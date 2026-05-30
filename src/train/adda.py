import logging
import os
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.optim import lr_scheduler

from src.metrics.mean_iou import MeanIoU
from src.models.discriminator import FCDiscriminator
from src.train.train_model import evaluate_model
from src.utils.utils import save_checkpoint

def adda_setup(cfg, device):

    discriminator = FCDiscriminator(num_classes=cfg.adda.adda_num_classes, ndf=cfg.adda.disc_ndf).to(device)
    
    disc_criterion = nn.MSELoss() if cfg.adda.ls_gan else nn.BCEWithLogitsLoss()
        
    # Optimizer setup
    lr = cfg.adda.adda_learning_rate
    wd = cfg.adda.adda_weight_decay
    beta1 = cfg.adda.adda_beta1
    beta2 = cfg.adda.adda_beta2
    match cfg.adda.adda_optimizer:
        case "SGD":
            disc_optimizer = optim.SGD(discriminator.parameters(), lr=lr, momentum=cfg.adda.adda_momentum, weight_decay=wd)
        case "Adam":
            disc_optimizer = optim.Adam(discriminator.parameters(), lr=lr, weight_decay=wd, betas=(beta1, beta2))
        case "AdamW":
            disc_optimizer = optim.AdamW(discriminator.parameters(), lr=lr, weight_decay=wd, betas=(beta1, beta2))
        case _:
            raise ValueError(f"Unsupported optimizer: {cfg.adda.adda_optimizer}")

    # Scheduler setup
    match cfg.adda.adda_scheduler:
        case "step_lr":
            disc_scheduler = lr_scheduler.StepLR(disc_optimizer, step_size=cfg.adda.adda_step_size, gamma=cfg.adda.adda_gamma)
        case "poly":
            disc_scheduler = lr_scheduler.LambdaLR(disc_optimizer, lr_lambda=lambda epoch: (1 - epoch / cfg.training.epochs) ** cfg.adda.adda_power)
        case "cosine":
            disc_scheduler = lr_scheduler.CosineAnnealingLR(disc_optimizer, T_max=cfg.training.epochs, eta_min=cfg.adda.adda_eta_min)
        case None:
            disc_scheduler = None
        case _:
            raise NotImplementedError(f"Unsupported scheduler type: {cfg.adda.adda_scheduler}")

    return discriminator, disc_criterion, disc_optimizer, disc_scheduler


def train_adda(
    generator, discriminator, gen_name, num_classes, lambda_adv, trainloader_source, trainloader_target, validloader, 
    criterion_gen, criterion_disc, optimizer_gen, optimizer_disc, scheduler, scheduler_disc, start_epoch, end_epoch,
    start_miou, bd_required, checkpoint_dir, device, log_frequency
):
    
    logging.info(f"{gen_name} - ADDA training")
    logging.info(f"Training epochs: {end_epoch}")
    
    def get_main_output(outputs):
        if "pidnet" in gen_name.lower():
            pred = outputs[1] if isinstance(outputs, (list, tuple)) else outputs
        elif isinstance(outputs, (list, tuple)):
            pred = outputs[0]
        else:
            pred = outputs
            
        return pred

    train_losses_gen_source, val_losses = [], []
    train_task_specific_losses_gen_source = {}
    adda_specific_losses = {"train_losses_gen_target": [], "train_losses_disc_source": [], "train_losses_disc_target": []}
    train_mious, val_mious = [], []
    train_ious, val_ious = [], []
    best_epoch, best_val_miou = start_epoch - 1, start_miou

    metric = MeanIoU(num_classes=num_classes).to(device)
    for epoch in range(start_epoch, end_epoch):
        
        # ADDA (source, target) data loaders
        def infinite_iterator(dataloader):
            while True:
                for batch in dataloader:
                    yield batch

        iter_source = infinite_iterator(trainloader_source) if len(trainloader_source) < len(trainloader_target) else iter(trainloader_source)
        iter_target = infinite_iterator(trainloader_target) if len(trainloader_target) < len(trainloader_source) else iter(trainloader_target)

        data_stream = zip(iter_source, iter_target)
        
        data_len, tot_batches = 0, max(len(trainloader_source), len(trainloader_target))
        train_loss_gen_source = 0.0
        epoch_task_specific_losses_gen_source, epoch_adda_specific_losses = {}, {"train_losses_gen_target": 0.0, "train_losses_disc_source": 0.0, "train_losses_disc_target": 0.0}
        metric.reset()

        generator.train()
        discriminator.train()

        for i, (batch_source, batch_target) in enumerate(data_stream):
            for idx in range(len(batch_source)):
                batch_source[idx] = batch_source[idx].to(device)
            
            if bd_required:
                inputs_source, gt_source = batch_source[0], batch_source[1:]
                masks_source = gt_source[0]
            else:
                inputs_source, gt_source = batch_source[0], batch_source[1]
                masks_source = gt_source
            
            inputs_target = batch_target[0].to(device)
                
            data_len += inputs_source.size(0)
            
            # Reset gradients
            optimizer_gen.zero_grad()
            optimizer_disc.zero_grad()

            # ===== Train generator =====
            # Freeze discriminator
            for param in discriminator.parameters():
                param.requires_grad = False

            # Train with source
            outputs_source = generator(inputs_source)
            pred_gen_source = get_main_output(outputs_source)
            batch_loss_gen_source, batch_task_specific_losses_gen_source = criterion_gen(outputs_source, gt_source)
            batch_loss_gen_source.backward()
            
            # Retain grads
            task_grads = [p.grad.detach() for p in generator.parameters() if p.grad is not None]
            grad_norm_task = torch.norm(torch.stack([g.norm(2) for g in task_grads]))
            old_grads = {p: p.grad.clone() for p in generator.parameters() if p.grad is not None}
            
            train_loss_gen_source += batch_loss_gen_source.item() * inputs_source.size(0)
            
            # Train with target
            outputs_target = generator(inputs_target)
            pred_gen_target = get_main_output(outputs_target)
            
            # ANALYTICAL TRACKING: 
            # Tell PyTorch to retain gradients for these intermediate tensors (normally dropped)
            pred_gen_target.retain_grad()  # Direct output of PIDNet (logits)
            
            preds_target = F.softmax(pred_gen_target, dim=1)
            preds_target.retain_grad()     # Softmax output (probabilities)
            
            outputs_gen_target = discriminator(preds_target)

            batch_loss_gen_target = criterion_disc(outputs_gen_target, torch.zeros_like(outputs_gen_target))
            
            epoch_adda_specific_losses["train_losses_gen_target"] += batch_loss_gen_target.item() * inputs_target.size(0)
            
            batch_loss_gen_target = lambda_adv * batch_loss_gen_target
            batch_loss_gen_target.backward()
            
            grad_in_softmax = preds_target.grad.norm(2).item() if preds_target.grad is not None else 0.0
            grad_out_softmax = pred_gen_target.grad.norm(2).item() if pred_gen_target.grad is not None else 0.0
            
            print(f"\n{'='*50}")
            print(f"--- ANALYTICAL PROPAGATION ANALYSIS (Iter {i+1}) ---")
            print(f"1. Gradient from Disc (on Probabilities): {grad_in_softmax:.6f}")
            print(f"2. Post-Softmax Gradient (on PIDNet Logits): {grad_out_softmax:.6f}")
            if grad_in_softmax > 0:
                print(f"-> Survival through Softmax Jacobian: {(grad_out_softmax/grad_in_softmax)*100:.2f}%")
            
            final_layer_grad = layer1_grad = None
            for name, p in generator.named_parameters():
                if 'final_layer.conv2.weight' in name and p.grad is not None:
                    adv_g = p.grad.detach() - old_grads[p] if p in old_grads else p.grad.detach()
                    final_layer_grad = adv_g.norm(2).item()
                if 'layer1.0.conv1.weight' in name and p.grad is not None:
                    adv_g = p.grad.detach() - old_grads[p] if p in old_grads else p.grad.detach()
                    layer1_grad = adv_g.norm(2).item()

            if final_layer_grad is not None and layer1_grad is not None:
                print(f"\n3. Adv Magnitude at Head (final_layer.conv2): {final_layer_grad:.6f}")
                print(f"4. Adv Magnitude at Base (layer1.0.conv1): {layer1_grad:.6f}")
                if final_layer_grad > 0:
                    print(f"-> Structural survival through PIDNet: {(layer1_grad/final_layer_grad)*100:.4f}%")
                else:
                    print("\n4. Could not compute structural survival (negative or zero gradient at final layer)")
            else:
                print("\n3. Could not compute structural survival (missing gradients in key layers)")
            print(f"{'='*50}\n")

            # Compute adversarial gradient norms
            adv_grads = [(p.grad.detach() - old_grads[p]) if p in old_grads else p.grad.detach() for p in generator.parameters() if p.grad is not None]
            grad_norm_adv = torch.norm(torch.stack([g.norm(2) for g in adv_grads]))
            
            # Compute total gradient norms
            grad_norm_total = torch.norm(torch.stack([p.grad.detach().norm(2) for p in generator.parameters() if p.grad is not None]))
            
            print(f"Iter {i+1} | Grad Norm Task: {grad_norm_task:.4f} | Grad Norm Adv: {grad_norm_adv:.4f} | Grad Norm Total: {grad_norm_total:.4f}")

            # ===== Train discriminator =====
            # Unfreeze discriminator
            for param in discriminator.parameters():
                param.requires_grad = True

            # Train with source
            pred_gen_source = pred_gen_source.detach()
            preds_source = F.softmax(pred_gen_source, dim=1)
            outputs_disc_source = discriminator(preds_source)

            batch_loss_disc_source = criterion_disc(outputs_disc_source, torch.zeros_like(outputs_disc_source))
            batch_loss_disc_source = batch_loss_disc_source / 2
            batch_loss_disc_source.backward()
            
            epoch_adda_specific_losses["train_losses_disc_source"] += batch_loss_disc_source.item() * inputs_source.size(0)

            # Train with target
            pred_gen_target = pred_gen_target.detach()
            preds_target = F.softmax(pred_gen_target, dim=1)
            outputs_disc_target = discriminator(preds_target)

            batch_loss_disc_target = criterion_disc(outputs_disc_target, torch.ones_like(outputs_disc_target))
            batch_loss_disc_target = batch_loss_disc_target / 2
            batch_loss_disc_target.backward()
            
            epoch_adda_specific_losses["train_losses_disc_target"] += batch_loss_disc_target.item() * inputs_target.size(0)

            #torch.nn.utils.clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, generator.parameters()), max_norm=35, norm_type=2)
            #torch.nn.utils.clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, discriminator.parameters()), max_norm=35, norm_type=2)
            
            optimizer_gen.step()
            optimizer_disc.step()
            
            for task, task_loss in batch_task_specific_losses_gen_source.items():
                if task not in epoch_task_specific_losses_gen_source:
                    epoch_task_specific_losses_gen_source[task] = 0.0
                epoch_task_specific_losses_gen_source[task] += task_loss.item() * inputs_source.size(0)
            
            metric.update(pred_gen_source, masks_source)
            current_miou, _ = metric.compute()

            if (i + 1) % log_frequency == 0:
                inner_losses_str = " | ".join([f"{k}: {v / data_len:.4f}" for k, v in epoch_task_specific_losses_gen_source.items()])
                logging.info(f"Epoch {epoch + 1}/{end_epoch} | Batch {i + 1}/{tot_batches} | Train Loss Gen Source: {train_loss_gen_source / data_len:.4f} | {inner_losses_str} | Train Loss Gen Target: {epoch_adda_specific_losses['train_losses_gen_target'] / data_len:.6f} | Train Loss Disc Source: {epoch_adda_specific_losses['train_losses_disc_source'] / data_len:.4f} | Train Loss Disc Target: {epoch_adda_specific_losses['train_losses_disc_target'] / data_len:.4f} | Train mIoU (%): {100 * current_miou.item():.2f}")
        
        # Loss aggregation
        train_loss_gen_source = train_loss_gen_source / data_len
        epoch_adda_specific_losses = {k: v / data_len for k, v in epoch_adda_specific_losses.items()}
        
        for task, task_loss in epoch_task_specific_losses_gen_source.items():
            epoch_task_specific_losses_gen_source[task] = task_loss / data_len
            if f"train_losses_{task}" not in train_task_specific_losses_gen_source:
                train_task_specific_losses_gen_source[f"train_losses_{task}"] = []

        # mIoU and IoU aggregation
        train_epoch_miou, train_epoch_ious = map(lambda x: x * 100, metric.compute())
        train_epoch_miou = train_epoch_miou.item()
        train_epoch_ious = train_epoch_ious.tolist()

        # Validation
        val_loss, val_epoch_miou, val_epoch_ious = evaluate_model(generator, gen_name, num_classes, validloader, criterion_gen, bd_required, epoch, end_epoch, device, log_frequency)
        
        train_losses_gen_source.append(float(train_loss_gen_source))
        val_losses.append(float(val_loss))
        for task, task_loss in epoch_task_specific_losses_gen_source.items():
            train_task_specific_losses_gen_source[f"train_losses_{task}"].append(float(task_loss))
        for adda_loss, adda_loss_value in epoch_adda_specific_losses.items():
            adda_specific_losses[adda_loss].append(float(adda_loss_value))
        train_mious.append(train_epoch_miou)
        val_mious.append(val_epoch_miou)
        train_ious.append(train_epoch_ious)
        val_ious.append(val_epoch_ious)
        
        # Logging
        logging.info(f"Epoch {epoch + 1}/{end_epoch} | Train Loss Gen Source: {train_loss_gen_source:.4f} | Train Loss Gen Target: {epoch_adda_specific_losses['train_losses_gen_target']:.6f} | Train Loss Disc Source: {epoch_adda_specific_losses['train_losses_disc_source']:.4f} | Train Loss Disc Target: {epoch_adda_specific_losses['train_losses_disc_target']:.4f} | Train mIoU (%): {train_epoch_miou:.2f} | Val Loss: {val_loss:.4f} | Val mIoU (%): {val_epoch_miou:.2f}")
        
        # Checkpointing
        if best_val_miou is None or val_epoch_miou > best_val_miou:
            prev_best_epoch = best_epoch
            best_val_miou = val_epoch_miou
            best_epoch = epoch
            chp_path = os.path.join(checkpoint_dir, f"best_{gen_name}_adda_{epoch + 1}.pth.tar")
            save_checkpoint(chp_path, epoch + 1, generator, disc_model=discriminator, optimizer=optimizer_gen, disc_optimizer=optimizer_disc, scheduler=scheduler, disc_scheduler=scheduler_disc, miou=val_epoch_miou, ious=val_epoch_ious)
            
            if prev_best_epoch >= start_epoch:
                prev_chp_path = os.path.join(checkpoint_dir, f"best_{gen_name}_adda_{prev_best_epoch + 1}.pth.tar")
                if os.path.exists(prev_chp_path):
                    os.remove(prev_chp_path)

        if scheduler is not None:
            scheduler.step()
        if scheduler_disc is not None:
            scheduler_disc.step()
            
    last_chp_path = os.path.join(checkpoint_dir, f"last_{gen_name}_adda_{end_epoch}.pth.tar")
    last_miou = val_mious[-1] if len(val_mious) > 0 else None
    last_ious = val_ious[-1] if len(val_ious) > 0 else None
    save_checkpoint(last_chp_path, epoch + 1, generator, disc_model=discriminator, optimizer=optimizer_gen, disc_optimizer=optimizer_disc, scheduler=scheduler, disc_scheduler=scheduler_disc, miou=last_miou, ious=last_ious)
    
    return {
        "train_losses": train_losses_gen_source,
        "val_losses": val_losses,
        **train_task_specific_losses_gen_source,
        **adda_specific_losses,
        "train_mious": train_mious,
        "val_mious": val_mious,
        "train_ious": train_ious,
        "val_ious": val_ious
    }