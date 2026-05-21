import json
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

def adda_multi_setup(cfg, device):
    num_discs = 2 # For PIDNet 3 outputs
    criterions_disc = [nn.BCEWithLogitsLoss() for _ in range(num_discs)]

    discriminators = [FCDiscriminator(num_classes=cfg.adda.adda_num_classes).to(device) for _ in range(num_discs)]
    
    optimizers_disc = []
    schedulers_disc = []

    for discriminator in discriminators:
        # Optimizer setup
        lr = cfg.adda.adda_learning_rate
        wd = cfg.adda.adda_weight_decay
        match cfg.adda.adda_optimizer:
            case "SGD":
                disc_optimizer = optim.SGD(discriminator.parameters(), lr=lr, momentum=cfg.adda.adda_momentum, weight_decay=wd)
            case "Adam":
                disc_optimizer = optim.Adam(discriminator.parameters(), lr=lr, weight_decay=wd)
            case "AdamW":
                disc_optimizer = optim.AdamW(discriminator.parameters(), lr=lr, weight_decay=wd)
            case _:
                raise ValueError(f"Unsupported optimizer: {cfg.adda.adda_optimizer}")
        optimizers_disc.append(disc_optimizer)
        
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
        schedulers_disc.append(disc_scheduler)

    return discriminators, criterions_disc, optimizers_disc, schedulers_disc


def train_adda_multi(
    generator, discriminators, gen_name, num_classes, lambda_adv, trainloader_source, trainloader_target, validloader, 
    criterion_gen, criterions_disc, optimizer_gen, optimizers_disc, scheduler, schedulers_disc, start_epoch, end_epoch,
    start_miou, bd_required, checkpoint_dir, device, log_frequency
):
    
    logging.info(f"{gen_name} - ADDA (multi-head) training")
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
    adda_specific_losses = {}
    for j in range(len(discriminators)):
        adda_specific_losses[f"train_losses_gen_target_disc{j}"] = []
        adda_specific_losses[f"train_losses_disc_source_disc{j}"] = []
        adda_specific_losses[f"train_losses_disc_target_disc{j}"] = []
    
    train_mious, val_mious = [], []
    train_ious, val_ious = [], []
    best_epoch, best_val_miou = start_epoch - 1, start_miou
    
    analytical_log = []
    metric = MeanIoU(num_classes).to(device)

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
        epoch_task_specific_losses_gen_source = {}
        epoch_adda_specific_losses = {}
        for j in range(len(discriminators)):
            epoch_adda_specific_losses[f"train_losses_gen_target_disc{j}"] = 0.0
            epoch_adda_specific_losses[f"train_losses_disc_source_disc{j}"] = 0.0
            epoch_adda_specific_losses[f"train_losses_disc_target_disc{j}"] = 0.0
            
        metric.reset()
        
        cum_diag = {"task": 0.0, "adv": 0.0, "total": 0.0, "adv_ratio": 0.0, "in_SM": 0.0, "out_SM": 0.0, "head_p": 0.0, "head_i": 0.0, "base": 0.0}
        cum_diag_disc = {str(j): {"src_mean": 0.0, "tgt_mean": 0.0, "grad_norm": 0.0} for j in range(len(discriminators))}

        generator.train()
        for disc in discriminators:
            disc.train()

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
            optimizer_gen.zero_grad(set_to_none=True)
            for optimizer_disc in optimizers_disc:
                optimizer_disc.zero_grad(set_to_none=True)

            # ===== Train generator =====
            # Freeze discriminator
            for disc in discriminators:
                for param in disc.parameters():
                    param.requires_grad = False

            # Train with source
            outputs_source = generator(inputs_source)
            pred_gen_source = get_main_output(outputs_source)
            batch_loss_gen_source, batch_task_specific_losses_gen_source = criterion_gen(outputs_source, gt_source)
            batch_loss_gen_source.backward()
            
            # Retain grads
            task_grads = [p.grad.detach() for p in generator.parameters() if p.grad is not None]
            grad_norm_task = torch.norm(torch.stack([g.norm(2) for g in task_grads]))
            old_grads = {p: p.grad.detach().clone() for p in generator.parameters() if p.grad is not None}
            
            train_loss_gen_source += batch_loss_gen_source.item() * inputs_source.size(0)
            
            # Train with target
            outputs_target = generator(inputs_target)
            
            main_pred_target = None
            main_preds_target = None
            
            # Ensure outputs are in a tuple/list format
            if not isinstance(outputs_target, (list, tuple)):
                outputs_target = [outputs_target]
                
            lambdas_adv = [lambda_adv / 10, lambda_adv] if len(outputs_target) > 1 else [lambda_adv]
            for out_idx, (out_target, disc, crit, current_lambda_adv) in enumerate(zip(outputs_target, discriminators, criterions_disc, lambdas_adv)):
                out_target.retain_grad()
                preds_target = F.softmax(out_target, dim=1)
                preds_target.retain_grad()
                
                outputs_gen_target = disc(preds_target)
                batch_loss_gen_target_disc = crit(outputs_gen_target, torch.zeros_like(outputs_gen_target))
                
                epoch_adda_specific_losses[f"train_losses_gen_target_disc{out_idx}"] += batch_loss_gen_target_disc.item() * inputs_target.size(0)
                
                # Apply current_lambda_adv and backpropagate
                scaled_loss = current_lambda_adv * batch_loss_gen_target_disc
                # Use retain_graph=True for all but the last discriminator to avoid clearing the graph if they share layers
                retain = (out_idx < len(discriminators) - 1)
                scaled_loss.backward(retain_graph=retain)
                
                if out_idx == 1 or len(outputs_target) == 1:
                    main_pred_target = out_target
                    main_preds_target = preds_target
            
            grad_in_softmax = main_preds_target.grad.norm(2).item() if main_preds_target is not None and main_preds_target.grad is not None else 0.0
            grad_out_softmax = main_pred_target.grad.norm(2).item() if main_pred_target is not None and main_pred_target.grad is not None else 0.0
            softmax_survival = (grad_out_softmax / grad_in_softmax * 100) if grad_in_softmax > 0 else 0.0
            
            head_p_grad = head_i_grad = layer1_grad = 0.0
            for name, p in generator.named_parameters():
                if 'seghead_p.conv2.weight' in name and p.grad is not None:
                    adv_g = p.grad.detach() - old_grads[p] if p in old_grads else p.grad.detach()
                    head_p_grad = adv_g.norm(2).item()
                if 'final_layer.conv2.weight' in name and p.grad is not None:
                    adv_g = p.grad.detach() - old_grads[p] if p in old_grads else p.grad.detach()
                    head_i_grad = adv_g.norm(2).item()
                if 'layer1.0.conv1.weight' in name and p.grad is not None:
                    adv_g = p.grad.detach() - old_grads[p] if p in old_grads else p.grad.detach()
                    layer1_grad = adv_g.norm(2).item()

            struct_survival = (layer1_grad / (head_p_grad + head_i_grad) * 100) if (head_p_grad + head_i_grad) > 0 else 0.0
            struct_survival_p = (layer1_grad / head_p_grad * 100) if head_p_grad > 0 else 0.0
            struct_survival_i = (layer1_grad / head_i_grad * 100) if head_i_grad > 0 else 0.0

            # Compute adversarial gradient norms
            adv_grads = [(p.grad.detach() - old_grads[p]) if p in old_grads else p.grad.detach() for p in generator.parameters() if p.grad is not None]
            grad_norm_adv = torch.norm(torch.stack([g.norm(2) for g in adv_grads])) if adv_grads else 0.0
            
            # Compute total gradient norms
            grad_norm_total = torch.norm(torch.stack([p.grad.detach().norm(2) for p in generator.parameters() if p.grad is not None]))
            
            g_task_val = grad_norm_task.item() if isinstance(grad_norm_task, torch.Tensor) else grad_norm_task
            g_adv_val = grad_norm_adv.item() if isinstance(grad_norm_adv, torch.Tensor) else grad_norm_adv
            adv_ratio = (g_adv_val / g_task_val) if g_task_val > 0 else 0.0
            
            cum_diag["task"] += g_task_val
            cum_diag["adv"] += g_adv_val
            cum_diag["total"] += grad_norm_total.item() if isinstance(grad_norm_total, torch.Tensor) else grad_norm_total
            cum_diag["adv_ratio"] += adv_ratio
            cum_diag["in_SM"] += grad_in_softmax
            cum_diag["out_SM"] += grad_out_softmax
            cum_diag["head_p"] += head_p_grad
            cum_diag["head_i"] += head_i_grad
            cum_diag["base"] += layer1_grad
            
            iter_stats = {
                "epoch": epoch + 1,
                "batch": i + 1,
                "grad_norm_task": g_task_val,
                "grad_norm_adv": g_adv_val,
                "grad_norm_total": grad_norm_total.item() if isinstance(grad_norm_total, torch.Tensor) else grad_norm_total,
                "adv_ratio": adv_ratio,
                "adv_flow_softmax": {
                    "in": grad_in_softmax,
                    "out": grad_out_softmax,
                    "survival_pct": softmax_survival
                },
                "struct": {
                    "head_p": head_p_grad,
                    "head_i": head_i_grad,
                    "base": layer1_grad,
                    "survival_pct_combined": struct_survival,
                    "survival_pct_p": struct_survival_p,
                    "survival_pct_i": struct_survival_i
                },
                "discriminators": {str(j): {} for j in range(len(discriminators))}
            }

            # ===== Train discriminator =====
            # Unfreeze discriminator
            for disc in discriminators:
                for param in disc.parameters():
                    param.requires_grad = True

            # Train with source
            if not isinstance(outputs_source, (list, tuple)):
                outputs_source = [outputs_source]
            
            for out_idx, (out_source, disc, crit) in enumerate(zip(outputs_source, discriminators, criterions_disc)):
                preds_source = F.softmax(out_source.detach(), dim=1)
                outputs_disc_source = disc(preds_source)
                iter_stats["discriminators"][str(out_idx)]["source_domain_mean"] = torch.sigmoid(outputs_disc_source).mean().item()
                batch_loss_disc_source_disc = crit(outputs_disc_source, torch.zeros_like(outputs_disc_source)) / 2
                
                batch_loss_disc_source_disc.backward()
                epoch_adda_specific_losses[f"train_losses_disc_source_disc{out_idx}"] += batch_loss_disc_source_disc.item() * 2 * inputs_source.size(0)

            # Train with target
            if not isinstance(outputs_target, (list, tuple)):
                outputs_target = [outputs_target]
                
            for out_idx, (out_target, disc, crit) in enumerate(zip(outputs_target, discriminators, criterions_disc)):
                preds_target = F.softmax(out_target.detach(), dim=1)
                outputs_disc_target = disc(preds_target)
                iter_stats["discriminators"][str(out_idx)]["target_domain_mean"] = torch.sigmoid(outputs_disc_target).mean().item()
                batch_loss_disc_target_disc = crit(outputs_disc_target, torch.ones_like(outputs_disc_target)) / 2

                batch_loss_disc_target_disc.backward()
                epoch_adda_specific_losses[f"train_losses_disc_target_disc{out_idx}"] += batch_loss_disc_target_disc.item() * 2 * inputs_target.size(0)

            for out_idx, disc in enumerate(discriminators):
                disc_grads = [p.grad.detach() for p in disc.parameters() if p.grad is not None]
                disc_grad_norm = torch.norm(torch.stack([g.norm(2) for g in disc_grads])) if disc_grads else 0.0
                iter_stats["discriminators"][str(out_idx)]["grad_norm"] = disc_grad_norm.item() if isinstance(disc_grad_norm, torch.Tensor) else disc_grad_norm
            
            for j_str, d_stats in iter_stats["discriminators"].items():
                cum_diag_disc[j_str]["src_mean"] += d_stats.get("source_domain_mean", 0.0)
                cum_diag_disc[j_str]["tgt_mean"] += d_stats.get("target_domain_mean", 0.0)
                cum_diag_disc[j_str]["grad_norm"] += d_stats.get("grad_norm", 0.0)

            analytical_log.append(iter_stats)

            #clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, generator.parameters()), max_norm=35, norm_type=2)
            #for disc in discriminators:
            #    clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, disc.parameters()), max_norm=35, norm_type=2)
            
            optimizer_gen.step()
            for opt in optimizers_disc:
                opt.step()
            
            for task, task_loss in batch_task_specific_losses_gen_source.items():
                if task not in epoch_task_specific_losses_gen_source:
                    epoch_task_specific_losses_gen_source[task] = 0.0
                epoch_task_specific_losses_gen_source[task] += task_loss.item() * inputs_source.size(0)
                
            metric.update(pred_gen_source, masks_source)
            current_miou, _ = metric.compute()

            if (i + 1) % log_frequency == 0:
                inner_losses_str = " | ".join([f"{k}: {v / data_len:.4f}" for k, v in epoch_task_specific_losses_gen_source.items()])
                
                adda_losses_str = ""
                for j in range(len(discriminators)):
                    adda_losses_str += f"Gen Target D{j}: {epoch_adda_specific_losses[f'train_losses_gen_target_disc{j}'] / data_len:.6f} | "
                    adda_losses_str += f"Disc Source D{j}: {epoch_adda_specific_losses[f'train_losses_disc_source_disc{j}'] / data_len:.4f} | "
                    adda_losses_str += f"Disc Target D{j}: {epoch_adda_specific_losses[f'train_losses_disc_target_disc{j}'] / data_len:.4f} | "
                
                logging.info(f"Epoch {epoch + 1}/{end_epoch} | Batch {i + 1}/{tot_batches} | Train Loss Gen Source: {train_loss_gen_source / data_len:.4f} | {inner_losses_str} | {adda_losses_str}Train mIoU (%): {100 * current_miou.item():.2f}")
                
                avg_task = cum_diag["task"] / (i + 1)
                avg_adv = cum_diag["adv"] / (i + 1)
                avg_total = cum_diag["total"] / (i + 1)
                avg_adv_ratio = cum_diag["adv_ratio"] / (i + 1)
                avg_in_sm = cum_diag["in_SM"] / (i + 1)
                avg_out_sm = cum_diag["out_SM"] / (i + 1)
                avg_head_p = cum_diag["head_p"] / (i + 1)
                avg_head_i = cum_diag["head_i"] / (i + 1)
                avg_base = cum_diag["base"] / (i + 1)
                
                avg_softmax_survival = (avg_out_sm / avg_in_sm * 100) if avg_in_sm > 0 else 0.0
                avg_struct_survival = (avg_base / (avg_head_p + avg_head_i) * 100) if (avg_head_p + avg_head_i) > 0 else 0.0
                avg_struct_survival_p = (avg_base / avg_head_p * 100) if avg_head_p > 0 else 0.0
                avg_struct_survival_i = (avg_base / avg_head_i * 100) if avg_head_i > 0 else 0.0

                disc_diag_str = " | ".join([
                    f"D{j} (Src: {cum_diag_disc[str(j)]['src_mean']/(i+1):.3f}, Tgt: {cum_diag_disc[str(j)]['tgt_mean']/(i+1):.3f}, Grad: {cum_diag_disc[str(j)]['grad_norm']/(i+1):.3f})"
                    for j in range(len(discriminators))
                ])

                logging.info(f"   -> [Diagnostics] GradNorm Task: {avg_task:.4f} | Adv: {avg_adv:.4f} | Total: {avg_total:.4f} | AdvRatio: {avg_adv_ratio:.4f} || "
                             f"AdvFlow Softmax: {avg_in_sm:.4f} -> {avg_out_sm:.4f} ({avg_softmax_survival:.1f}%) | "
                             f"Struct: Head (P:{avg_head_p:.4f}, I:{avg_head_i:.4f}) -> Base {avg_base:.4f} (Comb:{avg_struct_survival:.1f}%, P:{avg_struct_survival_p:.1f}%, I:{avg_struct_survival_i:.1f}%) || {disc_diag_str}")
                
            del task_grads, old_grads, adv_grads
            del outputs_source, outputs_target, batch_loss_gen_source
            del main_pred_target, main_preds_target
        
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
        
        adda_epoch_losses_str = ""
        for j in range(len(discriminators)):
            adda_epoch_losses_str += f"Gen Target D{j}: {epoch_adda_specific_losses[f'train_losses_gen_target_disc{j}']:.6f} | "
            adda_epoch_losses_str += f"Disc Source D{j}: {epoch_adda_specific_losses[f'train_losses_disc_source_disc{j}']:.4f} | "
            adda_epoch_losses_str += f"Disc Target D{j}: {epoch_adda_specific_losses[f'train_losses_disc_target_disc{j}']:.4f} | "
            
        logging.info(f"Epoch {epoch + 1}/{end_epoch} | Train Loss Gen Source: {train_loss_gen_source:.4f} | {adda_epoch_losses_str}Train mIoU (%): {train_epoch_miou:.2f} | Val Loss: {val_loss:.4f} | Val mIoU (%): {val_epoch_miou:.2f}")
        
        # Checkpointing
        if best_val_miou is None or val_epoch_miou > best_val_miou:
            prev_best_epoch = best_epoch
            best_val_miou = val_epoch_miou
            best_epoch = epoch
            chp_path = os.path.join(checkpoint_dir, f"best_{gen_name}_adda_{epoch + 1}.pth.tar")
            save_checkpoint(chp_path, epoch + 1, generator, disc_model=discriminators, optimizer=optimizer_gen, disc_optimizer=optimizers_disc, scheduler=scheduler, disc_scheduler=schedulers_disc, miou=val_epoch_miou, ious=val_epoch_ious)
            
            if prev_best_epoch >= start_epoch:
                prev_chp_path = os.path.join(checkpoint_dir, f"best_{gen_name}_adda_{prev_best_epoch + 1}.pth.tar")
                if os.path.exists(prev_chp_path):
                    os.remove(prev_chp_path)

        if scheduler is not None:
            scheduler.step()
        if schedulers_disc is not None:
            for s_disc in schedulers_disc:
                if s_disc is not None:
                    s_disc.step()
        
        with open(os.path.join(checkpoint_dir, "analytical_metrics.json"), "w") as f:
            json.dump(analytical_log, f, indent=4)
            
    last_chp_path = os.path.join(checkpoint_dir, f"last_{gen_name}_adda_{end_epoch}.pth.tar")
    last_miou = val_mious[-1] if len(val_mious) > 0 else None
    last_ious = val_ious[-1] if len(val_ious) > 0 else None
    save_checkpoint(last_chp_path, epoch + 1, generator, disc_model=discriminators, optimizer=optimizer_gen, disc_optimizer=optimizer_disc, scheduler=scheduler, disc_scheduler=schedulers_disc, miou=last_miou, ious=last_ious)
    
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