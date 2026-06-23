import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from src.dataset.dataset import LoveDA
from src.losses.deeplab_v2 import DeepLabLoss
from src.models.bisenet import BiSeNet
from src.models.deeplab_v2 import get_deeplab_v2
from src.models.stdc import STDC
from src.models.pidnet import get_pidnet

from src.losses.focal import FocalLoss
from src.losses.ohem import OHEMCrossEntropy
from src.losses.stdc import STDCLoss, DetailAggregateLoss
from src.losses.bisenet import BiSeNetLoss
from src.losses.bondary import BondaryLoss
from src.losses.pidnet import PIDNetLoss, PIDNetSemanticLoss

from src.metrics.mean_iou import MeanIoU
from src.utils.utils import load_checkpoint, save_checkpoint

def setup_model(cfg, device, backbone_name=None):
    model_name = cfg.model.model.lower()
    
    # Semantic criterion setup
    sem_kwargs = {}
    if cfg.training.weighted_loss:
        sem_kwargs['weight'] = LoveDA.get_class_weights(cfg.path.source, cfg.training.loss).to(device)
        
        logging.info("Using weighted loss")
    else:
        logging.info("Using unweighted loss")
        
    if cfg.training.loss == "cross_entropy":
        weight = sem_kwargs.get('weight', None)
        base_criterion = nn.CrossEntropyLoss(ignore_index=cfg.model.ignore_index, weight=weight)
    elif cfg.training.loss == "ohem":
        thres = getattr(cfg.training, "ohem_threshold", 0.8)
        min_kept = getattr(cfg.training, "ohem_keep", 32768)
        weight = sem_kwargs.get('weight', None)
        base_criterion = OHEMCrossEntropy(ignore_index=cfg.model.ignore_index, thres=thres, min_kept=min_kept, weight=weight)
    elif cfg.training.loss == "focal":
        gamma = getattr(cfg.training, "focal_gamma", 2.0)
        alpha = sem_kwargs.get('weight', None)
        base_criterion = FocalLoss(ignore_index=cfg.model.ignore_index, gamma=gamma, alpha=alpha)
    else:
        raise ValueError(f"Unsupported criterion: {cfg.training.loss}")
    
    # Model and loss initialization
    if "deeplab" in model_name:
        pretrained_model_path = os.path.join(cfg.path.weights, f"{cfg.model.model}.pth")
        model = get_deeplab_v2(
            num_classes=cfg.model.num_classes, pretrain=True, pretrain_model_path=pretrained_model_path
        )
        param_groups = [group["params"] for group in model.optim_parameters(lr=0)] # Learning rate will be set later in the universal setup
        criterion = DeepLabLoss(base_criterion)
    elif "bisenet" in model_name:
        model = BiSeNet(cfg.model.num_classes, backbone_name)
        param_groups = model.get_param_groups()
        criterion = BiSeNetLoss(base_criterion)
    elif "stdc" in model_name:
        pretrained_model = os.path.join(cfg.path.weights, f"{cfg.model.model}.pth")
        model = STDC(backbone_name, cfg.model.num_classes, pretrain_model=pretrained_model, use_boundary_8=True)
        param_groups = model.get_param_groups()
        detail_criterion = DetailAggregateLoss()
        criterion = STDCLoss(sem_loss=base_criterion, bd_loss=detail_criterion, ignore_index=cfg.model.ignore_index)
    elif "pidnet" in model_name:
        pretrained_weights = os.path.join(cfg.path.weights, f"{cfg.model.model}.pth")
        model, param_groups = get_pidnet(cfg.model.model, cfg.model.num_classes, pretrained_weights, imgnet_pretrained=True)
        
        args = {"weight": sem_kwargs.get('weight', None)}
        for k in ["focal_gamma", "ohem_threshold", "ohem_keep"]:
            if hasattr(cfg.training, k): args[k] = getattr(cfg.training, k)
            
        sem_loss = PIDNetSemanticLoss(type=cfg.training.loss, ignore_index=cfg.model.ignore_index, **args)
        bd_loss = BondaryLoss()
        criterion = PIDNetLoss(sem_loss=sem_loss, bd_loss=bd_loss, ignore_index=cfg.model.ignore_index)
    else:
        raise ValueError(f"Model unsupported for universal setup: {model_name}")
    
    if cfg.path.pretrained_path is not None:
        weights = torch.load(cfg.path.pretrained_path, map_location=device)
        if "model" in weights:
            weights = weights["model"]
        model.load_state_dict(weights, strict=True)
        logging.info(f"Pretrained weights loaded from {cfg.path.pretrained_path}")
        
    model = model.to(device)
    
    # Learning rate
    lr = cfg.training.learning_rate
    if isinstance(lr, dict):
        lr = lr[cfg.model.model]
        
    if isinstance(lr, (list, tuple)):
        opt_param_groups = [{'params': group, 'lr': lr[i]} for i, group in enumerate(param_groups)]
    else:
        assert isinstance(lr, (float, int)), f"Learning rate must be a float, int, list, tuple, or dict. Found: {type(lr)}"
        opt_param_groups = [{'params': group, 'lr': lr} for group in param_groups]
        
    # Weight decay
    wd = cfg.training.weight_decay
    if isinstance(wd, dict):
        wd = wd[cfg.model.model]
        
    if isinstance(wd, (list, tuple)):
        for i in range(len(opt_param_groups)):
            opt_param_groups[i]['weight_decay'] = wd[i]
    else:
        assert isinstance(wd, (float, int)), f"Weight decay must be a float, int, list, tuple, or dict. Found: {type(wd)}"
        for i in range(len(opt_param_groups)):
            opt_param_groups[i]['weight_decay'] = wd
    
    # Momentum
    momentum = cfg.training.momentum if hasattr(cfg.training, "momentum") else 0
    
    # Optimizer setup
    match cfg.training.optimizer:
        case "SGD":
            optimizer = optim.SGD(opt_param_groups, momentum=momentum)
        case "Adam":
            optimizer = optim.Adam(opt_param_groups)
        case "AdamW":
            optimizer = optim.AdamW(opt_param_groups)
        case _:
            raise ValueError(f"Unsupported optimizer: {cfg.training.optimizer}")
    
    # Scheduler setup
    match cfg.training.scheduler:
        case "step_lr":
            scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.training.step_size, gamma=cfg.training.gamma)
        case "poly":
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 
                min(1.0, ((epoch + 1) / cfg.training.warmup_epochs if cfg.training.warmup_epochs > 0 else 1.0)) * (1 - max(0.0, (epoch + 1 - cfg.training.warmup_epochs) / (cfg.training.epochs - cfg.training.warmup_epochs))) ** cfg.training.power
            )
        case "cosine":
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs, eta_min=cfg.training.eta_min)
        case _:
            raise NotImplementedError(f"Unsupported scheduler type: {cfg.training.scheduler}")
        
    if cfg.path.checkpoint_path is not None:
        loaded_checkpoint_info = load_checkpoint(cfg.path.checkpoint_path, model, device, optimizer=optimizer, scheduler=scheduler, epochs=cfg.training.epochs)
        logging.info(f"Checkpoint loaded from {cfg.path.checkpoint_path} | Resuming from epoch {loaded_checkpoint_info['epoch']} | mIoU (%): {loaded_checkpoint_info.get('miou', 'N/A'):.2f}")
    
    start_epoch = loaded_checkpoint_info['epoch'] if cfg.path.checkpoint_path is not None else 0
    start_miou = loaded_checkpoint_info.get('miou', None) if cfg.path.checkpoint_path is not None else None
    
    return model, criterion, optimizer, scheduler, start_epoch, start_miou

@torch.no_grad()
def evaluate_model(model, model_name, num_classes, dataloader, criterion, bd_required, epoch, tot_epochs, device, log_frequency):
    logging.info(f"{model_name} - Evaluation | Epoch {epoch + 1}/{tot_epochs}")
    model.eval()

    tot_loss = 0.0
    data_len, tot_batches = 0, len(dataloader)
    metric = MeanIoU(num_classes=num_classes).to(device)
    
    for i, batch in enumerate(dataloader):
        try:
            for idx in range(len(batch)):
                batch[idx] = batch[idx].to(device)
                    
            if bd_required:
                inputs, gt = batch[0], batch[1:]
                masks, boundaries = gt[0], gt[1]
            else:
                inputs, gt = batch[0], batch[1]
                masks = gt

            # Forward pass
            outputs = model(inputs)
            
            # Loss extraction
            if "pidnet" in model_name.lower():
                # PIDNetLoss expects ground truth as tuples if boundary is strictly needed
                loss_res = criterion(outputs, (masks, boundaries) if bd_required else (masks, None))
            else:
                loss_res = criterion(outputs, masks)
                
            # Handle scalar (BiSeNet/STDC eval mode) or tuple (PIDNet)
            loss = loss_res[0] if isinstance(loss_res, tuple) else loss_res
            
            tot_loss += loss.item() * inputs.size(0)
            data_len += inputs.size(0)
            
            # Pred Extraction (mIoU)
            if "pidnet" in model_name.lower():
                pred = outputs[1] if isinstance(outputs, (list, tuple)) else outputs
            elif isinstance(outputs, (list, tuple)):
                pred = outputs[0]
            else:
                pred = outputs
                
            # normalized_pred = torch.softmax(pred, dim=1)
            # assert normalized_pred.shape[0] == 1 and normalized_pred.shape[1] == num_classes
            
            # sq_normalized_pred = normalized_pred.squeeze(0)  # Remove batch dimension
            # sq_masks = masks.squeeze(0)  # Remove batch dimension
            
            # idx = f"{i:04d}"
            
            # DOMAIN = "urban"
            # if i == 0:
            #     SAVE_GT = not os.path.exists(os.path.join("eval_objects", DOMAIN, "gt"))
            
            # os.makedirs(os.path.join("eval_objects", DOMAIN, model_name), exist_ok=True)
            # torch.save(sq_normalized_pred.cpu(), os.path.join("eval_objects", DOMAIN, model_name, f"{idx}.pt"))
            
            # if SAVE_GT:
            #     os.makedirs(os.path.join("eval_objects", DOMAIN, "gt"), exist_ok=True)
            #     torch.save(sq_masks.cpu(), os.path.join("eval_objects", DOMAIN, "gt", f"{idx}.pt"))
                
            # Update mIoU metric
            metric.update(pred, masks)
            current_miou, _ = metric.compute()
            
        except RuntimeError as e:
            logging.error(f"Runtime error during evaluation at epoch {epoch + 1}, batch {i + 1}/{tot_batches}: {e}. Skipping this batch.")
        
        if (i + 1) % log_frequency == 0:
            logging.info(f"Epoch {epoch + 1}/{tot_epochs} | Batch {i + 1}/{tot_batches} | Loss: {tot_loss / data_len:.4f} | mIoU (%): {100 * current_miou.item():.2f}")
        
    miou, ious = map(lambda x: 100 * x, metric.compute())
    
    return tot_loss / data_len, miou.item(), ious.tolist()

def train_model(model, model_name, num_classes, trainloader, validloader, criterion, optimizer, scheduler, start_epoch, end_epoch, start_miou, bd_required, checkpoint_dir, device, log_frequency, validloader_adj=None):
    logging.info(f"{model_name} - Training")
    
    logging.info(f"Training epochs: {end_epoch} (from {start_epoch + 1} to {end_epoch})")
    
    train_losses, val_losses = [], []
    train_task_specific_losses = {}
    train_mious, val_mious = [], []
    train_ious, val_ious = [], []
    best_epoch, best_val_miou = start_epoch - 1, start_miou
    
    if validloader_adj is not None:
        val_losses_adj, val_mious_adj, val_ious_adj = [], [], []

    metric = MeanIoU(num_classes=num_classes).to(device)
    for epoch in range(start_epoch, end_epoch):
        
        data_len, tot_batches = 0, len(trainloader)
        train_loss, epoch_task_specific_losses = 0.0, {}
        
        model.train()

        metric.reset()
        for i, batch in enumerate(trainloader):
            for idx in range(len(batch)):
                batch[idx] = batch[idx].to(device)
                
            if bd_required:
                inputs, gt = batch[0], batch[1:]
                masks = gt[0]
            else:
                inputs, gt = batch[0], batch[1]
                masks = gt
            
            data_len += inputs.size(0)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Loss unpacking
            batch_loss, batch_task_specific_losses = criterion(outputs, gt)
            
            train_loss += batch_loss.item() * inputs.size(0)
            
            for task, task_loss in batch_task_specific_losses.items():
                if task not in epoch_task_specific_losses:
                    epoch_task_specific_losses[task] = 0.0
                epoch_task_specific_losses[task] += task_loss.item() * inputs.size(0)

            # Backward pass
            batch_loss.backward()
            optimizer.step()
            
            # Predictions Extraction
            if "pidnet" in model_name.lower():
                preds = outputs[1] if isinstance(outputs, (list, tuple)) else outputs
            elif isinstance(outputs, (list, tuple)):
                preds = outputs[0]
            else:
                preds = outputs
            
            metric.update(preds, masks)
            current_miou, _ = metric.compute()

            if (i + 1) % log_frequency == 0:
                inner_losses_str = " | ".join([f"{k}: {v / data_len:.4f}" for k, v in epoch_task_specific_losses.items()])
                logging.info(f"Epoch {epoch + 1}/{end_epoch} | Batch {i + 1}/{tot_batches} | Total Loss: {train_loss / data_len:.4f} | {inner_losses_str} | mIoU (%): {100 * current_miou.item():.2f}")

        # Loss aggregation
        train_loss = train_loss / data_len
        for task, task_loss in epoch_task_specific_losses.items():
            epoch_task_specific_losses[task] = task_loss / data_len
            if f"train_losses_{task}" not in train_task_specific_losses:
                train_task_specific_losses[f"train_losses_{task}"] = []
                
        # mIoU and IoU aggregation
        train_epoch_miou, train_epoch_ious = map(lambda x: x * 100, metric.compute())
        train_epoch_miou = train_epoch_miou.item()
        train_epoch_ious = train_epoch_ious.tolist()

        val_loss, val_epoch_miou, val_epoch_ious = evaluate_model(model, model_name, num_classes, validloader, criterion, bd_required, epoch, end_epoch, device, log_frequency)  
        
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        for task, task_loss in epoch_task_specific_losses.items():
            train_task_specific_losses[f"train_losses_{task}"].append(float(task_loss))
        train_mious.append(train_epoch_miou)
        val_mious.append(val_epoch_miou)
        train_ious.append(train_epoch_ious)
        val_ious.append(val_epoch_ious)
        
        if validloader_adj is not None:
            val_loss_adj, val_epoch_miou_adj, val_epoch_ious_adj = evaluate_model(model, model_name, num_classes, validloader_adj, criterion, bd_required, epoch, end_epoch, device, log_frequency)
            val_losses_adj.append(float(val_loss_adj))
            val_mious_adj.append(val_epoch_miou_adj)
            val_ious_adj.append(val_epoch_ious_adj)
        
        # Logging
        msg = f"Epoch {epoch + 1}/{end_epoch} | Train Loss: {train_loss:.4f} | Train mIoU (%): {train_epoch_miou:.2f} | Val Loss: {val_loss:.4f} | Val mIoU (%): {val_epoch_miou:.2f}"
        if validloader_adj is not None:
            msg += f" | Val Loss Adj: {val_loss_adj:.4f} | Val mIoU Adj (%): {val_epoch_miou_adj:.2f}"
        logging.info(msg)
        
        # Checkpointing
        if best_val_miou is None or val_epoch_miou > best_val_miou:
            prev_best_epoch = best_epoch
            best_val_miou = val_epoch_miou
            best_epoch = epoch
            chp_path = os.path.join(checkpoint_dir, f"best_{model_name}_{epoch + 1}.pth.tar")
            save_checkpoint(chp_path, epoch + 1, model, optimizer=optimizer, scheduler=scheduler, miou=val_epoch_miou, ious=val_epoch_ious)
            
            if prev_best_epoch >= start_epoch:
                prev_chp_path = os.path.join(checkpoint_dir, f"best_{model_name}_{prev_best_epoch + 1}.pth.tar")
                if os.path.exists(prev_chp_path):
                    os.remove(prev_chp_path)

        if scheduler is not None:
            scheduler.step()
            
    last_chp_path = os.path.join(checkpoint_dir, f"last_{model_name}_{end_epoch}.pth.tar")
    last_miou = val_mious[-1] if len(val_mious) > 0 else None
    last_ious = val_ious[-1] if len(val_ious) > 0 else None
    save_checkpoint(last_chp_path, end_epoch, model, optimizer=optimizer, scheduler=scheduler, miou=last_miou, ious=last_ious)

    res = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        **train_task_specific_losses,
        "train_mious": train_mious,
        "val_mious": val_mious,
        "train_ious": train_ious,
        "val_ious": val_ious
    }
    
    if validloader_adj is not None:
        res["val_losses_adj"] = val_losses_adj
        res["val_mious_adj"] = val_mious_adj
        res["val_ious_adj"] = val_ious_adj
    
    return res