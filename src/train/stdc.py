import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from src.metrics.metrics import compute_iou
from src.models.stdc import STDC
from src.utils.utils import save_checkpoint

class DetailAggregateLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DetailAggregateLoss, self).__init__()
        
        self.laplacian_kernel = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32
        ).reshape(1, 1, 3, 3).requires_grad_(False).type(torch.FloatTensor)
        
        self.fuse_kernel = torch.nn.Parameter(
            torch.tensor([[6./10], [3./10], [1./10]],
            dtype=torch.float32
        ).reshape(1, 3, 1, 1).type(torch.FloatTensor))
        
    def dice_loss(self, input, target):
        smooth = 1.0
        n = input.size(0)
        iflat = input.view(n, -1)
        tflat = target.view(n, -1)
        
        intersection = (iflat * tflat).sum(1)
        loss = 1 - ((
            2. * intersection + smooth) /
            (iflat.sum(1) + tflat.sum(1) + smooth
        ))
        
        return loss.mean()

    def forward(self, boundary_logits, gtmasks):

        # boundary_logits = boundary_logits.unsqueeze(1)
        boundary_targets = F.conv2d(gtmasks.unsqueeze(1).type(torch.FloatTensor), self.laplacian_kernel, padding=1)
        boundary_targets = boundary_targets.clamp(min=0)
        boundary_targets[boundary_targets > 0.1] = 1
        boundary_targets[boundary_targets <= 0.1] = 0

        boundary_targets_x2 = F.conv2d(gtmasks.unsqueeze(1).type(torch.FloatTensor), self.laplacian_kernel, stride=2, padding=1)
        boundary_targets_x2 = boundary_targets_x2.clamp(min=0)
        
        boundary_targets_x4 = F.conv2d(gtmasks.unsqueeze(1).type(torch.FloatTensor), self.laplacian_kernel, stride=4, padding=1)
        boundary_targets_x4 = boundary_targets_x4.clamp(min=0)

        boundary_targets_x8 = F.conv2d(gtmasks.unsqueeze(1).type(torch.FloatTensor), self.laplacian_kernel, stride=8, padding=1)
        boundary_targets_x8 = boundary_targets_x8.clamp(min=0)
    
        boundary_targets_x8_up = F.interpolate(boundary_targets_x8, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x4_up = F.interpolate(boundary_targets_x4, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x2_up = F.interpolate(boundary_targets_x2, boundary_targets.shape[2:], mode='nearest')
        
        boundary_targets_x2_up[boundary_targets_x2_up > 0.1] = 1
        boundary_targets_x2_up[boundary_targets_x2_up <= 0.1] = 0
        
        boundary_targets_x4_up[boundary_targets_x4_up > 0.1] = 1
        boundary_targets_x4_up[boundary_targets_x4_up <= 0.1] = 0
        
        boundary_targets_x8_up[boundary_targets_x8_up > 0.1] = 1
        boundary_targets_x8_up[boundary_targets_x8_up <= 0.1] = 0
        
        boundary_targets_pyramids = torch.stack((boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up), dim=1)
        
        boundary_targets_pyramids = boundary_targets_pyramids.squeeze(2)
        boundary_targets_pyramid = F.conv2d(boundary_targets_pyramids, self.fuse_kernel)

        boundary_targets_pyramid[boundary_targets_pyramid > 0.1] = 1
        boundary_targets_pyramid[boundary_targets_pyramid <= 0.1] = 0
        
        if boundary_logits.shape[-1] != boundary_targets.shape[-1]:
            boundary_logits = F.interpolate(
                boundary_logits, boundary_targets.shape[2:], mode='bilinear', align_corners=True
            )
        
        boundary_targets_pyramid = boundary_targets_pyramid.to(boundary_logits.device)
        bce_loss = F.binary_cross_entropy_with_logits(boundary_logits, boundary_targets_pyramid)
        dice_loss = self.dice_loss(torch.sigmoid(boundary_logits), boundary_targets_pyramid)
        
        return bce_loss, dice_loss

def stdc_model_setup(cfg, backbone_name, device):
    pretrained_model = os.path.join(cfg.path.weights, f"{cfg.model.model}.pth")
    
    # Using same configuration as STDC-Seg paper
    model = STDC(backbone_name, cfg.model.num_classes, pretrain_model=pretrained_model, use_boundary_8=True).to(device)
    
    match cfg.training.criterion:
        case "cross_entropy":
            criterion = nn.CrossEntropyLoss(weight=cfg.training.loss_weights, ignore_index=cfg.model.ignore_index)
        case "ohem":
            raise NotImplementedError("OHEM loss is not implemented yet for STDC")
            criterion = OhemCrossEntropy(weight=cfg.training.loss_weights, ignore_label=cfg.model.ignore_index, thresh=0.7, min_kept=100000)
        case "focal":
            raise NotImplementedError("Focal loss is not implemented yet for STDC")
            criterion = FocalLoss(weight=cfg.training.loss_weights, ignore_index=cfg.model.ignore_index, gamma=2.0)
        case _:
            raise ValueError(f"Unsupported loss type: {cfg.training.criterion}")
        
    detail_criterion = DetailAggregateLoss()

    match cfg.training.optimizer:
        case "SGD":
            optimizer = optim.SGD(model.parameters(), lr=cfg.training.learning_rate, momentum=cfg.training.momentum, weight_decay=cfg.training.weight_decay)
        case "Adam":
            optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
        case "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
        case _:
            raise ValueError(f"Unsupported optimizer type: {cfg.training.optimizer}")
    
    match cfg.training.scheduler:
        case "step_lr":
            scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.training.step_size, gamma=cfg.training.gamma)
        case "poly":
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / cfg.training.epochs) ** cfg.training.power)
        case "cosine":
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs, eta_min=cfg.training.eta_min)
        case _:
            raise NotImplementedError(f"Unsupported scheduler type: {cfg.training.scheduler}")
    
    return model, criterion, detail_criterion, optimizer, scheduler


@torch.no_grad()
def evaluate_stdc(model, num_classes, dataloader, criterion, device, epoch, tot_epochs, log_frequency):
    
    logging.info(f"STDC - Evaluation | Epoch {epoch + 1}/{tot_epochs}")

    model.eval()

    tot_loss = 0.0
    data_len, tot_batches = 0, len(dataloader)
    miou, ious = 0.0, torch.zeros(num_classes)

    for i, (inputs, masks) in enumerate(dataloader):

        data_len += inputs.size(0)

        inputs = inputs.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, masks)

        tot_loss += loss.item() * inputs.size(0)

        # Calculate mIoU
        batch_miou, batch_ious = compute_iou(outputs, masks, num_classes)
        miou += batch_miou * inputs.size(0)
        ious += batch_ious.cpu() * inputs.size(0)
        
        if (i + 1) % log_frequency == 0:
            logging.info(f"Epoch {epoch + 1}/{tot_epochs} | Batch {i + 1}/{tot_batches} | Loss: {tot_loss / data_len:.4f} | mIoU (%): {100 * miou / data_len:.2f}")
        
    tot_loss = tot_loss / data_len
    miou = 100 * miou / data_len
    ious = 100 * ious / data_len

    return tot_loss, miou, ious

def train_stdc(model, num_classes, trainloader, validloader, optimizer, scheduler, criterion, detail_criterion, num_epochs, checkpoint_dir, device, log_frequency):
    
    logging.info("STDC - Training")
    logging.info(f"Training epochs: {num_epochs}")
    
    train_losses, val_losses = [], []
    train_losses_sem, train_losses_aux16, train_losses_aux32, train_losses_boundary_bce, train_losses_boundary_dice = [], [], [], [], []
    train_mious, val_mious = [], []
    train_ious, val_ious = [], []
    best_val_miou, best_epoch = None, None

    for epoch in range(num_epochs):
        
        data_len, tot_batches = 0, len(trainloader)
        train_loss, train_loss_sem, train_loss_aux16, train_loss_aux32, train_loss_boundary_bce, train_loss_boundary_dice = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        train_epoch_miou, train_epoch_ious = 0.0, torch.zeros(num_classes)
        
        model.train()
        
        for i, (inputs, masks) in enumerate(trainloader):
            
            inputs = inputs.to(device)
            masks = masks.to(device)
            data_len += inputs.size(0)

            optimizer.zero_grad()
            
            # Using same configuration as STDC-Seg paper
            outputs, outputs16, outputs32, detail8 = model(inputs)
            
            loss1 = criterion(outputs, masks)
            loss2 = criterion(outputs16, masks)
            loss3 = criterion(outputs32, masks)
            boundary_bce_loss, boundary_dice_loss = detail_criterion(detail8, masks)
            loss = loss1 + loss2 + loss3 + boundary_bce_loss + boundary_dice_loss

            train_loss += loss.item() * inputs.size(0)
            train_loss_sem += loss1.item() * inputs.size(0)
            train_loss_aux16 += loss2.item() * inputs.size(0)
            train_loss_aux32 += loss3.item() * inputs.size(0)
            train_loss_boundary_bce += boundary_bce_loss.item() * inputs.size(0)
            train_loss_boundary_dice += boundary_dice_loss.item() * inputs.size(0)

            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            batch_miou, batch_ious = compute_iou(outputs, masks, num_classes)
            train_epoch_miou += batch_miou.item() * inputs.size(0)
            train_epoch_ious += batch_ious.cpu() * inputs.size(0)

            if (i + 1) % log_frequency == 0:
                logging.info(f"Epoch {epoch + 1}/{num_epochs} | Batch {i + 1}/{tot_batches} | Loss: {train_loss / data_len:.4f} | Semantic: {train_loss_sem / data_len:.4f} | Auxiliary (16): {train_loss_aux16 / data_len:.4f} | Auxiliary (32): {train_loss_aux32 / data_len:.4f} | Boundary BCE: {train_loss_boundary_bce / data_len:.4f} | Boundary Dice: {train_loss_boundary_dice / data_len:.4f} | mIoU (%): {100 * train_epoch_miou / data_len:.2f}")

        train_loss = train_loss / data_len
        train_loss_sem = train_loss_sem / data_len
        train_loss_aux16 = train_loss_aux16 / data_len
        train_loss_aux32 = train_loss_aux32 / data_len
        train_loss_boundary_bce = train_loss_boundary_bce / data_len
        train_loss_boundary_dice = train_loss_boundary_dice / data_len
        train_epoch_miou = 100 * train_epoch_miou / data_len
        train_epoch_ious = 100 * train_epoch_ious / data_len
        
        val_loss, val_epoch_miou, val_epoch_ious = evaluate_stdc(model, num_classes, validloader, criterion, device, epoch, num_epochs, log_frequency)
        
        #mious_per_category = get_mious_per_category(mious_per_class)

        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        train_losses_sem.append(float(train_loss_sem))
        train_losses_aux16.append(float(train_loss_aux16))
        train_losses_aux32.append(float(train_loss_aux32))
        train_losses_boundary_bce.append(float(train_loss_boundary_bce))
        train_losses_boundary_dice.append(float(train_loss_boundary_dice))
        train_mious.append(float(train_epoch_miou))
        val_mious.append(float(val_epoch_miou))
        train_ious.append(train_epoch_ious.tolist())
        val_ious.append(val_epoch_ious.tolist())
        
        logging.info(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} (Semantic: {train_loss_sem:.4f}, Auxiliary (16): {train_loss_aux16:.4f}, Auxiliary (32): {train_loss_aux32:.4f}, Boundary BCE: {train_loss_boundary_bce:.4f}, Boundary Dice: {train_loss_boundary_dice:.4f}) | Train mIoU (%): {train_epoch_miou:.2f} | Val Loss: {val_loss:.4f} | Val mIoU (%): {val_epoch_miou:.2f}")
        
        if best_val_miou is None or val_mious[-1] > best_val_miou:
            prev_best_epoch = best_epoch
            best_val_miou = val_mious[-1]
            best_epoch = epoch
            chp_path = os.path.join(checkpoint_dir, f"best_stdc_{epoch + 1}.pth.tar")
            save_checkpoint(chp_path, epoch, model, optimizer, scheduler, val_epoch_miou, val_epoch_ious)

            # Remove previous best checkpoint
            if prev_best_epoch is not None:
                prev_chp_path = os.path.join(checkpoint_dir, f"best_stdc_{prev_best_epoch + 1}.pth.tar")
                if os.path.exists(prev_chp_path):
                    os.remove(prev_chp_path)
                    
        if scheduler is not None:
            scheduler.step()
    
    # Save last epoch checkpoint
    last_chp_path = os.path.join(checkpoint_dir, f"last_stdc_{num_epochs}.pth.tar")
    save_checkpoint(last_chp_path, epoch, model, optimizer, scheduler, val_epoch_miou, val_epoch_ious)
    
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_losses_sem": train_losses_sem,
        "train_losses_sem_aux16": train_losses_aux16,
        "train_losses_sem_aux32": train_losses_aux32,
        "train_losses_boundary_bce": train_losses_boundary_bce,
        "train_losses_boundary_dice": train_losses_boundary_dice,
        "train_mious": train_mious,
        "val_mious": val_mious,
        "train_ious": train_ious,
        "val_ious": val_ious
    }