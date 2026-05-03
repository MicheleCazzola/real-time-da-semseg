import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from src.metrics.metrics import compute_iou
from src.models.deeplab_v2 import get_deeplab_v2
from src.utils.utils import save_checkpoint, resume_checkpoint

def deeplab_v2_model_setup(cfg, device):
    pretrained_model_path = os.path.join(cfg.path.weights, f"{cfg.model.model}.pth")
    model = get_deeplab_v2(
        num_classes=cfg.model.num_classes, pretrain=True, pretrain_model_path=pretrained_model_path
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: cfg.training.gamma ** (epoch // cfg.training.step_size))

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.model.ignore_index)
    
    return model, criterion, optimizer, scheduler


def train_deeplab_v2(model, num_classes, trainloader, num_epochs, criterion, optimizer, scheduler, device, new_chp_path, resume_training, resume_path, log_frequency=25):
    
    logging.info("DeeplabV2 - Training")

    if resume_training:
        start_epoch, model, optimizer, scheduler = resume_checkpoint(resume_path, model, optimizer, scheduler)
        end_epoch = start_epoch + num_epochs
        
        logging.info(f"Resuming training from checkpoint: {resume_path}")
        logging.info(f"Resumed epoch: {start_epoch} | Training epochs: {num_epochs} | Total epochs: {end_epoch}")
    else:
        start_epoch, end_epoch = 0, num_epochs
        logging.info(f"Starting training from scratch")
        logging.info(f"Training epochs: {num_epochs}")
        
    train_losses, train_mious, train_ious = [], [], []

    for epoch in range(start_epoch, end_epoch):
        model.train()
        epoch_loss = 0.0
        data_len, tot_batches = 0, len(trainloader)
        epoch_miou, epoch_ious = 0, torch.zeros(num_classes)
        for i, (images, masks) in enumerate(trainloader):
            images = images.to(device)
            masks = masks.to(device)
            data_len += images.size(0)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)
            
            # mIoU
            batch_miou, batch_ious = compute_iou(outputs, masks, num_classes)
            epoch_miou += batch_miou.item() * images.size(0)
            epoch_ious += batch_ious.cpu() * images.size(0)

            if (i + 1) % log_frequency == 0:
                logging.info(f"Epoch {epoch + 1}/{end_epoch} | Batch {i + 1}/{tot_batches} | Loss: {epoch_loss / (i + 1):.4f}")

        epoch_loss = epoch_loss / data_len
        epoch_miou = 100 * epoch_miou / data_len
        epoch_ious = 100 * epoch_ious / data_len
        
        train_losses.append(epoch_loss)
        train_mious.append(float(epoch_miou))
        train_ious.append(epoch_ious.tolist())
        
        logging.info(f"Epoch {epoch + 1}/{end_epoch} | Loss: {epoch_loss:.4f} | mIoU (%): {epoch_miou:.2f}%")
        
        chp_name = f"{new_chp_path.split('.pth.tar')[0]}_{epoch + 1}.pth.tar"
        save_checkpoint(chp_name, epoch, model, optimizer=optimizer, scheduler=scheduler)

        if scheduler is not None:
            scheduler.step()
            
    return train_losses, train_mious, train_ious


@torch.no_grad()
def evaluate_deeplab_v2(model, dataloader, criterion, num_classes, device, chp_path, start_epoch, num_epochs, log_frequency=25):
    
    logging.info("DeeplabV2 - Evaluation")
    logging.info(f"Evaluating checkpoint: {chp_path}")
    logging.info(f"Start epoch: {start_epoch + 1} | Num epochs: {num_epochs}")
    
    losses, mious, ious = [], [], []
    best_miou = None

    for epoch in range(start_epoch, start_epoch + num_epochs):

        chp_path_epoch = f"{chp_path.split('.pth.tar')[0]}_{epoch + 1}.pth.tar"
        chp_epoch, model, _, _ = resume_checkpoint(chp_path_epoch, model)
        
        assert chp_epoch == epoch, f"Checkpoint epoch {chp_epoch} does not match expected epoch {epoch}"

        model.to(device)
        
        epoch_loss = 0
        data_len, tot_batches = 0, len(dataloader)
        epoch_miou, epoch_ious = 0, torch.zeros(num_classes)
        
        model.eval()
        for i, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)
            data_len += images.size(0)

            # loss
            outputs = model(images)
            loss = criterion(outputs, masks)
            epoch_loss += loss.item() * images.size(0)

            # mIoU
            batch_miou, batch_ious = compute_iou(outputs, masks, num_classes)
            epoch_miou += batch_miou.item() * images.size(0)
            epoch_ious += batch_ious.cpu() * images.size(0)
            
            if (i + 1) % log_frequency == 0:
                logging.info(f"Epoch {epoch + 1}/{start_epoch + num_epochs} | Batch {i + 1}/{tot_batches} | Loss: {epoch_loss / (i + 1):.4f} | mIoU (%): {100 * epoch_miou / data_len:.2f}")
        
        epoch_miou = 100 * epoch_miou / data_len
        epoch_ious = 100 * epoch_ious / data_len
        epoch_loss = epoch_loss / data_len
        
        losses.append(epoch_loss)
        mious.append(float(epoch_miou))
        ious.append(epoch_ious.tolist())

        logging.info(f"Epoch: {epoch + 1}/{(start_epoch + num_epochs)} | Loss: {epoch_loss:.4f} | mIoU (%): {epoch_miou:.2f}%")
        
        if best_miou is None or epoch_miou > best_miou:
            best_miou = epoch_miou
            best_chp_path = f"{chp_path.split('.pth.tar')[0]}_best.pth.tar"
            save_checkpoint(best_chp_path, epoch, model, optimizer=None, scheduler=None, miou=epoch_miou, ious=epoch_ious)
            logging.info(f"New best mIoU: {best_miou:.2f}% | Best checkpoint saved at: {best_chp_path}")
        
    return losses, mious, ious