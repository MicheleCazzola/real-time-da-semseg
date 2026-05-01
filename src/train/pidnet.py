import torch
from torch.optim import lr_scheduler
import os
import logging

from src.losses.focal import FocalLoss
from src.losses.ohem import OhemCrossEntropy
from src.metrics.metrics import compute_iou
from src.losses.bondary import BondaryLoss
from src.losses.pidnet import PIDNetLoss, PIDNetCrossEntropy
from src.models.pidnet import PIDNet
from src.utils.utils import save_checkpoint


def get_pidnet(model_name, num_classes, pretrained_weights, imgnet_pretrained) -> PIDNet:

    if 's' in model_name:
        model = PIDNet(m=2, n=3, num_classes=num_classes, planes=32, ppm_planes=96, head_planes=128, augment=True)
    elif 'm' in model_name:
        model = PIDNet(m=2, n=3, num_classes=num_classes, planes=64, ppm_planes=96, head_planes=128, augment=True)
    else:
        model = PIDNet(m=3, n=4, num_classes=num_classes, planes=64, ppm_planes=112, head_planes=256, augment=True)

    if imgnet_pretrained:
        pretrained_state = torch.load(pretrained_weights, map_location='cpu')
        model_dict = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if (k in model_dict and v.shape == model_dict[k].shape)}
        model_dict.update(pretrained_state)
        msg = 'Loaded {} parameters!'.format(len(pretrained_state))
        logging.info('Attention!!!')
        logging.info(msg)
        logging.info('Over!!!')
        model.load_state_dict(model_dict, strict = False)
    else:
        pretrained_dict = torch.load(pretrained_weights, map_location='cpu')
        model_dict = model.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
        msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
        logging.info('Attention!!!')
        logging.info(msg)
        logging.info('Over!!!')
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict = False)

    return model


def get_pred_model(name, num_classes):

    if 's' in name:
        model = PIDNet(m=2, n=3, num_classes=num_classes, planes=32, ppm_planes=96, head_planes=128, augment=False)
    elif 'm' in name:
        model = PIDNet(m=2, n=3, num_classes=num_classes, planes=64, ppm_planes=96, head_planes=128, augment=False)
    else:
        model = PIDNet(m=3, n=4, num_classes=num_classes, planes=64, ppm_planes=112, head_planes=256, augment=False)

    return model


def pidnet_model_setup(cfg, device): 
    
    if cfg.training.optimizer == "SGD":
        assert cfg.training.momentum is not None, "Momentum value must be provided for SGD optimizer."
    
    pretrained_weights = os.path.join(cfg.path.weights, f"{cfg.model.model}.pth")
    
    match cfg.training.criterion:
        case "cross_entropy":
            sem_loss = PIDNetCrossEntropy(weight=cfg.training.loss_weights, ignore_label=cfg.model.ignore_index)
        case "ohem":
            sem_loss = OhemCrossEntropy(weight=cfg.training.loss_weights, ignore_label=cfg.model.ignore_index, thresh=0.7, min_kept=100000)
        case "focal":
            sem_loss = FocalLoss(weight=cfg.training.loss_weights, ignore_label=cfg.model.ignore_index, gamma=2.0)
        case _:
            raise ValueError(f"Unsupported loss type: {cfg.training.criterion}")
    
    bd_loss = BondaryLoss()
    
    model = get_pidnet(cfg.model.model, cfg.model.num_classes, pretrained_weights, imgnet_pretrained=True)
    criterion = PIDNetLoss(sem_loss=sem_loss, bd_loss=bd_loss, ignore_index=cfg.model.ignore_index)
    
    model = model.to(device)
    
    match cfg.training.optimizer:
        case "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr = cfg.training.learning_rate, momentum = cfg.training.momentum, weight_decay = cfg.training.weight_decay)
        case "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr = cfg.training.learning_rate, weight_decay = cfg.training.weight_decay)
        case "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr = cfg.training.learning_rate, weight_decay = cfg.training.weight_decay)
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
    
    return model, criterion, optimizer, scheduler


@torch.no_grad()
def evaluate_pidnet(model, num_classes, dataloader, criterion, epoch, tot_epochs, device, log_frequency):

    logging.info(f"PIDNet - Evaluation | Epoch {epoch + 1}/{tot_epochs}")
    
    model.eval()

    tot_loss = 0.0
    data_len, tot_batches = 0, len(dataloader)
    miou, ious = 0.0, torch.zeros(num_classes)

    for i, (inputs, masks, boundaries) in enumerate(dataloader):

        data_len += inputs.size(0)

        inputs = inputs.to(device)
        masks = masks.to(device)
        boundaries = boundaries.to(device)

        # Forward pass
        #loss, outputs, _, [loss_s, loss_b, loss_sb] = model(inputs, masks, boundaries)
        outputs = model(inputs)
        loss_s, loss_b, loss_sb = criterion(outputs, masks, boundaries)
        loss = loss_s + loss_b + loss_sb
        
        tot_loss += loss.item() * inputs.size(0)

        # Calculate mIoU: predictions == outputs[1]
        batch_miou, batch_ious = compute_iou(outputs[1], masks, num_classes)
        miou += batch_miou.item() * inputs.size(0)
        ious += batch_ious.cpu() * inputs.size(0)
        
        if (i + 1) % log_frequency == 0:
            logging.info(f"Epoch {epoch + 1}/{tot_epochs} | Batch {i + 1}/{tot_batches} | Loss: {tot_loss / data_len:.4f} | mIoU (%): {100 * miou / data_len:.2f}")
        
    tot_loss = tot_loss / data_len
    miou = 100 * miou / data_len
    ious = 100 * ious / data_len

    return tot_loss, miou, ious


def train_pidnet(model, num_classes, trainloader, validloader, criterion, optimizer, scheduler, num_epochs, checkpoint_dir, device, log_frequency):
    
    logging.info("PIDNet - Training")
    logging.info(f"Training epochs: {num_epochs}")
    
    train_losses, val_losses = [], []
    train_losses_s, train_losses_b, train_losses_sb = [], [], []
    train_mious, val_mious = [], []
    train_ious, val_ious = [], []
    best_val_miou, best_epoch = None, None

    for epoch in range(num_epochs):

        data_len, tot_batches = 0, len(trainloader)
        train_loss, train_loss_s, train_loss_b, train_loss_sb = 0.0, 0.0, 0.0, 0.0
        train_epoch_miou, train_epoch_ious = 0.0, torch.zeros(num_classes)
        
        model.train()

        for i, (inputs, masks, boundaries) in enumerate(trainloader):

            inputs = inputs.to(device)
            masks = masks.to(device)
            boundaries = boundaries.to(device)
            data_len += inputs.size(0)

            # Forward pass
            optimizer.zero_grad()
            #loss, outputs, _, [loss_s, loss_b, loss_sb] = model(inputs, masks, boundaries)
            outputs = model(inputs)
            loss_s, loss_b, loss_sb = criterion(outputs, masks, boundaries)
            loss = loss_s + loss_b + loss_sb
            
            train_loss += loss.item() * inputs.size(0)
            train_loss_s += loss_s.item() * inputs.size(0)
            train_loss_b += loss_b.item() * inputs.size(0)
            train_loss_sb += loss_sb.item() * inputs.size(0)

            # Backward pass
            loss.backward()
            optimizer.step()
            
            # mIoU: predictions == outputs[1]
            batch_miou, batch_ious = compute_iou(outputs[1], masks, num_classes)
            train_epoch_miou += batch_miou.item() * inputs.size(0)
            train_epoch_ious += batch_ious.cpu() * inputs.size(0)

            if (i + 1) % log_frequency == 0:
                logging.info(f"Epoch {epoch + 1}/{num_epochs} | Batch {i + 1}/{tot_batches} | Loss: {train_loss / data_len:.4f} (Semantic: {train_loss_s / data_len:.4f}, Boundary: {train_loss_b / data_len:.4f}, BAS: {train_loss_sb / data_len:.4f}) | mIoU (%): {100 * train_epoch_miou / data_len:.2f}")

        train_loss = train_loss / data_len
        train_loss_s = train_loss_s / data_len
        train_loss_b = train_loss_b / data_len
        train_loss_sb = train_loss_sb / data_len
        train_epoch_miou = 100 * train_epoch_miou / data_len
        train_epoch_ious = 100 * train_epoch_ious / data_len

        val_loss, val_epoch_miou, val_epoch_ious = evaluate_pidnet(model, num_classes, validloader, criterion, epoch, num_epochs, device, log_frequency)

        #mious_per_category = get_mious_per_category(mious_per_class)

        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        train_losses_s.append(float(train_loss_s))
        train_losses_b.append(float(train_loss_b))
        train_losses_sb.append(float(train_loss_sb))
        train_mious.append(float(train_epoch_miou))
        val_mious.append(float(val_epoch_miou))
        train_ious.append(train_epoch_ious.tolist())
        val_ious.append(val_epoch_ious.tolist())
        
        logging.info(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} (Semantic: {train_loss_s:.4f}, Boundary: {train_loss_b:.4f}, BAS: {train_loss_sb:.4f}) | Train mIoU (%): {train_epoch_miou:.2f} | Val Loss: {val_loss:.4f} | Val mIoU (%): {val_epoch_miou:.2f}")
        
        if best_val_miou is None or val_epoch_miou > best_val_miou:
            prev_best_epoch = best_epoch
            best_val_miou = val_epoch_miou
            best_epoch = epoch
            chp_path = os.path.join(checkpoint_dir, f"best_pidnet_{epoch + 1}.pth.tar")
            save_checkpoint(chp_path, epoch, model, optimizer, scheduler, val_epoch_miou, val_epoch_ious)
            
            # Remove previous best checkpoint
            if prev_best_epoch is not None:
                prev_chp_path = os.path.join(checkpoint_dir, f"best_pidnet_{prev_best_epoch + 1}.pth.tar")
                if os.path.exists(prev_chp_path):
                    os.remove(prev_chp_path)

        # Scheduler is None if learning rate is constant
        if scheduler is not None:
            scheduler.step()
            
    # Save last epoch checkpoint
    last_chp_path = os.path.join(checkpoint_dir, f"last_pidnet_{num_epochs}.pth.tar")
    save_checkpoint(last_chp_path, epoch, model, optimizer, scheduler, val_epoch_miou, val_epoch_ious)
    
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_losses_s": train_losses_s,
        "train_losses_b": train_losses_b,
        "train_losses_sb": train_losses_sb,
        "train_mious": train_mious,
        "val_mious": val_mious,
        "train_ious": train_ious,
        "val_ious": val_ious
    }