"""
    Instance Adaptive Self-Training (IAST) training loop implementation
    Adapted from:
    - github.com/Raykoooo/IAST
    - github.com/Junjue-Wang/LoveDA
"""

import os
from matplotlib import pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
import logging
import numpy as np

from torch.nn.utils import clip_grad
from tqdm import tqdm
from src.metrics.mean_iou import MeanIoU
from src.models.discriminator import FCDiscriminator
from src.train.train_model import evaluate_model
from src.dataset.utils import trainset_setup
from src.utils.utils import save_checkpoint

class EntropyLoss(torch.nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, logits, weight=None):
        val_num = weight[weight>0].numel()
        logits_log_softmax = torch.log_softmax(logits, dim=1)
        entropy = -torch.softmax(logits, dim=1) * weight * logits_log_softmax
        entropy_reg = torch.sum(entropy) / val_num
        return entropy_reg

class KLDLoss(torch.nn.Module):
    def __init__(self):
        super(KLDLoss, self).__init__()

    def forward(self, logits, weight):
        val_num = weight[weight>0].numel()
        logits_log_softmax = torch.log_softmax(logits, dim=1)
        num_classes = logits.size()[1]
        kld = - 1/num_classes * weight * logits_log_softmax
        kld_reg = torch.sum(kld) / val_num
        return kld_reg

def iast_setup(cfg, device):
    model_d = FCDiscriminator(num_classes=cfg.NUM_CLASSES).to(device)
    criterion_d = torch.nn.BCEWithLogitsLoss()
    criterion_ent = EntropyLoss()
    criterion_kd = KLDLoss()
    optimizer_d = optim.Adam(model_d.parameters(), lr=cfg.LEARNING_RATE_D, betas=(0.9, 0.99))
    scheduler_d = optim.lr_scheduler.LambdaLR(optimizer_d, lambda x: (1 - x / cfg.NUM_STEPS_STOP) ** 0.9)

    return model_d, criterion_d, criterion_ent, criterion_kd, optimizer_d, scheduler_d

def ias_thresh(conf_dict, n_class, alpha, w=None, gamma=1.0):
    if w is None:
        w = np.ones(n_class)
    # threshold
    cls_thresh = np.ones(n_class,dtype = np.float32)
    for idx_cls in np.arange(0, n_class):
        if conf_dict[idx_cls] != None:
            arr = np.array(conf_dict[idx_cls])
            cls_thresh[idx_cls] = np.percentile(arr, 100 * (1 - alpha * w[idx_cls] ** gamma))
    return cls_thresh

def generate_pseudo(model, dataloader, save_dir, epoch, n_class, pseudo_dict, device):
    logging.info(f'Start generate pseudo labels: {save_dir}')
    os.makedirs(os.path.join(save_dir, f'pseudo_labels_{epoch}'), exist_ok=True)
    model.eval()
    cls_thresh = np.ones(n_class) * 0.9
    for image, _, _, names in tqdm(dataloader, desc="Generating pseudo labels"):
        out = model(image.to(device))
        logits = out[1] if isinstance(out, tuple) or isinstance(out, list) else out
        logits = F.interpolate(logits, size=image.size()[2:], mode='bilinear', align_corners=True)
        max_items = logits.max(dim=1)
        label_pred = max_items[1].data.cpu().numpy()
        logits_pred = max_items[0].data.cpu().numpy()

        logits_cls_dict = {c: [cls_thresh[c]] for c in range(n_class)}
        for cls in range(n_class):
            logits_cls_dict[cls].extend(logits_pred[label_pred == cls].astype(np.float16))
        # instance adaptive selector
        tmp_cls_thresh = ias_thresh(logits_cls_dict, n_class, pseudo_dict['pl_alpha'],  w=cls_thresh, gamma=pseudo_dict['pl_gamma'])
        beta = pseudo_dict['pl_beta']
        cls_thresh = beta*cls_thresh + (1-beta)*tmp_cls_thresh
        cls_thresh[cls_thresh>=1] = 0.999

        np_logits = logits.data.cpu().numpy()
        for _i, fname in enumerate(names):
            # save pseudo label
            logit = np_logits[_i].transpose(1,2,0)
            label = np.argmax(logit, axis=2)
            logit_amax = np.amax(logit, axis=2)
            label_cls_thresh = np.apply_along_axis(lambda x: [cls_thresh[e] for e in x], 1, label)
            ignore_index = logit_amax < label_cls_thresh
            label += 1
            label[ignore_index] = 0
            assert label.shape == image.size()[2:], f"Label shape {label.shape} does not match image shape {image.size()[2:]}"
            plt.imsave(os.path.join(save_dir, f'pseudo_labels_{epoch}', fname), label.astype(np.uint8))
            #logging.info(f"Save pseudo label: {os.path.join(save_dir, f'pseudo_labels_{epoch}', fname)}")

    return f"pseudo_labels_{epoch}"

def train_iast(
    model, model_name, model_D, trainloader, trainloader_target_base, validloader, 
    criterion, criterion_D, criterion_ent, criterion_kd, optimizer, optimizer_D, scheduler, scheduler_D, 
    epochs, bd_required, cfg, iast_cfg, trainset_build_params, device, checkpoint_dir, regenerate, log_frequency
):
    
    pseudo_dir = os.path.join(cfg.path.root, "Train", cfg.path.target)
    if regenerate:
        # Generate pseudo labels at the beginning of training
        pseudo_pred_dir = generate_pseudo(model, trainloader_target_base, pseudo_dir, 0, iast_cfg.NUM_CLASSES, pseudo_dict=iast_cfg.PSEUDO_DICT, device=device)
    else:
        pseudo_pred_dir = "pseudo_labels_0"  # Assuming pseudo labels from epoch 0 are already generated and available
    
    _, trainloader_target = trainset_setup(
        cfg, cfg.path.target,
        trainset_build_params['g'], trainset_build_params['seed_worker'], trainset_build_params['num_workers'],
        mask_dir=pseudo_pred_dir, augmentations=trainset_build_params['augmentations'],
        boundaries=bd_required, reduce_factor=trainset_build_params['reduce_factor']
    )

    train_seg_losses, train_target_seg_losses, train_adv_losses, train_ent_reg_losses, train_kd_losses, train_disc_losses = [], [], [], [], [], []
    train_mious, train_ious = [], []
    val_losses, val_mious, val_ious = [], [], []
    best_epoch, best_val_miou = -1, None
    
    metric = MeanIoU(iast_cfg.NUM_CLASSES).to(device)
    for epoch in range(epochs):
        logging.info(f"Epoch {epoch + 1}/{epochs}")
        
        data_len, tot_batches = 0, min(len(trainloader), len(trainloader_target))
        epoch_seg_loss, epoch_target_seg_loss, epoch_adv_loss, epoch_ent_reg_loss, epoch_kd_reg_loss, epoch_disc_loss = 0, 0, 0, 0, 0, 0
        
        model.train()
        model_D.train()
        metric.reset()
        for i, (batch_source, batch_target) in enumerate(zip(trainloader, trainloader_target)):

            images_s, masks_s, boundaries_s = batch_source
            
            data_len += images_s.size(0)
            
            images_s = images_s.to(device)
            masks_s = masks_s.to(device)
            boundaries_s = boundaries_s.to(device)
            
            out_source = model(images_s)
            out_source_main = out_source[1] if isinstance(out_source, tuple) or isinstance(out_source, list) else out_source # for PIDNet, the second output is the intermediate feature map used for adversarial training

            images_t, masks_t, boundaries_t = batch_target
            
            images_t = images_t.to(device)
            masks_t = masks_t.to(device)
            boundaries_t = boundaries_t.to(device)
            
            out_target = model(images_t)
            out_target_main = out_target[1] if isinstance(out_target, tuple) or isinstance(out_target, list) else out_target # for PIDNet, the second output is the intermediate feature map used for adversarial training

            # defaut reg_weight
            if iast_cfg.DISCRIMINATOR['lambda_entropy_weight'] or iast_cfg.DISCRIMINATOR['lambda_kldreg_weight']:
                reg_val_matrix = torch.ones_like(masks_t).type_as(out_target_main)
                reg_val_matrix[masks_t == -1] = 0
                reg_val_matrix = reg_val_matrix.unsqueeze(dim=1)
                reg_ignore_matrix = 1 - reg_val_matrix
                reg_weight = torch.ones_like(out_target_main)
                reg_weight_val = reg_weight * reg_val_matrix
                reg_weight_ignore = reg_weight * reg_ignore_matrix
                del reg_ignore_matrix, reg_weight, reg_val_matrix

            loss_dict = dict()

            # forward discriminators
            s_D_logits = model_D(out_source_main.softmax(dim=1).detach())
            t_D_logits = model_D(out_target_main.softmax(dim=1).detach())

            is_source = torch.zeros_like(s_D_logits).to(device)
            is_target = torch.ones_like(t_D_logits).to(device)
            discriminator_loss = (criterion_D(s_D_logits, is_source) + criterion_D(t_D_logits, is_target)) / 2
            epoch_disc_loss += discriminator_loss.item() * images_s.size(0)
            
            # adv_losses
            t_D_logits = model_D(out_target_main.softmax(dim=1).detach())
            is_source = torch.zeros_like(t_D_logits).to(device)
            adv_loss = criterion_D(t_D_logits, is_source)
            epoch_adv_loss += adv_loss.item() * images_s.size(0)
            loss_dict['adv_loss'] = iast_cfg.DISCRIMINATOR['weight'] * adv_loss

            # update seg loss
            seg_loss = criterion(out_source, (masks_s, boundaries_s) if bd_required else masks_s)[0]
            epoch_seg_loss += seg_loss.item() * images_s.size(0)
            loss_dict['seg_loss'] = iast_cfg.SOURCE_LOSS_WEIGHT * seg_loss

            # pseudo label target seg loss
            target_seg_loss = criterion(out_target, (masks_t, boundaries_t) if bd_required else masks_t)[0]
            epoch_target_seg_loss += target_seg_loss.item() * images_s.size(0)
            loss_dict['target_seg_loss'] = iast_cfg.PSEUDO_LOSS_WEIGHT * target_seg_loss

            # entropy reg
            if iast_cfg.DISCRIMINATOR['lambda_entropy_weight'] > 0:
                entropy_reg_loss = criterion_ent(out_target_main, reg_weight_ignore)
                epoch_ent_reg_loss += entropy_reg_loss.item() * images_s.size(0)
                entropy_reg_loss =  entropy_reg_loss * iast_cfg.DISCRIMINATOR['lambda_entropy_weight']
                loss_dict['entropy_reg_loss'] = entropy_reg_loss
            # kld reg
            if iast_cfg.DISCRIMINATOR['lambda_kldreg_weight'] > 0:
                kld_reg_loss = criterion_kd(out_target_main, reg_weight_val)
                epoch_kd_reg_loss += kld_reg_loss.item() * images_s.size(0)
                kld_reg_loss =  kld_reg_loss * iast_cfg.DISCRIMINATOR['lambda_kldreg_weight']
                loss_dict['kld_reg_loss'] = kld_reg_loss

            # backward model
            optimizer.zero_grad()
            total_loss = sum(loss_dict.values())
            total_loss.backward()
            clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=35, norm_type=2)
            optimizer.step()

            # backward model_D
            optimizer_D.zero_grad()
            clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model_D.parameters()), max_norm=35, norm_type=2)
            discriminator_loss.backward()
            optimizer_D.step()
            
            metric.update(out_source_main, masks_s)
            current_miou, _ = metric.compute()

            if (i + 1) % log_frequency == 0:
                lr = scheduler.get_last_lr()[0]
                lr_D = scheduler_D.get_last_lr()[0]
                text = f'UDA iter = {i + 1}/{tot_batches} '
                for k, v in loss_dict.items():
                    text += f'{k} = {v:.3f} '
                text += f'd_loss = {discriminator_loss.item():.3f} '
                text += f'lr = {lr:.3f} '
                text += f'd_lr = {lr_D:.3f} '
                text += f'mIoU = {100 * current_miou.item():.2f} '
                logging.info(text)

        epoch_seg_loss = epoch_seg_loss / data_len
        epoch_target_seg_loss = epoch_target_seg_loss / data_len
        epoch_adv_loss = epoch_adv_loss / data_len
        epoch_ent_reg_loss = epoch_ent_reg_loss / data_len
        epoch_kd_reg_loss = epoch_kd_reg_loss / data_len 
        epoch_disc_loss = epoch_disc_loss / data_len
        train_epoch_miou, train_epoch_ious = map(
            lambda x: x * 100, metric.compute()
        )
        train_epoch_miou = train_epoch_miou.item()
        train_epoch_ious = train_epoch_ious.tolist()
        
        val_loss, val_epoch_miou, val_epoch_ious = evaluate_model(model, model_name, iast_cfg.NUM_CLASSES, validloader, criterion, bd_required, epoch, epochs, device, log_frequency)
        
        logging.info(f"Epoch {epoch + 1}/{epochs} | Train Seg Loss: {epoch_seg_loss:.4f} | Train Target Seg Loss: {epoch_target_seg_loss:.4f} | Train Adv Loss: {epoch_adv_loss:.4f} | Train Entropy Reg Loss: {epoch_ent_reg_loss:.4f} | Train KLD Reg Loss: {epoch_kd_reg_loss:.4f} | Train Disc Loss: {epoch_disc_loss:.4f} | Train mIoU (%): {train_epoch_miou:.2f} | Val Loss: {val_loss:.4f} | Val mIoU (%): {val_epoch_miou:.2f}")
        
        train_disc_losses.append(float(epoch_disc_loss))
        train_seg_losses.append(float(epoch_seg_loss))
        train_target_seg_losses.append(float(epoch_target_seg_loss))
        train_adv_losses.append(float(epoch_adv_loss))
        train_ent_reg_losses.append(float(epoch_ent_reg_loss))
        train_kd_losses.append(float(epoch_kd_reg_loss))
        train_mious.append(float(train_epoch_miou))
        train_ious.append(train_epoch_ious)
        val_losses.append(float(val_loss))
        val_mious.append(float(val_epoch_miou))
        val_ious.append(val_epoch_ious)
        
        # Checkpointing
        if best_val_miou is None or val_epoch_miou > best_val_miou:
            prev_best_epoch = best_epoch
            best_val_miou = val_epoch_miou
            best_epoch = epoch
            chp_path = os.path.join(checkpoint_dir, f"best_{model_name}_{epoch + 1}.pth.tar")
            save_checkpoint(chp_path, epoch + 1, model, optimizer=optimizer, scheduler=scheduler, miou=val_epoch_miou, ious=val_epoch_ious)
            
            if prev_best_epoch >= 0:
                prev_chp_path = os.path.join(checkpoint_dir, f"best_{model_name}_{prev_best_epoch + 1}.pth.tar")
                if os.path.exists(prev_chp_path):
                    os.remove(prev_chp_path)
            
        if epoch < epochs - 1 and (epoch + 1) % iast_cfg.GENERATE_PSEUDO_EVERY == 0:
            pseudo_dir = os.path.join(cfg.path.root, "Train", cfg.path.target)
            pseudo_pred_dir = generate_pseudo(model, trainloader_target_base, pseudo_dir, epoch + 1, iast_cfg.NUM_CLASSES, pseudo_dict=iast_cfg.PSEUDO_DICT, device=device)
            _, trainloader_target = trainset_setup(
                cfg, cfg.path.target,
                trainset_build_params['g'], trainset_build_params['seed_worker'], trainset_build_params['num_workers'],
                mask_dir=pseudo_pred_dir, augmentations=trainset_build_params['augmentations'],
                boundaries=bd_required, reduce_factor=trainset_build_params['reduce_factor']
            )
        
        if scheduler is not None:
            scheduler.step()
        if scheduler_D is not None:
            scheduler_D.step()
        
    return val_losses, val_mious, val_ious