import os
from torch.utils.data import DataLoader
import albumentations as A

from src.models.bisenet import BiSeNet
from src.models.pidnet import PIDNet
from src.models.stdc import STDC
from src.dataset.dataset import LoveDA
from src.utils.variables import TRAIN_DIR, VAL_DIR, IMG_PATH, MASK_PATH
from src.train.bisenet import evaluate_bisenet
from src.train.pidnet import evaluate_pidnet
from src.train.stdc import evaluate_stdc


def trainset_setup(cfg, domain, g, seed_worker, num_workers, augmentations=A.Compose([]), boundaries=False, reduce_factor=1):
    
    downscale = (
        A.Resize(cfg.data.downscale["height"], cfg.data.downscale["width"], p=1)
        if cfg.data.downscale is not None else A.NoOp()
    )
    
    train_transform = A.Compose([
        A.Normalize(mean=cfg.data.imagenet_mean, std=cfg.data.imagenet_std, p=1, max_pixel_value=255),
        downscale,
        augmentations,
        A.Resize(cfg.data.resize["height"], cfg.data.resize["width"], p=1),
        A.ToTensorV2(transpose_mask=True)
    ])
    
    train_root = os.path.join(cfg.path.root, cfg.path.train_dir)
    train_dataset = LoveDA(train_root, cfg.path.images, cfg.path.masks, directories=domain, transforms=train_transform, bd=boundaries, reduce_factor=reduce_factor)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.data.batch_size, shuffle=True, drop_last=True, num_workers=num_workers, worker_init_fn=seed_worker, generator=g
    )
    
    return train_dataset, train_loader

def validset_setup(cfg, domain, num_workers, g, seed_worker, boundaries=False, reduce_factor=1):
    val_transform = A.Compose([
        A.Normalize(mean=cfg.data.imagenet_mean, std=cfg.data.imagenet_std, p=1, max_pixel_value=255),
        A.ToTensorV2(transpose_mask=True)
    ])
    val_root = os.path.join(cfg.path.root, cfg.path.val_dir)
    val_dataset = LoveDA(val_root, cfg.path.images, cfg.path.masks, directories=domain, transforms=val_transform, bd=boundaries, reduce_factor=reduce_factor)
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.data.batch_size, shuffle=False, drop_last=False, num_workers=num_workers, worker_init_fn=seed_worker, generator=g
    )
    
    return val_dataset, val_loader

def evaluate_model(model, validloader, device, criterion):
    
    if isinstance(model, PIDNet):
        val_loss, val_miou, val_mious_per_class = evaluate_pidnet(model, validloader, device)
    elif isinstance(model, BiSeNet):
        val_loss, val_miou, val_mious_per_class = evaluate_bisenet(model, validloader, criterion, device)
    elif isinstance(model, STDC):
        val_loss, val_miou, val_mious_per_class = evaluate_stdc(model, validloader, criterion, device)
    else:
        raise NotImplementedError("ADDA training is only implemented for PIDNet, BiSeNet and STDC models.")
    
    return val_loss,val_miou,val_mious_per_class

def train_forward_source(model, inputs, masks, boundaries, criterion):

    if isinstance(model, PIDNet):
        source_loss, [_, source_output, _], _, _ = model(inputs, masks, boundaries)
        
    elif isinstance(model, BiSeNet):
        outputs, outputs16, outputs32 = model(inputs)
    
        loss1 = criterion(outputs, masks)
        loss2 = criterion(outputs16, masks)
        loss3 = criterion(outputs32, masks)
        
        source_output = outputs
        source_loss = loss1 + loss2 + loss3
        
    elif isinstance(model, STDC):
        # Must be modified when STDC will trained with boundary loss terms
        outputs, outputs16, outputs32 = model(inputs)
    
        loss1 = criterion(outputs, masks)
        loss2 = criterion(outputs16, masks)
        loss3 = criterion(outputs32, masks)
        
        source_output = outputs
        source_loss = loss1 + loss2 + loss3
    else:
        raise NotImplementedError("ADDA training is only implemented for PIDNet, BiSeNet and STDC models.")
    
    return source_loss, source_output

def train_forward_target(model, inputs, return_all=False):

    if isinstance(model, PIDNet):
        _, [target_outputs], _, _ = model(inputs, None, None)
        target_output = target_outputs[1]
        
    elif isinstance(model, BiSeNet):
        target_outputs = model(inputs)
        target_output = target_outputs[0]
        
    elif isinstance(model, STDC):
        # Must be modified when STDC will trained with boundary loss terms
        target_outputs = model(inputs)
        target_output = target_outputs[0]
    else:
        raise NotImplementedError("ADDA training is only implemented for PIDNet, BiSeNet and STDC models.")
    
    return (target_outputs, target_output) if return_all else target_output