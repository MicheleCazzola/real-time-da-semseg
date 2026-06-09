import os
from torch.utils.data import DataLoader
import albumentations as A

from src.dataset.dataset import LoveDA

def trainset_setup(cfg, domain, g, seed_worker, num_workers, split_dir=None, img_dir=None, mask_dir=None, resize=True, augmentations=None, boundaries=False, img_names=False, shuffle=True, drop_last=True, reduce_factor=1):
    
    if augmentations is None:
        augmentations = A.NoOp()
    
    downscale = (
        A.Resize(cfg.data.downscale["height"], cfg.data.downscale["width"], p=1)
        if cfg.data.downscale is not None else A.NoOp()
    )
    
    resize_transform = A.Resize(cfg.data.resize["height"], cfg.data.resize["width"], p=1) if resize else A.NoOp()
    
    train_transform = A.Compose([
        A.Normalize(mean=cfg.data.imagenet_mean, std=cfg.data.imagenet_std, p=1, max_pixel_value=255),
        downscale,
        augmentations,
        resize_transform,
        A.ToTensorV2(transpose_mask=True)
    ])
    
    split_dir = split_dir if split_dir is not None else cfg.path.train_dir
    img_dir = img_dir if img_dir is not None else cfg.path.images
    mask_dir = mask_dir if mask_dir is not None else cfg.path.masks
    
    data_root = os.path.join(cfg.path.root, split_dir)
    train_dataset = LoveDA(data_root, img_dir, mask_dir, directories=domain, transforms=train_transform, bd=boundaries, fname=img_names, reduce_factor=reduce_factor)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.data.batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers, worker_init_fn=seed_worker, generator=g
    )
    
    return train_dataset, train_loader

def validset_setup(cfg, domain, num_workers, g, seed_worker, boundaries=False, img_names=False, reduce_factor=1):
    val_transform = A.Compose([
        A.Normalize(mean=cfg.data.imagenet_mean, std=cfg.data.imagenet_std, p=1, max_pixel_value=255),
        A.ToTensorV2(transpose_mask=True)
    ])
    val_root = os.path.join(cfg.path.root, cfg.path.val_dir)
    val_dataset = LoveDA(val_root, cfg.path.images, cfg.path.masks, directories=domain, transforms=val_transform, bd=boundaries, fname=img_names, reduce_factor=reduce_factor)
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.data.batch_size, shuffle=False, drop_last=False, num_workers=num_workers, worker_init_fn=seed_worker, generator=g
    )
    
    return val_dataset, val_loader