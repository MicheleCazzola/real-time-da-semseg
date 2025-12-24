from torch.utils.data import DataLoader
import albumentations as A

from models.bisenet import BiSeNet
from models.pidnet import PIDNet
from models.stdc import STDC
from src.dataset.dataset import LoveDA
from src.utils.variables import TRAIN_DIR, VAL_DIR, IMG_PATH, MASK_PATH
from train.bisenet import evaluate_bisenet
from train.pidnet import evaluate_pidnet
from train.stdc import evaluate_stdc


def trainset_setup(avg, std, resize, dir_path, num_workers, batch_size, g, seed_worker, augmentations=A.NoOp(p=1)):
    train_transform = A.Compose([
        A.Normalize(mean=avg, std=std, p=1, always_apply=True, max_pixel_value=255),
        augmentations,
        A.Resize(resize, resize, p=1, always_apply=True)
    ])
    train_dataset = LoveDA(TRAIN_DIR, IMG_PATH, MASK_PATH, directories=dir_path, transforms=train_transform, bd=True)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, worker_init_fn=seed_worker, generator=g
    )
    
    return train_dataset, train_loader

def validset_setup(avg, std, dir_path, num_workers, batch_size, g, seed_worker):
    val_transform = A.Compose([
        A.Normalize(mean=avg, std=std, p=1, always_apply=True, max_pixel_value=255)
    ])
    val_dataset = LoveDA(VAL_DIR, IMG_PATH, MASK_PATH, directories=dir_path, transforms=val_transform, bd=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, worker_init_fn=seed_worker, generator=g
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