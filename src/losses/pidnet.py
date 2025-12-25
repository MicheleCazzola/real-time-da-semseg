import numpy as np
import torch
import torch.nn.functional as F

from src.dataset.dataset import generate_bd
from src.utils.variables import IGNORE_INDEX, device


def pidnet_loss(outputs, labels, sem_loss, bd_loss, bd_gt=None):
    loss_s = sem_loss(outputs[:-1], labels)
    
    if bd_gt is None:
        bd_gt = np.zeros_like(labels.cpu().numpy(), dtype=np.float32)
        for i, m in enumerate(labels):
            bd_gt[i] = generate_bd(m.cpu().numpy().astype(np.uint8))

        bd_gt = torch.from_numpy(bd_gt).to(device)

    loss_b = bd_loss(outputs[-1], bd_gt)

    filler = torch.ones_like(labels) * IGNORE_INDEX   
    bd_label = torch.where(F.sigmoid(outputs[-1][:,0,:,:]) > 0.8, labels, filler)

    loss_sb = sem_loss([outputs[-2]], bd_label)
    
    return loss_s, loss_b, loss_sb