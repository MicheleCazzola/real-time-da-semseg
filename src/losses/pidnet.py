import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset.dataset import generate_bd

def pidnet_loss(outputs, labels, sem_loss, bd_loss, ignore_index=-1, bd_gt=None):
    loss_s = sem_loss(outputs[:-1], labels)
    
    if bd_gt is None:
        bd_gt = np.zeros_like(labels.cpu().numpy(), dtype=np.float32)
        for i, m in enumerate(labels):
            bd_gt[i] = generate_bd(m.cpu().numpy().astype(np.uint8))

        bd_gt = torch.from_numpy(bd_gt).to(labels.device)

    loss_b = bd_loss(outputs[-1], bd_gt)

    filler = torch.ones_like(labels) * ignore_index
    bd_label = torch.where(F.sigmoid(outputs[-1][:,0,:,:]) > 0.8, labels, filler)

    loss_sb = sem_loss([outputs[-2]], bd_label)
    
    return loss_s, loss_b, loss_sb

class PIDNetCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(PIDNetCrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label
        )

    def _forward(self, score, target):

        loss = self.criterion(score, target)

        return loss

    def forward(self, score, target):

        # From original configs
        balance_weights = [0.4, 1.0]
        sb_weights = 1.0

        if len(balance_weights) == len(score):
            return sum([w * self._forward(x, target) for (w, x) in zip(balance_weights, score)])
        if len(score) == 1:
            return sb_weights * self._forward(score[0], target)
        
        raise ValueError("Lengths of prediction and target are not identical")

class PIDNetLoss(nn.Module):

    def __init__(self, sem_loss, bd_loss, ignore_index=-1):
        super(PIDNetLoss, self).__init__()
        self.sem_loss = sem_loss
        self.bd_loss = bd_loss
        self.ignore_index = ignore_index

    def forward(self, outputs, labels, bd_gt):
        
        assert labels is not None, "Labels must be provided for loss computation."
        
        # Resize outputs to match labels if necessary
        for i in range(len(outputs)):
            pw, ph = outputs[i].size(3), outputs[i].size(2)
            w, h = labels.size(2), labels.size(1)
            if pw != w or ph != h:
                outputs[i] = F.interpolate(
                    outputs[i], size=(h, w), mode='bilinear', align_corners=True
                )

        # Generate boundary ground truth if not provided (semi-supervised setting)
        if bd_gt is None:
            bd_gt = np.zeros_like(labels.cpu().numpy(), dtype=np.float32)
            for i, m in enumerate(labels):
                bd_gt[i] = generate_bd(m.cpu().numpy().astype(np.uint8))

            bd_gt = torch.from_numpy(bd_gt).to(labels.device)
            
        out_p, out_i, out_d = tuple(outputs)
    
        # Semantic loss
        loss_s = self.sem_loss([out_p, out_i], labels)
        
        # Boundary loss
        loss_b = self.bd_loss(out_d, bd_gt)
        
        # BAS loss
        filler = torch.ones_like(labels) * self.ignore_index
        bd_label = torch.where(F.sigmoid(out_d[:,0,:,:]) > 0.8, labels, filler)
        loss_sb = self.sem_loss([out_i], bd_label)

        return loss_s, loss_b, loss_sb