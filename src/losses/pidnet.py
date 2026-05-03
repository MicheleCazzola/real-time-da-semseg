import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset.dataset import generate_bd
from .focal import FocalLoss
from .ohem import OHEMCrossEntropy
from src.utils.variables import SemanticLoss

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

class PIDNetSemanticLoss(nn.Module):
    def __init__(self, type=SemanticLoss.CE, ignore_label=-1, **kwargs):
        super(PIDNetSemanticLoss, self).__init__()
        self.ignore_label = ignore_label
        self.loss_type = type
        
        if isinstance(self.loss_type, list):
            assert len(self.loss_type) == 2, "Loss type list must have 2 elements for out_p and out_i"
            self.criterion = [self._get_criterion(branch_loss_type, **kwargs) for branch_loss_type in self.loss_type]
        else:
            self.criterion = self._get_criterion(self.loss_type, **kwargs)
    
    def _get_criterion(self, loss_type, **kwargs):
        match loss_type:
            case SemanticLoss.CE.value:
                criterion = nn.CrossEntropyLoss(
                    weight=kwargs.get('class_weight'),
                    ignore_index=self.ignore_label
                )
            case SemanticLoss.OHEM.value:
                criterion = OHEMCrossEntropy(
                    ignore_label=self.ignore_label,
                    thres=kwargs.get('ohem_thres', 0.7),
                    min_kept=kwargs.get('ohem_min_kept', 100000),
                    weight=kwargs.get('class_weight')
                )
            case SemanticLoss.FOCAL.value:
                criterion = FocalLoss(
                    gamma=kwargs.get('focal_gamma'),
                    alpha=kwargs.get('class_weight'),
                    ignore_label=self.ignore_label
                )
            case _:
                raise ValueError(f"Unsupported loss type: {self.loss_type}")
        
        return criterion

    def forward(self, score, target):

        # From original configs
        balance_weights = [0.4, 1.0]
        sb_weights = 1.0
        
        criterions = self.criterion if isinstance(self.criterion, list) else [self.criterion] * len(score)

        if len(balance_weights) == len(score):
            return sum([w * criterion(x, target) for (w, x, criterion) in zip(balance_weights, score, criterions)])
        if len(score) == 1:
            criterion_i = criterions[-1]
            return sb_weights * criterion_i(score[0], target)
        
        raise ValueError("Lengths of prediction and target are not identical")

class PIDNetLoss(nn.Module):

    def __init__(self, sem_loss, bd_loss, ignore_index=-1):
        super(PIDNetLoss, self).__init__()
        self.sem_loss = sem_loss
        self.bd_loss = bd_loss
        self.ignore_index = ignore_index

    def forward(self, outputs, ground_truth):
        
        assert ground_truth is not None and len(ground_truth) == 2, "Ground truth must be provided for loss computation."
        
        masks, bd_gt = ground_truth
        
        # Resize outputs to match labels if necessary
        for i in range(len(outputs)):
            pw, ph = outputs[i].size(3), outputs[i].size(2)
            w, h = masks.size(2), masks.size(1)
            if pw != w or ph != h:
                outputs[i] = F.interpolate(
                    outputs[i], size=(h, w), mode='bilinear', align_corners=True
                )

        # Generate boundary ground truth if not provided (semi-supervised setting)
        if bd_gt is None:
            bd_gt = np.zeros_like(masks.cpu().numpy(), dtype=np.float32)
            for i, m in enumerate(masks):
                bd_gt[i] = generate_bd(m.cpu().numpy().astype(np.uint8))

            bd_gt = torch.from_numpy(bd_gt).to(masks.device)
            
        out_p, out_i, out_d = tuple(outputs)
    
        # Semantic loss
        loss_s = self.sem_loss([out_p, out_i], masks)
        
        # Boundary loss
        loss_b = self.bd_loss(out_d, bd_gt)
        
        # BAS loss
        filler = torch.ones_like(masks) * self.ignore_index
        bd_label = torch.where(F.sigmoid(out_d[:,0,:,:]) > 0.8, masks, filler)
        loss_sb = self.sem_loss([out_i], bd_label)

        loss = loss_s + loss_b + loss_sb
        
        return loss, {
            "semantic": loss_s,
            "boundary": loss_b,
            "bas": loss_sb
        }