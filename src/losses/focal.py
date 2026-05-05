import torch
import torch.nn as nn   
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, ignore_index=-1):
        super(FocalLoss, self).__init__()
        
        assert gamma >= 0, "Gamma should be non-negative"
        assert alpha is None or (isinstance(alpha, torch.Tensor) and alpha.dim() == 1), "Alpha should be a 1D tensor or None"
        
        self.ignore_index = ignore_index
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, score, target):

        ce_loss = F.cross_entropy(score, target, reduction='none', ignore_index=self.ignore_index)
        
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # Alpha-balancing 
        if self.alpha is not None:
            self.alpha = self.alpha.to(target.device)
            alpha_t = self.alpha[torch.clamp(target, min=0)]
            focal_loss = alpha_t * focal_loss

        # Mask out ignored labels
        valid_mask = (target != self.ignore_index)
        
        if valid_mask.sum() > 0:
            return focal_loss[valid_mask].mean()
        else:
            return torch.tensor(0.0, device=score.device)