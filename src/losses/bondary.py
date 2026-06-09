import torch
import torch.nn.functional as F
import torch.nn as nn

def weighted_bce(bd_pre, target, mask=None, ignore_index=-1):
    log_p = bd_pre.permute(0,2,3,1).contiguous().view(1, -1)
    target_t = target.view(1, -1)

    pos_index = (target_t == 1)
    neg_index = (target_t == 0)

    if mask is not None:
        mask_t = mask.contiguous().view(1, -1)
        valid = (mask_t != ignore_index)
        pos_index = pos_index & valid
        neg_index = neg_index & valid

    weight = torch.zeros_like(log_p)
    pos_num = pos_index.float().sum()
    neg_num = neg_index.float().sum()
    sum_num = pos_num + neg_num
    
    pos_weight = neg_num / (sum_num + 1e-6)
    neg_weight = pos_num / (sum_num + 1e-6)
    
    weight = torch.where(pos_index, pos_weight, weight)
    weight = torch.where(neg_index, neg_weight, weight)

    loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight=weight, reduction='mean')

    return loss

class BondaryLoss(nn.Module):
    def __init__(self, coeff_bce = 20.0):
        super(BondaryLoss, self).__init__()
        self.coeff_bce = coeff_bce

    def forward(self, bd_pre, bd_gt, mask=None, ignore_index=-1):
        bce_loss = self.coeff_bce * weighted_bce(bd_pre, bd_gt, mask=mask, ignore_index=ignore_index)
        return bce_loss