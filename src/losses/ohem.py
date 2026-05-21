import torch
import torch.nn as nn
import torch.nn.functional as F

class OHEMCrossEntropy(nn.Module):
    def __init__(self, ignore_index=-1, thres=0.7, min_kept=100000, weight=None):
        super(OHEMCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_index = ignore_index
        self.weight = weight

    def forward(self, score, target):
        
        if score.ndim == 4:
            score = score.permute(0, 2, 3, 1).contiguous().view(-1, score.size(1))
        if target.ndim == 3:
            target = target.view(-1)
        
        # Cross-entropy loss for each pixel
        pixel_losses = F.cross_entropy(
            score, 
            target, 
            weight=self.weight, 
            ignore_index=self.ignore_index, 
            reduction='none'
        ).view(-1)
        
        # Mask out ignored pixels
        mask = (target != self.ignore_index).view(-1)
        valid_pixels = mask.sum()
        
        if valid_pixels == 0:
            return torch.tensor(0.0, device=score.device)

        # Ground truth confidences
        pred = F.softmax(score, dim=1)
        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_index] = 0
        
        confidences = pred.gather(1, tmp_target.unsqueeze(1)).view(-1)
        confidences = confidences[mask]
        valid_losses = pixel_losses[mask]
        
        # Sort: low confidence = hard example
        confidences, ind = confidences.sort()
        valid_losses = valid_losses[ind]
        
        # Determine threshold
        kept_limit = min(self.min_kept, confidences.numel())
        if kept_limit > 0:
            min_value = confidences[kept_limit - 1]
            threshold = max(min_value.item(), self.thresh)
        else:
            threshold = self.thresh
            
        # Mask for hard examples
        ohem_mask = confidences < threshold
        kept_losses = valid_losses[ohem_mask]
        
        # Average over hard examples if any, otherwise average over all valid examples
        if kept_losses.numel() > 0:
            return kept_losses.mean()
        else:
            return valid_losses.mean()
