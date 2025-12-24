import torch.nn as nn
import torch.nn.functional as F

class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7, min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def _ce_forward(self, score, target):

        loss = self.criterion(score, target)

        return loss.mean()

    def _ohem_forward(self, score, target, **kwargs):

        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()

        min_value = pred[min(self.min_kept, pred.numel() - 1)]

        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, score, target):

        if not (isinstance(score, list) or isinstance(score, tuple)):
            score = [score]

        balance_weights = [0.4, 1.0]
        sb_weights = 1.0

        if len(balance_weights) == len(score):
            functions = [self._ce_forward] * (len(balance_weights) - 1) + [self._ohem_forward]
            return sum([w * func(x, target) for (w, x, func) in zip(balance_weights, score, functions)])

        elif len(score) == 1:
            return sb_weights * self._ohem_forward(score[0], target)

        else:
            raise ValueError("lengths of prediction and target are not identical!")
