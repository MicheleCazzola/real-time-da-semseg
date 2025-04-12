import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, ignore_label=-1, weight=None):
        super(FocalLoss, self).__init__()
        self.ignore_label = ignore_label
        self.gamma = gamma
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label
        )

    def _forward(self, score, target):

        ce_loss = self.criterion(score, target)

        pt = torch.exp(-ce_loss)
        focal_loss = torch.pow(1 - pt, self.gamma)

        if self.alpha is not None:
            return self.alpha * focal_loss
        return focal_loss

    def forward(self, score, target):

        if not (isinstance(score, list) or isinstance(score, tuple)):
            score = [score]

        # From original configs
        balance_weights = [0.4, 1.0]
        sb_weights = 1.0

        if len(balance_weights) == len(score):
            return sum([w * self._forward(x, target) for (w, x) in zip(balance_weights, score)])
        elif len(score) == 1:
            return sb_weights * self._forward(score[0], target)

        else:
            raise ValueError("lengths of prediction and target are not identical!")