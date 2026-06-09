import torch.nn as nn

class DeepLabLoss(nn.Module):
    def __init__(self, base_criterion):
        super(DeepLabLoss, self).__init__()
        self.base_criterion = base_criterion

    def forward(self, outputs, targets):
        loss = self.base_criterion(outputs, targets)
        return loss, {}