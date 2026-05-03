import torch
import torch.nn as nn

class BiSeNetLoss(nn.Module):
    def __init__(self, criterion):
        super(BiSeNetLoss, self).__init__()
        self.criterion = criterion

    def forward(self, outputs, masks):
        if isinstance(outputs, (list, tuple)) and len(outputs) != 3:
            raise ValueError(f"Expected outputs to be a list or tuple of 3 elements (output, output16, output32), but got {len(outputs)} elements.")
        
        # Evaluation
        if not isinstance(outputs, (list, tuple)):
            return self.criterion(outputs, masks)

        output, output16, output32 = outputs
        
        loss, loss16, loss32 = map(
            lambda x: self.criterion(x, masks), 
            [output, output16, output32]
        )
        
        tot_loss = loss + loss16 + loss32
        
        return tot_loss, {
            "semantic": loss,
            "semantic-aux16": loss16,
            "semantic-aux32": loss32
        }