import torch
from torchmetrics.segmentation import mean_iou

class MeanIoU(mean_iou.MeanIoU):
    def __init__(self, num_classes):
        super().__init__(num_classes=num_classes + 1, include_background=False, per_class=True, input_format="index")
        
    def update(self, preds, target):
        preds = preds.argmax(dim=1) if preds.dim() == 4 else preds
        # Shift preds and target by 1 to handle ignore_index=-1
        preds = preds + 1
        target = target + 1
        super().update(preds, target)
    
    def compute(self):
        per_class_ious = super().compute()
        # Filter out NaN or negative values (absent classes) to compute a valid mean
        valid_mask = (per_class_ious >= 0) & (~per_class_ious.isnan())
        miou = per_class_ious[valid_mask].mean() if valid_mask.any() else torch.tensor(0.0, device=per_class_ious.device)
        
        return miou, per_class_ious