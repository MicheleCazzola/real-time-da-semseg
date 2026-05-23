import torch
import torch.nn as nn
import torch.nn.functional as F

from .bisenet import BiSeNetLoss

class DetailAggregateLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DetailAggregateLoss, self).__init__()
        
        self.laplacian_kernel = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32
        ).reshape(1, 1, 3, 3).requires_grad_(False).type(torch.FloatTensor)
        
        self.fuse_kernel = torch.nn.Parameter(
            torch.tensor([[6./10], [3./10], [1./10]],
            dtype=torch.float32
        ).reshape(1, 3, 1, 1).type(torch.FloatTensor))
        
    def dice_loss(self, input, target):
        smooth = 1.0
        n = input.size(0)
        iflat = input.view(n, -1)
        tflat = target.view(n, -1)
        
        intersection = (iflat * tflat).sum(1)
        loss = 1 - ((
            2. * intersection + smooth) /
            (iflat.sum(1) + tflat.sum(1) + smooth
        ))
        
        return loss.mean()

    def forward(self, boundary_logits, gtmasks, ignore_index=-1):

        # Neutralize the ignore_index so Laplacian doesn't detect falsified boundaries
        gtmasks = gtmasks.clone()
        valid_mask = (gtmasks != ignore_index).unsqueeze(1).type(torch.FloatTensor).to(gtmasks.device)
        gtmasks[gtmasks == ignore_index] = 0

        # boundary_logits = boundary_logits.unsqueeze(1)
        boundary_targets = F.conv2d(gtmasks.unsqueeze(1).type(torch.FloatTensor), self.laplacian_kernel, padding=1)
        boundary_targets = boundary_targets.clamp(min=0)
        boundary_targets[boundary_targets > 0.1] = 1
        boundary_targets[boundary_targets <= 0.1] = 0

        boundary_targets_x2 = F.conv2d(gtmasks.unsqueeze(1).type(torch.FloatTensor), self.laplacian_kernel, stride=2, padding=1)
        boundary_targets_x2 = boundary_targets_x2.clamp(min=0)
        
        boundary_targets_x4 = F.conv2d(gtmasks.unsqueeze(1).type(torch.FloatTensor), self.laplacian_kernel, stride=4, padding=1)
        boundary_targets_x4 = boundary_targets_x4.clamp(min=0)

        boundary_targets_x8 = F.conv2d(gtmasks.unsqueeze(1).type(torch.FloatTensor), self.laplacian_kernel, stride=8, padding=1)
        boundary_targets_x8 = boundary_targets_x8.clamp(min=0)
    
        boundary_targets_x8_up = F.interpolate(boundary_targets_x8, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x4_up = F.interpolate(boundary_targets_x4, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x2_up = F.interpolate(boundary_targets_x2, boundary_targets.shape[2:], mode='nearest')
        
        boundary_targets_x2_up[boundary_targets_x2_up > 0.1] = 1
        boundary_targets_x2_up[boundary_targets_x2_up <= 0.1] = 0
        
        boundary_targets_x4_up[boundary_targets_x4_up > 0.1] = 1
        boundary_targets_x4_up[boundary_targets_x4_up <= 0.1] = 0
        
        boundary_targets_x8_up[boundary_targets_x8_up > 0.1] = 1
        boundary_targets_x8_up[boundary_targets_x8_up <= 0.1] = 0
        
        boundary_targets_pyramids = torch.stack((boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up), dim=1)
        
        boundary_targets_pyramids = boundary_targets_pyramids.squeeze(2)
        boundary_targets_pyramid = F.conv2d(boundary_targets_pyramids, self.fuse_kernel)

        boundary_targets_pyramid[boundary_targets_pyramid > 0.1] = 1
        boundary_targets_pyramid[boundary_targets_pyramid <= 0.1] = 0
        
        if boundary_logits.shape[-1] != boundary_targets.shape[-1]:
            boundary_logits = F.interpolate(
                boundary_logits, boundary_targets.shape[2:], mode='bilinear', align_corners=True
            )
        
        boundary_targets_pyramid = boundary_targets_pyramid.to(boundary_logits.device)
        
        # Apply validity mask to ignore padded areas (-1)
        valid_mask = valid_mask.to(boundary_logits.device)
        
        bce_loss = F.binary_cross_entropy_with_logits(boundary_logits, boundary_targets_pyramid, reduction='none')
        bce_loss = (bce_loss * valid_mask).sum() / (valid_mask.sum() + 1e-6)
        
        dice_loss = self.dice_loss(torch.sigmoid(boundary_logits) * valid_mask, boundary_targets_pyramid * valid_mask)
        
        return bce_loss, dice_loss

class STDCLoss(BiSeNetLoss):
    def __init__(self, sem_loss, bd_loss, ignore_index=-1):
        super(STDCLoss, self).__init__(sem_loss)
        self.bd_loss = bd_loss
        self.ignore_index = ignore_index

    def forward(self, outputs, masks):
        if isinstance(outputs, (list, tuple)) and len(outputs) != 4:
            raise ValueError(f"Expected outputs to be a list or tuple of 4 elements (output, output16, output32, output64), but got {len(outputs)} elements.")
        
        # Evaluation
        if not isinstance(outputs, (list, tuple)):
            return super().forward(outputs, masks)
        
        output, output16, output32, detail8 = outputs
        
        sem_loss, sem_losses = super().forward([output, output16, output32], masks)
        boundary_bce_loss, boundary_dice_loss = self.bd_loss(detail8, masks, ignore_index=self.ignore_index)
        
        tot_loss = sem_loss + boundary_bce_loss + boundary_dice_loss
        
        return tot_loss, {
            **sem_losses,
            "boundary_bce": boundary_bce_loss,
            "boundary_dice": boundary_dice_loss
        }