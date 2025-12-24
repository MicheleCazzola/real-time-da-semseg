import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from src.metrics.metrics import calculate_iou
from src.models.stdc import STDC
from src.utils.variables import num_classes, device, IGNORE_INDEX
from src.train.bisenet import train_bisenet

def stdc_model_setup(backbone_name, pretrained_weights, learning_rate, weight_decay, step_size, gamma):
    model = STDC(n_classes=num_classes, backbone=backbone_name, pretrain_model=pretrained_weights).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    
    return model, criterion, optimizer, scheduler

@torch.no_grad()
def evaluate_stdc(model, dataloader, criterion, device) -> tuple:

    model.eval()

    running_loss = 0.0
    data_len = 0
    iou_scores = 0.0
    ious_per_class = torch.zeros(num_classes)

    for i, (inputs, masks) in enumerate(dataloader):

        data_len += inputs.size(0)

        inputs = inputs.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs, _, _ = model(inputs)
        loss = criterion(outputs, masks)

        running_loss += loss.item()*inputs.size(0)

        # Calculate mIoU
        iou, iou_per_class = calculate_iou(outputs, masks, num_classes)
        iou_scores += iou.item() * inputs.size(0)
        ious_per_class += iou_per_class.cpu() * inputs.size(0)

    mIoU = iou_scores / data_len
    loss = running_loss / data_len
    mious_per_class = ious_per_class / data_len

    return loss, mIoU, mious_per_class


def train_stdc(model, trainloader, validloader, optimizer, scheduler, criterion, num_epochs, device, log_frequency):
    # Now they coincide
    # Possible improvement: add boundary losses in STDC for compliance with the original paper (need for a specific
    # function in that case)
    return train_bisenet(model, trainloader, validloader, optimizer, scheduler, criterion, num_epochs, device, log_frequency)