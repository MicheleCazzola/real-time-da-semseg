import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from metrics.metrics import calculate_iou
from src.models.bisenet import BiSeNet
from src.utils.variables import num_classes, device, IGNORE_INDEX
from utils.utils import get_mious_per_category

def bisenet_model_setup(backbone_name, learning_rate, weight_decay, step_size, gamma):
    
    model = BiSeNet(num_classes, backbone_name).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    
    return model, criterion, optimizer, scheduler


@torch.no_grad()
def evaluate_bisenet(model, dataloader, criterion, device) -> tuple:

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
        outputs = model(inputs)
        loss = criterion(outputs, masks)

        running_loss += loss.item()*inputs.size(0)

        # Calculate mIoU
        iou, iou_per_class = calculate_iou(outputs, masks, num_classes)
        iou_scores += iou * inputs.size(0)
        ious_per_class += iou_per_class.cpu() * inputs.size(0)

    mIoU = iou_scores / data_len
    loss = running_loss / data_len
    mious_per_class = ious_per_class / data_len

    return loss, mIoU, mious_per_class


def train_bisenet(model, trainloader, validloader, optimizer, scheduler, criterion, num_epochs, device, log_frequency):
    
    train_losses = []
    val_losses = []
    miou_scores = []
    mious_scores_per_category = []

    for epoch in range(num_epochs):
        
        print("### Training mode")
        
        model.train()
        running_loss = 0.0
        for i, (images, masks) in enumerate(trainloader):
            
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            
            outputs, outputs16, outputs32 = model(images)
            
            loss1 = criterion(outputs, masks)
            loss2 = criterion(outputs16, masks)
            loss3 = criterion(outputs32, masks)
            loss = loss1 + loss2 + loss3
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % log_frequency == 0:
                print(f"Processed {i + 1} batches, loss: {running_loss / (i+1)}")

        epoch_loss = running_loss / len(trainloader)
        train_losses.append(epoch_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")

        print("### Evaluation mode")
        val_loss, miou, mious_per_class = evaluate_bisenet(model, validloader, criterion, device)

        print(f"Validation mIoU: {miou * 100:.3f}%, Validation loss: {val_loss:.5f}")
        
        mious_per_category = get_mious_per_category(mious_per_class)

        val_losses.append(val_loss)
        miou_scores.append(miou)
        mious_scores_per_category.append(mious_per_category)

        print(f"Epoch: [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, mIoU: {(miou * 100):.2f}%")

        if scheduler is not None:
            scheduler.step()
            
    return train_losses, val_losses, miou_scores, mious_scores_per_category
        