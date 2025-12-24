import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from src.metrics.metrics import calculate_iou
from src.models.deeplab_v2 import get_deeplab_v2
from src.utils.variables import num_classes, device, DEEPLAB_V2_WEIGHTS, IGNORE_INDEX
from src.utils.utils import save_checkpoint, resume_checkpoint



def deeplab_v2_model_setup(learning_rate, weight_decay, step_size, gamma):
    model = get_deeplab_v2(num_classes=num_classes, pretrain=True, pretrain_model_path=DEEPLAB_V2_WEIGHTS).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    
    return model, criterion, optimizer, scheduler


def train_deeplab_v2(model, trainloader, num_epochs, criterion, optimizer, scheduler, chp_file_name_prefix, resume_training, resume_path):
    train_losses = []

    if resume_training:
        start_epoch, model, optimizer, scheduler = resume_checkpoint(resume_path, model, optimizer, scheduler)
    else:
        start_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        print("### Training mode")
        model.train()
        running_loss = 0.0
        for i, (images, masks) in enumerate(trainloader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs, _, _ = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 25 == 0:
                print(f"Processed {i + 1} batches, loss: {running_loss / (i+1)}")

        epoch_loss = running_loss / len(trainloader)
        train_losses.append(epoch_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")
        
        path = f"{chp_file_name_prefix}{epoch}.pth.tar"
        save_checkpoint(path, epoch, model, optimizer, scheduler)

        if scheduler is not None:
            scheduler.step()
            
    return train_losses


@torch.no_grad()
def evaluate_deeplab_v2(model, dataloader, criterion, start_epoch, end_epoch, chp_file_name_prefix, log_frequency=25):
    
    losses = []
    mious = []
    mious_per_class = []

    for epoch in range(start_epoch, end_epoch):

        model = get_deeplab_v2(num_classes=num_classes, pretrain=False)  # Assuming get_deeplab_v2 is defined

        path = f"{chp_file_name_prefix}{epoch}.pth.tar"
        _, model, _, _ = resume_checkpoint(path, model)

        model.to(device)
        
        print("### Evaluation mode")
        
        loss = 0
        data_len = 0
        iou_scores = 0
        ious_per_class = torch.zeros(num_classes)
        
        model.eval()
        for i, (inputs, masks) in enumerate(dataloader):
            inputs = inputs.to(device)
            masks = masks.to(device)
            data_len += inputs.size(0)

            # loss
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            loss += loss.item() * inputs.size(0)

            # mIoU
            iou, iou_per_class = calculate_iou(outputs, masks, num_classes)
            iou_scores += iou.item() * inputs.size(0)
            ious_per_class += iou_per_class.cpu() * inputs.size(0)
            
            if (i + 1) % log_frequency == 0:
                print(f"Processed {i + 1} batches: loss {loss / (i+1)}, mIoU: {miou / (i+1)}")
        
        miou = 100 * iou_scores / data_len
        miou_per_class = 100 * ious_per_class / data_len
        loss = loss / data_len
        
        losses.append(loss)
        mious.append(miou)
        mious_per_class.append(miou_per_class)

        print(f"Epoch: [{epoch+1}/{(end_epoch - start_epoch)}], Validation Loss: {loss:.4f}, mIoU: {(miou * 100):.2f}%")
        
    return losses, mious, mious_per_class