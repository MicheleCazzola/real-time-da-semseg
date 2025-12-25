import torch

from src.metrics.metrics import calculate_iou
from src.losses.bondary import BondaryLoss
from src.losses.cross_entropy import CrossEntropy
from src.utils.variables import num_classes, device, PIDNET_S_WEIGHTS, IGNORE_INDEX, categories
from src.models.pidnet import FullPIDNetModel, PIDNet
from src.utils.utils import get_mious_per_category


def get_pidnet(model_name, num_classes, pretrained_weights, imgnet_pretrained) -> PIDNet:

    if 's' in model_name:
        model = PIDNet(m=2, n=3, num_classes=num_classes, planes=32, ppm_planes=96, head_planes=128, augment=True)
    elif 'm' in model_name:
        model = PIDNet(m=2, n=3, num_classes=num_classes, planes=64, ppm_planes=96, head_planes=128, augment=True)
    else:
        model = PIDNet(m=3, n=4, num_classes=num_classes, planes=64, ppm_planes=112, head_planes=256, augment=True)

    if imgnet_pretrained:
        pretrained_state = torch.load(pretrained_weights, map_location='cpu')
        model_dict = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if (k in model_dict and v.shape == model_dict[k].shape)}
        model_dict.update(pretrained_state)
        msg = 'Loaded {} parameters!'.format(len(pretrained_state))
        print('Attention!!!')
        print(msg)
        print('Over!!!')
        model.load_state_dict(model_dict, strict = False)
    else:
        pretrained_dict = torch.load(pretrained_weights, map_location='cpu')
        model_dict = model.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
        msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
        print('Attention!!!')
        print(msg)
        print('Over!!!')
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict = False)

    return model


def get_pred_model(name, num_classes):

    if 's' in name:
        model = PIDNet(m=2, n=3, num_classes=num_classes, planes=32, ppm_planes=96, head_planes=128, augment=False)
    elif 'm' in name:
        model = PIDNet(m=2, n=3, num_classes=num_classes, planes=64, ppm_planes=96, head_planes=128, augment=False)
    else:
        model = PIDNet(m=3, n=4, num_classes=num_classes, planes=64, ppm_planes=112, head_planes=256, augment=False)

    return model


def pidnet_model_setup(pidnet_model, pretrained, learning_rate, weight_decay, step_size, gamma, optimizer_fun, momentum=None, ce_weights=None):
    
    if optimizer_fun is torch.optim.SGD:
        assert momentum is not None, "Momentum value must be provided for SGD optimizer."
    
    pidnet = get_pidnet(pidnet_model, num_classes, PIDNET_S_WEIGHTS, imgnet_pretrained=pretrained)
    model = FullPIDNetModel(pidnet, sem_loss=CrossEntropy(ignore_label=IGNORE_INDEX, weight=ce_weights), bd_loss=BondaryLoss())
    model = model.to(device)
    
    if optimizer_fun is torch.optim.SGD:
        optimizer = optimizer_fun(model.parameters(), lr = learning_rate, momentum = momentum, weight_decay = weight_decay)
    else:
        optimizer = optimizer_fun(model.parameters(), lr = learning_rate, weight_decay = weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    return model, optimizer, scheduler


@torch.no_grad()
def evaluate_pidnet(model, dataloader, device, log_frequency) -> tuple:

    model.eval()
    
    print("=== Evaluation ===")

    step = 0
    running_loss = 0.0
    data_len = 0
    iou_scores = 0.0
    ious_per_class = torch.zeros(num_classes)

    for (inputs, masks, boundaries) in dataloader:

        data_len += inputs.size(0)

        inputs = inputs.to(device)
        masks = masks.to(device)
        boundaries = boundaries.to(device)

        # Forward pass
        loss, outputs, _, [loss_s, loss_b, loss_sb] = model(inputs, masks, boundaries)
        running_loss += loss.item() * inputs.size(0)

        # Calculate mIoU
        iou, iou_per_class = calculate_iou(outputs[1], masks, num_classes)
        iou_scores += iou.item() * inputs.size(0)
        ious_per_class += iou_per_class.cpu() * inputs.size(0)
        
        if (step + 1) % log_frequency == 0:
            print(f"Iteration {step+1}, Loss: {running_loss / data_len:.3f}, mIoU: {100 * iou_scores / data_len:.2f}%"
                  f"\tLoss_s: {loss_s.item():.3f}, Loss_b: {loss_b.item():.3f}, Loss_sb: {loss_sb.item():.3f}")
        
        step += 1

    mIoU = 100 * iou_scores / data_len
    loss = running_loss / data_len
    mious_per_class = 100 * ious_per_class / data_len

    return loss, mIoU, mious_per_class


def train_pidnet(model, trainloader, validloader, optimizer, scheduler, num_epochs, device, log_frequency):
    
    val_losses = []
    train_losses = []
    miou_scores, miou_scores_per_category = [], []

    for epoch in range(num_epochs):

        data_len = 0
        current_step = 0
        train_loss = 0.0
        model.train()

        for (inputs, masks, boundaries) in trainloader:

            inputs = inputs.to(device)
            masks = masks.to(device)
            boundaries = boundaries.to(device)
            data_len += inputs.size(0)

            # Forward pass
            optimizer.zero_grad()
            loss, _, _, [loss_s, loss_b, loss_sb] = model(inputs, masks, boundaries)
            train_loss += loss.item() * inputs.size(0)

            # Backward pass
            loss.backward()
            optimizer.step()

            if current_step % log_frequency == 0:
                print(f"Epoch {epoch+1}, Iteration {current_step}, Current Loss: {train_loss/data_len:.3f} "
                      f"\tLoss_s: {loss_s.item():.3f}, Loss_b: {loss_b.item():.3f}, Loss_sb: {loss_sb.item():.3f}")

            current_step += 1

        train_loss /= data_len

        print(f"End of Epoch {epoch+1}")
        print(f"Training loss: {train_loss:.5f}")

        val_loss, val_miou, mious_per_class = evaluate_pidnet(model, validloader, device, log_frequency)

        print(f"Validation mIoU: {val_miou:.2f}%, Validation loss: {val_loss:.5f}")

        mious_per_category = get_mious_per_category(mious_per_class)

        val_losses.append(val_loss)
        train_losses.append(train_loss)
        miou_scores.append(val_miou)
        miou_scores_per_category.append(mious_per_category)

        print()
        
        # Scheduler is None if learning rate is constant
        if scheduler is not None:
            scheduler.step()
    
    return train_losses, val_losses, miou_scores, miou_scores_per_category