"""
Domain Adaptation via Cross-domain Mixed Sampling (DACS)
From the original implementation: github.com/vikolss/DACS
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F

from src.losses.pidnet import pidnet_loss
from src.models.bisenet import BiSeNet
from src.models.stdc import STDC
from src.dataset.dataset import generate_bd
from src.models.pidnet import FullPIDNetModel, PIDNet
from src.losses.bondary import BondaryLoss
from src.losses.cross_entropy import CrossEntropy
from src.train.pidnet import evaluate_pidnet, get_pidnet
from src.utils.variables import num_classes, PIDNET_S_WEIGHTS, IGNORE_INDEX, device, categories
from src.utils.utils import get_mious_per_category
from src.train.utils import evaluate_model, train_forward_source, train_forward_target


def oneMix(mask, data = None, target = None):
    #Mix
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0*data[0]+(1-stackedMask0)*data[1]).unsqueeze(0)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0*target[0]+(1-stackedMask0)*target[1]).unsqueeze(0)
    return data, target


def generate_class_mask(pred, classes):
    pred, classes = torch.broadcast_tensors(pred.unsqueeze(0), classes.unsqueeze(1).unsqueeze(2))
    N = pred.eq(classes).sum(0)
    return N


def mix(parameters, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = oneMix(mask = parameters["Mix"], data = data, target = target)
    return data, target

def update_ema_variables(ema_model, model, alpha_teacher, iteration):
    # Use the "true" average until the exponential average is more correct
    alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)

    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

def create_ema_model(model):
    
    # Old version that works only for PIDNet
    # pidnet = get_pidnet("pidnet_s", num_classes, PIDNET_S_WEIGHTS, imgnet_pretrained=True)
    # ema_model = FullModel(pidnet, sem_loss=CrossEntropy(ignore_label=IGNORE_INDEX), bd_loss=BondaryLoss())
    
    # for param in ema_model.parameters():
    #     param.detach_()
        
    # mp = list(model.parameters())
    # mcp = list(ema_model.parameters())
    # n = len(mp)
    # for i in range(0, n):
    #     mcp[i].data[:] = mp[i].data[:].clone()
    
    ema_model = copy.deepcopy(model)
    ema_model = ema_model.to(device)
    
    for p in ema_model.parameters():
        p.requires_grad_(False)
        
    return ema_model

def calc_U_loss(outputs, pseudo_labels, base_model, **model_kwargs):
    if isinstance(base_model, FullPIDNetModel):
        loss_s, loss_b, loss_sb = pidnet_loss(outputs, pseudo_labels, sem_loss=base_model.sem_loss, bd_loss=base_model.bd_loss)
        return loss_s + loss_b + loss_sb
    elif isinstance(base_model, (BiSeNet, STDC)):
        criterion = model_kwargs.get('criterion', None)
        if criterion is None:
            raise ValueError("Criterion must be provided for BiSeNet and STDC models.")
        loss = criterion(outputs, pseudo_labels)
        return loss
    else:
        raise NotImplementedError("calc_U_loss is not implemented for this model type.")


def dacs_setup(model):
    ema_model = create_ema_model(model)
    ema_model = ema_model.to(device)
    sem_loss = CrossEntropy(ignore_label=IGNORE_INDEX)
    bd_loss = BondaryLoss()
    
    return ema_model, sem_loss, bd_loss


def augment_and_mix(inputs, masks, target_inputs, targets_u_w):
    
    inputs_u_s = []
    targets_u = []
            
    for i in range(len(inputs)):
        classes = torch.unique(masks[i])
        nclasses = classes.shape[0]
        classes = (classes[torch.Tensor(np.random.choice(nclasses, int((nclasses+nclasses%2)/2),replace=False)).long()]).to(device)
        MixMask_i = generate_class_mask(masks[i], classes).unsqueeze(0).to(device)

        strong_parameters = {"Mix": MixMask_i}

        inputs_u_si, _ = mix(strong_parameters, data = torch.cat((inputs[i].unsqueeze(0),target_inputs[i].unsqueeze(0))))
        inputs_u_s.append(inputs_u_si)

        _, targets_ui = mix(strong_parameters, target = torch.cat((masks[i].unsqueeze(0),targets_u_w[i].unsqueeze(0))))
        targets_u.append(targets_ui)

    inputs_u_s = torch.cat(inputs_u_s)
    targets_u = torch.cat(targets_u).long().to(device)
    
    return inputs_u_s, targets_u, strong_parameters


def mix_pixel_weights(inputs, strong_parameters, pixelWiseWeight):
    
    pixel_weights = []
    onesWeights = torch.ones((pixelWiseWeight.shape)).to(device)
    for _ in range(len(inputs)):
        _, pixelWiseWeight_i = mix(
                    strong_parameters,
                    target = torch.cat((onesWeights[0].unsqueeze(0),pixelWiseWeight[0].unsqueeze(0)))
                )
        pixel_weights.append(pixelWiseWeight_i)

    pixel_weights = torch.cat(pixel_weights).to(device)
    
    return pixel_weights


def train_dacs(
    model, ema_model, trainloader_source, trainloader_target, validloader, optimizer, scheduler,
    num_epochs, device, log_frequency, pixel_weight=False, criterion=None, **model_kwargs
):
    assert model in [PIDNet, BiSeNet, STDC], "ADDA is only implemented for PIDNet, BiSeNet and STDC models."
    
    if (isinstance(model, BiSeNet) or isinstance(model, STDC)) and criterion is None:
        raise ValueError("Criterion must be provided for BiSeNet and STDC models when using ADDA.")
    
    ema_model.train()

    accumulated_loss_l, accumulated_loss_u = [], []
    train_losses, val_losses = [], []
    miou_scores, miou_scores_per_category = [], []
    
    for epoch in range(num_epochs):
        model.train()

        loss_u_value = 0
        loss_l_value = 0

        n = 0
        for (inputs, masks, boundaries), (target_inputs, _, _) in zip(trainloader_source, trainloader_target):
            optimizer.zero_grad()

            inputs = inputs.to(device)
            masks = masks.to(device)
            boundaries = boundaries.to(device)
            target_inputs = target_inputs.to(device)
            
            source_loss, _ = train_forward_source(model, inputs, masks, boundaries, criterion)

            # The original implementation uses the EMA model for pseudo-label generation
            # We found that using the original model gives better results
            
            # logits_u_w = train_forward_target(model, target_inputs)
            target_logits = train_forward_target(model, target_inputs)

            pseudo_label = torch.softmax(target_logits.detach(), dim=1)
            max_probs, targets_u_w = torch.max(pseudo_label, dim=1)

            inputs_u_s, targets_u, strong_parameters = augment_and_mix(inputs, masks, target_inputs, targets_u_w)
            
            outputs, logits_u_s = train_forward_target(model, inputs_u_s, return_all=True)

            if pixel_weight == "threshold_uniform":
                unlabeled_weight = torch.sum(max_probs.ge(0.968).long() == 1).item() / np.size(np.array(targets_u.cpu()))
                pixelWiseWeight = unlabeled_weight * torch.ones(max_probs.shape).to(device)
            elif pixel_weight == "threshold":
                pixelWiseWeight = max_probs.ge(0.968).float().to(device)
            elif pixel_weight == False:
                pixelWiseWeight = torch.ones(max_probs.shape).to(device)

            pixel_weights = mix_pixel_weights(inputs, strong_parameters, pixelWiseWeight)

            L_u = calc_U_loss(outputs, targets_u, model, **model_kwargs)
            L_u *= torch.mean(pixel_weights)

            loss = source_loss + L_u

            loss_l_value += source_loss.item()
            loss_u_value += L_u.item()

            loss.backward()
            optimizer.step()

            if n % log_frequency == 0:
                print('\tProcessed {0:d} batches, loss_l = {1:.3f}, loss_u = {2:.3f} loss = {3:.3f}'.format(n, loss_l_value/(n+1), loss_u_value/(n+1),(loss_l_value+loss_u_value)/(n+1)))

            n += 1

        loss_l_value /= len(trainloader_source)
        loss_u_value /= len(trainloader_target)

        accumulated_loss_l.append(loss_l_value)
        accumulated_loss_u.append(loss_u_value)
        train_losses.append(loss_l_value+loss_u_value)

        # Update learning rate
        if scheduler is not None:
            scheduler.step()
            
        # Update mean teacher network
        alpha_teacher = 0.99
        ema_model = update_ema_variables(ema_model = ema_model, model = model, alpha_teacher=alpha_teacher, iteration=epoch)

        print('iter = {0:6d}/{1:6d}, loss_l = {2:.3f}, loss_u = {3:.3f} loss = {4:.3f}'.format(epoch+1, num_epochs, loss_l_value, loss_u_value, loss_l_value + loss_u_value))

        val_loss, val_mean_iou, mious_per_class = evaluate_model(model, validloader, device, criterion)
        
        mious_per_category = get_mious_per_category(mious_per_class)
        
        print(f"Validation mIoU: {val_mean_iou*100:.3f}%, Validation loss: {val_loss:.5f}")
        
        for i, cat in enumerate(categories.keys()):
            print(f"{cat} mIoU: {mious_per_class[i]:.2f}")
        print()

        val_losses.append(val_loss)
        miou_scores.append(val_mean_iou)
        miou_scores_per_category.append(mious_per_category)
    
    return train_losses, val_losses, miou_scores, miou_scores_per_category