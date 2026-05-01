import logging
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.utils.clip_grad as clip_grad

from src.models.bisenet import BiSeNet
from src.models.discriminator import FCDiscriminator
from src.models.pidnet import PIDNet
from src.models.stdc import STDC
from src.train.utils import train_forward_source, train_forward_target, evaluate_model
from src.utils.utils import get_mious_per_category

def adda_setup(domain_lr, domain_wd):
    domain_criterion = nn.BCEWithLogitsLoss()

    model_domain = FCDiscriminator(num_classes)
    model_domain = model_domain.to(device)
    domain_optimizer = torch.optim.Adam(model_domain.parameters(), lr = domain_lr, weight_decay = domain_wd)
    
    return model_domain, domain_criterion, domain_optimizer


def train_adda(
    model, model_domain, lambda_domain, trainloader, trainloader_target, validloader, 
    criterion_domain, optimizer, optimizer_domain, scheduler, num_epochs, device, log_frequency, criterion=None
):
    assert model in [PIDNet, BiSeNet, STDC], "ADDA is only implemented for PIDNet, BiSeNet and STDC models."
    
    if (isinstance(model, BiSeNet) or isinstance(model, STDC)) and criterion is None:
        raise ValueError("Criterion must be provided for BiSeNet and STDC models when using ADDA.")
    
    train_losses, val_losses = [], []
    miou_scores, miou_scores_per_category = [], []

    for epoch in range(num_epochs):

        current_step = 0
        running_source_loss_seg = 0.0

        loss_G, loss_D = 0, 0

        model.train()
        model_domain.train()

        for (inputs, masks, boundaries), (target_inputs, _, _) in zip(trainloader, trainloader_target):

            inputs = inputs.to(device)
            masks = masks.to(device)
            boundaries = boundaries.to(device)
            target_inputs = target_inputs.to(device)

            # Train G
            for param in model_domain.parameters():
                param.requires_grad = False

            optimizer.zero_grad()
            optimizer_domain.zero_grad()

            # Train with source
            source_loss, source_output = train_forward_source(model, inputs, masks, boundaries, criterion)
                
            source_loss.backward()
            running_source_loss_seg += source_loss.item()

            # Train with target
            target_output = train_forward_target(model, target_inputs)
            preds = F.softmax(target_output, dim=1)
            D_out = model_domain(preds)

            domain_loss = lambda_domain * criterion_domain(D_out, torch.zeros_like(D_out))
            domain_loss.backward()
            loss_G += domain_loss.item()

            # Train D
            for param in model_domain.parameters():
                param.requires_grad = True

            # Train with source
            source_output = source_output.detach()
            preds = F.softmax(source_output, dim=1)
            D_out = model_domain(preds)

            domain_loss = criterion_domain(D_out, torch.zeros_like(D_out))
            domain_loss = domain_loss / 2
            domain_loss.backward()
            loss_D += domain_loss.item()

            # Train with target
            target_output = target_output.detach()
            preds = F.softmax(target_output, dim=1)
            D_out = model_domain(preds)

            domain_loss = criterion_domain(D_out, torch.ones_like(D_out))
            domain_loss = domain_loss / 2
            domain_loss.backward()
            loss_D += domain_loss.item()

            clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=35, norm_type=2)
            clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model_domain.parameters()), max_norm=35, norm_type=2)
            
            optimizer.step()
            optimizer_domain.step()

            if current_step % log_frequency == 0:
                logging.info(f"Epoch {epoch+1}, Iteration {current_step}, Source loss: {running_source_loss_seg/(current_step+1):5f}, Domain loss: {loss_G/(current_step+1):.5f} ({loss_D/(current_step+1):.5f}")
            current_step += 1

        train_loss = running_source_loss_seg/len(trainloader)
        train_domain_loss_G = loss_G/len(trainloader)
        train_domain_loss_D = loss_D/len(trainloader)

        logging.info(f"End of Epoch {epoch+1}")
        logging.info(f"Training loss: {train_loss:.5f}")
        logging.info(f"Domain loss G: {train_domain_loss_G:.5f}")
        logging.info(f"Domain loss D: {train_domain_loss_D:.5f}")

        val_loss, val_miou, val_mious_per_class = evaluate_model(model, validloader, device, criterion)
        mious_per_category = get_mious_per_category(val_mious_per_class)
        
        logging.info(f"Validation mIoU: {val_miou:.3f}%, Validation loss: {val_loss:.5f}")

        val_losses.append(val_loss)
        train_losses.append(train_loss)
        miou_scores.append(val_miou)
        miou_scores_per_category.append(mious_per_category)
        
        logging.info(f"Epoch {epoch+1} completed. Training loss: {train_loss:.5f}, Validation loss: {val_loss:.5f}, Validation mIoU: {val_miou:.3f}%")
        
        if scheduler is not None:
            scheduler.step()
    
    return train_losses, val_losses, miou_scores, miou_scores_per_category