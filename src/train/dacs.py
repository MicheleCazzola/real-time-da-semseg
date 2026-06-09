"""
    Domain Adaptation via Cross-domain Mixed Sampling (DACS)
    Adapted from: github.com/vikolss/DACS
"""
import copy
import logging
import os

import torch

from src.metrics.mean_iou import MeanIoU
from src.models.bisenet import BiSeNet
from src.models.pidnet import PIDNet
from src.models.stdc import STDC
from src.train.train_model import evaluate_model
from src.utils.utils import save_checkpoint


def oneMix(mask, data=None, target=None):
    # Mix
    if data is not None:
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0 * data[0] + (1 - stackedMask0) * data[1]).unsqueeze(0)
    if target is not None:
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0 * target[0] + (1 - stackedMask0) * target[1]).unsqueeze(0)
    return data, target


def generate_class_mask(pred, classes):
    pred, classes = torch.broadcast_tensors(pred.unsqueeze(0), classes.unsqueeze(1).unsqueeze(2))
    N = pred.eq(classes).sum(0)
    return N


def mix(parameters, data=None, target=None):
    assert data is not None or target is not None
    data, target = oneMix(mask=parameters["Mix"], data=data, target=target)
    return data, target



def update_ema_variables(ema_model, model, alpha_teacher, iteration):
    # Use the "true" average until the exponential average is more correct
    alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = alpha_teacher * ema_param.data[:] + (1 - alpha_teacher) * param.data[:]
    return ema_model


def create_ema_model(model, device):
    ema_model = copy.deepcopy(model)
    ema_model = ema_model.to(device)
    for p in ema_model.parameters():
        p.requires_grad_(False)
    return ema_model


def _extract_pred(outputs, model):
    if isinstance(model, PIDNet):
        return outputs[1] if isinstance(outputs, (list, tuple)) else outputs
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    return outputs


def _loss_value(loss_output):
    return loss_output[0] if isinstance(loss_output, tuple) else loss_output


def calc_u_loss(outputs, pseudo_labels, model, criterion):
    if isinstance(model, PIDNet):
        return _loss_value(criterion(outputs, (pseudo_labels, None)))
    return _loss_value(criterion(outputs, pseudo_labels))


def dacs_setup(cfg, model, device):
    _ = cfg
    ema_model = create_ema_model(model, device)
    return ema_model


def augment_and_mix(inputs, masks, target_inputs, targets_u_w, device, ignore_index):
    inputs_u_s = []
    targets_u = []
    mix_masks = []

    for i in range(len(inputs)):
        classes = torch.unique(masks[i])
        if ignore_index is not None:
            classes = classes[classes != ignore_index]
        nclasses = classes.shape[0]
        if nclasses == 0:
            selected_classes = classes
        else:
            k = int((nclasses + nclasses % 2) / 2)
            selected_classes = classes[torch.randperm(nclasses, device=classes.device)[:k]]
        mix_mask = generate_class_mask(masks[i], selected_classes).unsqueeze(0).to(device)
        mix_masks.append(mix_mask)

        strong_parameters = {"Mix": mix_mask}

        inputs_u_si, _ = mix(
            strong_parameters,
            data=torch.cat((inputs[i].unsqueeze(0), target_inputs[i].unsqueeze(0)))
        )
        inputs_u_s.append(inputs_u_si)

        _, targets_ui = mix(
            strong_parameters,
            target=torch.cat((masks[i].unsqueeze(0), targets_u_w[i].unsqueeze(0)))
        )
        targets_u.append(targets_ui)

    inputs_u_s = torch.cat(inputs_u_s)
    targets_u = torch.cat(targets_u).long().to(device)
    mix_masks = torch.cat(mix_masks)

    return inputs_u_s, targets_u, mix_masks


def mix_pixel_weights(inputs, mix_masks, pixel_wise_weight, device):
    pixel_weights = []
    for i in range(len(inputs)):
        ones_weights = torch.ones_like(pixel_wise_weight[i]).to(device)
        _, pixel_wise_weight_i = mix(
            {"Mix": mix_masks[i].unsqueeze(0)},
            target=torch.cat((ones_weights.unsqueeze(0), pixel_wise_weight[i].unsqueeze(0)))
        )
        pixel_weights.append(pixel_wise_weight_i)

    return torch.cat(pixel_weights).to(device)


def train_dacs(
    model, ema_model, model_name, num_classes, trainloader_source, trainloader_target, validloader,
    criterion, optimizer, scheduler, start_epoch, end_epoch, start_miou, bd_required, checkpoint_dir,
    device, log_frequency, pixel_weight="threshold", pseudo_threshold=0.968, use_ema_for_pseudo=True,
    alpha_teacher=0.99, ignore_index=-1
):
    assert isinstance(model, (PIDNet, BiSeNet, STDC)), "DACS is only implemented for PIDNet, BiSeNet and STDC models."
    if criterion is None:
        raise ValueError("Criterion must be provided for DACS training.")

    logging.info(f"{model_name} - DACS training")
    logging.info(f"Training epochs: {end_epoch} (from {start_epoch + 1} to {end_epoch})")

    train_losses, val_losses = [], []
    train_task_specific_losses = {}
    train_mious, val_mious = [], []
    train_ious, val_ious = [], []
    best_epoch, best_val_miou = start_epoch - 1, start_miou
    metric = MeanIoU(num_classes=num_classes).to(device)

    global_step = 0

    for epoch in range(start_epoch, end_epoch):
        model.train()
        ema_model.eval()

        data_len, tot_batches = 0, max(len(trainloader_source), len(trainloader_target))
        loss_l_value, loss_u_value = 0.0, 0.0
        epoch_task_specific_losses = {}
        metric.reset()

        def infinite_iterator(dataloader):
            while True:
                for batch in dataloader:
                    yield batch

        iter_source = infinite_iterator(trainloader_source) if len(trainloader_source) < len(trainloader_target) else iter(trainloader_source)
        iter_target = infinite_iterator(trainloader_target) if len(trainloader_target) < len(trainloader_source) else iter(trainloader_target)

        for i, (batch_source, batch_target) in enumerate(zip(iter_source, iter_target)):
            for idx in range(len(batch_source)):
                batch_source[idx] = batch_source[idx].to(device)

            if bd_required:
                inputs_source, gt_source = batch_source[0], batch_source[1:]
                masks_source = gt_source[0]
                boundaries_source = gt_source[1]
            else:
                inputs_source, gt_source = batch_source[0], batch_source[1]
                masks_source = gt_source
                boundaries_source = None

            target_inputs = batch_target[0].to(device)
            data_len += inputs_source.size(0)

            optimizer.zero_grad(set_to_none=True)

            outputs_source = model(inputs_source)
            if isinstance(model, PIDNet):
                source_loss_res = criterion(outputs_source, (masks_source, boundaries_source))
            else:
                source_loss_res = criterion(outputs_source, masks_source)
            
            if isinstance(source_loss_res, tuple):
                source_loss, batch_task_specific_losses = source_loss_res
            else:
                source_loss, batch_task_specific_losses = source_loss_res, {}

            teacher = ema_model if use_ema_for_pseudo else model
            with torch.no_grad():
                target_logits = teacher(target_inputs)
                target_pred = _extract_pred(target_logits, model)
                pseudo_label = torch.softmax(target_pred.detach(), dim=1)
                max_probs, targets_u_w = torch.max(pseudo_label, dim=1)

            inputs_u_s, targets_u, mix_masks = augment_and_mix(
                inputs_source, masks_source, target_inputs, targets_u_w, device, ignore_index
            )

            outputs_u_s = model(inputs_u_s)

            if pixel_weight == "threshold_uniform":
                unlabeled_weight = torch.sum(max_probs.ge(pseudo_threshold).long() == 1).item() / targets_u.numel()
                pixel_wise_weight = unlabeled_weight * torch.ones_like(max_probs).to(device)
            elif pixel_weight == "threshold":
                pixel_wise_weight = max_probs.ge(pseudo_threshold).float().to(device)
            else:
                pixel_wise_weight = torch.ones_like(max_probs).to(device)

            pixel_weights = mix_pixel_weights(inputs_u_s, mix_masks, pixel_wise_weight, device)

            loss_u = calc_u_loss(outputs_u_s, targets_u, model, criterion)
            loss_u = loss_u * torch.mean(pixel_weights)

            loss = source_loss + loss_u
            loss.backward()
            optimizer.step()

            loss_l_value += source_loss.item() * inputs_source.size(0)
            loss_u_value += loss_u.item() * inputs_source.size(0)

            for task, task_loss in batch_task_specific_losses.items():
                if task not in epoch_task_specific_losses:
                    epoch_task_specific_losses[task] = 0.0
                epoch_task_specific_losses[task] += task_loss.item() * inputs_source.size(0)

            pred_source = _extract_pred(outputs_source, model)
            metric.update(pred_source, masks_source)
            current_miou, _ = metric.compute()

            global_step += 1
            update_ema_variables(ema_model, model, alpha_teacher, global_step)

            if (i + 1) % log_frequency == 0:
                inner_losses_str = " | ".join([f"{k}: {v / data_len:.4f}" for k, v in epoch_task_specific_losses.items()])
                logging.info(
                    f"Epoch {epoch + 1}/{end_epoch} | Batch {i + 1}/{tot_batches} | "
                    f"Loss L: {loss_l_value / data_len:.4f} | {inner_losses_str} | Loss U: {loss_u_value / data_len:.4f} | "
                    f"Loss: {(loss_l_value + loss_u_value) / data_len:.4f} | "
                    f"Train mIoU (%): {100 * current_miou.item():.2f}"
                )

        train_loss = (loss_l_value + loss_u_value) / data_len
        train_epoch_miou, train_epoch_ious = map(lambda x: 100 * x, metric.compute())
        train_epoch_miou_val = train_epoch_miou.item() if isinstance(train_epoch_miou, torch.Tensor) else float(train_epoch_miou)

        val_loss, val_epoch_miou, val_epoch_ious = evaluate_model(
            model,
            model_name,
            num_classes,
            validloader,
            criterion,
            bd_required,
            epoch,
            end_epoch,
            device,
            log_frequency,
        )
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        train_mious.append(train_epoch_miou_val)
        val_mious.append(float(val_epoch_miou))
        train_ious.append(train_epoch_ious.tolist())
        val_ious.append(val_epoch_ious)

        for task, task_loss in epoch_task_specific_losses.items():
            if f"train_losses_{task}" not in train_task_specific_losses:
                train_task_specific_losses[f"train_losses_{task}"] = []
            train_task_specific_losses[f"train_losses_{task}"].append(float(task_loss / data_len))

        logging.info(
            f"Epoch {epoch + 1}/{end_epoch} | Train Loss: {train_loss:.4f} | "
            f"Train mIoU (%): {train_epoch_miou_val:.2f} | Val Loss: {val_loss:.4f} | "
            f"Val mIoU (%): {val_epoch_miou:.2f}"
        )

        if best_val_miou is None or val_epoch_miou > best_val_miou:
            prev_best_epoch = best_epoch
            best_val_miou = val_epoch_miou
            best_epoch = epoch
            chp_path = os.path.join(checkpoint_dir, f"best_{model_name}_dacs_{epoch + 1}.pth.tar")
            save_checkpoint(
                chp_path,
                epoch + 1,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                miou=val_epoch_miou,
                ious=val_epoch_ious,
            )

            if prev_best_epoch >= start_epoch:
                prev_chp_path = os.path.join(checkpoint_dir, f"best_{model_name}_dacs_{prev_best_epoch + 1}.pth.tar")
                if os.path.exists(prev_chp_path):
                    os.remove(prev_chp_path)

        if scheduler is not None:
            scheduler.step()

    last_chp_path = os.path.join(checkpoint_dir, f"last_{model_name}_dacs_{end_epoch}.pth.tar")
    last_miou = val_mious[-1] if len(val_mious) > 0 else None
    last_ious = val_ious[-1] if len(val_ious) > 0 else None
    last_epoch = end_epoch if end_epoch > start_epoch else start_epoch
    save_checkpoint(
        last_chp_path,
        last_epoch,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        miou=last_miou,
        ious=last_ious,
    )

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        **train_task_specific_losses,
        "train_mious": train_mious,
        "val_mious": val_mious,
        "train_ious": train_ious,
        "val_ious": val_ious
    }