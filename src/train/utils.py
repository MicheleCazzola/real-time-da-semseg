import os

from box.box import Box
import torch
import yaml

from src.dataset.utils import trainset_setup
from src.train.adda import adda_setup, train_adda
from src.train.dacs import dacs_setup, train_dacs
from src.train.iast import iast_setup, train_iast
from src.train.train_model import train_model
from src.utils.variables import ModelType, AdaptationMethod

def get_train_params(model_name):
    match model_name:
        case ModelType.DEEPLAB_V2.value:
            bd_required = False
            backbone_name = None
            make_train_specific_losses = lambda _: {}
        case ModelType.BISENET_V1.value | ModelType.BISENET_V1_RT.value:
            bd_required = False
            backbone_name = "resnet18" if model_name == ModelType.BISENET_V1_RT.value else "resnet101"
            make_train_specific_losses = lambda train_result: {
                "semantic": {
                    k: v
                    for k, v in train_result.items()
                    if k not in ["train_losses", "val_losses", "train_mious", "val_mious", "train_ious", "val_ious"]
                }
            }
        case ModelType.STDC1.value | ModelType.STDC2.value:
            bd_required = False
            backbone_name = "STDCNet813" if model_name == ModelType.STDC1.value else "STDCNet1446"
            make_train_specific_losses = lambda train_result: {
                "semantic": {
                    k: v
                    for k, v in train_result.items()
                    if "sem" in k
                    and k
                    not in ["train_losses", "val_losses", "train_mious", "val_mious", "train_ious", "val_ious"]
                },
                "boundary": {k: v for k, v in train_result.items() if "boundary" in k},
            }
        case ModelType.PIDNET_S.value | ModelType.PIDNET_M.value | ModelType.PIDNET_L.value:
            bd_required = True
            backbone_name = None
            make_train_specific_losses = lambda train_result: {
                "detail": {
                    k: v
                    for k, v in train_result.items()
                    if k not in ["train_losses", "val_losses", "train_mious", "val_mious", "train_ious", "val_ious"]
                }
            }
        case _:
            raise NotImplementedError(f"Model {model_name} not supported")
        
    return bd_required, backbone_name, make_train_specific_losses

def train(
    cfg, model, trainloader_source, trainloader_target, validloader, criterion, optimizer, scheduler, start_epoch, start_miou,
    bd_required, make_train_specific_losses, output_dir, device, g, seed_worker, num_workers, reduce_factor, augmentations, iast_regenerate
):
    if cfg.training.adaptation is None:
        train_result = train_model(
            model,
            cfg.model.model,
            cfg.model.num_classes,
            trainloader_source,
            validloader,
            criterion,
            optimizer,
            scheduler,
            start_epoch,
            cfg.training.last_epoch,
            start_miou,
            bd_required=bd_required,
            checkpoint_dir=output_dir,
            device=device,
            log_frequency=10,
        )
        train_specific_losses = make_train_specific_losses(train_result)
    elif cfg.training.adaptation == AdaptationMethod.ADDA.value:
        discriminator, disc_criterion, disc_optimizer, disc_scheduler = adda_setup(cfg, device)
        train_result = train_adda(
            model,
            discriminator,
            cfg.model.model,
            cfg.model.num_classes,
            cfg.adda.lambda_adv,
            trainloader_source,
            trainloader_target,
            validloader,
            criterion,
            disc_criterion,
            optimizer,
            disc_optimizer,
            scheduler,
            disc_scheduler,
            start_epoch,
            cfg.training.last_epoch,
            start_miou,
            bd_required=bd_required,
            checkpoint_dir=output_dir,
            device=device,
            log_frequency=10,
        )
        train_specific_losses = make_train_specific_losses(train_result)
        train_specific_losses.update(
            {
                "discriminator": {
                    "train_losses_adda_disc_source": train_result.get("train_losses_disc_source", []),
                    "train_losses_adda_disc_target": train_result.get("train_losses_disc_target", []),
                }
            }
        )
    elif cfg.training.adaptation == AdaptationMethod.DACS.value:
        ema_model = dacs_setup(cfg, model, device)
        train_result = train_dacs(
            model,
            ema_model,
            cfg.model.model,
            cfg.model.num_classes,
            trainloader_source,
            trainloader_target,
            validloader,
            criterion,
            optimizer,
            scheduler,
            start_epoch,
            cfg.training.last_epoch,
            start_miou,
            bd_required=bd_required,
            checkpoint_dir=output_dir,
            device=device,
            log_frequency=10,
            pixel_weight=cfg.dacs.pixel_weight,
            pseudo_threshold=cfg.dacs.pseudo_threshold,
            use_ema_for_pseudo=cfg.dacs.use_ema_for_pseudo,
            alpha_teacher=cfg.dacs.alpha_teacher,
            ignore_index=cfg.model.ignore_index
        )
        train_specific_losses = make_train_specific_losses(train_result)
    elif cfg.training.adaptation == AdaptationMethod.IAST.value:
        with open(os.path.join("configs", f"iast.yaml"), 'r') as f:
            iast_cfg = Box(yaml.safe_load(f))
            
        _, trainloader_target = trainset_setup(
            cfg,
            cfg.path.target,
            g,
            seed_worker,
            num_workers,
            reduce_factor=reduce_factor,
            boundaries=bd_required,
            img_names=True,
            shuffle=False,  # Important to keep track of pseudo-labels across epochs
            drop_last=False,  # Important to keep track of pseudo-labels across epochs
            resize=False
        )
        
        model_D, criterion_D, criterion_ent, criterion_kd, optimizer_D, scheduler_D = iast_setup(iast_cfg, device)
        criterion_ent = torch.nn.CrossEntropyLoss(ignore_index=cfg.model.ignore_index)
        criterion_kd = torch.nn.KLDivLoss(reduction='batchmean')
        train_result = train_iast(
            model,
            cfg.model.model,
            model_D,
            trainloader_source,
            trainloader_target,
            validloader,
            criterion,
            criterion_D,
            criterion_ent,
            criterion_kd,
            optimizer,
            optimizer_D,
            scheduler,
            scheduler_D,
            cfg.training.last_epoch,
            bd_required=bd_required,
            cfg=cfg,
            iast_cfg=iast_cfg,
            trainset_build_params={
                "reduce_factor": reduce_factor,
                "g": g,
                "seed_worker": seed_worker,
                "num_workers": num_workers,
                "augmentations": augmentations,
            },
            device=device,
            checkpoint_dir=output_dir,
            regenerate=iast_regenerate,
            log_frequency=10,
        )
        train_specific_losses = make_train_specific_losses(train_result)
    else:
        raise NotImplementedError(f"Adaptation method {cfg.adaptation.adaptation_method} not supported")
    
    return train_result, train_specific_losses