from src.train.adda import adda_setup, train_adda
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
    cfg, model, trainloader_source, trainloader_target, validloader, criterion, optimizer, scheduler, 
    start_epoch, start_miou, bd_required, make_train_specific_losses, output_dir, device
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
    else:
        raise NotImplementedError(f"Adaptation method {cfg.adaptation.adaptation_method} not supported")
    
    return train_result, train_specific_losses