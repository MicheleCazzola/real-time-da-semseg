import logging
import os
import yaml
import argparse
from box import Box
from datetime import datetime

import torch
import torch.multiprocessing

from src.train.iast import iast_setup, train_iast
torch.multiprocessing.set_sharing_strategy('file_system')

from src.dataset.dataset import LoveDA
from src.dataset.augmentations import get_augmentations, get_nop_augmentation
from src.metrics.resources import compute_performance_metrics
from src.train.adda_multi import adda_multi_setup, train_adda_multi
from src.train.train_model import train_model, setup_model, evaluate_model
from src.train.adda import adda_setup, train_adda
from src.utils.plot import plot_results
from src.utils.utils import set_default_config, set_seed, get_num_workers, setup_logger, get_device, save_results
from src.utils.variables import ModelType, Domain, AdaptationMethod
from src.train.utils import trainset_setup, validset_setup
from src.train.deeplab_v2 import deeplab_v2_model_setup, train_deeplab_v2, evaluate_deeplab_v2
from src.train.adda import adda_setup, train_adda
from src.train.dacs import dacs_setup, train_dacs

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a semantic segmentation model on urban and rural datasets.')
    parser.add_argument(
        '--from-config',
        type=str,
        default=os.path.join('configs', 'config.yaml'),
        help='Path to the YAML configuration file.',
    )
    parser.add_argument(
        '--model', type=str, choices=ModelType.values(), help='The model architecture to use for training.'
    )
    parser.add_argument('--source', type=str, choices=Domain.values(), help='The source domain for training.')
    parser.add_argument('--target', type=str, choices=Domain.values(), help='The target domain for training.')
    parser.add_argument(
        '--augment', action='store_true', help='Flag to indicate whether to apply data augmentation during training.'
    )
    parser.add_argument(
        '--adaptation',
        type=str,
        choices=AdaptationMethod.values(),
        help='The domain adaptation method to use for training.',
    )
    parser.add_argument('--train', action='store_true', help='Flag to indicate whether to train the model.')
    parser.add_argument(
        '--evaluate', action='store_true', help='Flag to indicate whether to evaluate the model on the test set.'
    )
    parser.add_argument(
        '--measure',
        action='store_true',
        help='Flag to indicate whether to compute performance metrics after evaluation.',
    )
    parser.add_argument(
        '--reduce-factor',
        type=float,
        default=1.0,
        help='Factor (ratio) by which to reduce the dataset size for faster experimentation.',
    )
    parser.add_argument('--epochs', type=int, help='Number of training epochs.')
    parser.add_argument(
        '--output-dir', type=str, help='Directory where outputs (checkpoints, logs, results) will be saved.'
    )
    parser.add_argument(
        '--iterations', type=int, default=1000, help='Number of iterations for performance measurement.'
    )
    parser.add_argument('--loss', type=str, help='Loss function to use for training.')
    parser.add_argument('--checkpoint-path', type=str, help='Path to a checkpoint to resume training or for evaluation.')
    parser.add_argument('--pretrained-path', type=str, help='Path to pretrained weights for model initialization.')
    parser.add_argument('--last-epoch', type=int, help='The last epoch number to train up to (used for training across multiple runs).')
    parser.add_argument('--iast-regenerate', action='store_true', help='Flag to indicate whether to regenerate pseudo-labels at the beginning of training and every N epochs during IAST training.')
    args = parser.parse_args()

    with open(args.from_config, 'r') as f:
        cfg = Box(yaml.safe_load(f))

    set_default_config(cfg, vars(args))

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(cfg.path.output_dir, cfg.model.model, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    setup_logger(output_dir)

    config_json = cfg.to_json(indent=4)
    logging.info(f"Configuration:\n{config_json}")

    # Auto-save parameters at run start
    config_path = os.path.join(output_dir, f"config.json")
    with open(config_path, 'w') as f:
        f.write(config_json)

    logging.info(f"Execution configuration saved in {config_path}")

    logging.info(f"Source domain: {cfg.path.source}, Target domain: {cfg.path.target}, Model: {cfg.model.model}")

    # Set seed for reproducibility
    g, seed_worker = set_seed()

    # Get device
    device = get_device()

    # Get number of workers for dataloader
    num_workers = get_num_workers(device)

    # Load dataset and dataloaders
    labels = LoveDA.id2label.values()

    # Load augmentations
    if cfg.data.augment:
        augmentations = get_augmentations(cfg.data.aug_prob, cfg.data.aug_prob_one_of, cfg.data.aug_names)
    else:
        augmentations = None

    match cfg.model.model:
        case ModelType.DEEPLAB_V2.value:
            bd_required = False
            backbone_name = None
            make_train_specific_losses = lambda _: {}
        case ModelType.BISENET_V1.value | ModelType.BISENET_V1_RT.value:
            bd_required = False
            backbone_name = "resnet18" if cfg.model.model == ModelType.BISENET_V1_RT.value else "resnet101"
            make_train_specific_losses = lambda train_result: {
                "semantic": {
                    k: v
                    for k, v in train_result.items()
                    if k not in ["train_losses", "val_losses", "train_mious", "val_mious", "train_ious", "val_ious"]
                }
            }
        case ModelType.STDC1.value | ModelType.STDC2.value:
            bd_required = False
            backbone_name = "STDCNet813" if cfg.model.model == ModelType.STDC1.value else "STDCNet1446"
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
            raise NotImplementedError(f"Model {cfg.model.model} not supported")

    model, criterion, optimizer, scheduler, start_epoch, start_miou = setup_model(cfg, device, backbone_name)

    if cfg.training.train:
        trainset_source, trainloader_source = trainset_setup(
            cfg,
            cfg.path.source,
            g,
            seed_worker,
            num_workers,
            augmentations=augmentations,
            reduce_factor=args.reduce_factor,
            boundaries=bd_required,
        )
        trainset_target, trainloader_target = trainset_setup(
            cfg,
            cfg.path.target,
            g,
            seed_worker,
            num_workers,
            split_dir=cfg.path.val_dir,
            augmentations=augmentations,
            reduce_factor=args.reduce_factor,
            boundaries=bd_required,
        )
        validset, validloader = validset_setup(
            cfg,
            cfg.path.target,
            num_workers,
            g,
            seed_worker,
            reduce_factor=args.reduce_factor,
            boundaries=bd_required,
        )
         
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
        elif cfg.training.adaptation == AdaptationMethod.ADDA_MULTI.value:
            discriminators, criterions_disc, disc_optimizers, disc_schedulers = adda_multi_setup(cfg, device)
            train_result = train_adda_multi(
                model,
                discriminators,
                cfg.model.model,
                cfg.model.num_classes,
                cfg.adda.lambda_adv,
                trainloader_source,
                trainloader_target,
                validloader,
                criterion,
                criterions_disc,
                optimizer,
                disc_optimizers,
                scheduler,
                disc_schedulers,
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
                pixel_weight=cfg.dacs.pixel_weight if hasattr(cfg, "dacs") else "threshold",
                pseudo_threshold=cfg.dacs.pseudo_threshold if hasattr(cfg, "dacs") else 0.968,
                use_ema_for_pseudo=cfg.dacs.use_ema_for_pseudo if hasattr(cfg, "dacs") else True,
                alpha_teacher=cfg.dacs.alpha_teacher if hasattr(cfg, "dacs") else 0.99,
                ignore_index=cfg.model.ignore_index
            )
            train_specific_losses = make_train_specific_losses(train_result)
        elif cfg.training.adaptation == AdaptationMethod.IAST.value:
            with open(os.path.join("configs", f"iast.yaml"), 'r') as f:
                iast_cfg = Box(yaml.safe_load(f))
                
            trainset_target, trainloader_target = trainset_setup(
                cfg,
                cfg.path.target,
                g,
                seed_worker,
                num_workers,
                reduce_factor=args.reduce_factor,
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
                    "reduce_factor": args.reduce_factor,
                    "g": g,
                    "seed_worker": seed_worker,
                    "num_workers": num_workers,
                    "augmentations": augmentations,
                },
                device=device,
                checkpoint_dir=output_dir,
                regenerate=args.iast_regenerate,
                log_frequency=10,
            )
            train_specific_losses = make_train_specific_losses(train_result)
        else:
            raise NotImplementedError(f"Adaptation method {cfg.adaptation.adaptation_method} not supported")

        save_results(os.path.join(output_dir, f"results.json"), **train_result)

        plot_results(
            dir_path=output_dir,
            labels=labels,
            main_losses=dict(
                zip(["train_losses", "val_losses"], list(map(train_result.get, ["train_losses", "val_losses"])))
            ),
            mean_ious=dict(
                zip(["train_mious", "val_mious"], list(map(train_result.get, ["train_mious", "val_mious"])))
            ),
            ious_per_class=dict(
                zip(["train_ious", "val_ious"], list(map(train_result.get, ["train_ious", "val_ious"])))
            ),
            train_losses=train_specific_losses,
            show=False,
        )
    elif cfg.training.evaluate:
        validset, validloader = validset_setup(
            cfg,
            cfg.path.target,
            num_workers,
            g,
            seed_worker,
            reduce_factor=args.reduce_factor,
            boundaries=bd_required
        )

        loss, miou, ious = evaluate_model(model, cfg.model.model, cfg.model.num_classes, validloader, criterion, bd_required, -1, 0, device, 1)
        
        save_results(os.path.join(output_dir, f"results.json"), val_loss=loss, val_miou=miou, val_ious=ious)

    if cfg.training.measure:
        resource_metrics = compute_performance_metrics(
            model,
            args.iterations,
            cfg.data.dimensions.height,
            cfg.data.dimensions.width,
            device,
            save_to=os.path.join(output_dir, f"performance_metrics.json"),
            return_out=False,
        )