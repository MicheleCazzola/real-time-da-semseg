import logging
import os
import yaml
from box import Box
from datetime import datetime

import argparse

from src.dataset.dataset import LoveDA
from src.dataset.augmentations import get_augmentations, get_nop_augmentation
from src.metrics.metrics import compute_performance_metrics
from src.train.train_rt_model import train_rt_model, setup_rt_model
from src.utils.plot import plot_class_distribution, plot_results
from src.utils.utils import load_model_weights, set_default_config, set_seed, get_num_workers, setup_logger, get_device, save_results
from src.utils.variables import ModelType, Domain, AdaptationMethod, urban_percentage, rural_percentage
from src.train.utils import trainset_setup, validset_setup
from src.train.deeplab_v2 import deeplab_v2_model_setup, train_deeplab_v2, evaluate_deeplab_v2
from src.train.adda import adda_setup, train_adda
from src.train.dacs import dacs_setup, train_dacs



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train a semantic segmentation model on urban and rural datasets.')
    parser.add_argument('--from-config', type=str, default=os.path.join('configs', 'config.yaml'), help='Path to the YAML configuration file.')
    parser.add_argument('--model', type=str, choices=ModelType.values(), help='The model architecture to use for training.')
    parser.add_argument('--source', type=str, choices=Domain.values(), help='The source domain for training.')
    parser.add_argument('--target', type=str, choices=Domain.values(), help='The target domain for training.')
    parser.add_argument('--augment', action='store_true', help='Flag to indicate whether to apply data augmentation during training.')
    parser.add_argument('--adaptation', type=str, choices=AdaptationMethod.values(), help='The domain adaptation method to use for training.')
    parser.add_argument('--train', action='store_true', help='Flag to indicate whether to train the model.')
    parser.add_argument('--evaluate', action='store_true', help='Flag to indicate whether to evaluate the model on the test set.')
    parser.add_argument('--measure', action='store_true', help='Flag to indicate whether to compute performance metrics after evaluation.')
    parser.add_argument('--reduce-factor', type=float, default=1.0, help='Factor (ratio) by which to reduce the dataset size for faster experimentation.')
    parser.add_argument('--epochs', type=int, help='Number of training epochs.')
    parser.add_argument('--output-dir', type=str, help='Directory where outputs (checkpoints, logs, results) will be saved.')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of iterations for performance measurement.')
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
    
    # Load dataset and dataloaders
    id2label = LoveDA.id2label.values()
    
    # Load augmentations
    if cfg.data.augment:
        augmentations = get_augmentations(cfg.data.aug_prob, cfg.data.aug_names)
    else:
        augmentations = get_nop_augmentation()
    
    match cfg.model.model:
        
        case ModelType.DEEPLAB_V2.value:
            model, criterion, optimizer, scheduler = deeplab_v2_model_setup(cfg, device)
                
            if cfg.training.train:
                trainset_source, trainloader_source = trainset_setup(cfg, cfg.path.source, g, seed_worker, num_workers, augmentations=augmentations)
                trainset_target, trainloader_target = trainset_setup(cfg, cfg.path.target, g, seed_worker, num_workers, augmentations=augmentations)
                validset, validloader = validset_setup(cfg, cfg.path.target, num_workers, g, seed_worker)

                chp_path = os.path.join(output_dir, cfg.model.checkpoint) if cfg.model.checkpoint else None
                new_chp_path = os.path.join(output_dir, f"{cfg.model.model}.pth.tar")
                
                train_losses, train_mious, train_ious = train_deeplab_v2(
                    model, cfg.model.num_classes, trainloader_source, cfg.training.epochs, criterion, optimizer,
                    scheduler, device, new_chp_path, cfg.training.resume, chp_path, log_frequency=10
                )
                val_losses, val_mious, val_ious = evaluate_deeplab_v2(
                    model, validloader, criterion, cfg.model.num_classes, device, new_chp_path,
                    start_epoch=0, num_epochs=cfg.training.epochs, log_frequency=10
                )
                
                save_results(
                    os.path.join(output_dir, f"results.json"),
                    train_losses=train_losses, train_mious=train_mious, train_ious=train_ious,
                    val_losses=val_losses, val_mious=val_mious, val_ious=val_ious
                )
                
                plot_results(
                    dir_path=output_dir,
                    id2label=id2label,
                    main_losses={"train_losses": train_losses, "val_losses": val_losses},
                    mean_ious={"train_mious": train_mious, "val_mious": val_mious},
                    ious_per_class={"train_ious": train_ious, "val_ious": val_ious},
                    train_losses=None,
                    show=False
                )
                
            if cfg.training.measure:
                resource_metrics = compute_performance_metrics(
                    model, cfg.model.num_classes, device, cfg.data.dimensions.height, cfg.data.dimensions.width,
                    iterations=args.iterations, mask_required=False, bd_required=False, save_to=os.path.join(output_dir, f"performance_metrics.json"), return_out=False
                )

        case ModelType.PIDNET_S.value | ModelType.PIDNET_M.value | ModelType.PIDNET_L.value:
            model, criterion, optimizer, scheduler = setup_rt_model(cfg, device)
            if cfg.training.train:
                trainset_source, trainloader_source = trainset_setup(cfg, cfg.path.source, g, seed_worker, num_workers, augmentations=augmentations, reduce_factor=args.reduce_factor, boundaries=True)
                trainset_target, trainloader_target = trainset_setup(cfg, cfg.path.target, g, seed_worker, num_workers, augmentations=augmentations, reduce_factor=args.reduce_factor, boundaries=True)
                validset, validloader = validset_setup(cfg, cfg.path.target, num_workers, g, seed_worker, reduce_factor=args.reduce_factor, boundaries=True)

                train_result = train_rt_model(
                    model, cfg.model.model, cfg.model.num_classes, trainloader_source, validloader,
                    criterion, optimizer, scheduler, cfg.training.epochs, bd_required=True, 
                    checkpoint_dir=output_dir, device=device, log_frequency=10
                )
                
                save_results(
                    os.path.join(output_dir, f"results.json"),
                    **train_result
                )
                
                train_specific_losses = {
                    "detail": {k: v for k, v in train_result.items() if k not in ["train_losses", "val_losses", "train_mious", "val_mious", "train_ious", "val_ious"]}
                }
                
                plot_results(
                    dir_path=output_dir,
                    id2label=id2label,
                    main_losses=dict(zip(["train_losses", "val_losses"], list(map(train_result.get, ["train_losses", "val_losses"])))),
                    mean_ious=dict(zip(["train_mious", "val_mious"], list(map(train_result.get, ["train_mious", "val_mious"])))),
                    ious_per_class=dict(zip(["train_ious", "val_ious"], list(map(train_result.get, ["train_ious", "val_ious"])))),
                    train_losses=train_specific_losses,
                    show=False
                )
            if cfg.training.measure:
                resource_metrics = compute_performance_metrics(
                    model, cfg.model.num_classes, device, cfg.data.dimensions.height, cfg.data.dimensions.width,
                    iterations=args.iterations, mask_required=False, bd_required=False, save_to=os.path.join(output_dir, f"performance_metrics.json"), return_out=False
                )

        case ModelType.BISENET_V1.value | ModelType.BISENET_V1_RT.value:
            backbone_name = "resnet18" if cfg.model.model == ModelType.BISENET_V1_RT.value else "resnet101"
            model, criterion, optimizer, scheduler = setup_rt_model(cfg, device, backbone_name)
                
            if cfg.training.train:
                trainset_source, trainloader_source = trainset_setup(cfg, cfg.path.source, g, seed_worker, num_workers, augmentations=augmentations, reduce_factor=args.reduce_factor, boundaries=False)
                trainset_target, trainloader_target = trainset_setup(cfg, cfg.path.target, g, seed_worker, num_workers, augmentations=augmentations, reduce_factor=args.reduce_factor, boundaries=False)
                validset, validloader = validset_setup(cfg, cfg.path.target, num_workers, g, seed_worker, reduce_factor=args.reduce_factor, boundaries=False)
                
                train_result = train_rt_model(
                    model, cfg.model.model, cfg.model.num_classes, trainloader_source, validloader,
                    criterion, optimizer, scheduler, cfg.training.epochs, bd_required=False,
                    checkpoint_dir=output_dir, device=device, log_frequency=10
                )
                
                save_results(
                    os.path.join(output_dir, f"results.json"),
                    **train_result
                )
                
                train_specific_losses = {
                    "semantic": {k: v for k, v in train_result.items() if k not in ["train_losses", "val_losses", "train_mious", "val_mious", "train_ious", "val_ious"]}
                }
                
                plot_results(
                    dir_path=output_dir,
                    id2label=id2label,
                    main_losses=dict(zip(["train_losses", "val_losses"], list(map(train_result.get, ["train_losses", "val_losses"])))),
                    mean_ious=dict(zip(["train_mious", "val_mious"], list(map(train_result.get, ["train_mious", "val_mious"])))),
                    ious_per_class=dict(zip(["train_ious", "val_ious"], list(map(train_result.get, ["train_ious", "val_ious"])))),
                    train_losses=train_specific_losses,
                    show=False
                )
            if cfg.training.measure:
                resource_metrics = compute_performance_metrics(
                    model, cfg.model.num_classes, device, cfg.data.dimensions.height, cfg.data.dimensions.width,
                    iterations=args.iterations, mask_required=False, bd_required=False, save_to=os.path.join(output_dir, f"performance_metrics.json"), return_out=False
                )
            
        case ModelType.STDC1.value | ModelType.STDC2.value:
            backbone_name = "STDCNet813" if cfg.model.model == ModelType.STDC1.value else "STDCNet1446"
            model, criterion, optimizer, scheduler = setup_rt_model(cfg, device, backbone_name)
                
            if cfg.training.train:
                trainset_source, trainloader_source = trainset_setup(cfg, cfg.path.source, g, seed_worker, num_workers, augmentations=augmentations, reduce_factor=args.reduce_factor, boundaries=False)
                trainset_target, trainloader_target = trainset_setup(cfg, cfg.path.target, g, seed_worker, num_workers, augmentations=augmentations, reduce_factor=args.reduce_factor, boundaries=False)
                validset, validloader = validset_setup(cfg, cfg.path.target, num_workers, g, seed_worker, reduce_factor=args.reduce_factor, boundaries=False)
                
                train_result = train_rt_model(
                    model, cfg.model.model, cfg.model.num_classes, trainloader_source, validloader,
                    criterion, optimizer, scheduler, cfg.training.epochs, bd_required=False,
                    checkpoint_dir=output_dir, device=device, log_frequency=10
                )
                
                save_results(
                    os.path.join(output_dir, f"results.json"),
                    **train_result
                )
                
                train_specific_losses = {
                    "semantic": {k: v for k, v in train_result.items() if "sem" in k and k not in ["train_losses", "val_losses", "train_mious", "val_mious", "train_ious", "val_ious"]},
                    "boundary": {k: v for k, v in train_result.items() if "boundary" in k}
                }
                
                plot_results(
                    dir_path=output_dir,
                    id2label=id2label,
                    main_losses=dict(zip(["train_losses", "val_losses"], list(map(train_result.get, ["train_losses", "val_losses"])))),
                    mean_ious=dict(zip(["train_mious", "val_mious"], list(map(train_result.get, ["train_mious", "val_mious"])))),
                    ious_per_class=dict(zip(["train_ious", "val_ious"], list(map(train_result.get, ["train_ious", "val_ious"])))),
                    train_losses=train_specific_losses,
                    show=False
                )
            if cfg.training.measure:
                resource_metrics = compute_performance_metrics(
                    model, cfg.model.num_classes, device, cfg.data.dimensions.height, cfg.data.dimensions.width,
                    iterations=args.iterations, mask_required=False, bd_required=False, save_to=os.path.join(output_dir, f"performance_metrics.json"), return_out=False
                )
            
        case _:
            raise NotImplementedError(f"Model {cfg.model.model} not implemented yet.")
        
    
    # logging.info(f"Sample image shape: {img.shape}, Sample mask shape: {mask.shape}, Sample boundary shape: {boundary.shape}")

    # logging.info255 * (img.permute(1, 2, 0).numpy() * cfg.data.imagenet_std + cfg.data.imagenet_mean).astype('uint8'))
    
    # plt.subplots(1, 3, figsize=(15, 5))
    # plt.subplot(1, 3, 1)
    # plt.imshow((255 * (img.permute(1, 2, 0).numpy() * cfg.data.imagenet_std + cfg.data.imagenet_mean)).astype('uint8'))
    # plt.imshow(mask.numpy(), alpha=0.5, cmap='jet')
    # plt.subplot(1, 3, 2)
    # plt.imshow(boundary, alpha=0.5, cmap='jet')
    # plt.subplot(1, 3, 3)
    # plt.imshow((255 * (img.permute(1, 2, 0).numpy() * cfg.data.imagenet_std + cfg.data.imagenet_mean)).astype('uint8'))
    # plt.show()