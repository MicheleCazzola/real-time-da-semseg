import json
import logging
import os
import yaml
from box import Box
from datetime import datetime

from src.dataset.dataset import LoveDA
from src.dataset.augmentations import get_augmentations
from src.metrics.resources import compute_performance_metrics
from src.train.train_model import setup_model, evaluate_model
from src.utils.plot import plot_results
from src.utils.utils import set_default_config, set_seed, get_num_workers, setup_logger, get_device, save_results
from src.dataset.utils import trainset_setup, validset_setup
from src.train.utils import get_train_params, train
from src.utils.utils import get_args

if __name__ == "__main__":

    args = get_args()
    
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

    bd_required, backbone_name, make_train_specific_losses = get_train_params(cfg.model.model)
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
        
        if cfg.training.double_eval:
            validset_adj, validloader_adj = validset_setup(
                cfg,
                "urban" if cfg.path.target == "rural" else "rural",
                num_workers,
                g,
                seed_worker,
                reduce_factor=args.reduce_factor,
                boundaries=bd_required,
            )
        else:
            validset_adj, validloader_adj = None, None
        
        train_result, train_specific_losses = train(
            cfg, model, trainloader_source, trainloader_target, validloader, validloader_adj, criterion, optimizer, scheduler, 
            start_epoch, start_miou, bd_required, make_train_specific_losses, output_dir, device
        )
         
        save_results(os.path.join(output_dir, f"results.json"), **train_result)

        plot_results(
            dir_path=output_dir,
            labels=labels,
            main_losses=dict(
                zip(["train_losses", "val_losses"] + (["val_losses_adj"] if validloader_adj is not None else []), list(map(train_result.get, ["train_losses", "val_losses"] + (["val_losses_adj"] if validloader_adj is not None else []))))
            ),
            mean_ious=dict(
                zip(["train_mious", "val_mious"] + (["val_mious_adj"] if validloader_adj is not None else []), list(map(train_result.get, ["train_mious", "val_mious"] + (["val_mious_adj"] if validloader_adj is not None else []))))
            ),
            ious_per_class=dict(
                zip(["train_ious", "val_ious"] + (["val_ious_adj"] if validloader_adj is not None else []), list(map(train_result.get, ["train_ious", "val_ious"] + (["val_ious_adj"] if validloader_adj is not None else []))))
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
        
        # with open(os.path.join(output_dir, f"uq_infos.json"), 'w') as f:
        #     json.dump(uq_infos, f, indent=4)

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