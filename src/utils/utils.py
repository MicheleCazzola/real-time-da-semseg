import os
import json
import logging
import torch
import random
import numpy as np
from box import Box
import logging
import argparse
from functools import partial

from src.utils.variables import ModelType, Domain, AdaptationMethod

def seed_worker(_):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Set seed for reproducibility
def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    g = torch.Generator()
    g.manual_seed(0)
    
    return g, partial(seed_worker)

def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        filename=f"{output_dir}/training.log",
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        encoding='utf-8'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)

def set_default_config(cfg, args):
    for key, value in cfg.items():
        if isinstance(value, (dict, Box)):
            set_default_config(value, args)
        elif key in args and args[key] is not None:
            cfg[key] = args[key]

# Checkpoint resume
def load_checkpoint(chp_path, model, device, disc_model=None, optimizer=None, disc_optimizer=None, scheduler=None, disc_scheduler=None, epochs=None):
    checkpoint = torch.load(chp_path, weights_only=False, map_location=device)
    
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])
    
    if disc_model is not None:
        if isinstance(disc_model, list):
            for d, s in zip(disc_model, checkpoint['disc_model']):
                d.load_state_dict(s)
        else:
            disc_model.load_state_dict(checkpoint['disc_model'])
        
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if disc_optimizer is not None:
        if isinstance(disc_optimizer, list):
            for d, s in zip(disc_optimizer, checkpoint['disc_optimizer']):
                d.load_state_dict(s)
        else:
            disc_optimizer.load_state_dict(checkpoint['disc_optimizer'])
            
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    if disc_scheduler is not None:
        if isinstance(disc_scheduler, list):
            for d, s in zip(disc_scheduler, checkpoint['disc_scheduler']):
                d.load_state_dict(s)
        else:
            disc_scheduler.load_state_dict(checkpoint['disc_scheduler'])
        
    best_miou = checkpoint.get('miou', None)
    ious = checkpoint.get('ious', None)
        
    return {
        "epoch": epoch,
        "miou": best_miou,
        "ious": ious
    }

# Checkpoint save
def save_checkpoint(path, epoch, model, disc_model=None, optimizer=None, disc_optimizer=None, scheduler=None, disc_scheduler=None, miou=None, ious=None):
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict()
    }
    
    if disc_model is not None:
        if isinstance(disc_model, list):
            checkpoint['disc_model'] = [d.state_dict() for d in disc_model]
        else:
            checkpoint['disc_model'] = disc_model.state_dict()
            
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    if disc_optimizer is not None:
        if isinstance(disc_optimizer, list):
            checkpoint['disc_optimizer'] = [d.state_dict() for d in disc_optimizer]
        else:
            checkpoint['disc_optimizer'] = disc_optimizer.state_dict()
            
    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()
    if disc_scheduler is not None:
        if isinstance(disc_scheduler, list):
            checkpoint['disc_scheduler'] = [d.state_dict() for d in disc_scheduler]
        else:
            checkpoint['disc_scheduler'] = disc_scheduler.state_dict()
            
    if miou is not None:
        checkpoint['miou'] = float(miou)
    if ious is not None:
        checkpoint['ious'] = ious.tolist() if isinstance(ious, torch.Tensor) else ious
        
    torch.save(checkpoint, path)

def get_num_workers(device):
    num_cpus = os.cpu_count()
    if device.type == 'cuda':
        return 0                # Colab environment
    elif device.type == 'mps':
        return min(6, num_cpus // 2)                
    else:
        return num_cpus // 2  # Use half of available CPUs for CPU training

def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available() else 
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

def save_results(destination, **results):
    json_str = json.dumps(results, indent=4)
    with open(destination, 'w') as f:
        f.write(json_str)
        
def get_args():
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
    parser.add_argument('--output-dir', type=str, help='Directory where outputs (checkpoints, logs, results) will be saved.')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of iterations for performance measurement.')
    parser.add_argument('--loss', type=str, help='Loss function to use for training.')
    parser.add_argument('--checkpoint-path', type=str, help='Path to a checkpoint to resume training or for evaluation.')
    parser.add_argument('--pretrained-path', type=str, help='Path to pretrained weights for model initialization.')
    parser.add_argument('--last-epoch', type=int, help='The last epoch number to train up to (used for training across multiple runs).')
    parser.add_argument('--warmup-epochs', type=int, help='Number of warmup epochs for learning rate scheduling.')
    parser.add_argument('--double-eval', action=argparse.BooleanOptionalAction, help='Flag to indicate whether to evaluate the model on both domains.')
    args = parser.parse_args()
    
    return args