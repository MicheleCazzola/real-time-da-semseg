from pathlib import Path
import os
import json
import logging
import torch
import random
import gdown
import numpy as np
from box import Box
import logging
from functools import partial

from src.utils.variables import categories

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
    """
    Overwrites YAML config parameters with command-line arguments (if present).
    Priority: argparse > YAML config.
    """
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
        disc_model.load_state_dict(checkpoint['disc_model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if disc_optimizer is not None:
        disc_optimizer.load_state_dict(checkpoint['disc_optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    if disc_scheduler is not None:
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
        checkpoint['disc_model'] = disc_model.state_dict()
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    if disc_optimizer is not None:
        checkpoint['disc_optimizer'] = disc_optimizer.state_dict()
    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()
    if disc_scheduler is not None:
        checkpoint['disc_scheduler'] = disc_scheduler.state_dict()
    if miou is not None:
        checkpoint['miou'] = float(miou)
    if ious is not None:
        checkpoint['ious'] = ious.tolist() if isinstance(ious, torch.Tensor) else ious
        
    torch.save(checkpoint, path)
    
# Load model weights
def load_model_weights(weights_dir_name, weights_model_name, file_id):
    weights_dir = Path(weights_dir_name)
    if not weights_dir.exists():
        weights_dir.mkdir(exist_ok=True)

    weights_model = Path(weights_model_name)
    if not weights_model.exists():
        gdown.download(id=file_id, output=str(weights_model), quiet=False)
        
    return weights_dir, weights_model

def get_num_workers(device):
    num_cpus = os.cpu_count()
    if device.type == 'cuda':
        return 0                # Colab environment -> Issues with multiple workers and CUDA, set to 0 for safe execution
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

def get_mious_per_category(mious_per_class):
    mious_per_category = {}
    for i, cat in enumerate(categories.keys()):
        if cat in mious_per_category:
            mious_per_category[cat] += [mious_per_class[i].item()]
        else:
            mious_per_category[cat] = [mious_per_class[i].item()]
            
        logging.info(f"{cat} mIoU: {mious_per_class[i]:.2f}%")
        
    return mious_per_category

def save_results(destination, **results):
    json_str = json.dumps(results, indent=4)
    with open(destination, 'w') as f:
        f.write(json_str)