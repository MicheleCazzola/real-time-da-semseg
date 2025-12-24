from pathlib import Path
import torch
import random
import gdown
import numpy as np

from src.utils.variables import device, categories

# Set seed for reproducibility
def set_seed():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def seed_worker(_):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)
    
    return g, seed_worker

# Checkpoint resume
def resume_checkpoint(resume_path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(resume_path)
    iteration = checkpoint['iteration'] + 1
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
      optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
      scheduler.load_state_dict(checkpoint['scheduler'])
    return iteration, model, optimizer, scheduler

# Checkpoint save
def save_checkpoint(path, iteration, model, optimizer, scheduler):
    checkpoint = {
        'iteration': iteration,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
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

def get_num_workers():
    return 2 if device == "cuda" else 0

def get_mious_per_category(mious_per_class):
    mious_per_category = {}
    for i, cat in enumerate(categories.keys()):
        if cat in mious_per_category:
            mious_per_category[cat] += [mious_per_class[i].item()]
        else:
            mious_per_category[cat] = [mious_per_class[i].item()]
            
        print(f"{cat} mIoU: {mious_per_class[i]:.2f}%")
        
    return mious_per_category