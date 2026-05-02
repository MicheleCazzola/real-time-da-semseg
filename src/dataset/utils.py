import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset.dataset import LoveDA
from src.utils.variables import categories, device, IMG_PATH, MASK_PATH, RURAL_PATH, TRAIN_DIR, URBAN_PATH

def compute_class_distribution(seed_worker, g):
    urban_dataset = LoveDA(TRAIN_DIR, IMG_PATH, MASK_PATH, directories=URBAN_PATH)
    urban_loader = DataLoader(urban_dataset, batch_size=64, worker_init_fn=seed_worker, generator=g)

    rural_dataset = LoveDA(TRAIN_DIR, IMG_PATH, MASK_PATH, directories=RURAL_PATH)
    rural_loader = DataLoader(rural_dataset, batch_size=64, worker_init_fn=seed_worker, generator=g)

    urban_classes = dict()
    rural_classes = dict()

    for (_, masks) in urban_loader:

        masks = masks.to(device)

        for i, cat in enumerate(categories.keys()):
            if cat in urban_classes:
                urban_classes[cat] += torch.count_nonzero(masks == i)
            else:
                urban_classes[cat] = torch.count_nonzero(masks == i)

    for (_, masks) in rural_loader:

        masks = masks.to(device)

        for i, cat in enumerate(categories.keys()):
            if cat in rural_classes:
                rural_classes[cat] += torch.count_nonzero(masks == i)
            else:
                rural_classes[cat] = torch.count_nonzero(masks == i)
                
    return urban_classes, rural_classes

def calc_weights(percentages):
    percentages = np.array(percentages)
    proportions = percentages / 100  # Divide by 100 to convert percentages to fractions

    # Calculate class weights inversely proportional to proportions
    class_weights = 1 / proportions

    # Optional: Normalize weights so the mean is 1
    normalized_weights = class_weights / np.mean(class_weights)

    alpha = 0.5  # Adjust this hyperparameter
    softened_weights = 1 / (proportions ** alpha)
    softened_weights /= np.mean(softened_weights)

    normalized_weights_v2 = class_weights / max(class_weights)

    return list(class_weights), list(normalized_weights), list(softened_weights), list(normalized_weights_v2)