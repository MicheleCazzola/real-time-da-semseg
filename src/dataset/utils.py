import os
from pathlib import Path
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
import wget
import zipfile
from google.colab import drive

from src.dataset.dataset import LoveDA
from src.utils.variables import categories, device, DRIVE_BASE_DIR, DATA_DIR, IMG_PATH, MASK_PATH, RURAL_PATH, TRAIN_DIR, URBAN_PATH, VAL_DIR, TEST_DIR, TRAIN_ZIP, VAL_ZIP, TEST_ZIP

def download_to_gdrive():
    data_path_dir = f'{DRIVE_BASE_DIR}/{DATA_DIR}'
    train_path_zip = f'{DRIVE_BASE_DIR}/{TRAIN_ZIP}'
    val_path_zip = f'{DRIVE_BASE_DIR}/{VAL_ZIP}'
    test_path_zip = f'{DRIVE_BASE_DIR}/{TEST_ZIP}'
    train_path_dir = f'{DRIVE_BASE_DIR}/{TRAIN_DIR}'
    val_path_dir = f'{DRIVE_BASE_DIR}/{VAL_DIR}'
    test_path_dir = f'{DRIVE_BASE_DIR}/{TEST_DIR}'

    drive.mount(DRIVE_BASE_DIR)

    download_directory = Path(data_path_dir)
    if not download_directory.exists():
        download_directory.mkdir(exist_ok=True)

    train_zip = Path(train_path_zip)
    if not train_zip.exists():
        wget.download('https://zenodo.org/record/5706578/files/Train.zip?download=1', train_path_zip)

    val_zip = Path(val_path_zip)
    if not val_zip.exists():
        wget.download('https://zenodo.org/record/5706578/files/Val.zip?download=1', val_path_zip)

    test_zip = Path(test_path_zip)
    if not test_zip.exists():
        wget.download('https://zenodo.org/record/5706578/files/Test.zip?download=1', test_path_zip)

def extract_from_gdrive():

    drive_path_dir = '/content/drive'
    mydrive_path_dir = f'{drive_path_dir}/MyDrive'
    data_path_dir = f'{mydrive_path_dir}/{DATA_DIR}'
    train_path_zip = f'{mydrive_path_dir}/{TRAIN_ZIP}'
    val_path_zip = f'{mydrive_path_dir}/{VAL_ZIP}'
    test_path_zip = f'{mydrive_path_dir}/{TEST_ZIP}'
    train_path_dir = f'{mydrive_path_dir}/{TRAIN_DIR}'
    val_path_dir = f'{mydrive_path_dir}/{VAL_DIR}'
    test_path_dir = f'{mydrive_path_dir}/{TEST_DIR}'

    drive.mount(drive_path_dir)

    train_dir = Path(TRAIN_DIR)
    if not train_dir.exists():
        
        with zipfile.ZipFile(train_path_zip, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        shutil.move(f'{DATA_DIR}/Train', TRAIN_DIR)

    val_dir = Path(VAL_DIR)
    if not val_dir.exists():
        with zipfile.ZipFile(val_path_zip, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        shutil.move(f'{DATA_DIR}/Val', VAL_DIR)

    #test_dir = Path(TEST_DIR)
    #if not test_dir.exists():
    #    !unzip -q {test_path_zip} -d {DATA_DIR}
    #    !mv {DATA_DIR}/Test {TEST_DIR}
    

def copy_to_gdrive():
    drive_path_dir = '/content/drive'
    mydrive_path_dir = f'{drive_path_dir}/MyDrive'
    data_path_dir = f'{mydrive_path_dir}/{DATA_DIR}'
    train_path_zip = f'{mydrive_path_dir}/{TRAIN_ZIP}'
    val_path_zip = f'{mydrive_path_dir}/{VAL_ZIP}'
    test_path_zip = f'{mydrive_path_dir}/{TEST_ZIP}'

    drive.mount(drive_path_dir)

    # Create the directory if it doesn't exist
    os.makedirs(data_path_dir, exist_ok=True)

    # Copy the zip files using shutil.copy
    if not os.path.exists(train_path_zip):
        shutil.copy(TRAIN_ZIP, train_path_zip)
        print(f"Copied {TRAIN_ZIP} to {train_path_zip}")

    if not os.path.exists(val_path_zip):
        shutil.copy(VAL_ZIP, val_path_zip)
        print(f"Copied {VAL_ZIP} to {val_path_zip}")

    if not os.path.exists(test_path_zip):
        shutil.copy(TEST_ZIP, test_path_zip)
        print(f"Copied {TEST_ZIP} to {test_path_zip}")
        

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