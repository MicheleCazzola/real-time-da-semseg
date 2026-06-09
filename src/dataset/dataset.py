from collections import OrderedDict
import os

from PIL import Image
from pathlib import Path
import numpy as np
import cv2
import torch
from torchvision.datasets.vision import VisionDataset

class LoveDA(VisionDataset):
    label2color = OrderedDict(
        Background=(255, 255, 255),
        Building=(255, 0, 0),
        Road=(255, 255, 0),
        Water=(0, 0, 255),
        Barren=(159, 129, 183),
        Forest=(0, 255, 0),
        Agricultural=(255, 195, 128),
    )

    label2id = OrderedDict(
        Background=0,
        Building=1,
        Road=2,
        Water=3,
        Barren=4,
        Forest=5,
        Agricultural=6
    )
    
    id2label = OrderedDict({
        v: k for k, v in label2id.items()
    })
    
    class_frequency_train = {
        "urban": OrderedDict({
            "Background": 0.4845071829571744,
            "Building": 0.2120044261084477,
            "Road": 0.09279747200463612,
            "Water": 0.03731724537886264,
            "Barren": 0.0756555935761753,
            "Forest": 0.07916728358161661,
            "Agricultural": 0.01855079639308729
        }),
        "rural": OrderedDict({
            "Background": 0.2528211363277947,
            "Building": 0.026519406365771797,
            "Road": 0.019447644795621704,
            "Water": 0.0860276709433357,
            "Barren": 0.032679162738100336,
            "Forest": 0.22840263991146437,
            "Agricultural": 0.3541023389179114
        }),
        "all": OrderedDict({
            "Background": 0.3578595226769841,
            "Building": 0.11061185582123938,
            "Road": 0.05270190244779463,
            "Water": 0.06394406006512347,
            "Barren": 0.05216318006219974,
            "Forest": 0.1607445256521755,
            "Agricultural": 0.20197495327448317
        })
    }
    
    @staticmethod
    def get_class_weights(domain, loss_type):
        domain = domain.lower()
        if domain not in LoveDA.class_frequency_train:
            raise ValueError(f"Domain '{domain}' not valid")
        class_freq = LoveDA.class_frequency_train[domain]
        
        sorted_freqs = sorted([(LoveDA.label2id[k], v) for k, v in class_freq.items()], key=lambda x: x[0])
        freqs = np.array([f for _, f in sorted_freqs])
        
        if loss_type == "focal":
            # Focal: Normalized Inverse Frequency
            weights = 1.0 / freqs
        elif loss_type == "cross_entropy":
            # CE: Median Frequency Balancing
            median_freq = np.median(freqs)
            weights = median_freq / freqs
        elif loss_type == "ohem":
            # OHEM: Smoothed Log-Inverse to limit double penalty
            weights = 1.0 / np.log(1.02 + freqs)
        else:
            raise ValueError(f"Loss type '{loss_type}' not supported for class weights")
            
        # Normalization
        weights = weights / np.sum(weights)
        
        return torch.tensor(weights, dtype=torch.float32)

    def __init__(self, root, img, mask, directories=None, transforms=None, bd=False, fname=False, reduce_factor=1):
        super(LoveDA, self).__init__(root)

        root_path = Path(root)

        assert root_path.is_dir(), f"root should be a directory"
        assert directories is not None, f"at least one directory should be specified"
        assert 0 < reduce_factor <= 1, f"reduce_factor should be in the range (0, 1]"

        self.root = root
        self.img_path = img
        self.mask_path = mask
        self.transforms = transforms
        self.bd = bd
        self.fname = fname
        self.reduce_factor = reduce_factor

        self.image_names = []

        directories = [directories] if isinstance(directories, str) else directories

        for d in directories:
            d = d.title()
            image_names = self._load_images(root_path, d, img)
            self.image_names.extend([(d, image_name) for image_name in image_names])
        
        if self.reduce_factor < 1:
            self._reduce()

    def __getitem__(self, index):
        dir, image_name = self.image_names[index]
        image_path = os.path.join(self.root, dir, self.img_path, image_name)
        mask_path = os.path.join(self.root, dir, self.mask_path, image_name)

        image = self._pil_loader(image_path, "RGB")
        mask = self._pil_loader(mask_path, "L")

        image = np.array(image)
        mask = np.array(mask)
        
        assert image.shape[:2] == mask.shape, f"Image and mask shapes do not match for {image_name}: {image.shape} vs {mask.shape}"

        if self.transforms is not None:
            data = self.transforms(image=image, mask=mask)
            image = data['image']
            mask = data['mask']
          
        # Map classes in [1-7] to [0-6] and ignored from 0 to -1
        mask = mask.long() - 1
        
        result = (image, mask)

        if self.bd:
            bd = generate_bd(mask.numpy().astype(np.uint8))
            result += (bd,)
        if self.fname:
            result += (image_name,)

        return result

    def __len__(self):
        length = len(self.image_names)
        return length
    
    def _reduce(self):
        reduced_length = int(len(self.image_names) * self.reduce_factor)
        selected_indices = np.random.choice(self.__len__(), reduced_length, replace=False)
        self.image_names = [self.image_names[i] for i in selected_indices]
    
    def _pil_loader(self, path, codify):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert(codify)

    def _load_images(self, root_path, directory, img):
        directory_path = root_path / directory
        img_path = directory_path / img
        if not img_path.is_dir():
            raise RuntimeError(f"Image path {img_path} is not a directory")
        
        images = [item.name for item in img_path.iterdir()]

        return images

def generate_bd(mask, edge_pad=False, edge_size=2):

    y_k_size = 6
    x_k_size = 6

    edge = cv2.Canny(mask, 0.1, 0.2)
    kernel = np.ones((edge_size, edge_size), np.uint8)

    if edge_pad:
        edge = edge[y_k_size:-y_k_size, x_k_size:-x_k_size]
        edge = np.pad(edge, ((y_k_size,y_k_size),(x_k_size,x_k_size)), mode='constant')
    edge = (cv2.dilate(edge, kernel, iterations=1) > 50).astype(np.float32)

    return edge