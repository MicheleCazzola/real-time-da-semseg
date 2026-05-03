from collections import OrderedDict
import logging
import os

from PIL import Image
from pathlib import Path
import numpy as np
import cv2
from torchvision.datasets.vision import VisionDataset

def pil_loader(path, codify):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert(codify)

def load_images(root_path, directory, img, mask):
    directory_path = root_path / directory
    img_path = directory_path / img
    mask_path = directory_path / mask
    if not img_path.is_dir() or not mask_path.is_dir():
        raise RuntimeError("folder structure different from expected")

    images = [item.name for item in img_path.iterdir()]
    masks = [item.name for item in mask_path.iterdir()]

    if set(images) != set(masks):
        raise RuntimeError("images and masks do not match")

    return images

def generate_bd(mask, edge_pad=False, is_flip=False, edge_size=2):

    y_k_size = 6
    x_k_size = 6

    edge = cv2.Canny(mask, 0, 8)
    kernel = np.ones((edge_size, edge_size), np.uint8)

    if edge_pad:
        edge = edge[y_k_size:-y_k_size, x_k_size:-x_k_size]
        edge = np.pad(edge, ((y_k_size,y_k_size),(x_k_size,x_k_size)), mode='constant')
    edge = (cv2.dilate(edge, kernel, iterations=1) > 50).astype(np.float32)

    return edge

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
    
    id2label = {v: k for k, v in label2id.items()}
    
    def __init__(self, root, img, mask, directories=None, transforms=None, bd=False, reduce_factor=1):
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
        self.reduce_factor = reduce_factor

        self.image_names = []

        directories = [directories] if isinstance(directories, str) else directories

        for d in directories:
          image_names = load_images(root_path, d, img, mask)
          self.image_names.extend([(d, image_name) for image_name in image_names])
        
        if self.reduce_factor < 1:
            self._reduce()

    def __getitem__(self, index):
        dir, image_name = self.image_names[index]
        image_path = os.path.join(self.root, dir, self.img_path, image_name)
        mask_path = os.path.join(self.root, dir, self.mask_path, image_name)

        image = pil_loader(image_path, "RGB")
        mask = pil_loader(mask_path, "L")

        image = np.array(image)
        mask = np.array(mask)

        if self.transforms is not None:
            data = self.transforms(image=image, mask=mask)
            image = data['image']
            mask = data['mask']
          
        # Map classes in [1-7] to [0-6] and ignored from 0 to -1
        mask = mask.long() - 1

        if self.bd:
            bd = generate_bd(mask.numpy().astype(np.uint8))

            return image, mask, bd

        return image, mask

    def __len__(self):
        length = len(self.image_names)
        return length
    
    def _reduce(self):
        reduced_length = int(len(self.image_names) * self.reduce_factor)
        selected_indices = np.random.choice(self.__len__(), reduced_length, replace=False)
        self.image_names = [self.image_names[i] for i in selected_indices]