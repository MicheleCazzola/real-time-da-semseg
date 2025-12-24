from PIL import Image
from pathlib import Path
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms as transforms

from src.utils.variables import RGB, grayscale

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
    edge = (cv2.dilate(edge, kernel, iterations=1)>50)*1.0

    return edge

class LoveDA(VisionDataset):
    def __init__(self, root, img, mask, directories=None, transforms=None, bd=False):
        super(LoveDA, self).__init__(root)

        root_path = Path(root)

        if not root_path.is_dir():
            raise RuntimeError("root should be a directory")

        self.root = root
        self.img_path = img
        self.mask_path = mask
        self.transforms = transforms

        self.image_names = []

        self.bd = bd

        if directories is None:
            raise RuntimeError("at least one directory must be passed")

        directories = [directories] if isinstance(directories, str) else directories

        for d in directories:
          image_names = load_images(root_path, d, img, mask)
          self.image_names.extend([(d, image_name) for image_name in image_names])

    def __getitem__(self, index):
        dir, image_name = self.image_names[index]
        image_path = f'{self.root}/{dir}/{self.img_path}/{image_name}'
        mask_path = f'{self.root}/{dir}/{self.mask_path}/{image_name}'

        image = pil_loader(image_path, RGB)
        mask = pil_loader(mask_path, grayscale)

        image = np.array(image)
        mask = np.array(mask)

        if self.transforms is not None:
          data = self.transforms(image=image, mask=mask)
          image = data['image']
          mask = data['mask']

        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask).squeeze(0)
        mask = transforms.ToPILImage()(mask)
        mask = transforms.PILToTensor()(mask).squeeze(0).long()

        mask = mask - 1

        if self.bd:
            bd = generate_bd(mask.numpy().astype(np.uint8))

            return image, mask, bd

        return image, mask

    def __len__(self):
        length = len(self.image_names)
        return length