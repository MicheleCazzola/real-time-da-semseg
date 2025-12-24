import albumentations as A



def get_augmentations(resize, aug_prob, aug_indices):

    augmentations = [
        A.ShiftScaleRotate(p=1),
        A.GridDistortion(p=1),
        A.RandomCrop(height=resize, width=resize, p=1),
        A.HorizontalFlip(p=1),
        A.GaussianBlur(p=1),
        A.GridDropout(p=1),
        A.ColorJitter(p=1),
        A.GaussNoise(var_limit=(0.2, 0.3), p=1),
        A.ChannelDropout(p=1),
        A.RandomSizedCrop(min_max_height=(resize//8, resize), height=resize, width=resize, p=1),
    ]

    return A.Compose([augmentations[i] for i in aug_indices], p=aug_prob)