import albumentations as A

def get_nop_augmentation():
    return A.Compose([], p=1)

def get_augmentations(aug_prob, aug_prob_one_of, aug_params):
    
    alone_augs = ["RC"]
    one_of_augs, prob_one_of = ["HF", "VF", "RR"], aug_prob_one_of

    augmentations = {
        "AFF": A.Affine(p=1, **aug_params.get("AFF", {})),
        "GD": A.GridDistortion(p=1, **aug_params.get("GD", {})),
        "RC": A.RandomCrop(p=1, **aug_params.get("RC", {"height": 512, "width": 512})),
        "HF": A.HorizontalFlip(p=1, **aug_params.get("HF", {})),
        "VF": A.VerticalFlip(p=1, **aug_params.get("VF", {})),
        "RR": A.RandomRotate90(p=1, **aug_params.get("RR", {})),
        "GB": A.GaussianBlur(p=1, **aug_params.get("GB", {})),
        "GDO": A.GridDropout(p=1, **aug_params.get("GDO", {})),
        "CJ": A.ColorJitter(p=1, **aug_params.get("CJ", {})),
        "GN": A.GaussNoise(p=1, **aug_params.get("GN", {})),
        "CD": A.ChannelDropout(p=1, **aug_params.get("CD", {})),
        "RSC": A.RandomSizedCrop(p=1, **aug_params.get("RSC", {"min_max_height": (128, 1024), "size": (512, 512)})),
    }

    return A.Compose(
        [augmentations[aug] for aug in alone_augs if aug in aug_params] + 
        [A.OneOf([augmentations[aug] for aug in one_of_augs if aug in aug_params], p=prob_one_of)],
        p=aug_prob
    )