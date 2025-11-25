import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_augmentations(image_size: int):
    """
    Build augmentation pipeline for training.
    Includes flips, brightness changes, distortions, and resizing.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # A.ShiftScaleRotate(
            #         shift_limit = 0.1,
            #         scale_limit = 0.1,
            #         rotate_limit = 15,
            #         p = 0.5,
            # ),
            A.Affine(
                scale=(0.9, 1.1),
                rotate=(-15, 15),
                shear=(-10, 10),
                p=0.5,
            ),
            A.RandomBrightnessContrast(p=0.5),
            A.CLAHE(p=0.5),
            A.ElasticTransform(alpha=0.1, p=0.1),
            A.GridDistortion(p=0.1),
            A.OpticalDistortion(p=0.1),
            ToTensorV2(),
        ],
    )


def get_val_augmentations(image_size: int):
    """
    Build validation and inference augmentation pipeline.
    Only resize and tensor conversion.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            ToTensorV2(),
        ],
    )
