import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


# Histogram Equalization (HE): A traditional method that redistributes pixel intensities based on the global histogram of the image. It often makes the whole image brighter or darker uniformly.
# Adaptive Histogram Equalization (AHE): Instead of one global histogram, the image is divided into smaller regions (tiles), and each region gets its own histogram equalization. This enhances local contrast.
# CLAHE (Contrast Limited AHE): Improves on AHE by limiting contrast amplification. This prevents noise in uniform areas (like sky or skin) from being exaggerated


# def get_train_augmentations(image_size: int = 256, mode: str = "rgb", num_channels: int = 3):
def get_train_augmentations(image_size: int = 256, mode: str = "rgb"):
    """
    Build augmentation pipeline for training with geometric and color transforms.
    
    Includes flips, rotations, brightness/contrast adjustments, and resizing.
    Color transforms only applied for RGB mode.
    
    Args:
        image_size (int): Target image size (square). Defaults to 256.
        mode (str): Image mode ('rgb', 'filtered', or 'concat'). Color transforms only for rgb/filtered.
    
    Returns:
        A.Compose: Albumentations composition of augmentations.
    """
    # Base transforms that work with any number of channels
    base_transforms = [
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]
    # Color transforms only work with RGB (3 channels)
    if mode in ["rgb", "filtered"]:
        color_transforms = [
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.3),
        ]
        base_transforms.extend(color_transforms)

    final_transforms = [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]

    base_transforms.extend(final_transforms)
    return A.Compose(base_transforms)


def get_val_augmentations(image_size: int = 256, mode: str = "rgb"):
    """
    Build validation and inference augmentation pipeline.
    
    Minimal transforms with padding and normalization only.
    
    Args:
        image_size (int): Target image size (square). Defaults to 256.
        mode (str): Image mode. Defaults to "rgb".
    
    Returns:
        A.Compose: Albumentations composition of augmentations.
    """
    return A.Compose(
        [
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
            ),
            # A.CenterCrop(height=image_size, width=image_size),
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
    )
