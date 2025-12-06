import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


# Histogram Equalization (HE): A traditional method that redistributes pixel intensities based on the global histogram of the image. It often makes the whole image brighter or darker uniformly.
# Adaptive Histogram Equalization (AHE): Instead of one global histogram, the image is divided into smaller regions (tiles), and each region gets its own histogram equalization. This enhances local contrast.
# CLAHE (Contrast Limited AHE): Improves on AHE by limiting contrast amplification. This prevents noise in uniform areas (like sky or skin) from being exaggerated


def get_train_augmentations(image_size: int = 256):
    """
    Build augmentation pipeline for training.
    Includes flips, brightness changes, distortions, and resizing.
    """
    return A.Compose(
        [
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
            ),
            A.Resize(image_size, image_size),
            # A.RandomCrop(height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            # A.ShiftScaleRotate(
            #         shift_limit = 0.1,
            #         scale_limit = 0.1,
            #         rotate_limit = 15,
            #         p = 0.5,
            # ),
            # A.Affine(
            #     scale=(0.9, 1.1),
            #     rotate=(-15, 15),
            #     shear=(-10, 10),
            #     p=0.5,
            # ),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.3),
            # A.CLAHE(p=0.5), #alrady used as a filter
            # A.ElasticTransform(alpha=0.1, p=0.1),
            # A.GridDistortion(p=0.1),
            # A.OpticalDistortion(p=0.1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
    )


def get_val_augmentations(image_size: int = 256):
    """
    Build validation and inference augmentation pipeline.
    """
    return A.Compose(
        [
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
            ),
            # A.CenterCrop(height=image_size, width=image_size),
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
    )
