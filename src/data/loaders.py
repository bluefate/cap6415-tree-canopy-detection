from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.annotations import AnnotationEntry
from src.data.masks import build_multiclass_mask


class ImageMaskDataset(Dataset):
    """
    Dataset that returns image and mask pairs.
    Expects a list of AnnotationEntry objects and a directory with images.

    ImageMaskDataset:

    • loads image
    • generates mask polygons
    • applies transforms
    • handles numpy vs tensor images
    • handles numpy vs tensor masks
    • ensures output shapes:

    image: (3, H, W)

    mask: (1, H, W)

    Neded for segmentation training.
    """

    def __init__(
        self,
        entries: List[AnnotationEntry],
        image_dir: Path,
        classes: Optional[List[str]] = None,
        transform=None,
    ):
        """
        Initialize image-mask dataset.
        
        Args:
            entries (List[AnnotationEntry]): List of annotation entries.
            image_dir (Path): Directory containing image files.
            classes (List[str], optional): List of class names for multi-class segmentation.
            transform: Albumentations composition for augmentation.
        """
        self.entries = entries
        self.image_dir = Path(image_dir)
        self.classes = classes
        self.transform = transform

    def __len__(self) -> int:
        """
        Get dataset size.
        
        Returns:
            int: Number of samples in dataset.
        """
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get image-mask pair by index.
        
        Args:
            idx (int): Index of sample.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Image tensor [C, H, W] and mask tensor [H, W].
        """
        entry = self.entries[idx]
        img_path = self.image_dir / entry.image_path.name

        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Failed to read {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = build_multiclass_mask(entry)

        if self.transform:
            processed = self.transform(image=image, mask=mask)
            image = processed["image"]
            mask = processed["mask"]

            # Ensure mask values are proper class indices (0,1,2)
            if mask.max() > 2:
                mask = (mask / 255).astype(np.uint8)

        # Safety check: Convert image to tensor if not already done
        if not isinstance(image, torch.Tensor):
            img_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        else:
            img_t = image.float()
            if img_t.ndim == 3 and img_t.shape[0] != 3:
                img_t = img_t.permute(2, 0, 1)
            # Only normalize if not already normalized by Albumentations
            min_val = img_t.min().item()
            max_val = img_t.max().item()
            is_already_normalized = (min_val >= -5.0 and max_val <= 5.0) or (
                min_val < 0 and max_val <= 10.0
            )
            if not is_already_normalized and img_t.max() > 1.0:
                img_t = img_t / 255.0

        # Convert mask to tensor
        if isinstance(mask, torch.Tensor):
            mask_t = mask.long()
        else:
            mask_t = torch.from_numpy(mask).long()

        while mask_t.ndim > 2:
            mask_t = mask_t.squeeze(0)

        assert mask_t.ndim == 2, f"Mask should be 2D [H, W], got {mask_t.shape}"
        return img_t, mask_t


class ImageOnlyDataset(Dataset):
    """
    Dataset for inference. Returns image tensors only.

    ImageOnlyDataset:

    • load image
    • apply transforms
    • safely convert to CHW tensor
    • return image name + tensor

    Supports:
    • PIL Image
    • NumPy array
    • Torch tensor
    • File paths

    Used only for inference
    """

    def __init__(
        self,
        image_dir: Path,
        transform=None,
    ):
        """
        Initialize image-only dataset for inference.
        
        Args:
            image_dir (Path): Directory containing PNG images.
            transform: Albumentations composition for preprocessing.
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.files = sorted(
            [f for f in self.image_dir.glob("*.*") if f.suffix.lower() in [".png"]],
        )

    def __len__(self) -> int:
        """Return number of images in dataset."""
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        """
        Get image at specified index.
        
        Args:
            idx (int): Index of image to retrieve.
        
        Returns:
            tuple: (filename, image_tensor) where image_tensor has shape (C, H, W).
        """
        path = self.files[idx]
        image = self._load_image(path)

        if self.transform:
            processed = self.transform(image=image)
            image = processed["image"]

        # Safety check: Convert image to tensor if not already done
        if not isinstance(image, torch.Tensor):
            img_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        else:
            img_t = image.float()
            if img_t.ndim == 3 and img_t.shape[0] != 3:
                img_t = img_t.permute(2, 0, 1)
            # Only normalize if not already normalized by Albumentations
            min_val = img_t.min().item()
            max_val = img_t.max().item()
            is_already_normalized = (min_val >= -5.0 and max_val <= 5.0) or (
                min_val < 0 and max_val <= 10.0
            )
            if not is_already_normalized and img_t.max() > 1.0:
                img_t = img_t / 255.0

        return path.name, img_t

    def _load_image(
        self, source: Union[str, Path, np.ndarray, torch.Tensor, Image.Image]
    ) -> np.ndarray:
        """
        Load image from various source types with format handling.
        
        Args:
            source: Image source (path, array, tensor, or PIL Image).
        
        Returns:
            np.ndarray: RGB image array.
        
        Raises:
            ValueError: If image cannot be loaded from source.
        """
        if isinstance(source, np.ndarray):
            img = source
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=2)
            return img

        if isinstance(source, torch.Tensor):
            arr = source.cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] == 3:
                arr = arr.transpose(1, 2, 0)
            return arr

        if isinstance(source, Image.Image):
            return np.array(source)

        path = str(source)
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Failed to read {source}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
