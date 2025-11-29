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
        self.entries = entries
        self.image_dir = Path(image_dir)
        self.classes = classes
        self.transform = transform

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        entry = self.entries[idx]
        img_path = self.image_dir / entry.image_path.name

        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Failed to read {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W = image.shape[:2]

        # Build multi-class mask (values: 0=background, 1=individual, 2=group)
        mask = build_multiclass_mask(entry)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Convert image to tensor
        if isinstance(image, torch.Tensor):
            img_t = image.float()
            if img_t.ndim == 3 and img_t.shape[0] != 3:
                img_t = img_t.permute(2, 0, 1)
        else:
            img_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Convert mask to tensor [H, W] -> [1, H, W]
        if isinstance(mask, torch.Tensor):
            mask_t = mask.long()
        else:
            mask_t = torch.from_numpy(mask).long()

        # Ensure mask has shape [H, W]
        if mask_t.ndim == 3 and mask_t.shape[0] == 1:
            mask_t = mask_t.squeeze(0)

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

    def __init__(self, image_dir: Path, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.files = sorted(
            [f for f in self.image_dir.glob("*.*") if f.suffix.lower() in [".png"]],
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:

        path = self.files[idx]

        image = self._load_image(path)

        if self.transform:
            processed = self.transform(image=image)
            image = processed["image"]

        if isinstance(image, torch.Tensor):
            img_t = image.float()
            if img_t.ndim == 3 and img_t.shape[0] != 3:
                img_t = img_t.permute(2, 0, 1)
        else:
            img_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        return path.name, img_t

    def _load_image(
        self, source: Union[str, Path, np.ndarray, torch.Tensor, Image.Image]
    ) -> np.ndarray:
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
