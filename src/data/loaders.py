from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.annotations import AnnotationEntry
from src.data.masks import build_multi_mask


class ImageMaskDataset(Dataset):
    """
    Dataset that returns image and mask pairs.
    Expects a list of AnnotationEntry objects and a directory with images.
    """

    def __init__(
            self,
            entries: List[AnnotationEntry],
            image_dir: Path,
            classes: Optional[List[str]] = None,
            transform = None,
    ):
        self.entries = entries
        self.image_dir = Path(image_dir)
        self.classes = classes
        self.transform = transform

    def __len__( self ) -> int:
        return len(self.entries)

    def __getitem__( self, idx: int ) -> Tuple[torch.Tensor, torch.Tensor]:
        entry = self.entries[idx]
        img_path = self.image_dir / entry.image_path.name

        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Failed to read {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W = image.shape[:2]

        if self.classes is None:
            polys = [item.segmentation for item in entry.items]
        else:
            polys = [
                item.segmentation
                for item in entry.items
                if item.cls in self.classes
            ]

        mask = build_multi_mask(polys, W, H)

        if self.transform:
            augmented = self.transform(image = image, mask = mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # img_t = torch.tensor(image.transpose(2, 0, 1)).float() / 255.0
        # produces an error
        # augmented["image"] is sometimes a NumPy array and sometimes a PyTorch tensor, depending on your augmentation pipeline
        # Convert image
        if isinstance(image, torch.Tensor):
            # already CHW float Tensor from ToTensorV2
            img_t = image.float()
        else:
            # numpy HWC array
            img_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # mask_t = torch.tensor(mask).unsqueeze(0).float()
        # same
        # Convert image
        ## if isinstance(mask, torch.Tensor):
        ##     mask_t = mask.float()
        ## else:
        ##     mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        if isinstance(mask, torch.Tensor):
            mask_t = mask.float()
            if mask_t.ndim == 2:
                mask_t = mask_t.unsqueeze(0)
        else:
            if mask.ndim == 2:
                mask = np.expand_dims(mask, 0)
            mask_t = torch.from_numpy(mask).float()

        return img_t, mask_t


class ImageOnlyDataset(Dataset):
    """
    Dataset for inference. Returns image tensors only.
    """

    def __init__( self, image_dir: Path, transform = None ):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.files = sorted(
                [f for f in self.image_dir.glob("*.*") if f.suffix.lower() in [".tif"]],
        )

    def __len__( self ) -> int:
        return len(self.files)

    def __getitem__( self, idx: int ) -> Tuple[str, torch.Tensor]:
        path = self.files[idx]

        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Failed to read {path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            processed = self.transform(image = image)
            image = processed["image"]

        img_t = torch.tensor(image.transpose(2, 0, 1)).float() / 255.0
        return path.name, img_t
