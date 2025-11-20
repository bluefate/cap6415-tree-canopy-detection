# From C:\github\Tree-Canopy-Detection\src\data\annotations.py
import json
from pathlib import Path
from typing import List

import cv2
import numpy as np


class AnnotationItem:
    """
    Hold annotation data for one object.
    Stores class name, segmentation polygon, confidence score, and the computed bounding box.
    """

    def __init__( self, cls: str, segmentation: List[float], confidence: float = 1.0 ):
        self.cls = cls
        self.segmentation = segmentation
        self.confidence = confidence
        self.bbox = self.compute_bbox(segmentation)

    @staticmethod
    def compute_bbox( seg: List[float] ) -> List[int]:
        """
        Convert a flat segmentation list into a bounding box.
        Returns [x1, y1, x2, y2].
        """
        if seg is None or len(seg) < 4:
            return [0, 0, 0, 0]
        xs = seg[0::2]
        ys = seg[1::2]
        return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


class AnnotationEntry:
    """
    Hold all annotations for one image.
    Includes path, width, height, and a list of AnnotationItem objects.
    """

    def __init__( self, image_path: Path, width: int, height: int, items: List[AnnotationItem] ):
        self.image_path = image_path
        self.width = width
        self.height = height
        self.items = items

    def to_mask( self ) -> np.ndarray:
        """
        Build a binary mask from all polygons in this entry.
        """
        mask = np.zeros((self.height, self.width), dtype = np.uint8)
        for item in self.items:
            seg = item.segmentation
            if seg is None or len(seg) < 4:
                continue
            poly = np.array(seg, dtype = np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [poly], 1)
        return mask


def load_json_annotations( json_path: Path ) -> List[AnnotationEntry]:
    """
    Load annotation entries from a JSON file that contains images and annotations.
    Returns a list of AnnotationEntry objects.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Missing annotation file {json_path}")

    with open(json_path, "r", encoding = "utf8") as f:
        data = json.load(f)

    entries = []

    for img in data.get("images", []):
        fname = img.get("file_name")
        width = int(img.get("width", 0))
        height = int(img.get("height", 0))
        anns = img.get("annotations", [])

        items = []
        for ann in anns:
            cls = ann.get("class", "unknown")
            seg = ann.get("segmentation", [])
            conf = float(ann.get("confidence_score", 1.0))
            items.append(AnnotationItem(cls, seg, conf))

        entry = AnnotationEntry(Path(fname), width, height, items)
        entries.append(entry)

    return entries


def get_unique_classes( entries: List[AnnotationEntry] ) -> List[str]:
    """
    Return a sorted list of unique classes across all entries.
    """
    classes = set()
    for entry in entries:
        for item in entry.items:
            classes.add(item.cls)
    return sorted(classes)


# From C:\github\Tree-Canopy-Detection\src\data\augmentations.py
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_augmentations( image_size: int ):
    """
    Build augmentation pipeline for training.
    Includes flips, brightness changes, distortions, and resizing.
    """
    return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p = 0.5),
                A.VerticalFlip(p = 0.5),
                A.ShiftScaleRotate(
                        shift_limit = 0.1,
                        scale_limit = 0.1,
                        rotate_limit = 15,
                        p = 0.5,
                ),
                A.Affine(
                        scale = (0.9, 1.1),
                        rotate = (-15, 15),
                        shear = (-10, 10),
                        p = 0.5,
                ),
                A.RandomBrightnessContrast(p = 0.5),
                A.CLAHE(p = 0.5),
                A.ElasticTransform(alpha = 0.1, p = 0.1),
                A.GridDistortion(p = 0.1),
                A.OpticalDistortion(p = 0.1),
                ToTensorV2(),
            ],
    )


def get_val_augmentations( image_size: int ):
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


# From C:\github\Tree-Canopy-Detection\src\data\loaders.py
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.annotations import AnnotationEntry
from src.data.masks import build_multi_mask


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

        if self.classes is None:
            polys = [item.segmentation for item in entry.items]
        else:
            polys = [
                item.segmentation for item in entry.items if item.cls in self.classes
            ]

        mask = build_multi_mask(polys, W, H)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # img_t = torch.tensor(image.transpose(2,0,1)).float() / 255.0
        # produces an error
        # augmented["image"] is sometimes a NumPy array and sometimes a PyTorch tensor, depending on your augmentation pipeline
        # Convert image
        if isinstance(image, torch.Tensor):
            # already CHW float Tensor from ToTensorV2
            img_t = image.float()
            if img_t.ndim == 3 and img_t.shape[0] != 3:
                img_t = img_t.permute(2, 0, 1)
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

        ##---------------------------------------
        # if isinstance(mask, torch.Tensor):
        #     mask_t = mask.float()
        #     if mask_t.ndim == 2:
        #         mask_t = mask_t.unsqueeze(0)
        # else:
        #     if mask.ndim == 2:
        #         mask = np.expand_dims(mask, 0)
        #     mask_t = torch.from_numpy(mask).float()
        ##---------------------------------------
        # eliminates all shape variance
        if isinstance(mask, torch.Tensor):
            mask_t = mask.float()
        else:
            mask = mask.astype("float32")
            if mask.ndim == 2:
                mask = mask[None, ...]
            mask_t = torch.from_numpy(mask)

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
            [
                f
                for f in self.image_dir.glob("*.*")
                if f.suffix.lower() in [".tif", ".jpg", ".png"]
            ],
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


# From C:\github\Tree-Canopy-Detection\src\data\masks.py
from pathlib import Path
from typing import List

import cv2
import numpy as np


def build_binary_mask( segmentation: List[float], width: int, height: int ) -> np.ndarray:
    """
    Convert one segmentation polygon into a binary mask.
    segmentation is a flat list of coordinates.
    """
    mask = np.zeros((height, width), dtype = np.uint8)
    if segmentation is None or len(segmentation) < 4:
        return mask
    poly = np.array(segmentation, dtype = np.int32).reshape(-1, 2)
    cv2.fillPoly(mask, [poly], 1)
    return mask


def build_multi_mask( polygons: List[List[float]], width: int, height: int ) -> np.ndarray:
    """
    Convert multiple segmentation polygons into one mask.
    """
    mask = np.zeros((height, width), dtype = np.uint8)
    for seg in polygons:
        if seg is None or len(seg) < 4:
            continue
        poly = np.array(seg, dtype = np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [poly], 1)
    return mask


def save_mask( mask: np.ndarray, path: Path ) -> None:
    """
    Save a binary mask. Values are written as 0 or 255.
    """
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)
    out = (mask * 255).astype(np.uint8)
    cv2.imwrite(str(path), out)


def load_mask( path: Path ) -> np.ndarray:
    """
    Load a binary mask from disk. Converts 255 to 1.
    """
    path = Path(path)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Missing mask file {path}")
    return (img > 127).astype(np.uint8)


def mask_to_overlay( image: np.ndarray, mask: np.ndarray, alpha: float = 0.4 ) -> np.ndarray:
    """
    Overlay a binary mask on an RGB image. Mask is shown in red.
    """
    if image.max() <= 1.0:
        img_u8 = (image * 255).astype(np.uint8)
    else:
        img_u8 = image.astype(np.uint8)

    mask_u8 = (mask * 255).astype(np.uint8)
    mask_rgb = np.zeros_like(img_u8)
    mask_rgb[:, :, 0] = mask_u8

    overlay = cv2.addWeighted(img_u8, 1 - alpha, mask_rgb, alpha, 0)
    return overlay


# From C:\github\Tree-Canopy-Detection\src\data\__init__.py


