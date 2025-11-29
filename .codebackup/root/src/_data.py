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

    def __init__(self, cls: str, segmentation: List[float], confidence: float = 1.0):
        self.cls = cls
        self.segmentation = segmentation
        self.confidence = confidence
        self.bbox = self.compute_bbox(segmentation)

    @staticmethod
    def compute_bbox(seg: List[float]) -> List[int]:
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

    def __init__(
        self, image_path: Path, width: int, height: int, items: List[AnnotationItem]
    ):
        self.image_path = image_path
        self.width = width
        self.height = height
        self.items = items

    def to_mask(self) -> np.ndarray:
        """
        Build a binary mask from all polygons in this entry.
        """
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        for item in self.items:
            seg = item.segmentation
            if seg is None or len(seg) < 4:
                continue
            poly = np.array(seg, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [poly], 1)
        return mask


def load_json_annotations(json_path: Path) -> List[AnnotationEntry]:
    """
    Load annotation entries from a JSON file that contains images and annotations.
    Returns a list of AnnotationEntry objects.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Missing annotation file {json_path}")

    with open(json_path, "r", encoding="utf8") as f:
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


def get_unique_classes(entries: List[AnnotationEntry]) -> List[str]:
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


# From C:\github\Tree-Canopy-Detection\src\data\enhance_masks.py
import torch
import cv2

from exploration.enhancement import clahe_enhance
from src.data.loaders import ImageMaskDataset


class EnhancedImageMaskDataset(ImageMaskDataset):
    """
    Extended dataset that applies filter enhancements during loading.

    Can operate in 3 modes:
    1. 'rgb' - Original 3-channel RGB
    2. 'filtered' - Top 3 filters as RGB channels
    3. 'concat' - 6-channel (RGB + 3 filters)
    """

    def __init__(
        self,
        entries,
        image_dir,
        mode="rgb",
        filter_names=None,
        classes=None,
        transform=None,
    ):
        super().__init__(entries, image_dir, classes, transform)
        self.mode = mode
        self.filter_names = filter_names or ["laplacian", "sobel", "clahe"]

    def __getitem__(self, idx: int):
        """Override to ensure mask dtype is float for BCEWithLogitsLoss."""
        entry = self.entries[idx]
        img_path = self.image_dir / entry.image_path.name

        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Failed to read {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Build mask
        if self.classes is None:
            from src.data.masks import build_multiclass_mask

            mask = build_multiclass_mask(entry)
        else:
            from src.data.annotations import AnnotationEntry
            from src.data.masks import build_multiclass_mask

            filtered_items = [item for item in entry.items if item.cls in self.classes]
            filtered_entry = AnnotationEntry(
                entry.image_path, entry.width, entry.height, filtered_items
            )
            mask = build_multiclass_mask(filtered_entry)

        # Apply filters based on mode
        if self.mode == "filtered":
            image = self.apply_filters(image)
        elif self.mode == "concat":
            filtered_img = self.apply_filters(image)
            # Concatenate RGB + filtered
            image = np.concatenate([image, filtered_img], axis=2)

        # Apply transforms
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

        # Convert mask to float tensor - CRITICAL for BCEWithLogitsLoss
        if isinstance(mask, torch.Tensor):
            mask_t = mask.float()
        else:
            mask_t = torch.from_numpy(mask).float()

        # Ensure mask has shape [1, H, W]
        if mask_t.ndim == 2:
            mask_t = mask_t.unsqueeze(0)

        # Ensure binary values
        mask_t = (mask_t > 0.5).float()

        return img_t, mask_t

    def apply_filters(self, img):
        """
        Apply specified filters and return as 3-channel image.

        Handles both kernel-based and algorithmic filters with consistent
        naming through the centralized filter registry approach.
        """
        from src.exploration.kernels import get_kernels
        from src.exploration.enhancement import to_gray
        import cv2
        import numpy as np

        gray = to_gray(img)

        # loading kernel bank dynamically
        kernel_bank = get_kernels("all")

        # create callable filters dynamically from kernel bank
        dynamic_kernels = {
            name.lower(): lambda k=kernel: cv2.filter2D(gray, -1, k)
            for name, kernel in kernel_bank.items()
        }

        dynamic_kernels.update(
            {
                # Edge detection
                "laplacian": lambda: cv2.Laplacian(gray, cv2.CV_64F),
                "sobel": lambda: cv2.Sobel(gray, cv2.CV_64F, 1, 0)
                + cv2.Sobel(gray, cv2.CV_64F, 0, 1),
                "sobel_x_cv": lambda: cv2.Sobel(gray, cv2.CV_64F, 1, 0),
                "sobel_y_cv": lambda: cv2.Sobel(gray, cv2.CV_64F, 0, 1),
                "canny": lambda: cv2.Canny(gray, 50, 150),
                # Enhancement
                "clahe": lambda: clahe_enhance(gray),
                "histogram_eq": lambda: cv2.equalizeHist(gray),
                # Gaussian blur variants - THIS FIXES THE gaussian_3x3 issue
                "gaussian_3x3": lambda: cv2.GaussianBlur(gray, (3, 3), 1.0),
                "gaussian_5x5": lambda: cv2.GaussianBlur(gray, (5, 5), 1.5),
                "gaussian_7x7": lambda: cv2.GaussianBlur(gray, (7, 7), 2.0),
                "gaussian_9x9": lambda: cv2.GaussianBlur(gray, (9, 9), 2.5),
                # Other useful filters
                "bilateral": lambda: cv2.bilateralFilter(gray, 9, 75, 75),
                "median_3x3": lambda: cv2.medianBlur(gray, 3),
                "median_5x5": lambda: cv2.medianBlur(gray, 5),
            }
        )

        channels = []
        for fname in self.filter_names:
            key = fname.lower()
            if key not in dynamic_kernels:
                raise KeyError(
                    f"Unknown filter '{fname}'. Available: {list(dynamic_kernels.keys())}"
                )

            filtered = dynamic_kernels[key]()
            filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(
                np.uint8
            )
            channels.append(filtered)

        # Ensure at least 3 channels
        if len(channels) == 0:
            raise RuntimeError(f"No filters produced output for {self.filter_names}")

        while len(channels) < 3:
            channels.append(channels[-1])

        return np.stack(channels[:3], axis=2)

    @classmethod
    def get_available_filters(cls):
        """
        Get list of all available filter names.
        """
        from src.exploration.kernels import get_kernels

        # Get kernel-based filters
        kernel_bank = get_kernels("all")
        kernel_names = [name.lower() for name in kernel_bank.keys()]

        # Algorithmic filters (always available)
        algorithmic = [
            "laplacian",
            "sobel",
            "clahe",
            "gaussian_3x3",
            "gaussian_5x5",
            "gaussian_7x7",
            "gaussian_9x9",
            "sobel_x_cv",
            "sobel_y_cv",
            "canny",
            "histogram_eq",
            "bilateral",
            "median_3x3",
            "median_5x5",
        ]

        return sorted(set(kernel_names + algorithmic))


# From C:\github\Tree-Canopy-Detection\src\data\image_loader.py
"""
Image loading utilities that handle multiple formats.
"""

from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image

from utils.helpers import p


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load an image with automatic format handling and fallback for TIFFs.
    Tries OpenCV first (fastest), falls back to PIL for problematic TIFFs.
    Automatically uses PNG version if it exists alongside TIFF.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Check for PNG version first (more reliable than TIFF)
    png_path = image_path.with_suffix(".png")
    if png_path.exists() and png_path != image_path:
        image_path = png_path

    # Try OpenCV first (fastest)
    img = cv2.imread(str(image_path))

    if img is not None:
        # OpenCV loads as BGR, convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    # Fall back to PIL (better TIFF support)
    try:
        pil_img = Image.open(image_path)
        img = np.array(pil_img)

        # Ensure RGB format
        if img.ndim == 2:  # Grayscale
            img = np.stack([img, img, img], axis=2)
        elif img.shape[2] == 4:  # RGBA
            img = img[:, :, :3]

        return img

    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {str(e)}")


def validate_image_directory(image_dir: Path) -> dict:
    """
    Validate all images in a directory can be loaded.
    """
    from src.utils.helpers import p

    image_dir = Path(image_dir)

    # Find all image files
    image_files = []
    for ext in ["*.tif"]:
        # for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
        image_files.extend(image_dir.glob(ext))

    results = {
        "total": len(image_files),
        "valid": 0,
        "invalid": 0,
        "problematic_files": [],
    }

    p("Validating images", f"{len(image_files)} files")

    for img_path in image_files:
        try:
            # Quick validation using PIL
            pil_img = Image.open(img_path)
            pil_img.verify()
            results["valid"] += 1
        except Exception:
            results["invalid"] += 1
            results["problematic_files"].append(str(img_path))

    p("Valid images", results["valid"])
    p("Invalid images", results["invalid"])

    if results["problematic_files"]:
        p("Problematic files", "")
        for path in results["problematic_files"][:10]:
            p("", f"  {path}", color1=c.SALMON)

    return results


def apply_all_filters(img):
    """
    Apply all available filters to an image and return dict of results.
    Uses unified registry from kernels + algorithmic filters.
    """
    from src.exploration.kernels import get_kernels, apply_kernel_using_convolution
    from src.exploration.enhancement import to_gray, clahe_enhance

    gray = to_gray(img)
    target_h, target_w = img.shape[:2]

    kernel_bank = get_kernels("all")

    # Create unified filter registry
    filter_registry = {}

    # Add kernel-based filters
    for kname, kernel in kernel_bank.items():
        filter_registry[kname.lower()] = (
            lambda k=kernel: apply_kernel_using_convolution(gray, k)
        )

    # Add algorithmic filters
    filter_registry.update(
        {
            "laplacian": lambda: cv2.Laplacian(gray, cv2.CV_64F),
            "sobel": lambda: cv2.Sobel(gray, cv2.CV_64F, 1, 0)
            + cv2.Sobel(gray, cv2.CV_64F, 0, 1),
            "clahe": lambda: clahe_enhance(gray, clip=2.0, tile=8),
            "gaussian_3x3": lambda: cv2.GaussianBlur(gray, (3, 3), 1.0),
            "gaussian_5x5": lambda: cv2.GaussianBlur(gray, (5, 5), 1.5),
            "gaussian_7x7": lambda: cv2.GaussianBlur(gray, (7, 7), 2.0),
        }
    )

    # Apply all filters
    results = {}
    for fname, filter_func in filter_registry.items():
        try:
            filtered = filter_func()

            # Normalize
            if filtered.ndim == 3:
                filtered = cv2.cvtColor(filtered, cv2.COLOR_RGB2GRAY)
            if filtered.shape != (target_h, target_w):
                filtered = cv2.resize(
                    filtered, (target_w, target_h), interpolation=cv2.INTER_LINEAR
                )
            if filtered.dtype != np.uint8:
                filtered = cv2.normalize(
                    filtered, None, 0, 255, cv2.NORM_MINMAX
                ).astype(np.uint8)

            results[fname] = filtered
        except Exception as e:
            p("Warning", f"Filter '{fname}' failed: {e}", color1=c.ORANGE)
            continue

    return results


def apply_filters(self, img):
    """
    Apply specified filters and return as 3-channel image.
    Handles both kernel-based and algorithmic filters.
    """
    from src.exploration.kernels import get_kernels, apply_kernel_using_convolution
    from src.exploration.enhancement import to_gray, clahe_enhance
    import cv2
    import numpy as np

    # Convert to grayscale for filter application
    gray = to_gray(img)
    target_h, target_w = img.shape[:2]

    # Load kernel bank dynamically
    kernel_bank = get_kernels("all")

    # Create unified filter registry
    filter_registry = {}

    # Add kernel-based filters
    for kname, kernel in kernel_bank.items():
        # Use closure to capture kernel value
        filter_registry[kname.lower()] = (
            lambda k=kernel: apply_kernel_using_convolution(gray, k)
        )

    # Add algorithmic filters
    filter_registry.update(
        {
            "laplacian": lambda: cv2.Laplacian(gray, cv2.CV_64F),
            "sobel": lambda: cv2.Sobel(gray, cv2.CV_64F, 1, 0)
            + cv2.Sobel(gray, cv2.CV_64F, 0, 1),
            "clahe": lambda: clahe_enhance(gray, clip=2.0, tile=8),
            "gaussian_3x3": lambda: cv2.GaussianBlur(gray, (3, 3), 1.0),
            "gaussian_5x5": lambda: cv2.GaussianBlur(gray, (5, 5), 1.5),
            "gaussian_7x7": lambda: cv2.GaussianBlur(gray, (7, 7), 2.0),
        }
    )

    # Apply requested filters
    channels = []
    for fname in self.filter_names:
        key = fname.lower()

        if key not in filter_registry:
            available = sorted(filter_registry.keys())
            raise KeyError(
                f"Unknown filter '{fname}'. "
                f"Available filters ({len(available)}): {available[:10]}..."
            )

        try:
            # Apply filter
            filtered = filter_registry[key]()

            # Ensure 2D (grayscale)
            if filtered.ndim == 3:
                filtered = cv2.cvtColor(filtered, cv2.COLOR_RGB2GRAY)

            # Resize if needed
            if filtered.shape != (target_h, target_w):
                filtered = cv2.resize(
                    filtered, (target_w, target_h), interpolation=cv2.INTER_LINEAR
                )

            # Normalize to uint8
            if filtered.dtype != np.uint8:
                filtered = cv2.normalize(
                    filtered, None, 0, 255, cv2.NORM_MINMAX
                ).astype(np.uint8)

            channels.append(filtered)

        except Exception as e:
            raise RuntimeError(f"Filter '{fname}' failed: {e}")

    # Ensure we have at least 3 channels
    if len(channels) == 0:
        raise RuntimeError(f"No filters produced output for {self.filter_names}")

    while len(channels) < 3:
        channels.append(channels[-1].copy())

    # Stack into 3-channel image
    result = np.stack(channels[:3], axis=2)

    return result


def create_enhanced_image(img, filter_names):
    """
    Create multi-channel enhanced image using specified filters.
    Returns 3-channel image suitable for model input.
    """
    filters_dict = apply_all_filters(img)

    # Get target shape from original image
    target_h, target_w = img.shape[:2]

    channels = []
    for fname in filter_names[:3]:  # Take up to 3 filters
        filtered = filters_dict[fname]

        # Ensure it's 2D (grayscale)
        if filtered.ndim == 3:
            filtered = cv2.cvtColor(filtered, cv2.COLOR_RGB2GRAY)

        # Resize to match target dimensions if needed
        if filtered.shape != (target_h, target_w):
            filtered = cv2.resize(
                filtered, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )

        # Normalize to 0-255
        if filtered.dtype != np.uint8:
            filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(
                np.uint8
            )

        channels.append(filtered)

    # If we have fewer than 3 filters, pad with the last one
    while len(channels) < 3:
        channels.append(channels[-1].copy())

    # Verify all channels have same shape
    # shapes = [ch.shape for ch in channels[:3]]
    # if len(set(shapes)) != 1:
    #     p("Warning", f"Channel shape mismatch: {shapes}")
    #     # Force resize all to target
    #     channels = [cv2.resize(ch, (target_w, target_h)) if ch.shape != (target_h, target_w) else ch
    #                 for ch in channels[:3]]

    # Stack into 3-channel image
    enhanced = np.stack(channels[:3], axis=2)
    return enhanced


# From C:\github\Tree-Canopy-Detection\src\data\loaders.py
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

        # Build mask - handle class filtering
        if self.classes is None:
            mask = build_multiclass_mask(entry)
        else:
            # Create filtered entry with only specified classes
            from src.data.annotations import AnnotationEntry

            filtered_items = [item for item in entry.items if item.cls in self.classes]
            filtered_entry = AnnotationEntry(
                entry.image_path, entry.width, entry.height, filtered_items
            )
            mask = build_multiclass_mask(filtered_entry)

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
            mask_t = mask.float()
        else:
            mask_t = torch.from_numpy(mask).float()

        # Ensure mask has shape [1, H, W]
        if mask_t.ndim == 2:
            mask_t = mask_t.unsqueeze(0)

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


# From C:\github\Tree-Canopy-Detection\src\data\masks.py
from pathlib import Path
from typing import List

import cv2
import numpy as np


# Class mapping for Solafune competition
CLASS_TO_ID = {
    "individual_tree": 1,
    "group_of_trees": 2,
}

ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}


def build_binary_mask(segmentation: List[float], width: int, height: int) -> np.ndarray:
    """
    Convert one segmentation polygon into a binary mask.
    segmentation is a flat list of coordinates.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    if segmentation is None or len(segmentation) < 4:
        return mask
    poly = np.array(segmentation, dtype=np.int32).reshape(-1, 2)
    cv2.fillPoly(mask, [poly], 1)
    return mask


def build_multiclass_mask(entry, class_to_id: dict = None) -> np.ndarray:
    """
    Build mask with class indices for multi-class segmentation.
    """
    if class_to_id is None:
        class_to_id = CLASS_TO_ID

    mask = np.zeros((entry.height, entry.width), dtype=np.uint8)

    for item in entry.items:
        seg = item.segmentation
        if seg is None or len(seg) < 6:
            continue

        # Get class ID (0 if unknown class)
        class_id = class_to_id.get(item.cls, 0)

        poly = np.array(seg, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [poly], class_id)

    return mask


def save_mask(mask: np.ndarray, path: Path) -> None:
    """
    Save a binary mask. Values are written as 0 or 255.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = (mask * 255).astype(np.uint8)
    cv2.imwrite(str(path), out)


def load_mask(path: Path) -> np.ndarray:
    """
    Load a binary mask from disk. Converts 255 to 1.
    """
    path = Path(path)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Missing mask file {path}")
    return (img > 127).astype(np.uint8)


def mask_to_overlay(
    image: np.ndarray, mask: np.ndarray, alpha: float = 0.4
) -> np.ndarray:
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


