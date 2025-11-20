"""
Robust image loading utilities that handle multiple formats.
"""
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image


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
    png_path = image_path.with_suffix('.png')
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
    for ext in ['*.tif']:
    # for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
        image_files.extend(image_dir.glob(ext))

    results = {
        "total": len(image_files),
        "valid": 0,
        "invalid": 0,
        "problematic_files": []
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
            print(f"  {path}")

    return results