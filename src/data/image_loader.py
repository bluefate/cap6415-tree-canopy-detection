"""
Image loading utilities that handle multiple formats.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.data.annotations import AnnotationEntry
from src.utils.helpers import c, p


def load_image(image_dir: Path, entry: AnnotationEntry) -> np.ndarray:
    """
    Load image with automatic format handling and TIFF fallback.
    
    Tries OpenCV first (fastest), falls back to PIL for problematic TIFFs. 
    Automatically uses PNG version if available. Ensures RGB output.
    
    Args:
        image_dir (Path): Directory containing images.
        entry (AnnotationEntry): Annotation entry with image path.
    
    Returns:
        np.ndarray: RGB image array.
    
    Raises:
        FileNotFoundError: If image not found.
        ValueError: If image cannot be loaded by OpenCV or PIL.
    """
    image_path = Path(image_dir / entry.image_path.name)

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
    Validate all images in directory can be loaded successfully.
    
    Checks each image file using PIL.open().verify() and reports results.
    
    Args:
        image_dir (Path): Directory containing image files.
    
    Returns:
        dict: Results with keys: total, valid, invalid, problematic_files (list).
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

    p("Validating images", f"{len(image_files)} files", color1=c.BLACK)

    for img_path in image_files:
        try:
            # Quick validation using PIL
            pil_img = Image.open(img_path)
            pil_img.verify()
            results["valid"] += 1
        except Exception:
            results["invalid"] += 1
            results["problematic_files"].append(str(img_path))

    p("Valid images", results["valid"], color1=c.BLACK)
    p("Invalid images", results["invalid"], color1=c.BLACK)

    if results["problematic_files"]:
        p("Problematic files", "")
        for path in results["problematic_files"][:10]:
            p("", f"  {path}", color1=c.SALMON)

    return results


def apply_all_filters(img):
    """
    Apply all available filters to image and return normalized results.
    
    Applies both kernel-based filters (Sobel, Laplacian, Gaussian, etc.) and 
    algorithmic filters (CLAHE, etc.). All outputs normalized to uint8 with 
    original image dimensions.
    
    Args:
        img (np.ndarray): Input RGB image.
    
    Returns:
        dict: Mapping of filter names to normalized grayscale output arrays.
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



def create_enhanced_image(img, filter_names):
    """
    Create multi-channel enhanced image from specified filters.
    
    Applies named filters using apply_all_filters(), normalizes outputs, 
    and stacks into 3-channel array suitable for model input.
    
    Args:
        img (np.ndarray): Input RGB image.
        filter_names (list): Names of filters to apply (first 3 used).
    
    Returns:
        np.ndarray: 3-channel enhanced image with same height/width as input.
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
