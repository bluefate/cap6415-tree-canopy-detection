"""
Image loading utilities that handle multiple formats.
"""
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image

from src.exploration.enhancement import clahe_enhance, to_gray
from src.exploration.filters import cv2_apply_gaussian, cv2_apply_laplacian, cv2_apply_sobel
from src.exploration.kernels import apply_kernel_using_convolution, get_kernels


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
            p("", f"  {path}", color1 = c.SALMON)

    return results

def apply_all_filters( img ):
    """Apply comprehensive set of filters to one image."""
    gray = to_gray(img)

    filters = { }

    # Edge detection filters
    filters['sobel'] = cv2_apply_sobel(img)
    filters['laplacian'] = cv2_apply_laplacian(img)

    kernels = get_kernels()
    filters['scharr_x'] = apply_kernel_using_convolution(gray, kernels['Scharr_X'])
    filters['scharr_y'] = apply_kernel_using_convolution(gray, kernels['Scharr_Y'])
    filters['scharr_combined'] = np.sqrt(
            filters['scharr_x'].astype(float) ** 2 +
            filters['scharr_y'].astype(float) ** 2
    ).astype(np.uint8)

    # Contrast enhancement
    filters['clahe'] = clahe_enhance(gray, clip = 2.0, tile = 8)
    filters['hist_eq'] = cv2.equalizeHist(gray)

    # Smoothing (often used before edge detection)
    filters['gaussian_3x3'] = cv2_apply_gaussian(img, ksize = 3, sigma = 1.0)
    filters['gaussian_5x5'] = cv2_apply_gaussian(img, ksize = 5, sigma = 1.5)

    # Custom kernels
    filters['high_pass'] = apply_kernel_using_convolution(gray, kernels['High_Pass_3x3'])
    filters['sharpen'] = apply_kernel_using_convolution(gray, kernels['Sharpen_Basic'])

    # Gradient magnitude (Sobel components combined)
    filters['gradient_mag'] = cv2_apply_sobel(img)

    return filters


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
            filtered = cv2.resize(filtered, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Normalize to 0-255
        if filtered.dtype != np.uint8:
            filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

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