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
    Convert polygon segmentation into a binary mask.
    
    Args:
        segmentation (List[float]): Flat list of polygon coordinates [x1, y1, x2, y2, ...].
        width (int): Mask width.
        height (int): Mask height.
    
    Returns:
        np.ndarray: Binary mask (0 = background, 1 = object).
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    if segmentation is None or len(segmentation) < 4:
        return mask
    poly = np.array(segmentation, dtype=np.int32).reshape(-1, 2)
    cv2.fillPoly(mask, [poly], 1)
    return mask


def build_multiclass_mask(entry, class_to_id: dict = None) -> np.ndarray:
    """
    Build multi-class segmentation mask with class indices.
    
    Background=0, individual_tree=1, group_of_trees=2, unknown classes are skipped.
    
    Args:
        entry (AnnotationEntry): Annotation entry with image dimensions and items.
        class_to_id (dict, optional): Mapping from class name to ID. Defaults to standard mapping.
    
    Returns:
        np.ndarray: Multi-class mask with dtype=uint8.
    """
    if class_to_id is None:
        class_to_id = CLASS_TO_ID

    mask = np.zeros((entry.height, entry.width), dtype=np.uint8)

    for item in entry.items:
        seg = item.segmentation
        if seg is None or len(seg) < 6:
            continue

        # Get class ID, skip unknown classes (don't assign to background)
        class_id = class_to_id.get(item.cls, None)
        if class_id is None:
            # Skip unknown classes instead of assigning to background
            continue

        poly = np.array(seg, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [poly], class_id)

    return mask


def save_mask(mask: np.ndarray, path: Path) -> None:
    """
    Save binary mask to disk with values converted to 0 or 255.
    
    Args:
        mask (np.ndarray): Binary mask (values 0-1).
        path (Path): Output file path.
    
    Returns:
        None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = (mask * 255).astype(np.uint8)
    cv2.imwrite(str(path), out)


def load_mask(path: Path) -> np.ndarray:
    """
    Load binary mask from disk and convert 255 values to 1.
    
    Args:
        path (Path): Path to mask file.
    
    Returns:
        np.ndarray: Binary mask (0-1 values).
    
    Raises:
        FileNotFoundError: If mask file does not exist.
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
    Overlay a binary mask on an RGB image with semi-transparency.
    
    Args:
        image (np.ndarray): RGB image (0-255 or 0-1 range).
        mask (np.ndarray): Binary mask (0-1 values).
        alpha (float): Blending factor (0-1). Defaults to 0.4.
    
    Returns:
        np.ndarray: Overlaid image with mask shown in red.
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
