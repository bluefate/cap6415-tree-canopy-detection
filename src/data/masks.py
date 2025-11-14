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
