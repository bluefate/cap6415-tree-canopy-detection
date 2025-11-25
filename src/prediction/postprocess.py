import cv2
import numpy as np


def refine_mask(mask: np.ndarray, min_area: int = 20) -> np.ndarray:
    """
    Clean small artifacts in a binary mask.
    Removes connected components smaller than min_area.
    """
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 1
    return cleaned


def group_mask_threshold(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Apply a direct threshold to a probability mask.
    """
    return (mask > threshold).astype(np.uint8)


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Create a red overlay of the mask on top of an RGB image.
    """
    if image.max() <= 1.0:
        img_u8 = (image * 255).astype(np.uint8)
    else:
        img_u8 = image.astype(np.uint8)

    mask_u8 = (mask * 255).astype(np.uint8)

    mask_rgb = np.zeros_like(img_u8)
    mask_rgb[:, :, 0] = mask_u8

    return cv2.addWeighted(img_u8, 1 - alpha, mask_rgb, alpha, 0)
