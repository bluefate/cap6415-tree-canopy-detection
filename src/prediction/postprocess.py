import cv2
import numpy as np


def refine_mask(mask: np.ndarray, min_area: int = 20) -> np.ndarray:
    """
    Refine binary mask by removing small connected components (noise).

    Args:
        mask (np.ndarray): Binary mask to refine.
        min_area (int): Minimum area threshold for connected components. Defaults to 20.

    Returns:
        np.ndarray: Refined mask with small components removed.
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
    Convert probabilistic mask to binary using threshold.

    Args:
        mask (np.ndarray): Probabilistic mask (0-1 values).
        threshold (float): Threshold value for binarization. Defaults to 0.5.

    Returns:
        np.ndarray: Binary mask (0 or 1).
    """
    return (mask > threshold).astype(np.uint8)


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Overlay binary mask on RGB image with semi-transparency.

    Args:
        image (np.ndarray): RGB image (0-255 or 0-1 range).
        mask (np.ndarray): Binary mask (0-1 values).
        alpha (float): Blending factor. Defaults to 0.4.

    Returns:
        np.ndarray: Blended image with mask overlay.
    """
    if image.max() <= 1.0:
        img_u8 = (image * 255).astype(np.uint8)
    else:
        img_u8 = image.astype(np.uint8)

    mask_u8 = (mask * 255).astype(np.uint8)

    mask_rgb = np.zeros_like(img_u8)
    mask_rgb[:, :, 0] = mask_u8

    return cv2.addWeighted(img_u8, 1 - alpha, mask_rgb, alpha, 0)
