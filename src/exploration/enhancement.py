import cv2
import numpy as np


def to_gray(image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to grayscale.
    """
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def equalize_hist(gray: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization to a grayscale image.
    """
    return cv2.equalizeHist(gray)


def clahe_enhance(gray: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    """
    Apply CLAHE to improve local contrast.
    """
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return clahe.apply(gray)


def sharpen(gray: np.ndarray) -> np.ndarray:
    """
    Apply a basic sharpening filter to enhance edges.
    """
    kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )
    out = cv2.filter2D(gray, -1, kernel)
    return np.clip(out, 0, 255).astype(np.uint8)


def normalize(gray: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values to zero to one.
    """
    g = gray.astype(np.float32)
    m = g.min()
    M = g.max()
    if M <= m:
        return g
    return (g - m) / (M - m)


def enhance_image_for_segmentation(image: np.ndarray) -> tuple:
    """
    Full enhancement pipeline used in notebooks.
    Returns enhanced image and intermediate stages.

    Steps:
    1. convert to grayscale
    2. equalize histogram
    3. apply CLAHE
    4. sharpen
    5. normalize to zero to one
    """

    stages = {}

    gray = to_gray(image)
    stages["gray"] = gray

    eq = equalize_hist(gray)
    stages["equalized"] = eq

    clahe = clahe_enhance(eq)
    stages["clahe"] = clahe

    sharp = sharpen(clahe)
    stages["sharpened"] = sharp

    norm = normalize(sharp)
    stages["normalized"] = norm

    return norm, stages
