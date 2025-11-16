import cv2
import numpy as np


def cv2_apply_gaussian( image: np.ndarray, ksize: int = 5, sigma: float = 1.0 ) -> np.ndarray:
    """
    Apply Gaussian blur to an RGB image.
    """
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def cv2_apply_sobel( image: np.ndarray ) -> np.ndarray:
    """
    Apply Sobel edge detection to a grayscale or RGB image.
    If RGB, converts to grayscale internally.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    mag = np.sqrt(dx * dx + dy * dy)
    mag = np.clip(mag, 0, 255).astype(np.uint8)
    return mag


def cv2_apply_laplacian( image: np.ndarray ) -> np.ndarray:
    """
    Apply Laplacian edge detection.
    Converts to grayscale if needed.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    out = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
    return out


def normalize_image( image: np.ndarray ) -> np.ndarray:
    """
    Normalize to range zero to one.
    """
    img = image.astype(np.float32)
    m = img.min()
    M = img.max()
    if M <= m:
        return img
    return (img - m) / (M - m)
