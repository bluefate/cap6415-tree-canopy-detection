import cv2
import numpy as np


def cv2_apply_gaussian(
    image: np.ndarray, ksize: int = 5, sigma: float = 1.0
) -> np.ndarray:
    """
    Apply Gaussian blur filter to image.
    
    Args:
        image (np.ndarray): Input image (RGB or grayscale).
        ksize (int): Kernel size. Defaults to 5.
        sigma (float): Gaussian standard deviation. Defaults to 1.0.
    
    Returns:
        np.ndarray: Blurred image.
    """
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def cv2_apply_sobel(image: np.ndarray) -> np.ndarray:
    """
    Apply Sobel edge detection filter.
    
    Computes directional derivatives to highlight edges. Converts RGB to grayscale 
    internally if needed.
    
    Args:
        image (np.ndarray): Input image (RGB or grayscale).
    
    Returns:
        np.ndarray: Edge-detected image with magnitude values clipped to [0, 255].
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


def cv2_apply_laplacian(image: np.ndarray) -> np.ndarray:
    """
    Apply Laplacian edge detection filter.
    
    Detects edges as zero-crossings of the Laplacian operator. Converts RGB to 
    grayscale internally if needed.
    
    Args:
        image (np.ndarray): Input image (RGB or grayscale).
    
    Returns:
        np.ndarray: Edge-detected image with absolute values clipped to [0, 255].
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    out = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
    return out


