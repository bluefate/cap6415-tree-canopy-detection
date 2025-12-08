import cv2
import numpy as np


def to_gray(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale.
    
    If image is already grayscale, returns unchanged.
    
    Args:
        image (np.ndarray): Input image (RGB or grayscale).
    
    Returns:
        np.ndarray: Grayscale image.
    """
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def equalize_hist(gray: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalization to improve contrast.
    
    Enhances contrast globally by stretching the histogram across the full range.
    
    Args:
        gray (np.ndarray): Grayscale image.
    
    Returns:
        np.ndarray: Histogram-equalized grayscale image.
    """
    return cv2.equalizeHist(gray)


def clahe_enhance(gray: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Improves local contrast while preventing noise amplification in homogeneous regions.
    
    Args:
        gray (np.ndarray): Grayscale image.
        clip (float): Clip limit for histogram. Defaults to 2.0.
        tile (int): Size of grid tiles. Defaults to 8.
    
    Returns:
        np.ndarray: CLAHE-enhanced grayscale image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return clahe.apply(gray)


def sharpen(gray: np.ndarray) -> np.ndarray:
    """
    Apply unsharp masking filter to enhance edges.
    
    Uses a standard sharpening kernel to enhance high-frequency details.
    
    Args:
        gray (np.ndarray): Grayscale image.
    
    Returns:
        np.ndarray: Sharpened grayscale image with values clipped to [0, 255].
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
    Normalize pixel values to [0, 1] range using min-max scaling.
    
    Args:
        gray (np.ndarray): Grayscale image.
    
    Returns:
        np.ndarray: Normalized image as float32 in range [0, 1].
    """
    g = gray.astype(np.float32)
    m = g.min()
    M = g.max()
    if M <= m:
        return g
    return (g - m) / (M - m)


def enhance_image_for_segmentation(image: np.ndarray) -> tuple:
    """
    Full image enhancement pipeline for segmentation preprocessing.
    
    Applies a sequence of enhancement techniques: grayscale conversion, histogram 
    equalization, CLAHE, sharpening, and normalization. Intermediate stages are 
    returned for analysis and visualization.
    
    Args:
        image (np.ndarray): Input RGB image.
    
    Returns:
        tuple: (normalized_image, stages_dict) where stages_dict contains:
            - 'gray': Grayscale image
            - 'equalized': After histogram equalization
            - 'clahe': After CLAHE enhancement
            - 'sharpened': After sharpening filter
            - 'normalized': Final normalized output [0, 1]
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
