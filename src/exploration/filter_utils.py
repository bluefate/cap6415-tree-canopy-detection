"""
Filter utilities with automatic shape normalization.

This module provides helpers for applying filters and creating multi-channel
enhanced images without shape mismatch errors.
"""

import cv2
import numpy as np

from src.utils.helpers import c, p


def get_input_channels(mode, filters, model_name):
    """
    Get correct input channel count for model and mode combination.
    
    Determines appropriate number of input channels based on processing mode 
    and model architecture.
    
    Args:
        mode (str): Processing mode ('rgb', 'filtered', or 'concat').
        filters (list): List of filters to apply (used in 'concat' mode).
        model_name (str): Name of model ('simple_cnn', 'unet', etc.).
    
    Returns:
        int: Number of input channels (typically 3 or 6).
    """
    if mode == "rgb":
        return 3
    elif mode == "filtered":
        return 3  # Filters replace RGB channels
    elif mode == "concat":
        if model_name in ["simple_cnn", "unet"]:
            # These models don't support 6 channels - use RGB instead
            p(
                "WARNING",
                f"{model_name} doesn't support concat mode, using RGB",
                color1=c.ORANGE,
            )
            return 3
        else:
            num_filters = len(filters) if filters else 0
            return 3 + num_filters  # RGB + actual filter count
    else:
        return 3  # Default fallback


def normalize_filter_output(filtered, target_shape, dtype=np.uint8):
    """
    Normalize filter output to consistent shape and dtype.
    
    Handles grayscale conversion, resizing, and dtype normalization to ensure
    consistent output across different filter implementations.

    Args:
        filtered (np.ndarray): Filter output (any shape, any dtype).
        target_shape (tuple): Target (height, width) shape.
        dtype (type): Output dtype. Defaults to np.uint8.

    Returns:
        np.ndarray: Normalized 2D array with target shape and dtype.
    """
    target_h, target_w = target_shape

    # Convert to grayscale if needed
    if filtered.ndim == 3:
        filtered = cv2.cvtColor(filtered, cv2.COLOR_RGB2GRAY)

    # Resize if shape doesn't match
    if filtered.shape != (target_h, target_w):
        filtered = cv2.resize(
            filtered, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )

    # Normalize dtype
    if filtered.dtype != dtype:
        if dtype == np.uint8:
            # Normalize to 0-255 range
            filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(
                np.uint8
            )
        else:
            filtered = filtered.astype(dtype)

    return filtered


def apply_filters_safe(img, filter_functions, filter_names=None):
    """
    Apply multiple filters safely with error handling and normalization.

    Args:
        img (np.ndarray): Input RGB image.
        filter_functions (dict or list): Filter functions to apply. Either dict of 
                                        {name: callable} or list of callables.
        filter_names (list, optional): Names for filters if using list format.

    Returns:
        dict: Dictionary mapping filter names to normalized output arrays.
    """
    target_shape = img.shape[:2]

    if isinstance(filter_functions, dict):
        filters_dict = filter_functions
    elif isinstance(filter_functions, list):
        if filter_names is None:
            filter_names = [f"filter_{i}" for i in range(len(filter_functions))]
        filters_dict = dict(zip(filter_names, filter_functions))
    else:
        raise ValueError("filter_functions must be dict or list")

    results = {}
    for name, func in filters_dict.items():
        try:
            filtered = func()
            normalized = normalize_filter_output(filtered, target_shape)
            results[name] = normalized
        except Exception as e:
            p("Warning", f"Filter '{name}' failed: {e}", color1=c.ORANGE)
            # Return zeros as fallback
            results[name] = np.zeros(target_shape, dtype=np.uint8)

    return results


def create_multichannel_image(filter_outputs, channel_names=None, num_channels=3):
    """
    Create multi-channel image from filter outputs.

    Args:
        filter_outputs (dict or list): Filter output arrays. Dict of {name: 2D array} 
                                       or list of 2D arrays.
        channel_names (list, optional): Specific channel names to use in order. 
                                       If None, uses first num_channels filters.
        num_channels (int): Number of output channels. Defaults to 3.

    Returns:
        np.ndarray: Multi-channel image with shape (H, W, num_channels).
    
    Raises:
        ValueError: If channel shapes are mismatched or invalid input format.
    """
    if isinstance(filter_outputs, dict):
        if channel_names is None:
            # Use first num_channels filters
            channel_names = list(filter_outputs.keys())[:num_channels]
        channels = [filter_outputs[name] for name in channel_names]
    elif isinstance(filter_outputs, (list, tuple)):
        channels = list(filter_outputs)[:num_channels]
    else:
        raise ValueError("filter_outputs must be dict or list")

    # Pad with last channel if needed
    while len(channels) < num_channels:
        channels.append(channels[-1].copy())

    # Verify all channels have same shape
    shapes = [ch.shape for ch in channels[:num_channels]]
    if len(set(shapes)) != 1:
        raise ValueError(f"Channel shape mismatch: {shapes}")

    # Stack
    multichannel = np.stack(channels[:num_channels], axis=2)
    return multichannel


def create_enhanced_image_robust(img, filter_names, filter_registry=None):
    """
    Robustly create enhanced multi-channel image with automatic error handling.
    
    Applies filters from registry to image, normalizes outputs, and stacks into 
    multi-channel result.

    Args:
        img (np.ndarray): Input RGB image.
        filter_names (list): Names of filters to apply.
        filter_registry (dict, optional): Dict of {name: callable}. If None, uses 
                                         default filters (laplacian, sobel, clahe).

    Returns:
        np.ndarray: 3-channel enhanced image.
    """
    from src.exploration.enhancement import clahe_enhance, to_gray
    from src.exploration.filters import cv2_apply_laplacian, cv2_apply_sobel

    gray = to_gray(img)

    # Default filter registry
    if filter_registry is None:
        filter_registry = {
            "laplacian": lambda: cv2_apply_laplacian(img),
            "sobel": lambda: cv2_apply_sobel(img),
            "clahe": lambda: clahe_enhance(gray),
        }

    # Apply filters safely
    filter_outputs = apply_filters_safe(img, filter_registry)

    # Create 3-channel image
    enhanced = create_multichannel_image(
        filter_outputs, channel_names=filter_names, num_channels=3
    )

    return enhanced


# Convenience function for quick testing
def test_filter_pipeline(img_path, filter_names=["laplacian", "sobel", "clahe"]):
    """
    Test complete filter pipeline on an image file.

    Args:
        img_path (str): Path to image file to process.
        filter_names (list): List of filters to apply. 
                            Defaults to ["laplacian", "sobel", "clahe"].

    Returns:
        tuple: (original_img, enhanced_img, filter_outputs_dict)
    
    Raises:
        ValueError: If image file cannot be loaded.
    """
    import cv2

    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply filters
    enhanced = create_enhanced_image_robust(img, filter_names)

    # Get individual filter outputs for inspection
    from src.exploration.enhancement import clahe_enhance, to_gray
    from src.exploration.filters import cv2_apply_laplacian, cv2_apply_sobel

    gray = to_gray(img)
    filter_registry = {
        "laplacian": lambda: cv2_apply_laplacian(img),
        "sobel": lambda: cv2_apply_sobel(img),
        "clahe": lambda: clahe_enhance(gray),
    }

    filter_outputs = apply_filters_safe(img, filter_registry)

    return img, enhanced, filter_outputs
