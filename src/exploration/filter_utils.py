"""
Filter utilities with automatic shape normalization.

This module provides helpers for applying filters and creating multi-channel
enhanced images without shape mismatch errors.
"""

import cv2
import numpy as np

from src.utils.helpers import c, p, t


def get_input_channels(mode, filters, model_name):
    """Get correct input channels for model and mode combination."""
    if mode == "rgb":
        return 3
    elif mode == "filtered":
        return 3  # Filters replace RGB channels
    elif mode == "concat":
        if model_name in ["simple_cnn", "unet"]:
            # These models don't support 6 channels - use RGB instead
            print(f"WARNING: {model_name} doesn't support concat mode, using RGB")
            return 3
        else:
            return 6  # RGB + 3 filters
    else:
        return 3  # Default fallback


def normalize_filter_output(filtered, target_shape, dtype=np.uint8):
    """
    Normalize filter output to consistent shape and dtype.

    Args:
        filtered: Filter output (any shape, any dtype)
        target_shape: Target (height, width) tuple
        dtype: Output dtype (default: np.uint8)

    Returns:
        Normalized 2D array with shape=target_shape and dtype=dtype
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
    Apply multiple filters and return as dictionary with normalized outputs.

    Args:
        img: Input RGB image
        filter_functions: Dict of {name: callable} or list of callables
        filter_names: Optional list of names (if filter_functions is a list)

    Returns:
        Dict of {filter_name: normalized_output}
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
        filter_outputs: Dict of {name: 2D array} or list of 2D arrays
        channel_names: List of channel names to use (in order)
        num_channels: Number of channels in output (default: 3)

    Returns:
        Multi-channel image with shape (H, W, num_channels)
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
    Robust version of create_enhanced_image with automatic error handling.

    Args:
        img: Input RGB image
        filter_names: List of filter names to apply
        filter_registry: Optional dict of {name: callable}. If None, uses default filters.

    Returns:
        3-channel enhanced image
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
    Test the complete filter pipeline on an image.

    Args:
        img_path: Path to image file
        filter_names: List of filters to apply

    Returns:
        Tuple of (original_img, enhanced_img, filter_outputs_dict)
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


if __name__ == "__main__":
    # Quick test
    from src.utils.config import Config
    from src.data.annotations import load_json_annotations

    config = Config.load()
    entries = load_json_annotations(config.paths.annotations)

    # Test on first image
    img_path = config.paths.train_images / entries[0].image_path.name

    t("Testing filter pipeline")

    original, enhanced, filters = test_filter_pipeline(img_path)

    p("Original shape", original.shape)
    p("Enhanced shape", enhanced.shape)
    p("Filter outputs")
    for name, output in filters.items():
        p("", f"  {name}: {output.shape}")

    p("SUCCESS", "Filter pipeline working correctly", color1=c.GREEN)
