from data.enhance_masks import EnhancedImageMaskDataset


def get_available_filters():
    """
    Get list of all available filter names from the system.
    """
    try:
        available = EnhancedImageMaskDataset.get_available_filters()
        return available
    except Exception as e:
        p(
            "Warning",
            f"Could not load filters dynamically: {e}",
            color1=c.ORANGE,
            color2=c.ORANGE,
        )
        # Fallback to known filters
        return [
            "laplacian",
            "sobel",
            "clahe",
            "gaussian_3x3",
            "gaussian_5x5",
            "gaussian_7x7",
            "sobel_x",
            "sobel_y",
            "laplacian_3x3",
            "sharpen_basic",
            "high_pass_3x3",
            "edge_enhance",
            "gaussian_3x3_sigma1",
            "gaussian_5x5_sigma1",
            "gaussian_7x7_sigma1",
        ]


def validate_filter_set(filter_names, available_filters):
    """
    Validate a list of filter names against available filters.
    """
    invalid = []
    suggestions = {}

    for fname in filter_names:
        if fname.lower() not in [f.lower() for f in available_filters]:
            invalid.append(fname)
            # Try to find suggestion
            for avail in available_filters:
                if fname.lower() in avail.lower() or avail.lower() in fname.lower():
                    suggestions[fname] = avail
                    break

    return len(invalid) == 0, invalid, suggestions
