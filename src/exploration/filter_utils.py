"""Filter channel helpers for model input sizing."""

from src.utils.helpers import c, p


def get_input_channels(mode, filters, model_name):
    """Return input channel count for mode/model (rgb=3, filtered=3, concat=3+N)."""
    if mode == "rgb":
        return 3
    if mode == "filtered":
        return 3
    if mode == "concat":
        if model_name in ["simple_cnn", "unet"]:
            p(
                "WARNING",
                f"{model_name} doesn't support concat mode, using RGB",
                color1=c.ORANGE,
            )
            return 3
        return 3 + (len(filters) if filters else 0)
    return 3
