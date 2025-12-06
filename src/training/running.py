import copy

import torch

from src.data.enhance_masks import EnhancedImageMaskDataset
from src.utils.helpers import c, p


def get_version_config(
    config,
    filters,
    notebook,
    model_name,
    mode,
    in_channels,
    best_model_tracker=None,
    i=None,
    experiments=None,
):
    version_root = config.paths.models / notebook / model_name / mode
    if filters:
        filter_str = "_".join(filters)
        version_root = version_root / filter_str
    # Add image size to path to differentiate models
    version_root = version_root / f"size_{config.train.image_size}"

    version_root.mkdir(parents=True, exist_ok=True)
    p("Version root", version_root)

    # saving custom config
    exp_config = copy.deepcopy(config)
    exp_config.extra["experiment"] = {
        "model_name": model_name,
        "input_mode": mode,
        "filter_names": filters,
        "input_channels": in_channels,
    }

    # experiment key
    key = f"{model_name}_{mode}"
    if filters:
        filter_key = "_".join(filters)
        key = f"{key}_{filter_key}"
    p("key", key)

    # Check if experiment already completed successfully
    best_model_path = version_root / "best_model.pth"
    checkpoint_path_check = version_root / "checkpoint.pth"

    best_model_exists = False
    if best_model_tracker:
        if best_model_path.exists():
            p(f"SKIPPING {i}/{len(experiments)}", key, color1=c.CYAN, color2=c.CYAN)
            p("[Info]", f"Already trained: {best_model_path}", color1=c.CYAN)

            # Still check if this is the best model overall
            if checkpoint_path_check.exists():
                try:
                    ckpt = torch.load(checkpoint_path_check, map_location="cpu")
                    val_loss = ckpt.get("best_val_loss", float("inf"))
                    p("[Info]", f"Previous Val Loss: {val_loss:.6f}", color1=c.CYAN)

                    if val_loss < best_model_tracker["best_val_loss"]:
                        best_model_tracker["best_val_loss"] = val_loss
                        best_model_tracker["best_experiment"] = key
                        best_model_tracker["best_model_path"] = best_model_path
                        best_model_tracker["best_version_dir"] = version_root
                        p(
                            "\t\t🏆 BEST MODEL (from previous run)",
                            key,
                            color1=c.ORANGE,
                            bold=True,
                        )
                except Exception as e:
                    p("Warning", f"Could not load checkpoint: {e}", color1=c.ORANGE)

            best_model_exists = True

    return (
        key,
        version_root,
        exp_config,
        best_model_path,
        checkpoint_path_check,
        best_model_exists,
    )


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
