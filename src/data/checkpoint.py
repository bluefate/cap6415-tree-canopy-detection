import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from matplotlib import pyplot as plt

from prediction.submission import export_submission
from src.models.zoo import MODEL_BUILDERS
from src.prediction.pipeline import Predictor
from src.utils.helpers import c, p, t


def find_checkpoint_directories(base_path: Path, config) -> List[Path]:
    """
    Find all checkpoint directories (checkpoints, checkpoints_copy, checkpoints2, etc.)
    """
    checkpoint_dirs = []

    if not base_path.exists():
        p(f"Base path does not exist: {base_path}", color1=c.RED)
        return checkpoint_dirs

    # Look for exact match 'checkpoints'
    checkpoints_dir = base_path / "checkpoints"
    if checkpoints_dir.exists() and checkpoints_dir.is_dir():
        checkpoint_dirs.append(checkpoints_dir)

    # Look for variations like checkpoints_copy, checkpoints2, etc.
    root = config.paths.root
    for item in root.iterdir():
        if item.is_dir():
            item_name_lower = item.name.lower()
            # Check for checkpoint variations OR numbered directories (like 03, 07, 10, 11)
            if item.name != "checkpoints" and (
                "checkpoint" in item_name_lower
                or "model" in item_name_lower
                or item.name.startswith("ckpt")
                or
                # Look for numbered variations
                re.match(r"checkpoints?\d+", item_name_lower)
                or
                # Look for suffix variations
                re.match(r"checkpoints?_\w+", item_name_lower)
                or
                # NEW: Look for pure numbered directories (like 03, 07, 10, 11)
                re.match(r"^\d{2,3}$", item.name)
            ):
                checkpoint_dirs.append(item)

    p(f"Searching for checkpoint directories in: {base_path}")
    if checkpoint_dirs:
        p(f"Found checkpoint directories")
        for dir_path in checkpoint_dirs:
            p(f"  {dir_path.name}")
    else:
        p(f"No checkpoint directories found. Available directories")
        for item in base_path.iterdir():
            if item.is_dir():
                p(f"  {item.name}")

    return sorted(checkpoint_dirs)


def extract_model_info_from_path(model_path: Path) -> Dict[str, str]:
    """
    Extract model information from the structured file path.
    """
    parts = model_path.parts
    path_str = str(model_path)

    # Find the checkpoint directory index
    checkpoint_idx = -1
    for i, part in enumerate(parts):
        if "checkpoint" in part.lower():
            checkpoint_idx = i
            break

    if checkpoint_idx == -1:
        p(
            f"Warning: Could not find checkpoint directory in path: {model_path}",
            color1=c.ORANGE,
        )
        return extract_model_info_fallback(model_path)

    # Extract information from structured path
    try:
        # Start from after checkpoint directory
        remaining_parts = parts[checkpoint_idx + 1 :]

        if len(remaining_parts) < 3:  # Need at least model/mode/version
            return extract_model_info_fallback(model_path)

        model_name = remaining_parts[0]
        mode = remaining_parts[1]

        # Check if there's a size directory
        if len(remaining_parts) >= 4 and remaining_parts[2].startswith("size_"):
            image_size = remaining_parts[2].replace("size_", "")
            version = remaining_parts[3]
        else:
            # No size directory, extract from other clues
            version = remaining_parts[2]
            image_size = extract_size_from_context(model_path)

        # Extract filters from path if present
        filters = extract_filters_from_path(model_path)

        return {
            "model_name": model_name,
            "mode": mode,
            "image_size": image_size,
            "version": version,
            "filters": filters,
            "structured_path": True,
        }

    except Exception as e:
        p(f"Error parsing structured path {model_path}: {str(e)}", color1=c.ORANGE)
        return extract_model_info_fallback(model_path)


def extract_model_info_fallback(model_path: Path) -> Dict[str, str]:
    """
    Fallback method to extract model information when structured path parsing fails.
    """
    path_str = str(model_path).lower()

    # Extract model name
    model_name = "unknown"
    model_keywords = [
        "simple_cnn",
        "unet",
        "smp_unet",
        "smp_fpn",
        "smp_linknet",
        "smp_deeplabv3",
        "smp_deeplabv3plus",
        "yolov8n",
        "yolov8s",
        "yolov8m",
        "yolov8l",
        "resnet",
        "efficientnet",
    ]

    if model_name == "unknown":
        parts = model_path.parts
        for part in parts:
            if part.lower().startswith("smp_") or part.lower().startswith("yolov8"):
                model_name = part.lower()
                break

    for keyword in model_keywords:
        if keyword in path_str:
            model_name = keyword
            break

    # Extract mode/input type
    mode = "rgb"
    if "filtered" in path_str:
        mode = "filtered"
    elif "concat" in path_str:
        mode = "concat"
    elif "gray" in path_str or "grey" in path_str:
        mode = "grayscale"

    # Extract image size
    image_size = extract_size_from_context(model_path)

    # Extract version info
    parts = model_path.parts
    version_info = []
    for part in parts:
        if part.startswith("v") and len(part) <= 6:  # Like v1, v2, v10, etc.
            version_info.append(part)
        elif "version" in part.lower():
            version_info.append(part)

    version = "_".join(version_info) if version_info else "unknown"

    # Extract filters
    filters = extract_filters_from_path(model_path)

    return {
        "model_name": model_name,
        "mode": mode,
        "image_size": image_size,
        "version": version,
        "filters": filters,
        "structured_path": False,
    }


def extract_size_from_context(model_path: Path) -> str:
    """
    Extract image size from path context or filename.
    """
    path_str = str(model_path)

    # Look for size_XXX pattern
    size_match = re.search(r"size_(\d+)", path_str)
    if size_match:
        return size_match.group(1)

    # Look for XXX pattern in various contexts
    size_patterns = [
        r"(\d+)(?:x\d+)?(?:_rgb|_filtered|_concat)",
        r"(\d+)_rgb",
        r"(\d+)_filtered",
        r"(\d+)_concat",
        r"plan_(\d+)_",
        r"size(\d+)",
    ]

    for pattern in size_patterns:
        match = re.search(pattern, path_str)
        if match:
            return match.group(1)

    return "unknown"


def extract_filters_from_path(model_path: Path) -> str:
    """
    Extract filter information from the path.
    """
    path_str = str(model_path).lower()

    # Check for filter directories or indicators
    if "filter" in path_str:
        if "sobel" in path_str:
            return "sobel"
        elif "canny" in path_str:
            return "canny"
        elif "gaussian" in path_str:
            return "gaussian"
        else:
            return "custom"

    # Check for concat mode (which often involves filters)
    if "concat" in path_str:
        return "concat_filters"

    return "none"


def get_model_training_info(model_path: Path) -> Dict[str, Any]:
    """
    Extract training information from the model checkpoint.
    """
    try:
        checkpoint = torch.load(model_path, map_location="cpu")

        info = {
            "epoch": checkpoint.get("epoch", None),
            "val_loss": checkpoint.get("val_loss", None),
            "train_loss": checkpoint.get("train_loss", None),
            "best_val_loss": checkpoint.get("best_val_loss", None),
            "learning_rate": checkpoint.get("learning_rate", None),
            "model_state_available": "model" in checkpoint,
            "optimizer_state_available": "optimizer" in checkpoint,
        }

        # Try to get additional metrics if available
        for metric in ["iou", "accuracy", "precision", "recall", "f1_score", "dice"]:
            if metric in checkpoint:
                info[metric] = checkpoint[metric]

        return info

    except Exception as e:
        return {"error": str(e), "epoch": None, "val_loss": None}


def scan_all_models(base_path: Path, config) -> List[Dict[str, Any]]:
    """
    Scan all checkpoint directories and collect model information.
    """
    p("Base_path", base_path)
    checkpoint_dirs = find_checkpoint_directories(base_path, config)
    all_models: List[Dict[str, Any]] = []
    p("Checkpoint directories found", checkpoint_dirs)

    t(f"Scanning Checkpoint Directories")
    p(f"Found {len(checkpoint_dirs)} checkpoint directories")

    for checkpoint_dir in checkpoint_dirs:
        p(f"Scanning: {checkpoint_dir.name}", color1=c.BLUE)

        # Find all .pth files recursively
        model_files = list(checkpoint_dir.rglob("*.pth"))
        p(f"  Found {len(model_files)} model files")

        # Show directory structure for first few files
        if len(model_files) > 0:
            p(f"  Sample paths")
            for model_file in model_files[:3]:
                rel_path = model_file.relative_to(checkpoint_dir)
                p(f"    {rel_path}")
            if len(model_files) > 3:
                p(f"    ... and {len(model_files) - 3} more")

        for model_file in model_files:
            try:
                # Extract basic info using improved parser
                model_info = extract_model_info_from_path(model_file)

                # Show parsing result for first few models
                if len(all_models) < 3:
                    p(
                        f"  Parsed {model_file.name}: "
                        f"{model_info['model_name']}/"
                        f"{model_info['mode']}/"
                        f"{model_info['image_size']}/"
                        f"{model_info['version']}",
                        color1=c.CYAN,
                    )

                # File stats
                file_stats = model_file.stat()
                created_time = datetime.fromtimestamp(file_stats.st_mtime)
                file_size = file_stats.st_size / (1024 * 1024)  # MB

                # Training info
                training_info = get_model_training_info(model_file)

                # Robust submission detection in this folder
                submission_path = next(
                    (
                        f
                        for f in model_file.parent.glob("*")
                        if f.name.lower().startswith("submission")
                        and f.suffix.lower() == ".json"
                    ),
                    None,
                )
                has_submission = submission_path is not None

                try:
                    rel_path = str(model_file.relative_to(config.paths.root))
                except ValueError:
                    rel_path = str(
                        model_file
                    )  # fallback to full path if not under root

                # Combine all information
                model_data: Dict[str, Any] = {
                    "file_path": str(model_file),
                    # "relative_path": str(model_file.relative_to(base_path)),
                    "relative_path": rel_path,
                    "checkpoint_dir": checkpoint_dir.name,
                    "file_name": model_file.name,
                    "file_size_mb": round(file_size, 2),
                    "created_date": created_time,
                    "has_submission": has_submission,
                    "submission_path": (
                        str(submission_path) if submission_path is not None else None
                    ),
                    **model_info,
                    **training_info,
                }

                all_models.append(model_data)

            except Exception as e:
                p(f"  Error processing {model_file}: {str(e)}", color1=c.ORANGE)
                import traceback

                traceback.print_exc()

    p(f"Total models found: {len(all_models)}")

    # Summary of discovered models
    if all_models:
        t("Model Discovery Summary")
        model_counts: Dict[str, int] = {}
        mode_counts: Dict[str, int] = {}

        for model in all_models:
            model_name = model.get("model_name", "unknown")
            mode = model.get("mode", "unknown")

            model_counts[model_name] = model_counts.get(model_name, 0) + 1
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

        p("Models by type", model_counts)
        p("Models by mode", mode_counts)

        # Models with and without submissions
        with_sub = sum(1 for m in all_models if m.get("has_submission"))
        p(f"Models with submissions: {with_sub}/{len(all_models)}")

    return all_models


def generate_submission_for_model(model_path: Path, config) -> Optional[Path]:
    """
    Generate a submission file for a given model checkpoint (.pth).

    Uses the same building blocks as your prediction notebooks:
    - build Predictor with a valid model_name and image_size
    - run inference on config.paths.eval_images
    - call export_submission to write SUBMISSION.json in the model folder
    """
    try:
        # Normalize to Path
        if not isinstance(model_path, Path):
            model_path = Path(model_path)

        p("Trying to create submission for", str(model_path))

        if not model_path.exists():
            p("Error", f"Model file not found: {model_path}", color1=c.RED)
            return None

        # Check if a submission already exists in this folder (case insensitive)
        existing_sub = next(
            (
                f
                for f in model_path.parent.glob("*")
                if f.name.lower().startswith("submission")
                and f.suffix.lower() == ".json"
            ),
            None,
        )
        if existing_sub is not None:
            p("Submission already exists for", str(model_path.parent))
            return existing_sub

        # IMPORTANT
        # For submissions we always use the fallback parser
        # so we never treat "03", "10", "11" as model names.
        # model_info = extract_model_info_fallback(model_path)
        #
        # model_name = model_info.get("model_name", "unet")
        # image_size_str = model_info.get("image_size", "unknown")

        # Scan path to find a valid model name
        model_name = next(
            (
                part.lower()
                for part in model_path.parts
                if part.lower() in MODEL_BUILDERS
            ),
            "unet",
        )
        # Try to find a known model from path
        known_models = set(MODEL_BUILDERS.keys())
        path_parts = [p.lower() for p in model_path.parts]

        # Direct match
        model_name = next((p for p in path_parts if p in known_models), None)

        # Optional remap (you can expand this as needed)
        name_map = {
            "unet": "smp_unet",  # only if you want this behavior
            "deeplabv3": "smp_deeplabv3",
            "deeplabv3plus": "smp_deeplabv3plus",
        }
        if model_name is None:
            # fallback: detect and remap if applicable
            for p in path_parts:
                if p in name_map:
                    model_name = name_map[p]
                    break

        # Fallback hard default
        if not model_name:
            model_name = "smp_unet"


        # Get image size using fallback logic
        model_info = extract_model_info_fallback(model_path)
        image_size_str = model_info.get("image_size", "unknown")

        # Convert image size to int if possible
        image_size: Optional[int] = None
        if image_size_str and image_size_str != "unknown":
            try:
                image_size = int(image_size_str)
            except ValueError:
                image_size = None

        # Fallback to config.train.image_size or 256
        if image_size is None:
            train_cfg = getattr(config, "train", None)
            cfg_size = getattr(train_cfg, "image_size", None) if train_cfg else None
            if cfg_size is not None:
                try:
                    image_size = int(cfg_size)
                except ValueError:
                    image_size = 256
            else:
                image_size = 256

        p(
            "Parsed model for submission",
            f"name={model_name}",
            f"image_size={image_size}",
        )

        # Evaluation images directory (same as used in prediction notebooks)
        eval_dir = Path(config.paths.eval_images)
        if not eval_dir.exists():
            p(
                "Error",
                f"Evaluation images directory not found: {eval_dir}",
                color1=c.RED,
            )
            return None

        t(f"=== === Generating submission for: {model_path.name} === ===")

        # Build predictor and run on evaluation folder
        predictor = Predictor(
            model_path=model_path,
            model_name=model_name,
            image_size=image_size,
        )

        results = predictor.run_on_folder(
            eval_dir,
            transform=None,
            num_samples=None,
        )

        # Write SUBMISSION.json alongside the checkpoint
        submission_path = model_path.parent / "SUBMISSION.json"
        export_submission(results, submission_path, config)

        p("Submission created", str(submission_path), color1=c.GREEN)
        return submission_path

    except Exception as e:
        p(
            "Failed to generate submission",
            f"{e} | model={model_name} | path={model_path}",
            color1=c.RED,
        )

        import traceback

        traceback.print_exc()
        return None


def generate_missing_submissions(models_list: List[Dict[str, Any]], config) -> int:
    """
    Generate submission files for all models that do not have them.
    Work at the version folder level, so each model directory gets
    at most one SUBMISSION.json.
    """
    t("Generating Missing Submissions")

    # Filter models that currently have no submission
    models_without_submissions = [m for m in models_list if not m.get("has_submission")]
    p(f"Models without submissions: {len(models_without_submissions)}")

    if not models_without_submissions:
        p("All models already have submissions!", color1=c.GREEN)
        return 0

    # Work once per parent directory
    unique_dirs: Dict[Path, Path] = {}
    for m in models_without_submissions:
        model_path = Path(m["file_path"])
        model_dir = model_path.parent
        if model_dir not in unique_dirs:
            unique_dirs[model_dir] = model_path

    dirs_to_process = list(unique_dirs.items())
    generated_count = 0

    for idx, (model_dir, model_path) in enumerate(dirs_to_process, start=1):
        p("")
        p(f"Progress: {idx}/{len(dirs_to_process)}")
        p("Creating submission for", str(model_path))

        sub_path = generate_submission_for_model(model_path, config)
        if sub_path is not None:
            generated_count += 1

    p("")
    p(
        f"Submissions generated: {generated_count}/{len(dirs_to_process)}",
        color1=c.CYAN,
    )

    return generated_count


def rank_models_by_performance(
    models_list: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Rank models from best to worst based on validation loss and other metrics.
    """

    def calculate_score(model: Dict[str, Any]) -> float:
        """Calculate a composite score for ranking."""
        score = 0.0

        # Primary metric: validation loss (lower is better)
        val_loss = model.get("val_loss")
        if val_loss is not None:
            # Convert to score (higher is better)
            score += max(0, 10 - val_loss * 10)  # Assuming val_loss is usually < 1

        # Secondary metrics (higher is better)
        for metric in ["iou", "accuracy", "precision", "recall", "f1_score", "dice"]:
            value = model.get(metric)
            if value is not None:
                score += value * 5  # Weight these metrics

        # Bonus for having submission
        if model.get("has_submission"):
            score += 1

        # Small penalty for very old models (encourage recent experiments)
        created_date = model.get("created_date")
        if created_date:
            days_old = (datetime.now() - created_date).days
            if days_old > 30:
                score -= 0.1

        return score

    # Filter out models with errors
    valid_models = [m for m in models_list if m.get("error") is None]

    # Sort by score (highest first)
    ranked_models = sorted(valid_models, key=calculate_score, reverse=True)

    return ranked_models


def create_model_summary_dataframe(models_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a comprehensive DataFrame for model analysis.
    """
    df_data = []

    for model in models_list:
        row = {
            "Model Name": model.get("model_name", "unknown"),
            "Mode": model.get("mode", "unknown"),
            "Image Size": model.get("image_size", "unknown"),
            "Version": model.get("version", "unknown"),
            "Filters": model.get("filters", "none"),
            "Checkpoint Dir": model.get("checkpoint_dir", "unknown"),
            "File Name": model.get("file_name", "unknown"),
            "File Size (MB)": model.get("file_size_mb", 0),
            "Created Date": model.get("created_date"),
            "Epochs": model.get("epoch"),
            "Val Loss": model.get("val_loss"),
            "Train Loss": model.get("train_loss"),
            "Best Val Loss": model.get("best_val_loss"),
            "IoU": model.get("iou"),
            "Accuracy": model.get("accuracy"),
            "Precision": model.get("precision"),
            "Recall": model.get("recall"),
            "F1 Score": model.get("f1_score"),
            "Dice": model.get("dice"),
            "Has Submission": model.get("has_submission", False),
            "Submission Path": model.get("submission_path"),
            "File Path": model.get("file_path"),
            "Error": model.get("error"),
        }
        df_data.append(row)

    df = pd.DataFrame(df_data)

    # Sort by validation loss (best first)
    df = df.sort_values(by=["Val Loss"], ascending=True, na_position="last")

    return df


def print_model_rankings(ranked_models: List[Dict[str, Any]], top_n: int = 10) -> None:
    """
    Print a nicely formatted ranking of the top models.
    """
    t(f"Top {min(top_n, len(ranked_models))} Model Rankings")

    for i, model in enumerate(ranked_models[:top_n]):
        rank = i + 1
        model_name = model.get("model_name", "unknown")
        mode = model.get("mode", "unknown")
        val_loss = model.get("val_loss")
        epochs = model.get("epoch")
        created_date = model.get("created_date")
        has_submission = model.get("has_submission", False)

        # Format date
        date_str = (
            created_date.strftime("%Y-%m-%d %H:%M") if created_date else "unknown"
        )

        # Format validation loss
        val_loss_str = f"{val_loss:.6f}" if val_loss is not None else "N/A"

        # Status indicators
        sub_indicator = "✓" if has_submission else "✗"

        p(
            f"{rank:2d}. {model_name} ({mode})",
            f"Val Loss: {val_loss_str} | Epochs: {epochs} | {date_str} | Sub: {sub_indicator}",
            color1=c.GREEN if rank <= 3 else c.BLUE if rank <= 5 else c.BLACK,
        )

        # Show additional metrics if available
        metrics = []
        for metric_name, display_name in [
            ("iou", "IoU"),
            ("accuracy", "Acc"),
            ("f1_score", "F1"),
        ]:
            value = model.get(metric_name)
            if value is not None:
                metrics.append(f"{display_name}: {value:.3f}")

        if metrics:
            p(f"    {' | '.join(metrics)}")


def create_submission_links_report(models_list: List[Dict[str, Any]]) -> str:
    """
    Create a markdown report with clickable submission links.
    Re verifies submission presence for each model directory.
    """
    # Re verify submission status for all models
    for model in models_list:
        model_path = Path(model["file_path"])
        submission_path = next(
            (
                f
                for f in model_path.parent.glob("*")
                if f.name.lower().startswith("submission")
                and f.suffix.lower() == ".json"
            ),
            None,
        )
        has_submission = submission_path is not None
        model["has_submission"] = has_submission
        if has_submission:
            model["submission_path"] = str(submission_path)

    models_with_submissions = [m for m in models_list if m.get("has_submission")]
    ranked_models = rank_models_by_performance(models_with_submissions)

    report_lines = [
        "# Model Submission Links Report",
        "",
        f"Total models with submissions: {len(models_with_submissions)}",
        "",
        "## Ranked by Performance (Best to Worst)",
        "",
    ]

    for i, model in enumerate(ranked_models):
        rank = i + 1
        model_name = model.get("model_name", "unknown")
        mode = model.get("mode", "unknown")
        val_loss = model.get("val_loss")
        submission_path = model.get("submission_path")
        created_date = model.get("created_date")

        # Format validation loss
        val_loss_str = f"{val_loss:.6f}" if val_loss is not None else "N/A"

        # Format date
        date_str = created_date.strftime("%Y-%m-%d") if created_date else "unknown"

        # Create submission link
        rel_submission_path = Path(submission_path).name if submission_path else "N/A"
        link = (
            f"[{rel_submission_path}]({submission_path})"
            if submission_path
            else "No submission"
        )

        report_lines.extend(
            [
                f"### {rank}. {model_name} ({mode}) - Val Loss: {val_loss_str}",
                f"- **Date Created:** {date_str}",
                f"- **Submission:** {link}",
                f"- **Model Path:** `{model.get('relative_path', 'unknown')}`",
                "",
            ]
        )

    return "\n".join(report_lines)


def plot_model_performance_overview(df: pd.DataFrame) -> None:
    """
    Create visualizations showing model performance overview.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Model Performance Overview", fontsize=16, fontweight="bold")

    # Filter valid models (with validation loss)
    valid_df = df.dropna(subset=["Val Loss"])

    if len(valid_df) == 0:
        p("No models with validation loss data for visualization", color1=c.ORANGE)
        plt.close(fig)
        return

    # 1. Validation Loss Distribution
    axes[0, 0].hist(
        valid_df["Val Loss"], bins=min(20, len(valid_df)), alpha=0.7, color="skyblue"
    )
    axes[0, 0].set_xlabel("Validation Loss")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Validation Loss Distribution")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Performance by Model Type
    if "Model Name" in valid_df.columns:
        model_performance = valid_df.groupby("Model Name")["Val Loss"].agg(
            ["mean", "min", "count"]
        )
        model_performance = model_performance.sort_values("mean")

        x_pos = range(len(model_performance))
        axes[0, 1].bar(x_pos, model_performance["mean"], alpha=0.7, color="lightcoral")
        axes[0, 1].set_xlabel("Model Type")
        axes[0, 1].set_ylabel("Average Val Loss")
        axes[0, 1].set_title("Average Performance by Model Type")
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(model_performance.index, rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

    # 3. Training Progress (Epochs vs Val Loss)
    epochs_df = valid_df.dropna(subset=["Epochs"])
    if len(epochs_df) > 0:
        scatter = axes[1, 0].scatter(
            epochs_df["Epochs"],
            epochs_df["Val Loss"],
            alpha=0.6,
            c=epochs_df["Val Loss"],
            cmap="viridis",
        )
        axes[1, 0].set_xlabel("Epochs Trained")
        axes[1, 0].set_ylabel("Validation Loss")
        axes[1, 0].set_title("Epochs vs Validation Loss")
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 0])

    # 4. Submission Status
    submission_counts = df["Has Submission"].value_counts()
    colors = ["lightgreen" if x else "lightcoral" for x in submission_counts.index]
    labels = [
        "Has Submission" if x else "Missing Submission" for x in submission_counts.index
    ]

    axes[1, 1].pie(
        submission_counts.values, labels=labels, colors=colors, autopct="%1.1f%%"
    )
    axes[1, 1].set_title("Submission Status")

    plt.tight_layout()
    plt.show()


def plot_model_timeline(df: pd.DataFrame) -> None:
    """
    Create a timeline visualization of model training.
    """
    date_df = df.dropna(subset=["Created Date"])

    if len(date_df) == 0:
        p("No date information available for timeline", color1=c.ORANGE)
        return

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Group by date and count models
    date_df["Date Only"] = date_df["Created Date"].dt.date
    daily_counts = date_df.groupby("Date Only").size()

    ax.plot(
        daily_counts.index, daily_counts.values, marker="o", linewidth=2, markersize=6
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Models Created")
    ax.set_title("Model Creation Timeline")
    ax.grid(True, alpha=0.3)

    # Format x-axis dates
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main_model_tracking_pipeline(config):
    """
    Main pipeline for model tracking and submission management.
    """
    p()
    t("Model Tracking and Submission Management Pipeline")

    # Step 1: Scan all models
    p()
    t("Step 1: Scanning All Models")
    models_list = scan_all_models(config.paths.models, config)

    if len(models_list) == 0:
        p("No models found! Check your paths.", color1=c.RED)
        return

    # Step 2: Generate missing submissions
    p()
    t("Step 2: Generating Missing Submissions")
    generated_count = generate_missing_submissions(models_list, config)

    # Step 3: Rank models by performance
    p()
    t("Step 3: Ranking Models by Performance")
    ranked_models = rank_models_by_performance(models_list)

    # Step 4: Create summary DataFrame
    p()
    t("Step 4: Creating Summary DataFrame")
    df_summary = create_model_summary_dataframe(models_list)

    # Step 5: Print rankings
    p()
    t("Step 5: Top Model Rankings")
    print_model_rankings(ranked_models, top_n=15)

    # Step 6: Create visualizations
    p()
    t("Step 6: Creating Performance Visualizations")
    plot_model_performance_overview(df_summary)
    # plot_model_timeline(df_summary)

    # Step 7: Generate submission links report
    p()
    t("Step 7: Generating Submission Links Report")
    report_content = create_submission_links_report(models_list)

    # Save report to file
    p()
    report_path = config.paths.models / "model_submission_links_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    p(f"Submission links report saved: {report_path}", color1=c.GREEN)

    # Step 8: Save detailed CSV
    p()
    t("Step 8: Saving Detailed Model Information")
    csv_path = config.paths.models / "all_models_detailed.csv"
    df_summary.to_csv(csv_path, index=False)
    p(f"Detailed model CSV saved: {csv_path}", color1=c.GREEN)

    # Summary statistics
    p()
    t("Pipeline Summary")
    total_models = len(models_list)
    models_with_submissions = sum(1 for m in models_list if m.get("has_submission"))
    models_with_val_loss = sum(1 for m in models_list if m.get("val_loss") is not None)

    p()
    p(f"Total models found: {total_models}")
    p(f"Models with submissions: {models_with_submissions}")
    p(f"Models with validation loss: {models_with_val_loss}")
    p(f"Submissions generated: {generated_count}")

    if ranked_models:
        best_model = ranked_models[0]
        p(
            f"Best model: {best_model.get('model_name')} ({best_model.get('mode')}) - Val Loss: {best_model.get('val_loss')}",
            color1=c.GREEN,
        )

    return {
        "models_list": models_list,
        "ranked_models": ranked_models,
        "summary_df": df_summary,
        "report_path": report_path,
        "csv_path": csv_path,
        "generated_submissions": generated_count,
    }


def interactive_model_explorer(models_list: List[Dict[str, Any]]):
    """
    Interactive explorer for examining individual models.
    """
    t("Interactive Model Explorer")

    # Filter models with submissions
    models_with_subs = [m for m in models_list if m.get("has_submission")]

    if not models_with_subs:
        p("No models with submissions found!", color1=c.RED)
        return

    p(f"Available models with submissions: {len(models_with_subs)}")

    # List first 10 models for selection
    for i, model in enumerate(models_with_subs[:10]):
        model_name = model.get("model_name", "unknown")
        mode = model.get("mode", "unknown")
        val_loss = model.get("val_loss")
        val_loss_str = f"{val_loss:.6f}" if val_loss else "N/A"

        p(f"{i + 1:2d}. {model_name} ({mode}) - Val Loss: {val_loss_str}")

    if len(models_with_subs) > 10:
        p(f"... and {len(models_with_subs) - 10} more models")


def show_model_details(model_data: Dict[str, Any]):
    """
    Show detailed information about a specific model.
    """
    t(f"Model Details: {model_data.get('file_name', 'Unknown')}")

    # Basic information
    p("Model Name", model_data.get("model_name", "unknown"))
    p("Mode", model_data.get("mode", "unknown"))
    p("Image Size", model_data.get("image_size", "unknown"))
    p("Version", model_data.get("version", "unknown"))
    p("Filters", model_data.get("filters", "none"))
    p("Checkpoint Directory", model_data.get("checkpoint_dir", "unknown"))

    # File information
    p("File Size", f"{model_data.get('file_size_mb', 0):.2f} MB")
    created_date = model_data.get("created_date")
    if created_date:
        p("Created Date", created_date.strftime("%Y-%m-%d %H:%M:%S"))

    # Training information
    p("Epochs Trained", model_data.get("epoch", "N/A"))

    val_loss = model_data.get("val_loss")
    if val_loss is not None:
        p("Validation Loss", f"{val_loss:.6f}", color1=c.GREEN)

    train_loss = model_data.get("train_loss")
    if train_loss is not None:
        p("Training Loss", f"{train_loss:.6f}")

    # Additional metrics
    metrics = ["iou", "accuracy", "precision", "recall", "f1_score", "dice"]
    available_metrics = {
        m: model_data.get(m) for m in metrics if model_data.get(m) is not None
    }

    if available_metrics:
        t("Performance Metrics")
        for metric, value in available_metrics.items():
            p(metric.replace("_", " ").title(), f"{value:.4f}")

    # Submission information
    has_submission = model_data.get("has_submission", False)
    p(
        "Has Submission",
        "Yes" if has_submission else "No",
        color1=c.GREEN if has_submission else c.RED,
    )

    if has_submission:
        submission_path = model_data.get("submission_path")
        if submission_path:
            p("Submission Path", submission_path, color1=c.BLUE)


def generate_best_model_report(ranked_models: List[Dict[str, Any]], config):
    """
    Generate a comprehensive report for the best performing model.
    """
    if not ranked_models:
        p("No models available for best model report", color1=c.RED)
        return

    best_model = ranked_models[0]

    t("Best Model Report Generation")

    # Create report content
    report_lines = [
        "# Best Model Performance Report",
        "",
        "## Model Information",
        f"- **Model Type:** {best_model.get('model_name', 'unknown')}",
        f"- **Input Mode:** {best_model.get('mode', 'unknown')}",
        f"- **Image Size:** {best_model.get('image_size', 'unknown')}",
        f"- **Version:** {best_model.get('version', 'unknown')}",
        f"- **Filters Applied:** {best_model.get('filters', 'none')}",
        "",
        "## Training Results",
        f"- **Epochs Trained:** {best_model.get('epoch', 'N/A')}",
        (
            f"- **Validation Loss:** {best_model.get('val_loss', 'N/A'):.6f}"
            if best_model.get("val_loss")
            else "- **Validation Loss:** N/A"
        ),
        (
            f"- **Training Loss:** {best_model.get('train_loss', 'N/A'):.6f}"
            if best_model.get("train_loss")
            else "- **Training Loss:** N/A"
        ),
        "",
        "## Performance Metrics",
    ]

    # performance metrics
    metrics = ["iou", "accuracy", "precision", "recall", "f1_score", "dice"]
    for metric in metrics:
        value = best_model.get(metric)
        if value is not None:
            metric_name = metric.replace("_", " ").title()
            report_lines.append(f"- **{metric_name}:** {value:.4f}")

    # file information
    created_date = best_model.get("created_date")
    report_lines.extend(
        [
            "",
            "## File Information",
            f"- **File Path:** `{best_model.get('file_path', 'unknown')}`",
            f"- **File Size:** {best_model.get('file_size_mb', 0):.2f} MB",
            (
                f"- **Created Date:** {created_date.strftime('%Y-%m-%d %H:%M:%S')}"
                if created_date
                else "- **Created Date:** Unknown"
            ),
            f"- **Checkpoint Directory:** {best_model.get('checkpoint_dir', 'unknown')}",
        ]
    )

    # submission information
    if best_model.get("has_submission"):
        submission_path = best_model.get("submission_path", "")
        report_lines.extend(
            [
                "",
                "## Submission",
                f"- **Submission Available:** Yes",
                f"- **Submission Path:** `{submission_path}`",
                f"- **Download Link:** [SUBMISSION.json]({submission_path})",
            ]
        )
    else:
        report_lines.extend(
            [
                "",
                "## Submission",
                "- **Submission Available:** No - Generate submission file needed",
            ]
        )

    # Save report
    report_content = "\n".join(report_lines)
    best_model_report_path = config.paths.models / "BEST_MODEL_REPORT.md"

    with open(best_model_report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    p(f"Best model report saved: {best_model_report_path}", color1=c.GREEN)

    # Also save best model info in simple text format (compatible with existing code)
    best_model_info_path = config.paths.models / "BEST_MODEL.txt"
    with open(best_model_info_path, "w") as f:
        f.write(f"model: {best_model.get('model_name', 'unknown')}\n")
        f.write(f"path: {best_model.get('file_path', 'unknown')}\n")
        f.write(f"val_loss: {best_model.get('val_loss', 'N/A')}\n")
        f.write(f"mode: {best_model.get('mode', 'unknown')}\n")
        f.write(f"version: {best_model.get('version', 'unknown')}\n")

    p(f"Best model info saved: {best_model_info_path}", color1=c.GREEN)

    return best_model_report_path
