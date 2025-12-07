# From C:\github\Tree-Canopy-Detection\src\data\annotations.py
import json
from pathlib import Path
from typing import List

import cv2
import numpy as np


class AnnotationItem:
    """
    Hold annotation data for one object.
    Stores class name, segmentation polygon, confidence score, and the computed bounding box.
    """

    def __init__(self, cls: str, segmentation: List[float], confidence: float = 1.0):
        self.cls = cls
        self.segmentation = segmentation
        self.confidence = confidence
        self.bbox = self.compute_bbox(segmentation)

    @staticmethod
    def compute_bbox(seg: List[float]) -> List[int]:
        """
        Convert a flat segmentation list into a bounding box.
        Returns [x1, y1, x2, y2].
        """
        if seg is None or len(seg) < 4:
            return [0, 0, 0, 0]
        xs = seg[0::2]
        ys = seg[1::2]
        return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


class AnnotationEntry:
    """
    Hold all annotations for one image.
    Includes path, width, height, and a list of AnnotationItem objects.
    """

    def __init__(
        self, image_path: Path, width: int, height: int, items: List[AnnotationItem]
    ):
        self.image_path = image_path
        self.width = width
        self.height = height
        self.items = items

    def to_mask(self) -> np.ndarray:
        """
        Build a binary mask from all polygons in this entry.
        """
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        for item in self.items:
            seg = item.segmentation
            if seg is None or len(seg) < 4:
                continue
            poly = np.array(seg, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [poly], 1)
        return mask


def load_json_annotations(json_path: Path) -> List[AnnotationEntry]:
    """
    Load annotation entries from a JSON file that contains images and annotations.
    Returns a list of AnnotationEntry objects.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Missing annotation file {json_path}")

    with open(json_path, "r", encoding="utf8") as f:
        data = json.load(f)

    entries = []

    for img in data.get("images", []):
        fname = img.get("file_name")
        width = int(img.get("width", 0))
        height = int(img.get("height", 0))
        anns = img.get("annotations", [])

        items = []
        for ann in anns:
            cls = ann.get("class", "unknown")
            seg = ann.get("segmentation", [])
            conf = float(ann.get("confidence_score", 1.0))
            items.append(AnnotationItem(cls, seg, conf))

        entry = AnnotationEntry(Path(fname), width, height, items)
        entries.append(entry)

    return entries


def get_unique_classes(entries: List[AnnotationEntry]) -> List[str]:
    """
    Return a sorted list of unique classes across all entries.
    """
    classes = set()
    for entry in entries:
        for item in entry.items:
            classes.add(item.cls)
    return sorted(classes)


# From C:\github\Tree-Canopy-Detection\src\data\augmentations.py
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


# Histogram Equalization (HE): A traditional method that redistributes pixel intensities based on the global histogram of the image. It often makes the whole image brighter or darker uniformly.
# Adaptive Histogram Equalization (AHE): Instead of one global histogram, the image is divided into smaller regions (tiles), and each region gets its own histogram equalization. This enhances local contrast.
# CLAHE (Contrast Limited AHE): Improves on AHE by limiting contrast amplification. This prevents noise in uniform areas (like sky or skin) from being exaggerated


# def get_train_augmentations(image_size: int = 256, mode: str = "rgb", num_channels: int = 3):
def get_train_augmentations(image_size: int = 256, mode: str = "rgb"):
    """
    Build augmentation pipeline for training.
    Includes flips, brightness changes, distortions, and resizing.
    """
    # Base transforms that work with any number of channels
    base_transforms = [
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]
    # Color transforms only work with RGB (3 channels)
    if mode in ["rgb", "filtered"]:
        color_transforms = [
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.3),
        ]
        base_transforms.extend(color_transforms)

    final_transforms = [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]

    # if num_channels == 3:
    #     # RGB mode - full augmentations
    #     color_transforms = [
    #         A.RandomBrightnessContrast(p=0.5),
    #         A.HueSaturationValue(p=0.3),
    #     ]
    #     base_transforms.extend(color_transforms)
    #
    #     # Standard ImageNet normalization
    #     final_transforms = [
    #         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #         ToTensorV2(),
    #     ]
    # elif num_channels == 6:
    #     # Concat mode - custom normalization
    #     # Normalize first 3 channels (RGB) with ImageNet stats
    #     # Normalize last 3 channels (filters) differently
    #     final_transforms = [
    #         CustomNormalize6Channel(),
    #         ToTensorV2(),
    #     ]
    # else:
    #     # Fallback
    #     final_transforms = [
    #         A.Normalize(mean=(0.5,) * num_channels, std=(0.5,) * num_channels),
    #         ToTensorV2(),
    #     ]

    base_transforms.extend(final_transforms)
    return A.Compose(base_transforms)


def get_val_augmentations(image_size: int = 256, mode: str = "rgb"):
    """
    Build validation and inference augmentation pipeline.
    """
    return A.Compose(
        [
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
            ),
            # A.CenterCrop(height=image_size, width=image_size),
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
    )


# From C:\github\Tree-Canopy-Detection\src\data\checkpoint.py
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


def extract_model_info_from_path(model_path: Path, config) -> Dict[str, str]:
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
        return extract_model_info_fallback(model_path, config)

    # Extract information from structured path
    try:
        # Start from after checkpoint directory
        remaining_parts = parts[checkpoint_idx + 1 :]

        if len(remaining_parts) < 3:  # Need at least model/mode/version
            return extract_model_info_fallback(model_path, config)

        model_name = remaining_parts[0]
        mode = remaining_parts[1]

        # Check if there's a size directory
        if len(remaining_parts) >= 4 and remaining_parts[2].startswith("size_"):
            image_size = remaining_parts[2].replace("size_", "")
            version = remaining_parts[3]
        else:
            # No size directory, extract from other clues
            version = remaining_parts[2]
            image_size = extract_size_from_context(model_path, config)

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
        return extract_model_info_fallback(model_path, config)


def extract_model_info_fallback(model_path: Path, config) -> Dict[str, str]:
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
        "segformer",
    ]

    # Method 1: Direct substring search
    for keyword in sorted(model_keywords, key=len, reverse=True):
        if keyword in path_str:
            model_name = keyword
            break

    # Method 2: Check path parts if Method 1 failed
    if model_name == "unknown":
        parts = model_path.parts
        for part in parts:
            part_lower = part.lower()
            if part_lower in model_keywords:
                model_name = part_lower
                break

    # Method 3: Pattern-based detection for edge cases
    if model_name == "unknown":
        if "smp_unet" in path_str or ("smp" in path_str and "unet" in path_str):
            model_name = "smp_unet"
        elif "simple_cnn" in path_str:
            model_name = "simple_cnn"
        elif "/unet/" in path_str and "smp" not in path_str:
            model_name = "unet"
        elif "yolov8" in path_str:
            if "yolov8n" in path_str:
                model_name = "yolov8n"
            elif "yolov8s" in path_str:
                model_name = "yolov8s"
            elif "yolov8m" in path_str:
                model_name = "yolov8m"
            elif "yolov8l" in path_str:
                model_name = "yolov8l"

    # Fallback - but avoid "unknown"
    if model_name == "unknown":
        model_name = "smp_unet"

    # Extract mode/input type
    mode = "rgb"
    if "filtered" in path_str:
        mode = "filtered"
    elif "concat" in path_str:
        mode = "concat"
    elif "gray" in path_str or "grey" in path_str:
        mode = "grayscale"

    # Extract image size
    image_size = extract_size_from_context(model_path, config)

    # Extract version info
    parts = model_path.parts
    version_info = []
    for part in parts:
        if part.startswith("v") and len(part) <= 6:
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


def extract_size_from_context(model_path: Path, config) -> str:
    """
    Extract image size from path context or filename.
    """
    path_str = str(model_path)
    image_size = "unknown"

    # Look for size_XXX pattern
    size_match = re.search(r"size_(\d+)", path_str)
    if size_match:
        image_size = size_match.group(1)
    else:
        # Look for other patterns
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
                image_size = match.group(1)
                break

    if image_size == "unknown":
        try:
            image_size = str(config.train.image_size)
        except:
            image_size = "512"

    return image_size


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
                model_info = extract_model_info_from_path(model_file, config)

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


def get_correct_model_name_from_path(model_path):
    """Fixed model name detection specifically for your case"""
    from src.models.zoo import MODEL_BUILDERS

    path_str = str(model_path).lower()
    known_models = MODEL_BUILDERS.keys()

    # Check for exact matches first (prioritize longer names)
    for model_name in sorted(known_models, key=len, reverse=True):
        if model_name in path_str:
            return model_name

    # Manual fixes for your specific case
    if "/smp_unet" in path_str:
        return "smp_unet"
    elif "smp" in path_str and "unet" in path_str:
        return "smp_unet"
    elif "simple_cnn" in path_str:
        return "simple_cnn"
    elif "unet" in path_str:
        return "unet"  # Only if no smp prefix

    # Fallback
    return "smp_unet"


def generate_submission_for_model(model_path: Path, config) -> Optional[Path]:
    """
    Generate a submission file for a given model checkpoint (.pth).
    """
    model_name = "smp_unet"

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

        try:
            model_info = extract_model_info_fallback(model_path, config)
            model_name = model_info.get("model_name", "smp_unet")

            # Verify the model name is valid
            if model_name not in MODEL_BUILDERS:
                print(
                    f"Warning: {model_name} not found in MODEL_BUILDERS, using smp_unet"
                )
                model_name = "smp_unet"

            p(f"Using model name: {model_name} for path: {model_path}")
        except Exception as e:
            p(f"Error extracting model info: {e}, using default: {model_name}")

        # Get image size using fallback logic
        image_size_str = (
            model_info.get("image_size", "unknown")
            if "model_info" in locals()
            else "unknown"
        )

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

        t(f"Generating submission for: {model_path.name}")

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
        # Filter out invalid model names (pure numbers like "03", "07")
        valid_df_filtered = valid_df[
            valid_df["Model Name"].apply(lambda x: str(x).isalpha() or "_" in str(x))
        ]

        model_performance = valid_df_filtered.groupby("Model Name")["Val Loss"].agg(
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


def fix_models_list_before_analysis(models_list):
    """
    Call this function to fix model names in your models_list before creating visualizations.
    """
    for model in models_list:
        current_name = model.get("model_name", "unknown")

        # If current name is a directory number or unknown, extract real model name
        if current_name in ["03", "07", "10", "11", "unknown"]:
            file_path = model.get("file_path", "")

            # Use our fixed extraction function
            fixed_info = extract_model_info_fallback(Path(file_path))
            model["model_name"] = fixed_info["model_name"]

            # Also update other fields that might be wrong
            if model.get("mode", "unknown") == "unknown":
                model["mode"] = fixed_info["mode"]
            if model.get("image_size", "unknown") == "unknown":
                model["image_size"] = fixed_info["image_size"]

    return models_list


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
    models_list = fix_models_list_before_analysis(models_list)
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


# From C:\github\Tree-Canopy-Detection\src\data\CustomNormalize6Channel.py
class CustomNormalize6Channel:
    """Custom normalization for 6-channel images (RGB + 3 filters)."""

    def __init__(self):
        # ImageNet stats for RGB channels
        self.rgb_mean = [0.485, 0.456, 0.406]
        self.rgb_std = [0.229, 0.224, 0.225]

        # Custom stats for filter channels (you might want to compute these)
        self.filter_mean = [0.5, 0.5, 0.5]
        self.filter_std = [0.5, 0.5, 0.5]

    def __call__(self, image, **kwargs):
        if len(image.shape) == 3 and image.shape[2] == 6:
            # Normalize RGB channels (0:3)
            for i in range(3):
                image[:, :, i] = (
                    image[:, :, i] / 255.0 - self.rgb_mean[i]
                ) / self.rgb_std[i]

            # Normalize filter channels (3:6)
            for i in range(3, 6):
                image[:, :, i] = (
                    image[:, :, i] / 255.0 - self.filter_mean[i - 3]
                ) / self.filter_std[i - 3]

        return {"image": image}


# From C:\github\Tree-Canopy-Detection\src\data\enhance_masks.py
import albumentations as A
import cv2
import numpy as np
import torch
from albumentations import ToTensorV2

from src.data.loaders import ImageMaskDataset
from src.data.masks import build_multiclass_mask


class EnhancedImageMaskDataset(ImageMaskDataset):
    """
    Extended dataset that applies filter enhancements during loading.

    Can operate in 3 modes:
    1. 'rgb' - Original 3-channel RGB
    2. 'filtered' - Top 3 filters as RGB channels
    3. 'concat' - 6-channel (RGB + 3 filters)
    """

    def __init__(
        self,
        entries,
        image_dir,
        mode="rgb",
        filter_names=None,
        classes=None,
        transform=None,
    ):
        super().__init__(entries, image_dir, classes, transform)
        self.mode = mode
        self.filter_names = filter_names or ["laplacian", "sobel", "clahe"]

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        img_path = self.image_dir / entry.image_path.name

        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Failed to read {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Build mask
        if self.classes is None:
            mask = build_multiclass_mask(entry)
        else:
            from src.data.annotations import AnnotationEntry

            filtered_items = [item for item in entry.items if item.cls in self.classes]
            filtered_entry = AnnotationEntry(
                entry.image_path, entry.width, entry.height, filtered_items
            )
            mask = build_multiclass_mask(filtered_entry)

        # Apply filters BEFORE transforms to avoid double normalization
        if self.mode == "filtered":
            image = self.apply_filters_to_enhanced_image(image)
        elif self.mode == "concat":
            filtered_img = self.apply_filters_to_enhanced_image(image)
            image = np.concatenate([image, filtered_img], axis=2)

        # Apply transforms with mode awareness
        if self.transform:
            # Check if transform function accepts mode parameter
            try:
                if hasattr(self.transform, "transforms"):
                    # It's an Albumentations compose - apply directly
                    processed = self.transform(image=image, mask=mask)
                else:
                    # It's our custom function - this shouldn't happen with current code
                    processed = self.transform(image=image, mask=mask)
            except TypeError as e:
                if "HueSaturationValue" in str(e):
                    print(f"Skipping problematic transform for {self.mode} mode")
                    # Apply minimal transforms only
                    minimal_transform = A.Compose(
                        [
                            A.Resize(256, 256),
                            A.Normalize(
                                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                            ),
                            ToTensorV2(),
                        ]
                    )
                    processed = minimal_transform(image=image, mask=mask)
                else:
                    raise e

            image = processed["image"]
            mask = processed["mask"]

            # Normalize mask to class indices 0,1,2
            if mask.max() > 2:
                mask = (mask / 255).astype(np.uint8)

        # Safety check: Convert image to tensor if not already done
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        else:
            # Ensure proper format if already tensor
            if image.ndim == 3 and image.shape[0] not in [3, 6]:  # Not CHW format
                image = image.permute(2, 0, 1)
            # Only normalize if not already normalized by Albumentations
            min_val = image.min().item()
            max_val = image.max().item()
            is_already_normalized = (min_val >= -5.0 and max_val <= 5.0) or (
                min_val < 0 and max_val <= 10.0
            )
            if not is_already_normalized and image.max() > 1.0:
                image = image / 255.0

        # Convert mask to tensor
        if isinstance(mask, torch.Tensor):
            mask_t = mask.long()
        else:
            mask_t = torch.from_numpy(mask).long()

        while mask_t.ndim > 2:
            mask_t = mask_t.squeeze(0)

        assert mask_t.ndim == 2, f"Mask should be 2D [H, W], got {mask_t.shape}"

        return image, mask_t

    def apply_filters_to_enhanced_image(self, img):
        """
        Apply specified filters and return as 3-channel image.
        """
        from src.data.image_loader import create_enhanced_image

        return create_enhanced_image(img, self.filter_names)

    @classmethod
    def get_available_filters(cls):
        """
        Get list of all available filter names.
        """
        from src.exploration.kernels import get_kernels

        # Get kernel-based filters
        kernel_bank = get_kernels("all")
        kernel_names = [name.lower() for name in kernel_bank.keys()]

        # Algorithmic filters (always available)
        algorithmic = [
            "laplacian",
            "sobel",
            "clahe",
            "gaussian_3x3",
            "gaussian_5x5",
            "gaussian_7x7",
            "gaussian_9x9",
            "sobel_x_cv",
            "sobel_y_cv",
            "canny",
            "histogram_eq",
            "bilateral",
            "median_3x3",
            "median_5x5",
        ]

        return sorted(set(kernel_names + algorithmic))


# From C:\github\Tree-Canopy-Detection\src\data\image_loader.py
"""
Image loading utilities that handle multiple formats.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.data.annotations import AnnotationEntry
from src.utils.helpers import c, p


def load_image(image_dir: Path, entry: AnnotationEntry) -> np.ndarray:
    """
    Load an image with automatic format handling and fallback for TIFFs.
    Tries OpenCV first (fastest), falls back to PIL for problematic TIFFs.
    Automatically uses PNG version if it exists alongside TIFF.
    """
    image_path = Path(image_dir / entry.image_path.name)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Check for PNG version first (more reliable than TIFF)
    png_path = image_path.with_suffix(".png")
    if png_path.exists() and png_path != image_path:
        image_path = png_path

    # Try OpenCV first (fastest)
    img = cv2.imread(str(image_path))

    if img is not None:
        # OpenCV loads as BGR, convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    # Fall back to PIL (better TIFF support)
    try:
        pil_img = Image.open(image_path)
        img = np.array(pil_img)

        # Ensure RGB format
        if img.ndim == 2:  # Grayscale
            img = np.stack([img, img, img], axis=2)
        elif img.shape[2] == 4:  # RGBA
            img = img[:, :, :3]

        return img

    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}: {str(e)}")


def validate_image_directory(image_dir: Path) -> dict:
    """
    Validate all images in a directory can be loaded.
    """
    from src.utils.helpers import p

    image_dir = Path(image_dir)

    # Find all image files
    image_files = []
    for ext in ["*.tif"]:
        # for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
        image_files.extend(image_dir.glob(ext))

    results = {
        "total": len(image_files),
        "valid": 0,
        "invalid": 0,
        "problematic_files": [],
    }

    p("Validating images", f"{len(image_files)} files", color1=c.BLACK)

    for img_path in image_files:
        try:
            # Quick validation using PIL
            pil_img = Image.open(img_path)
            pil_img.verify()
            results["valid"] += 1
        except Exception:
            results["invalid"] += 1
            results["problematic_files"].append(str(img_path))

    p("Valid images", results["valid"], color1=c.BLACK)
    p("Invalid images", results["invalid"], color1=c.BLACK)

    if results["problematic_files"]:
        p("Problematic files", "")
        for path in results["problematic_files"][:10]:
            p("", f"  {path}", color1=c.SALMON)

    return results


def apply_all_filters(img):
    """
    Apply all available filters to an image and return dict of results.
    Uses unified registry from kernels + algorithmic filters.
    """
    from src.exploration.kernels import get_kernels, apply_kernel_using_convolution
    from src.exploration.enhancement import to_gray, clahe_enhance

    gray = to_gray(img)
    target_h, target_w = img.shape[:2]

    kernel_bank = get_kernels("all")

    # Create unified filter registry
    filter_registry = {}

    # Add kernel-based filters
    for kname, kernel in kernel_bank.items():
        filter_registry[kname.lower()] = (
            lambda k=kernel: apply_kernel_using_convolution(gray, k)
        )

    # Add algorithmic filters
    filter_registry.update(
        {
            "laplacian": lambda: cv2.Laplacian(gray, cv2.CV_64F),
            "sobel": lambda: cv2.Sobel(gray, cv2.CV_64F, 1, 0)
            + cv2.Sobel(gray, cv2.CV_64F, 0, 1),
            "clahe": lambda: clahe_enhance(gray, clip=2.0, tile=8),
            "gaussian_3x3": lambda: cv2.GaussianBlur(gray, (3, 3), 1.0),
            "gaussian_5x5": lambda: cv2.GaussianBlur(gray, (5, 5), 1.5),
            "gaussian_7x7": lambda: cv2.GaussianBlur(gray, (7, 7), 2.0),
        }
    )

    # Apply all filters
    results = {}
    for fname, filter_func in filter_registry.items():
        try:
            filtered = filter_func()

            # Normalize
            if filtered.ndim == 3:
                filtered = cv2.cvtColor(filtered, cv2.COLOR_RGB2GRAY)
            if filtered.shape != (target_h, target_w):
                filtered = cv2.resize(
                    filtered, (target_w, target_h), interpolation=cv2.INTER_LINEAR
                )
            if filtered.dtype != np.uint8:
                filtered = cv2.normalize(
                    filtered, None, 0, 255, cv2.NORM_MINMAX
                ).astype(np.uint8)

            results[fname] = filtered
        except Exception as e:
            p("Warning", f"Filter '{fname}' failed: {e}", color1=c.ORANGE)
            continue

    return results


# def apply_filters(self, img):
#     """
#     Apply specified filters and return as 3-channel image.
#     Handles both kernel-based and algorithmic filters.
#     """
#     from src.exploration.kernels import get_kernels, apply_kernel_using_convolution
#     from src.exploration.enhancement import to_gray, clahe_enhance
#     import cv2
#     import numpy as np
#
#     # Convert to grayscale for filter application
#     gray = to_gray(img)
#     target_h, target_w = img.shape[:2]
#
#     # Load kernel bank dynamically
#     kernel_bank = get_kernels("all")
#
#     # Create unified filter registry
#     filter_registry = {}
#
#     # Add kernel-based filters
#     for kname, kernel in kernel_bank.items():
#         # Use closure to capture kernel value
#         filter_registry[kname.lower()] = (
#             lambda k=kernel: apply_kernel_using_convolution(gray, k)
#         )
#
#     # Add algorithmic filters
#     filter_registry.update(
#         {
#             "laplacian": lambda: cv2.Laplacian(gray, cv2.CV_64F),
#             "sobel": lambda: cv2.Sobel(gray, cv2.CV_64F, 1, 0)
#             + cv2.Sobel(gray, cv2.CV_64F, 0, 1),
#             "clahe": lambda: clahe_enhance(gray, clip=2.0, tile=8),
#             "gaussian_3x3": lambda: cv2.GaussianBlur(gray, (3, 3), 1.0),
#             "gaussian_5x5": lambda: cv2.GaussianBlur(gray, (5, 5), 1.5),
#             "gaussian_7x7": lambda: cv2.GaussianBlur(gray, (7, 7), 2.0),
#         }
#     )
#
#     # Apply requested filters
#     channels = []
#     for fname in self.filter_names:
#         key = fname.lower()
#
#         if key not in filter_registry:
#             available = sorted(filter_registry.keys())
#             raise KeyError(
#                 f"Unknown filter '{fname}'. "
#                 f"Available filters ({len(available)}): {available[:10]}..."
#             )
#
#         try:
#             # Apply filter
#             filtered = filter_registry[key]()
#
#             # Ensure 2D (grayscale)
#             if filtered.ndim == 3:
#                 filtered = cv2.cvtColor(filtered, cv2.COLOR_RGB2GRAY)
#
#             # Resize if needed
#             if filtered.shape != (target_h, target_w):
#                 filtered = cv2.resize(
#                     filtered, (target_w, target_h), interpolation=cv2.INTER_LINEAR
#                 )
#
#             # Normalize to uint8
#             if filtered.dtype != np.uint8:
#                 filtered = cv2.normalize(
#                     filtered, None, 0, 255, cv2.NORM_MINMAX
#                 ).astype(np.uint8)
#
#             channels.append(filtered)
#
#         except Exception as e:
#             raise RuntimeError(f"Filter '{fname}' failed: {e}")
#
#     # Ensure we have at least 3 channels
#     if len(channels) == 0:
#         raise RuntimeError(f"No filters produced output for {self.filter_names}")
#
#     while len(channels) < 3:
#         channels.append(channels[-1].copy())
#
#     # Stack into 3-channel image
#     result = np.stack(channels[:3], axis=2)
#
#     return result


def create_enhanced_image(img, filter_names):
    """
    Create multi-channel enhanced image using specified filters.
    Returns 3-channel image suitable for model input.
    """
    filters_dict = apply_all_filters(img)

    # Get target shape from original image
    target_h, target_w = img.shape[:2]

    channels = []
    for fname in filter_names[:3]:  # Take up to 3 filters
        filtered = filters_dict[fname]

        # Ensure it's 2D (grayscale)
        if filtered.ndim == 3:
            filtered = cv2.cvtColor(filtered, cv2.COLOR_RGB2GRAY)

        # Resize to match target dimensions if needed
        if filtered.shape != (target_h, target_w):
            filtered = cv2.resize(
                filtered, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )

        # Normalize to 0-255
        if filtered.dtype != np.uint8:
            filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(
                np.uint8
            )

        channels.append(filtered)

    # If we have fewer than 3 filters, pad with the last one
    while len(channels) < 3:
        channels.append(channels[-1].copy())

    # Verify all channels have same shape
    # shapes = [ch.shape for ch in channels[:3]]
    # if len(set(shapes)) != 1:
    #     p("Warning", f"Channel shape mismatch: {shapes}")
    #     # Force resize all to target
    #     channels = [cv2.resize(ch, (target_w, target_h)) if ch.shape != (target_h, target_w) else ch
    #                 for ch in channels[:3]]

    # Stack into 3-channel image
    enhanced = np.stack(channels[:3], axis=2)
    return enhanced


# From C:\github\Tree-Canopy-Detection\src\data\loaders.py
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.annotations import AnnotationEntry
from src.data.masks import build_multiclass_mask


class ImageMaskDataset(Dataset):
    """
    Dataset that returns image and mask pairs.
    Expects a list of AnnotationEntry objects and a directory with images.

    ImageMaskDataset:

    • loads image
    • generates mask polygons
    • applies transforms
    • handles numpy vs tensor images
    • handles numpy vs tensor masks
    • ensures output shapes:

    image: (3, H, W)

    mask: (1, H, W)

    Neded for segmentation training.
    """

    def __init__(
        self,
        entries: List[AnnotationEntry],
        image_dir: Path,
        classes: Optional[List[str]] = None,
        transform=None,
    ):
        self.entries = entries
        self.image_dir = Path(image_dir)
        self.classes = classes
        self.transform = transform

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        entry = self.entries[idx]
        img_path = self.image_dir / entry.image_path.name

        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Failed to read {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = build_multiclass_mask(entry)

        if self.transform:
            processed = self.transform(image=image, mask=mask)
            image = processed["image"]
            mask = processed["mask"]

            # Ensure mask values are proper class indices (0,1,2)
            if mask.max() > 2:
                mask = (mask / 255).astype(np.uint8)

        # Safety check: Convert image to tensor if not already done
        if not isinstance(image, torch.Tensor):
            img_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        else:
            img_t = image.float()
            if img_t.ndim == 3 and img_t.shape[0] != 3:
                img_t = img_t.permute(2, 0, 1)
            # Only normalize if not already normalized by Albumentations
            min_val = img_t.min().item()
            max_val = img_t.max().item()
            is_already_normalized = (min_val >= -5.0 and max_val <= 5.0) or (
                min_val < 0 and max_val <= 10.0
            )
            if not is_already_normalized and img_t.max() > 1.0:
                img_t = img_t / 255.0

        # Convert mask to tensor
        if isinstance(mask, torch.Tensor):
            mask_t = mask.long()
        else:
            mask_t = torch.from_numpy(mask).long()

        while mask_t.ndim > 2:
            mask_t = mask_t.squeeze(0)

        assert mask_t.ndim == 2, f"Mask should be 2D [H, W], got {mask_t.shape}"
        return img_t, mask_t


class ImageOnlyDataset(Dataset):
    """
    Dataset for inference. Returns image tensors only.

    ImageOnlyDataset:

    • load image
    • apply transforms
    • safely convert to CHW tensor
    • return image name + tensor

    Supports:
    • PIL Image
    • NumPy array
    • Torch tensor
    • File paths

    Used only for inference
    """

    def __init__(
        self,
        image_dir: Path,
        transform=None,
    ):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.files = sorted(
            [f for f in self.image_dir.glob("*.*") if f.suffix.lower() in [".png"]],
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        path = self.files[idx]
        image = self._load_image(path)

        if self.transform:
            processed = self.transform(image=image)
            image = processed["image"]

        # Safety check: Convert image to tensor if not already done
        if not isinstance(image, torch.Tensor):
            img_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        else:
            img_t = image.float()
            if img_t.ndim == 3 and img_t.shape[0] != 3:
                img_t = img_t.permute(2, 0, 1)
            # Only normalize if not already normalized by Albumentations
            min_val = img_t.min().item()
            max_val = img_t.max().item()
            is_already_normalized = (min_val >= -5.0 and max_val <= 5.0) or (
                min_val < 0 and max_val <= 10.0
            )
            if not is_already_normalized and img_t.max() > 1.0:
                img_t = img_t / 255.0

        return path.name, img_t

    def _load_image(
        self, source: Union[str, Path, np.ndarray, torch.Tensor, Image.Image]
    ) -> np.ndarray:
        if isinstance(source, np.ndarray):
            img = source
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=2)
            return img

        if isinstance(source, torch.Tensor):
            arr = source.cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] == 3:
                arr = arr.transpose(1, 2, 0)
            return arr

        if isinstance(source, Image.Image):
            return np.array(source)

        path = str(source)
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Failed to read {source}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


# From C:\github\Tree-Canopy-Detection\src\data\masks.py
from pathlib import Path
from typing import List

import cv2
import numpy as np


# Class mapping for Solafune competition
CLASS_TO_ID = {
    "individual_tree": 1,
    "group_of_trees": 2,
}

ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}


def build_binary_mask(segmentation: List[float], width: int, height: int) -> np.ndarray:
    """
    Convert one segmentation polygon into a binary mask.
    segmentation is a flat list of coordinates.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    if segmentation is None or len(segmentation) < 4:
        return mask
    poly = np.array(segmentation, dtype=np.int32).reshape(-1, 2)
    cv2.fillPoly(mask, [poly], 1)
    return mask


def build_multiclass_mask(entry, class_to_id: dict = None) -> np.ndarray:
    """
    Build mask with class indices for multi-class segmentation.
    Background=0, individual_tree=1, group_of_trees=2
    """
    if class_to_id is None:
        class_to_id = CLASS_TO_ID

    mask = np.zeros((entry.height, entry.width), dtype=np.uint8)

    for item in entry.items:
        seg = item.segmentation
        if seg is None or len(seg) < 6:
            continue

        # Get class ID, skip unknown classes (don't assign to background)
        class_id = class_to_id.get(item.cls, None)
        if class_id is None:
            # Skip unknown classes instead of assigning to background
            continue

        poly = np.array(seg, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [poly], class_id)

    return mask


def save_mask(mask: np.ndarray, path: Path) -> None:
    """
    Save a binary mask. Values are written as 0 or 255.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = (mask * 255).astype(np.uint8)
    cv2.imwrite(str(path), out)


def load_mask(path: Path) -> np.ndarray:
    """
    Load a binary mask from disk. Converts 255 to 1.
    """
    path = Path(path)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Missing mask file {path}")
    return (img > 127).astype(np.uint8)


def mask_to_overlay(
    image: np.ndarray, mask: np.ndarray, alpha: float = 0.4
) -> np.ndarray:
    """
    Overlay a binary mask on an RGB image. Mask is shown in red.
    """
    if image.max() <= 1.0:
        img_u8 = (image * 255).astype(np.uint8)
    else:
        img_u8 = image.astype(np.uint8)

    mask_u8 = (mask * 255).astype(np.uint8)
    mask_rgb = np.zeros_like(img_u8)
    mask_rgb[:, :, 0] = mask_u8

    overlay = cv2.addWeighted(img_u8, 1 - alpha, mask_rgb, alpha, 0)
    return overlay


# From C:\github\Tree-Canopy-Detection\src\data\__init__.py


