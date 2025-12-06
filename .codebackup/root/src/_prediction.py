# From C:\github\Tree-Canopy-Detection\src\prediction\pipeline.py
from pathlib import Path

import cv2
import numpy as np
import torch

from src.models.zoo import build_model
from src.utils.logging import Logger


class Predictor:
    """
    Full inference pipeline for tree canopy prediction.
    Loads a trained model, applies preprocessing, runs inference,
    and returns refined masks and overlays.
    """

    def __init__(
        self, model_path: Path, model_name: str = "unet", image_size: int = 256
    ):
        self.model_path = Path(model_path)
        self.model_name = model_name
        self.image_size = image_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = Logger()

        self.model = build_model(model_name, in_channels=3, out_channels=3).to(
            self.device
        )
        self._load_weights()

    def _load_weights(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Missing model weights {self.model_path}")
        state = torch.load(self.model_path, map_location=self.device)
        if "model" in state:
            self.model.load_state_dict(state["model"])
        else:
            self.model.load_state_dict(state)
        self.model.eval()
        self.logger.info(f"Loaded model weights from {self.model_path}")

    def predict_tensor(self, tensor: torch.Tensor) -> np.ndarray:
        """Run inference on a single tensor. Returns a numpy mask."""
        tensor = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(tensor)

        # Handle multi-class output
        if pred.shape[1] == 3:  # Multi-class
            pred_classes = torch.argmax(pred, dim=1).cpu().squeeze().numpy()
            # Return binary mask (any tree class > 0)
            return (pred_classes > 0).astype(np.uint8)
        else:  # Binary
            pred = torch.sigmoid(pred).cpu().squeeze().numpy()
            return (pred > 0.5).astype(np.uint8)

    def _load_weights(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Missing model weights {self.model_path}")
        state = torch.load(self.model_path, map_location=self.device)
        if "model" in state:
            self.model.load_state_dict(state["model"])
        else:
            self.model.load_state_dict(state)
        self.model.eval()
        self.logger.info(f"Loaded weights: {self.model_path}")

    def predict_sliding_window(
        self, image: np.ndarray, tile_size: int, overlap: float = 0.25
    ) -> np.ndarray:
        """
        Perform inference using sliding window to preserve resolution.
        """
        h, w = image.shape[:2]
        stride = int(tile_size * (1 - overlap))

        # Initialize outputs
        # shape: [3, H, W] for 3 classes
        prob_map = torch.zeros((3, h, w), device=self.device)
        count_map = torch.zeros((1, h, w), device=self.device)

        # Preprocessing (Normalization matches training)
        # Manually normalize since we aren't using Albumentations pipeline here easily
        img_norm = image.astype(np.float32) / 255.0
        img_norm = (img_norm - np.array([0.485, 0.456, 0.406])) / np.array(
            [0.229, 0.224, 0.225]
        )
        tensor_img = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()  # [C, H, W]

        # Sliding window
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y2 = min(h, y + tile_size)
                x2 = min(w, x + tile_size)
                y1 = max(0, y2 - tile_size)
                x1 = max(0, x2 - tile_size)

                crop = tensor_img[:, y1:y2, x1:x2].unsqueeze(0).to(self.device)

                with torch.no_grad():
                    # Get raw logits
                    output = self.model(crop)
                    # Apply softmax to get probabilities
                    output = torch.nn.functional.softmax(output, dim=1)

                prob_map[:, y1:y2, x1:x2] += output.squeeze(0)
                count_map[:, y1:y2, x1:x2] += 1.0

        # Average results
        prob_map /= count_map

        # Argmax to get class indices
        pred_mask = torch.argmax(prob_map, dim=0).cpu().numpy().astype(np.uint8)

        return pred_mask

    def predict_image(self, image: np.ndarray) -> np.ndarray:
        """
        Run inference directly on an RGB numpy image.
        Resizing and conversion done here.
        """
        img = cv2.resize(image, (self.image_size, self.image_size))
        img_t = torch.tensor(img.transpose(2, 0, 1)).float() / 255.0
        return self.predict_tensor(img_t)

    def _resize_prediction_to_original(self, pred_mask, original_shape):
        """
        Resize prediction mask back to original image dimensions.
        """
        orig_h, orig_w = original_shape[:2]
        if pred_mask.shape != (orig_h, orig_w):
            pred_mask = cv2.resize(
                pred_mask.astype(np.uint8),
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )
        return pred_mask

    def run_on_folder(self, image_dir: Path, transform=None, num_samples: int = None):
        """
        Run prediction on a folder using sliding window.
        """
        image_files = sorted(
            list(image_dir.glob("*.png")) + list(image_dir.glob("*.tif"))
        )
        results = []

        total = (
            len(image_files)
            if num_samples is None
            else min(num_samples, len(image_files))
        )

        for idx in range(total):
            img_path = image_files[idx]

            # Load RGB
            original_img = cv2.imread(str(img_path))
            if original_img is None:
                continue
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

            # Predict using sliding window (tile_size matches training size)
            pred_mask = self.predict_sliding_window(
                original_img, tile_size=self.image_size
            )

            # Create overlay
            mask_rgb = np.zeros_like(original_img)
            mask_rgb[pred_mask == 1] = [0, 255, 0]  # Individual = Green
            mask_rgb[pred_mask == 2] = [255, 255, 0]  # Group = Yellow

            overlay = cv2.addWeighted(original_img, 0.7, mask_rgb, 0.3, 0)

            results.append(
                {
                    "name": img_path.name,
                    "image": original_img,
                    "mask": pred_mask,
                    "overlay": overlay,
                }
            )

            if (idx + 1) % 5 == 0:
                print(f"Processed {idx + 1}/{total} images")

        return results

    # def run_on_folder(self, image_dir: Path, transform=None, num_samples: int = None):
    #     """
    #     Run prediction on a folder of images using ImageOnlyDataset.
    #     Returns list of dictionaries with masks and overlays.
    #     """
    #     dataset = ImageOnlyDataset(image_dir, transform=transform)
    #     results = []
    #
    #     total = len(dataset) if num_samples is None else min(num_samples, len(dataset))
    #
    #     for idx in range(total):
    #         name, img_t = dataset[idx]
    #
    #         # Load original image to get true dimensions
    #         img_path = image_dir / name
    #         original_img = cv2.imread(str(img_path))
    #         if original_img is None:
    #             # Try with different extensions
    #             for ext in [".png", ".tiff"]:
    #                 alt_path = img_path.with_suffix(ext)
    #                 if alt_path.exists():
    #                     original_img = cv2.imread(str(alt_path))
    #                     break
    #
    #         if original_img is not None:
    #             original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    #             original_shape = original_img.shape
    #         else:
    #             # Fallback to processed image shape
    #             if isinstance(img_t, torch.Tensor):
    #                 original_shape = img_t.permute(1, 2, 0).shape
    #             else:
    #                 original_shape = img_t.shape
    #             original_img = (
    #                 img_t
    #                 if not isinstance(img_t, torch.Tensor)
    #                 else img_t.permute(1, 2, 0).cpu().numpy()
    #             )
    #
    #         # Convert HWC -> CHW safely for both numpy and torch
    #         if isinstance(img_t, torch.Tensor):
    #             img_chw = img_t.unsqueeze(0).float()
    #             base = img_t.permute(1, 2, 0).cpu().numpy()
    #         else:
    #             img_chw = (
    #                 torch.from_numpy(img_t.transpose(2, 0, 1)).unsqueeze(0).float()
    #             )
    #             base = img_t
    #
    #         with torch.no_grad():
    #             pred = self.model(img_chw.to(self.device)).cpu()
    #
    #         # Convert multi-class logits to class predictions
    #         if pred.shape[1] == 3:  # Multi-class
    #             pred_classes = torch.argmax(pred, dim=1).squeeze().numpy()  # [H, W]
    #             pred_bin_for_overlay = (pred_classes > 0).astype(np.uint8)
    #         else:  # Binary
    #             pred = torch.sigmoid(pred).squeeze().numpy()
    #             pred_classes = (pred > 0.5).astype(np.uint8)
    #             pred_bin_for_overlay = pred_classes
    #
    #         # Resize predictions to match original image dimensions
    #         pred_classes_resized = self._resize_prediction_to_original(
    #             pred_classes, original_shape
    #         )
    #         pred_bin_resized = self._resize_prediction_to_original(
    #             pred_bin_for_overlay, original_shape
    #         )
    #
    #         # Create overlay with original image dimensions
    #         pred_u8 = pred_bin_resized * 255
    #         overlay = np.zeros_like(original_img)
    #         overlay = overlay.copy()
    #         overlay[:, :, 0] = pred_u8
    #         overlay = cv2.addWeighted(
    #             original_img.astype(np.uint8), 0.6, overlay.astype(np.uint8), 0.4, 0
    #         )
    #
    #         results.append(
    #             {
    #                 "name": name,
    #                 "image": original_img,  # Use original size image
    #                 "mask": pred_classes_resized,  # Use resized mask
    #                 "overlay": overlay,  # Use properly sized overlay
    #             }
    #         )
    #
    #     return results


# From C:\github\Tree-Canopy-Detection\src\prediction\postprocess.py
import cv2
import numpy as np


def refine_mask(mask: np.ndarray, min_area: int = 20) -> np.ndarray:
    """
    Clean small artifacts in a binary mask.
    Removes connected components smaller than min_area.
    """
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 1
    return cleaned


def group_mask_threshold(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Apply a direct threshold to a probability mask.
    """
    return (mask > threshold).astype(np.uint8)


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Create a red overlay of the mask on top of an RGB image.
    """
    if image.max() <= 1.0:
        img_u8 = (image * 255).astype(np.uint8)
    else:
        img_u8 = image.astype(np.uint8)

    mask_u8 = (mask * 255).astype(np.uint8)

    mask_rgb = np.zeros_like(img_u8)
    mask_rgb[:, :, 0] = mask_u8

    return cv2.addWeighted(img_u8, 1 - alpha, mask_rgb, alpha, 0)


# From C:\github\Tree-Canopy-Detection\src\prediction\submission.py
import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from src.utils.helpers import p, t


def mask_to_polygons_multiclass(
    mask: np.ndarray, id_to_class: dict = None
) -> List[dict]:
    """
    Convert a multi-class mask to list of annotation dicts with correct class names.

    Expected mask values:
        0 = background
        1 = individual_tree
        2 = group_of_trees
    """
    if id_to_class is None:
        id_to_class = {1: "individual_tree", 2: "group_of_trees"}

    annotations = []

    # Process each tree class (skip background = 0)
    for class_id in [1, 2]:
        class_name = id_to_class.get(class_id, f"class_{class_id}")

        # Extract binary mask for this class
        binary_mask = (mask == class_id).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            # Filter small artifacts
            if cv2.contourArea(cnt) < 25:  # Min 25 pixels
                continue

            if len(cnt) >= 3:
                # Flatten: [[x1,y1], [x2,y2]] -> [x1,y1,x2,y2]
                segmentation = cnt.reshape(-1).tolist()

                if len(segmentation) >= 6:  # At least 3 points
                    annotations.append(
                        {
                            "class": class_name,
                            "confidence_score": 1.0,
                            "segmentation": segmentation,
                        }
                    )

    return annotations


def export_submission(
    results: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Convert prediction results into expected submission JSON structure.
    Uses sample_answer.json as template to preserve cm_resolution and scene_type.

    IMPORTANT: Per admin guidance, only annotations should be modified.
    cm_resolution and scene_type come from the template.
    """
    # Load template - use raw string (r"...") for Windows paths
    template_path = Path(
        "/content/drive/MyDrive/TreeCanopyProject/data/data1/sample_answer.json"
    )
    # template_path = Path(
    #     r"C:\github\Tree-Canopy-Detection\src\data\data1\sample_answer.json"
    # )

    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    # Load template
    with open(template_path, "r", encoding="utf8") as f:
        submission = json.load(f)

    # Build lookup from results by filename
    results_lookup = {}
    for r in results:
        fname = r.get("name", "")
        # Normalize to .tif extension
        fname_key = Path(fname).stem + ".tif"
        results_lookup[fname_key] = r

    # Update only the annotations in template
    for img_entry in submission.get("images", []):
        fname = img_entry.get("file_name", "")

        if fname in results_lookup:
            r = results_lookup[fname]
            mask = r.get("mask", None)

            if mask is not None:
                annotations = mask_to_polygons_multiclass(mask)
            else:
                annotations = []

            # Replace ONLY annotations, keep cm_resolution and scene_type from template
            img_entry["annotations"] = annotations
        else:
            # Image not in predictions - clear annotations
            img_entry["annotations"] = []

    # Save JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf8") as f:
        json.dump(submission, f, indent=4)

    # Logging
    t("Submission Export Complete")
    p("✓ Submission saved", output_path)
    p("✓ Total images", len(submission.get("images", [])))
    p(
        "✓ Total annotations",
        sum(len(img["annotations"]) for img in submission.get("images", [])),
    )

    # Count by class
    individual_count = sum(
        1
        for img in submission.get("images", [])
        for ann in img.get("annotations", [])
        if ann.get("class") == "individual_tree"
    )
    group_count = sum(
        1
        for img in submission.get("images", [])
        for ann in img.get("annotations", [])
        if ann.get("class") == "group_of_trees"
    )
    p("✓ individual_tree annotations", individual_count)
    p("✓ group_of_trees annotations", group_count)


def extract_cm_resolution(fname: str) -> int:
    """
    Extract resolution in cm from filenames like 'forest_10cm_001.tif'
    Returns integer resolution (eg: 10).
    """
    name = Path(fname).stem.lower()

    # Search for patterns like 5cm, 10cm, 20cm, etc.
    import re

    match = re.search(r"(\d+)\s*cm", name)
    if match:
        return int(match.group(1))

    # If no information found, return fallback
    return 10


# From C:\github\Tree-Canopy-Detection\src\prediction\validation.py
"""
Submission validation utilities for Tree Canopy Detection
Validates submission JSON format according to competition requirements
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import torch

from src.utils.helpers import c, p, t


def validate_submission_format(submission_path: Path) -> Dict[str, Any]:
    """
    Validate submission matches required format.

    Args:
        submission_path: Path to submission JSON file

    Returns:
        Dictionary containing:
        - valid: bool indicating if validation passed
        - total_images: number of images
        - total_annotations: total annotation count
        - class_distribution: dict of class counts
        - resolution_distribution: dict of resolution counts
        - issues: list of validation issues found
    """
    with open(submission_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    issues = []

    # Check top-level structure
    if "images" not in data:
        issues.append("Missing 'images' key at top level")
        return {
            "valid": False,
            "total_images": 0,
            "total_annotations": 0,
            "class_distribution": {},
            "resolution_distribution": {},
            "issues": issues,
        }

    images = data["images"]

    # Statistics
    total_annotations = 0
    class_stats = defaultdict(int)
    res_stats = defaultdict(int)

    for i, img in enumerate(images):
        # Check required image fields
        required_img_fields = [
            "file_name",
            "width",
            "height",
            "cm_resolution",
            "scene_type",
            "annotations",
        ]

        for field in required_img_fields:
            if field not in img:
                issues.append(f"Image {i}: Missing '{field}'")

        if "annotations" in img:
            for j, ann in enumerate(img["annotations"]):
                # Check required annotation fields
                required_ann_fields = ["class", "confidence_score", "segmentation"]

                for field in required_ann_fields:
                    if field not in ann:
                        issues.append(f"Image {i}, Annotation {j}: Missing '{field}'")

                # Validate class name
                if "class" in ann:
                    if ann["class"] not in ["individual_tree", "group_of_trees"]:
                        issues.append(
                            f"Image {i}, Annotation {j}: Invalid class '{ann['class']}'"
                        )
                    else:
                        class_stats[ann["class"]] += 1

                # Validate segmentation format
                if "segmentation" in ann:
                    seg = ann["segmentation"]

                    if not isinstance(seg, list):
                        issues.append(
                            f"Image {i}, Annotation {j}: Segmentation must be a list"
                        )
                    elif len(seg) < 6:
                        issues.append(
                            f"Image {i}, Annotation {j}: Segmentation must have "
                            f"at least 6 coordinates (got {len(seg)})"
                        )
                    elif len(seg) % 2 != 0:
                        issues.append(
                            f"Image {i}, Annotation {j}: Segmentation must have "
                            f"even number of coordinates (got {len(seg)})"
                        )

                # Validate confidence score
                if "confidence_score" in ann:
                    score = ann["confidence_score"]

                    if not isinstance(score, (int, float)):
                        issues.append(
                            f"Image {i}, Annotation {j}: confidence_score must be "
                            f"numeric (got {type(score).__name__})"
                        )
                    elif not (0 <= score <= 1):
                        issues.append(
                            f"Image {i}, Annotation {j}: confidence_score must be "
                            f"between 0 and 1 (got {score})"
                        )

                total_annotations += 1

        if "cm_resolution" in img:
            res_stats[img["cm_resolution"]] += 1

    stats = {
        "valid": len(issues) == 0,
        "total_images": len(images),
        "total_annotations": total_annotations,
        "class_distribution": dict(class_stats),
        "resolution_distribution": dict(res_stats),
        "issues": issues[:50],  # Limit to first 50 issues for readability
    }

    return stats


def print_validation_results(stats: Dict[str, Any]) -> None:
    """
    Pretty print validation results.

    Args:
        stats: Validation statistics dictionary from validate_submission_format
    """
    t("VALIDATION RESULTS")

    status_symbol = "PASSED" if stats["valid"] else "FAILED"
    p("Status", status_symbol)
    p("Total images", stats["total_images"])
    p("Total annotations", stats["total_annotations"])

    if stats["class_distribution"]:
        p("\nClass distribution:")
        for cls, count in stats["class_distribution"].items():
            p(f"  {cls}: {count}")

    if stats["resolution_distribution"]:
        p("\nResolution distribution:")
        for res, count in stats["resolution_distribution"].items():
            p(f"  {res}cm: {count}")

    if stats["issues"]:
        num_issues = len(stats["issues"])
        p("\nIssues found ({num_issues}):")

        # Show first 20 issues
        for issue in stats["issues"][:20]:
            p(f"  - {issue}")

        if num_issues > 20:
            p(f"  ... and {num_issues - 20} more issues")


# if __name__ == "__main__":
#     import sys
#
#     if len(sys.argv) > 1:
#         submission_path = Path(sys.argv[1])
#     else:
#         submission_path = Path("checkpoints/FINAL_SUBMISSION.json")
#
#     if submission_path.exists():
#         p(f"Validating: {submission_path}\n")
#         stats = validate_submission_format(submission_path)
#         print_validation_results(stats)
#
#         # Exit with error code if validation failed
#         sys.exit(0 if stats["valid"] else 1)
#     else:
#         p(f"Error: Submission file not found: {submission_path}")
#         sys.exit(1)


def validate_data_loader(data_loader, name="DataLoader"):
    """Validate data loader outputs for debugging."""
    t(f"Validating {name}")
    for i, (images, masks) in enumerate(data_loader):
        p(
            f"Image Batch {i}:",
            f"  Images: {images.shape}, dtype={images.dtype}, range=[{images.min():.3f}, {images.max():.3f}]",
        )
        p(
            f"Mask Batch {i}:",
            f"  Masks: {masks.shape}, dtype={masks.dtype}, unique={torch.unique(masks).tolist()}",
        )

        # Check for invalid values
        if torch.isnan(images).any():
            p("WARNING", "NaN values in images!", color1=c.ORANGE)
        if torch.isinf(images).any():
            p("WARNING", "Inf values in images!", color1=c.ORANGE)
        if masks.max() >= 3:
            p(
                "WARNING",
                f"Invalid mask values > 2: {torch.unique(masks).tolist()}",
                color1=c.ORANGE,
            )

        if i >= 2:  # Only check first few batches
            break


def analyze_validation_metrics(
    val_loss=None,
    iou=None,
    accuracy=None,
    precision=None,
    recall=None,
    f1_score=None,
    individual_tree_iou=None,
    group_tree_iou=None,
    dice=None,
    model_name="Model",
    verbose=True,
):
    """
    Analyze validation metrics and provide performance assessment.

    Args:
        val_loss: Validation loss (lower is better)
        iou: Intersection over Union (0-1, higher is better)
        accuracy: Pixel accuracy (0-1, higher is better)
        precision: Precision (0-1, higher is better)
        recall: Recall (0-1, higher is better)
        f1_score: F1 score (0-1, higher is better)
        individual_tree_iou: IoU for individual trees
        group_tree_iou: IoU for tree groups
        dice: Dice coefficient (0-1, higher is better)
        model_name: Name for display
        verbose: Print detailed analysis

    Returns:
        dict: Analysis results with scores and recommendations
    """

    def get_performance_level(value, thresholds):
        """Get performance level based on thresholds [poor, fair, good, excellent]"""
        if value >= thresholds[2]:
            return "excellent"
        elif value >= thresholds[1]:
            return "good"
        elif value >= thresholds[0]:
            return "fair"
        else:
            return "poor"

    def get_loss_level(loss, thresholds):
        """Get performance level for loss (lower is better)"""
        if loss <= thresholds[0]:
            return "excellent"
        elif loss <= thresholds[1]:
            return "good"
        elif loss <= thresholds[2]:
            return "fair"
        else:
            return "poor"

    # Define thresholds for different metrics
    thresholds = {
        "iou": [0.3, 0.5, 0.7],  # poor < 30%, fair 30-50%, good 50-70%, excellent >70%
        "accuracy": [
            0.7,
            0.85,
            0.95,
        ],  # poor < 70%, fair 70-85%, good 85-95%, excellent >95%
        "precision": [
            0.5,
            0.7,
            0.85,
        ],  # poor < 50%, fair 50-70%, good 70-85%, excellent >85%
        "recall": [
            0.5,
            0.7,
            0.85,
        ],  # poor < 50%, fair 50-70%, good 70-85%, excellent >85%
        "f1": [0.5, 0.7, 0.85],  # poor < 50%, fair 50-70%, good 70-85%, excellent >85%
        "dice": [
            0.5,
            0.7,
            0.85,
        ],  # poor < 50%, fair 50-70%, good 70-85%, excellent >85%
        "loss": [
            0.3,
            0.6,
            1.0,
        ],  # excellent < 0.3, good 0.3-0.6, fair 0.6-1.0, poor >1.0
    }

    results = {
        "model_name": model_name,
        "metrics": {},
        "overall_score": 0,
        "performance_level": "poor",
        "recommendations": [],
        "concerns": [],
        "strengths": [],
    }

    # Analyze each metric
    metrics_analyzed = 0
    total_score = 0

    if val_loss is not None:
        level = get_loss_level(val_loss, thresholds["loss"])
        score_map = {"excellent": 4, "good": 3, "fair": 2, "poor": 1}
        score = score_map[level]
        results["metrics"]["val_loss"] = {
            "value": val_loss,
            "level": level,
            "score": score,
        }
        total_score += score
        metrics_analyzed += 1

        if level == "poor":
            results["concerns"].append(
                f"High validation loss ({val_loss:.3f}) suggests poor convergence"
            )
        elif level == "excellent":
            results["strengths"].append(
                f"Excellent low validation loss ({val_loss:.3f})"
            )

    # Helper function for standard metrics
    def analyze_metric(value, name, threshold_key):
        if value is None:
            return 0
        level = get_performance_level(value, thresholds[threshold_key])
        score_map = {"excellent": 4, "good": 3, "fair": 2, "poor": 1}
        score = score_map[level]
        results["metrics"][name] = {"value": value, "level": level, "score": score}

        if level == "poor":
            results["concerns"].append(
                f"Low {name} ({value:.1%}) indicates poor {name.replace('_', ' ')}"
            )
        elif level == "excellent":
            results["strengths"].append(f"Excellent {name} ({value:.1%})")

        return score

    # Analyze all metrics
    if iou is not None:
        total_score += analyze_metric(iou, "iou", "iou")
        metrics_analyzed += 1

    if accuracy is not None:
        total_score += analyze_metric(accuracy, "accuracy", "accuracy")
        metrics_analyzed += 1

    if precision is not None:
        total_score += analyze_metric(precision, "precision", "precision")
        metrics_analyzed += 1

    if recall is not None:
        total_score += analyze_metric(recall, "recall", "recall")
        metrics_analyzed += 1

    if f1_score is not None:
        total_score += analyze_metric(f1_score, "f1_score", "f1")
        metrics_analyzed += 1

    if dice is not None:
        total_score += analyze_metric(dice, "dice", "dice")
        metrics_analyzed += 1

    # Class-specific IoUs
    if individual_tree_iou is not None:
        total_score += analyze_metric(individual_tree_iou, "individual_tree_iou", "iou")
        metrics_analyzed += 1

    if group_tree_iou is not None:
        total_score += analyze_metric(group_tree_iou, "group_tree_iou", "iou")
        metrics_analyzed += 1

    # Calculate overall performance
    if metrics_analyzed > 0:
        avg_score = total_score / metrics_analyzed
        results["overall_score"] = avg_score

        if avg_score >= 3.5:
            results["performance_level"] = "excellent"
        elif avg_score >= 2.5:
            results["performance_level"] = "good"
        elif avg_score >= 1.5:
            results["performance_level"] = "fair"
        else:
            results["performance_level"] = "poor"

    # Generate recommendations
    if results["performance_level"] == "poor":
        results["recommendations"].extend(
            [
                "Try a more powerful model (UNet, YOLOv8)",
                "Reduce learning rate (try 1e-4 or 1e-5)",
                "Increase training epochs",
                "Check data quality and annotations",
                "Consider data augmentation",
            ]
        )
    elif results["performance_level"] == "fair":
        results["recommendations"].extend(
            [
                "Fine-tune hyperparameters",
                "Try different augmentations",
                "Consider ensemble methods",
            ]
        )
    elif results["performance_level"] == "good":
        results["recommendations"].extend(
            [
                "Model is performing well",
                "Consider minor hyperparameter tuning for optimization",
            ]
        )
    else:
        results["recommendations"].append(
            "Excellent performance! Model is ready for deployment"
        )

    # Print analysis if verbose
    if verbose:
        t(f"{model_name} Performance Analysis")
        p("Overall Performance", f"{results['performance_level'].upper()}")
        p("Overall Score", f"{results['overall_score']:.2f}/4.0")
        p()

        if results["metrics"]:
            t("Metric Breakdown:")
            for metric, data in results["metrics"].items():
                icon = {"excellent": "ðŸ¥‡", "good": "ðŸ¥ˆ", "fair": "ðŸ¥‰", "poor": "ðŸ˜¡"}[
                    data["level"]
                ]
                col = {
                    "excellent": c.GREEN,
                    "good": c.BLUE,
                    "fair": c.ORANGE,
                    "poor": c.RED,
                }[data["level"]]
                if "loss" in metric:
                    p(
                        f"  {icon} {metric.replace('_', ' ').title()}",
                        f"{data['value']:.4f} ({data['level']})",
                        color1=col,
                    )
                else:
                    p(
                        f"  {icon} {metric.replace('_', ' ').title()}",
                        f"{data['value']:.1%} ({data['level']})",
                        color1=col,
                    )
            p()

        if results["strengths"]:
            t("Strengths:")
            for strength in results["strengths"]:
                p(f"  â€¢ {strength}", color1=c.BLACK)
            p()

        if results["concerns"]:
            t("Concerns:")
            for concern in results["concerns"]:
                p(f"  â€¢ {concern}", color1=c.BLACK)
            p()

        if results["recommendations"]:
            t("Recommendations:")
            for rec in results["recommendations"]:
                p(f"  â€¢ {rec}", color1=c.BLACK)

        p("=" * 50)

    return results


# From C:\github\Tree-Canopy-Detection\src\prediction\__init__.py


