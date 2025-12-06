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


def validate_data_batch(images, masks, batch_idx=0):
    """Validate a batch of data for common issues."""
    p(f"Batch {batch_idx} validation:")
    p(f"  Images: shape={images.shape}, dtype={images.dtype}")
    p(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
    p(f"  Masks: shape={masks.shape}, dtype={masks.dtype}")
    p(f"  Mask unique values: {torch.unique(masks).tolist()}")

    # Check for issues
    issues = []
    if torch.isnan(images).any():
        issues.append("NaN values in images")
    if torch.isinf(images).any():
        issues.append("Inf values in images")
    if images.min() < -3 or images.max() > 3:
        issues.append(
            f"Images outside expected range: [{images.min():.3f}, {images.max():.3f}]"
        )
    if masks.max() >= 3:
        issues.append(f"Invalid mask values >= 3: {torch.unique(masks).tolist()}")
    if masks.min() < 0:
        issues.append(f"Negative mask values: {torch.unique(masks).tolist()}")

    if issues:
        p("Warning", f"ISSUES FOUND: {issues}", color1 = c.ORANGE)
    else:
        p("Batch looks good")
    print()


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
                icon = {"excellent": "🥇", "good": "🥈", "fair": "🥉", "poor": "😡"}[
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
                p(f"  • {strength}", color1=c.BLACK)
            p()

        if results["concerns"]:
            t("Concerns:")
            for concern in results["concerns"]:
                p(f"  • {concern}", color1=c.BLACK)
            p()

        if results["recommendations"]:
            t("Recommendations:")
            for rec in results["recommendations"]:
                p(f"  • {rec}", color1=c.BLACK)

        p("=" * 50)

    return results
