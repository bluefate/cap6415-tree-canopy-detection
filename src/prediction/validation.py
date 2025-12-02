"""
Submission validation utilities for Tree Canopy Detection
Validates submission JSON format according to competition requirements
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

from src.utils.helpers import p, t


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


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        submission_path = Path(sys.argv[1])
    else:
        submission_path = Path("checkpoints/FINAL_SUBMISSION.json")

    if submission_path.exists():
        p(f"Validating: {submission_path}\n")
        stats = validate_submission_format(submission_path)
        print_validation_results(stats)

        # Exit with error code if validation failed
        sys.exit(0 if stats["valid"] else 1)
    else:
        p(f"Error: Submission file not found: {submission_path}")
        sys.exit(1)
