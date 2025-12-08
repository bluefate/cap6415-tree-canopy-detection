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
    Convert multi-class mask to list of polygon annotations.
    
    Extracts contours for each tree class and converts to polygon segmentation format.
    
    Args:
        mask (np.ndarray): Multi-class mask with values 0=background, 1=individual_tree, 2=group_of_trees.
        id_to_class (dict, optional): Mapping of class IDs to names.
    
    Returns:
        List[dict]: List of annotation dictionaries with class, confidence, and segmentation.
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


def export_submission(results: List[Dict[str, Any]], output_path: Path, config) -> None:
    """
    Export prediction results in required submission JSON format.
    
    Uses template file to preserve cm_resolution and scene_type while updating annotations
    with model predictions.
    
    Args:
        results (List[Dict]): Prediction results with image names and masks.
        output_path (Path): Path to save submission JSON file.
        config: Configuration object with paths including template file.
    
    Returns:
        None
    """
    # Load template - use raw string (r"...") for Windows paths
    # template_path = Path(
    #     "/content/drive/MyDrive/TreeCanopyProject/data/data1/sample_answer.json"
    # )
    # template_path = Path(
    #     r"C:\github\Tree-Canopy-Detection\src\data\data1\sample_answer.json"
    # )
    template_path = config.paths.template

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
    Extract resolution in centimeters from image filename.
    
    Parses filenames like 'forest_10cm_001.tif' to extract resolution value.
    
    Args:
        fname (str): Image filename.
    
    Returns:
        int: Resolution in centimeters (default 10 if not found).
    """
    name = Path(fname).stem.lower()

    # Search for patterns like 5cm, 10cm, 20cm, etc.
    import re

    match = re.search(r"(\d+)\s*cm", name)
    if match:
        return int(match.group(1))

    # If no information found, return fallback
    return 10
