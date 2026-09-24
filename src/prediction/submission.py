import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from src.utils.helpers import p, t


def separate_instances(
    mask: np.ndarray,
    min_area: int = 25,
    dist_thresh: float = 0.35,
) -> np.ndarray:
    """
    Split connected canopy regions into instance-like components per class.

    Distance transform + watershed so contour extraction yields separate
    polygons (semantic → instance post-process for weighted mAP).

    Args:
        mask: Multiclass mask {0=bg, 1=individual_tree, 2=group_of_trees}.
        min_area: Drop components smaller than this (pixels).
        dist_thresh: Fraction of max distance used as seed threshold.

    Returns:
        Multiclass mask with class ids; watershed ridges left as 0 so
        external contours separate adjacent trees of the same class.
    """
    out = np.zeros_like(mask, dtype=np.uint8)

    for class_id in (1, 2):
        binary = (mask == class_id).astype(np.uint8)
        if int(binary.sum()) == 0:
            continue

        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        peak = float(dist.max()) if dist.size else 0.0
        if peak <= 0:
            out[binary == 1] = class_id
            continue

        _, sure_fg = cv2.threshold(dist, dist_thresh * peak, 255, cv2.THRESH_BINARY)
        sure_fg = sure_fg.astype(np.uint8)
        n_labels, markers = cv2.connectedComponents(sure_fg)
        if n_labels <= 1:
            # No seeds — keep whole component as one instance
            n_cc, cc = cv2.connectedComponents(binary)
            for lab in range(1, n_cc):
                comp = cc == lab
                if int(comp.sum()) >= min_area:
                    out[comp] = class_id
            continue

        markers = markers.astype(np.int32)
        # Background must be 0; shift labels so 0 stays bg for watershed
        markers[binary == 0] = 0
        ws = cv2.cvtColor(binary * 255, cv2.COLOR_GRAY2BGR)
        cv2.watershed(ws, markers)

        for lab in range(1, int(markers.max()) + 1):
            comp = markers == lab
            if int(comp.sum()) < min_area:
                continue
            out[comp] = class_id
        # Leave markers == -1 (boundaries) as 0 → separates contours

    return out


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
                instance_mask = separate_instances(mask)
                annotations = mask_to_polygons_multiclass(instance_mask)
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


