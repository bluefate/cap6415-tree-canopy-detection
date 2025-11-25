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
    Now handles multi-class predictions properly.
    """
    images = []

    for r in results:
        image = r.get("image", None)
        mask = r.get("mask", None)
        fname = r.get("name", "")

        # Enforce .tif extension
        fname = Path(fname).stem + ".tif"

        # Dimensions
        if image is not None:
            h, w = image.shape[:2]
        elif mask is not None:
            h, w = mask.shape[:2] if mask.ndim >= 2 else (0, 0)
        else:
            raise ValueError(f"Missing image/mask for result entry: {r}")

        try:
            if mask is not None:
                annotations = mask_to_polygons_multiclass(mask)
            else:
                annotations = []
        except Exception as e:
            p("Warning", f"Polygon conversion failed for {fname}: {e}", color1=c.ORANGE)
            annotations = []

        # Build entry directly (no longer using build_submission_entry for polygons)
        entry = {
            "file_name": fname,
            "width": w,
            "height": h,
            "scene_type": r.get("scene_type", "unknown"),
            "cm_resolution": extract_cm_resolution(fname),
            "annotations": annotations,  # ← Already has correct class names!
        }

        images.append(entry)

    # Save JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    submission = {"images": images}
    with output_path.open("w", encoding="utf8") as f:
        json.dump(submission, f, indent=2)

    # Logging
    t("Submission Export Complete")
    p("✓ Submission saved", output_path)
    p("✓ Total images", len(images))
    p("✓ Total annotations", sum(len(img["annotations"]) for img in images))

    # Count by class
    individual_count = sum(
        1
        for img in images
        for ann in img["annotations"]
        if ann["class"] == "individual_tree"
    )
    group_count = sum(
        1
        for img in images
        for ann in img["annotations"]
        if ann["class"] == "group_of_trees"
    )
    p("✔ individual_tree annotations", individual_count)
    p("✔ group_of_trees annotations", group_count)

    # Show sample
    if images:
        t("Sample")
        sample = images[0]

        p("file_name", sample["file_name"])
        p("width", sample["width"])
        p("height", sample["height"])
        p("scene_type", sample["scene_type"])
        p("cm_resolution", sample["cm_resolution"])
        p("annotations", f"{len(sample['annotations'])} polygons")

        if sample["annotations"]:
            ann = sample["annotations"][0]
            p("  class", ann["class"])
            p("  confidence", ann["confidence_score"])
            p("  segmentation points", len(ann["segmentation"]))


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
