import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from src.utils.helpers import p, t


def mask_to_polygons(mask: np.ndarray) -> List[List[int]]:
    """
    Convert a binary mask to COCO-style polygon lists.
    Returns a list of polygon coordinate lists.
    """
    mask = (mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []

    for cnt in contours:
        if len(cnt) >= 3:
            cnt = cnt.reshape(-1, 2).tolist()
			# Flatten: [[x1,y1], [x2,y2]] -> [x1,y1,x2,y2]
            flat = [coord for pt in cnt for coord in pt]
            polygons.append(flat)

    return polygons


def build_submission_entry(
        file_name: str,
        width: int,
        height: int,
        polygons: List[List[int]],
        scene_type: str = "unknown",
        cm_resolution: int = 10,
) -> Dict[str, Any]:
    """
    Build one image level submission entry.
    """
    annotations = []
    for poly in polygons:
        if len(poly) >= 6:
            annotations.append({
                "class": "tree",
                "confidence_score": 1.0,
                "segmentation": poly,
            })

    return {
        "file_name": file_name,
        "width": width,
        "height": height,
        "scene_type": scene_type,
        "cm_resolution": cm_resolution,
        "annotations": annotations,
    }


def export_submission(
        results: List[Dict[str, Any]],
        output_path: Path,
) -> None:
    """
    Convert prediction results into expected submission JSON structure.
    Enforces required fields and valid geometry.
    """

    images = []

    for r in results:
        image = r.get("image", None)
        mask = r.get("mask", None)
        fname = r.get("name", "")

        # enforce .tif extension
        fname = Path(fname).stem + ".tif"

        # Dimensions
        if image is not None:
            h, w = image.shape[:2]
        elif mask is not None:
            h, w = mask.shape
        else:
            raise ValueError(f"Missing image/mask for result entry: {r}")

        # Convert mask to polygons
        try:
            polygons = mask_to_polygons(mask) if mask is not None else []
        except Exception as e:
            p("Warning", f"Polygon conversion failed for {fname}: {e}", color1=c.ORANGE)
            polygons = []

        # Build final entry with required keys
        entry = build_submission_entry(
            file_name=fname,
            width=w,
            height=h,
            polygons=polygons,
            scene_type=r.get("scene_type", "unknown"),
            cm_resolution = extract_cm_resolution(fname),
        )

        images.append(entry)

    # Save JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    submission = {"images": images}
    # with open(output_path, "w", encoding = "utf8") as f:
    with output_path.open("w", encoding="utf8") as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)

    # Logging
    t("Submission Export Complete")
    p("✓ Submission saved", output_path)
    p("✓ Total images", len(images))
    p("✓ Total annotations", sum(len(img["annotations"]) for img in images))

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
