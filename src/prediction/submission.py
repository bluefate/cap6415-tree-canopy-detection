import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from src.utils.helpers import p, t


def mask_to_polygons( mask: np.ndarray ) -> List[List[int]]:
    """
    Convert a binary mask into COCO style polygon lists.
    Returns a list of polygon coordinate lists.
    """
    mask = (mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []

    for cnt in contours:
        if len(cnt) >= 3:
            cnt = cnt.reshape(-1, 2).tolist()
            # Flatten: [[x1,y1], [x2,y2]] -> [x1,y1,x2,y2]
            flat = [coord for point in cnt for coord in point]
            polygons.append(flat)

    return polygons


def build_submission_entry(
        file_name: str,
        width: int,
        height: int,
        polygons: List[List[int]],
        # scene_type: str = "unknown",
        # cm_resolution: int = 0,
) -> Dict[str, Any]:
    """
    Build one image level submission item.
    """
    annotations = []
    for poly in polygons:
        if len(poly) >= 6:
            annotations.append(
                {
                    "class": "tree",
                    "confidence_score": 1.0,
                    "segmentation": poly,
                },
        )

    return {
        "file_name": file_name,
        "width": width,
        "height": height,
        # "cm_resolution": cm_resolution,
        # "scene_type": scene_type,
        # "annotations": annotations,
    }


def export_submission(
        results: List[Dict[str, Any]],
        output_path: Path,
) -> None:
    """
    Convert a list of prediction results into a submission JSON file.
    Each result entry must contain:
    file, image, mask

    Saves output_path as a JSON file.
    """

    images  = []
    for r in results:
        image = r["image"]
        mask = r["mask"]
        fname = r["name"]

        fname = Path(fname).stem + ".tif"

        # Get dimensions from mask or image
        if "image" in r:
            h, w = r["image"].shape[:2]
        else:
            h, w = mask.shape

        # Convert mask to polygons
        polygons = mask_to_polygons(mask)

        # Build entry
        entry = build_submission_entry(
                file_name = fname,
                width = w,
                height = h,
                polygons = polygons,
        )
        images .append(entry)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents = True, exist_ok = True)

    # Write JSON
    submission = {"images": images}

    with open(output_path, "w", encoding = "utf8") as f:
        json.dump(submission, f, indent=2, ensure_ascii=False)

    t("Submission Export Complete")

    # Validation
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
        p("annotations", f"{len(sample['annotations'])} polygons")

        if sample["annotations"]:
            ann = sample["annotations"][0]
            p("  class", ann["class"])
            p("  confidence", ann["confidence_score"])
            p("  segmentation points", len(ann["segmentation"]))



