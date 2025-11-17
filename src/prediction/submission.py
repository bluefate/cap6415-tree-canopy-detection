import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np


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
            flat = [coord for point in cnt for coord in point]
            polygons.append(flat)

    return polygons


def build_submission_entry(
        file_name: str,
        width: int,
        height: int,
        polygons: List[List[int]],
        scene_type: str = "unknown",
        cm_resolution: int = 0,
) -> Dict[str, Any]:
    """
    Build one image level submission item.
    """
    annotations = []
    for poly in polygons:
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
        "cm_resolution": cm_resolution,
        "scene_type": scene_type,
        "annotations": annotations,
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

    dataset = []
    for r in results:
        image = r["image"]
        mask = r["mask"]
        fname = r["name"]

        h, w = mask.shape
        polygons = mask_to_polygons(mask)

        entry = build_submission_entry(
                file_name = fname,
                width = w,
                height = h,
                polygons = polygons,
        )
        dataset.append(entry)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents = True, exist_ok = True)

    with open(output_path, "w", encoding = "utf8") as f:
        json.dump({ "images": dataset }, f, indent = 4)
