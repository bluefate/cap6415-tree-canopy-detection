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
        """
        Initialize an annotation item.

        Args:
            cls (str): Class name ('individual_tree' or 'group_of_trees').
            segmentation (list): Flat list of polygon coordinates [x1, y1, x2, y2, ...].
            confidence (float): Confidence score. Defaults to 1.0.
        """
        self.cls = cls
        self.segmentation = segmentation
        self.confidence = confidence
        self.bbox = self.compute_bbox(segmentation)

    @staticmethod
    def compute_bbox(seg: List[float]) -> List[int]:
        """
        Convert a flat segmentation polygon to bounding box coordinates.

        Args:
            seg (List[float]): Flat list of polygon coordinates [x1, y1, x2, y2, ...].

        Returns:
            List[int]: Bounding box as [x1, y1, x2, y2] (top-left and bottom-right corners).
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
        """
        Initialize an annotation entry for a single image.

        Args:
            image_path (Path): Path to the image file.
            width (int): Image width in pixels.
            height (int): Image height in pixels.
            items (list): List of AnnotationItem objects for this image.
        """
        self.image_path = image_path
        self.width = width
        self.height = height
        self.items = items

    def to_mask(self) -> np.ndarray:
        """
        Generate a binary mask from all polygon annotations in this entry.

        Fills all polygons to create a single-channel mask with 0 (background) and 1 (object).

        Returns:
            np.ndarray: Binary mask array of shape (height, width) with uint8 dtype.
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
    Load annotation entries from JSON file with images and annotations.

    Expected JSON structure:
    {
        "images": [
            {
                "file_name": "image1.jpg",
                "width": 1024,
                "height": 1024,
                "annotations": [
                    {
                        "class": "individual_tree",
                        "segmentation": [x1, y1, x2, y2, ...],
                        "confidence_score": 0.95
                    }
                ]
            }
        ]
    }

    Args:
        json_path (Path): Path to annotation JSON file.

    Returns:
        List[AnnotationEntry]: List of annotation entries for each image.

    Raises:
        FileNotFoundError: If annotation file does not exist.
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
    Extract and return sorted list of unique classes across all annotation entries.

    Args:
        entries (List[AnnotationEntry]): List of annotation entries.

    Returns:
        List[str]: Sorted list of unique class names.
    """
    classes = set()
    for entry in entries:
        for item in entry.items:
            classes.add(item.cls)
    return sorted(classes)
