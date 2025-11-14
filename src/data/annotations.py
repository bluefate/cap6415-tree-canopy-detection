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

    def __init__( self, cls: str, segmentation: List[float], confidence: float = 1.0 ):
        self.cls = cls
        self.segmentation = segmentation
        self.confidence = confidence
        self.bbox = self.compute_bbox(segmentation)

    @staticmethod
    def compute_bbox( seg: List[float] ) -> List[int]:
        """
        Convert a flat segmentation list into a bounding box.
        Returns [x1, y1, x2, y2].
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

    def __init__( self, image_path: Path, width: int, height: int, items: List[AnnotationItem] ):
        self.image_path = image_path
        self.width = width
        self.height = height
        self.items = items

    def to_mask( self ) -> np.ndarray:
        """
        Build a binary mask from all polygons in this entry.
        """
        mask = np.zeros((self.height, self.width), dtype = np.uint8)
        for item in self.items:
            seg = item.segmentation
            if seg is None or len(seg) < 4:
                continue
            poly = np.array(seg, dtype = np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [poly], 1)
        return mask


def load_json_annotations( json_path: Path ) -> List[AnnotationEntry]:
    """
    Load annotation entries from a JSON file that contains images and annotations.
    Returns a list of AnnotationEntry objects.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Missing annotation file {json_path}")

    with open(json_path, "r", encoding = "utf8") as f:
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


def get_unique_classes( entries: List[AnnotationEntry] ) -> List[str]:
    """
    Return a sorted list of unique classes across all entries.
    """
    classes = set()
    for entry in entries:
        for item in entry.items:
            classes.add(item.cls)
    return sorted(classes)
