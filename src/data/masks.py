from pathlib import Path
from typing import List

import cv2
import numpy as np


# Class mapping for Solafune competition
CLASS_TO_ID = {
    "individual_tree": 1,
    "group_of_trees": 2,
}

def build_multiclass_mask(entry, class_to_id: dict = None) -> np.ndarray:
    """
    Build multi-class segmentation mask with class indices.
    
    Background=0, individual_tree=1, group_of_trees=2, unknown classes are skipped.
    
    Args:
        entry (AnnotationEntry): Annotation entry with image dimensions and items.
        class_to_id (dict, optional): Mapping from class name to ID. Defaults to standard mapping.
    
    Returns:
        np.ndarray: Multi-class mask with dtype=uint8.
    """
    if class_to_id is None:
        class_to_id = CLASS_TO_ID

    mask = np.zeros((entry.height, entry.width), dtype=np.uint8)

    for item in entry.items:
        seg = item.segmentation
        if seg is None or len(seg) < 6:
            continue

        # Get class ID, skip unknown classes (don't assign to background)
        class_id = class_to_id.get(item.cls, None)
        if class_id is None:
            # Skip unknown classes instead of assigning to background
            continue

        poly = np.array(seg, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [poly], class_id)

    return mask


def save_mask(mask: np.ndarray, path: Path) -> None:
    """
    Save binary mask to disk with values converted to 0 or 255.
    
    Args:
        mask (np.ndarray): Binary mask (values 0-1).
        path (Path): Output file path.
    
    Returns:
        None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = (mask * 255).astype(np.uint8)
    cv2.imwrite(str(path), out)


