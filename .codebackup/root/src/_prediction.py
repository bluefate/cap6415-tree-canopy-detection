# From C:\github\Tree-Canopy-Detection\src\prediction\pipeline.py
from pathlib import Path

import cv2
import numpy as np
import torch

from src.data.loaders import ImageOnlyDataset
from src.models.zoo import build_model
from src.utils.logging import Logger


class Predictor:
    """
    Full inference pipeline for tree canopy prediction.
    Loads a trained model, applies preprocessing, runs inference,
    and returns refined masks and overlays.
    """

    def __init__(
        self, model_path: Path, model_name: str = "unet", image_size: int = 256
    ):
        self.model_path = Path(model_path)
        self.model_name = model_name
        self.image_size = image_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = Logger()

        self.model = build_model(model_name, in_channels=3, out_channels=3).to(
            self.device
        )
        self._load_weights()

    def _load_weights(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Missing model weights {self.model_path}")
        state = torch.load(self.model_path, map_location=self.device)
        if "model" in state:
            self.model.load_state_dict(state["model"])
        else:
            self.model.load_state_dict(state)
        self.model.eval()
        self.logger.info(f"Loaded model weights from {self.model_path}")

    def predict_tensor(self, tensor: torch.Tensor) -> np.ndarray:
        """Run inference on a single tensor. Returns a numpy mask."""
        tensor = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(tensor)

        # Handle multi-class output
        if pred.shape[1] == 3:  # Multi-class
            pred_classes = torch.argmax(pred, dim=1).cpu().squeeze().numpy()
            # Return binary mask (any tree class > 0)
            return (pred_classes > 0).astype(np.uint8)
        else:  # Binary
            pred = torch.sigmoid(pred).cpu().squeeze().numpy()
            return (pred > 0.5).astype(np.uint8)

    def predict_image(self, image: np.ndarray) -> np.ndarray:
        """
        Run inference directly on an RGB numpy image.
        Resizing and conversion done here.
        """
        img = cv2.resize(image, (self.image_size, self.image_size))
        img_t = torch.tensor(img.transpose(2, 0, 1)).float() / 255.0
        return self.predict_tensor(img_t)

    def run_on_folder(self, image_dir: Path, transform=None, num_samples: int = None):
        """
        Run prediction on a folder of images using ImageOnlyDataset.
        Returns list of dictionaries with masks and overlays.
        """
        dataset = ImageOnlyDataset(image_dir, transform=transform)
        results = []

        total = len(dataset) if num_samples is None else min(num_samples, len(dataset))

        for idx in range(total):
            name, img_t = dataset[idx]

            # Convert HWC -> CHW safely for both numpy and torch
            if isinstance(img_t, torch.Tensor):
                # img_chw = img_t.permute(2,0,1).unsqueeze(0).float()
                img_chw = img_t.unsqueeze(0).float()
                base = img_t.permute(1, 2, 0).cpu().numpy()
            else:
                img_chw = (
                    torch.from_numpy(img_t.transpose(2, 0, 1)).unsqueeze(0).float()
                )
                base = img_t

            with torch.no_grad():
                # pred = self.model(img_chw.to(self.device)).cpu().squeeze().numpy()
                pred = self.model(img_chw.to(self.device)).cpu()

            # Convert multi-class logits to class predictions
            if pred.shape[1] == 3:  # Multi-class
                # Get class with highest probability
                pred_classes = torch.argmax(pred, dim=1).squeeze().numpy()  # [H, W]

                # For overlay visualization, create binary mask
                pred_bin_for_overlay = (pred_classes > 0).astype(np.uint8)
            else:  # Binary
                pred = torch.sigmoid(pred).squeeze().numpy()
                pred_classes = (pred > 0.5).astype(np.uint8)
                pred_bin_for_overlay = pred_classes

            pred_u8 = pred_bin_for_overlay * 255
            overlay = np.zeros_like(base)
            overlay = overlay.copy()
            overlay[:, :, 0] = pred_u8
            overlay = cv2.addWeighted(
                base.astype(np.uint8), 0.6, overlay.astype(np.uint8), 0.4, 0
            )

            results.append(
                {
                    "name": name,
                    "image": base,
                    "mask": pred_classes,
                    "overlay": overlay,
                }
            )

        return results


# From C:\github\Tree-Canopy-Detection\src\prediction\postprocess.py
import cv2
import numpy as np


def refine_mask(mask: np.ndarray, min_area: int = 20) -> np.ndarray:
    """
    Clean small artifacts in a binary mask.
    Removes connected components smaller than min_area.
    """
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 1
    return cleaned


def group_mask_threshold(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Apply a direct threshold to a probability mask.
    """
    return (mask > threshold).astype(np.uint8)


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Create a red overlay of the mask on top of an RGB image.
    """
    if image.max() <= 1.0:
        img_u8 = (image * 255).astype(np.uint8)
    else:
        img_u8 = image.astype(np.uint8)

    mask_u8 = (mask * 255).astype(np.uint8)

    mask_rgb = np.zeros_like(img_u8)
    mask_rgb[:, :, 0] = mask_u8

    return cv2.addWeighted(img_u8, 1 - alpha, mask_rgb, alpha, 0)


# From C:\github\Tree-Canopy-Detection\src\prediction\submission.py
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

    Expected mask values:
        0 = background
        1 = individual_tree
        2 = group_of_trees
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
    Uses sample_answer.json as template to preserve cm_resolution and scene_type.

    IMPORTANT: Per admin guidance, only annotations should be modified.
    cm_resolution and scene_type come from the template.
    """
    # Load template - use raw string (r"...") for Windows paths
    template_path = Path(
        r"C:\github\Tree-Canopy-Detection\src\data\data1\sample_answer.json"
    )

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


# From C:\github\Tree-Canopy-Detection\src\prediction\__init__.py


