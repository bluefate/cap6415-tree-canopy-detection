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

    def __init__( self, model_path: Path, model_name: str = "unet", image_size: int = 256 ):
        self.model_path = Path(model_path)
        self.model_name = model_name
        self.image_size = image_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = Logger()

        self.model = build_model(model_name, in_channels = 3, out_channels = 1).to(self.device)
        self._load_weights()

    def _load_weights( self ):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Missing model weights {self.model_path}")
        state = torch.load(self.model_path, map_location = self.device)
        if "model" in state:
            self.model.load_state_dict(state["model"])
        else:
            self.model.load_state_dict(state)
        self.model.eval()
        self.logger.info(f"Loaded model weights from {self.model_path}")

    def predict_tensor( self, tensor: torch.Tensor ) -> np.ndarray:
        """
        Run inference on a single tensor. Returns a numpy mask.
        """
        tensor = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(tensor)
        pred = torch.sigmoid(pred).cpu().squeeze().numpy()
        return (pred > 0.5).astype(np.uint8)

    def predict_image( self, image: np.ndarray ) -> np.ndarray:
        """
        Run inference directly on an RGB numpy image.
        Resizing and conversion done here.
        """
        img = cv2.resize(image, (self.image_size, self.image_size))
        img_t = torch.tensor(img.transpose(2,0,1)).float() / 255.0
        return self.predict_tensor(img_t)

    def run_on_folder( self, image_dir: Path, transform = None, num_samples: int = None ):
        """
        Run prediction on a folder of images using ImageOnlyDataset.
        Returns list of dictionaries with masks and overlays.
        """
        dataset = ImageOnlyDataset(image_dir, transform = transform)
        results = []

        total = len(dataset) if num_samples is None else min(num_samples, len(dataset))

        for idx in range(total):
            name, img_t = dataset[idx]

            # Convert HWC -> CHW safely for both numpy and torch
            if isinstance(img_t, torch.Tensor):
                # img_chw = img_t.permute(2,0,1).unsqueeze(0).float()
                img_chw = img_t.unsqueeze(0).float()
                base = img_t.permute(1,2,0).cpu().numpy()
            else:
                img_chw = torch.from_numpy(img_t.transpose(2,0,1)).unsqueeze(0).float()
                base = img_t

            with torch.no_grad():
                pred = self.model(img_chw.to(self.device)).cpu().squeeze().numpy()

            pred_bin = (pred > 0.5).astype(np.uint8)

            pred_u8 = pred_bin * 255
            overlay = np.zeros_like(base)
            overlay = overlay.copy()
            overlay[:, :, 0] = pred_u8
            overlay = cv2.addWeighted(base.astype(np.uint8), 0.6, overlay.astype(np.uint8), 0.4, 0)

            results.append({
                "name": name,
                "image": base,
                "mask": pred_bin,
                "overlay": overlay,
            })

        return results


# From C:\github\Tree-Canopy-Detection\src\prediction\postprocess.py
import cv2
import numpy as np


def refine_mask( mask: np.ndarray, min_area: int = 20 ) -> np.ndarray:
    """
    Clean small artifacts in a binary mask.
    Removes connected components smaller than min_area.
    """
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity = 8)

    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 1
    return cleaned


def group_mask_threshold( mask: np.ndarray, threshold: float = 0.5 ) -> np.ndarray:
    """
    Apply a direct threshold to a probability mask.
    """
    return (mask > threshold).astype(np.uint8)


def overlay_mask( image: np.ndarray, mask: np.ndarray, alpha: float = 0.4 ) -> np.ndarray:
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


# From C:\github\Tree-Canopy-Detection\src\prediction\__init__.py


