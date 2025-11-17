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
        img_t = torch.tensor(img.transpose(2, 0, 1)).float() / 255.0
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
                # img_chw = img_t.permute(2, 0, 1).unsqueeze(0).float()
                img_chw = img_t.unsqueeze(0).float()
                base = img_t.permute(1, 2, 0).cpu().numpy()
            else:
                img_chw = torch.from_numpy(img_t.transpose(2, 0, 1)).unsqueeze(0).float()
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
