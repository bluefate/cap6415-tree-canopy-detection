from pathlib import Path

import cv2
import numpy as np
import torch

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
        """
        Initialize prediction pipeline with trained model.
        
        Args:
            model_path (Path): Path to trained model weights.
            model_name (str): Model architecture name. Defaults to "unet".
            image_size (int): Target image size for inference. Defaults to 256.
        """
        self.model_path = Path(model_path)
        self.model_name = model_name
        self.image_size = image_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = Logger()

        self.model = build_model(model_name, in_channels=3, out_channels=3).to(
            self.device
        )
        # model_args = MODEL_BENCHMARKS.get(model_name, {})
        # model_args.update({"in_channels": 3, "out_channels": 3})
        # self.model = build_model(model_name, **model_args).to(self.device)

        self._load_weights()

    def _load_weights(self):
        """
        Load trained model weights from checkpoint file.
        
        Returns:
            None
        
        Raises:
            FileNotFoundError: If model weights file does not exist.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Missing model weights {self.model_path}")
        state = torch.load(self.model_path, map_location=self.device)

        # if "model" in state:
        #     self.model.load_state_dict(state["model"])
        # else:
        #     self.model.load_state_dict(state)
        try:
            if "model" in state:
                self.model.load_state_dict(state["model"])
            else:
                self.model.load_state_dict(state)
        except Exception as e:
            # self.logger.error(f"Failed to load weights from {self.model_path}: {e}")
            raise

        self.model.eval()
        self.logger.info(f"Loaded model weights from {self.model_path}")

    def predict_tensor(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Run inference on a single image tensor.
        
        Args:
            tensor (torch.Tensor): Image tensor of shape [C, H, W] (normalized).
        
        Returns:
            np.ndarray: Prediction mask of shape [H, W] with class indices (0, 1, 2).
        """
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

    def predict_sliding_window(
        self, image: np.ndarray, tile_size: int, overlap: float = 0.25
    ) -> np.ndarray:
        """
        Perform inference using sliding window to preserve resolution.
        """
        h, w = image.shape[:2]
        stride = int(tile_size * (1 - overlap))

        # Initialize outputs
        # shape: [3, H, W] for 3 classes
        prob_map = torch.zeros((3, h, w), device=self.device)
        count_map = torch.zeros((1, h, w), device=self.device)

        # Preprocessing (Normalization matches training)
        # Manually normalize since we aren't using Albumentations pipeline here easily
        img_norm = image.astype(np.float32) / 255.0
        img_norm = (img_norm - np.array([0.485, 0.456, 0.406])) / np.array(
            [0.229, 0.224, 0.225]
        )
        tensor_img = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()  # [C, H, W]

        # Sliding window
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y2 = min(h, y + tile_size)
                x2 = min(w, x + tile_size)
                y1 = max(0, y2 - tile_size)
                x1 = max(0, x2 - tile_size)

                crop = tensor_img[:, y1:y2, x1:x2].unsqueeze(0).to(self.device)

                with torch.no_grad():
                    # Get raw logits
                    output = self.model(crop)
                    # Apply softmax to get probabilities
                    output = torch.nn.functional.softmax(output, dim=1)

                prob_map[:, y1:y2, x1:x2] += output.squeeze(0)
                count_map[:, y1:y2, x1:x2] += 1.0

        # Average results
        prob_map /= count_map

        # Argmax to get class indices
        pred_mask = torch.argmax(prob_map, dim=0).cpu().numpy().astype(np.uint8)

        return pred_mask

    def predict_image(self, image: np.ndarray) -> np.ndarray:
        """
        Run inference directly on an RGB numpy image.
        Resizing and conversion done here.
        """
        img = cv2.resize(image, (self.image_size, self.image_size))
        img_t = torch.tensor(img.transpose(2, 0, 1)).float() / 255.0
        return self.predict_tensor(img_t)

    def _resize_prediction_to_original(self, pred_mask, original_shape):
        """
        Resize prediction mask back to original image dimensions.
        """
        orig_h, orig_w = original_shape[:2]
        if pred_mask.shape != (orig_h, orig_w):
            pred_mask = cv2.resize(
                pred_mask.astype(np.uint8),
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )
        return pred_mask

    def run_on_folder(self, image_dir: Path, transform=None, num_samples: int = None):
        """
        Run prediction on a folder using the same preprocessing as training.
        """
        # Use the same transforms as training for consistency
        if transform is None:
            from src.data.augmentations import get_val_augmentations

            transform = get_val_augmentations(self.image_size)

        # Use ImageOnlyDataset with our fixed normalization
        from src.data.loaders import ImageOnlyDataset  # Our fixed version

        dataset = ImageOnlyDataset(image_dir, transform=transform)

        results = []
        total = len(dataset) if num_samples is None else min(num_samples, len(dataset))

        for idx in range(total):
            name, img_t = dataset[idx]

            # Load original image for overlay (full resolution)
            img_path = image_dir / name
            original_img = cv2.imread(str(img_path))
            if original_img is None:
                continue
            original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

            # Predict on processed tensor
            with torch.no_grad():
                pred = self.model(img_t.unsqueeze(0).to(self.device))
                pred_classes = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

            # Resize prediction to original image size
            if pred_classes.shape != original_img.shape[:2]:
                pred_classes = cv2.resize(
                    pred_classes.astype(np.uint8),
                    (original_img.shape[1], original_img.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            # Create overlay
            mask_rgb = np.zeros_like(original_img)
            mask_rgb[pred_classes == 1] = [0, 255, 0]  # Individual = Green
            mask_rgb[pred_classes == 2] = [255, 255, 0]  # Group = Yellow

            overlay = cv2.addWeighted(original_img, 0.7, mask_rgb, 0.3, 0)

            results.append(
                {
                    "name": name,
                    "image": original_img,
                    "mask": pred_classes,
                    "overlay": overlay,
                }
            )

            if (idx + 1) % 5 == 0:
                print(f"Processed {idx + 1}/{total} images")

        return results
