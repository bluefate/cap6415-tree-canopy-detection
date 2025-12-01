# From C:\github\Tree-Canopy-Detection\src\training\engine.py
from pathlib import Path

import torch

from src.models.zoo import build_model
from src.training.trainer import Trainer
from src.utils.config import Config


def prepare_optimizer(model: torch.nn.Module, lr: float):
    """
    Build Adam optimizer for the given model.
    """
    return torch.optim.Adam(model.parameters(), lr=lr)


def prepare_criterion():
    """Multiclass cross entropy + dice loss for 3-class segmentation."""
    return torch.nn.CrossEntropyLoss()


def run_training(
    config: Config,
    train_loader,
    val_loader,
    version_root: Path,
    model_name: str = "unet",
):
    """
    Builds model, optimizer, criterion, and Trainer. Then runs training.
    """
    image_size = config.train.image_size
    lr = config.train.learning_rate

    model = build_model(model_name, in_channels=3, out_channels=3)
    optimizer = prepare_optimizer(model, lr)
    criterion = prepare_criterion()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        version_root=version_root,
    )

    trainer.run()
    return trainer


# From C:\github\Tree-Canopy-Detection\src\training\metrics.py
import numpy as np
import torch
import torch.nn.functional as F


def _to_numpy(pred: torch.Tensor, true: torch.Tensor):
    """
    Convert prediction and ground truth tensors to numpy arrays.
    Applies sigmoid to prediction if needed, then thresholds at 0.5.
    """
    if isinstance(pred, torch.Tensor):
        pred = torch.sigmoid(pred).detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()

    pred_bin = (pred > 0.5).astype(np.uint8)
    true_bin = (true > 0.5).astype(np.uint8)
    return pred_bin, true_bin


def compute_confusion(pred_bin: np.ndarray, true_bin: np.ndarray):
    """
    Compute basic confusion counts.
    """
    tp = np.logical_and(pred_bin == 1, true_bin == 1).sum()
    fp = np.logical_and(pred_bin == 1, true_bin == 0).sum()
    fn = np.logical_and(pred_bin == 0, true_bin == 1).sum()
    tn = np.logical_and(pred_bin == 0, true_bin == 0).sum()
    return tp, fp, fn, tn


def compute_metrics(pred: torch.Tensor, true: torch.Tensor):
    """
    Compute IoU, Dice, Accuracy for binary segmentation.
    """
    # Handle SegFormer output format
    if hasattr(pred, "logits"):
        pred = pred.logits

    # Convert to tensor if needed
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred)

    # Get target spatial size
    if isinstance(true, torch.Tensor):
        target_size = true.shape[-2:]  # (H, W)
    else:
        target_size = true.shape[-2:]

    # Resize prediction to match target size if needed
    if pred.shape[-2:] != target_size:
        pred = F.interpolate(
            pred, size=target_size, mode="bilinear", align_corners=False
        )

    # Apply sigmoid for binary segmentation
    if isinstance(pred, torch.Tensor):
        pred = torch.sigmoid(pred).detach().cpu().numpy()

    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()

    pred_bin = pred > 0.5
    true_bin = true > 0.5

    tp = (pred_bin & true_bin).sum()
    fp = (pred_bin & ~true_bin).sum()
    fn = (~pred_bin & true_bin).sum()
    tn = (~pred_bin & ~true_bin).sum()

    inter = float(tp)
    union = float(tp + fp + fn)

    return {
        "iou": inter / union if union > 0 else 0,
        "dice": (2 * tp) / (2 * tp + fp + fn + 1e-8),
        "acc": (tp + tn) / (tp + tn + fp + fn + 1e-8),
        "precision": tp / (tp + fp + 1e-8),
        "recall": tp / (tp + fn + 1e-8),
    }


def compute_metrics_multiclass(
    pred: torch.Tensor, true: torch.Tensor, num_classes: int = 3
):
    """
    Compute per-class IoU and mean IoU for multi-class segmentation.
    """
    # Handle SegFormer output format
    if hasattr(pred, "logits"):
        pred = pred.logits

    # Convert to tensor if needed
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred)

    # Get target spatial size
    if isinstance(true, torch.Tensor):
        target_size = true.shape[-2:]
    else:
        target_size = true.shape[-2:]

    # Resize prediction to match target size if needed
    if pred.shape[-2:] != target_size:
        pred = F.interpolate(
            pred, size=target_size, mode="bilinear", align_corners=False
        )

    # BINARY MODE: If pred has 1 channel, use binary metrics
    if pred.shape[1] == 1:
        # Squeeze channel dimension from true if present
        if isinstance(true, torch.Tensor) and true.ndim == 4 and true.shape[1] == 1:
            true_binary = true.squeeze(1)  # [B, H, W]
        else:
            true_binary = true

        # Apply sigmoid and threshold
        pred_binary = torch.sigmoid(pred).squeeze(1) > 0.5  # [B, H, W]

        if isinstance(true_binary, torch.Tensor):
            true_binary = true_binary > 0.5

            # Convert to numpy
            pred_np = pred_binary.detach().cpu().numpy()
            true_np = true_binary.detach().cpu().numpy()
        else:
            pred_np = pred_binary.detach().cpu().numpy()
            true_np = true_binary > 0.5

        # Compute binary metrics
        tp = (pred_np & true_np).sum()
        fp = (pred_np & ~true_np).sum()
        fn = (~pred_np & true_np).sum()
        tn = (~pred_np & ~true_np).sum()

        inter = float(tp)
        union = float(tp + fp + fn)

        return {
            "iou": inter / union if union > 0 else 0,
            "dice": (2 * tp) / (2 * tp + fp + fn + 1e-8),
            "acc": (tp + tn) / (tp + tn + fp + fn + 1e-8),
            "precision": tp / (tp + fp + 1e-8),
            "recall": tp / (tp + fn + 1e-8),
        }

    # MULTI-CLASS MODE: If pred has C > 1 channels
    # Convert logits to class predictions
    if isinstance(pred, torch.Tensor):
        pred_classes = torch.argmax(pred, dim=1).detach().cpu().numpy()  # [B, H, W]
    else:
        pred_classes = np.argmax(pred, axis=1)

    if isinstance(true, torch.Tensor):
        # Squeeze channel dimension if present
        if true.ndim == 4 and true.shape[1] == 1:
            true = true.squeeze(1)
        true_classes = true.detach().cpu().numpy()  # [B, H, W]
    else:
        true_classes = true

    # Compute IoU for each class
    ious = {}
    class_names = {0: "background", 1: "individual_tree", 2: "group_of_trees"}

    for cls_id in range(num_classes):
        pred_mask = pred_classes == cls_id
        true_mask = true_classes == cls_id

        intersection = np.logical_and(pred_mask, true_mask).sum()
        union = np.logical_or(pred_mask, true_mask).sum()

        iou = float(intersection) / float(union + 1e-8)
        ious[f"iou_{class_names.get(cls_id, f"class_{cls_id}")}"] = iou

    # Mean IoU (excluding background class 0)
    if num_classes > 1:
        tree_ious = [ious[f"iou_{class_names[i]}"] for i in range(1, num_classes)]
        ious["mean_iou"] = sum(tree_ious) / len(tree_ious)
    else:
        ious["mean_iou"] = ious.get("iou_background", 0.0)

    # Overall pixel accuracy
    correct = (pred_classes == true_classes).sum()
    total = pred_classes.size
    ious["acc"] = float(correct) / float(total)

    # Aliases for compatibility
    ious["iou"] = ious["mean_iou"]
    ious["dice"] = 0.0  # Placeholder

    return ious


# From C:\github\Tree-Canopy-Detection\src\training\running.py
from data.enhance_masks import EnhancedImageMaskDataset


def get_available_filters():
    """
    Get list of all available filter names from the system.
    """
    try:
        available = EnhancedImageMaskDataset.get_available_filters()
        return available
    except Exception as e:
        p(
            "Warning",
            f"Could not load filters dynamically: {e}",
            color1=c.ORANGE,
            color2=c.ORANGE,
        )
        # Fallback to known filters
        return [
            "laplacian",
            "sobel",
            "clahe",
            "gaussian_3x3",
            "gaussian_5x5",
            "gaussian_7x7",
            "sobel_x",
            "sobel_y",
            "laplacian_3x3",
            "sharpen_basic",
            "high_pass_3x3",
            "edge_enhance",
            "gaussian_3x3_sigma1",
            "gaussian_5x5_sigma1",
            "gaussian_7x7_sigma1",
        ]


def validate_filter_set(filter_names, available_filters):
    """
    Validate a list of filter names against available filters.
    """
    invalid = []
    suggestions = {}

    for fname in filter_names:
        if fname.lower() not in [f.lower() for f in available_filters]:
            invalid.append(fname)
            # Try to find suggestion
            for avail in available_filters:
                if fname.lower() in avail.lower() or avail.lower() in fname.lower():
                    suggestions[fname] = avail
                    break

    return len(invalid) == 0, invalid, suggestions


# From C:\github\Tree-Canopy-Detection\src\training\trainer.py
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from src.training.metrics import compute_metrics_multiclass
from src.utils.logging import Logger
from src.utils.versioning import VersionManager


class Trainer:
    """
    Handle model training, validation, checkpointing, and versioning.
    Uses mixed precision when CUDA is available.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Any,
        version_root: Path,
    ):
        # define device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
        }

        # versioning paths
        self.version_mgr = VersionManager(version_root)
        self.version_dir = self.version_mgr.resolve_version(self._extract_cfg())
        self.paths = self.version_mgr.get_paths(self.version_dir)

        # Initialize logger
        self.logger = Logger(self.paths["log"], cfg=config)
        self.logger.header("Training started")
        self.logger.info(self.model)
        # LOG MODEL

        # GPU memory optimization
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True  # Speed up training
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(
                f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.train.scheduler_factor,
            patience=config.train.scheduler_patience,
        )

        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda"))

        # checkpointing
        self.best_val_loss = float("inf")
        self.start_epoch = 0

        if self.paths["checkpoint"].exists():
            self._load_checkpoint()

    def _extract_cfg(self) -> Dict[str, Any]:
        """
        Convert the config object into a dictionary.
        """
        return {
            "paths": self.cfg.paths.dict(),
            "train": self.cfg.train.dict(),
            "extra": self.cfg.extra,
        }

    def _save_checkpoint(
        self,
        epoch: int,
        is_best: bool,
        train_loss: float = 0.0,
        val_metrics: dict = None,
    ) -> None:
        """
        Save model, optimizer, scaler, and metrics.
        """
        state = {
            "epoch": epoch,
            "final_epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "train_loss": train_loss,
        }
        if val_metrics is not None:
            state.update(
                {
                    "val_loss": val_metrics.get("loss", 0.0),
                    "val_accuracy": val_metrics.get("acc", 0.0),
                    "val_iou": val_metrics.get("iou", 0.0),
                    "val_iou_individual": val_metrics.get("iou_individual_tree", 0.0),
                    "val_iou_group": val_metrics.get("iou_group_of_trees", 0.0),
                    "val_dice": val_metrics.get("dice", 0.0),
                    "iou": val_metrics.get("iou", 0.0),
                    "accuracy": val_metrics.get("acc", 0.0),
                    "f1_score": val_metrics.get("dice", 0.0),
                    "precision": val_metrics.get("precision", 0.0),
                    "recall": val_metrics.get("recall", 0.0),
                }
            )

        torch.save(state, self.paths["checkpoint"])
        if is_best:
            torch.save(state, self.paths["best"])

    def _load_checkpoint(self) -> None:
        """
        Resume training from the latest checkpoint.
        """
        data = torch.load(self.paths["checkpoint"], map_location=self.device)
        self.model.load_state_dict(data["model"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.scaler.load_state_dict(data["scaler"])
        self.best_val_loss = data.get("best_val_loss", float("inf"))
        self.start_epoch = data.get("epoch", 0) + 1
        self.logger.info(f"Resuming from epoch {self.start_epoch}")

    def train_epoch(self) -> float:
        """
        Train one epoch and return the average loss.
        """
        self.model.train()
        total_loss = 0.0

        for images, masks in self.train_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                preds = self.model(images)
                loss = self.criterion(preds, masks)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            # Clear GPU cache periodically (every 10 batches)
            if self.device.type == "cuda" and (len(self.train_loader) % 10 == 0):
                torch.cuda.empty_cache()

        return total_loss / max(1, len(self.train_loader))

    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate one epoch and return metrics.
        """
        self.model.eval()
        total_loss = 0.0
        total_iou = 0.0
        total_iou_individual = 0.0
        total_iou_group = 0.0
        total_acc = 0.0

        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                preds = self.model(images)
                loss = self.criterion(preds, masks)
                total_loss += loss.item()

                # Use multiclass metrics
                m = compute_metrics_multiclass(preds, masks, num_classes=3)

                total_iou += m.get("iou", m.get("mean_iou", 0.0))
                total_iou_individual += m.get("iou_individual_tree", 0.0)
                total_iou_group += m.get("iou_group_of_trees", 0.0)
                total_acc += m["acc"]

        n = max(1, len(self.val_loader))

        return {
            "loss": total_loss / n,
            "iou": total_iou / n,  # This is now mean_iou of tree classes
            "iou_individual_tree": total_iou_individual / n,
            "iou_group_of_trees": total_iou_group / n,
            "dice": 0.0,  # Placeholder
            "acc": total_acc / n,
        }

    def run(self) -> None:
        """
        Run full training loop using configuration settings.
        """
        epochs = self.cfg.train.epochs
        patience = self.cfg.train.early_stop_patience
        no_improve = 0

        for epoch in range(self.start_epoch, epochs):
            print()
            self.logger.warn(f"Epoch {epoch + 1}")

            train_loss = self.train_epoch()
            val = self.validate_epoch()

            self.logger.info(
                f"Train loss {train_loss:.4f}, "
                f"Val loss {val['loss']:.4f}, "
                f"IoU {val['iou']:.4f} (ind={val['iou_individual_tree']:.4f},"
                f"grp={val['iou_group_of_trees']:.4f}), "
                f"Acc {val['acc']:.4f}"
            )

            is_best = val["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val["loss"]
                no_improve = 0
                self.logger.info("New best model")
            else:
                no_improve += 1

            self._save_checkpoint(epoch, is_best, train_loss, val)

            if no_improve >= patience:
                self.logger.info("Early stop triggered")
                break

            self.scheduler.step(val["loss"])

        self.logger.warn("Training complete")


# From C:\github\Tree-Canopy-Detection\src\training\__init__.py


