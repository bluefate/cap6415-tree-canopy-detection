import random
from pathlib import Path
from typing import Any, Dict

import psutil
import torch
from torch.utils.data import DataLoader

from src.training.metrics import compute_metrics_multiclass
from src.utils.logging import Logger
from src.utils.versioning import VersionManager


def create_splits(entries, seed=42):
    """Create consistent train/val splits for all experiments."""
    random.seed(seed)
    shuffled_entries = entries.copy()
    random.shuffle(shuffled_entries)

    split_idx = int(0.8 * len(shuffled_entries))
    train_entries = shuffled_entries[:split_idx]
    val_entries = shuffled_entries[split_idx:]

    return train_entries, val_entries


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
        self.criterion = criterion.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
            "val_iou": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
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

        self.logger.info(f"CPU cores: {psutil.cpu_count()}")
        self.logger.info(f"RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
        self.logger.info(f"Disk space: {psutil.disk_usage('/').free / 1e9:.1f} GB")

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
            "history": self.history,
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
        self.history = data.get("history", self.history)
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
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0

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
                total_precision += m.get("precision", 0.0)
                total_recall += m.get("recall", 0.0)

                # Calculate F1 score from precision and recall
                prec = m.get("precision", 0.0)
                rec = m.get("recall", 0.0)
                f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
                total_f1 += f1

        n = max(1, len(self.val_loader))

        return {
            "loss": total_loss / n,
            "iou": total_iou / n,
            "iou_individual_tree": total_iou_individual / n,
            "iou_group_of_trees": total_iou_group / n,
            "acc": total_acc / n,
            "precision": total_precision / n,
            "recall": total_recall / n,
            "f1_score": total_f1 / n,
            "dice": total_f1 / n,  # Use F1 as Dice approximation
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
                f"IoU {val['iou']:.4f} (ind={val['iou_individual_tree']:.4f}, "
                f"grp={val['iou_group_of_trees']:.4f}), "
                f"Acc {val['acc']:.4f}, "
                f"Prec {val['precision']:.4f}, "
                f"Rec {val['recall']:.4f}, "
                f"F1 {val['f1_score']:.4f}"
            )

            # store epoch history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val["loss"])
            self.history["val_iou"].append(val["iou"])
            self.history["val_precision"].append(val["precision"])
            self.history["val_recall"].append(val["recall"])
            self.history["val_f1"].append(val["f1_score"])
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["lr"].append(current_lr)

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
