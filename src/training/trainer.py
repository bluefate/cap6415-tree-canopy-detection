from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from src.training.metrics import compute_metrics
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
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.version_mgr = VersionManager(version_root)
        self.version_dir = self.version_mgr.resolve_version(self._extract_cfg())
        self.paths = self.version_mgr.get_paths(self.version_dir)

        self.logger = Logger(self.paths["log"])
        self.logger.header("Training started")

        self.logger.header(self.model)

        self.scaler = torch.cuda.amp.GradScaler(enabled = (self.device.type == "cuda"))
        self.best_val_loss = float("inf")
        self.start_epoch = 0

        if self.paths["checkpoint"].exists():
            self._load_checkpoint()

    def _extract_cfg( self ) -> Dict[str, Any]:
        """
        Convert the config object into a dictionary.
        """
        return {
            "paths": self.cfg.paths.dict(),
            "train": self.cfg.train.dict(),
            "extra": self.cfg.extra,
        }

    def _save_checkpoint( self, epoch: int, is_best: bool ) -> None:
        """
        Save model, optimizer, scaler, and metrics.
        """
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        torch.save(state, self.paths["checkpoint"])
        if is_best:
            torch.save(state, self.paths["best"])

    def _load_checkpoint( self ) -> None:
        """
        Resume training from the latest checkpoint.
        """
        data = torch.load(self.paths["checkpoint"], map_location = self.device)
        self.model.load_state_dict(data["model"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.scaler.load_state_dict(data["scaler"])
        self.best_val_loss = data.get("best_val_loss", float("inf"))
        self.start_epoch = data.get("epoch", 0) + 1
        self.logger.info(f"Resuming from epoch {self.start_epoch}")

    def train_epoch( self ) -> float:
        """
        Train one epoch and return the average loss.
        """
        self.model.train()
        total_loss = 0.0

        for images, masks in self.train_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled = (self.device.type == "cuda")):
                preds = self.model(images)
                loss = self.criterion(preds, masks)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / max(1, len(self.train_loader))

    def validate_epoch( self ) -> Dict[str, float]:
        """
        Validate one epoch and return metrics.
        """
        self.model.eval()
        total_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0
        total_acc = 0.0

        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                preds = self.model(images)
                loss = self.criterion(preds, masks)
                total_loss += loss.item()

                m = compute_metrics(preds, masks)
                total_iou += m["iou"]
                total_dice += m["dice"]
                total_acc += m["acc"]

        n = max(1, len(self.val_loader))
        return {
            "loss": total_loss / n,
            "iou": total_iou / n,
            "dice": total_dice / n,
            "acc": total_acc / n,
        }

    def run( self ) -> None:
        """
        Run full training loop using configuration settings.
        """
        epochs = self.cfg.train.epochs
        patience = self.cfg.train.early_stop_patience
        no_improve = 0

        for epoch in range(self.start_epoch, epochs):
            self.logger.warn(f"Epoch {epoch + 1}")

            train_loss = self.train_epoch()
            val = self.validate_epoch()

            self.logger.info(f"Train loss {train_loss:.4f}, Val loss {val['loss']:.4f}, IoU {val['iou']:.4f}, Dice {val['dice']:.4f}, Acc {val['acc']:.4f}")

            is_best = val["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val["loss"]
                no_improve = 0
                self.logger.info("New best model")
            else:
                no_improve += 1

            self._save_checkpoint(epoch, is_best)

            if no_improve >= patience:
                self.logger.info("Early stop triggered")
                break

        self.logger.warn("Training complete")
