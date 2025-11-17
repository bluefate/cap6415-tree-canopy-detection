from pathlib import Path

import torch

from src.models.zoo import build_model
from src.training.trainer import Trainer
from src.utils.config import Config


def prepare_optimizer( model: torch.nn.Module, lr: float ):
    """
    Build Adam optimizer for the given model.
    """
    return torch.optim.Adam(model.parameters(), lr = lr)


def prepare_criterion():
    """
    Binary cross entropy loss for segmentation.
    """
    return torch.nn.BCELoss()


def run_training(
        config: Config,
        train_loader,
        val_loader,
        version_root: Path,
        model_name: str = "unet",
):
    """
    High level training utility.
    Builds model, optimizer, criterion, and Trainer. Then runs training.
    """
    image_size = config.train.image_size
    lr = config.train.learning_rate

    model = build_model(model_name, in_channels = 3, out_channels = 1)
    optimizer = prepare_optimizer(model, lr)
    criterion = prepare_criterion()

    trainer = Trainer(
            model = model,
            optimizer = optimizer,
            criterion = criterion,
            train_loader = train_loader,
            val_loader = val_loader,
            config = config,
            version_root = version_root,
    )

    trainer.run()
    return trainer
