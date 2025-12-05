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


# Cross-Entropy Loss
# def prepare_criterion():
#     """Multiclass cross entropy + dice loss for 3-class segmentation."""
#     return torch.nn.CrossEntropyLoss()

def prepare_criterion():
    """Weighted cross entropy for imbalanced 3-class segmentation."""
    weights = torch.tensor([0.5, 2.0, 2.0])  # [background, individual, group]
    return torch.nn.CrossEntropyLoss(weight=weights.cuda() if torch.cuda.is_available() else weights)

def run_training(
    config: Config,
    train_loader,
    val_loader,
    version_root: Path,
    model_name: str = "unet",
    in_channels: int = 3,
):
    """
    Builds model, optimizer, criterion, and Trainer. Then runs training.
    """
    image_size = config.train.image_size
    lr = config.train.learning_rate

    model = build_model(model_name, in_channels=in_channels, out_channels=3)
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
