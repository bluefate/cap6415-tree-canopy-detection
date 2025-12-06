from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss

from src.data.masks import CLASS_TO_ID
from src.models.zoo import build_model
from src.prediction.validation import validate_data_loader
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # weights = torch.tensor([1.00, 2.50, 7.00], device=device)  # [background, individual, group]
    # weights = torch.tensor([1.0, 5.0, 5.0], device=device)  # [background, individual, group]
    weights = torch.tensor(
        [0.5, 2.0, 3.0], device=device
    )  # [background, individual, group]

    return CrossEntropyLoss(weight=weights)


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

    validate_data_loader(train_loader, "Training")
    validate_data_loader(val_loader, "Validation")

    n_classes = len(CLASS_TO_ID) + 1  # +1 for background

    model = build_model(model_name, in_channels=in_channels, out_channels=n_classes)
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
