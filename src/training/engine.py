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
    # weights = torch.tensor([1.00, 2.50, 7.00])  # [background, individual, group]
    weights = torch.tensor([1.0, 5.0, 5.0])  # [background, individual, group]
    return torch.nn.CrossEntropyLoss(
        weight=weights.cuda() if torch.cuda.is_available() else weights
    )


# look for this after train_loader to verify class inbalance
# all_masks = []
# for _, mask in train_loader:
#     all_masks.append(mask.flatten())
# all_masks = torch.cat(all_masks)
# print(f"Class distribution: {torch.bincount(all_masks, minlength=3)}")
# print(f"Class percentages: {torch.bincount(all_masks, minlength=3).float() / len(all_masks) * 100}")


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

    n_classes = len(
        torch.unique(torch.cat([m.flatten() for _, m in train_loader.dataset]))
    )

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
