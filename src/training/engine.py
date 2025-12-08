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
    Create Adam optimizer for model training.
    
    Args:
        model (torch.nn.Module): Model to optimize.
        lr (float): Learning rate.
    
    Returns:
        torch.optim.Adam: Configured Adam optimizer.
    """
    return torch.optim.Adam(model.parameters(), lr=lr)


# Cross-Entropy Loss
# def prepare_criterion():
#     """Multiclass cross entropy + dice loss for 3-class segmentation."""
#     return torch.nn.CrossEntropyLoss()


def prepare_criterion():
    """
    Create weighted cross-entropy loss for imbalanced 3-class segmentation.
    
    Classes: 0=background (weight 0.5), 1=individual_tree (weight 2.0), 2=group_of_trees (weight 3.0).
    
    Returns:
        torch.nn.CrossEntropyLoss: Loss function with class weights.
    """
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
    Setup and execute model training.
    
    Builds model, optimizer, and loss function, then runs training loop via Trainer.
    
    Args:
        config (Config): Configuration object with training parameters.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        version_root (Path): Directory for saving checkpoints and logs.
        model_name (str): Name of model architecture. Defaults to "unet".
        in_channels (int): Number of input channels. Defaults to 3.
    
    Returns:
        Trainer: Trained trainer instance with results.
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
