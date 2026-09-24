import torch

from src.utils.helpers import p


def load_best_model(model_name: str, config, notebook="10", mode="rgb", filters=None, **model_kwargs):
    """
    Load the best trained model with correct path structure.
    
    Constructs model directory path from configuration and training parameters,
    uses VersionManager to find latest version, and loads weights.
    
    Args:
        model_name (str): Name of model architecture (e.g., 'unet', 'simple_cnn').
        config: Configuration object with paths and train settings.
        notebook (str): Notebook version identifier. Defaults to "10".
        mode (str): Processing mode ('rgb', 'filtered', 'concat'). Defaults to 'rgb'.
        filters (list or str, optional): Filter names if mode='filtered' or 'concat'.
        **model_kwargs: Extra build_model args (e.g. encoder_name for SMP).
    
    Returns:
        torch.nn.Module: Model loaded on appropriate device (CUDA or CPU) in eval mode.
    
    Raises:
        RuntimeError: If model directory or weights not found.
    """
    from src.models.zoo import build_model
    from src.utils.versioning import VersionManager

    # Construct the exact path used by training
    model_dir = config.paths.models / notebook / model_name / mode

    # Add filters to path if used
    if filters:
        if isinstance(filters, list):
            filter_str = "_".join(filters)
        else:
            filter_str = str(filters)
        model_dir = model_dir / filter_str

    # Add image size to path
    model_dir = model_dir / f"size_{config.train.image_size}"

    p(f"Looking for models in: {model_dir}")

    if not model_dir.exists():
        print(f"Model directory doesn't exist: {model_dir}")
        # Show what actually exists
        base_dir = config.paths.models
        if base_dir.exists():
            print("Available structure:")
            for item in base_dir.rglob("*"):
                if item.is_dir() or item.suffix == ".pth":
                    print(f"  {item.relative_to(base_dir)}")
        raise RuntimeError(f"No model directory found: {model_dir}")

    # Use VersionManager to find latest version
    vm = VersionManager(model_dir)
    version_dir = vm.find_latest()

    if version_dir is None:
        print(f"No version directories found in: {model_dir}")
        print("Contents:")
        for item in model_dir.glob("*"):
            print(f"  {item.name}")
        raise RuntimeError("No trained models found")

    # Try best_model.pth first, then checkpoint.pth
    best_path = version_dir / "best_model.pth"
    checkpoint_path = version_dir / "checkpoint.pth"

    if best_path.exists():
        model_path = best_path
        print(f"Loading best model from: {best_path}")
    elif checkpoint_path.exists():
        model_path = checkpoint_path
        print(f"Loading checkpoint from: {checkpoint_path}")
    else:
        print(f"No model files found in: {version_dir}")
        print("Available files:")
        for item in version_dir.glob("*"):
            print(f"  {item.name}")
        raise RuntimeError(f"No model weights found in: {version_dir}")

    # Determine input channels based on mode and filters
    if mode == "concat" and filters:
        in_channels = 6  # RGB + 3 filters
    else:
        in_channels = 3  # RGB or filtered RGB

    # Build and load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        model_name,
        in_channels=in_channels,
        out_channels=3,
        **model_kwargs,
    )

    # Load weights
    state = torch.load(model_path, map_location=device)
    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)

    model.to(device)
    model.eval()

    print(f"Successfully loaded model from: {model_path}")
    return model


def diagnose_model_directory(config):
    """
    Diagnose and display actual model directory structure.
    
    Prints out full directory tree and file structure to help debug missing models.
    
    Args:
        config: Configuration object with paths settings.
    """
    models_dir = config.paths.models
    print(f"Models directory: {models_dir}")
    print(f"Exists: {models_dir.exists()}")

    if models_dir.exists():
        print("\nDirectory structure:")
        for item in models_dir.rglob("*"):
            if item.is_dir() or item.suffix == ".pth":
                depth = len(item.relative_to(models_dir).parts)
                indent = "  " * (depth - 1)
                print(f"{indent}{item.name}")
