import shutil
from pathlib import Path

import cv2
import torch

from data.annotations import load_json_annotations
from data.augmentations import get_val_augmentations
from data.loaders import ImageMaskDataset
from models.zoo import build_model
from src.utils.helpers import c, p, t


def check_config(config):
    """
    Verify configuration loads correctly and all critical paths exist.
    
    Args:
        config: Configuration object to validate.
    
    Returns:
        bool: True if all checks pass, False otherwise.
    """
    t("Checking Configuration")
    try:
        p("✓ Config loaded", config.paths.root, color1 = c.GREEN)

        # Check critical paths
        checks = [
            ('annotations', config.paths.annotations),
            ('train_images', config.paths.train_images),
            ('eval_images', config.paths.eval_images),
            ('models', config.paths.models),
            ('notebooks', config.paths.notebooks),
        ]

        for name, path in checks:
            if path and Path(path).exists():
                p(f"✓ {name}", "exists", color1 = c.GREEN)
            else:
                p(f"✗ {name}", f"missing: {path}", color1 = c.RED)

        return True

    except Exception as e:
        p("✗ Config failed", str(e), color1 = c.RED)
        return False


def check_gpu():
    """
    Check GPU availability, device count, and memory capacity.
    
    Tests CUDA availability and performs a test memory allocation to verify GPU is working.
    
    Returns:
        bool: True if GPU is available and working, False if CPU only.
    """
    t("Checking GPU")

    if torch.cuda.is_available():
        p("✓ CUDA available", torch.cuda.get_device_name(0), color1 = c.GREEN)
        p("GPU count", torch.cuda.device_count())

        # Check memory
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        p("GPU memory", f"{total:.1f} GB")

        # Test allocation
        try:
            test = torch.zeros((1000, 1000)).cuda()
            del test
            torch.cuda.empty_cache()
            p("✓ GPU allocation", "working", color1 = c.GREEN)
        except Exception as e:
            p("✗ GPU allocation", str(e), color1 = c.RED)

        return True
    else:
        p("✗ CUDA not available", "will use CPU (slow)", color1 = c.ORANGE)
        return False


def check_data(config):
    """
    Verify data can be loaded from configured paths.
    
    Checks that annotation file exists, contains valid entries, and sample images can be loaded.
    
    Args:
        config: Configuration object with data paths.
    
    Returns:
        bool: True if all data checks pass, False otherwise.
    """
    t("Checking Data")

    try:
        entries = load_json_annotations(config.paths.annotations)

        p("✓ Annotations loaded", f"{len(entries)} images", color1 = c.GREEN)

        # Check first entry
        entry = entries[0]
        img_path = config.paths.train_images / entry.image_path.name

        if img_path.exists():
            p("✓ Sample image", "found", color1 = c.GREEN)

            img = cv2.imread(str(img_path))
            if img is not None:
                p("✓ Image loading", f"shape={img.shape}", color1 = c.GREEN)
            else:
                p("✗ Image loading", "failed", color1 = c.RED)
        else:
            p("✗ Sample image", f"not found: {img_path}", color1 = c.RED)

        return True

    except Exception as e:
        p("✗ Data check failed", str(e), color1 = c.RED)
        return False


def check_dataset(config):
    """
    Test dataset creation and sample loading.
    
    Verifies that ImageMaskDataset can be instantiated and samples can be loaded with correct shapes.
    
    Args:
        config: Configuration object with training parameters.
    
    Returns:
        bool: True if dataset checks pass, False otherwise.
    """
    t("Checking Dataset")

    try:
        entries = load_json_annotations(config.paths.annotations)
        transform = get_val_augmentations(config.train.image_size)

        dataset = ImageMaskDataset(
                entries[:5],
                config.paths.train_images,
                transform = transform
        )

        p("✓ Dataset created", f"{len(dataset)} samples", color1 = c.GREEN)

        # Test loading
        img_t, mask_t = dataset[0]

        p("✓ Image shape", img_t.shape, color1 = c.GREEN)
        p("✓ Mask shape", mask_t.shape, color1 = c.GREEN)

        # Verify shapes
        if img_t.ndim == 3 and img_t.shape[0] == 3:
            p("✓ Image format", "correct [3, H, W]", color1 = c.GREEN)
        else:
            p("✗ Image format", f"wrong: {img_t.shape}", color1 = c.RED)

        # Multi-class segmentation: masks should be [H, W] with class indices (0, 1, 2)
        if mask_t.ndim == 2:
            p("✓ Mask format", f"correct [H, W] for multi-class", color1 = c.GREEN)
            unique_vals = torch.unique(mask_t)
            if torch.all((unique_vals >= 0) & (unique_vals <= 2)):
                p("✓ Mask values", f"valid classes: {unique_vals.tolist()}", color1 = c.GREEN)
            else:
                p("✗ Mask values", f"invalid: {unique_vals.tolist()}", color1 = c.RED)
        elif mask_t.ndim == 3 and mask_t.shape[0] == 1:
            p("⚠ Mask format", "[1, H, W] - should be [H, W] for CrossEntropyLoss", color1 = c.ORANGE)
        else:
            p("✗ Mask format", f"wrong: {mask_t.shape}", color1 = c.RED)

        return True

    except Exception as e:
        p("✗ Dataset check failed", str(e), color1 = c.RED)

        # traceback.print_exc()
        return False


def check_model():
    """
    Test model creation and forward pass.
    
    Verifies that a simple CNN model can be created, parameters counted,
    and a forward pass produces correct output shape.
    
    Returns:
        bool: True if model checks pass, False otherwise.
    """
    t("Checking Model")

    try:
        model = build_model('simple_cnn', in_channels = 3, out_channels = 3)

        p("✓ Model created", "simple_cnn", color1 = c.GREEN)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        p("Model parameters", f"{total_params:,}")

        # Test forward pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        test_input = torch.randn(2, 3, 256, 256).to(device)

        with torch.no_grad():
            output = model(test_input)

        p("✓ Forward pass", f"output shape={output.shape}", color1 = c.GREEN)

        # Multi-class segmentation: 3 channels (background, individual_tree, group_of_trees)
        expected_shape = (2, 3, 256, 256)

        if output.shape == expected_shape:
            p("✓ Output shape", "correct (multi-class)", color1 = c.GREEN)
        else:
            p("✗ Output shape", f"expected {expected_shape}, got {output.shape}", color1 = c.RED)

        return True

    except Exception as e:
        p("✗ Model check failed", str(e), color1 = c.RED)

        # traceback.print_exc()
        return False


def check_disk_space(config):
    """
    Check available disk space at project root.
    
    Args:
        config: Configuration object with root path.
    
    Returns:
        None (prints disk space information to console).
    """
    t("Checking Disk Space")

    try:
        total, used, free = shutil.disk_usage(config.paths.root)

        free_gb = free / (1024 ** 3)

        p("Free space", f"{free_gb:.1f} GB")

        if free_gb > 10:
            p("✓ Sufficient space", ">10 GB available", color1 = c.GREEN)
            return True
        elif free_gb > 5:
            p("⚠ Limited space", f"{free_gb:.1f} GB (needs >10 GB)", color1 = c.ORANGE)
            return True
        else:
            p("✗ Insufficient space", f"{free_gb:.1f} GB (needs >10 GB)", color1 = c.RED)
            return False

    except Exception as e:
        p("✗ Disk check failed", str(e), color1 = c.ORANGE)
        return True






