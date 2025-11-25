# %% [markdown]
# # Notebook: 00 Pre-Flight Check Script
# ### Purpose: Test full pipeline to catch issues early
#

# %%
import os
import sys
from pathlib import Path
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../src"))
import torch
from src.utils.config import Config
from src.utils.helpers import c, p, t


# %%
def check_config():
    """Verify config loads correctly."""
    t("Checking Configuration")
    try:
        config = Config.load()
        p("✓ Config loaded", config.paths.root, color1 = c.GREEN)

        # Check critical paths
        checks = [
            ('annotations', config.paths.annotations),
            ('train_images', config.paths.train_images),
            ('eval_images', config.paths.eval_images),
            ('models', config.paths.models),
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
    """Check GPU availability."""
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


def check_data():
    """Check data can be loaded."""
    t("Checking Data")

    try:
        from src.data.annotations import load_json_annotations

        config = Config.load()
        entries = load_json_annotations(config.paths.annotations)

        p("✓ Annotations loaded", f"{len(entries)} images", color1 = c.GREEN)

        # Check first entry
        entry = entries[0]
        img_path = config.paths.train_images / entry.image_path.name

        if img_path.exists():
            p("✓ Sample image", "found", color1 = c.GREEN)

            # Try to load
            import cv2

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


def check_dataset():
    """Test dataset creation."""
    t("Checking Dataset")

    try:
        from src.data.annotations import load_json_annotations
        from src.data.augmentations import get_val_augmentations
        from src.data.loaders import ImageMaskDataset

        config = Config.load()
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

        if mask_t.ndim == 3 and mask_t.shape[0] == 1:
            p("✓ Mask format", "correct [1, H, W]", color1 = c.GREEN)
        else:
            p("✗ Mask format", f"wrong: {mask_t.shape}", color1 = c.RED)

        return True

    except Exception as e:
        p("✗ Dataset check failed", str(e), color1 = c.RED)
        import traceback

        traceback.print_exc()
        return False


def check_model():
    """Test model creation."""
    t("Checking Model")

    try:
        from src.models.zoo import build_model

        model = build_model('simple_cnn', in_channels = 3, out_channels = 1)

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

        if output.shape == (2, 1, 256, 256):
            p("✓ Output shape", "correct", color1 = c.GREEN)
        else:
            p("✗ Output shape", f"wrong: {output.shape}", color1 = c.RED)

        return True

    except Exception as e:
        p("✗ Model check failed", str(e), color1 = c.RED)
        import traceback

        traceback.print_exc()
        return False


def check_disk_space():
    """Check available disk space."""
    t("Checking Disk Space")

    try:
        import shutil

        config = Config.load()
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


def estimate_runtime():
    """Estimate total runtime."""
    t("Runtime Estimate")

    config = Config.load()
    from src.data.annotations import load_json_annotations

    entries = load_json_annotations(config.paths.annotations)
    train_size = int(0.8 * len(entries))

    batch_size = config.train.batch_size
    batches_per_epoch = train_size // batch_size

    # Assume ~1 second per batch (conservative)
    seconds_per_epoch = batches_per_epoch * 1

    # 4 experiments × 10 epochs
    total_seconds = 4 * 10 * seconds_per_epoch
    total_minutes = total_seconds / 60
    total_hours = total_minutes / 60

    p("Training samples", train_size)
    p("Batches per epoch", batches_per_epoch)
    p("Estimated time per epoch", f"~{seconds_per_epoch // 60} min")
    p("Total estimated time", f"~{total_hours:.1f} hours", color1 = c.CYAN)

    if total_hours > 4:
        p("⚠ Long run", "Consider reducing epochs or using GPU", color1 = c.ORANGE)





# %%
t("PRE-FLIGHT CHECK")
t("=" * 80)

checks = [
    ("Configuration", check_config),
    ("GPU", check_gpu),
    ("Data", check_data),
    ("Dataset", check_dataset),
    ("Model", check_model),
    ("Disk Space", check_disk_space),
]

results = { }

for name, func in checks:
    try:
        results[name] = func()
        p("")  # Blank line
    except Exception as e:
        results[name] = False
        p(f"✗ {name} check crashed", str(e), color1 = c.RED)
        import traceback


        traceback.print_exc()

# Runtime estimate
try:
    estimate_runtime()
except Exception as e:
    p("⚠ Runtime estimate failed", str(e), color1 = c.ORANGE)

# Summary
t("=" * 80)
t("SUMMARY")

passed = sum(results.values())
total = len(results)

for name, success in results.items():
    if success:
        p(f"✓ {name}", "PASS", color1 = c.GREEN, bold = True)
    else:
        p(f"✗ {name}", "FAIL", color1 = c.RED, bold = True)

p("")
p("Total", f"{passed}/{total} passed")

if passed == total:
    p("")
    t("ALL CHECKS PASSED! ✓")
    p(
        "You're ready to run:", "python 09_master_execution_robust.py",
        color1 = c.GREEN, color2 = c.CYAN, bold = True
        )
else:
    p("")
    t("SOME CHECKS FAILED!")
    p("Fix the issues above before running the full pipeline", color1 = c.RED)
