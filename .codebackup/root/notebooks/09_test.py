# %% [markdown]
# # Notebook: 09 Test
# ### Purpose: Used for shape verification diagnostic to make sure  data loaders produce correct shapes
#

# %%

from IPython import get_ipython


if not "google.colab" in str(get_ipython()):
    from pathlib import Path


    root = Path("C:/github/Tree-Canopy-Detection")

else:
    from pathlib import Path


    root = Path("/content/CAP6415_F25_project-Tree-Canopy-Detection")

    # noinspection PyUnresolvedReferences
    from google.colab import drive

    import os
    import subprocess
    import sys


    os.chdir("/content")
    drive.mount("/content/drive")

    # Load environment variables
    env_path = "/content/drive/MyDrive/TreeCanopyProject/.env"
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value

    # Clone repository
    repo_path = "/content/CAP6415_F25_project-Tree-Canopy-Detection"
    github_token = os.getenv("TOKEN")

    if not os.path.exists(repo_path):
        if github_token:
            # #!git config --global user.email "jherna65@fau.edu"
            subprocess.run(
                    ["git", "config", "--global", "user.email", "jherna65@fau.edu"],
                    check = True,
            )

            # #!git config --global user.name "bluefate"
            subprocess.run(
                    ["git", "config", "--global", "user.name", "bluefate"], check = True
            )

            clone_url = f"https://bluefate:{github_token}@github.com/bluefate/CAP6415_F25_project-Tree-Canopy-Detection.git"

            # #!git clone $clone_url
            subprocess.run(["git", "clone", clone_url], check = True)
            print("Repository cloned")
        else:
            print("ERROR: No token")
    else:
        print("Repository already exists")

    # Set paths and pull latest
    if os.path.exists(repo_path):
        os.chdir(repo_path)
        if github_token:
            # #!git reset --hard HEAD
            # #!git pull
            subprocess.run(["git", "pull"], check = True)
        sys.path.insert(0, repo_path)
        sys.path.insert(0, os.path.join(repo_path, "src"))
        print("Setup complete")

    # Set paths
    if os.path.exists(repo_path):
        os.chdir(repo_path)
        sys.path.insert(0, repo_path)
        sys.path.insert(0, os.path.join(repo_path, "src"))
        print("Setup complete")

    print("Requirements")
    # Install packages
    # # !pip install -r requirements.txt
    subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check = True
    )

# %% [markdown]
# #### Imports and setup

# %%
import os
import sys

from src.utils.tester import p_test


sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../src"))

p_test()

# %%

import cv2
from src.data.image_loader import apply_all_filters, create_enhanced_image
import torch
from src.models.zoo import list_available_models
from src.data.annotations import load_json_annotations
from src.data.augmentations import get_train_augmentations
from src.data.loaders import ImageMaskDataset
from src.data.enhance_masks import EnhancedImageMaskDataset
from src.utils.config import Config
from src.utils.helpers import p, t, c
from src.models.zoo import build_model
from torch.nn import CrossEntropyLoss



# %%

config = Config.load(root = root)
entries = load_json_annotations(config.paths.annotations)
config.show()


# %%
def print_versions():
    from src.utils.helpers import p
    import sys

    import numpy as np
    import pandas as pd
    import torch


    p("Python", sys.version)
    p("Numpy", np.__version__)
    p("Panda", pd.__version__)
    p("Torch", torch.__version__)


print_versions()

# %% [markdown]
# #### Test 1: Original ImageMaskDataset
#

# %%

t("Testing Original ImageMaskDataset")

train_tf = get_train_augmentations(config.train.image_size)
dataset = ImageMaskDataset(entries[:5], config.paths.train_images, transform = train_tf)

img_t, mask_t = dataset[0]

p("Image shape", img_t.shape)
p("Mask shape", mask_t.shape)
p("Image dtype", img_t.dtype)
p("Mask dtype", mask_t.dtype)

# Expected:
# Image: [3, H, W] or [C, H, W]
# Mask: [1, H, W]

if mask_t.ndim == 2:
    p("✓ SUCCESS", f"Mask shape is correct [H, W] for multi-class = {mask_t.shape}", color1 = c.BLACK, color2 = c.BLUE)
    # Verify class values
    unique_vals = torch.unique(mask_t)
    p("Mask unique values", unique_vals.tolist(), color1 = c.ORANGE)
elif mask_t.ndim == 3 and mask_t.shape[0] == 1:
    p("WARNING", "Mask has [1, H, W] - should be [H, W] for CrossEntropyLoss", color1 = c.ORANGE)
else:
    p("ERROR", f"Unexpected mask shape: {mask_t.shape}", color1 = c.RED)



# %% [markdown]
# #### Test 2: EnhancedImageMaskDataset
#

# %%



config = Config.load(root = root)
entries = load_json_annotations(config.paths.annotations)

# Load test image
entry = entries[0]
img_path = config.paths.train_images / entry.image_path.name
img = cv2.imread(str(img_path))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Apply all filters (assumes apply_all_filters function exists in notebook)
filters = apply_all_filters(img)

# Get top 3 filter names (adjust based on your results)
test_filters = list(filters.keys())[:3]

# Test create_enhanced_image
try:
    enhanced = create_enhanced_image(img, test_filters)
    p("SUCCESS: Enhanced image shape", enhanced.shape, color1 = c.GREEN)
    p("Original image shape", img.shape, color1 = c.GREEN)
    p("Filters used", test_filters, color1 = c.GREEN)
except Exception as e:
    p("ERROR", str(e), color1 = c.RED, color2 = c.RED)
    import traceback


    traceback.print_exc()


# %%
t("Testing EnhancedImageMaskDataset")

for mode in ['rgb', 'filtered']:
    dataset = EnhancedImageMaskDataset(
            entries[:5],
            config.paths.train_images,
            mode = mode,
            transform = train_tf
    )

    img_t, mask_t = dataset[0]

    p(f"Mode: {mode} - Image shape", img_t.shape)
    p(f"Mode: {mode} - Mask shape", mask_t.shape)

    # Verify shapes
    try:
        assert img_t.ndim == 3, f"Image should be 3D, got {img_t.ndim}D"
        assert mask_t.ndim == 2, f"Mask should be 2D [H, W], got {mask_t.shape}D"
        p(f"Mode {mode}", "Shapes are correct!", color1 = c.BLACK, color2 = c.BLUE)
    except AssertionError as e:
        p("ASSERTION FAILED", str(e), color1 = c.RED, color2 = c.RED)


# %% [markdown]
# #### Test 3: DataLoader Batches
#

# %%
from torch.utils.data import DataLoader


t("Testing DataLoader Batches")

dataset = EnhancedImageMaskDataset(
        entries[:10],
        config.paths.train_images,
        mode = 'rgb',
        transform = train_tf
)

loader = DataLoader(dataset, batch_size = 4, shuffle = False)

# Get first batch
images, masks = next(iter(loader))

p("Batch image shape", images.shape)
p("Batch mask shape", masks.shape)

# Expected:
# Images: [batch_size, channels, H, W]  e.g., [4, 3, 256, 256]
# Masks: [batch_size, 1, H, W]          e.g., [4, 1, 256, 256]

try:
    assert images.ndim == 4, f"Batch images should be 4D, got {images.ndim}D"
    assert masks.ndim == 3, f"Batch masks should be 3D [B, H, W], got {masks.ndim}D"
    assert masks.shape[0] == images.shape[0], f"Batch sizes don't match"
    p("SUCCESS", "DataLoader produces correct batch shapes!", color1 = c.BLACK, color2 = c.BLUE)
except AssertionError as e:
    p("ASSERTION FAILED", str(e), color1 = c.RED, color2 = c.RED)


# %% [markdown]
# #### Test 4: Model Forward Pass
#

# %%



t("Testing Model Forward Pass")

# Build model
model = build_model('simple_cnn', in_channels = 3, out_channels = 3)
model.eval()

# Get a batch
images, masks = next(iter(loader))

# Forward pass
with torch.no_grad():
    preds = model(images)

p("Input shape", images.shape)
p("Output shape", preds.shape)
p("Target shape", masks.shape)

# Verify shapes match for loss computation
# CrossEntropyLoss expects: predictions=[B, C, H, W], targets=[B, H, W]
try:
    expected_pred_shape = (masks.shape[0], 3, masks.shape[1], masks.shape[2])
    assert preds.shape == expected_pred_shape, \
        f"Prediction {preds.shape} should be {expected_pred_shape} for target {masks.shape}"
    p("SUCCESS", "Model output shape correct for CrossEntropyLoss!", color1 = c.BLACK, color2 = c.BLUE)
except AssertionError as e:
    p("ASSERTION FAILED", str(e), color1 = c.RED, color2 = c.RED)

# %% [markdown]
# #### Test 5: Loss Computation
#

# %%
t("Testing Loss Computation")

criterion = CrossEntropyLoss()

# Compute loss (should not raise error)
try:
    loss = criterion(preds, masks)
    p("Loss value", loss.item())
    p("SUCCESS", "Loss computation works!", color1 = c.BLACK, color2 = c.BLUE)
except Exception as e:
    p("ERROR", str(e), color1 = c.RED, color2 = c.RED)
    import traceback


    traceback.print_exc()


# %% [markdown]
# #### Test 6: Comprehensive Dataset Verification Function

# %%
def verify_dataset_shapes():
    """
    Comprehensive test that verifies all datasets return correct shapes.
    This is the master verification function that should ALWAYS pass after fixes.
    """

    entries = load_json_annotations(config.paths.annotations)
    train_tf = get_train_augmentations(config.train.image_size, mode = mode)

    test_results = []

    # ========================================================================
    # Test 1: ImageMaskDataset
    # ========================================================================
    try:
        t("Test 1: ImageMaskDataset")
        dataset1 = ImageMaskDataset(entries[:5], config.paths.train_images, transform = train_tf)
        img_t, mask_t = dataset1[0]

        p("Image shape", img_t.shape)
        p("Mask shape", mask_t.shape)

        assert img_t.ndim == 3, f"Image should be 3D, got {img_t.ndim}D"
        assert mask_t.ndim == 2, f"Mask should be 2D [H, W], got {mask_t.shape}D"

        p("Test 1", "PASS", color1 = c.BLACK, color2 = c.BLUE, bold = True)
        test_results.append(("ImageMaskDataset", True, None))

    except Exception as e:
        p("Test 1", f"FAIL: {e}", color1 = c.RED, color2 = c.RED, bold = True)
        test_results.append(("ImageMaskDataset", False, str(e)))

    # ========================================================================
    # Test 2: EnhancedImageMaskDataset (RGB mode)
    # ========================================================================
    try:
        t("Test 2: EnhancedImageMaskDataset (RGB mode)")
        dataset2 = EnhancedImageMaskDataset(
                entries[:5],
                config.paths.train_images,
                mode = 'rgb',
                transform = train_tf
        )
        img_t, mask_t = dataset2[0]

        p("Image shape", img_t.shape)
        p("Mask shape", mask_t.shape)

        assert img_t.ndim == 3, f"Image should be 3D, got {img_t.ndim}D"
        assert mask_t.ndim == 2, f"Mask should be 2D [H, W], got {mask_t.shape}D"

        p("Test 2", "PASS", color1 = c.BLACK, color2 = c.BLUE, bold = True)
        test_results.append(("EnhancedImageMaskDataset (rgb)", True, None))

    except Exception as e:
        p("Test 2", f"FAIL: {e}", color1 = c.RED, color2 = c.RED, bold = True)
        test_results.append(("EnhancedImageMaskDataset (rgb)", False, str(e)))

    # ========================================================================
    # Test 3: EnhancedImageMaskDataset (filtered mode)
    # ========================================================================
    try:
        t("Test 3: EnhancedImageMaskDataset (filtered mode)")
        dataset3 = EnhancedImageMaskDataset(
                entries[:5],
                config.paths.train_images,
                mode = 'filtered',
                transform = train_tf
        )
        img_t, mask_t = dataset3[0]

        p("Image shape", img_t.shape)
        p("Mask shape", mask_t.shape)

        assert img_t.ndim == 3, f"Image should be 3D, got {img_t.ndim}D"
        assert mask_t.ndim == 2, f"Mask should be 2D [H, W], got {mask_t.shape}D"

        p("Test 3", "PASS", color1 = c.BLACK, color2 = c.BLUE, bold = True)
        test_results.append(("EnhancedImageMaskDataset (filtered)", True, None))

    except Exception as e:
        p("Test 3", f"FAIL: {e}", color1 = c.RED, color2 = c.RED, bold = True)
        test_results.append(("EnhancedImageMaskDataset (filtered)", False, str(e)))

    # ========================================================================
    # Test 4: DataLoader Batching
    # ========================================================================
    try:
        t("Test 4: DataLoader Batching")
        dataset = EnhancedImageMaskDataset(
                entries[:10],
                config.paths.train_images,
                mode = 'rgb',
                transform = train_tf
        )
        loader = DataLoader(dataset, batch_size = 4, shuffle = False)
        images, masks = next(iter(loader))

        p("Batch image shape", images.shape)
        p("Batch mask shape", masks.shape)

        assert images.ndim == 4, f"Batch images should be 4D, got {images.ndim}D"
        assert masks.ndim == 3, f"Batch masks should be 3D, got {masks.ndim}D"
        # assert masks.shape[1] == 1, f"Masks should have 1 channel, got {masks.shape[1]}"

        p("Test 4", "PASS", color1 = c.BLACK, color2 = c.BLUE, bold = True)
        test_results.append(("DataLoader Batching", True, None))

    except Exception as e:
        p("Test 4", f"FAIL: {e}", color1 = c.RED, color2 = c.RED, bold = True)
        test_results.append(("DataLoader Batching", False, str(e)))

    # ========================================================================
    # Test 5: Model Forward Pass
    # ========================================================================
    try:
        t("Test 5: Model Forward Pass")
        model = build_model('simple_cnn', in_channels = 3, out_channels = 3)
        model.eval()

        with torch.no_grad():
            preds = model(images)

        p("Input shape", images.shape)
        p("Output shape", preds.shape)
        p("Target shape", masks.shape)

        assert preds.shape == (masks.shape[0], 3, masks.shape[1], masks.shape[
            2]), f"Prediction {preds.shape} should be [B, 3, H, W] for target {masks.shape} [B, H, W]"

        p("Test 5", "PASS", color1 = c.BLACK, color2 = c.BLUE, bold = True)
        test_results.append(("Model Forward Pass", True, None))

    except Exception as e:
        p("Test 5", f"FAIL: {e}", color1 = c.RED, color2 = c.RED, bold = True)
        test_results.append(("Model Forward Pass", False, str(e)))

    # ========================================================================
    # Test 6: Loss Computation
    # ========================================================================
    try:
        t("Test 6: Loss Computation")
        criterion = CrossEntropyLoss()
        loss = criterion(preds, masks)

        p("Loss value", loss.item())

        p("Test 6", "PASS", color1 = c.BLACK, color2 = c.BLUE, bold = True)
        test_results.append(("Loss Computation", True, None))

    except Exception as e:
        p("Test 6", f"FAIL: {e}", color1 = c.RED, color2 = c.RED, bold = True)
        test_results.append(("Loss Computation", False, str(e)))

    # ========================================================================
    # Summary
    # ========================================================================
    t("Test Summary")

    passed = sum(1 for _, success, _ in test_results if success)
    failed = len(test_results) - passed

    for test_name, success, error in test_results:
        if success:
            p(test_name, "PASS", color1 = c.BLACK, color2 = c.BLUE)
        else:
            p(test_name, f"FAIL: {error}", color1 = c.RED, color2 = c.RED)

    p("")
    p("Total Tests", len(test_results))
    p("Passed", passed, color1 = c.GREEN)
    p("Failed", failed, color1 = c.RED if failed > 0 else c.GREEN)

    if failed == 0:
        p("")
        p("ALL TESTS PASSED!", "Dataset shapes are correct!", color1 = c.BLACK, color2 = c.BLUE, bold = True)
        return True
    else:
        p("")
        p("SOME TESTS FAILED!", "Check errors above", color1 = c.RED, color2 = c.RED, bold = True)
        return False


all_passed = verify_dataset_shapes()


# %% [markdown]
# #### Test CLAHE Error Fix

# %%
from src.exploration.enhancement import clahe_enhance, to_gray


# %%
t("Testing CLAHE with different input types")

# Test 1: Grayscale
try:
    gray = to_gray(img)
    clahe_result = clahe_enhance(gray)
    p("✓ CLAHE on grayscale", "PASS", color1 = c.GREEN)
except Exception as e:
    p("✗ CLAHE on grayscale", str(e), color1 = c.RED)

# Test 2: RGB (works via to_gray)
try:
    gray = to_gray(img)  # Convert first
    clahe_result = clahe_enhance(gray)
    p("✓ CLAHE on RGB (converted)", "PASS", color1 = c.GREEN)
except Exception as e:
    p("✗ CLAHE on RGB", str(e), color1 = c.RED)

# Test 3: EnhancedImageMaskDataset with CLAHE filter
try:
    dataset = EnhancedImageMaskDataset(
            entries[:5],
            config.paths.train_images,
            mode = 'filtered',
            filter_names = ['laplacian', 'sobel', 'clahe'],
            transform = train_tf
    )
    img_t, mask_t = dataset[0]
    p("✓ EnhancedDataset with CLAHE", "PASS", color1 = c.GREEN)
    p("  Shape", img_t.shape)
except Exception as e:
    p("✗ EnhancedDataset with CLAHE", str(e), color1 = c.RED)

# %%


# List all models
list_available_models()

# Build YOLOv8 small
model = build_model('yolov8s', in_channels = 3, out_channels = 3)

# Count parameters
params = sum(p.numel() for p in model.parameters())
p(f"Parameters: {params / 1e6:.2f}M")

# Test forward pass
x = torch.randn(2, 3, 256, 256)
with torch.no_grad():
    y = model(x)

p(f"Input shape: {x.shape}")
p(f"Output shape: {y.shape}")

# Verify output
assert y.shape == (2, 3, 256, 256), "Output shape mismatch!"
p("✓ YOLOv8 model test passed!")

