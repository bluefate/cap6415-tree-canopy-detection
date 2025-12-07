# %% [markdown]
# # Notebook: 04 Evaluation
# ### Purpose: load predictions, compute metrics, visualize overlays, and summarize performance.
#

# %%

from IPython import get_ipython


if not "google.colab" in str(get_ipython()):
    from pathlib import Path


    root = Path("C:/github/Tree-Canopy-Detection")

else:
    from pathlib import Path


    root = Path("/content/drive/MyDrive/TreeCanopyProject")

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

# %%
import os
import sys


sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../src"))

import random

import cv2
import numpy as np
import torch

from src.data.annotations import load_json_annotations
from src.data.augmentations import get_val_augmentations
from src.data.loaders import ImageMaskDataset
from src.exploration.visualize import show_side_by_side
from src.models.zoo import MODEL_BUILDERS
from src.training.metrics import compute_metrics
from src.utils.config import Config
from src.utils.helpers import c, init_notebook, p, t
from src.exploration.evaluation import diagnose_model_directory, load_best_model


config = Config.load(root = root)

init_notebook(config.train.seed)

train_dir = config.paths.train_images
mask_dir = config.paths.train_masks
annotations_path = config.paths.annotations
entries = load_json_annotations(annotations_path)

# if not "google.colab" in str(get_ipython()):
#     config.train.batch_size = 2
#     config.train.num_workers = 1
#     config.train.image_size = 32
#     config.train.epochs = 2

# %%
p("Models", MODEL_BUILDERS)
#config.show()
p("Batch", config.train.batch_size)
p("Epochs", config.train.epochs)
p("Learning Rate", config.train.learning_rate, precision = 9)
p("Image Size", config.train.image_size)


# %% [markdown]
# #### Validation setup

# %%
val_tf = get_val_augmentations(config.train.image_size)
dataset = ImageMaskDataset(entries, train_dir, transform = val_tf)

# %% [markdown]
# #### Loading Model

# %%
diagnose_model_directory(config)

# %%
# Using output from notebook 3
model = load_best_model("simple_cnn", config, notebook = "03", mode = "rgb")

# %% [markdown]
# #### Batch evaluation

# %%
all_metrics = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for idx in range(len(dataset)):
    img_t, mask_t = dataset[idx]

    img_np = img_t.permute(1, 2, 0).numpy()
    true_np = mask_t.squeeze().numpy()

    # Move tensors to the same device as model
    img_t = img_t.to(device)
    mask_t = mask_t.to(device)

    with torch.no_grad():
        pred = model(img_t.unsqueeze(0))
    metrics = compute_metrics(pred, mask_t.unsqueeze(0))
    all_metrics.append(metrics)

# %% [markdown]
# #### Summarize metrics

# %%

# Extract all available metrics
iou_vals = [m["iou"] for m in all_metrics]
dice_vals = [m["dice"] for m in all_metrics]
acc_vals = [m["acc"] for m in all_metrics]
precision_vals = [m.get("precision", 0) for m in all_metrics]
recall_vals = [m.get("recall", 0) for m in all_metrics]

# Calculate F1 scores
f1_vals = []
for p_val, r_val in zip(precision_vals, recall_vals):
    if p_val + r_val > 0:
        f1 = 2 * (p_val * r_val) / (p_val + r_val)
    else:
        f1 = 0
    f1_vals.append(f1)

# Display comprehensive metrics
p("=== COMPREHENSIVE METRICS ===", color1 = c.CYAN, bold = True)
p("")
p("IoU (Intersection over Union):")
p("  Mean IoU", f"{np.mean(iou_vals):.4f}")
p("  Std IoU", f"{np.std(iou_vals):.4f}")
p("")

p("Dice Coefficient:")
p("  Mean Dice", f"{np.mean(dice_vals):.4f}")
p("  Std Dice", f"{np.std(dice_vals):.4f}")
p("")

p("Accuracy:")
p("  Mean Accuracy", f"{np.mean(acc_vals):.4f}")
p("  Std Accuracy", f"{np.std(acc_vals):.4f}")
p("")

p("Precision:")
p("  Mean Precision", f"{np.mean(precision_vals):.4f}")
p("  Std Precision", f"{np.std(precision_vals):.4f}")
p("")

p("Recall:")
p("  Mean Recall", f"{np.mean(recall_vals):.4f}")
p("  Std Recall", f"{np.std(recall_vals):.4f}")
p("")

p("F1 Score:")
p("  Mean F1", f"{np.mean(f1_vals):.4f}")
p("  Std F1", f"{np.std(f1_vals):.4f}")
p("")

# Summary table
p("=== SUMMARY TABLE ===", color1 = c.GREEN, bold = True)
p(f"{'Metric':<12} {'Mean':<8} {'Std':<8}")
p("-" * 28)
p(f"{'IoU':<12} {np.mean(iou_vals):<8.4f} {np.std(iou_vals):<8.4f}")
p(f"{'Dice':<12} {np.mean(dice_vals):<8.4f} {np.std(dice_vals):<8.4f}")
p(f"{'Accuracy':<12} {np.mean(acc_vals):<8.4f} {np.std(acc_vals):<8.4f}")
p(f"{'Precision':<12} {np.mean(precision_vals):<8.4f} {np.std(precision_vals):<8.4f}")
p(f"{'Recall':<12} {np.mean(recall_vals):<8.4f} {np.std(recall_vals):<8.4f}")
p(f"{'F1 Score':<12} {np.mean(f1_vals):<8.4f} {np.std(f1_vals):<8.4f}")

# %% [markdown]
# #### Visualizations Samples

# %%
for _ in range(5):
    idx = random.randint(0, len(dataset) - 1)
    t(f"Image {idx}")

    img_t, mask_t = dataset[idx]

    # Get original image to preserve dimensions
    entry = entries[idx]
    img_path = train_dir / entry.image_path.name
    original_img = cv2.imread(str(img_path))
    if original_img is not None:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        original_shape = original_img.shape
        img_display = original_img
    else:
        # Fallback to processed image
        img_display = img_t.permute(1, 2, 0).numpy()
        if img_display.max() <= 1.0:
            img_display = (img_display * 255).astype(np.uint8)
        original_shape = img_display.shape

    # Get ground truth mask
    mask = mask_t.squeeze().numpy()

    # CRITICAL: Move input to same device as model
    device = next(model.parameters()).device
    img_t = img_t.to(device)

    with torch.no_grad():
        pred = model(img_t.unsqueeze(0)).cpu()

    # Handle multi-class output
    if pred.shape[1] == 3:  # Multi-class
        pred_classes = torch.argmax(pred, dim = 1).squeeze().numpy()
        pred_bin = (pred_classes > 0).astype(np.uint8)  # Any tree = 1
    else:  # Binary
        pred = torch.sigmoid(pred).squeeze().numpy()
        pred_bin = (pred > 0.5).astype(np.uint8)

    # CRITICAL: Resize predictions to match original image dimensions
    orig_h, orig_w = original_shape[:2]
    if pred_classes.shape != (orig_h, orig_w):
        pred_classes_resized = cv2.resize(
                pred_classes, (orig_w, orig_h),
                interpolation = cv2.INTER_NEAREST
        )
        pred_bin_resized = cv2.resize(
                pred_bin, (orig_w, orig_h),
                interpolation = cv2.INTER_NEAREST
        )
    else:
        pred_classes_resized = pred_classes
        pred_bin_resized = pred_bin

    # Resize ground truth mask to match if needed
    if mask.shape != (orig_h, orig_w):
        mask_resized = cv2.resize(
                mask.astype(np.uint8), (orig_w, orig_h),
                interpolation = cv2.INTER_NEAREST
        )
    else:
        mask_resized = mask

    # Create colored masks for better visualization
    mask_rgb = np.zeros((orig_h, orig_w, 3), dtype = np.uint8)
    mask_rgb[mask_resized == 1] = (0, 255, 0)  # individual trees - green
    mask_rgb[mask_resized == 2] = (255, 255, 0)  # groups - yellow

    pred_rgb = np.zeros((orig_h, orig_w, 3), dtype = np.uint8)
    pred_rgb[pred_classes_resized == 1] = (0, 255, 0)  # individual trees - green
    pred_rgb[pred_classes_resized == 2] = (255, 255, 0)  # groups - yellow

    # Create overlay
    overlay = cv2.addWeighted(img_display, 0.6, pred_rgb, 0.4, 0)

    titles = [f"Image {idx}", "Ground Truth", "Prediction", "Overlay"]

    show_side_by_side(
            img_display, mask_rgb, pred_rgb, overlay,
            titles = titles,
            cmaps = [None, None, None, None]
    )

# %%
