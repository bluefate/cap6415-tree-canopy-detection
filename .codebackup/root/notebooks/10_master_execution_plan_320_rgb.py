# %% [markdown]
# <a href="https://colab.research.google.com/github/bluefate/CAP6415_F25_project-Tree-Canopy-Detection/blob/main/notebooks/10_master_execution_plan_320_rgb.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%
import os
import sys
from IPython.display import HTML



if 'google.colab' in str(get_ipython()):
    HTML("""
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        pre, code, .output pre {
            font-family: 'JetBrains Mono', monospace !important;
        }
    </style>
    """)

    os.chdir('/content')

    from google.colab import drive
    drive.mount('/content/drive')

    # Load environment variables
    env_path = '/content/drive/MyDrive/TreeCanopyProject/.env'
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

    # Clone repository
    repo_path = '/content/CAP6415_F25_project-Tree-Canopy-Detection'
    github_token = os.getenv('TOKEN')

    if not os.path.exists(repo_path):
        if github_token:
            # !git config --global user.email "jherna65@fau.edu"
            # !git config --global user.name "bluefate"
            clone_url = f"https://bluefate:{github_token}@github.com/bluefate/CAP6415_F25_project-Tree-Canopy-Detection.git"
            # !git clone $clone_url
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
            # !git pull
        sys.path.insert(0, repo_path)
        sys.path.insert(0, os.path.join(repo_path, 'src'))
        print("Setup complete")

    # Set paths
    if os.path.exists(repo_path):
        os.chdir(repo_path)
        sys.path.insert(0, repo_path)
        sys.path.insert(0, os.path.join(repo_path, 'src'))
        print("Setup complete")

    print("Requirements")
    # Install packages
    # !pip install -r requirements.txt

# %%
# # !git fetch origin
# # !git reset --hard origin/main

# %% [markdown]
# # Notebook: 10 Master Execution Plan
# ### Purpose: Complete pipeline from data preparation to final submission

# %%

import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  #enable for debugging ONLY

sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../src"))
from src.data.image_loader import validate_image_directory
from src.exploration.class_explorer import color_mask, load_image, mask_all
from src.exploration.visualize import show_side_by_side
from src.models.zoo import build_model, MODEL_EXPERIMENTS
from src.training.running import get_available_filters, get_version_config, validate_filter_set
from src.utils.helpers import format_time
from src.utils.image_converter import ImageConverter
from torch.utils.data import DataLoader
from src.data.annotations import load_json_annotations
from src.data.augmentations import get_train_augmentations, get_val_augmentations
from src.data.loaders import ImageMaskDataset
from src.training.engine import run_training
from src.utils.config import Config
from src.utils.helpers import c, init_notebook, p, t
from src.data.enhance_masks import EnhancedImageMaskDataset
from src.utils.versioning import VersionManager
import matplotlib.pyplot as plt
from IPython.display import clear_output, display


config = Config.load()
#config = Config.load(Path("..").resolve() / "config_PROD.yaml")
init_notebook(config.train.seed)
config.show()



# %% [markdown]
# #### Step 1: Data Preparation
#
# - TIFF → PNG conversion (notebook 00_preprocess_images)
# - Annotation loading
# - Mask generation
# - Data augmentation
#
#

# %%
t("Loading Annotations")
entries = load_json_annotations(config.paths.annotations)
p("Total images", len(entries))

if config.paths.eval_images and config.paths.eval_images.exists():
    # Check if conversion is needed
    tiff_files = list(config.paths.eval_images.glob("*.tif")) + list(config.paths.eval_images.glob("*.tiff"))
    png_files = list(config.paths.eval_images.glob("*.png"))

    p("TIFF files found", len(tiff_files), color1 = c.BLACK)
    p("PNG files found", len(png_files), color1 = c.BLACK)

    if len(tiff_files) > 0:
        p("Converting TIFF to PNG...", color1 = c.ORANGE)
        eval_converter = ImageConverter(
                source_dir = config.paths.eval_images,
                target_dir = config.paths.eval_images
        )
        eval_stats = eval_converter.convert_batch(overwrite = False)
        p("Converted", eval_stats.get("converted", 0), color1 = c.ORANGE)
        p("Skipped (already exist)", eval_stats.get("skipped", 0), color1 = c.ORANGE)
        p("Failed", eval_stats.get("failed", 0), color1 = c.ORANGE)
    else:
        p("No TIFF files to convert", color1 = c.GREEN)

    # Validate
    eval_validation = validate_image_directory(config.paths.eval_images)
    p("Valid images after conversion", eval_validation.get("valid", 0), color1 = c.BLACK)
else:
    p("Warning", "Eval images directory not found", color1 = c.ORANGE)

# Analyze class distribution
individual_count = sum(1 for e in entries if any(item.cls == "individual_tree" for item in e.items))
group_count = sum(1 for e in entries if any(item.cls == "group_of_trees" for item in e.items))

p("Images with individual trees", individual_count)
p("Images with tree groups", group_count)



# %%
# Add this after your imports and before training
import torch
from torch.cuda.amp import GradScaler, autocast

# Optimize CUDA
torch.backends.cudnn.benchmark = True
torch.cuda.set_per_process_memory_fraction(0.95)

# Mixed precision scaler
scaler = GradScaler()

p("CUDA optimizations enabled")

# %% [markdown]
# #### Step 2: Filter Experimentation
#
# **Action:** Run and retrieve notebook 08 to identify top 3 filters
#
# **Expected Output:**
# - Filter ranking CSV
# - Top 3 filter names
# - Visual comparisons
#
#

# %% [markdown]
# ##### Validate Available Filters

# %%


# Get available filters
t("Validating Available Filters")
AVAILABLE_FILTERS = get_available_filters()
p("Available filters count", len(AVAILABLE_FILTERS))
p("Sample filters", AVAILABLE_FILTERS, show = 15, color1 = c.ORANGE)

# %% [markdown]
# #### Step 3: Enhanced Dataset Creation
#
# - Create training dataset with filter-enhanced inputs
# - Apply filters as additional channels.
#
#

# %%
t("Dataset modes")

val_transform = get_val_augmentations(config.train.image_size)
sample_entries = entries[:5]

# for mode in ['rgb', 'filtered', 'concat']:
for mode in ['rgb', 'filtered']:
    try:
        dataset = EnhancedImageMaskDataset(
                sample_entries,
                config.paths.train_images,
                mode = mode,
                transform = val_transform
        )

        img_t, mask_t = dataset[0]
        p(f"Mode {mode}", f"\n\tImage shape: {img_t.shape}, \n\tMask shape: {mask_t.shape}", color1 = c.ORANGE)
    except Exception as e:
        p(f"Mode: {mode}", f"FAILED: {e}", color1 = c.RED, color2 = c.RED)


# %% [markdown]
# #### Step 4: Model Training Comparison
#
# **Experiment Design:**
# Comparing model performance across input types:
# 1. Baseline: RGB only
# 2. Filtered: Top 3 filters as channels
# 3. Concat: RGB + Filters (6 channels)
#
# **Models to test:**
# - SimpleCNN (fast baseline)
# - UNet (standard architecture)
# - SMP UNet + ResNet34 (transfer learning)
#
#

# %% [markdown]
# ##### Experiment setup

# %%
t("Filter Sets")

# Define filter combinations to test
filter_sets = {
    'classic':        ['laplacian', 'sobel', 'clahe'],
    'gaussian':       ['gaussian_3x3', 'gaussian_5x5', 'gaussian_7x7'],
    'kernel_sharpen': ['sharpen_basic', 'high_pass_3x3', 'edge_enhance'],
    'kernel_edge':    ['sobel_x', 'sobel_y', 'laplacian_3x3'],
    'combined':       ['laplacian', 'gaussian_5x5', 'clahe'],
}

# Validate all filter sets before proceeding

all_valid = True

for set_name, filters in filter_sets.items():
    is_valid, invalid, suggestions = validate_filter_set(filters, AVAILABLE_FILTERS)

    if is_valid:
        p(f"Filter set '{set_name}'", "VALID", color1 = c.GREEN)
    else:
        p(f"Filter set '{set_name}'", f"INVALID: {invalid}", color1 = c.RED, color2 = c.RED)
        if suggestions:
            p("  Suggestions", suggestions, color1 = c.ORANGE)
        all_valid = False

if not all_valid:
    p("")
    p("WARNING", "Some filter sets have invalid filter names!", color1 = c.RED, color2 = c.RED)
    p("", "Check filter names against AVAILABLE_FILTERS", color1 = c.ORANGE)

p("")
p("filter_sets", filter_sets, color1 = c.ORANGE)

# %%

t("Available experiments")

all_experiments = MODEL_EXPERIMENTS(filter_sets)

# Concat mode: 6 channels (RGB + 3 filters)
# Note: This requires model architecture modification for 6-channel input

p("All experiments", all_experiments, show = 100, color1 = c.ORANGE)

# %%
t("Setup experiments to run ")

#experiments = [all_experiments[81]]  # 81: ('yolov8l', 'filtered', ['sharpen_basic', 'high_pass_3x3', 'edge_enhance'])
#experiments = [all_experiments[67]]  # 67: ('yolov8s', 'filtered', ['sharpen_basic', 'high_pass_3x3', 'edge_enhance'])
#experiments = all_experiments
# experiments = [all_experiments[0]]
#experiments = [exp for exp in all_experiments if exp[0] == 'simple_cnn']
experiments = [exp for exp in all_experiments if exp[1] == 'rgb']

p("Experiments to Run", experiments, show = 50, color1 = c.RED)


# %%
import datetime
import torch


# Assume 'p', 't', 'c', 'entries', and 'config' are defined in the context.

def format_time( seconds ):
    """Converts a total number of seconds into a human-readable D days, HH:MM:SS format."""
    td = datetime.timedelta(seconds = int(seconds))
    time_str = str(td)

    # Handle the 'days' case (e.g., "1 day, 0:03:20" -> "1d 0h 3m 20s")
    if 'day' in time_str:
        parts = time_str.split(', ')
        days = parts[0].replace(' days', 'd').replace(' day', 'd')
        hms = parts[1].split(':')
        return f"{days} {hms[0].zfill(1)}h {hms[1].zfill(2)}m {hms[2].zfill(2)}s"

    # If less than a day, output Hh Mm Ss (e.g., "3:25:45" -> "3h 25m 45s")
    hms = time_str.split(':')
    # Use lstrip('0') to show '3h' instead of '03h' unless it's '0h'
    return f"{hms[0].lstrip('0')}h {hms[1]}m {hms[2]}s"


def estimate_runtime( experiments ):
    """
    Estimate training runtime with GPU/CPU awareness, model complexity,
    and experiment mode (rgb, filtered, concat) awareness.
    """

    # --- Setup and Initialization ---
    train_size = int(0.8 * len(entries))
    batch_size = config.train.batch_size
    batches_per_epoch = max(1, train_size // batch_size)
    epochs = config.train.epochs
    total_seconds = 0.0

    # Device detection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'

    if device.type == 'cuda':
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory < 4:  # Low-end GPU
            base_seconds = 2.0
        elif gpu_memory < 8:  # Mid-range GPU
            base_seconds = 1.0
        else:  # High-end GPU
            base_seconds = 0.5
    else:
        base_seconds = 5.0  # CPU is significantly slower

    # Mode timing multipliers (updated for clarity and to incorporate overhead)
    mode_multiplier = {
        "rgb":      1.0,  # Standard 3-channel input
        "filtered": 1.5,  # Filter computation overhead
        "concat":   2.5  # 6-channel input + filter overhead (higher than 2.0 to account for extra memory/ops)
    }

    model_complexity = {
        "simple_cnn":        1.0,
        "unet":              2.5,
        "smp_deeplabv3plus": 4.5,
        "segformer":         3.5,
        "yolov8n":           1.8,
        "yolov8l":           6.0,
    }

    # --- Header and Pre-Run Info ---
    t("Runtime Estimate")
    p("Device", device_name, color1 = c.GREEN)
    p("Total Experiments", len(experiments), color1 = c.BLACK, color2 = c.ORANGE)
    p("Training samples", train_size, color1 = c.BLACK)
    p("Batches per epoch", batches_per_epoch, color1 = c.BLACK)
    p("Epochs per experiment", epochs, color1 = c.BLACK)
    p("Batch size", batch_size, color1 = c.BLACK)
    p("-" * 70, color1 = c.ORANGE)
    p()

    p("Per-Experiment Estimates:", color1 = c.CYAN, bold = True)
    p("-" * 70, color1 = c.CYAN)

    # --- Calculation Loop ---
    for model_name, mode, filters in experiments:

        # Get multipliers, defaulting to 1.0 if model/mode not found
        mode_mult = mode_multiplier.get(mode, 1.0)
        complexity_mult = model_complexity.get(model_name, 1.0)

        # Total multiplier
        total_mult = mode_mult * complexity_mult

        # Time calculation: Base * Device/Complexity Multipliers * (Batches * Epochs)
        sec_per_batch = base_seconds * total_mult

        exp_seconds = epochs * batches_per_epoch * sec_per_batch
        total_seconds += exp_seconds

        # Format filter string
        filter_str = f"[{', '.join(filters)}...]" if filters and len(filters) > 0 else "none"

        # Display
        exp_label = f"{model_name:20s} | {mode:8s} | {filter_str:20s}"

        p(exp_label, format_time(exp_seconds), color1 = c.BLUE, color2 = c.BLACK)

    p()
    t("Totals")

    total_hours = total_seconds / 3600

    p("Total experiments", len(experiments))
    p("Total batches", batches_per_epoch * epochs * len(experiments))

    p("Estimated total time", format_time(total_seconds), color1 = c.GREEN, bold = True)

    # Time breakdown and Warnings
    if total_hours >= 24:
        p("Estimated completion", f"~{total_hours / 24:.1f} days", color1 = c.ORANGE)
        p("⚠ WARNING", "Training will take over 24 hours!", color1 = c.ORANGE, bold = True)
        p("Consider", "Reducing epochs or selecting fewer models", color1 = c.ORANGE)
    elif total_hours >= 8:
        p("Estimated completion", f"~{total_hours:.1f} hours", color1 = c.ORANGE)
        p("⚠ NOTE", "Long training session - consider running overnight", color1 = c.ORANGE)
    else:
        p("Estimated completion", f"~{total_seconds / 60:.0f} minutes", color1 = c.GREEN)

    if device.type == 'cpu':
        p("⚠ CPU DETECTED", "Training on CPU is 10-20x slower than GPU", color1 = c.RED, bold = True)
    p()


estimate_runtime(experiments)


# %%
def analyze_class_distribution(train_loader, val_loader=None):
    """
    Analyze class distribution in training (and optionally validation) data.
    Call this before training to understand class imbalance and get weight recommendations.
    """

    p()
    t("CLASS DISTRIBUTION ANALYSIS")


    def analyze_loader(loader, name):
        all_masks = []
        total_images = 0

        for _, mask in loader:
            all_masks.append(mask.flatten())
            total_images += mask.shape[0]

        all_masks = torch.cat(all_masks)
        counts = torch.bincount(all_masks, minlength=3)
        total_pixels = counts.sum().item()

        print(f"\n{name}:")
        print(f"  Total images: {total_images}")
        print(f"  Total pixels: {total_pixels:,}")
        print()

        class_names = ["background", "individual_tree", "group_of_trees"]
        percentages = []

        for i, (class_name, count) in enumerate(zip(class_names, counts)):
            pct = count.item() / total_pixels * 100
            percentages.append(pct)
            print(f"  Class {i} ({class_name:15}): {count.item():>10,} pixels ({pct:5.2f}%)")

        return counts, percentages

    # Analyze training data
    train_counts, train_pcts = analyze_loader(train_loader, "TRAINING SET")

    # Analyze validation data if provided
    if val_loader is not None:
        val_counts, val_pcts = analyze_loader(val_loader, "VALIDATION SET")

    # Calculate recommended weights (inverse frequency)
    p()
    t("RECOMMENDED WEIGHTS")

    total = train_counts.sum().float()
    frequencies = train_counts.float() / total

    # Method 1: Inverse frequency
    inv_freq_weights = 1.0 / (frequencies + 1e-8)
    inv_freq_weights = inv_freq_weights / inv_freq_weights.sum() * 3  # Normalize to sum=3

    # Method 2: Balanced weights (sklearn-style)
    n_classes = 3
    n_samples = total.item()
    balanced_weights = n_samples / (n_classes * train_counts.float() + 1e-8)
    balanced_weights = balanced_weights / balanced_weights.min()  # Normalize so min=1

    # Method 3: Simple practical weights (background down, trees up)
    bg_ratio = train_pcts[0] / 100
    simple_weights = torch.tensor([
        0.3,  # Background (reduce)
        1.0 / (train_pcts[1] / 100 + 0.1),  # Individual tree
        1.0 / (train_pcts[2] / 100 + 0.1),  # Group of trees
    ])
    simple_weights = simple_weights / simple_weights.sum() * 3

    # using print because p nt outputing weights correctly.
    print("\nMethod 1 - Inverse Frequency:")
    print(f"  weights = torch.tensor([{inv_freq_weights[0]:.2f}, {inv_freq_weights[1]:.2f}, {inv_freq_weights[2]:.2f}])")

    print("\nMethod 2 - Balanced (sklearn-style):")
    print(f"  weights = torch.tensor([{balanced_weights[0]:.2f}, {balanced_weights[1]:.2f}, {balanced_weights[2]:.2f}])")

    print("\nMethod 3 - Simple Practical:")
    print(f"  weights = torch.tensor([{simple_weights[0]:.2f}, {simple_weights[1]:.2f}, {simple_weights[2]:.2f}])")


    return {
        "train_counts": train_counts,
        "train_percentages": train_pcts,
        "inv_freq_weights": inv_freq_weights,
        "balanced_weights": balanced_weights,
        "simple_weights": simple_weights,
    }



# %%
def create_experiment_tracker():
    """Create a tracker to store and plot experiment results."""
    return {
        "models": [],
        "val_loss": [],
        "iou": [],
        "iou_individual": [],
        "iou_group": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }

def update_tracker(tracker, model_name, checkpoint_path):
    """Update tracker with results from a completed experiment."""
    if not checkpoint_path.exists():
        return False

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        tracker["models"].append(model_name)
        tracker["val_loss"].append(ckpt.get("val_loss", ckpt.get("best_val_loss", 0)))
        tracker["iou"].append(ckpt.get("val_iou", ckpt.get("iou", 0)))
        tracker["iou_individual"].append(ckpt.get("val_iou_individual", 0))
        tracker["iou_group"].append(ckpt.get("val_iou_group", 0))
        tracker["precision"].append(ckpt.get("val_precision", ckpt.get("precision", 0)))
        tracker["recall"].append(ckpt.get("val_recall", ckpt.get("recall", 0)))
        tracker["f1"].append(ckpt.get("val_f1_score", ckpt.get("f1_score", 0)))
        return True
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False

def plot_experiment_results(tracker, save_path=None):
    """Plot comparison of all completed experiments."""
    if len(tracker["models"]) == 0:
        print("No experiments completed yet.")
        return

    # One row, six columns
    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    fig.suptitle("Experiment Comparison", fontsize=14, fontweight="bold")

    models = tracker["models"]
    x = range(len(models))
    colors = plt.cm.tab10(range(len(models)))

    def setup_xticks(ax):
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=0, ha="center", fontsize=8)

    # Plot 1: Val Loss (lower is better)
    ax = axes[0]
    ax.bar(x, tracker["val_loss"], color=colors)
    ax.set_ylabel("Val Loss")
    ax.set_title("Validation Loss (lower=better)")
    setup_xticks(ax)
    ax.axhline(y=min(tracker["val_loss"]), color="green", linestyle="--", alpha=0.5)

    # Plot 2: IoU (higher is better)
    ax = axes[1]
    ax.bar(x, tracker["iou"], color=colors)
    ax.set_ylabel("IoU")
    ax.set_title("Mean IoU (higher=better)")
    setup_xticks(ax)
    ax.axhline(y=max(tracker["iou"]), color="green", linestyle="--", alpha=0.5)

    # Plot 3: IoU by class
    ax = axes[2]
    width = 0.35
    x_arr = np.arange(len(models))
    ax.bar(x_arr - width / 2, tracker["iou_individual"], width, label="Individual", color="forestgreen")
    ax.bar(x_arr + width / 2, tracker["iou_group"], width, label="Group", color="gold")
    ax.set_ylabel("IoU")
    ax.set_title("IoU by Class")
    ax.set_xticks(x_arr)
    ax.set_xticklabels(models, rotation=45, ha="center", fontsize=8)
    ax.legend()

    # Plot 4: Precision
    ax = axes[3]
    ax.bar(x, tracker["precision"], color=colors)
    ax.set_ylabel("Precision")
    ax.set_title("Precision (higher=better)")
    setup_xticks(ax)

    # Plot 5: Recall
    ax = axes[4]
    ax.bar(x, tracker["recall"], color=colors)
    ax.set_ylabel("Recall")
    ax.set_title("Recall (higher=better)")
    setup_xticks(ax)

    # Plot 6: F1 Score
    ax = axes[5]
    ax.bar(x, tracker["f1"], color=colors)
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score (higher=better)")
    setup_xticks(ax)
    ax.axhline(y=max(tracker["f1"]), color="green", linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()

    # Print summary table
    p()
    t(f"{'Model':<20} {'Loss':>8} {'IoU':>8} {'Ind':>8} {'Grp':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")

    best_iou_idx = np.argmax(tracker["iou"])
    for i, model in enumerate(models):
        marker = " 🏆" if i == best_iou_idx else ""
        print(
            f"{model:<20} {tracker['val_loss'][i]:>8.4f} {tracker['iou'][i]:>8.4f} "
            f"{tracker['iou_individual'][i]:>8.4f} {tracker['iou_group'][i]:>8.4f} "
            f"{tracker['precision'][i]:>8.4f} {tracker['recall'][i]:>8.4f} {tracker['f1'][i]:>8.4f}{marker}"
        )

def plot_training_history(trainer, title_prefix = ""):
    history = trainer.history

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    val_iou = history.get("val_iou", [])
    val_precision = history.get("val_precision", [])
    val_recall = history.get("val_recall", [])
    val_f1 = history.get("val_f1", [])
    lr = history.get("lr", [])

    n_epochs = len(train_loss)
    if n_epochs == 0:
        print("No history to plot")
        return

    epochs = range(1, n_epochs + 1)

    fig, axes = plt.subplots(1, 4, figsize=(22, 4))
    fig.suptitle(f"{title_prefix} epoch metrics", fontsize=14, fontweight="bold")

    # 1. Loss curves
    ax = axes[0]
    ax.plot(epochs, train_loss, label="Train loss")
    ax.plot(epochs, val_loss, label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss per epoch")
    ax.legend()

    # 2. IoU and F1
    ax = axes[1]
    if len(val_iou) == n_epochs:
        ax.plot(epochs, val_iou, label="Val IoU")
    if len(val_f1) == n_epochs:
        ax.plot(epochs, val_f1, label="Val F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("IoU and F1 per epoch")
    ax.legend()

    # 3. Precision and Recall
    ax = axes[2]
    if len(val_precision) == n_epochs:
        ax.plot(epochs, val_precision, label="Val precision")
    if len(val_recall) == n_epochs:
        ax.plot(epochs, val_recall, label="Val recall")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Precision and Recall per epoch")
    ax.legend()

    # 4. Learning rate
    ax = axes[3]
    if len(lr) == n_epochs:
        ax.plot(epochs, lr, label="LR")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("LR schedule")
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()



# %%
def get_input_channels( mode, filters ):
    """Determine input channels based on mode."""
    if mode == "concat" and filters:
        return 6  # RGB + 3 filters
    else:
        return 3  # RGB or filtered RGB


# %%
# Initialize best model tracker
best_model_tracker = {
    "best_val_loss":    float('inf'),
    "best_iou":         0.0,
    "best_experiment":  None,
    "best_model_path":  None,
    "best_version_dir": None,
}

experiment_tracker = create_experiment_tracker()

# %%
# plot_experiment_results(experiment_tracker)

# %%
t("Train Experiments")



# Running experiments
results = { }
analysis_printed = False
for i, (model_name, mode, filters) in enumerate(experiments, 1):
    p("\n\n")
    t(f"Experiment {i}/{len(experiments)}")
    p("Model", model_name, color1 = c.ORANGE)
    p("Mode", mode, color1 = c.ORANGE)
    p("Filters", filters, color1 = c.ORANGE)

    try:
        # Prepare data
        train_entries = entries[: int(0.8 * len(entries))]
        val_entries = entries[int(0.8 * len(entries)):]

        train_tf = get_train_augmentations(config.train.image_size)
        val_tf = get_val_augmentations(config.train.image_size)

        train_ds = EnhancedImageMaskDataset(
                train_entries,
                config.paths.train_images,
                mode = mode,
                filter_names = filters,
                transform = train_tf,
        )
        val_ds = EnhancedImageMaskDataset(
                val_entries,
                config.paths.train_images,
                mode = mode,
                filter_names = filters,
                transform = val_tf,
        )
        # Determine input channels
        sample_img, _ = train_ds[0]
        in_channels = sample_img.shape[0]
        # Skip 6-channel experiments not supported yet
        if in_channels == 6 and model_name in ["simple_cnn", "unet"]:
            # p("Warning", f"Skipping 6-channel experiment for {model_name}", color1 = c.ORANGE)
            # continue
            in_channels = get_input_channels(mode, filters)

        train_loader = DataLoader(
                train_ds,
                batch_size = config.train.batch_size,
                shuffle = True,
                num_workers = config.train.num_workers,
                pin_memory = True if torch.cuda.is_available() else False,
        )

        sample_img, sample_mask = next(iter(train_loader))
        print(f"Image range: [{sample_img.min():.3f}, {sample_img.max():.3f}]")
        print(f"Mask unique values: {torch.unique(sample_mask).tolist()}")
        print(f"Mask value counts: {torch.bincount(sample_mask.flatten(), minlength=3).tolist()}")


        val_loader = DataLoader(
                val_ds,
                batch_size = config.train.batch_size,
                shuffle = False,
                num_workers = config.train.num_workers,
                pin_memory = True if torch.cuda.is_available() else False,
        )

        if not analysis_printed:
            analysis_printed = True
            # using softer balance'
            # weights = torch.tensor([1.00, 2.50, 6.00]) in prepare_criterion
            analysis = analyze_class_distribution(train_loader, val_loader)


        key, version_root, exp_config, best_model_path, checkpoint_path_check, best_model_exists = get_version_config(
                config, filters, "10", model_name, mode, in_channels, best_model_tracker, i, experiments
        )
        if best_model_exists:
            continue

        trainer = run_training(
                config = exp_config,
                train_loader = train_loader,
                val_loader = val_loader,
                version_root = version_root,
                model_name = model_name,
                in_channels = in_channels,
        )

        if trainer is not None:
            results[key] = trainer

            # Check if this is the best model
            version_dir = trainer.version_dir
            checkpoint_path = version_dir / "checkpoint.pth"

            if checkpoint_path.exists():
                try:
                    ckpt = torch.load(checkpoint_path, map_location = "cpu")
                    val_loss = ckpt.get("best_val_loss", float('inf'))
                    p("[Info]", f"Val Loss: {val_loss:.6f}", color1 = c.GREEN)

                    if val_loss < best_model_tracker["best_val_loss"]:
                        best_model_tracker["best_val_loss"] = val_loss
                        best_model_tracker["best_experiment"] = key
                        best_model_tracker["best_model_path"] = version_dir / "best_model.pth"
                        best_model_tracker["best_version_dir"] = version_dir
                        p("\t\t🏆 NEW BEST MODEL", key, color1 = c.ORANGE, color2 = c.ORANGE, bold = True)

                    update_tracker(experiment_tracker, key, checkpoint_path)
                    # plot_experiment_results(experiment_tracker)
                    plot_training_history(trainer, title_prefix=key)


                except Exception as e:
                    p("Warning", f"Could not load checkpoint: {e}", color1 = c.ORANGE, color2 = c.ORANGE)

            p(f"COMPLETED {i}/{len(experiments)}", key, color1 = c.GREEN, color2 = c.GREEN)
        else:
            p(f"SKIPPED {i}/{len(experiments)}", key, color1 = c.ORANGE, color2 = c.ORANGE)

    except Exception as e:
        p(f"FAILED {i}/{len(experiments)}", key, color1 = c.RED, color2 = c.RED)
        p("[Error]", str(e), color1 = c.RED)
        results[key] = { "status": "ERROR", "error": str(e) }
        continue


# %%

# Summary
t("Experiment Results Summary")
successful = sum(1 for v in results.values() if not isinstance(v, dict) or v.get("status") != "ERROR")
failed = sum(1 for v in results.values() if isinstance(v, dict) and "status" in v)
skipped = sum(1 for v in results.values() if v is None)

p("Total experiments", len(experiments))
p("Successful", successful - skipped, color1 = c.BLUE)
p("Skipped", skipped, color1 = c.ORANGE)
p("Failed", failed, color1 = c.RED if failed > 0 else c.RED)


# %%

# %%

# %% [markdown]
# ##### Model Comparison & Best Model Selection

# %%
# List of metrics to extract
metrics_to_extract = [
    'best_val_loss',
    'epoch',
    'final_epoch',
    'train_loss',
    'val_loss',
    'val_accuracy',
    'val_iou',
    'iou',
    'accuracy',
    'f1_score',
    'precision',
    'recall',
]

# %%

# %%
t("Collecting Experiment Results")

experiment_results = []

# Iterate through all experiments
for i, (model_name, mode, filters) in enumerate(experiments, 1):
    p()
    p(f"Experiment {i}: {model_name}, {mode}, {filters}", color1 = c.MAGENTA, bold = True)

    key, version_root, exp_config, best_model_path, checkpoint_path_check, best_model_exists = get_version_config(
            config, filters, "10", model_name, mode, in_channels, best_model_tracker, i, experiments
    )

    # Check if directory exists
    if not version_root.exists():
        p(f"Directory does not exist: {version_root}", color1 = c.ORANGE)
        continue

    # Find ALL version subdirectories
    version_dirs = sorted(list(version_root.glob("v*")))

    if not version_dirs:
        p(f"No version directories found in {version_root}", color1 = c.ORANGE)
        continue

    # Use LATEST version
    latest_version_dir = version_dirs[-1]
    p(f"Found {len(version_dirs)} versions, using latest: {latest_version_dir.name}", color1 = c.GREEN)

    # Look for checkpoint files
    checkpoint_files = []
    for pattern in ["checkpoint.pth", "best_model.pth"]:
        found = list(latest_version_dir.glob(pattern))
        checkpoint_files.extend(found)

    if not checkpoint_files:
        p(f"No checkpoint files in {latest_version_dir.name}", color1 = c.ORANGE)
        continue

    # Prefer checkpoint.pth over best_model.pth
    checkpoint_files.sort(key = lambda x: 0 if x.name == "checkpoint.pth" else 1)
    checkpoint_path = checkpoint_files[0]

    p(f"Loading: {checkpoint_path.name}", color1 = c.CYAN)

    try:
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location = "cpu")

        # Extract metrics with fallbacks
        metrics_dict = { }

        for metric in metrics_to_extract:
            value = None

            # Try direct key
            if metric in ckpt:
                value = ckpt[metric]

            # Try nested in 'metrics'
            elif 'metrics' in ckpt and metric in ckpt['metrics']:
                value = ckpt['metrics'][metric]

            # Handle epoch/final_epoch aliases
            elif metric == 'final_epoch' and 'epoch' in ckpt:
                value = ckpt['epoch']
            elif metric == 'epoch' and 'final_epoch' in ckpt:
                value = ckpt['final_epoch']

            # Handle other aliases
            elif metric == 'iou' and 'val_iou' in ckpt:
                value = ckpt['val_iou']
            elif metric == 'accuracy' and 'val_accuracy' in ckpt:
                value = ckpt['val_accuracy']

            if value is not None:
                metrics_dict[metric] = value

        # Get filter string
        filters_str = str(filters) if filters else 'none'

        # Try to load config snapshot for extra info
        config_snapshot_path = latest_version_dir / "config_snapshot.json"
        additional_info = { }
        if config_snapshot_path.exists():
            try:
                with open(config_snapshot_path, 'r') as f:
                    snapshot = json.load(f)
                    additional_info = snapshot.get('config', { }).get('extra', { }).get('experiment', { })
            except Exception:
                pass

        # Build result entry
        result_entry = {
            'model':           model_name,
            'mode':            mode,
            'filters':         filters_str,
            'version':         latest_version_dir.name,
            'checkpoint_path': str(checkpoint_path),
            'version_path':    str(latest_version_dir),
            **additional_info,
            **metrics_dict
        }

        experiment_results.append(result_entry)

        p(f"✓ Loaded successfully - Metrics: {len(metrics_dict)}/{len(metrics_to_extract)}", color1 = c.GREEN)

        # Show missing metrics
        missing = [m for m in metrics_to_extract if m not in metrics_dict]
        if missing and len(missing) < 5:
            p(f"  Missing: {', '.join(missing)}", color1 = c.ORANGE)

    except Exception as e:
        p("ERROR loading checkpoint", str(e), color1 = c.RED)
        continue

# Convert to DataFrame
df_results = pd.DataFrame(experiment_results)

# %%
# Display results
p()
t("Results Summary")
p("Total experiments found", len(df_results))

if len(df_results) > 0:
    p("Available columns:", list(df_results.columns), show = 50, color1 = c.CYAN)

    # Select columns for display
    display_columns = ['model', 'mode', 'filters', 'version']
    for col in ['best_val_loss', 'final_epoch', 'val_accuracy', 'val_iou', 'iou']:
        if col in df_results.columns:
            display_columns.append(col)

    p("Displaying columns:", display_columns, color1 = c.CYAN)

    # Sort by best_val_loss if available
    sort_column = 'best_val_loss' if 'best_val_loss' in df_results.columns else display_columns[0]

    try:
        df_sorted = df_results.sort_values(by = sort_column, ascending = True)

        p()
        p("Top 10 Experiments", color1 = c.BLUE, bold = True)

        # Format display
        display_df = df_sorted[display_columns].head(10).copy()
        for col in display_df.columns:
            if display_df[col].dtype == float:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")

        print(display_df.to_string(index = False))

    except Exception as e:
        p(f"Error displaying results: {e}", color1 = c.RED)

else:
    p("No experiment results found!", color1 = c.RED, bold = True)

# %%
# Aggregation summary
agg_dict = { }
for col in metrics_to_extract:
    if col in df_results.columns:
        agg_dict[col] = ['min', 'mean', 'count']

if agg_dict and len(df_results) > 0:
    p()
    p("Detailed Summary:", color1 = c.BLUE, bold = True)

    summary = df_results.groupby(['model', 'mode']).agg(agg_dict).reset_index()
    summary.columns = [' '.join(col).strip() for col in summary.columns.values]

    # Format floats
    float_cols = [col for col in summary.columns if 'min' in col or 'mean' in col]
    for col in float_cols:
        summary[col] = summary[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")

    try:
        summary = summary.sort_values('best_val_loss min', ascending = True)
    except:
        pass

    print(summary.to_string(index = False))
else:
    p("No aggregatable data", color1 = c.ORANGE)




# %%
summary

# %%
# Convert to DataFrame for easy analysis
df_results = pd.DataFrame(experiment_results)

p("Total experiments found", len(df_results))

# Print available columns for debugging
p("Available columns:", list(df_results.columns), show = 50, color1 = c.RED)

# Dynamically select columns based on available metrics
display_columns = ['model', 'mode', 'filters']
additional_cols = metrics_to_extract

# Add additional columns that have non-None values
for col in additional_cols:
    if col in df_results.columns:
        display_columns.append(col)

p("Columns to display:", display_columns, color1 = c.CYAN)


# %%
t("Top 10 Experiments")

# Check if we have any sortable columns
sort_column = 'best_val_loss' if 'best_val_loss' in df_results.columns else display_columns[0]

try:
    # Sort and select top 10 experiments
    sorted_results = df_results[display_columns].sort_values(
            by = sort_column, ascending = True
    )

    # Format float columns to limit decimal places
    float_cols = [col for col in display_columns if df_results[col].dtype == float]
    for col in float_cols:
        sorted_results[col] = sorted_results[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else x)

    p(sorted_results.head(10).to_string(index = False), color1 = c.BLACK)

except Exception as e:
    p(f"Error sorting results: {e}", color1 = c.RED)
    p(sorted_results.head(10).to_string(index = False), color1 = c.BLACK)

# %%
sorted_results

# %%
# Prepare aggregation dictionary dynamically
agg_dict = { }
for col in additional_cols:
    if col in df_results.columns:
        agg_dict[col] = ['min', 'mean', 'count']

if agg_dict:
    p("\nDetailed Summary:", color1 = c.BLUE, bold = True)

    # Create summary with dynamic columns
    summary = df_results.groupby(['model', 'mode']).agg(agg_dict).reset_index()

    # Flatten multi-level column names for readability
    summary.columns = [' '.join(col).strip() for col in summary.columns.values]

    # Format float columns to limit decimal places
    float_cols = [col for col in summary.columns if 'min' in col or 'mean' in col]
    for col in float_cols:
        summary[col] = summary[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else x)

    # Sort if possible
    try:
        summary = summary.sort_values('best_val_loss min', ascending = True)
    except:
        pass

    p(summary.to_string(index = False), color1 = c.BLACK)

else:
    p("No aggregatable columns found", color1 = c.RED)

# %%
summary

# %%

t("Best Model Selection")

if len(df_results) == 0:
    p("ERROR", "No experiments to compare", color1 = c.RED)
else:
    # Get best model (lowest validation loss)
    best_row = df_results.iloc[0]

    p("=" * 80, color1 = c.CYAN, bold = True)
    p("BEST MODEL", color1 = c.GREEN, bold = True)
    p("=" * 80, color1 = c.CYAN, bold = True)
    p("")
    p("Model", best_row['model'], color1 = c.GREEN)
    p("Input Mode", best_row['mode'], color1 = c.GREEN)
    p("Filters", best_row['filters'], color1 = c.GREEN)
    p("Best Val Loss", f"{best_row['best_val_loss']:.6f}", color1 = c.GREEN)
    p("Final Epoch", best_row['final_epoch'], color1 = c.GREEN)
    # Extract version from version_path if 'version' is not present
    version = best_row.get('version', Path(best_row['version_path']).name)
    p("Version", version, color1 = c.GREEN)

    p("Path", best_row['version_path'], color1 = c.BLUE)
    p("")

    # Check if this is better than previous best
    previous_best_path = config.paths.models / "BEST_MODEL.txt"

    is_new_best = False
    improvement = None

    if previous_best_path.exists():
        with open(previous_best_path, 'r') as f:
            lines = f.readlines()
            prev_loss = None
            for line in lines:
                if "best_val_loss:" in line:
                    try:
                        prev_loss = float(line.split(':')[1].strip())
                        break
                    except:
                        pass

        if prev_loss is not None:
            improvement = prev_loss - best_row['best_val_loss']

            if best_row['best_val_loss'] < prev_loss:
                is_new_best = True
                p("IMPROVEMENT", f"{improvement:.6f} (lower is better)", color1 = c.GREEN, bold = True)
            else:
                p("NO IMPROVEMENT", f"Previous best was {prev_loss:.6f}", color1 = c.ORANGE)
    else:
        is_new_best = True
        p("FIRST RUN", "No previous best to compare", color1 = c.CYAN)

    # Save new best model info
    if is_new_best:
        from datetime import datetime


        with open(previous_best_path, 'w') as f:
            f.write(f"model: {best_row['model']}\n")
            f.write(f"mode: {best_row['mode']}\n")
            f.write(f"filters: {best_row['filters']}\n")
            f.write(f"best_val_loss: {best_row['best_val_loss']}\n")
            f.write(f"epoch: {best_row['final_epoch']}\n")
            f.write(f"version: {best_row['version']}\n")
            f.write(f"path: {best_row['version_path']}\n")
            f.write(f"timestamp: {datetime.now().isoformat()}\n")

        p("")
        p("✓ Saved new best model info", str(previous_best_path), color1 = c.GREEN)

        # Copy best model to top-level for easy access
        best_model_src = Path(best_row['version_path']) / "best_model.pth"
        best_model_dst = config.paths.models / "BEST_MODEL.pth"

        if best_model_src.exists():
            import shutil


            shutil.copy2(best_model_src, best_model_dst)
            p("✓ Copied best model", str(best_model_dst), color1 = c.GREEN)

# %%
t("Performance Comparison Visualization")

if len(df_results) > 0:
    import matplotlib.pyplot as plt


    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize = (15, 10))
    fig.suptitle('Model Performance Comparison', fontsize = 16, fontweight = 'bold')

    # Plot 1: Validation Loss by Model
    ax1 = axes[0, 0]
    df_pivot = df_results.pivot_table(values = 'best_val_loss', index = 'model', columns = 'mode', aggfunc = 'min')
    df_pivot.plot(kind = 'bar', ax = ax1, color = ['#3498db', '#e74c3c'])
    ax1.set_title('Best Validation Loss by Model & Mode')
    ax1.set_ylabel('Validation Loss')
    ax1.set_xlabel('Model')
    ax1.legend(title = 'Input Mode')
    ax1.grid(True, alpha = 0.3)

    # Plot 2: Loss Distribution
    ax2 = axes[0, 1]
    df_results.boxplot(column = 'best_val_loss', by = 'model', ax = ax2)
    ax2.set_title('Validation Loss Distribution by Model')
    ax2.set_ylabel('Validation Loss')
    ax2.set_xlabel('Model')
    plt.sca(ax2)
    plt.xticks(rotation = 45)

    # Plot 3: Mode Comparison
    ax3 = axes[1, 0]
    mode_stats = df_results.groupby('mode')['best_val_loss'].agg(['mean', 'min', 'max'])
    mode_stats.plot(kind = 'bar', ax = ax3, color = ['#2ecc71', '#f39c12', '#e74c3c'])
    ax3.set_title('Input Mode Performance')
    ax3.set_ylabel('Validation Loss')
    ax3.set_xlabel('Input Mode')
    ax3.legend(['Mean', 'Min', 'Max'])
    ax3.grid(True, alpha = 0.3)

    # Plot 4: Convergence Speed
    ax4 = axes[1, 1]
    for _, row in df_results.head(5).iterrows():
        label = f"{row['model']}-{row['mode']}"
        ax4.scatter(row['final_epoch'], row['best_val_loss'], s = 100, label = label)

    ax4.set_title('Convergence: Epochs vs Loss (Top 5)')
    ax4.set_xlabel('Final Epoch')
    ax4.set_ylabel('Best Validation Loss')
    ax4.legend(fontsize = 8)
    ax4.grid(True, alpha = 0.3)

    plt.tight_layout()

    # Save figure
    plot_path = config.paths.models / "experiment_comparison.png"
    plt.savefig(plot_path, dpi = 150, bbox_inches = 'tight')
    p("✓ Saved comparison plot", str(plot_path), color1 = c.GREEN)

    plt.show()

# %%
t("Exporting Results")
from datetime import datetime


if len(df_results) > 0:
    # Save full results
    csv_path = config.paths.models / "all_experiments.csv"
    df_results.to_csv(csv_path, index = False)
    p("✓ Saved experiment results", str(csv_path), color1 = c.GREEN)

    # Create summary report
    summary_path = config.paths.models / "EXPERIMENT_SUMMARY.txt"

    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EXPERIMENT SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total experiments: {len(df_results)}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")

        f.write("BEST MODEL:\n")
        best = df_results.iloc[0]
        f.write(f"  Model: {best['model']}\n")
        f.write(f"  Mode: {best['mode']}\n")
        f.write(f"  Filters: {best['filters']}\n")
        f.write(f"  Best Val Loss: {best['best_val_loss']:.6f}\n")
        f.write(f"  Path: {best['version_path']}\n\n")

        f.write("TOP 5 MODELS:\n")
        f.write("-" * 80 + "\n")
        for idx, row in df_results.head(5).iterrows():
            f.write(f"{idx + 1}. {row['model']} ({row['mode']}) - Loss: {row['best_val_loss']:.6f}\n")
            f.write(f"   Filters: {row['filters']}\n")
            f.write(f"   Path: {row['version_path']}\n\n")

        f.write("\nSTATISTICS BY MODEL:\n")
        f.write("-" * 80 + "\n")
        model_stats = df_results.groupby('model')['best_val_loss'].agg(['count', 'mean', 'min', 'max'])
        f.write(model_stats.to_string())
        f.write("\n\n")

        f.write("STATISTICS BY MODE:\n")
        f.write("-" * 80 + "\n")
        mode_stats = df_results.groupby('mode')['best_val_loss'].agg(['count', 'mean', 'min', 'max'])
        f.write(mode_stats.to_string())

    p("✓ Saved summary report", str(summary_path), color1 = c.GREEN)

    p("")
    p("=" * 80, color1 = c.CYAN, bold = True)
    p("EXPERIMENT COMPARISON COMPLETE", color1 = c.GREEN, bold = True)
    p("=" * 80, color1 = c.CYAN, bold = True)


# %%

# %% [markdown]
# #### Step 5: Class-Specific Training
#
# **Strategy:**
# Trainning separate models for:
# 1. Individual trees
# 2. Groups of trees
# 3. Combined predictions
#
# **Rationale:**
# - Individual trees have distinct boundaries
# - Tree groups have larger, more diffuse edges
# - Specialized models may perform better
#
#

# %%
def train_class_specific_model( class_name, model_name = 'simple_cnn' ):
    """
    Train a model for a specific class.
    """
    t(f"Training {model_name} for class: {class_name}")

    # Filter entries by class
    class_entries = [
        e for e in entries
        if any(item.cls == class_name for item in e.items)
    ]

    p(f"Images with {class_name}", len(class_entries))

    if len(class_entries) < 10:
        p("Warning", "Insufficient samples for training", color1 = c.ORANGE)
        return None

    # Split
    split_idx = int(0.8 * len(class_entries))
    train_entries = class_entries[:split_idx]
    val_entries = class_entries[split_idx:]

    # Create datasets with class filter
    train_tf = get_train_augmentations(config.train.image_size)
    val_tf = get_val_augmentations(config.train.image_size)

    train_ds = ImageMaskDataset(
            train_entries,
            config.paths.train_images,
            classes = [class_name],
            transform = train_tf
    )

    val_ds = ImageMaskDataset(
            val_entries,
            config.paths.train_images,
            classes = [class_name],
            transform = val_tf
    )

    train_loader = DataLoader(
            train_ds,
            batch_size = config.train.batch_size,
            shuffle = True,
            num_workers = config.train.num_workers
    )

    val_loader = DataLoader(
            val_ds,
            batch_size = config.train.batch_size,
            shuffle = False,
            num_workers = config.train.num_workers
    )

    # Train using config epochs
    trainer = run_training(
            config = config,
            train_loader = train_loader,
            val_loader = val_loader,
            version_root = config.paths.models,
            model_name = model_name,
            #in_channels = in_channels,
    )

    return trainer


p("", "Class-specific training configured")


# %%
# ## Train individual tree model
# try:
#     trainer_individual = train_class_specific_model('individual_tree', 'simple_cnn')
# except Exception as e:
#     p("Failed to train individual_tree model", str(e), color1=c.RED)
#     trainer_individual = None
#
# try:
#     trainer_group = train_class_specific_model('group_of_trees', 'simple_cnn')
# except Exception as e:
#     p("Failed to train group_of_trees model", str(e), color1=c.RED)
#     trainer_group = None

# %% [markdown]
# #### Step 6: Ensemble Predictions
#
# **Approach:**
# Combine predictions from multiple models:
# 1. RGB-trained model
# 2. Filter-enhanced model
# 3. Class-specific models
#
# **Fusion methods:**
# - Average (simple)
# - Weighted average (based on validation IoU)
# - Majority voting (threshold-based)
#
#

# %%
def ensemble_predict( models_and_weights, image_tensor, device = 'cpu' ):
    """
    Combine predictions from multiple models.
    """
    predictions = []
    total_weight = sum(w for _, w in models_and_weights)

    for model, weight in models_and_weights:
        model = model.to(device).eval()
        with torch.no_grad():
            pred = torch.sigmoid(model(image_tensor.to(device)))
        predictions.append(pred.cpu() * weight)

    # Weighted average
    avg_pred = sum(predictions) / total_weight
    return avg_pred


p("", "Ensemble prediction function ready")



# %%

# %%

# %%

# %% [markdown]
# #### Step 7: Submission Generation
#
# **Current Status:**
# - Prediction pipeline exists (notebook 05)
# - Submission export implemented (`export_submission`)
#

# %%
p("eval_images", config.paths.eval_images)

# %%
from src.prediction.pipeline import Predictor
from src.prediction.submission import export_submission


def generate_submission( model_path, model_name, eval_dir, output_path ):
    """
    Generate final submission JSON.
    """
    t("Generating Submission")

    # Initialize predictor
    predictor = Predictor(
            model_path = model_path,
            model_name = model_name,
            image_size = config.train.image_size
    )

    # Run predictions
    val_tf = get_val_augmentations(config.train.image_size)

    results = predictor.run_on_folder(eval_dir, transform = val_tf)

    p("Predictions generated", len(results))

    # Export submission
    export_submission(results, output_path)
    p("Submission saved", output_path)

    with open(output_path, 'r') as f:
        data = json.load(f)

    p("Images in submission", len(data.get('images', [])))

    # Check first entry structure
    if data.get('images'):
        first_img = data['images'][0]
        p("Sample entry keys", list(first_img.keys()), show = 15)
        if first_img.get('annotations'):
            first_ann = first_img['annotations'][0]
            p("Sample annotation keys", list(first_ann.keys()))

    return output_path


# %%
from pathlib import Path
import pandas as pd

t("Generating Submission For Each Experiment From all_experiments.csv")

csv_path = config.paths.models / "all_experiments.csv"

if not csv_path.exists():
    p("Warning", f"No all_experiments.csv found at {csv_path}", color1=c.ORANGE)
else:
    df_exp = pd.read_csv(csv_path)

    if len(df_exp) == 0:
        p("Warning", "all_experiments.csv is empty, no experiments to export", color1=c.ORANGE)
    else:
        num_ok = 0
        num_missing = 0

        for idx, row in df_exp.iterrows():
            model_name = str(row.get("model", "unknown_model"))
            mode = str(row.get("mode", "unknown_mode"))
            version_path_str = row.get("version_path", "")

            if not isinstance(version_path_str, str) or not version_path_str:
                p("Skipping row", f"{idx} (no valid version_path)", color1=c.ORANGE)
                num_missing += 1
                continue

            version_path = Path(version_path_str)

            if not version_path.exists():
                p(
                    "Skipping experiment",
                    f"{model_name} ({mode}) missing folder: {version_path}",
                    color1=c.ORANGE,
                )
                num_missing += 1
                continue

            best_model_path = version_path / "best_model.pth"
            checkpoint_path = version_path / "checkpoint.pth"

            if best_model_path.exists():
                model_weights = best_model_path
                weight_type = "best_model.pth"
            elif checkpoint_path.exists():
                model_weights = checkpoint_path
                weight_type = "checkpoint.pth"
            else:
                p(
                    "Skipping experiment",
                    f"{model_name} ({mode}) no best_model.pth or checkpoint.pth in {version_path}",
                    color1=c.ORANGE,
                )
                num_missing += 1
                continue

            p(
                "Generating submission for",
                f"{model_name} ({mode}) using {weight_type}",
                color1=c.GREEN,
            )
            p("From", str(version_path), color1=c.BLUE)

            output_file = version_path / "SUBMISSION.json"

            generate_submission(
                model_path=model_weights,
                model_name=model_name,
                eval_dir=config.paths.eval_images,
                output_path=output_file,
            )

            p("✓ Submission generated", str(output_file), color1=c.GREEN, bold=True)
            num_ok += 1

        p(
            "Summary",
            f"Submissions generated for {num_ok} experiments, skipped {num_missing}",
            color1=c.CYAN,
        )


# %%

t("Generating Final Submission with Best Model")

# Load best model info
best_model_info_path = config.paths.models / "BEST_MODEL.txt"

if best_model_info_path.exists():
    with open(best_model_info_path, 'r') as f:
        lines = f.readlines()
        best_model_name = None
        best_model_path = None

        for line in lines:
            if "model:" in line:
                best_model_name = line.split(':')[1].strip()
            elif "path:" in line:
                best_model_path = Path(line.split(':', 1)[1].strip())

        if best_model_path and best_model_name:
            model_weights = best_model_path / "best_model.pth"

            p("Using best model", best_model_name, color1 = c.GREEN)
            p("From", str(best_model_path), color1 = c.BLUE)

            # from datetime import datetime
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Generate submission
            #output_file = config.paths.models / f"FINAL_SUBMISSION_{timestamp}.json"
            output_file = config.paths.models / "FINAL_SUBMISSION.json"

            generate_submission(
                    model_path = model_weights,
                    model_name = best_model_name,
                    eval_dir = config.paths.eval_images,
                    output_path = output_file
            )

            p("✓ Final submission generated", str(output_file), color1 = c.GREEN, bold = True)
else:
    p("Warning", "No best model info found, skipping final submission", color1 = c.ORANGE)

# %%

# %% [markdown]
# #### Generate submissions for all trained experiments

# %%
best_submissions = []

# Extract unique model names and modes from experiments
all_model_names = set(exp[0] for exp in experiments)
all_input_modes = set(exp[1] for exp in experiments)

for model_name in all_model_names:
    for input_mode in all_input_modes:

        base_path = config.paths.models / model_name / input_mode

        if not base_path.exists():
            continue

        # Check for direct versions (no filter subdirectory)
        vm = VersionManager(base_path)
        latest_version = vm.find_latest()

        if latest_version is not None:
            model_path = latest_version / "best_model.pth"
            if model_path.exists():
                output_file = latest_version / "submission.json"

                p("Generating submission", f"{model_name}/{input_mode}", color1 = c.MAGENTA)

                try:
                    generate_submission(
                            model_path = model_path,
                            model_name = model_name,
                            eval_dir = config.paths.eval_images,
                            output_path = output_file
                    )

                    best_submissions.append(
                            {
                                "model":      model_name,
                                "mode":       input_mode,
                                "filters":    None,
                                "version":    str(latest_version),
                                "submission": str(output_file)
                            }
                    )
                    p("Saved submission", str(output_file), color1 = c.GREEN)

                except Exception as e:
                    p("Failed", str(e), color1 = c.RED)

        # Check for filter subdirectories (for 'filtered' mode)
        if input_mode == 'filtered':
            for filter_subdir in base_path.iterdir():
                if not filter_subdir.is_dir():
                    continue
                if filter_subdir.name.startswith('v'):
                    continue

                vm_filter = VersionManager(filter_subdir)
                latest_filter_version = vm_filter.find_latest()

                if latest_filter_version is None:
                    continue

                model_path = latest_filter_version / "best_model.pth"
                if not model_path.exists():
                    continue

                output_file = latest_filter_version / "submission.json"
                filter_name = filter_subdir.name

                p("Generating submission", f"{model_name}/{input_mode}/{filter_name}", color1 = c.MAGENTA)

                try:
                    generate_submission(
                            model_path = model_path,
                            model_name = model_name,
                            eval_dir = config.paths.eval_images,
                            output_path = output_file
                    )

                    best_submissions.append(
                            {
                                "model":      model_name,
                                "mode":       input_mode,
                                "filters":    filter_name,
                                "version":    str(latest_filter_version),
                                "submission": str(output_file)
                            }
                    )
                    p("Saved submission", str(output_file), color1 = c.GREEN)

                except Exception as e:
                    p("Failed", str(e), color1 = c.RED)

p("")
t("Submission Generation Complete")
p("Total submissions generated", len(best_submissions))

# %%
# Build global list of all best experiment submissions
best_overall = None

for item in best_submissions:
    ckpt_path = Path(item["version"]) / "checkpoint.pth"

    if not ckpt_path.exists():
        continue

    try:
        ckpt = torch.load(ckpt_path, map_location = "cpu")
        val_loss = ckpt.get("best_val_loss", None)
        if val_loss is None:
            continue

        if best_overall is None or val_loss < best_overall["best_val_loss"]:
            best_overall = {
                "model":         item["model"],
                "mode":          item["mode"],
                "version":       item["version"],
                "submission":    item["submission"],
                "best_val_loss": val_loss
            }
    except Exception as e:
        p("Warning", f"Could not load checkpoint {ckpt_path}: {e}", color1 = c.ORANGE)
        continue

# Append overall best only if not already present
if best_overall:
    already_present = any(
            s.get("version") == best_overall["version"] and s.get("overall_best")
            for s in best_submissions
    )
    if not already_present:
        best_submissions.append(
                {
                    "model":         best_overall["model"],
                    "mode":          best_overall["mode"],
                    "version":       best_overall["version"],
                    "submission":    best_overall["submission"],
                    "best_val_loss": best_overall["best_val_loss"],
                    "overall_best":  True
                }
        )
    else:
        p("best already in list")

# ===== Summary =====
t("Best Submissions Summary")

for item in best_submissions:
    is_overall = item.get("overall_best", False)
    label = "\nOVERALL BEST" if is_overall else "\nExperiment"

    p(label, f"{item['model']} | {item['mode']}", color1 = c.MAGENTA)
    p("Version", item["version"])
    p("Submission", item["submission"])

    if "best_val_loss" in item:
        p("Val Loss", item["best_val_loss"])
    p("")


# %% [markdown]
# #### Build global list of all best experiment submissions

# %%

best_overall = None

for item in best_submissions:
    ckpt_path = Path(item["version"]) / "checkpoint.pth"

    if not ckpt_path.exists():
        continue

    try:
        ckpt = torch.load(ckpt_path, map_location = "cpu")
        val_loss = ckpt.get("best_val_loss", None)
        if val_loss is None:
            continue

        if best_overall is None or val_loss < best_overall["best_val_loss"]:
            best_overall = {
                "model":         item["model"],
                "mode":          item["mode"],
                "filters":       item.get("filters"),
                "version":       item["version"],
                "submission":    item["submission"],
                "best_val_loss": val_loss
            }
    except Exception as e:
        p("Warning", f"Could not load checkpoint {ckpt_path}: {e}", color1 = c.ORANGE)
        continue

# Append overall best only if not already present
if best_overall:
    already_present = any(
            s.get("version") == best_overall["version"] and s.get("overall_best")
            for s in best_submissions
    )
    if not already_present:
        best_submissions.append(
                {
                    "model":         best_overall["model"],
                    "mode":          best_overall["mode"],
                    "filters":       best_overall.get("filters"),
                    "version":       best_overall["version"],
                    "submission":    best_overall["submission"],
                    "best_val_loss": best_overall["best_val_loss"],
                    "overall_best":  True
                }
        )
    else:
        p("best already in list")


def visualize_experiment_predictions( item, image_dir, num_samples = 3 ):
    """
    Visualize predictions for a single experiment with correct preprocessing and sizing.
    """
    from pathlib import Path
    from src.models.zoo import build_model
    from src.data.augmentations import get_val_augmentations
    from src.exploration.visualize import show_side_by_side
    import random
    import cv2
    import torch
    import numpy as np

    model_name = item["model"]
    mode = item["mode"]
    version_path = Path(item["version"])
    filters_str = item.get("filters")

    # Parse filters from string if needed
    if filters_str and filters_str != "none" and filters_str != "None":
        if isinstance(filters_str, str):
            if filters_str.startswith("["):
                import ast

                try:
                    filters = ast.literal_eval(filters_str)
                except:
                    filters = [filters_str.strip("[]'\" ")]
            else:
                suffixes = { 'x', 'y', '3x3', '5x5', '7x7' }
                parts = filters_str.split("_")
                filters = []
                i = 0
                while i < len(parts) and len(filters) < 3:
                    if i + 1 < len(parts) and parts[i + 1] in suffixes:
                        filters.append(f"{parts[i]}_{parts[i + 1]}")
                        i += 2
                    else:
                        filters.append(parts[i])
                        i += 1
        else:
            filters = filters_str
    else:
        filters = None

    # Find model weights
    best_model_path = version_path / "best_model.pth"
    checkpoint_path = version_path / "checkpoint.pth"

    if best_model_path.exists():
        model_path = best_model_path
    elif checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location = "cpu")
        if "model" not in checkpoint:
            p("No model state in checkpoint", str(checkpoint_path), color1 = c.ORANGE)
            return
        model_path = checkpoint_path
    else:
        p("No model weights found", str(version_path), color1 = c.ORANGE)
        return

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Determine input channels based on mode and filters
        if mode == "concat" and filters:
            in_channels = 6  # RGB + 3 filters
        else:
            in_channels = 3  # RGB or filtered RGB

        # Build model with correct input channels
        model = build_model(model_name, in_channels = in_channels, out_channels = 3).to(device)

        # Load weights
        state = torch.load(model_path, map_location = device)
        if "model" in state:
            model.load_state_dict(state["model"])
        else:
            model.load_state_dict(state)
        model.eval()

        val_tf = get_val_augmentations(config.train.image_size)

        # Get image files
        image_files = sorted(list(image_dir.glob("*.png")))
        if not image_files:
            image_files = sorted(list(image_dir.glob("*.tif")))

        if not image_files:
            p("No images found in", str(image_dir), color1 = c.ORANGE)
            return

        # Sample random images
        sample_files = random.sample(image_files, min(num_samples, len(image_files)))

        for img_path in sample_files:
            # Load original image to preserve dimensions
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            original_shape = img_rgb.shape

            # Apply same preprocessing as training
            if mode == "filtered" and filters:
                from src.data.image_loader import create_enhanced_image

                try:
                    processed_img = create_enhanced_image(img_rgb, filters)
                except Exception as e:
                    p("Filter preprocessing failed", str(e), color1 = c.ORANGE)
                    processed_img = img_rgb
            elif mode == "concat" and filters:
                from src.data.image_loader import create_enhanced_image

                try:
                    filtered_img = create_enhanced_image(img_rgb, filters)
                    # Concatenate RGB + filtered for 6 channels
                    processed_img = np.concatenate([img_rgb, filtered_img], axis = 2)
                except Exception as e:
                    p("Concat preprocessing failed", str(e), color1 = c.ORANGE)
                    processed_img = img_rgb
            else:
                processed_img = img_rgb

            # Apply transforms
            augmented = val_tf(image = processed_img)
            img_tensor = augmented["image"]

            # Ensure correct format [C, H, W]
            if isinstance(img_tensor, torch.Tensor):
                if img_tensor.ndim == 3 and img_tensor.shape[-1] in [3, 6]:
                    img_tensor = img_tensor.permute(2, 0, 1)
                img_tensor = img_tensor.float()
            else:
                img_tensor = torch.from_numpy(img_tensor.transpose(2, 0, 1)).float()

            # Normalize to 0-1 if needed
            if img_tensor.max() > 1.0:
                img_tensor = img_tensor / 255.0

            img_tensor = img_tensor.unsqueeze(0).to(device)

            # Predict
            with torch.no_grad():
                pred = model(img_tensor)

            # Convert to mask
            if pred.shape[1] == 3:  # Multi-class
                pred_classes = torch.argmax(pred, dim = 1).squeeze().cpu().numpy()
            else:  # Binary
                pred_classes = (torch.sigmoid(pred).squeeze().cpu().numpy() > 0.5).astype(np.uint8)

            pred_mask = pred_classes.astype(np.uint8)

            # CRITICAL: Resize prediction to match original image dimensions
            orig_h, orig_w = original_shape[:2]
            if pred_mask.shape != (orig_h, orig_w):
                pred_mask_resized = cv2.resize(
                        pred_mask, (orig_w, orig_h),
                        interpolation = cv2.INTER_NEAREST
                )
            else:
                pred_mask_resized = pred_mask

            # Color mask for display using original dimensions
            mask_rgb = np.zeros((orig_h, orig_w, 3), dtype = np.uint8)
            mask_rgb[pred_mask_resized == 1] = (0, 255, 0)  # individual trees - green
            mask_rgb[pred_mask_resized == 2] = (255, 255, 0)  # groups - yellow

            # Create overlay with original image (no resizing)
            overlay = cv2.addWeighted(img_rgb, 0.6, mask_rgb, 0.4, 0)

            # Show results with properly sized components
            show_side_by_side(
                    img_rgb, pred_mask_resized, mask_rgb, overlay,
                    titles = (f"Original: {img_path.name}", "Pred Classes", "Color Mask", "Overlay"),
                    cmaps = [None, "gray", None, None],
                    maxcolumns = 4,
            )

    except Exception as e:
        p("Visualization failed", str(e), color1 = c.RED)
        import traceback

        traceback.print_exc()


# ===== Summary with Visualizations =====
t("Best Submissions Summary with Predictions")

image_dir = config.paths.eval_images

for item in best_submissions:
    is_overall = item.get("overall_best", False)
    label = "OVERALL BEST" if is_overall else "Experiment"

    p("")
    p("=" * 70, color1 = c.CYAN)
    p(label, f"{item['model']} | {item['mode']}", color1 = c.MAGENTA, bold = is_overall)
    p("=" * 70, color1 = c.CYAN)
    p("Version", item["version"])
    p("Filters", item.get("filters", "none"))
    p("Submission", item["submission"])

    if "best_val_loss" in item:
        p("Val Loss", f"{item['best_val_loss']:.6f}", color1 = c.GREEN)

    # Visualize predictions for this experiment
    t(f"Predictions: {item['model']} ({item['mode']})")
    visualize_experiment_predictions(item, image_dir, num_samples = 1)

    p("")

# %%

t("Visualizing Training Data (Ground Truth)")

image_dir = config.paths.train_images
p("Train images dir", image_dir)

# sample three entries
sample_entries = random.sample(entries, 1)

for e in sample_entries:
    p("Image", e.image_path.name)

    try:
        img = load_image(image_dir, e)
        mask = mask_all(e)
        mask_rgb = color_mask(e)
        overlay = cv2.addWeighted(img, 0.6, mask_rgb, 0.4, 0)

        show_side_by_side(
                img, mask, mask_rgb, overlay,
                titles = ("Original", "Mask", "Color Mask", "Overlay"),
                maxcolumns = 4
        )
    except Exception as e_viz:
        p("Failed to visualize", str(e_viz), color1 = c.ORANGE)


# %%
# %%
t("Visualizing sample predictions")

def find_available_model_weights(config, preferred_model=None):
    """Find available trained model weights."""
    models_dir = config.paths.models

    # Search for .pth files recursively
    model_files = []

    if models_dir.exists():
        # Look for all .pth files
        for pth_file in models_dir.rglob("*.pth"):
            model_files.append(pth_file)

    # Prioritize best_model.pth over checkpoint.pth
    best_models = [f for f in model_files if f.name == "best_model.pth"]
    checkpoint_models = [f for f in model_files if f.name == "checkpoint.pth"]

    # Return first best_model, then first checkpoint
    if best_models:
        return best_models[0], best_models[0].parent
    elif checkpoint_models:
        return checkpoint_models[0], checkpoint_models[0].parent

    return None, None


# Try to find model weights
test_model_weights, test_version_dir = find_available_model_weights(config, "simple_cnn")

if test_model_weights is None:
    p("No trained models found!", color1 = c.RED, bold = True)
    p("Available directories in models:", color1 = c.ORANGE)

    models_dir = config.paths.models
    if models_dir.exists():
        for item in models_dir.iterdir():
            if item.is_dir():
                p("  ", item.name)
                # Show subdirectories
                for subitem in item.iterdir():
                    if subitem.is_dir():
                        p("    ", subitem.name)
    else:
        p("Models directory doesn't exist:", models_dir)
else:
    p("Using model weights", test_model_weights, color1 = c.GREEN)

    try:
        # Extract model info from path
        parts = test_model_weights.parts
        if "simple_cnn" in str(test_model_weights):
            test_best_model_name = "simple_cnn"
        elif "unet" in str(test_model_weights):
            test_best_model_name = "unet"
        else:
            # Try to extract from path structure
            for part in reversed(parts):
                if part in ["simple_cnn", "unet", "yolov8s", "yolov8m", "yolov8l"]:
                    test_best_model_name = part
                    break
            else:
                test_best_model_name = "simple_cnn"  # fallback

        p("Detected model name", test_best_model_name, color1 = c.CYAN)

        # Determine mode and filters from path
        test_mode = "rgb"  # default
        test_filters = None

        if "filtered" in str(test_model_weights):
            test_mode = "filtered"
            # Try to extract filter names from path
            for part in parts:
                if any(f in part for f in ["sobel", "laplacian", "gaussian", "clahe"]):
                    # This part contains filter names
                    test_filters = part.split("_")
                    break

        p("Mode", test_mode, color1 = c.CYAN)
        p("Filters", test_filters, color1 = c.CYAN)

        # Load and run prediction
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Determine input channels
        in_channels = 6 if test_mode == "concat" and test_filters else 3

        # Build model
        model = build_model(test_best_model_name, in_channels = in_channels, out_channels = 3).to(device)

        # Load weights
        state = torch.load(test_model_weights, map_location = device)
        if "model" in state:
            model.load_state_dict(state["model"])
        else:
            model.load_state_dict(state)
        model.eval()

        p("Model loaded successfully", color1 = c.GREEN)

        # Test visualization with sample images
        eval_dir = config.paths.eval_images
        if eval_dir.exists():
            image_files = list(eval_dir.glob("*.png")) + list(eval_dir.glob("*.tif"))
            if image_files:
                sample_img = random.choice(image_files)
                p("Testing with", sample_img.name, color1 = c.CYAN)

                # Create test item for visualization
                test_item = {
                    "model":   test_best_model_name,
                    "mode":    test_mode,
                    "filters": test_filters,
                    "version": str(test_version_dir)
                }

                # Use the updated visualization function
                visualize_experiment_predictions(test_item, eval_dir, num_samples = 1)
            else:
                p("No images found in eval directory", color1 = c.ORANGE)
        else:
            p("Eval directory not found", eval_dir, color1 = c.ORANGE)

    except Exception as e:
        p("Failed to visualize predictions", str(e), color1 = c.RED)
        import traceback


        traceback.print_exc()


# %%
def test_best_models_by_version(config, eval_dir, num_samples=2):
    """Generate predictions for the best model from each version/experiment."""
    models_dir = config.paths.models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Find all best_model.pth files (prioritize over checkpoint.pth)
    best_models = list(models_dir.rglob("best_model.pth"))
    checkpoint_models = list(models_dir.rglob("checkpoint.pth"))

    # Group by experiment (model + mode + version)
    experiments = {}

    for model_path in best_models + checkpoint_models:
        # Extract experiment info from path
        parts = model_path.parts

        # Find model name
        model_name = "simple_cnn"
        for part in parts:
            if part in ["simple_cnn", "unet", "yolov8s", "yolov8m", "yolov8l"]:
                model_name = part
                break

        # Find mode
        mode = "rgb"
        if "filtered" in str(model_path):
            mode = "filtered"
        elif "concat" in str(model_path):
            mode = "concat"

        # Find version/size info
        version_info = []
        for part in parts:
            if part.startswith("v") or "size" in part:
                version_info.append(part)
        version_str = "_".join(version_info) if version_info else "unknown"

        # Create experiment key
        exp_key = f"{model_name}_{mode}_{version_str}"

        # Prioritize best_model.pth over checkpoint.pth
        if exp_key not in experiments:
            experiments[exp_key] = model_path
        elif model_path.name == "best_model.pth" and experiments[exp_key].name == "checkpoint.pth":
            experiments[exp_key] = model_path

    print(f"Found {len(experiments)} unique experiments:")
    for exp_key, model_path in experiments.items():
        print(f"  {exp_key}: {model_path}")

    # Get sample images
    image_files = list(eval_dir.glob("*.png")) + list(eval_dir.glob("*.tif"))
    sample_images = random.sample(image_files, min(num_samples, len(image_files)))

    # Test each experiment's best model
    for exp_key, model_path in experiments.items():
        try:
            # Extract model info
            model_name = exp_key.split("_")[0]
            mode = exp_key.split("_")[1]

            print(f"\n=== {exp_key} ===")
            print(f"Path: {model_path}")

            # Load model
            in_channels = 3
            if mode == "concat":
                in_channels = 6

            model = build_model(model_name, in_channels=in_channels, out_channels=3).to(device)

            state = torch.load(model_path, map_location=device)
            if "model" in state:
                model.load_state_dict(state["model"])
                if 'val_loss' in state:
                    print(f"Validation loss: {state['val_loss']:.4f}")
                if 'epoch' in state:
                    print(f"Trained for {state['epoch']} epochs")
            else:
                model.load_state_dict(state)

            model.eval()

            # Test predictions
            fig, axes = plt.subplots(len(sample_images), 3, figsize=(12, 4*len(sample_images)))
            if len(sample_images) == 1:
                axes = axes.reshape(1, -1)

            for i, img_path in enumerate(sample_images):
                # Load and preprocess
                img = cv2.imread(str(img_path))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_float = img_rgb.astype(np.float32) / 255.0

                transform = get_val_augmentations(config.train.image_size)
                img_transformed = transform(image=img_float)['image']

                # Predict
                with torch.no_grad():
                    img_batch = img_transformed.unsqueeze(0).to(device)
                    prediction = model(img_batch)
                    pred_probs = torch.softmax(prediction, dim=1)
                    pred_classes = torch.argmax(pred_probs, dim=1)

                # Stats
                pred_np = pred_classes.squeeze().cpu().numpy()
                bg_pct = (pred_np == 0).sum() / pred_np.size * 100
                tree_pct = ((pred_np == 1) | (pred_np == 2)).sum() / pred_np.size * 100

                # Plot
                axes[i, 0].imshow(img_rgb)
                axes[i, 0].set_title(f"{img_path.name}")
                axes[i, 0].axis('off')

                axes[i, 1].imshow(pred_np, cmap='viridis')
                axes[i, 1].set_title(f"Prediction\nTrees: {tree_pct:.1f}%")
                axes[i, 1].axis('off')

                # Tree probability
                tree_prob = torch.max(pred_probs[0, 1:], dim=0)[0].cpu().numpy()
                axes[i, 2].imshow(img_rgb)
                axes[i, 2].imshow(tree_prob, alpha=0.6, cmap='hot')
                axes[i, 2].set_title("Tree Confidence")
                axes[i, 2].axis('off')

            plt.suptitle(exp_key)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Failed to test {exp_key}: {e}")

# Run the test
test_best_models_by_version(config, eval_dir, num_samples=2)


# %%
# %%
def fix_submission_visualization( submission_results, image_dir ):
    """
    Fix mask sizing issues in submission visualization.
    """
    for result in submission_results:
        img_name = result["name"]
        img_path = image_dir / img_name

        # Load original image to get dimensions
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            # Try with .png extension
            alt_path = img_path.with_suffix(".png")
            if alt_path.exists():
                img_bgr = cv2.imread(str(alt_path))
            if img_bgr is None:
                continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]

        # Get the prediction mask from the result
        pred_mask = result["mask"]

        # Resize mask to match image if needed
        if pred_mask.shape != (orig_h, orig_w):
            pred_mask_resized = cv2.resize(
                    pred_mask.astype(np.uint8),
                    (orig_w, orig_h),
                    interpolation = cv2.INTER_NEAREST
            )
        else:
            pred_mask_resized = pred_mask

        # Update the result with properly sized mask
        result["mask"] = pred_mask_resized

        # Create proper color mask and overlay
        mask_rgb = np.zeros((orig_h, orig_w, 3), dtype = np.uint8)
        mask_rgb[pred_mask_resized == 1] = (0, 255, 0)  # individual trees
        mask_rgb[pred_mask_resized == 2] = (255, 255, 0)  # groups

        # Create overlay with properly sized components
        overlay = cv2.addWeighted(img_rgb, 0.6, mask_rgb, 0.4, 0)
        result["overlay"] = overlay
        result["image"] = img_rgb

    return submission_results


# %%
t("Visualizing predictions from submission")

submission_path = config.paths.models / "FINAL_SUBMISSION.json"
p("Submission file", submission_path)

try:
    with open(submission_path, "r", encoding = "utf8") as f:
        data = json.load(f)

    images = data.get("images", [])
    if not images:
        p("No images found in submission", color1 = c.ORANGE)
    else:
        # Convert submission data to results format for processing
        submission_results = []

        for entry in images[:5]:  # Process first 5 images
            fname = entry["file_name"]

            # Get image dimensions
            h = entry.get("height", 512)  # fallback values
            w = entry.get("width", 512)

            # Reconstruct mask from annotations
            mask = np.zeros((h, w), dtype = np.uint8)

            for ann in entry.get("annotations", []):
                seg = ann.get("segmentation", [])
                if not seg:
                    continue

                pts = np.array(seg, dtype = np.float32).reshape(-1, 2)
                pts = np.round(pts).astype(np.int32)
                cls = ann.get("class", "individual_tree")

                if cls == "individual_tree":
                    class_id = 1
                elif cls == "group_of_trees":
                    class_id = 2
                else:
                    class_id = 1

                cv2.fillPoly(mask, [pts], color = class_id)

            submission_results.append(
                    {
                        "name":    fname,
                        "mask":    mask,
                        "image":   None,  # Will be loaded in fix function
                        "overlay": None  # Will be created in fix function
                    }
            )

        # Fix sizing issues
        fixed_results = fix_submission_visualization(submission_results, config.paths.eval_images)

        # Visualize the fixed results
        for result in fixed_results:
            if result["image"] is not None:
                p("Eval image from submission", result["name"])

                show_side_by_side(
                        result["image"],
                        result["mask"],
                        result["overlay"],
                        titles = ("Original", "Pred mask", "Overlay"),
                        cmaps = [None, "gray", None],
                        maxcolumns = 3,
                )

except Exception as e_viz:
    p("Failed to visualize submission predictions", str(e_viz), color1 = c.ORANGE)
    import traceback


    traceback.print_exc()

# %%

# %%
from src.prediction.validation import analyze_validation_metrics

# Load best model info first
best_model_info_path = config.paths.models / "BEST_MODEL.txt"
best_model_name = "simple_cnn"  # default fallback

if best_model_info_path.exists():
    with open(best_model_info_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "model:" in line:
                best_model_name = line.split(':')[1].strip()
                break

if trainer is not None:
    checkpoint_path = trainer.paths["checkpoint"]
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location = "cpu")

        p(f"Analyzing validation metrics for", best_model_name, color1 = c.ORANGE)

        analyze_validation_metrics(
                val_loss = ckpt.get("val_loss", 0),
                iou = ckpt.get("val_iou", ckpt.get("iou", 0)),
                accuracy = ckpt.get("val_accuracy", ckpt.get("accuracy", 0)),
                precision = ckpt.get("val_precision", ckpt.get("precision", 0)),
                recall = ckpt.get("val_recall", ckpt.get("recall", 0)),
                f1_score = ckpt.get("val_f1_score", ckpt.get("f1_score", 0)),
                individual_tree_iou = ckpt.get("val_iou_individual", 0),
                group_tree_iou = ckpt.get("val_iou_group", 0),
                dice = ckpt.get("val_dice", ckpt.get("dice", 0)),
                model_name = best_model_name
        )
