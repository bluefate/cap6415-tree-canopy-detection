# %% [markdown]
# # Notebook: 10 Master Execution Plan
# ### Purpose: Complete pipeline from data preparation to final submission

# %%

import copy
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd

from exploration.class_explorer import color_mask, load_image, mask_all
from src.exploration.visualize import show_side_by_side


#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  #enable for debugging ONLY

sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../src"))

import torch
from torch.utils.data import DataLoader
from src.data.annotations import load_json_annotations
from src.data.augmentations import get_train_augmentations, get_val_augmentations
from src.data.loaders import ImageMaskDataset
from src.training.engine import run_training
from src.utils.config import Config
from src.utils.helpers import c, init_notebook, p, t
from src.data.enhance_masks import EnhancedImageMaskDataset
from src.utils.versioning import VersionManager


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

# Analyze class distribution
individual_count = sum(1 for e in entries if any(item.cls == "individual_tree" for item in e.items))
group_count = sum(1 for e in entries if any(item.cls == "group_of_trees" for item in e.items))

p("Images with individual trees", individual_count)
p("Images with tree groups", group_count)



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
def get_available_filters():
    """
    Get list of all available filter names from the system.
    """
    try:
        available = EnhancedImageMaskDataset.get_available_filters()
        return available
    except Exception as e:
        p("Warning", f"Could not load filters dynamically: {e}", color1 = c.ORANGE, color2 = c.ORANGE)
        # Fallback to known filters
        return [
            'laplacian', 'sobel', 'clahe',
            'gaussian_3x3', 'gaussian_5x5', 'gaussian_7x7',
            'sobel_x', 'sobel_y', 'laplacian_3x3',
            'sharpen_basic', 'high_pass_3x3', 'edge_enhance',
            'gaussian_3x3_sigma1', 'gaussian_5x5_sigma1', 'gaussian_7x7_sigma1',
        ]


def validate_filter_set( filter_names, available_filters ):
    """
    Validate a list of filter names against available filters.
    """
    invalid = []
    suggestions = { }

    for fname in filter_names:
        if fname.lower() not in [f.lower() for f in available_filters]:
            invalid.append(fname)
            # Try to find suggestion
            for avail in available_filters:
                if fname.lower() in avail.lower() or avail.lower() in fname.lower():
                    suggestions[fname] = avail
                    break

    return len(invalid) == 0, invalid, suggestions


# Get available filters
t("Validating Available Filters")
AVAILABLE_FILTERS = get_available_filters()
p("Available filters count", len(AVAILABLE_FILTERS))
p("Sample filters", AVAILABLE_FILTERS, show = 15)

# %% [markdown]
# #### Step 3: Enhanced Dataset Creation
#
# - Create training dataset with filter-enhanced inputs
# - Apply filters as additional channels.
#
#

# %%
t("Testing Enhanced Dataset")

val_transform = get_val_augmentations(config.train.image_size)
sample_entries = entries[:5]

# Testing each mode
for mode in ['rgb', 'filtered', 'concat']:
    try:
        dataset = EnhancedImageMaskDataset(
                sample_entries,
                config.paths.train_images,
                mode = mode,
                transform = val_transform
        )

        img_t, mask_t = dataset[0]
        p(f"Mode: {mode}", f"Image shape: {img_t.shape}, Mask shape: {mask_t.shape}")
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

# %%
def train_experiment( model_name, input_mode, filter_names = None, num_epochs = None ):
    """
    Run one training experiment and return metrics.
    Supports RGB (3ch), filtered (3ch), and concat (6ch) modes.
    """
    # Use config epochs if not specified
    if num_epochs is None:
        num_epochs = config.train.epochs

    t(f"Experiment: {model_name} with {input_mode} input")

    # Prepare data
    train_entries = entries[:int(0.8 * len(entries))]
    val_entries = entries[int(0.8 * len(entries)):]

    train_tf = get_train_augmentations(config.train.image_size)
    val_tf = get_val_augmentations(config.train.image_size)

    train_ds = EnhancedImageMaskDataset(
            train_entries,
            config.paths.train_images,
            mode = input_mode,
            filter_names = filter_names,
            transform = train_tf
    )

    val_ds = EnhancedImageMaskDataset(
            val_entries,
            config.paths.train_images,
            mode = input_mode,
            filter_names = filter_names,
            transform = val_tf
    )

    # Determine input channels
    sample_img, _ = train_ds[0]
    in_channels = sample_img.shape[0]

    p("Input channels", in_channels)
    p("Mode", input_mode)
    if filter_names:
        p("Filters", filter_names)

    # Need to modify model for non-standard input
    # Skip 6-channel experiments if model doesn't support it
    if in_channels == 6 and model_name in ['simple_cnn', 'unet']:
        p("Warning", f"Skipping 6-channel experiment for {model_name}", color1 = c.ORANGE)
        p("", "  (requires custom model architecture)", color1 = c.ORANGE)
        # For now, skip 6-channel experiments
        return None

    # Create data loaders
    train_loader = DataLoader(
            train_ds,
            batch_size = config.train.batch_size,
            shuffle = True,
            num_workers = config.train.num_workers,
            pin_memory = True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
            val_ds,
            batch_size = config.train.batch_size,
            shuffle = False,
            num_workers = config.train.num_workers,
            pin_memory = True if torch.cuda.is_available() else False
    )

    exp_config = copy.deepcopy(config)
    exp_config.train.epochs = num_epochs

    # Adding experiment metadata
    exp_config.extra['experiment'] = {
        'model_name':     model_name,
        'input_mode':     input_mode,
        'filter_names':   filter_names,
        'input_channels': in_channels,
    }

    # model-specific directory
    version_root = config.paths.models / model_name / input_mode
    if filter_names:
        filter_str = "_".join(filter_names)
        version_root = version_root / filter_str[:30]  # Limit path length

    version_root.mkdir(parents = True, exist_ok = True)

    p("Version root", version_root)
    # Train (VersionManager handles versioning inside run_training -> Trainer)
    trainer = run_training(
            config = exp_config,
            train_loader = train_loader,
            val_loader = val_loader,
            #version_root = config.paths.models,
            version_root = version_root,  # Model-specific path
            model_name = model_name
    )

    return trainer


t("Experiments configured")



# %% [markdown]
# ##### Experiment setup

# %%
# Define filter combinations to test
filter_sets = {
    'classic':        ['laplacian', 'sobel', 'clahe'],
    'gaussian':       ['gaussian_3x3', 'gaussian_5x5', 'gaussian_7x7'],
    'kernel_sharpen': ['sharpen_basic', 'high_pass_3x3', 'edge_enhance'],
    'kernel_edge':    ['sobel_x', 'sobel_y', 'laplacian_3x3'],
    'combined':       ['laplacian', 'gaussian_5x5', 'clahe'],
}

# Validate all filter sets before proceeding
t("Validating Filter Sets")
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
p("filter_sets", filter_sets)

# %%

# TODO  - EXPLORE other methods (subtract filters maybe)
# Define experiments
experiments = []

# Baseline: RGB only
for model in ['simple_cnn', 'unet']:
    experiments.append((model, 'rgb', None))

# Filtered mode: 3 channels from filters
for model in ['simple_cnn', 'unet']:
    for set_name, filters in filter_sets.items():
        experiments.append((model, 'filtered', filters))

# Concat mode: 6 channels (RGB + 3 filters)
# Note: This requires model architecture modification for 6-channel input
# for model in ['simple_cnn', 'unet']:
#     experiments.append((model, 'concat', filter_sets['combined']))

p("experiments", experiments, show = 15)

# %%

# # TODO  - TESTING - remove
# # Minimal experiment set for initial testing
# t("Testing")
# filter_sets = dict(list(filter_sets.items())[:1])
# p("filter_sets", filter_sets)
#
# experiments = [
#     ('unet', 'filtered', ['sharpen_basic', 'high_pass_3x3', 'edge_enhance']),
#     ('simple_cnn', 'rgb', None),
# ]
# p("experiments", experiments, show = 15)


# %%
def estimate_runtime( experiments, filter_sets ):
    """
    Estimate training runtime using:
      - experiments list
      - filter_sets dict
    Adjusts time for rgb vs filtered vs concat.
    """

    t("Runtime Estimate")

    train_size = int(0.8 * len(entries))
    batch_size = config.train.batch_size
    batches_per_epoch = max(1, train_size // batch_size)
    epochs = config.train.epochs

    # Baseline timing assumption (seconds per batch)
    base_seconds = 1.0

    total_seconds = 0.0

    p("Experiments", len(experiments))
    p("Filter sets", len(filter_sets))
    p("Epochs per experiment", epochs)

    # Mode timing multipliers
    mode_multiplier = {
        "rgb":      1.0,
        "filtered": 1.5,
        "concat":   2.0
    }

    for model_name, mode, filters in experiments:

        # base seconds per batch
        sec_per_batch = base_seconds * mode_multiplier.get(mode, 1.0)

        exp_seconds = epochs * batches_per_epoch * sec_per_batch
        total_seconds += exp_seconds

        p(f"{model_name} | {mode}", f"{exp_seconds / 60:.2f} min")

    total_minutes = total_seconds / 60
    total_hours = total_minutes / 60
    p()
    t("Totals")
    p("Training samples", train_size)
    p("Batches per epoch", batches_per_epoch)
    p("Estimated total time", f"~{total_hours:.2f} hours", color1 = c.RED, color2 = c.RED)

    if total_hours > 4:
        p("⚠ Long run", "Reduce epochs or filter sets", color1 = c.ORANGE)


estimate_runtime(experiments, filter_sets)


# %%
def summarize_experiments( experiments, filter_sets ):
    """
    Summarize experiment count based on:
    -  experiments: list of (model, mode, filters)
    -  filter_sets: dict {name: [filters]} used externally
    """

    t("TOTAL EXPERIMENTS")

    # Extract models from experiments
    models = sorted({ m for m, _, _ in experiments })

    # Count experiments by mode
    num_rgb = sum(1 for m, mode, f in experiments if mode == "rgb")
    num_filtered = sum(1 for m, mode, f in experiments if mode == "filtered")
    num_concat = sum(1 for m, mode, f in experiments if mode == "concat")

    # Total
    total = len(experiments)

    # Print counts
    p("Models", len(models), color1 = c.BLUE)
    p("Filter sets", len(filter_sets), color1 = c.SALMON)
    p("RGB experiments", num_rgb)
    p("Filtered experiments", num_filtered)
    p("Concat experiments", num_concat)
    p("TOTAL EXPERIMENTS TO RUN", total, color1 = c.RED, color2 = c.RED)

    # Print experiment combinations
    p("\n", "Experiment combinations", color1 = c.BLACK)
    for model, mode, filters in experiments:
        p("", f"{model} | {mode} | {filters}")


summarize_experiments(experiments, filter_sets)


# %%

# Initialize best model tracker
best_model_tracker = {
    "best_val_loss":    float('inf'),
    "best_iou":         0.0,
    "best_experiment":  None,
    "best_model_path":  None,
    "best_version_dir": None,
}

# Running experiments
results = { }

num_epochs = config.train.epochs
p("Using epochs from config", num_epochs)

for i, (model_name, mode, filters) in enumerate(experiments, 1):
    p("\n\n")
    t(f"Experiment {i}/{len(experiments)}")
    p("Model", model_name)
    p("Mode", mode)
    p("Filters", filters)

    # experiment key
    key = f"{model_name}_{mode}"
    if filters:
        filter_key = "_".join(filters)[:20]
        key = f"{key}_{filter_key}"

    try:
        trainer = train_experiment(
                model_name,
                mode,
                filters,
                num_epochs = config.train.epochs
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

                    if val_loss < best_model_tracker["best_val_loss"]:
                        best_model_tracker["best_val_loss"] = val_loss
                        best_model_tracker["best_experiment"] = key
                        best_model_tracker["best_model_path"] = version_dir / "best_model.pth"
                        best_model_tracker["best_version_dir"] = version_dir

                        p("🏆 NEW BEST MODEL!", key, color1 = c.GREEN, bold = True)
                        p("  Val Loss", f"{val_loss:.6f}", color1 = c.GREEN)
                except Exception as e:
                    p("Warning", f"Could not load checkpoint: {e}", color1 = c.ORANGE)

            p("EXPERIMENT COMPLETED", key, color1 = c.GREEN, color2 = c.GREEN)
        else:
            p("EXPERIMENT SKIPPED", key, color1 = c.ORANGE, color2 = c.ORANGE)

    except Exception as e:
        p("EXPERIMENT FAILED", key, color1 = c.RED, color2 = c.RED)
        p("[Error]", str(e), color1 = c.RED)
        results[key] = { "status": "ERROR", "error": str(e) }
        continue

# Summary
t("Experiment Results Summary")
successful = sum(1 for v in results.values() if not isinstance(v, dict) or v.get("status") != "ERROR")
failed = sum(1 for v in results.values() if isinstance(v, dict) and "status" in v)
skipped = sum(1 for v in results.values() if v is None)

p("Total experiments", len(experiments))
p("Successful", successful - skipped, color1 = c.GREEN)
p("Skipped", skipped, color1 = c.ORANGE)
p("Failed", failed, color1 = c.RED if failed > 0 else c.GREEN)



# %% [markdown]
# ##### Model Comparison & Best Model Selection

# %%
t("Collecting Experiment Results")

experiment_results = []

for model_name in ['simple_cnn', 'unet']:
    for input_mode in ['rgb', 'filtered']:

        version_root = config.paths.models / model_name / input_mode

        if not version_root.exists():
            continue

        versions = sorted(version_root.glob("v*"))

        for version_dir in versions:
            checkpoint_path = version_dir / "checkpoint.pth"

            if not checkpoint_path.exists():
                continue

            try:
                # Load checkpoint to get metrics
                ckpt = torch.load(checkpoint_path, map_location = "cpu")

                best_val_loss = ckpt.get("best_val_loss", float('inf'))
                epoch = ckpt.get("epoch", 0)

                # Load config snapshot to get filter info
                config_snapshot_path = version_dir / "config_snapshot.json"
                filter_names = None

                if config_snapshot_path.exists():
                    with open(config_snapshot_path, 'r') as f:
                        snapshot = json.load(f)
                        exp_config = snapshot.get('config', { }).get('extra', { }).get('experiment', { })
                        filter_names = exp_config.get('filter_names')

                experiment_results.append(
                        {
                            'model':           model_name,
                            'mode':            input_mode,
                            'version':         version_dir.name,
                            'filters':         str(filter_names) if filter_names else 'none',
                            'best_val_loss':   best_val_loss,
                            'final_epoch':     epoch,
                            'checkpoint_path': str(checkpoint_path),
                            'version_path':    str(version_dir),
                        }
                )

            except Exception as e:
                p("Warning", f"Could not load {checkpoint_path}: {e}", color1 = c.ORANGE)
                continue

# Convert to DataFrame for easy analysis
df_results = pd.DataFrame(experiment_results)

if len(df_results) == 0:
    p("No experiment results found", color1 = c.RED)
else:
    p("Total experiments found", len(df_results))

    # Sort by validation loss (lower is better)
    df_results = df_results.sort_values('best_val_loss')

    p("")
    p("Top 5 Experiments by Validation Loss", color1 = c.CYAN, bold = True)
    print(df_results[['model', 'mode', 'filters', 'best_val_loss', 'final_epoch']].head(10).to_string(index = False))

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
    p("Version", best_row['version'], color1 = c.GREEN)
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
    plt.xticks(rotation = 0)

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
            model_name = model_name
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

    p(val_tf, color1 = c.RED)
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
        p("Sample entry keys", list(first_img.keys()))
        if first_img.get('annotations'):
            first_ann = first_img['annotations'][0]
            p("Sample annotation keys", list(first_ann.keys()))

    return output_path


# %%

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

            # Generate submission
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

# Get all possible model names from MODEL_BUILDERS
all_model_names = ['simple_cnn', 'unet', 'smp_unet', 'smp_fpn', 'smp_linknet',
                   'smp_deeplabv3', 'smp_deeplabv3plus', 'segformer']

# All possible input modes
all_input_modes = ['rgb', 'filtered', 'concat']

for model_name in all_model_names:
    for input_mode in all_input_modes:

        base_path = config.paths.models / model_name / input_mode

        if not base_path.exists():
            continue

        # Check for direct versions (no filter subdirectory)
        # e.g., models/simple_cnn/rgb/v001/
        vm = VersionManager(base_path)
        latest_version = vm.find_latest()

        if latest_version is not None:
            model_path = latest_version / "best_model.pth"
            if model_path.exists():
                output_file = latest_version / "submission.json"

                t(f"eval_images path: {config.paths.eval_images}")

                p("\nGenerating submission", f"{model_name}/{input_mode}", color1 = c.MAGENTA)

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
                    p("Saved submission", output_file)

                except Exception as e:
                    p("Failed", str(e), color1 = c.RED)

        # Check for filter subdirectories (for 'filtered' mode)
        # e.g., models/simple_cnn/filtered/laplacian_sobel_clahe/v001/
        if input_mode == 'filtered':
            for filter_subdir in base_path.iterdir():
                if not filter_subdir.is_dir():
                    continue
                if filter_subdir.name.startswith('v'):  # Skip version dirs
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

                p("\nGenerating submission", f"{model_name}/{input_mode}/{filter_name}", color1 = c.MAGENTA)

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
                    p("Saved submission", output_file)

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

#Append overall best
if best_overall:
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



# %%



image_dir = config.paths.train_images
t(image_dir)

# sample three entries
sample_entries = random.sample(entries, 5)

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
t("Visualizing sample predictions")

image_dir = config.paths.eval_images
t(image_dir)

try:
    # Use the best model that was selected earlier
    predictor = Predictor(
            model_path = model_weights,
            model_name = best_model_name,
            image_size = config.train.image_size,
    )

    val_tf = get_val_augmentations(config.train.image_size)

    # Run prediction on a small subset of eval images
    results = predictor.run_on_folder(
            image_dir,
            transform = val_tf,
            num_samples = 5,
    )

    for r in results:
        name = r["name"]
        img = r["image"]
        pred_mask = r["mask"]
        overlay = r["overlay"]

        p("Eval image", name)

        show_side_by_side(
                img, pred_mask, overlay,
                titles = ("Original", "Pred mask", "Overlay"),
                maxcolumns = 3,
        )


except Exception as e_viz:
    p("Failed to visualize predictions", str(e_viz), color1 = c.ORANGE)

# %%
import numpy as np


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
        num_samples = min(5, len(images))
        sample_entries = random.sample(images, num_samples)

        for entry in sample_entries:
            fname = entry["file_name"]
            img_path = config.paths.eval_images / fname

            # Optional: fallback to .png if only converted files exist
            if not img_path.exists():
                alt_png = img_path.with_suffix(".png")
                if alt_png.exists():
                    img_path = alt_png

            p("Eval image from submission", fname)

            img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                p("Could not read image", str(img_path), color1 = c.ORANGE)
                continue

            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            h = entry.get("height", img.shape[0])
            w = entry.get("width", img.shape[1])
            mask = np.zeros((h, w), dtype = np.uint8)

            # Rebuild multi class mask from polygons
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

            # Simple color mask for display
            mask_rgb = np.zeros_like(img)
            mask_rgb[mask == 1] = (0, 255, 0)  # individual trees
            mask_rgb[mask == 2] = (255, 255, 0)  # groups

            overlay = cv2.addWeighted(img, 0.6, mask_rgb, 0.4, 0)

            show_side_by_side(
                    img, mask, mask_rgb, overlay,
                    titles = ("Original", "Pred mask (ids)", "Color mask", "Overlay"),
                    maxcolumns = 4,
            )

except Exception as e_viz:
    p("Failed to visualize submission predictions", str(e_viz), color1 = c.ORANGE)

