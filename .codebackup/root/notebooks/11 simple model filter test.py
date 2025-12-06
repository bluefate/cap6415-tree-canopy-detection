# %% [markdown]
# # Simple Model + Filter Experimentation
# ### Goal: Find the best model-filter combination for tree segmentation
#
# This notebook systematically tests:
# - Multiple models (SimpleCNN, UNet, YOLOv8)
# - Multiple filter combinations
# - RGB baseline vs filtered inputs
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

# %%
import os
import sys

import pandas as pd
from torch.utils.data import DataLoader


sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../src"))

from src.data.annotations import load_json_annotations
from src.data.augmentations import get_train_augmentations, get_val_augmentations
from src.data.enhance_masks import EnhancedImageMaskDataset
from src.training.engine import run_training
from src.training.running import get_available_filters, get_version_config, validate_filter_set
from src.utils.config import Config
from src.utils.helpers import estimate_runtime_by_epcoh, init_notebook, p, t, c


# %% [markdown]
# ### 1. Setup and Configuration
#

# %%
config = Config.load(root = root)
init_notebook(config.train.seed)

# Load data
entries = load_json_annotations(config.paths.annotations)
p("Total images loaded", len(entries), color1 = c.GREEN)

# Split data (80/20)
split_idx = int(0.8 * len(entries))
train_entries = entries[:split_idx]
val_entries = entries[split_idx:]

p("Training samples", len(train_entries))
p("Validation samples", len(val_entries))


# %% [markdown]
# ### 2. Available Filters
# #
# Let's see what filters we have available from your kernel bank and algorithmic filters.
#

# %%
t("Available Filters")
AVAILABLE_FILTERS = get_available_filters()
p("Total filters available", len(AVAILABLE_FILTERS))
p("  Filter categories:")
p("  Edge detection: laplacian, sobel, sobel_x, sobel_y, canny")
p("  Smoothing: gaussian_3x3, gaussian_5x5, gaussian_7x7, bilateral")
p("  Enhancement: clahe, histogram_eq")
p("  Sharpening: sharpen_basic, edge_enhance, high_pass")


# %% [markdown]
# ### 3. Define Experiment Matrix
# #
# We'll test:
# - **Models**: SimpleCNN (fast), UNet (standard), YOLOv8s (advanced)
# - **Input modes**: RGB (baseline), Filtered (3 filter channels)
# - **Filter sets**: Edge detection, smoothing, enhancement, mixed
#

# %%
t("Experiment Configuration")

# Define filter combinations to test
filter_sets = {
    "edge_detection": ["laplacian", "sobel", "sobel_x"],
    "edge_kernel":    ["sobel_x", "sobel_y", "laplacian_3x3"],
    "smoothing":      ["gaussian_3x3", "gaussian_5x5", "gaussian_7x7"],
    "enhancement":    ["clahe", "bilateral", "median_3x3"],
    "mixed_best":     ["laplacian", "gaussian_5x5", "clahe"],
    "sharpening":     ["sharpen_basic", "edge_enhance", "high_pass_3x3"],
}

# Validate filter sets
p("\nValidating filter sets...")
all_valid = True
for set_name, filters in filter_sets.items():
    is_valid, invalid, suggestions = validate_filter_set(filters, AVAILABLE_FILTERS)
    status = "✓ VALID" if is_valid else f"✗ INVALID: {invalid}"
    color = c.GREEN if is_valid else c.RED
    p(f"  {set_name}", status, color1 = color)
    if not is_valid:
        all_valid = False
        if suggestions:
            p(f"    Suggestions", suggestions, color1 = c.ORANGE)

if not all_valid:
    raise ValueError(
            "Some filter sets contain invalid filters. Please fix before continuing."
    )

# Define models to test
models_to_test = ["simple_cnn", "unet", "yolov8s"]

# IMPORTANT NOTE: The current run_training() function hardcodes 3-channel input.
# This means we can only test RGB mode effectively with the existing infrastructure.
# Filtered mode (3 filter channels) will be skipped for now unless filters are applied
# as preprocessing that still results in 3-channel outputs.

# Generate all experiments: (model, mode, filters)
experiments = []
for model_name in models_to_test:
    # RGB baseline (no filters) - this will work
    experiments.append((model_name, "rgb", None))

    # For filtered mode, we'll test but may need to skip if channels != 3
    # The dataloader will create 3-channel outputs from the filters
    for set_name, filters in filter_sets.items():
        experiments.append((model_name, "filtered", filters))

p(f"\nTotal experiments", len(experiments), color1 = c.ORANGE)
p("Sample experiments:")
for i, exp in enumerate(experiments[:5]):
    model, mode, filters = exp
    filter_str = filters[:2] if filters else "None"
    p(f"  {i + 1}", f"{model} | {mode} | {filter_str}...", color1 = c.CYAN)


# %% [markdown]
# ### 4. Quick Runtime Estimate
#
#

# %%
estimated_hours = estimate_runtime_by_epcoh(experiments, epochs_per_exp = config.train.epochs)


# %% [markdown]
# ### 5. Run Experiments
# #
# This will train each model-filter combination and track performance.
#

# %%
t("Starting Experiments")

# Results tracker
results = []
best_model_tracker = {
    "best_val_loss": float("inf"),
    "best_iou":      0.0,
    "experiment":    None,
    "model_path":    None,
}

# Prepare transforms
train_tf = get_train_augmentations(config.train.image_size, mode = mode)
val_tf = get_val_augmentations(config.train.image_size, mode = mode)

for i, (model_name, mode, filters) in enumerate(experiments, 1):
    p("\n" + "=" * 80)
    t(f"Experiment {i}/{len(experiments)}")
    p("Model", model_name, color1 = c.CYAN)
    p("Mode", mode, color1 = c.CYAN)
    p("Filters", filters if filters else "None", color1 = c.CYAN)

    try:
        # Create a unique config for this experiment to prevent checkpoint reuse
        import copy
        import time


        key, version_root, exp_config, best_model_path, checkpoint_path_check, best_model_exists = get_version_config(
                config, filters, "11", model_name, mode, in_channels, best_model_tracker, i, experiments
        )
        if best_model_exists:
            continue

        # Create datasets
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

        # Check input channels
        sample_img, _ = train_ds[0]
        in_channels = sample_img.shape[0]
        p("Input channels", in_channels)

        # NOTE: run_training always builds models with 3 input channels
        # So filtered mode (3 filter channels) works, but concat mode (6 channels) won't
        if mode == "concat":
            p(
                    "⚠ Warning",
                    "Skipping concat mode - not supported by run_training",
                    color1 = c.ORANGE,
            )
            continue

        # Create dataloaders
        train_loader = DataLoader(
                train_ds, batch_size = config.train.batch_size, shuffle = True, num_workers = 2
        )
        val_loader = DataLoader(
                val_ds, batch_size = config.train.batch_size, shuffle = False, num_workers = 2
        )

        # Train model (run_training builds the model internally)
        p(f"Training {model_name}...", color1 = c.GREEN)
        trainer = run_training(
                exp_config,
                train_loader,
                val_loader,
                config.paths.models,
                version_root = version_root,
                model_name = model_name,
        )

        # Extract results
        version_info = trainer.version_manager.version_info
        history = trainer.history

        result = {
            "experiment_id":    i,
            "model":            model_name,
            "mode":             mode,
            "filters":          str(filters) if filters else "none",
            "best_val_loss":    (
                min(history["val_loss"]) if history["val_loss"] else float("inf")
            ),
            "final_train_loss": (
                history["train_loss"][-1] if history["train_loss"] else None
            ),
            "best_epoch":       (
                history["val_loss"].index(min(history["val_loss"])) + 1
                if history["val_loss"]
                else None
            ),
            "total_epochs":     len(history["train_loss"]),
            "version_path":     str(trainer.version_dir),
        }

        results.append(result)

        # Update best model tracker
        if result["best_val_loss"] < best_model_tracker["best_val_loss"]:
            best_model_tracker.update(
                    {
                        "best_val_loss": result["best_val_loss"],
                        "experiment":    f"{model_name}_{mode}_{filters}",
                        "model_path":    trainer.version_dir / "best_model.pth",
                    }
            )

        p(
                "✓ Experiment complete",
                f"Val Loss: {result['best_val_loss']:.4f}",
                color1 = c.GREEN,
        )

    except Exception as e:
        p("✗ Experiment failed", str(e), color1 = c.RED)
        results.append(
                {
                    "experiment_id": i,
                    "model":         model_name,
                    "mode":          mode,
                    "filters":       str(filters) if filters else "none",
                    "error":         str(e),
                }
        )
        continue

p("\n" + "=" * 80)
t("All Experiments Complete!")


# %% [markdown]
# ### 6. Analyze Results
#

# %%
t("Results Analysis")

# Create DataFrame
df_results = pd.DataFrame(results)

# Filter out failed experiments
df_success = df_results[~df_results["best_val_loss"].isna()].copy()
df_failed = df_results[df_results["best_val_loss"].isna()]

p("Successful experiments", len(df_success), color1 = c.GREEN)
p("Failed experiments", len(df_failed), color1 = c.RED)

if len(df_success) > 0:
    # Sort by performance
    df_success = df_success.sort_values("best_val_loss")

    p("\n" + "=" * 80)
    p("TOP 5 MODELS", "", color1 = c.GREEN, bold = True)
    p("=" * 80)

    for idx, row in df_success.head(5).iterrows():
        p(f"\n{idx + 1}. {row['model'].upper()} ({row['mode']})")
        p("  Filters", row["filters"])
        p("  Val Loss", f"{row['best_val_loss']:.6f}")
        p("  Best Epoch", f"{row['best_epoch']}/{row['total_epochs']}")
        p("  Path", row["version_path"])

    p("\n" + "=" * 80)
    p("BEST MODEL OVERALL", "", color1 = c.CYAN, bold = True)
    p("=" * 80)
    best = df_success.iloc[0]
    p("Model", best["model"])
    p("Mode", best["mode"])
    p("Filters", best["filters"])
    p("Val Loss", f"{best['best_val_loss']:.6f}")
    p("Path", best["version_path"])


# %% [markdown]
# ### 7. Performance Comparison
#

# %%
if len(df_success) > 0:
    t("Performance by Model Type")

    model_comparison = (
        df_success.groupby("model")
        .agg({ "best_val_loss": ["mean", "min", "std"] })
        .round(6)
    )

    p(model_comparison)

    t("Performance by Input Mode")

    mode_comparison = (
        df_success.groupby("mode")
        .agg({ "best_val_loss": ["mean", "min", "std"] })
        .round(6)
    )

    p(mode_comparison)

    t("Best Filter Sets")

    # Only compare filtered experiments
    df_filtered = df_success[df_success["mode"] == "filtered"].copy()
    if len(df_filtered) > 0:
        filter_comparison = (
            df_filtered.groupby("filters")
            .agg({ "best_val_loss": ["mean", "min"] })
            .sort_values(("best_val_loss", "mean"))
            .round(6)
        )

        p(filter_comparison.head(10))


# %% [markdown]
# ### 8. Export Results
#

# %%
t("Exporting Results")

if len(results) > 0:
    # Save all results
    results_path = config.paths.models / "model_filter_experiments.csv"
    df_results.to_csv(results_path, index = False)
    p("✓ Saved results", str(results_path), color1 = c.GREEN)

    # Save summary
    if len(df_success) > 0:
        summary_path = config.paths.models / "EXPERIMENT_SUMMARY.txt"
        with open(summary_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL + FILTER EXPERIMENT SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Total experiments: {len(results)}\n")
            f.write(f"Successful: {len(df_success)}\n")
            f.write(f"Failed: {len(df_failed)}\n\n")

            f.write("BEST MODEL:\n")
            f.write("-" * 80 + "\n")
            best = df_success.iloc[0]
            f.write(f"Model: {best['model']}\n")
            f.write(f"Mode: {best['mode']}\n")
            f.write(f"Filters: {best['filters']}\n")
            f.write(f"Val Loss: {best['best_val_loss']:.6f}\n")
            f.write(f"Path: {best['version_path']}\n\n")

            f.write("TOP 5 MODELS:\n")
            f.write("-" * 80 + "\n")
            for i, (idx, row) in enumerate(df_success.head(5).iterrows(), 1):
                f.write(f"\n{i}. {row['model']} ({row['mode']})\n")
                f.write(f"   Filters: {row['filters']}\n")
                f.write(f"   Val Loss: {row['best_val_loss']:.6f}\n")
                f.write(f"   Epoch: {row['best_epoch']}/{row['total_epochs']}\n")

        p("✓ Saved summary", str(summary_path), color1 = c.GREEN)


# %% [markdown]
# ### 9. Visualization (Optional)
# #
# Create comparison plots if you want visual analysis.
#

# %%
import matplotlib.pyplot as plt


if len(df_success) > 5:
    t("Creating Visualizations")

    fig, axes = plt.subplots(2, 2, figsize = (15, 10))

    # Plot 1: Val loss by model
    ax1 = axes[0, 0]
    for model in df_success["model"].unique():
        model_data = df_success[df_success["model"] == model]
        ax1.scatter(
                range(len(model_data)),
                model_data["best_val_loss"],
                label = model,
                s = 100,
                alpha = 0.6,
        )
    ax1.set_title("Validation Loss by Model")
    ax1.set_xlabel("Experiment Index")
    ax1.set_ylabel("Best Val Loss")
    ax1.legend()
    ax1.grid(True, alpha = 0.3)

    # Plot 2: RGB vs Filtered comparison
    ax2 = axes[0, 1]
    mode_stats = df_success.groupby("mode")["best_val_loss"].agg(["mean", "min", "max"])
    mode_stats.plot(kind = "bar", ax = ax2)
    ax2.set_title("RGB vs Filtered Mode Performance")
    ax2.set_ylabel("Validation Loss")
    ax2.set_xlabel("Input Mode")
    ax2.legend(["Mean", "Min", "Max"])
    ax2.grid(True, alpha = 0.3)

    # Plot 3: Training convergence (top 5)
    ax3 = axes[1, 0]
    for idx, row in df_success.head(5).iterrows():
        label = f"{row['model']}-{row['mode']}"
        ax3.scatter(row["best_epoch"], row["best_val_loss"], s = 150, label = label)
    ax3.set_title("Convergence Speed (Top 5)")
    ax3.set_xlabel("Best Epoch")
    ax3.set_ylabel("Best Val Loss")
    ax3.legend(fontsize = 8)
    ax3.grid(True, alpha = 0.3)

    # Plot 4: Model comparison boxplot
    ax4 = axes[1, 1]
    df_success.boxplot(column = "best_val_loss", by = "model", ax = ax4)
    ax4.set_title("Loss Distribution by Model")
    ax4.set_xlabel("Model")
    ax4.set_ylabel("Best Val Loss")
    plt.suptitle("")  # Remove default title

    plt.tight_layout()

    plot_path = config.paths.models / "experiment_comparison.png"
    plt.savefig(plot_path, dpi = 150, bbox_inches = "tight")
    p("✓ Saved plot", str(plot_path), color1 = c.GREEN)

    plt.show()


# %% [markdown]
# ### 10. Recommendations
# #
# Based on the results, here's what to do next:
#

# %%
if len(df_success) > 0:
    t("Recommendations")

    best = df_success.iloc[0]

    p("=" * 80)
    p("NEXT STEPS")
    p("=" * 80)
    p(f"1. Use the best model: {best['model']} with {best['mode']} mode")
    p(f"   Filters: {best['filters']}")
    p(f"   Achieved Val Loss: {best['best_val_loss']:.6f}")
    p(f"2. Load the model from: {best['version_path']}")
    p(f"3. Run predictions using notebook 05_prediction.py")
    p(f"4. Generate submission file using notebook 07_submission.py")

    # Additional insights
    if best["mode"] == "filtered":
        p(f"INSIGHT: Filter enhancement improved performance!")
        p(f"   Consider using these filters: {best['filters']}")
    else:
        p(f"INSIGHT: RGB baseline performed best.")
        p(f"   Filters didn't improve performance in this case.")

    # Check convergence
    if best["best_epoch"] < best["total_epochs"] * 0.5:
        p(f"⚡ Model converged quickly (epoch {best['best_epoch']}/{best['total_epochs']})")
        p(f"   Could reduce epochs for faster training in future runs.")

    p("=" * 80)

