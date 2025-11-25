# %% [markdown]
# # Notebook: 09 Master Execution Plan
# ### Purpose: Complete pipeline from data preparation to final submission

# %%
import os
import sys
from pathlib import Path


#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  #enable for debugging ONLY

sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../src"))


def timeout_handler( signum, frame ):
    raise TimeoutError("Training took too long")


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
p("Sample filters", AVAILABLE_FILTERS[:15])

# %% [markdown]
# #### Step 3: Enhanced Dataset Creation
#
# - Create training dataset with filter-enhanced inputs
# - Apply filters as additional channels.
#
#

# %%
t("Testing Enhanced Dataset")

val_tf = get_val_augmentations(config.train.image_size)
sample_entries = entries[:5]

# Testing each mode
for mode in ['rgb', 'filtered', 'concat']:
    try:
        dataset = EnhancedImageMaskDataset(
                sample_entries,
                config.paths.train_images,
                mode = mode,
                transform = val_tf
        )

        img_t, mask_t = dataset[0]
        p(f"Mode: {mode}", f"Image shape: {img_t.shape}, Mask shape: {mask_t.shape}")
    except Exception as e:
        p(f"Mode: {mode}", f"FAILED: {e}", color1=c.RED, color2=c.RED)


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

    # Configure training
    import copy

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


p("", "Experiments configured")



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

p("experiments", experiments)

# %%
# # Minimal experiment set for initial testing
# t("Testing")
# filter_sets = dict(list(filter_sets.items())[:1])
# p("filter_sets", filter_sets)
#
# experiments = [
#     ('simple_cnn', 'rgb', None),
# ]
# p("experiments", experiments)


# %%
def estimate_runtime( experiments, filter_sets ):
    """
    Estimate training runtime using:
      - experiments list
      - filter_sets dict
    Adjusts time for rgb vs filtered vs concat.
    """

    t("Runtime Estimate")

    # # Load config and dataset size
    # config = Config.load()
    # from src.data.annotations import load_json_annotations
    # entries = load_json_annotations(config.paths.annotations)

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
      experiments: list of (model, mode, filters)
      filter_sets: dict {name: [filters]} used externally
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
# Running experiments
results = { }

# Pull epoch count from config
num_epochs = config.train.epochs
p("Using epochs from config", num_epochs)

for i, (model_name, mode, filters) in enumerate(experiments, 1):
    p("")
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
        results[key] = train_experiment(
                model_name,
                mode,
                filters,
                num_epochs = config.train.epochs
        )

        # Mark successful completion
        if results[key] is not None:
            p("EXPERIMENT COMPLETED", key, color1 = c.GREEN, color2 = c.GREEN)
        else:
            p("EXPERIMENT SKIPPED", key, color1 = c.ORANGE, color2 = c.ORANGE)


    except KeyError as e:
        # Filter name errors
        p("EXPERIMENT FAILED", key, color1 = c.RED, color2 = c.RED)
        p("[Filter Error]", str(e), color1 = c.RED)
        p("Skipping to next experiment", color1 = c.BLUE)
        results[key] = { "status": "FILTER_ERROR", "error": str(e) }
        continue

    except RuntimeError as e:
        # Data loading or model errors
        p("EXPERIMENT FAILED", key, color1 = c.RED, color2 = c.RED)
        p("[Runtime Error]", str(e), color1 = c.RED)
        p("Skipping to next experiment", color1 = c.BLUE)
        results[key] = { "status": "RUNTIME_ERROR", "error": str(e) }
        continue

    except Exception as e:
        # Catch-all for other errors
        p("EXPERIMENT FAILED", key, color1 = c.RED, color2 = c.RED)
        p("[Error]", str(e), color1 = c.RED)
        p("Skipping to next experiment", color1 = c.BLUE)
        results[key] = { "status": "ERROR", "error": str(e) }
        continue

# Summary of experiment results
t("Experiment Results Summary")
successful = sum(1 for v in results.values() if not isinstance(v, dict) or v.get("status") != "ERROR")
failed = sum(1 for v in results.values() if isinstance(v, dict) and "status" in v)
skipped = sum(1 for v in results.values() if v is None)

p("Total experiments", len(experiments))
p("Successful", successful - skipped, color1 = c.GREEN)
p("Skipped", skipped, color1 = c.ORANGE)
p("Failed", failed, color1 = c.RED if failed > 0 else c.GREEN)



# # If running all experiments:
# experiments = [
#     # === Simple CNN ===
#     ('simple_cnn', 'rgb', None),
#     ('simple_cnn', 'filtered', ['laplacian', 'sobel', 'clahe']),
#     ('simple_cnn', 'filtered', ['gaussian_3x3', 'gaussian_5x5', 'gaussian_7x7']),
#     ('simple_cnn', 'filtered', ['sharpen_basic', 'high_pass_3x3', 'edge_enhance']),
#     ('simple_cnn', 'filtered', ['sobel_x', 'sobel_y', 'laplacian_3x3']),
#
#     # === UNet ===
#     ('unet', 'rgb', None),
#     ('unet', 'filtered', ['laplacian', 'sobel', 'clahe']),
#     ('unet', 'filtered', ['gaussian_3x3', 'gaussian_5x5', 'gaussian_7x7']),
#     ('unet', 'filtered', ['sobel_x', 'sobel_y', 'laplacian_3x3']),
# ]



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



# %% [markdown]
# #### Step 7: Submission Generation
#
# **Current Status:**
# - Prediction pipeline exists (notebook 05)
# - Submission export implemented (`export_submission`)
#

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

    # Validate JSON structure
    import json

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


# %% [markdown]
# #### Generate submissions for all trained experiments

# %%
best_submissions = []

# Generate submissions for all trained experiments
for model_name in ['simple_cnn', 'unet']:
    for input_mode in ['rgb', 'filtered']:

        version_root = config.paths.models / model_name / input_mode

        if not version_root.exists():
            p("Skipping", f"{model_name}/{input_mode} (not trained yet)")
            continue

        vm = VersionManager(version_root)
        latest_version = vm.find_latest()

        if latest_version is None:
            p("No versions found for", f"{model_name}/{input_mode}")
            continue

        model_path = latest_version / "best_model.pth"

        if not model_path.exists():
            p("Model not found", model_path)
            continue

        p("\nGenerating submission", f"{model_name}/{input_mode}", color1 = c.MAGENTA, color2 = c.MAGENTA)

        output_file = latest_version / "submission.json"

        try:
            submission_path = generate_submission(
                    model_path = model_path,
                    model_name = model_name,
                    eval_dir = config.paths.eval_images,
                    output_path = output_file
            )

            # Track best submission file per experiment
            best_submissions.append(
                    {
                        "model":      model_name,
                        "mode":       input_mode,
                        "version":    str(latest_version),
                        "submission": str(output_file)
                    }
            )

            p("Saved submission", submission_path)

        except Exception as e:
            p("Failed to generate submission", str(e), color1 = c.RED)
            continue

        p("")





# %%
# Build global list of all best experiment submissions
best_submissions = []

for model_name in ['simple_cnn', 'unet']:
    for input_mode in ['rgb', 'filtered']:

        version_root = config.paths.models / model_name / input_mode

        if not version_root.exists():
            continue

        vm = VersionManager(version_root)
        latest_version = vm.find_latest()

        if latest_version is None:
            continue

        model_path = latest_version / "best_model.pth"
        if not model_path.exists():
            continue

        submission_file = latest_version / "submission.json"

        best_submissions.append(
                {
                    "model":      model_name,
                    "mode":       input_mode,
                    "version":    str(latest_version),
                    "submission": str(submission_file)
                }
        )

# ===== Find best overall experiment =====

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

# Append overall best
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
import cv2
from exploration.class_explorer import load_image, mask_all, color_mask
from src.exploration.visualize import show_side_by_side
import random


image_dir = config.paths.train_images

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
                img,
                mask,
                mask_rgb,
                overlay,
                titles = ("Original", "Mask", "Color Mask", "Overlay"),
                maxcolumns = 6
        )
    except Exception as e_viz:
        p("Failed to visualize", str(e_viz), color1 = c.ORANGE)


# %%
