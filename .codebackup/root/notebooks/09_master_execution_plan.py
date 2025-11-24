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
    dataset = EnhancedImageMaskDataset(
            sample_entries,
            config.paths.train_images,
            mode = mode,
            transform = val_tf
    )

    img_t, mask_t = dataset[0]
    p(f"Mode: {mode}", f"Image shape: {img_t.shape}, Mask shape: {mask_t.shape}")



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
    if num_epochs is not None:
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
# Running experiments

# Define filter combinations to test
filter_sets = {
    'classic':        ['laplacian', 'sobel', 'clahe'],
    'gaussian':       ['gaussian_3x3', 'gaussian_5x5', 'gaussian_7x7'],
    'kernel_sharpen': ['sharpen_basic', 'high_pass_3x3', 'edge_enhance'],
    'kernel_edge':    ['sobel_x', 'sobel_y', 'laplacian_3x3'],
    'combined':       ['laplacian', 'gaussian_5x5', 'clahe'],
}

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

results = { }
for model_name, mode, filters in experiments:
    key = f"{model_name}_{mode}"
    results[key] = train_experiment(model_name, mode, filters, num_epochs = 10)

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

# Minimal experiment set for initial testing
experiments = [
    ('simple_cnn', 'rgb', None),
]


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
        p("Warning", "Insufficient samples for training")
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

    train_loader = DataLoader(train_ds, batch_size = config.train.batch_size, shuffle = True)
    val_loader = DataLoader(val_ds, batch_size = config.train.batch_size, shuffle = False)

    # Train
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
## Train individual tree model
trainer_individual = train_class_specific_model('individual_tree', 'simple_cnn')
trainer_group = train_class_specific_model('group_of_trees', 'simple_cnn')


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

        p(f"Generating submission for {model_name}/{input_mode}", color1 = c.RED)

        submission_path = generate_submission(
                model_path = model_path,
                model_name = model_name,
                eval_dir = config.paths.eval_images,
                output_path = latest_version / "submission.json"
        )

        p("Saved submission", submission_path)
        p()

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


# %%
