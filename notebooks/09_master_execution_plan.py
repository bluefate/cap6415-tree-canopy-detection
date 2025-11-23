#%% md
# # Notebook: 09 Master Execution Plan
# ### Purpose: Complete pipeline from data preparation to final submission
#%%
import os
import sys


sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../src"))

import torch
from torch.utils.data import DataLoader

from src.data.annotations import load_json_annotations
from src.data.augmentations import get_train_augmentations, get_val_augmentations
from src.data.loaders import ImageMaskDataset
from src.training.engine import run_training
from src.utils.config import Config
from src.utils.helpers import init_notebook, p, t
from src.data.enhance_masks import EnhancedImageMaskDataset


config = Config.load()
init_notebook(config.train.seed)


#%% md
# #### Stage 1: Data Preparation
# 
# - TIFF → PNG conversion (notebook 00_preprocess_images)
# - Annotation loading
# - Mask generation
# - Data augmentation
# 
# 
#%%
t("Loading Annotations")
entries = load_json_annotations(config.paths.annotations)
p("Total images", len(entries))

# Analyze class distribution
individual_count = sum(1 for e in entries if any(item.cls == "individual_tree" for item in e.items))
group_count = sum(1 for e in entries if any(item.cls == "group_of_trees" for item in e.items))

p("Images with individual trees", individual_count)
p("Images with tree groups", group_count)


#%% md
# #### Stage 2: Filter Experimentation
# 
# **Action:** Run and retrieve notebook 08 to identify top 3 filters
# 
# **Expected Output:**
# - Filter ranking CSV
# - Top 3 filter names
# - Visual comparisons
# 
# 
#%% md
# #### Stage 3: Enhanced Dataset Creation
# 
# - Create training dataset with filter-enhanced inputs
# - Apply filters as additional channels.
# 
# 
#%%
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


#%% md
# #### Stage 4: Model Training Comparison
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
#%%
def train_experiment( model_name, input_mode, filter_names = None, num_epochs = None ):
    """
    Run one training experiment and return metrics.
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

    # Determine input channels
    sample_img, _ = train_ds[0]
    in_channels = sample_img.shape[0]
    p("Input channels", in_channels)

    # Build model
    if in_channels != 3:
        # Need to modify model for non-standard input
        p("Warning", "6-channel input requires model modification")
        # For now, skip 6-channel experiments
        return None

    # Use modified config for this experiment
    import copy

    exp_config = copy.deepcopy(config)
    if num_epochs is not None:
        exp_config.train.epochs = num_epochs

    # Train
    trainer = run_training(
            config = exp_config,
            train_loader = train_loader,
            val_loader = val_loader,
            version_root = config.paths.models,
            model_name = model_name
    )

    return trainer


p("", "Experiments configured")


#%%
# Running experiments

experiments = [
    ('simple_cnn', 'rgb', None),
    ('simple_cnn', 'filtered', ['laplacian', 'sobel', 'clahe']),
    ('unet', 'rgb', None),
    ('unet', 'filtered', ['laplacian', 'sobel', 'clahe']),
]

results = { }
for model_name, mode, filters in experiments:
    key = f"{model_name}_{mode}"
    results[key] = train_experiment(model_name, mode, filters, num_epochs = 10)

#%% md
# #### Stage 5: Class-Specific Training
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
#%%
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
#%%
## Train individual tree model
trainer_individual = train_class_specific_model('individual_tree', 'simple_cnn')
trainer_group = train_class_specific_model('group_of_trees', 'simple_cnn')
#%% md
# #### Stage 6: Ensemble Predictions
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
#%%
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


#%%
### TEST PSEUDO CODE
models = [
    (model_rgb, 1.0),
    (model_filtered, 1.2),  # Higher weight if better validation IoU
    (model_individual, 0.8),
    (model_group, 0.8),
]

final_prediction = ensemble_predict(models, input_tensor)
#%% md
# #### Stage 7: Submission Generation
# 
# **Current Status:**
# - Prediction pipeline exists (notebook 05)
# - Submission export implemented (`export_submission`)
# 
#%%
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
#%%
submission_path = generate_submission(
        model_path = config.paths.models / "v001" / "best_model.pth",
        model_name = "simple_cnn",
        eval_dir = config.paths.eval_images,
        output_path = config.paths.models / "submission.json"
)
#%% md
# #### Completed
# - Data preparation (TIFF→PNG, annotations, masks)
# - Exploration tools (kernels, filters, visualization)
# - Training infrastructure (models, metrics, versioning)
# - Prediction pipeline
# 
# 
# #### Ideas, To Do
# 
# - Identify top 3 filters
# - Update config.yaml with TOP_3_FILTERS
# 
# Run Stage 4 - Model Comparison
# - Train SimpleCNN on RGB
# - Train SimpleCNN on filtered input
# - Compare IoU/Dice scores
# 
# Run Stage 5 (Class-Specific Training)
# - Train model for individual_tree
# - Train model for group_of_trees
# - Compare with combined model
# 
# Other
# - Implement ensemble predictions
# - Validate submission JSON