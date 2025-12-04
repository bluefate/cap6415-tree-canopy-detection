# %% [markdown]
# # Notebook: 03 Training Test
# ### Purpose: create train and validation datasets, build augmentations, run Trainer, save versioned checkpoints.
#

# %%
import os
import sys

from torch.nn import CrossEntropyLoss

from training.running import get_version_config


sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../src"))

import random

import torch

from src.data.annotations import load_json_annotations
from src.data.augmentations import get_train_augmentations, get_val_augmentations
from src.data.loaders import ImageMaskDataset
from src.models.zoo import MODEL_BUILDERS
from src.training.engine import run_training
from src.utils.config import Config
from src.utils.helpers import init_notebook, p, t, c


config = Config.load()

init_notebook(config.train.seed)

train_dir = config.paths.train_images
annotations_path = config.paths.annotations
entries = load_json_annotations(annotations_path)

# Shuffle entries
random.shuffle(entries)


# %% [markdown]
# #### Dataset split

# %%
# Compute number of validation samples (20 percent of dataset)
val_count = max(1, int(0.2 * len(entries)))

# Split validation set, and training set
val_entries = entries[:val_count]
train_entries = entries[val_count:]

# Build augmentation pipelines for training and validation
train_tf = get_train_augmentations(config.train.image_size)
val_tf = get_val_augmentations(config.train.image_size)

# Build dataset objects that load image-mask pairs and apply transforms
train_ds = ImageMaskDataset(train_entries, train_dir, transform = train_tf)
val_ds = ImageMaskDataset(val_entries, train_dir, transform = val_tf)

# Build dataloaders
train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size = config.train.batch_size,
        shuffle = True,
        num_workers = config.train.num_workers,
)

val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size = config.train.batch_size,
        shuffle = False,
        num_workers = config.train.num_workers,
)

p("Train samples", len(train_ds))
p("Val samples", len(val_ds))

# %% [markdown]
# #### Available Models

# %%
p("Models", MODEL_BUILDERS)
#config.show()
p("Batch", config.train.batch_size)
p("Epochs", config.train.epochs)
p("Learning Rate", config.train.learning_rate, precision = 9)
p("Image Size", config.train.image_size)


# %% [markdown]
# #### Run Training

# %%
# Create structured path: checkpoints/notebook_eval/[model]/[mode]/[version]
model_name, mode, in_channels = "simple_cnn", "rgb", "3"

key, version_root, exp_config, best_model_path, checkpoint_path_check, best_model_exists = get_version_config(
        config, None, "03", model_name, mode, in_channels, None, None, None
)

trainer = run_training(
        config = exp_config,
        train_loader = train_loader,
        val_loader = val_loader,
        version_root = version_root,
        model_name = model_name,
)

# %% [markdown]
#

# %%
from models.zoo import build_model


# Verify tensor types
t("Tensor Type Verification")
sample_img, sample_mask = train_ds[0]
p(f"Image dtype: {sample_img.dtype}, shape: {sample_img.shape}")
p(f"Mask dtype: {sample_mask.dtype}, shape: {sample_mask.shape}", color1 = c.BLUE)
p(f"Mask values: min={sample_mask.min()}, max={sample_mask.max()}", color1 = c.BLACK)
p(f"Mask unique values: {torch.unique(sample_mask)}", color1 = c.BLACK)

# Test a batch
batch_imgs, batch_masks = next(iter(train_loader))
p(f"\nBatch image dtype: {batch_imgs.dtype}, shape: {batch_imgs.shape}")
p(f"Batch mask dtype: {batch_masks.dtype}, shape: {batch_masks.shape}", color1 = c.BLUE)

# Test with model
model = build_model('simple_cnn', in_channels = 3, out_channels = 3)
with torch.no_grad():
    preds = model(batch_imgs[:1])
p(f"\nModel output dtype: {preds.dtype}, shape: {preds.shape}", color1 = c.BLACK)

# Test loss
criterion = CrossEntropyLoss()
loss = criterion(preds, batch_masks[:1])
p(f"Loss computed successfully: {loss.item()}", color1 = c.BLACK)


# %%
from prediction.validation import analyze_validation_metrics


if trainer is not None:
    checkpoint_path = trainer.paths["checkpoint"]
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location = "cpu")

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
                model_name = "simple_cnn"
        )

