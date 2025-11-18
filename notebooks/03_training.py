# %% [markdown]
# # Notebook: 03 Training
# ## Purpose: create train and validation datasets, build augmentations, run Trainer, save versioned checkpoints.
#

# %%

import random

import torch

from src.data.annotations import load_json_annotations
from src.data.augmentations import get_train_augmentations, get_val_augmentations
from src.data.loaders import ImageMaskDataset
from src.models.zoo import MODEL_BUILDERS
from src.training.engine import run_training
from src.utils.config import Config
from src.utils.helpers import init_notebook, p


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
version_root = config.paths.models
trainer = run_training(
        config = config,
        train_loader = train_loader,
        val_loader = val_loader,
        version_root = version_root,
        model_name = "simple_cnn",
)
