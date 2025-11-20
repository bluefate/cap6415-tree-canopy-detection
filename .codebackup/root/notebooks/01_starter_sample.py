# %% [markdown]
# # Notebook: 01 Starter Sample
# ### Purpose: load data, inspect shapes, verify dataset logic, preview masks, confirm training setup.

# %%
import random

import cv2
import numpy as np
import torch

from src.data.annotations import load_json_annotations
from src.data.augmentations import get_val_augmentations
from src.data.loaders import ImageMaskDataset
from src.exploration.visualize import show_image, show_mask, show_overlay, show_side_by_side
from src.utils.config import Config
from src.utils.helpers import init_notebook, p, t


config = Config.load()

init_notebook(config.train.seed)

train_dir = config.paths.train_images
mask_dir = config.paths.train_masks
annotations_path = config.paths.annotations
entries = load_json_annotations(annotations_path)

# %%
# annotation count
p("Annotation entries", len(entries))


# %% [markdown]
# #### Build simple dataset with validation transforms
#

# %%
val_tf = get_val_augmentations(config.train.image_size)
dataset = ImageMaskDataset(entries, train_dir, transform = val_tf)
p("Dataset size", len(dataset))

# %% [markdown]
# #### Sample inspection

# %%
for _ in range(1):
    idx = random.randint(0, len(dataset) - 1)
    t(f"Image {idx}")

    img_t, mask_t = dataset[idx]

    img = img_t.permute(1, 2, 0).numpy()
    mask = mask_t.squeeze().numpy()

    titles = [f"Image {idx}", f"Mask", "Overlay"]

    show_image(img, f"Sample {idx}")
    show_mask(mask, "Mask")
    show_overlay(img, mask, 0.4, "Overlay")


# %%
for _ in range(3):
    idx = random.randint(0, len(dataset) - 1)
    t(f"Image {idx}")

    img_t, mask_t = dataset[idx]

    img = img_t.permute(1, 2, 0).numpy()
    mask = mask_t.squeeze().numpy()

    titles = [f"Image {idx}", f"Mask", "Overlay"]

    if img.max() <= 1.0:
        base = (img * 255).astype(np.uint8)
    else:
        base = img.astype(np.uint8)

    mask_u8 = (mask * 255).astype(np.uint8)
    mask_rgb = np.zeros_like(base)
    mask_rgb[:, :, 0] = mask_u8
    overlay = cv2.addWeighted(base, 0.6, mask_rgb, 0.4, 0)

    show_side_by_side(img, mask, overlay, titles = titles)




# %% [markdown]
# #### Testing forward pass with a small model

# %% [markdown]
# - sample_img is a tensor shaped [3, H, W]
# - sample_mask is a tensor shaped [1, H, W]

# %%
sample_img, sample_mask = dataset[0]

# %% [markdown]
# - adding a batch dimension

# %%
sample_img = sample_img.unsqueeze(0)

# %% [markdown]
# - now the image is shaped [1, 3, H, W]
# - batch size is 1

# %%
from src.models.sample_model_provided import SampleModelProvided


model = SampleModelProvided(in_channels = 3, out_channels = 1)
model.eval()

with torch.no_grad():
    out = model(sample_img)

p("Forward pass result shape", out.shape)


# %%
model = SampleModelProvided(in_channels = 3, out_channels = 3)
model.eval()

with torch.no_grad():
    out = model(sample_img)

p("Forward pass result shape", out.shape)
