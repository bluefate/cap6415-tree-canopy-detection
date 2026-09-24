# %% [markdown]
# # Notebook: 01 Starter Sample
# ### Purpose: load data, inspect shapes, verify dataset logic, preview masks, confirm training setup.

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
from src.exploration.visualize import show_image, show_mask, show_overlay, show_side_by_side
from src.utils.config import Config
from src.utils.helpers import init_notebook, p, t


config = Config.load(root = root)

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

    # Handle normalized images properly for display
    if img_t.min() < 0:  # ImageNet normalized
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_denorm = torch.clamp(img_t * std + mean, 0.0, 1.0)
        img = img_denorm.permute(1, 2, 0).numpy()
    else:  # Already in [0,1] range
        img = img_t.permute(1, 2, 0).numpy()
    mask = mask_t.squeeze().numpy()

    show_image(img, f"Sample {idx}")
    show_mask(mask, "Mask")
    show_overlay(img, mask, 0.4, "Overlay")

    t("Checking show_side_by_side()")
    titles = [f"Image {idx}", f"Mask", "Overlay"]

    img_t = show_image(img, return_img = True)
    mask = show_mask(mask, return_img = True)
    overlay = show_overlay(img, mask, return_img = True)

    show_side_by_side(img, mask, overlay, titles = titles)


# %%
for _ in range(5):
    idx = random.randint(0, len(dataset) - 1)
    t(f"Image {idx}")

    img_t, mask_t = dataset[idx]

    # Handle normalized images properly for display
    if img_t.min() < 0:  # ImageNet normalized
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_denorm = torch.clamp(img_t * std + mean, 0.0, 1.0)
        img = img_denorm.permute(1, 2, 0).numpy()
    else:  # Already in [0,1] range
        img = img_t.permute(1, 2, 0).numpy()
    mask = mask_t.squeeze().numpy()

    titles = [f"Image {idx}", "Mask", "Overlay"]

    # Convert image to uint8 for cv2 operations
    if img.max() <= 1.0:
        base = (img * 255).astype(np.uint8)
    else:
        base = img.astype(np.uint8)

    # Handle mask properly for overlay
    mask_binary = (mask > 0).astype(np.uint8)  # Convert class indices to binary
    mask_rgb = np.zeros_like(base)
    mask_rgb[:, :, 0] = mask_binary * 255  # Red channel for trees
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


model = SampleModelProvided(in_channels = 3, out_channels = 3)
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
