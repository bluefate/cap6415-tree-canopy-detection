# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import random
import shutil
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from PIL import Image
from src.utils.config import inject_config_vars, load_config
from src.utils.helpers import format_number, init_this_notebook, p

# %% [markdown]
# ### Import configuration loader

# %%
config = load_config()
inject_config_vars(config)

# %%
init_this_notebook(SEED)

# %% [markdown]
# ### Extract images and remove __MACOSX folders

# %%
# Mapping of zip files to their extraction targets
paths = {
    PATHS_TRAIN_IMAGES_ZIP: PATHS_TRAIN_IMAGES,
    PATHS_EVAL_IMAGES_ZIP: PATHS_EVAL_IMAGES,
}

# --- Extract zip archives ---
for zip_file, extract_to in paths.items():
    if not zip_file.exists():
        p("Zip file not found", zip_file)
        continue

    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_file, "r") as zf:
        zf.extractall(extract_to)
    p("Extracted", f"{zip_file.relative_to(ROOT)} -> {extract_to.relative_to(ROOT)}")

    # Remove __MACOSX folders if present
    for macosx_dir in extract_to.rglob("__MACOSX"):
        if macosx_dir.is_dir():
            shutil.rmtree(macosx_dir)
            p("Removed: ", macosx_dir.relative_to(ROOT))

    print("")

# %% [markdown]
# ### Display one random image from each folder

# %%
# preview
for folder in paths.values():
    if not folder.exists():
        print(f"Folder not found: {folder}")
        continue

    images = [
        f
        for f in folder.glob("*.*")
        if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
    ]
    if not images:
        p("", f"No images found in {folder.relative_to(ROOT)}")
        continue

    p(folder.relative_to(ROOT))

    # img_path = random.choice(images)
    # with Image.open(img_path) as img:
    #     plt.figure(figsize=(3, 3), dpi=100)
    #     plt.imshow(img)
    #     plt.title(img_path.name, fontsize=8)
    #     plt.axis("off")
    #     plt.tight_layout()
    #     plt.show()

    # Pick up to 10 random images
    num_samples = min(10, len(images))
    sample_imgs = random.sample(images, num_samples)

    rows, cols = 2, 5
    plt.figure(figsize=(12, 6), dpi=100)

    for i, img_path in enumerate(sample_imgs, 1):
        with Image.open(img_path) as img:
            plt.subplot(rows, cols, i)
            plt.imshow(img)
            plt.title(img_path.name, fontsize=8)
            plt.axis("off")

    plt.tight_layout()
    plt.show()

# %%
import json
import os

import cv2
import numpy as np
from tqdm import tqdm


def create_masks_from_custom_json(json_path, images_dir, output_dir):
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    images = data.get("images", [])
    p(f"Found {len(images)} images in annotation file")

    for img_entry in tqdm(images, desc="Creating masks", ncols=80, colour="green"):
        file_name = img_entry.get("file_name")
        width = img_entry.get("width")
        height = img_entry.get("height")
        annotations = img_entry.get("annotations", [])

        mask = np.zeros((height, width), dtype=np.uint8)

        for annotation in annotations:
            segmentation = annotation.get("segmentation", [])
            if not segmentation:
                continue

            # segmentation is expected to be [x1, y1, x2, y2, ...]
            poly = np.array(segmentation).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [poly], color=1)

        mask_img = (mask * 255).astype(np.uint8)
        out_path = output_dir / file_name
        Image.fromarray(mask_img).save(out_path)

    p("Masks saved to", output_dir)
    p("Total generated", len(os.listdir(output_dir)))


create_masks_from_custom_json(
    PATHS_DATA / "train_annotations.json",
    PATHS_TRAIN_IMAGES,
    PATHS_TRAIN_MASKS,
)

# %%
images = [
    f
    for f in PATHS_TRAIN_MASKS.glob("*.*")
    if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
]

num_samples = min(5, len(images))
sample_imgs = random.sample(images, num_samples)


rows, cols = 2, 5
plt.figure(figsize=(12, 6), dpi=100)
for i, img_path in enumerate(sample_imgs, 1):
    with Image.open(img_path) as img:
        plt.subplot(rows, cols, i)
        plt.imshow(img)
        plt.title(img_path.name, fontsize=8)
        plt.axis("off")

plt.tight_layout()
plt.show()


## ---
# Channel Order
# Pillow (Image.open)	RGB	[255, 0, 0] = red
# OpenCV (cv2.imread)	BGR	[0, 0, 255] = red


plt.figure(figsize=(12, 6), dpi=100)
for i, img_path in enumerate(sample_imgs, 1):
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]

    # Show original image
    plt.subplot(rows, cols, i)
    plt.imshow(img_rgb)
    plt.title(
        f"RGB mean: {red.mean():.0f}, {green.mean():.0f}, {blue.mean():.0f}", fontsize=8
    )
    plt.axis("off")

plt.tight_layout()
plt.show()

# %%
p("images", len(os.listdir(PATHS_TRAIN_IMAGES)))
p("masks", len(os.listdir(PATHS_TRAIN_MASKS)))
p("eval images", len(os.listdir(PATHS_EVAL_IMAGES)))


# %%
def show_image_with_mask(image_path, mask_path, alpha=0.4):
    """
    Display an image and its mask overlayed.
    alpha controls mask transparency (0 = invisible, 1 = opaque).
    """

    image = cv2.imread(str(image_path))
    if image is None:
        p("Failed to read image", image_path, color="red")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        p("Failed to read mask", mask_path, color="orange")
        return

    # Create red overlay for mask
    mask_rgb = np.zeros_like(image)
    mask_rgb[:, :, 0] = mask  # Red channel

    overlay = cv2.addWeighted(image, 1 - alpha, mask_rgb, alpha, 0)

    plt.figure(figsize=(6, 3), dpi=100)
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Image", fontsize=8)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title("With Mask Overlay", fontsize=8)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


nums = random.sample(range(1, 151), 20)
for n in nums:
    sample_img = sorted(list(PATHS_TRAIN_IMAGES.glob("*")))[n]
    sample_mask = PATHS_TRAIN_MASKS / sample_img.name
    show_image_with_mask(sample_img, sample_mask)

# %%
