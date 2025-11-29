# %% [markdown]
# # Notebook: 07 Pipeline Orchestration
# ### Imports and setup

# %% [markdown]
# #### Imports and setup

# %%
import os
import sys


sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../src"))
from collections import defaultdict
import cv2
import numpy as np
from torch.utils.data import DataLoader

from exploration.class_explorer import explore_bboxes
from src.data.annotations import load_json_annotations
from src.data.augmentations import get_train_augmentations, get_val_augmentations
from src.data.loaders import ImageMaskDataset
from src.exploration.filters import cv2_apply_gaussian, cv2_apply_laplacian, cv2_apply_sobel
from src.exploration.kernels import apply_kernel_using_convolution, get_kernels
from src.exploration.visualize import show_side_by_side
from src.prediction.pipeline import Predictor
from src.training.engine import run_training
from src.utils.config import Config
from src.utils.helpers import init_notebook, p, t
from src.utils.versioning import VersionManager

# %%
config = Config.load()
init_notebook(config.train.seed)

entries = load_json_annotations(config.paths.annotations)
image_dir = config.paths.train_images


# %%
explore_bboxes(entries[42], image_dir)

# %% [markdown]
# #### Step 1: Identify single class and group images

# %%
# individual_tree images
single_individual = [
    e for e in entries
    if any(item.cls == "individual_tree" for item in e.items)
]

# Images with at least one group_of_trees
single_group = []
for entry in entries:
    cls_set = { item.cls for item in entry.items }
    if "group_of_trees" in cls_set:
        single_group.append(entry)

# single_group = [
#     e for e in entries
#     if any(item.cls == "group_of_trees" for item in e.items)
# ]

p("single_individual", len(single_individual))
p("single_group", len(single_group))


# %% [markdown]
# #### Step 2: Extract each bbox from each single image and inspect

# %%
sample_entry = single_individual[0]


# %%
# def crop_bbox( entry, item ):
#     img = cv2.imread(str(image_dir / entry.image_path.name))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     x1, y1, x2, y2 = item.bbox
#     return img[y1:y2, x1:x2]


def crop_bbox( entry, item ):
    img_path = image_dir / entry.image_path.name
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x1, y1, x2, y2 = item.bbox
    return img[y1:y2, x1:x2]


def crop_mask( entry, item ):
    mask = np.zeros((entry.height, entry.width), dtype = np.uint8)
    seg = item.segmentation
    if seg and len(seg) >= 4:
        poly = np.array(seg, dtype = np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [poly], 1)
    x1, y1, x2, y2 = item.bbox
    return mask[y1:y2, x1:x2]


def pad_to_size( img, target_h, target_w ):
    h, w = img.shape[:2]
    pad_h = target_h - h
    pad_w = target_w - w

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    return cv2.copyMakeBorder(
            img,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value = [0, 0, 0]
    )



# %% [markdown]
# ##### Collect all crops first and record their shapes

# %%
# bboxes = []
# for item in sample_entry.items:
#     p(crop_bbox(sample_entry, item))
#     bboxes.append(crop_bbox(sample_entry, item))
#     masks.append(crop_mask(sample_entry, item))
#
# max_h = max([arr.shape[0] for arr in bboxes])
# max_w = max([arr.shape[1] for arr in bboxes])
#
# padded_bboxes = [pad_to_size(arr, max_h, max_w) for arr in bboxes]
# padded_masks = [pad_to_size(arr, max_h, max_w) for arr in masks]
#
# overlays = []
# for bb, mm in zip(padded_bboxes, padded_masks):
#     rgb = np.array(bb)
#     rgb[:, :, 0] = np.maximum(rgb[:, :, 0], mm)
#     overlays.append(rgb)
#
# show_side_by_side([padded_bboxes[0], padded_masks[0], overlays[0]], ["bbox", "mask", "overlay"])
#


# %%
crops = []
#crops = [crop_bbox(sample_entry, item) for item in sample_entry.items]
for item in sample_entry.items:
    c = crop_bbox(sample_entry, item)
    crops.append(c)

heights = [c.shape[0] for c in crops]
widths = [c.shape[1] for c in crops]
max_h = max(heights)
max_w = max(widths)

# %% [markdown]
# ##### Build padded crops, padded masks, and overlays.

# %%
padded_crops = []
padded_masks = []
padded_overlays = []

for idx, item in enumerate(sample_entry.items):
    crop = crops[idx]
    cmask = crop_mask(sample_entry, item)

    crop_p = pad_to_size(crop, max_h, max_w)
    mask_p = pad_to_size(cmask, max_h, max_w)

    overlay = cv2.addWeighted(
            crop_p.astype(np.uint8),
            0.6,
            np.dstack([mask_p * 255, np.zeros_like(mask_p), np.zeros_like(mask_p)]).astype(np.uint8),
            0.4,
            0
    )

    padded_crops.append(crop_p)
    padded_masks.append(mask_p)
    padded_overlays.append(overlay)


# %%
num_show = 9
indexes = list(range(min(num_show, len(padded_crops))))

triplets, titles = [], []

for idx in indexes:
    triplets.extend(
            [
                padded_crops[idx],
                padded_masks[idx],
                padded_overlays[idx]
            ]
    )
    titles.extend(
            [
                f"crop {idx}",
                f"mask {idx}",
                f"overlay {idx}"
            ]
    )

show_side_by_side(
        *triplets,
        titles = tuple(titles),
        maxcolumns = 9
)


# %% [markdown]
# ##### FOR TRAINING

# %% [markdown]
# ##### Prepare all individual crops for training

# %%
# Step 1: Group items by image to minimize disk loads
image_to_items = defaultdict(list)
for e_idx, entry in enumerate(single_individual):
    for j_idx, item in enumerate(entry.items):
        if item.cls == "individual_tree":
            image_to_items[entry.image_path.name].append((entry, item, e_idx, j_idx))

p("Unique images to process", len(image_to_items))

# %%
# Step 2: Process each image once
padded_crops_ind = []
padded_masks_ind = []
idx_entry_map = []
idx_item_map = []

for img_name, items_list in image_to_items.items():
    # Load image ONCE per unique image
    entry = items_list[0][0]
    img_path = image_dir / img_name

    img = cv2.imread(str(img_path))
    if img is None:
        p("Warning: Failed to load", img_name, color1 = "red")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Collect ALL crops and masks for THIS image
    local_crops = []
    local_masks = []
    local_entries = []

    for entry, item, e_idx, j_idx in items_list:
        # Crop directly from loaded image
        x1, y1, x2, y2 = item.bbox
        crop = img[y1:y2, x1:x2]

        # Create mask
        mask = np.zeros((entry.height, entry.width), dtype = np.uint8)
        seg = item.segmentation
        if seg and len(seg) >= 4:
            poly = np.array(seg, dtype = np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [poly], 1)
        mask_crop = mask[y1:y2, x1:x2]

        local_crops.append(crop)
        local_masks.append(mask_crop)
        local_entries.append((e_idx, j_idx))

    # Calculate max dimensions for THIS IMAGE ONLY
    local_heights = [c.shape[0] for c in local_crops]
    local_widths = [c.shape[1] for c in local_crops]
    local_max_h = max(local_heights)
    local_max_w = max(local_widths)

    # Pad to local max (not global)
    for crop, mask, (e_idx, j_idx) in zip(local_crops, local_masks, local_entries):
        padded_crops_ind.append(pad_to_size(crop, local_max_h, local_max_w))
        padded_masks_ind.append(pad_to_size(mask, local_max_h, local_max_w))
        idx_entry_map.append(e_idx)
        idx_item_map.append(j_idx)

p("total padded crops", len(padded_crops_ind))
p("total padded masks", len(padded_masks_ind))
t("Extraction complete!")

# %% [markdown]
# ##### Sample padded and original images

# %%
# Sort by area (for visualization)
sizes = [(i, padded_crops_ind[i].shape) for i in range(len(padded_crops_ind))]
sizes_sorted = sorted(sizes, key = lambda x: x[1][0] * x[1][1], reverse = True)
#sizes_sorted = sizes_sorted + sorted(sizes, key=lambda x: x[1][0]*x[1][1])

#if using vector instead
# sizes = [(i, raw_crops_ind[i].shape) for i in range(len(raw_crops_ind))]
# sizes_sorted = sorted(sizes, key = lambda x: x[1][0] * x[1][1])
# #sizes_sorted = sorted(sizes, key = lambda x: x[1][0] * x[1][1], reverse = True)

# %%
# Visualize top 5 largest crops with their source images
for i in range(min(5, len(sizes_sorted))):
    idx = sizes_sorted[i][0]

    entry = single_individual[idx_entry_map[idx]]

    # Load full image for display
    full_img = cv2.imread(str(image_dir / entry.image_path.name))
    full_img = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)

    # Get the already-padded crop and mask
    padded_crop = padded_crops_ind[idx]
    padded_mask = padded_masks_ind[idx]

    overlay = cv2.addWeighted(
            padded_crop.astype(np.uint8),
            0.6,
            np.dstack([padded_mask * 255, np.zeros_like(padded_mask), np.zeros_like(padded_mask)]).astype(np.uint8),
            0.4,
            0
    )

    show_side_by_side(
            full_img, padded_crop, overlay,
            titles = (f"orig {idx}", f"padded {idx}", f"overlay {idx}")
    )

# %%
# grid of crops
idxs = [x[0] for x in sizes_sorted[:num_show]]

tiplets, titles = [], []
for i in idxs:
    crop_p = padded_crops_ind[i]
    mask_p = padded_masks_ind[i]
    overlay = cv2.addWeighted(
            crop_p.astype(np.uint8),
            0.6,
            np.dstack([mask_p * 255, np.zeros_like(mask_p), np.zeros_like(mask_p)]).astype(np.uint8),
            0.4,
            0
    )

    triplets.extend([crop_p, mask_p, overlay])
    titles.extend([f"crop {i}", f"mask {i}", f"overlay {i}"])

show_side_by_side(*triplets, titles = tuple(titles), maxcolumns = 12)

#### Step 3: Apply filters and kernel exploration on crops

# %% [markdown]
# ##### Direct crop

# %%
crop = crop_bbox(sample_entry, sample_entry.items[0])
gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

ga = cv2_apply_gaussian(crop)
so = cv2_apply_sobel(crop)
la = cv2_apply_laplacian(crop)

show_side_by_side(crop, ga, so, la, titles = ("orig", "gauss", "sobel", "lap"))


# %% [markdown]
# ##### Padded Crop

# %%
padded = pad_to_size(crop, max_h, max_w)

ga_p = cv2_apply_gaussian(padded)
so_p = cv2_apply_sobel(padded)
la_p = cv2_apply_laplacian(padded)

show_side_by_side(
        padded,
        ga_p,
        so_p,
        la_p,
        titles = ("padded orig", "padded gauss", "padded sobel", "padded lap"),
        maxcolumns = 4
)



# %%
kernels = get_kernels()
lap_k = kernels["Laplacian_3x3"]
lap_resp = apply_kernel_using_convolution(gray, lap_k)

show_side_by_side(gray, lap_resp, titles = ("gray", "lap3x3"))


# %% [markdown]
# #### Step 4: Train model on only individual class

# %%
train_tf = get_train_augmentations(config.train.image_size)
val_tf = get_val_augmentations(config.train.image_size)

train_ds_ind = ImageMaskDataset(
        single_individual,
        image_dir,
        classes = ["individual_tree"],
        transform = train_tf
)

val_ds_ind = ImageMaskDataset(
        single_individual[:20],
        image_dir,
        classes = ["individual_tree"],
        transform = val_tf
)

train_loader_ind = DataLoader(train_ds_ind, batch_size = config.train.batch_size, shuffle = True)
val_loader_ind = DataLoader(val_ds_ind, batch_size = config.train.batch_size, shuffle = False)

trainer_ind = run_training(
        config,
        train_loader_ind,
        val_loader_ind,
        config.paths.models,
        model_name = "simple_cnn"
)

# %% [markdown]
# #### Step 5: Train model on only group class

# %%

vm = VersionManager(config.paths.models)
version_dir = vm.find_latest()
model_path = version_dir / "best_model.pth"

predictor_ind = Predictor(
        model_path,
        model_name = "simple_cnn",
        image_size = config.train.image_size
)

results_single = predictor_ind.run_on_folder(
        image_dir,
        transform = get_val_augmentations(config.train.image_size)
)

all_singles = results_single

len(all_singles)




# %% [markdown]
# #### Step 6: Run single-class model on all images and save predictions

# %%

train_ds_grp = ImageMaskDataset(
        single_group,
        image_dir,
        classes = ["group_of_trees"],
        transform = train_tf
)

val_ds_grp = ImageMaskDataset(
        single_group[:20],
        image_dir,
        classes = ["group_of_trees"],
        transform = val_tf
)

train_loader_grp = DataLoader(train_ds_grp, batch_size = config.train.batch_size, shuffle = True)
val_loader_grp = DataLoader(val_ds_grp, batch_size = config.train.batch_size, shuffle = False)

trainer_grp = run_training(
        config,
        train_loader_grp,
        val_loader_grp,
        config.paths.models,
        model_name = "simple_cnn"
)


# %% [markdown]
# #### Step 7: Run group-class model on all images

# %%
# Load group model
vm = VersionManager(config.paths.models)
version_dir = vm.find_latest()
model_path = version_dir / "best_model.pth"

predictor_grp = Predictor(
        model_path,
        model_name = "simple_cnn",
        image_size = config.train.image_size
)

results_group = predictor_grp.run_on_folder(
        image_dir,
        transform = get_val_augmentations(config.train.image_size)
)

all_groups = results_group
p("all_groups", len(all_groups))

# %% [markdown]
# #### Step 8: Final combined training

# %%
combined_entries = single_individual + single_group

train_ds_comb = ImageMaskDataset(combined_entries, image_dir, transform = train_tf)
val_ds_comb = ImageMaskDataset(combined_entries[:40], image_dir, transform = val_tf)

train_loader_final = DataLoader(train_ds_comb, batch_size = config.train.batch_size, shuffle = True)
val_loader_final = DataLoader(val_ds_comb, batch_size = config.train.batch_size, shuffle = False)

trainer_final = run_training(
        config,
        train_loader_final,
        val_loader_final,
        config.paths.models,
        model_name = "simple_cnn"
)

