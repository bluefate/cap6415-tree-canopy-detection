# %% [markdown]
# # Notebook: 07 Data Analysis
# ### Purpose: Analyze dataset characteristics, class distributions, and image properties

# %% [markdown]
# #### Imports and setup

# %%
import os
import sys
from collections import defaultdict

import torch


sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../src"))
import cv2
import numpy as np
from torch.utils.data import DataLoader
from src.exploration.content import crop_bbox, crop_bbox_with_context, crop_mask_with_context, crop_mask, pad_to_size, \
    expand_crop_with_background
from src.exploration.class_explorer import explore_bboxes
from src.data.annotations import load_json_annotations
from src.data.augmentations import get_train_augmentations, get_val_augmentations
from src.data.loaders import ImageMaskDataset
from src.exploration.filters import cv2_apply_gaussian, cv2_apply_laplacian, cv2_apply_sobel
from src.exploration.kernels import apply_kernel_using_convolution, get_kernels
from src.exploration.visualize import show_side_by_side
from src.prediction.pipeline import Predictor
from src.training.engine import run_training
from src.utils.config import Config
from src.utils.helpers import init_notebook, p, t, c
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
# Class distribution

individual_entries = [e for e in entries if any(item.cls == "individual_tree" for item in e.items)]
group_entries = [e for e in entries if any(item.cls == "group_of_trees" for item in e.items)]
both_entries = [e for e in entries if
                any(item.cls == "individual_tree" for item in e.items) and
                any(item.cls == "group_of_trees" for item in e.items)]

p("Total images", len(entries))
p("Images with individual trees", len(individual_entries))
p("Images with tree groups", len(group_entries))
p("Images with both classes", len(both_entries))



# %%
t("Sample Visualizations")

# Show examples of each class type
sample_individual = individual_entries[0] if individual_entries else None
sample_group = group_entries[0] if group_entries else None
sample_both = both_entries[0] if both_entries else None

if sample_individual:
    p("Individual tree example:")
    explore_bboxes(sample_individual, image_dir)

if sample_group:
    p("Tree group example:")
    explore_bboxes(sample_group, image_dir)


# %%
t("Dataset Report")
# dataset_report(entries, image_dir)

# %%
# t("Build Context-Aware Crops for Individual Trees")
#
# # Process individual tree crops with natural context
# context_crops_ind = []
# context_masks_ind = []
# context_info = []
#
# # Group items by image for efficiency
# image_to_items = defaultdict(list)
# for e_idx, entry in enumerate(individual_entries):
#     for j_idx, item in enumerate(entry.items):
#         if item.cls == "individual_tree":
#             image_to_items[entry.image_path.name].append((entry, item, e_idx, j_idx))
#
# p("Unique images to process", len(image_to_items))
#
# for img_name, items_list in image_to_items.items():
#     p(f"Processing {img_name} with {len(items_list)} individual trees")
#
#     for entry, item, e_idx, j_idx in items_list:
#         try:
#             # Get crop with natural context
#             crop, context_bbox = crop_bbox_with_context(image_dir, entry, item, context_factor=1.5)
#             mask_crop = crop_mask_with_context(entry, item, context_bbox)
#
#             # Resize to standard size
#             crop_resized, mask_resized = resize_to_standard(crop, mask_crop, target_size=256)
#
#             context_crops_ind.append(crop_resized)
#             context_masks_ind.append(mask_resized)
#             context_info.append({
#                 'entry_idx': e_idx,
#                 'item_idx': j_idx,
#                 'image_name': img_name,
#                 'original_bbox': item.bbox,
#                 'context_bbox': context_bbox,
#                 'crop_shape': crop.shape,
#                 'resized_shape': crop_resized.shape
#             })
#
#         except Exception as e:
#             p(f"Failed to process {img_name} item {j_idx}: {e}", color1="red")
#
# p("Total context crops generated", len(context_crops_ind))
# p("All crops resized to", context_crops_ind[0].shape if context_crops_ind else "N/A")


# %%
# sample
if len(individual_entries) > 0:
    p("Quick demo of context-aware cropping on 3 samples:")

    demo_images = []
    demo_titles = []

    for i in range(min(3, len(individual_entries))):
        entry = individual_entries[i]
        if entry.items:
            item = entry.items[0]  # First item only

            try:
                # Show the concept
                crop, context_bbox = crop_bbox_with_context(image_dir, entry, item, context_factor = 1.5)
                mask_crop = crop_mask_with_context(entry, item, context_bbox)

                # Quick resize for display
                if crop.shape[0] > 200 or crop.shape[1] > 200:
                    crop = cv2.resize(crop, (200, 200))
                    mask_crop = cv2.resize(mask_crop, (200, 200), interpolation = cv2.INTER_NEAREST)

                # Create overlay
                mask_rgb = np.zeros_like(crop)
                mask_rgb[:, :, 1] = mask_crop * 255
                overlay = cv2.addWeighted(crop, 0.7, mask_rgb, 0.3, 0)

                demo_images.extend([crop, overlay])
                demo_titles.extend([f"Context Crop {i + 1}", f"Overlay {i + 1}"])

            except Exception as e:
                p(f"Demo failed for sample {i + 1}: {e}", color1 = "orange")

    if demo_images:
        show_side_by_side(
                *demo_images,
                titles = tuple(demo_titles),
                maxcolumns = 2
        )

# %% [markdown]
# #### Step 2: Extract each bbox from each single image and inspect

# %%
sample_entry = individual_entries[0]


# %% [markdown]
# ##### Collect all crops first and record their shapes

# %%
# Extract some individual bbox crops for inspection
individual_crops = []
individual_masks = []

individual_crops_count = 5
for e_idx, entry in enumerate(individual_entries[:5]):
    for j_idx, item in enumerate(entry.items):
        if item.cls == "individual_tree":
            crop = crop_bbox(image_dir, entry, item)
            mask_crop = crop_mask(entry, item)
            individual_crops.append(crop)
            individual_masks.append(mask_crop)

            if len(individual_crops) >= individual_crops_count:
                break
    if len(individual_crops) >= individual_crops_count:
        break

# Calculate dimensions
all_heights = [crop.shape[0] for crop in individual_crops]
all_widths = [crop.shape[1] for crop in individual_crops]
max_h = max(all_heights)
max_w = max(all_widths)

p("Individual crop count", len(individual_crops))
p("Max dimensions needed", f"{max_h}x{max_w}")

# Show first few crops
if individual_crops:
    triplets = []
    titles = []

    for i in range(min(individual_crops_count, len(individual_crops))):
        crop = individual_crops[i]
        mask_crop = individual_masks[i]

        # Create overlay
        overlay = crop.copy()
        overlay[mask_crop > 0] = [0, 255, 0]  # Green for trees
        overlay = cv2.addWeighted(crop, 0.7, overlay, 0.3, 0)

        triplets.extend([crop, mask_crop * 255, overlay])
        titles.extend([f"crop {i}", f"mask {i}", f"overlay {i}"])

    show_side_by_side(*triplets, titles = tuple(titles), maxcolumns = 6)

# %%
# crops = []
# #crops = [crop_bbox(sample_entry, item) for item in sample_entry.items]
# for item in sample_entry.items:
#     c = crop_bbox(sample_entry, item)
#     crops.append(c)
#
# heights = [c.shape[0] for c in crops]
# widths = [c.shape[1] for c in crops]
# max_h = max(heights)
# max_w = max(widths)

##### Build padded crops, padded masks, and overlays.

# padded_crops = []
# padded_masks = []
# padded_overlays = []
#
# for idx, item in enumerate(sample_entry.items):
#     crop = crops[idx]
#     cmask = crop_mask(sample_entry, item)
#
#     crop_p = pad_to_size(crop, max_h, max_w)
#     mask_p = pad_to_size(cmask, max_h, max_w)
#
#     overlay = cv2.addWeighted(
#             crop_p.astype(np.uint8),
#             0.6,
#             np.dstack([mask_p * 255, np.zeros_like(mask_p), np.zeros_like(mask_p)]).astype(np.uint8),
#             0.4,
#             0
#     )
#
#     padded_crops.append(crop_p)
#     padded_masks.append(mask_p)
#     padded_overlays.append(overlay)


# num_show = 9
# indexes = list(range(min(num_show, len(padded_crops))))
#
# triplets, titles = [], []
#
# for idx in indexes:
#     triplets.extend(
#             [
#                 padded_crops[idx],
#                 padded_masks[idx],
#                 padded_overlays[idx]
#             ]
#     )
#     titles.extend(
#             [
#                 f"crop {idx}",
#                 f"mask {idx}",
#                 f"overlay {idx}"
#             ]
#     )
#
# show_side_by_side(
#         *triplets,
#         titles = tuple(titles),
#         maxcolumns = 9
# )

##### FOR TRAINING

##### Prepare all individual crops for training

# # Step 1: Group items by image to minimize disk loads
# image_to_items = defaultdict(list)
# for e_idx, entry in enumerate(individual_entries):
#     for j_idx, item in enumerate(entry.items):
#         if item.cls == "individual_tree":
#             image_to_items[entry.image_path.name].append((entry, item, e_idx, j_idx))
#
# p("Unique images to process", len(image_to_items))

# %%
# # Step 2: Process each image once
# background_crops_ind = []
# background_masks_ind = []
# idx_entry_map = []
# idx_item_map = []
#
# for img_name, items_list in image_to_items.items():
#     # Load image ONCE per unique image
#     entry = items_list[0][0]
#     img_path = image_dir / img_name
#
#     img = cv2.imread(str(img_path))
#     if img is None:
#         p("Warning: Failed to load", img_name, color1 = "red")
#         continue
#
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#     # Collect ALL crops and masks for THIS image
#     local_crops = []
#     local_masks = []
#     local_entries = []
#
#     for entry, item, e_idx, j_idx in items_list:
#         # Crop directly from loaded image
#         x1, y1, x2, y2 = item.bbox
#         crop = img[y1:y2, x1:x2]
#
#         # Create mask
#         mask = np.zeros((entry.height, entry.width), dtype = np.uint8)
#         seg = item.segmentation
#         if seg and len(seg) >= 4:
#             poly = np.array(seg, dtype = np.int32).reshape(-1, 2)
#             cv2.fillPoly(mask, [poly], 1)
#         mask_crop = mask[y1:y2, x1:x2]
#
#         local_crops.append(crop)
#         local_masks.append(mask_crop)
#         local_entries.append((e_idx, j_idx))
#
#     # Calculate max dimensions for THIS IMAGE ONLY
#     local_heights = [c.shape[0] for c in local_crops]
#     local_widths = [c.shape[1] for c in local_crops]
#     local_max_h = max(local_heights)
#     local_max_w = max(local_widths)
#
#     # Pad to local max (not global)
#     for crop, mask, (e_idx, j_idx) in zip(local_crops, local_masks, local_entries):
#         background_crops_ind.append(pad_to_size(crop, local_max_h, local_max_w))
#         background_masks_ind.append(pad_to_size(mask, local_max_h, local_max_w))
#         idx_entry_map.append(e_idx)
#         idx_item_map.append(j_idx)
#
# p("total padded crops", len(background_crops_ind))
# p("total padded masks", len(background_masks_ind))
# t("Extraction complete!")

# # Step 2: Process each image once
# background_crops_ind = []
# background_masks_ind = []
# idx_entry_map = []
# idx_item_map = []
#
# for img_name, items_list in image_to_items.items():
#     # Load image ONCE per unique image
#     entry = items_list[0][0]
#     img_path = image_dir / img_name
#
#     img = cv2.imread(str(img_path))
#     if img is None:
#         p("Warning: Failed to load", img_name, color1 = "red")
#         continue
#
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#     # Collect ALL crops and masks for THIS image
#     local_crops = []
#     local_masks = []
#     local_bboxes = []
#     local_entries = []
#
#     for entry, item, e_idx, j_idx in items_list:
#         # Crop directly from loaded image
#         x1, y1, x2, y2 = item.bbox
#         crop = img[y1:y2, x1:x2]
#
#         # Create mask
#         mask = np.zeros((entry.height, entry.width), dtype = np.uint8)
#         seg = item.segmentation
#         if seg and len(seg) >= 4:
#             poly = np.array(seg, dtype = np.int32).reshape(-1, 2)
#             cv2.fillPoly(mask, [poly], 1)
#         mask_crop = mask[y1:y2, x1:x2]
#
#         local_crops.append(crop)
#         local_masks.append(mask_crop)
#         local_bboxes.append(item.bbox)
#         local_entries.append((e_idx, j_idx))
#
#     # Calculate max dimensions for THIS IMAGE ONLY
#     local_heights = [c.shape[0] for c in local_crops]
#     local_widths = [c.shape[1] for c in local_crops]
#     local_max_h = max(local_heights)
#     local_max_w = max(local_widths)
#
#     # Expand to include background instead of padding
#     for crop, mask_crop, bbox, (e_idx, j_idx) in zip(local_crops, local_masks, local_bboxes, local_entries):
#         expanded_img, expanded_mask = expand_crop_with_background(
#                 img, mask, bbox, local_max_h, local_max_w
#         )
#
#         background_crops_ind.append(expanded_img)
#         background_masks_ind.append(expanded_mask)
#         idx_entry_map.append(e_idx)
#         idx_item_map.append(j_idx)
#
# p("total background crops", len(background_crops_ind))
# p("total background masks", len(background_masks_ind))
# t("Background expansion complete!")

##### Sample padded and original images

# %%
# # Sort by area (for visualization)
# sizes = [(i, background_crops_ind[i].shape) for i in range(len(background_crops_ind))]
# sizes_sorted = sorted(sizes, key = lambda x: x[1][0] * x[1][1], reverse = True)
# #sizes_sorted = sizes_sorted + sorted(sizes, key=lambda x: x[1][0]*x[1][1])
#
# #if using vector instead
# # sizes = [(i, raw_crops_ind[i].shape) for i in range(len(raw_crops_ind))]
# # sizes_sorted = sorted(sizes, key = lambda x: x[1][0] * x[1][1])
# # #sizes_sorted = sorted(sizes, key = lambda x: x[1][0] * x[1][1], reverse = True)

# %%
# # Visualize top 5 largest crops with their source images
# for i in range(min(5, len(sizes_sorted))):
#     idx = sizes_sorted[i][0]
#
#     entry = individual_entries[idx_entry_map[idx]]
#
#     # Load full image for display
#     full_img = cv2.imread(str(image_dir / entry.image_path.name))
#     full_img = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)
#
#     # Get the already-padded crop and mask
#     padded_crop = background_crops_ind[idx]
#     padded_mask = background_masks_ind[idx]
#
#     overlay = cv2.addWeighted(
#             padded_crop.astype(np.uint8),
#             0.6,
#             np.dstack([padded_mask * 255, np.zeros_like(padded_mask), np.zeros_like(padded_mask)]).astype(np.uint8),
#             0.4,
#             0
#     )
#
#     show_side_by_side(
#             full_img, padded_crop, overlay,
#             titles = (f"orig {idx}", f"padded {idx}", f"overlay {idx}")
#     )

# %%
# # grid of crops
# idxs = [x[0] for x in sizes_sorted[:num_show]]
#
# # tiplets, titles = [], []
# # for i in idxs:
# #     crop_p = background_crops_ind[i]
# #     mask_p = background_masks_ind[i]
# #     overlay = cv2.addWeighted(
# #             crop_p.astype(np.uint8),
# #             0.6,
# #             np.dstack([mask_p * 255, np.zeros_like(mask_p), np.zeros_like(mask_p)]).astype(np.uint8),
# #             0.4,
# #             0
# #     )
# #
# #     triplets.extend([crop_p, mask_p, overlay])
# #     titles.extend([f"crop {i}", f"mask {i}", f"overlay {i}"])
# #
# # show_side_by_side(*triplets, titles = tuple(titles), maxcolumns = 12)
#
#
# if len(background_crops_ind) >= 5:
#     triplets = []
#     titles = []
#
#     for i in range(5):
#         crop_img = background_crops_ind[i]
#         crop_mask = background_masks_ind[i]
#
#         # Create overlay
#         overlay = crop_img.copy()
#         overlay[crop_mask > 0] = [0, 255, 0]  # Green for trees
#         overlay = cv2.addWeighted(crop_img, 0.7, overlay, 0.3, 0)
#
#         triplets.extend([crop_img, crop_mask * 255, overlay])
#         titles.extend([f"bg crop {i}", f"mask {i}", f"overlay {i}"])
#
#     show_side_by_side(*triplets, titles = tuple(titles), maxcolumns = 12)
#


# %% [markdown]
# #### Step 3: Apply filters and kernel exploration on crops

# %% [markdown]
# ##### Direct crop

# %%
crop = crop_bbox(image_dir, sample_entry, sample_entry.items[0])
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



# %% [markdown]
# ##### Background Expanded Crop

# %%
# Create full image mask for background expansion
full_mask = np.zeros((sample_entry.height, sample_entry.width), dtype = np.uint8)
sample_seg = sample_entry.items[0].segmentation
if sample_seg and len(sample_seg) >= 4:
    poly = np.array(sample_seg, dtype = np.int32).reshape(-1, 2)
    cv2.fillPoly(full_mask, [poly], 1)

# Load full image
img_path = image_dir / sample_entry.image_path.name
full_img = cv2.imread(str(img_path))
full_img = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)

# Expand with background
expanded, expanded_mask = expand_crop_with_background(

        full_img, full_mask, sample_entry.items[0].bbox, max_h, max_w
)

ga_p = cv2_apply_gaussian(expanded)
so_p = cv2_apply_sobel(expanded)
la_p = cv2_apply_laplacian(expanded)

show_side_by_side(
        expanded,
        ga_p,
        so_p,
        la_p,
        titles = ("expanded orig", "expanded gauss", "expanded sobel", "expanded lap"),
        maxcolumns = 4
)




# %%
kernels = get_kernels()
lap_k = kernels["Laplacian_3x3"]
gray_expanded = cv2.cvtColor(expanded, cv2.COLOR_RGB2GRAY)
lap_resp = apply_kernel_using_convolution(gray_expanded, lap_k)

show_side_by_side(gray_expanded, lap_resp, titles = ("gray expanded", "lap3x3"))


# %% [markdown]
# ### FOR TRAINING
# ##### Prepare all individual crops for training

# %% [markdown]
# #### Step 1: Group items by image to minimize disk loads

# %%
image_to_items = defaultdict(list)
for e_idx, entry in enumerate(individual_entries):
    for j_idx, item in enumerate(entry.items):
        if item.cls == "individual_tree":
            image_to_items[entry.image_path.name].append((entry, item, e_idx, j_idx))

p("Unique images to process", len(image_to_items))


# %% [markdown]
# #### Step 2: Process each image once

# %%
processed = 0
max_demo_samples = 5

# %%
# background_crops_ind = []
# background_masks_ind = []
# idx_entry_map = []
# idx_item_map = []
#
# # Just process 5 samples to demonstrate the concept

#
# for img_name, items_list in image_to_items.items():
#     if processed >= max_demo_samples:
#         break
#
#     # Load image
#     entry = items_list[0][0]
#     img_path = image_dir / img_name
#     img = cv2.imread(str(img_path))
#     if img is None:
#         p("Warning: Failed to load", img_name, color1 = "red")
#         continue
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#     # Create full image mask once
#     full_mask = np.zeros((entry.height, entry.width), dtype = np.uint8)
#
#     # Collect ALL crops and masks for THIS image
#     local_crops = []
#     local_masks = []
#     local_bboxes = []
#     local_entries = []
#
#     try:
#         for entry, item, e_idx, j_idx in items_list:
#             # Crop directly from loaded image
#             x1, y1, x2, y2 = item.bbox
#             crop = img[y1:y2, x1:x2]
#
#             # Create mask for this specific item
#             item_mask = np.zeros((entry.height, entry.width), dtype = np.uint8)
#             seg = item.segmentation
#             if seg and len(seg) >= 4:
#                 poly = np.array(seg, dtype = np.int32).reshape(-1, 2)
#                 cv2.fillPoly(item_mask, [poly], 1)
#             mask_crop = item_mask[y1:y2, x1:x2]
#
#             local_crops.append(crop)
#             local_masks.append(mask_crop)
#             local_bboxes.append(item.bbox)
#             local_entries.append((e_idx, j_idx))
#
#             # Calculate max dimensions for this iamge only
#             local_heights = [c.shape[0] for c in local_crops]
#             local_widths = [c.shape[1] for c in local_crops]
#             local_max_h = max(local_heights)
#             local_max_w = max(local_widths)
#
#             # Expand to include background instead of padding
#             for crop, mask_crop, bbox, (e_idx, j_idx) in zip(local_crops, local_masks, local_bboxes, local_entries):
#                 # Create item-specific mask for expansion
#                 item_full_mask = np.zeros((entry.height, entry.width), dtype = np.uint8)
#                 item_full_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = mask_crop
#
#                 expanded_img, expanded_mask = expand_crop_with_background(
#                         img, item_full_mask, bbox, local_max_h, local_max_w
#                 )
#
#                 background_crops_ind.append(expanded_img)
#                 background_masks_ind.append(expanded_mask)
#                 idx_entry_map.append(e_idx)
#                 idx_item_map.append(j_idx)
#                 processed += 1
#
#     except Exception as e:
#         p(f"Failed to process {img_name}: {e}", color1 = "orange")
# p("total background crops", len(background_crops_ind))
# p("total background masks", len(background_masks_ind))
# t("Background expansion complete!")


# %%
# %%
t("Background Expansion Demo (Sample Only)")

# Just process samples to demonstrate the concept
background_crops_ind = []
background_masks_ind = []

for img_name, items_list in image_to_items.items():
    if processed >= max_demo_samples:
        break

    # Load image
    entry = items_list[0][0]
    img_path = image_dir / img_name
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process just one item from this image
    entry, item, e_idx, j_idx = items_list[0]

    try:
        # Get crop
        x1, y1, x2, y2 = item.bbox
        crop = img[y1:y2, x1:x2]

        # Create mask
        item_mask = np.zeros((entry.height, entry.width), dtype = np.uint8)
        seg = item.segmentation
        if seg and len(seg) >= 4:
            poly = np.array(seg, dtype = np.int32).reshape(-1, 2)
            cv2.fillPoly(item_mask, [poly], 1)

        # Use fixed target size instead of variable max dimensions
        target_h, target_w = 256, 256

        expanded_img, expanded_mask = expand_crop_with_background(
                img, item_mask, item.bbox, target_h, target_w
        )

        background_crops_ind.append(expanded_img)
        background_masks_ind.append(expanded_mask)
        processed += 1

    except Exception as e:
        p(f"Failed to process {img_name}: {e}", color1 = "orange")

p(f"{max_demo_samples} Demo background crops processed", len(background_crops_ind))


# %%
# Visualize background-expanded crops
if len(background_crops_ind) >= max_demo_samples:
    triplets = []
    titles = []

    for i in range(max_demo_samples):
        crop_img = background_crops_ind[i]
        crop_mask = background_masks_ind[i]

        # Create overlay
        overlay = crop_img.copy()
        overlay[crop_mask > 0] = [0, 255, 0]  # Green for trees
        overlay = cv2.addWeighted(crop_img, 0.7, overlay, 0.3, 0)

        triplets.extend([crop_img, crop_mask * 255, overlay])
        titles.extend([f"bg crop {i}", f"mask {i}", f"overlay {i}"])

    show_side_by_side(*triplets, titles = tuple(titles), maxcolumns = 6)


# %% [markdown]
# #### Step 4: Train model on only individual class

# %%
train_tf = get_train_augmentations(config.train.image_size)
val_tf = get_val_augmentations(config.train.image_size)

train_ds_ind = ImageMaskDataset(
        individual_entries,
        image_dir,
        classes = ["individual_tree"],
        transform = train_tf
)

val_ds_ind = ImageMaskDataset(
        individual_entries[:20],
        image_dir,
        classes = ["individual_tree"],
        transform = val_tf
)

train_loader_ind = DataLoader(train_ds_ind, batch_size = config.train.batch_size, shuffle = True)
val_loader_ind = DataLoader(val_ds_ind, batch_size = config.train.batch_size, shuffle = False)

# CREATE UNIQUE PATH for individual tree model
individual_model_path = config.paths.models / "07" / "individual_tree_only" / "simple_cnn" / "rgb"

trainer_ind = run_training(
        config,
        train_loader_ind,
        val_loader_ind,
        individual_model_path,  # Use unique path
        model_name = "simple_cnn"
)

# %% [markdown]
# #### Step 5: Train model on only group class

# %%
# Load individual tree model

individual_vm = VersionManager(individual_model_path)
individual_version_dir = individual_vm.find_latest()
individual_model_file = individual_version_dir / "best_model.pth"

predictor_ind = Predictor(
        individual_model_file,
        model_name = "simple_cnn",
        image_size = config.train.image_size
)

results_single = predictor_ind.run_on_folder(
        image_dir,
        transform = get_val_augmentations(config.train.image_size)
)

p("Individual tree predictions", len(results_single))

# %% [markdown]
# #### Step 6: Run single-class model on all images and save predictions

# %%

train_ds_grp = ImageMaskDataset(
        group_entries,
        image_dir,
        classes = ["group_of_trees"],
        transform = train_tf
)

val_ds_grp = ImageMaskDataset(
        group_entries[:20],
        image_dir,
        classes = ["group_of_trees"],
        transform = val_tf
)

train_loader_grp = DataLoader(train_ds_grp, batch_size = config.train.batch_size, shuffle = True)
val_loader_grp = DataLoader(val_ds_grp, batch_size = config.train.batch_size, shuffle = False)

# CREATE UNIQUE PATH for group tree model
group_model_path = config.paths.models / "07" / "group_trees_only" / "simple_cnn" / "rgb"

trainer_grp = run_training(
        config,
        train_loader_grp,
        val_loader_grp,
        group_model_path,  # Use unique path
        model_name = "simple_cnn"
)

# %% [markdown]
# #### Step 7: Run group-class model on all images

# %%
# Load group tree model
group_vm = VersionManager(group_model_path)
group_version_dir = group_vm.find_latest()
group_model_file = group_version_dir / "best_model.pth"

predictor_grp = Predictor(
        group_model_file,
        model_name = "simple_cnn",
        image_size = config.train.image_size
)

results_group = predictor_grp.run_on_folder(
        image_dir,
        transform = get_val_augmentations(config.train.image_size)
)

all_groups = results_group
p("Group tree predictions", len(all_groups))


# %% [markdown]
# #### Step 8: Final combined training

# %%

import copy
import time


combined_entries = individual_entries + group_entries

train_ds_comb = ImageMaskDataset(combined_entries, image_dir, transform = train_tf)
val_ds_comb = ImageMaskDataset(combined_entries[:40], image_dir, transform = val_tf)

train_loader_final = DataLoader(train_ds_comb, batch_size = config.train.batch_size, shuffle = True)
val_loader_final = DataLoader(val_ds_comb, batch_size = config.train.batch_size, shuffle = False)

# CREATE UNIQUE PATH for combined model
combined_model_path = config.paths.models / "07" / "combined_classes" / "simple_cnn" / "rgb"

model_name, mode, in_channels = "simple_cnn", "rgb", "3"

exp_config = copy.deepcopy(config)
exp_config.extra['force_retrain'] = time.time()

trainer_final = run_training(
        exp_config,
        train_loader_final,
        val_loader_final,
        combined_model_path,  # Use unique path
        model_name = "simple_cnn"
)

# %%
if trainer_final is not None:
    t("Final Combined Trainer Metrics")

    # Load checkpoint to get metrics
    checkpoint_path = trainer_final.paths["checkpoint"]
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location = "cpu")

        p("")

        p("Model", "simple_cnn (combined)", color1 = c.GREEN)
        p("Best Val Loss", f"{ckpt.get('best_val_loss', 0):.6f}", color1 = c.GREEN)
        p("Final Epoch", ckpt.get('final_epoch', ckpt.get('epoch', 0)), color1 = c.GREEN)
        p("Train Loss", f"{ckpt.get('train_loss', 0):.6f}", color1 = c.GREEN)
        p("")

        p("Validation Metrics", color1 = c.CYAN, bold = True)
        p("Val Loss", f"{ckpt.get('val_loss', 0):.6f}", color1 = c.BLUE)
        p("Val IoU", f"{ckpt.get('val_iou', ckpt.get('iou', 0)):.4f}", color1 = c.BLUE)
        p("Val Accuracy", f"{ckpt.get('val_accuracy', ckpt.get('accuracy', 0)):.4f}", color1 = c.BLUE)
        p("Individual Tree IoU", f"{ckpt.get('val_iou_individual', 0):.4f}", color1 = c.BLUE)
        p("Group Tree IoU", f"{ckpt.get('val_iou_group', 0):.4f}", color1 = c.BLUE)
        p("")

        p("Version", trainer_final.version_dir.name, color1 = c.CYAN)
        p("Path", str(trainer_final.version_dir), color1 = c.BLUE)
        p("")

    else:
        p("ERROR", "No checkpoint found!", color1 = c.RED)
else:
    p("WARNING", "Trainer is None - training was skipped", color1 = c.ORANGE)

# %%
from prediction.validation import analyze_validation_metrics


if trainer_final is not None:
    checkpoint_path = trainer_final.paths["checkpoint"]
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location = "cpu")

        analyze_validation_metrics(
                val_loss = ckpt.get("val_loss", 0),
                iou = ckpt.get("val_iou", ckpt.get("iou", 0)),
                accuracy = ckpt.get("val_accuracy", ckpt.get("accuracy", 0)),
                individual_tree_iou = ckpt.get("val_iou_individual", 0),
                group_tree_iou = ckpt.get("val_iou_group", 0),
                model_name = "Group Tree Model"
        )


# %%
