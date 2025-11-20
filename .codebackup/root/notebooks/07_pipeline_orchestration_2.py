# %% [markdown]
# # Notebook: 07 Pipeline Orchestration
# ### Imports and setup

# %% [markdown]
# #### Imports and setup

# %%

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.annotations import load_json_annotations
from src.data.augmentations import get_train_augmentations, get_val_augmentations
from src.data.loaders import ImageMaskDataset
from src.exploration.filters import cv2_apply_gaussian, cv2_apply_laplacian, cv2_apply_sobel
from src.exploration.kernels import apply_kernel_using_convolution, get_kernels
from src.exploration.visualize import show_side_by_side
from src.prediction.pipeline import Predictor
from src.training.engine import run_training
from src.utils.config import Config
from src.utils.helpers import init_notebook, p
from src.utils.versioning import VersionManager


config = Config.load()
init_notebook(config.train.seed)

entries = load_json_annotations(config.paths.annotations)
image_dir = config.paths.train_images


# %%
def crop_mask( entry, item ):
    mask = np.zeros((entry.height, entry.width), dtype = np.uint8)
    seg = item.segmentation
    if seg and len(seg) >= 4:
        poly = np.array(seg, dtype = np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [poly], 1)
    x1, y1, x2, y2 = item.bbox
    return mask[y1:y2, x1:x2]


def crop_bbox( entry, item ):
    img = cv2.imread(str(image_dir / entry.image_path.name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x1, y1, x2, y2 = item.bbox
    return img[y1:y2, x1:x2]


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
# #### Step 1: Identify single class and group images

# %%
single_individual = [
    e for e in entries
    if any(item.cls == "individual_tree" for item in e.items)
]

single_group = [
    e for e in entries
    if any(item.cls == "group_of_trees" for item in e.items)
]

mixed = [
    e for e in entries
    if any(i.cls == "individual_tree" for i in e.items)
       and any(i.cls == "group_of_trees" for i in e.items)
]

p("single_individual", len(single_individual))
p("single_group", len(single_group))
p("mixed", len(mixed))


# single_individual = [e for e in entries if all(item.cls == "individual_tree" for item in e.items)]
# # single_group = [e for e in entries if all(item.cls == "group_of_trees" for item in e.items)]
#
# single_group = []
# for entry in entries:
#     cls_set = { item.cls for item in entry.items }
#     if "group_of_trees" in cls_set:
#         single_group.append(entry)
#
# mixed = [e for e in entries if
#          any(i.cls == "individual_tree" for i in e.items) and any(i.cls == "group_of_trees" for i in e.items)]



# %% [markdown]
# #### Step 2: Extract each bbox from each single image and inspect

# %%
sample_entry = single_individual[0]


# %%

# %% [markdown]
# ##### Collect all crops first and record their shapes

# %%
# for item in sample_entry.items:
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



# %%
crops = [crop_bbox(sample_entry, item) for item in sample_entry.items]
heights = [c.shape[0] for c in crops]
widths = [c.shape[1] for c in crops]
max_h, max_w = max(heights), max(widths)

# %% [markdown]
# ##### Build padded crops, padded masks, and overlays.

# %%

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
    triplets.extend([padded_crops[idx], padded_masks[idx], padded_overlays[idx]])
    titles.extend([f"crop {idx}", f"mask {idx}", f"overlay {idx}"])

show_side_by_side(
        *triplets,
        titles = tuple(titles),
        maxcolumns = 9
)


# %% [markdown]
# ##### FOR TRAINING

# %% [markdown]
# ##### all images

# %%
# raw_crops_ind = []
# raw_masks_ind = []
# idx_entry_map = []
# idx_item_map = []
#
# for e_idx, entry in enumerate(single_individual):
#     for j_idx, item in enumerate(entry.items):
#         if item.cls == "individual_tree":
#             raw_crops_ind.append(crop_bbox(entry, item))
#             raw_masks_ind.append(crop_mask(entry, item))
#             idx_entry_map.append(e_idx)
#             idx_item_map.append(j_idx)
#
# # sampel by largest descending
# # indexed = [(i, raw_crops_ind[i].shape[0]) for i in range(len(raw_crops_ind))]
# # indexed = sorted(indexed, key=lambda x: x[1], reverse=True)
#
# heights = [c.shape[0] for c in raw_crops_ind]
# widths = [c.shape[1] for c in raw_crops_ind]
#
# max_h = max(heights)
# max_w = max(widths)
#
# padded_crops_ind = []
# padded_masks_ind = []
#
# for c, m in zip(raw_crops_ind, raw_masks_ind):
#     padded_crops_ind.append(pad_to_size(c, max_h, max_w))
#     padded_masks_ind.append(pad_to_size(m, max_h, max_w))

raw_crops_ind = []
raw_masks_ind = []
idx_entry_map = []
idx_item_map = []

for e_idx, entry in enumerate(single_individual):
    for j_idx, item in enumerate(entry.items):
        if item.cls == "individual_tree":
            raw_crops_ind.append(crop_bbox(entry, item))
            raw_masks_ind.append(crop_mask(entry, item))
            idx_entry_map.append(e_idx)
            idx_item_map.append(j_idx)

heights = [c.shape[0] for c in raw_crops_ind]
widths = [c.shape[1] for c in raw_crops_ind]
max_h, max_w = max(heights), max(widths)

padded_crops_ind = [pad_to_size(c, max_h, max_w) for c in raw_crops_ind]
padded_masks_ind = [pad_to_size(m, max_h, max_w) for m in raw_masks_ind]


# %%
p("total raw crops", len(raw_crops_ind))
p("total raw masks", len(raw_masks_ind))
p("total padded crops", len(padded_crops_ind))
p("total padded masks", len(padded_masks_ind))
p("max_h", max_h)
p("max_w", max_w)


# %%

class PaddedCropDataset(torch.utils.data.Dataset):

    def __init__( self, images, masks, transform = None ):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__( self ):
        return len(self.images)

    def __getitem__( self, idx ):
        img = self.images[idx]
        msk = self.masks[idx]

        if self.transform:
            out = self.transform(image = img, mask = msk)
            img, msk = out["image"], out["mask"]

        if isinstance(img, torch.Tensor):
            img_t = img.float()
        else:
            img_t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        if isinstance(msk, torch.Tensor):
            msk_t = msk.float()
            if msk_t.ndim == 2:
                msk_t = msk_t.unsqueeze(0)
        else:
            msk = msk.astype("float32")
            if msk.ndim == 2:
                msk = msk[None, ...]
            msk_t = torch.from_numpy(msk)

        return img_t, msk_t


# %% [markdown]
# ##### sample padded and orginal images

# %%

# %%
# sizes = [(i, raw_crops_ind[i].shape) for i in range(len(raw_crops_ind))]
# sizes_sorted = sorted(sizes, key = lambda x: x[1][0] * x[1][1])
# #sizes_sorted = sorted(sizes, key = lambda x: x[1][0] * x[1][1], reverse = True)


sizes = [(i, padded_crops_ind[i].shape) for i in range(len(padded_crops_ind))]
sizes_sorted = sorted(sizes, key = lambda x: x[1][0] * x[1][1], reverse = True)
#sizes_sorted = sizes_sorted + sorted(sizes, key=lambda x: x[1][0]*x[1][1])


for i in range(min(5, len(sizes_sorted))):
    idx = sizes_sorted[i][0]

    entry = single_individual[idx_entry_map[idx]]
    full_img = cv2.imread(str(image_dir / entry.image_path.name))
    full_img = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)

    crop = raw_crops_ind[idx]
    mask = raw_masks_ind[idx]

    local_crops = [crop_bbox(entry, it) for it in entry.items if it.cls == "individual_tree"]
    local_heights = [c.shape[0] for c in local_crops]
    local_widths = [c.shape[1] for c in local_crops]
    lh, lw = max(local_heights), max(local_widths)

    padded_crop = pad_to_size(crop, lh, lw)
    padded_mask = pad_to_size(mask, lh, lw)

    overlay = cv2.addWeighted(
            padded_crop.astype(np.uint8),
            0.6,
            np.dstack([padded_mask * 255, np.zeros_like(padded_mask), np.zeros_like(padded_mask)]).astype(np.uint8),
            0.4,
            0
    )

    show_side_by_side(
            full_img, crop, padded_crop, overlay,
            titles = (f"orig {idx}", f"crop {idx}", f"padded {idx}", f"overlay {idx}")
    )

# %%


idxs = [x[0] for x in sizes_sorted[:num_show]]

#idxs = list(range(min(num_show, len(padded_crops_ind))))

# # get heights
# sizes = [(i, padded_crops_ind[i].shape[1]) for i in range(len(padded_crops_ind))]
# # sort by height descending
# sizes = sorted(sizes, key=lambda x: x[1], reverse=True)
# # take top four indexes
# idxs = [x[0] for x in sizes[:num_show]]

# sizes = [(i, padded_crops_ind[i].shape) for i in range(len(padded_crops_ind))]
# sizes_sorted = sorted(sizes, key = lambda x: x[1][0] * x[1][1], reverse = True)
# idxs = [x[0] for x in sizes_sorted[:num_show]]

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


# %% [markdown]
# #### Step 3: Apply filters and kernel exploration on crops

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

# padded_crops_ind = []
# padded_masks_ind = []
# idx_entry_map = []
# idx_item_map = []

# for e_idx, entry in enumerate(single_individual):
#     local_crops = []
#     local_masks = []
#     local_item_ids = []
#
#     for j_idx, item in enumerate(entry.items):
#         if item.cls != "individual_tree":
#             continue
#         c = crop_bbox(entry, item)
#         m = crop_mask(entry, item)
#         local_crops.append(c)
#         local_masks.append(m)
#         local_item_ids.append(j_idx)
#
#     if len(local_crops) == 0:
#         continue
#
#     heights = [c.shape[0] for c in local_crops]
#     widths = [c.shape[1] for c in local_crops]
#
#     max_h = max(heights)
#     max_w = max(widths)
#
#     for c, m in zip(local_crops, local_masks):
#         padded_crops_ind.append(pad_to_size(c, max_h, max_w))
#         padded_masks_ind.append(pad_to_size(m, max_h, max_w))
#         idx_entry_map.append(e_idx)
#
# train_ds_ind = PaddedCropDataset(
#         padded_crops_ind,
#         padded_masks_ind,
#         transform = train_tf
# )
#


# %%

train_tf = get_train_augmentations(config.train.image_size)
val_tf = get_val_augmentations(config.train.image_size)

train_ds_ind = ImageMaskDataset(single_individual, image_dir, classes = ["individual_tree"], transform = train_tf)
val_ds_ind = ImageMaskDataset(single_individual[:20], image_dir, classes = ["individual_tree"], transform = val_tf)

train_loader_ind = DataLoader(train_ds_ind, batch_size = config.train.batch_size, shuffle = True)
val_loader_ind = DataLoader(val_ds_ind, batch_size = config.train.batch_size, shuffle = False)

trainer_ind = run_training(
        config,
        train_loader_ind,
        val_loader_ind,
        config.paths.models,
        model_name = "simple_cnn"
)



# #----
# train_tf = get_train_augmentations(config.train.image_size)
# val_tf = get_val_augmentations(config.train.image_size)
#
# train_ds_ind = PaddedCropDataset(
#         padded_crops_ind,
#         padded_masks_ind,
#         transform = train_tf
# )
#
# val_ds_ind = PaddedCropDataset(
#         padded_crops_ind[:20],
#         padded_masks_ind[:20],
#         transform = val_tf
# )
#
# from torch.utils.data import DataLoader
#
#
# train_loader = DataLoader(
#         train_ds_ind,
#         batch_size = config.train.batch_size,
#         shuffle = True
# )
#
# val_loader = DataLoader(
#         val_ds_ind,
#         batch_size = config.train.batch_size,
#         shuffle = False
# )
#
# trainer = run_training(
#         config,
#         train_loader,
#         val_loader,
#         config.paths.models,
#         model_name = "simple_cnn"
# )



# %% [markdown]
# #### Step 5: Train model on only group class

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
# #### Step 6: Run single-class model on all images and save predictions

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

# %% [markdown]
# #### Step 8: Final combined training

# %%
combined_entries = single_individual + single_group + mixed

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


# %%

# %%
