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
import torch
from src.utils.config import load_config
from src.utils.helpers import format_number, init_this_notebook, p

cfg = load_config()
# inject_config_vars(config)

cfg.show()

init_this_notebook(cfg.SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# %%
import os

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ---------------------------------------------------------------
# 2. DATASET
# ---------------------------------------------------------------
class SegmentationDataset(Dataset):
    # Custom PyTorch dataset for image segmentation
    def __init__(self, img_dir, mask_dir=None, transform=None):
        self.img_dir = img_dir  # Folder containing input images
        self.mask_dir = mask_dir  # Folder containing segmentation masks
        self.transform = transform  # Albumentations transform pipeline
        self.img_files = sorted(os.listdir(img_dir))  # List of all image filenames

    def __len__(self):
        # Return total number of images
        return len(self.img_files)

    def __getitem__(self, idx):
        # Load one image (and its mask if available)
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # Read image with OpenCV (loads as BGR)
        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        # Convert BGR → RGB for compatibility with PyTorch and Albumentations
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = None
        if self.mask_dir and os.path.exists(os.path.join(self.mask_dir, img_name)):
            # Build mask path using same filename as image
            mask_path = os.path.join(self.mask_dir, img_name)
            if not os.path.exists(mask_path):
                raise RuntimeError(f"Missing mask: {mask_path}")

            # Read mask in grayscale (1 channel)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Failed to read mask: {mask_path}")

            # Normalize to [0,1] and expand to shape (H, W, 1)
            mask = mask / 255.0
            mask = np.expand_dims(mask, axis=-1)

        # Apply augmentations if defined
        if self.transform:
            if mask is not None:
                # Apply same transform to image and mask together
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            else:
                # Apply only on image if no mask
                image = self.transform(image=image)["image"]

        # Return tuple (image, mask) if mask exists, else image only
        return (image, mask) if mask is not None else image


# ---------------------------------------------------------------
# Data augmentations (Albumentations)
# ---------------------------------------------------------------
train_transform = A.Compose(
    [
        A.Resize(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),  # Resize to fixed input size
        A.HorizontalFlip(p=0.5),  # Random horizontal flip
        A.VerticalFlip(p=0.5),  # Random vertical flip
        A.RandomBrightnessContrast(p=0.2),  # Slight brightness/contrast change
        ToTensorV2(),  # Convert to PyTorch tensor
    ]
)

# Validation transform: only resize + tensor conversion
val_transform = A.Compose([A.Resize(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), ToTensorV2()])

# ---------------------------------------------------------------
# Dataset instances for training and validation
# ---------------------------------------------------------------
train_ds = SegmentationDataset(
    cfg.PATHS_TRAIN_IMAGES, cfg.PATHS_TRAIN_MASKS, transform=train_transform
)
val_ds = SegmentationDataset(
    cfg.PATHS_EVAL_IMAGES, cfg.PATHS_EVAL_MASKS, transform=val_transform
)

# ---------------------------------------------------------------
# DataLoaders (for batching and parallel data loading)
# ---------------------------------------------------------------
train_loader = DataLoader(
    train_ds,
    batch_size=cfg.BATCH_SIZE,  # Number of samples per batch
    shuffle=True,  # Shuffle data every epoch
    num_workers=cfg.NUM_WORKERS,  # Number of parallel data-loading threads
)
val_loader = DataLoader(
    val_ds,
    batch_size=cfg.BATCH_SIZE,
    shuffle=False,  # No shuffling for validation
    num_workers=cfg.NUM_WORKERS,
)


# %%
# ---------------------------------------------------------------
# 3. MODEL (U-Net)
# ---------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        c1 = self.enc1(x)
        c2 = self.enc2(self.pool(c1))
        c3 = self.enc3(self.pool(c2))
        c4 = self.enc4(self.pool(c3))
        b = self.bottleneck(self.pool(c4))
        u4 = self.up4(b)
        u4 = torch.cat([u4, c4], dim=1)
        d4 = self.dec4(u4)
        u3 = self.up3(d4)
        u3 = torch.cat([u3, c3], dim=1)
        d3 = self.dec3(u3)
        u2 = self.up2(d3)
        u2 = torch.cat([u2, c2], dim=1)
        d2 = self.dec2(u2)
        u1 = self.up1(d2)
        u1 = torch.cat([u1, c1], dim=1)
        d1 = self.dec1(u1)
        return torch.sigmoid(self.out(d1))


model = UNet().to(DEVICE)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)


p("UNet", model)

# %%
from PIL import Image


def validate_dataset(ds, name):
    print(f"Validating {name} dataset...")
    for i in range(min(20, len(ds))):  # check first few samples
        try:
            img, mask = ds[i]
        except Exception as e:
            print(f"Error at index {i}: {e}")
            raise
    print(f"{name} dataset OK ({len(ds)} samples).")


validate_dataset(train_ds, "train")
# validate_dataset(val_ds, "val")

# %%
# ---------------------------------------------------------------
# 4. TRAINING
# ---------------------------------------------------------------
# This section defines the training and validation loops for the model.
# - `train_epoch`: Runs one training pass over the dataset, applying gradient updates.
# - `val_epoch`: Evaluates model performance without updating weights.
# - Both functions use tqdm for progress visualization and handle device placement.
# - Inputs are normalized to float32 and scaled to [0, 1] to match model expectations.
# - The training loop iterates over epochs, logs losses, and saves the final model.

import os
from pathlib import Path

import torch
from tqdm import tqdm


def train_epoch(loader):
    model.train()
    loss_sum = 0
    for imgs, masks in tqdm(loader, desc="Training", leave=False):
        try:
            # imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            imgs = imgs.to(DEVICE).float() / 255.0
            masks = masks.to(DEVICE)

            preds = model(imgs)
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        except Exception as e:
            print(f"Error during training batch: {e}")
            break
    return loss_sum / max(1, len(loader))


def val_epoch(loader):
    model.eval()
    loss_sum = 0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Validation", leave=False):
            try:
                # imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                imgs = imgs.to(DEVICE).float() / 255.0
                masks = masks.to(DEVICE)

                preds = model(imgs)
                loss = criterion(preds, masks)
                loss_sum += loss.item()
            except Exception as e:
                p(f"Error during validation batch", e, "red")
                break
    return loss_sum / max(1, len(loader))


# --- Training loop ---
for epoch in range(1, cfg.EPOCHS + 1):
    p("Epoch", f"{epoch}/{cfg.EPOCHS}")
    try:
        tr_loss = train_epoch(train_loader)
        val_loss = val_epoch(val_loader)
        p(" - ", f"Train Loss {tr_loss:.4f} | Val Loss: {val_loss:.4f}")
    except Exception as e:
        p(f"Error in epoch {epoch}", e, "red")
        break

# --- Save model ---
model_path = cfg.PATHS_MODELS / "unet_canopy.pth"
torch.save(model.state_dict(), model_path)
p("Model saved", model_path.relative_to(cfg.ROOT))

# %%
# ---------------------------------------------------------------
# 5. INFERENCE EXAMPLE
# ---------------------------------------------------------------
model.eval()
sample_img = os.listdir(cfg.PATHS_EVAL_IMAGES)[0]
img = cv2.imread(os.path.join(cfg.PATHS_EVAL_IMAGES, sample_img))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
aug = val_transform(image=img)
img_tensor = aug["image"].unsqueeze(0).to(DEVICE)

with torch.no_grad():
    pred = model(img_tensor).squeeze().cpu().numpy()
mask = (pred > 0.5).astype(np.uint8)

plt.imshow(mask, cmap="gray")
plt.title("Predicted Tree Canopy Mask")
plt.axis("off")
plt.show()

# %%

# %%
val_transform = A.Compose([A.Resize(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), ToTensorV2()])
