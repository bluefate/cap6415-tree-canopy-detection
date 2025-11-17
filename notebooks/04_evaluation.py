# %% [markdown]
# # Notebook: 04 Evaluation
# ## Purpose: load predictions, compute metrics, visualize overlays, and summarize performance.
#

# %%

from src.utils.config import Config
from src.utils.helpers import init_notebook, p, t



config = Config.load()

init_notebook(config.train.seed)


# %%
import numpy as np
import torch
from src.utils.versioning import VersionManager
from src.data.annotations import load_json_annotations
from src.data.augmentations import get_val_augmentations
from src.data.loaders import ImageMaskDataset
from src.exploration.visualize import show_image, show_mask, show_overlay
from src.models.zoo import build_model, MODEL_BUILDERS
from src.training.metrics import compute_metrics
from models.zoo import MODEL_BUILDERS

train_dir = config.paths.train_images
mask_dir = config.paths.train_masks
annotations_path = config.paths.annotations
entries = load_json_annotations(annotations_path)

# %%
p("Models", MODEL_BUILDERS)
#config.show()
p("Batch", config.train.batch_size)
p("Epochs", config.train.epochs)
p("Learning Rate", config.train.learning_rate, precision = 9)
p("Image Size", config.train.image_size)


# %% [markdown]
# #### Validation setup

# %%
val_tf = get_val_augmentations(config.train.image_size)
dataset = ImageMaskDataset(entries, train_dir, transform = val_tf)


# %% [markdown]
# #### Loading Model

# %%
def load_best_model(model_name: str, config):
    t(model_name)
    from src.models.zoo import build_model
    from src.utils.versioning import VersionManager

    vm = VersionManager(config.paths.models)
    version_dir = vm.find_latest()
    best_path = version_dir / "best_model.pth"
    p("Using model version", version_dir.name)

    model = build_model(model_name, in_channels=3, out_channels=1)
    state = torch.load(best_path, map_location="cpu")
    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)

    #return model.eval()
    return model

model = load_best_model("simple_cnn", config)
model.eval()


# %% [markdown]
# #### Batch evaluation

# %%

all_metrics = []

for idx in range(len(dataset)):
    img_t, mask_t = dataset[idx]

    img_np = img_t.permute(1, 2, 0).numpy()
    true_np = mask_t.squeeze().numpy()

    with torch.no_grad():
        pred = model(img_t.unsqueeze(0))
    metrics = compute_metrics(pred, mask_t.unsqueeze(0))
    all_metrics.append(metrics)


# %% [markdown]
# #### Summarize metrics

# %%
iou_vals = [m["iou"] for m in all_metrics]
dice_vals = [m["dice"] for m in all_metrics]
acc_vals = [m["acc"] for m in all_metrics]

p("Mean IoU", np.mean(iou_vals))
p("Mean Dice", np.mean(dice_vals))
p("Mean Accuracy", np.mean(acc_vals))

# %% [markdown]
# #### Visualizations Samples

# %%
from exploration.visualize import show_side_by_side
import random
import cv2
import numpy as np
import torch

for _ in range(5):
    idx = random.randint(0, len(dataset) - 1)
    t(f"Image {idx}")

    img_t, mask_t = dataset[idx]

    img = img_t.permute(1, 2, 0).numpy()
    mask = mask_t.squeeze().numpy()

    with torch.no_grad():
        pred = model(img_t.unsqueeze(0)).cpu().squeeze().numpy()

    pred_bin = (pred > 0.5).astype(np.uint8)

    # show_image(img, f"Image {idx}")
    # show_mask(mask, "Ground Truth")
    # show_mask(pred_bin, "Prediction")
    # show_overlay(img, pred_bin, 0.4, "Overlay")


    titles = [f"Image {idx}", "Ground Truth", "Prediction", "Overlay"]

    if img.max() <= 1.0:
        base = (img * 255).astype(np.uint8)
    else:
        base = img.astype(np.uint8)

    mask_u8 = (mask > 0.5).astype(np.uint8) * 255
    pred_u8 = pred_bin * 255

    mask_rgb = np.zeros_like(base)
    mask_rgb[:, :, 0] = mask_u8

    pred_rgb = np.zeros_like(base)
    pred_rgb[:, :, 0] = pred_u8

    overlay = cv2.addWeighted(base, 0.6, pred_rgb, 0.4, 0)

    show_side_by_side(base, mask_u8, pred_u8, overlay,
                      titles=titles,
                      cmaps = [None, "gray", "gray", None]
                      )


# %%
