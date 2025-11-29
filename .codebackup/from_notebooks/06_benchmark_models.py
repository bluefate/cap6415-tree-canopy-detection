# %% [markdown]
# # Notebook: 06 Benchmark Models
# ### Purpose: benchmark segmentation models on a shared dataset and compare quality and speed.

# %%
import os
import sys


sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../src"))

import time

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.data.annotations import load_json_annotations
from src.data.augmentations import get_val_augmentations
from src.data.loaders import ImageMaskDataset
from src.models.zoo import build_model, MODEL_BENCHMARKS
from src.training.metrics import compute_metrics
from src.utils.config import Config
from src.utils.helpers import init_notebook, p


config = Config.load()

init_notebook(config.train.seed)

train_dir = config.paths.train_images
annotations_path = config.paths.annotations
entries = load_json_annotations(annotations_path)


# %% [markdown]
# #### Load validation entries

# %%
val_count = max(8, int(0.1 * len(entries)))
val_entries = entries[:val_count]

val_tf = get_val_augmentations(config.train.image_size)
val_ds = ImageMaskDataset(val_entries, train_dir, transform = val_tf)

p("Validation samples", len(val_ds))

# %% [markdown]
# #### Model definitions

# %%
p("Models", MODEL_BENCHMARKS)
#config.show()
p("Batch", config.train.batch_size)
p("Epochs", config.train.epochs)
p("Learning Rate", config.train.learning_rate, precision = 9)
p("Image Size", config.train.image_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# #### Benchmarking

# %%
rows = []

for name, params in MODEL_BENCHMARKS.items():
    p("Testing model", name)

    try:
        model = build_model(name, in_channels = 3, out_channels = 1, **params).to(device)


    except Exception as e:
        p("Failed to load model", name)
        p("Error", e)
        continue

    model.eval()

    total_iou = 0.0
    total_dice = 0.0
    total_acc = 0.0
    total_time = 0.0

    start_mem = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0

    with torch.no_grad():
        for img_t, mask_t in val_ds:
            img_t = img_t.unsqueeze(0).to(device)
            mask_t = mask_t.unsqueeze(0).to(device)

            t0 = time.time()
            pred = model(img_t)
            t1 = time.time()

            total_time += (t1 - t0)

            m = compute_metrics(pred, mask_t)
            total_iou += m["iou"]
            total_dice += m["dice"]
            total_acc += m["acc"]

    n = len(val_ds)
    end_mem = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    mem_used = end_mem - start_mem
    params_num = sum(p.numel() for p in model.parameters())

    rows.append(
            {
                "model":         name,
                "params":        params_num,
                "avg_iou":       total_iou / n,
                "avg_dice":      total_dice / n,
                "avg_acc":       total_acc / n,
                "avg_time_sec":  total_time / n,
                "gpu_mem_bytes": int(mem_used),
            }
    )

df = pd.DataFrame(rows)
df

# %% [markdown]
# #### Plot IoU, Speed, Memory

# %%
# IoU bar chart
plt.figure(figsize = (12, 6))
plt.bar(df["model"], df["avg_iou"])
plt.xticks(rotation = 45)
plt.ylabel("IoU")
plt.title("Model IoU comparison")
plt.show()


# %%
# speed bar chart
plt.figure(figsize = (12, 6))
plt.bar(df["model"], df["avg_time_sec"])
plt.xticks(rotation = 45)
plt.ylabel("Avg time sec")
plt.title("Model speed comparison")
plt.show()

# %%
# memory chart
plt.figure(figsize = (12, 6))
plt.bar(df["model"], df["gpu_mem_bytes"])
plt.xticks(rotation = 45)
plt.ylabel("GPU memory bytes")
plt.title("Model memory comparison")
plt.show()

# %% [markdown]
# #### Export benchmark results

# %%
out_path = config.paths.models / "benchmark_results.csv"
df.to_csv(out_path, index = False)
p("Saved", out_path)

# %%
