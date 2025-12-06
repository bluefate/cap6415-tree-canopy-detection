# %% [markdown]
# # Notebook: 06 Benchmark Models
# ### Purpose: benchmark segmentation models on a shared dataset and compare quality and speed.

# %%

from IPython import get_ipython


if not "google.colab" in str(get_ipython()):
    from pathlib import Path


    root = Path("C:/github/Tree-Canopy-Detection")

else:
    from pathlib import Path


    root = Path("/content/CAP6415_F25_project-Tree-Canopy-Detection")

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

import time
import matplotlib.pyplot as plt
import pandas as pd
import torch
from src.data.annotations import load_json_annotations
from src.data.augmentations import get_val_augmentations
from src.data.loaders import ImageMaskDataset
from src.models.zoo import build_model, MODEL_BENCHMARKS
from src.utils.config import Config
from src.utils.helpers import init_notebook, p


config = Config.load(root = root)

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
        model = build_model(name, in_channels = 3, out_channels = 3, **params).to(device)
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

            # Use multiclass metrics for consistency
            from src.training.metrics import compute_metrics_multiclass


            m = compute_metrics_multiclass(pred, mask_t, num_classes = 3)

            total_iou += m.get("iou", m.get("mean_iou", 0.0))
            total_dice += m.get("dice", 0.0)
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
