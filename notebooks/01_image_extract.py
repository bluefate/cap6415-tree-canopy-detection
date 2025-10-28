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
# PROJECT_ROOT = None

# for parent in [Path.cwd(), *Path.cwd().parents]:
#     if (parent / "src").exists():
#         if str(parent) not in sys.path:
#             sys.path.append(str(parent))
#             print(f"Added to sys.path: {parent}")
#         PROJECT_ROOT = parent
#         break


# if PROJECT_ROOT is None:
#     raise RuntimeError("Could not locate project root (folder containing 'src').")

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

    p("", f"No images found in {folder.relative_to(ROOT)}")

    img_path = random.choice(images)
    with Image.open(img_path) as img:
        plt.figure(figsize=(2, 2))
        plt.imshow(img)
        plt.title(f"{folder.relative_to(ROOT)}/{img_path.name}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

# %%
p("", "test")

# %%
