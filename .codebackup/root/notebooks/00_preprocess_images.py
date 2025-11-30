# %% [markdown]
# # Notebook: 00 Preprocess Images
# ### Purpose: Convert TIFF files to PNG for stable training and validate image integrity
#

# %%
import os
import sys


sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../src"))

from pathlib import Path

from src.data.image_loader import validate_image_directory
from src.utils.config import Config
from src.utils.helpers import init_notebook, p, t
from src.utils.image_converter import ImageConverter


config = Config.load()
init_notebook(config.train.seed)


# %% [markdown]
# #### Step 1: Validate Current Images

# %%
t("Validating train images")
train_results = validate_image_directory(config.paths.train_images)

if config.paths.eval_images and config.paths.eval_images.exists():
    t("Validating eval images")
    eval_results = validate_image_directory(config.paths.eval_images)


# %%
if config.paths.eval_images and config.paths.eval_images.exists():
    t("Validating eval images")
    eval_results = validate_image_directory(config.paths.eval_images)


# %% [markdown]
# #### Step 2: Convert TIFF to PNG
#
# Why convert?
# - TIFF files (especially GeoTIFFs) can cause system crashes
# - OpenCV has inconsistent TIFF support
# - PNG is lossless, universally supported, and faster to load
# - Your crashes are likely caused by TIFF decompression issues
#

# %%
t("Converting train images")

converter = ImageConverter(
        source_dir = config.paths.train_images,
        target_dir = config.paths.train_images
)

stats = converter.convert_batch(overwrite = False)


# %%
if config.paths.eval_images and config.paths.eval_images.exists():
    t("Converting eval images")

    eval_converter = ImageConverter(
            source_dir = config.paths.eval_images,
            target_dir = config.paths.eval_images
    )

    eval_stats = eval_converter.convert_batch(overwrite = False)

# %% [markdown]
# #### Step 3: Update Annotations
#
# Automatically creates timestamped backups before modifying
#

# %%
if config.paths.annotations and Path(config.paths.annotations).exists():
    t("Updating annotations to reference PNG files")

    # Backups are created automatically with timestamps
    # e.g., annotations.json.backup_20241120_143022
    #       annotations.json.backup (latest)

    converter.update_annotations(
            annotations_path = config.paths.annotations,
            create_backup = True
    )


# %% [markdown]
# #### Step 4: Validate Converted Images
#

# %%
t("Validating converted images")
final_results = validate_image_directory(config.paths.train_images)


# %%
# Print summary
t("Conversion Summary")
p("Original valid", train_results["valid"])
p("Original invalid", train_results["invalid"])
p("Final valid", final_results["valid"])
p("Final invalid", final_results["invalid"])

if stats["failed"] > 0:
    t("Failed Conversions")
    for err in stats["errors"]:
        p(err["file"], err["error"])


# %%
if config.paths.eval_images and config.paths.eval_images.exists():
    t("Validating converted eval images")
    final_eval_results = validate_image_directory(config.paths.eval_images)
    p("Eval final valid", final_eval_results["valid"])
    p("Eval final invalid", final_eval_results["invalid"])


# %% [markdown]
# #### Convert Eval Images
#

# %%
if config.paths.eval_images and config.paths.eval_images.exists():
    t("Converting eval images")

    eval_converter = ImageConverter(
            source_dir = config.paths.eval_images,
            target_dir = config.paths.eval_images
    )

    eval_stats = eval_converter.convert_batch(overwrite = False)


# %% [markdown]
# #### Restore Annotations from Backup (if needed)
#
# If something went wrong, restore the original annotations
#

# %%
# from src.utils.image_converter import ImageConverter
#
# success = ImageConverter.restore_annotations_from_backup(
#     annotations_path=config.paths.annotations
# )
#
# if success:
#     p("Annotations restored successfully")
# else:
#     p("Failed to restore annotations")
#
#     # List available backups
#     backup_dir = Path(config.paths.annotations).parent
#     backups = list(backup_dir.glob("*.backup*"))
#     p("Available backups", len(backups))
#     for backup in backups:
#         p("", backup.name)
