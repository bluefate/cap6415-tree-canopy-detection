# %% [markdown]
# # Notebook: 05 Prediction
# ### Purpose: load a trained model, run inference, preview overlays, and optionally export a submission JSON.

# %%
import os
import sys


sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../src"))
from src.data.augmentations import get_val_augmentations
from src.exploration.visualize import show_side_by_side
from src.models.zoo import MODEL_BUILDERS
from src.prediction.pipeline import Predictor
from src.prediction.submission import export_submission
from src.utils.config import Config
from src.utils.helpers import c, init_notebook, p
from src.utils.versioning import VersionManager


config = Config.load()

init_notebook(config.train.seed)


# %% [markdown]
# #### Select model version

# %%
p("Models", MODEL_BUILDERS)
#config.show()
p("Batch", config.train.batch_size)
p("Epochs", config.train.epochs)
p("Learning Rate", config.train.learning_rate, precision = 9)
p("Image Size", config.train.image_size)

# %%
model_name = "simple_cnn"

# Search for models in structured paths first
structured_paths = [
    config.paths.models / model_name / "rgb",
    config.paths.models / "notebook_eval" / model_name / "rgb",
    config.paths.models / model_name / "rgb" / f"size_{config.train.image_size}",
]

version_dir = None
vm = None

# Try structured paths first
for path in structured_paths:
    if path.exists():
        p(f"Checking path: {path}", color1=c.CYAN)
        vm = VersionManager(path)
        version_dir = vm.find_latest()
        if version_dir is not None:
            p(f"Found model in: {path}", color1=c.GREEN)
            break

# Fallback to general search
if version_dir is None:
    p("Structured paths not found, searching generally...", color1=c.ORANGE)
    vm = VersionManager(config.paths.models)
    version_dir = vm.find_latest()

# Add null check for missing models
if version_dir is None:
    p("No trained models found!", color1=c.RED, bold=True)
    p("Available directories:", color1=c.ORANGE)
    models_dir = config.paths.models
    if models_dir.exists():
        p("Top-level directories:")
        for item in models_dir.iterdir():
            if item.is_dir():
                p("  ", item.name)
                # Check subdirectories
                for subitem in item.iterdir():
                    if subitem.is_dir():
                        p("    ", subitem.name)
                        # Check for version directories
                        for subsubitem in subitem.iterdir():
                            if subsubitem.is_dir() and subsubitem.name.startswith('v'):
                                p("      ", subsubitem.name)
    else:
        p("Models directory doesn't exist:", models_dir)

    p("\nTo fix this issue:")
    p("1. Run notebook 03 (training) first, OR")
    p("2. Run notebook 10 with a simple experiment like:", color1=c.CYAN)
    p("   experiments = [('simple_cnn', 'rgb', None)]", color1=c.CYAN)

    raise RuntimeError("No trained models found. Please run training first.")

p(f"Using model from: {version_dir}", color1=c.GREEN)

model_path = version_dir / "best_model.pth"

# Check if model file exists
if not model_path.exists():
    checkpoint_path = version_dir / "checkpoint.pth"
    if checkpoint_path.exists():
        p("Using checkpoint instead of best_model", color1=c.ORANGE)
        model_path = checkpoint_path
    else:
        raise RuntimeError(f"No model weights found in {version_dir}")

predictor = Predictor(
    model_path=model_path,
    model_name=model_name,
    image_size=config.train.image_size,
)

# %% [markdown]
# #### Run on evaluation folder

# %%
eval_dir = config.paths.eval_images
p("eval_dir", eval_dir)

transform = get_val_augmentations(config.train.image_size)

results = predictor.run_on_folder(eval_dir, transform = transform, num_samples = 10)


# %% [markdown]
# #### Visualizations Samples

# %%
for r in results:
    img = r["image"]
    mask = r["mask"]
    overlay = r["overlay"]

    # show_image(img, r["name"])
    # show_mask(mask, "Predicted mask")
    # show_overlay(img, mask, 0.4, "Overlay")

    show_side_by_side(
            img, mask, overlay,
            titles = [r["name"], "Predicted mask", "Overlay"],
            cmaps = [None, "gray", None],
    )



# %% [markdown]
# ### Export submission file

# %%
vm = VersionManager(config.paths.models)
version_folder = vm.find_latest()

if version_folder is None:
    p("No model version found for export", color1=c.RED)
else:
    p("Exporting submission for version:", version_folder.name)

    out_path = version_folder / "submission.json"
    export_submission(results, out_path)
    p("Submission Saved", out_path)
