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
from src.utils.helpers import init_notebook, p
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

vm = VersionManager(config.paths.models)
version_dir = vm.find_latest()
model_path = version_dir / "best_model.pth"

predictor = Predictor(
        model_path = model_path,
        model_name = model_name,
        image_size = config.train.image_size,
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

p("Exporting submission for version:", version_folder.name)

out_path = version_folder / "submission.json"
export_submission(results, out_path)
p("Submission Saved", out_path)
