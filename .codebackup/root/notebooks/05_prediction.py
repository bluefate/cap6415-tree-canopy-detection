# %% [markdown]
# # Notebook: 05 Prediction
# ### Purpose: load a trained model, run inference, preview overlays, and optionally export a submission JSON.

# %%

from IPython import get_ipython


if not "google.colab" in str(get_ipython()):
    from pathlib import Path


    root = Path("C:/github/Tree-Canopy-Detection")

else:
    from pathlib import Path


    root = Path("/content/drive/MyDrive/TreeCanopyProject")

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
import torch


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
from src.exploration.evaluation import load_best_model


config = Config.load(root = root)

init_notebook(config.train.seed)

# if not "google.colab" in str(get_ipython()):
#     config.train.batch_size = 2
#     config.train.num_workers = 1
#     config.train.image_size = 32
#     config.train.epochs = 2

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

try:
    model = load_best_model(model_name, config, notebook = "03", mode = "rgb")

    # Save model temporarily for Predictor class
    temp_model_path = config.paths.models / "temp_best_model.pth"
    torch.save({ "model": model.state_dict() }, temp_model_path)

    predictor = Predictor(
            model_path = temp_model_path,
            model_name = model_name,
            image_size = config.train.image_size,
    )

    p(f"Successfully created predictor with {model_name}", color1 = c.GREEN)

except RuntimeError as e:
    p("No trained models found!", color1 = c.RED, bold = True)
    p("Available directories:", color1 = c.ORANGE)
    models_dir = config.paths.models
    if models_dir.exists():
        p("Top-level directories:")
        for item in models_dir.iterdir():
            if item.is_dir():
                p("  ", item.name)
    else:
        p("Models directory doesn't exist:", models_dir)

    raise e

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
    p("No model version found for export", color1 = c.RED)
else:
    p("Exporting submission for version:", version_folder.name)

    out_path = version_folder / "submission.json"
    export_submission(results, out_path)
    p("Submission Saved", out_path)
