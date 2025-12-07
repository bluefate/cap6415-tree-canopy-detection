# %% [markdown]
# # Notebook: 12 Model Tracker and Submission Manager
# ### Purpose: Track all models in checkpoint directories and manage submissions
#

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
import sys
import os


sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../src"))

#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  #enable for debugging ONLY

import numpy as np
import torch
from src.utils.config import Config
from src.utils.helpers import init_notebook, p, t, c
from src.data.checkpoint import generate_best_model_report, main_model_tracking_pipeline, show_model_details
import random


# Initialize configuration
config = Config.load(root = root)

init_notebook(config.train.seed)


# %%
torch.manual_seed(config.train.seed)
np.random.seed(config.train.seed)
random.seed(config.train.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(config.train.seed)
    torch.cuda.manual_seed_all(config.train.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
p("Device", device)
p("Config loaded", config.paths.root)


# %% [markdown]
# ### Execute Pipeline
#

# %%

# Run the main pipeline
t("Executing Model Tracking and Submission Management Pipeline")

try:
    results = main_model_tracking_pipeline(config)
    p("Pipeline completed successfully!", color1 = c.GREEN)
except Exception as e:
    p(f"Pipeline failed with error: {str(e)}", color1 = c.RED)
    import traceback


    traceback.print_exc()


# %% [markdown]
# ### Show details for the best model
#

# %%
if 'results' in locals() and results:
    ranked_models = results.get("ranked_models", [])
    if ranked_models:
        t("Best Model Details")
        show_model_details(ranked_models[0])


# %% [markdown]
# ### Generate Final Best Model Report
#

# %%

if 'results' in locals() and results:
    ranked_models = results.get("ranked_models", [])
    if ranked_models:
        best_report_path = generate_best_model_report(ranked_models, config)

