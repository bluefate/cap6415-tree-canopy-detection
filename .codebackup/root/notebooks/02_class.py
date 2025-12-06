# %% [markdown]
# # Notebook: 02 Class Annotation Exploration
# ### Purpose: This notebook provides utilities for exploring masks, overlays, classes, and bounding boxes

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

# %% [markdown]
# #### Config and Setup

# %%
import os
import sys


sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../src"))

from src.utils.config import Config
from src.utils.helpers import init_notebook


config = Config.load(root = root)
init_notebook(config.train.seed)



# %%
import os
import sys


sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../src"))

from src.data.annotations import get_unique_classes, load_json_annotations
from src.exploration.class_explorer import *
from src.utils.config import Config
from src.utils.helpers import init_notebook, p


config = Config.load(root = root)
init_notebook(config.train.seed)



# %%

annotations_path = config.paths.annotations

entries = load_json_annotations(annotations_path)
unique_classes = get_unique_classes(entries)

p("Loaded annotation entries", len(entries))
p("Unique annotation classes", unique_classes)


# %%
entry = entries[config.train.seed]
p("Classes found in entry", count_classes(entry))
# p("Number of classes found", class_distribution(entry))

# %%


show_single_class(config, entry, "individual_tree")
show_single_class(config, entry, "group_of_trees")
show_all_classes(config, entry)
show_per_class(config, entry)
show_overlay_all(config, entry)
show_overlay_by_class(config, entry)



# %% [markdown]
# #### Explore Masks Per Class

# %%
explore_image(config, entry)

# %% [markdown]
# #### Color Mask and Overlay

# %%
explore_color_overlay(config, entry)

# %% [markdown]
# #### Bounding Box Exploration

# %%
explore_bboxes(entry, config.paths.train_images)

# %% [markdown]
# #### Dataset-wide Report

# %%
dataset_report(config, entries)
