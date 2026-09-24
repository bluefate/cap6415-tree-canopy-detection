# %% [markdown]
# <a href="https://colab.research.google.com/github/bluefate/CAP6415_F25_project-Tree-Canopy-Detection/blob/main/notebooks/_run_all.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Notebook: Run All
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
import glob
import os
import subprocess
import sys

import nbformat


repo_root = "/content/CAP6415_F25_project-Tree-Canopy-Detection"
source_folder = os.path.join(repo_root, "notebooks")

sys.path.append(os.path.join(repo_root, "src"))
from src.utils.helpers import p, t, c


t("Run All Notebooks")

notebooks = glob.glob(os.path.join(source_folder, "*.ipynb"))
notebooks = [nb for nb in notebooks if os.path.basename(nb) in
             ["10_master_execution_plan_512_all.ipynb", "12_model_tracker_submission_manager.ipynb"]
             ]
notebooks = sorted(notebooks)

for nb in notebooks:
    print(os.path.basename(nb))


# %%
import os
from nbclient import NotebookClient
from nbclient.exceptions import CellTimeoutError


failed_notebooks = []

# Set per cell timeout in seconds. Use None for no timeout.
#PER_CELL_TIMEOUT = 3600  # one hour per cell
PER_CELL_TIMEOUT = None

for nb_path in notebooks:
    nb_name = os.path.basename(nb_path)
    try:
        p()
        t(f"Executing: {nb_name}")

        nb = nbformat.read(nb_path, as_version = 4)
        client = NotebookClient(
                nb,
                timeout = PER_CELL_TIMEOUT,
                kernel_name = "python3",
        )
        client.execute()

        # Save executed notebook
        nbformat.write(nb, nb_path)
        p(f"Finished & saved: {nb_name}", color1 = c.BLUE)

        # Path relative to repo root
        rel_nb_path = os.path.relpath(nb_path, repo_root)

        # Git add/commit/push from repo root
        subprocess.run(
                ["git", "add", rel_nb_path],
                check = True,
                cwd = repo_root,
        )
        commit_msg = f"Auto-update {nb_name}"
        subprocess.run(
                ["git", "commit", "-m", commit_msg],
                check = False,
                cwd = repo_root,
        )
        subprocess.run(
                ["git", "push"],
                check = True,
                cwd = repo_root,
        )

        p(f"Committed and pushed: {nb_name}", color1 = c.GREEN)

    except CellTimeoutError as e:
        p(
                f"Timeout while executing {nb_name}",
                str(e),
                color1 = c.RED,
                color2 = c.BLACK,
        )
        failed_notebooks.append(nb_path)

    except Exception as e:
        p(f"Error while executing {nb_name}", str(e), color1 = c.RED, color2 = c.BLACK)
        failed_notebooks.append(nb_path)

t("Execution completed.")

if failed_notebooks:
    p("\n\nNotebooks that failed:", color1 = c.MAGENTA)
    for nb_path in failed_notebooks:
        p("", nb_path)

