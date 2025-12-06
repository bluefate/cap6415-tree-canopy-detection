# %% [markdown]
# <a href="https://colab.research.google.com/github/bluefate/CAP6415_F25_project-Tree-Canopy-Detection/blob/main/notebooks/00%20preflight%20check.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Notebook: 00 Pre-Flight Check Script
# ### Purpose: Test full pipeline to catch issues early
#

# %%
import traceback

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
from src.utils.preflight import check_config, check_data, check_gpu, check_model, check_dataset, check_disk_space
from src.utils.config import Config
from src.utils.helpers import init_notebook
from src.utils.helpers import c, p, simple_estimate_runtime, t


config = Config.load(root = root)
init_notebook(config.train.seed)



# %%
t("PRE-FLIGHT CHECK")

# %%
checks = [
    ("Configuration", lambda: check_config(config)),
    ("GPU", lambda: check_gpu()),
    ("Data", lambda: check_data(config)),
    ("Dataset", lambda: check_dataset(config)),
    ("Model", lambda: check_model()),
    ("Disk Space", lambda: check_disk_space(config)),
]

results = { }

for name, func in checks:
    try:
        results[name] = func()
        p("")  # Blank line
    except Exception as e:
        results[name] = False
        p(f"✗ {name} check crashed", str(e), color1 = c.RED)
        traceback.print_exc()


# %%

# Runtime estimate
try:
    simple_estimate_runtime(config)
except Exception as e:
    p("⚠ Runtime estimate failed", str(e), color1 = c.ORANGE)


# %%

# Summary
t("SUMMARY")

passed = sum(results.values())
total = len(results)

for name, success in results.items():
    if success:
        p(f"{name}", "✓", color1 = c.GREEN, bold = True)
    else:
        p(f"{name}", "✗", color1 = c.RED, bold = True)

