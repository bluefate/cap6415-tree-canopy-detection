# %% [markdown]
# <a href="https://colab.research.google.com/github/bluefate/CAP6415_F25_project-Tree-Canopy-Detection/blob/main/notebooks/00%20colab%20setup.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Tree Canopy Detection - Google Colab Setup
# ### CAP-6415 Project Setup and Environment Configuration
#
# This notebook sets up the complete environment for the Tree Canopy Detection project including:
# - GitHub repository cloning
# - Dependencies installation
# - GPU/hardware verification
# - Storage configuration
# - Project structure setup
#
# **Author:** jherna65
#
# **Course:** CAP6415 - 2025
#
# **Project:** Tree Canopy Detection

# %% [markdown]
# ## System Information and GPU Check

# %%
# Check system information and GPU availability
import platform
from datetime import datetime

import psutil
import torch


print("=" * 60)
print("🌳 TREE CANOPY DETECTION - SYSTEM CHECK")
print("=" * 60)
print(f"Setup Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Platform: {platform.platform()}")
print(f"Python: {platform.python_version()}")
print(f"CPU Cores: {psutil.cpu_count()}")
print(f"RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
print(f"Disk Space: {psutil.disk_usage('/').free / 1e9:.1f} GB free")

# GPU Check
print("=" * 60)
print("🚀 GPU CONFIGURATION")
print("=" * 60)

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU Available: {gpu_name}")
    print(f"GPU Memory: {gpu_memory:.1f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")

    # Test GPU allocation
    try:
        test_tensor = torch.zeros((1000, 1000)).cuda()
        del test_tensor
        torch.cuda.empty_cache()
        print("✅ GPU Allocation: Working")
    except Exception as e:
        print(f"❌ GPU Allocation Error: {e}")
else:
    print("❌ No GPU available - will use CPU (training will be slower)")
    print("Consider upgrading to Colab Pro for GPU access")



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
