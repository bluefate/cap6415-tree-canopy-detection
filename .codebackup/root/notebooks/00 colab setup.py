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
# **Course:** CAP6415 F25
# **Project:** Tree Canopy Detection

# %% [markdown]
# ## Step 1: System Information and GPU Check

# %%
# Check system information and GPU availability
import os
import psutil
import torch
import platform
from datetime import datetime

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



# %% [markdown]
# # Step 2: Google Drive Integration

# %%
from google.colab import drive
import shutil

drive.mount('/content/drive')

# Create project backup directory in Drive
drive_project_path = '/content/drive/MyDrive/TreeCanopyProject'
os.makedirs(drive_project_path, exist_ok=True)

# Create subdirectories for organization
subdirs = [
    'data',
    'models',
    'checkpoints',
    'outputs',
    'logs',
    'backups'
]


for subdir in subdirs:
    path = os.path.join(drive_project_path, subdir)
    os.makedirs(path, exist_ok=True)
    print(f"Created: {path}")



# %% [markdown]
# # Step 3: GitHub Repository Setup

# %%
# !ls /content/drive/MyDrive/TreeCanopyProject


# %%
import os

env_path = '/content/drive/MyDrive/TreeCanopyProject/.env'

if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value
    print("✅ Environment variables loaded")
else:
    print("❌ .env file not found")

github_token = os.getenv('TOKEN')

if github_token:
    # !git config --global user.email "jherna65@fau.edu"
    # !git config --global user.name "bluefate"

    clone_url = f"https://bluefate:{github_token}@github.com/bluefate/CAP6415_F25_project-Tree-Canopy-Detection.git"
    # !git clone $clone_url
    # %cd CAP6415_F25_project-Tree-Canopy-Detection
    # !git pull

    print("✅ Setup complete")
else:
    print("❌ Add GITHUB_TOKEN to .env file")


# %% [markdown]
# # Step 4: Installing Requirements

# %%
# !pip install -r requirements.txt


# %%
