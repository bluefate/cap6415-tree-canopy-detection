Solafune Tree Canopy Detection challenge - Week 1 Project Summary
==================================================================

Overview
----------------------------------------

Work so far was focused on establishing the foundation for the Solafune Tree Canopy Detection
challenge. Key goals included documenting competition requirements, building reusable tooling, and
exploring the training data.

Documentation & Configuration
----------------------------------------

- Documented competition context, submission format, and environment/model setup in the project
  README for quick reference.
- Outlined repository structure and environment setup requirements.
- Created a centralized YAML configuration file to define dataset locations, model checkpoints, and
  core
  hyperparameters.
- Implemented `src/utils/config.py` to automatically locate the project root, load environment
  variables, parse the YAML
  file, normalize paths, and expose global settings.
- Added helper utilities in `src/utils/helpers.py`, including a rich-print helper (`p`) and other
  convenience functions.

Data Preparation & Exploration
----------------------------------------

- Developed exploratory notebooks (`notebooks/00_info.ipynb` and corresponding script) to verify the
  Python/Jupyter
  environment and dependency installation.
- Built `notebooks/01_image_extract.ipynb` to:
    * Load configuration values.
    * Unpack provided training and evaluation ZIP archives while removing extraneous `__MACOSX`
      folders.
    * Generate binary mask TIFFs from custom JSON polygon annotations.
    * Visualize random image samples, channel statistics, and overlay mask previews to confirm
      annotation alignment.
- Counted and verified available training, mask, and evaluation assets.

Model Prototyping & Training Experiments
----------------------------------------

- Implemented starter notebook (`notebooks/01_starter_sample.ipynb`) featuring a basic PyTorch U-Net
  example as from
  project discussions.
- Authored a training notebook (`notebooks/02_Train.ipynb`) to experiment with:
    * Albumentations-based preprocessing pipelines for training and validation.
    * Comprehensive data augmentation recipes with reusable data loaders that materialize augmented
      samples in memory.
    * A flexible dataset wrapper supporting Tensor/NumPy conversions and hashing to prevent
      duplicate augmentations.
    * Created a model using segmentation_models_pytorch's ResNet34-backed U-Net, optimized with Dice
      loss and
      qualitative prediction checks.
    * Sanity-check utilities for model and data verification.
