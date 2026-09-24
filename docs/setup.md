# Tree Canopy Detection - Setup and Execution Guide

This guide explains how to run the project in both Jupyter on a local machine and Google Colab.

The best way to reproduce experiments is to:

1. Configure paths and settings in `config.yml`.
2. Set `PROJECT_ROOT` and related variables in a `.env` file.
3. Place the Solafune data files in the paths expected by `config.yml`.
4. Run the preprocessing notebooks.
5. Run either the individual experiment notebooks or one of the master execution plans.

Most notebooks read their settings from `config.yml` / `config.yaml`.
Only the `13_master_execution_plan_*.ipynb` / `14_master_execution_plan_*.ipynb` notebooks have additional manual overrides for the image
size hardcoded, or other values that overwrite the configured size.

```python
config.train.image_size = 512
```

---

## 1. Project root and configuration

The project expects a single project root. Use one of these examples.

In Colab:

```python
root = Path("/content/drive/MyDrive/TreeCanopyProject")
```

In Windows:

```python
root = Path("C:/github/Tree-Canopy-Detection")
```

`config.yml` should live under this root. For example:

```text
C:/github/Tree-Canopy-Detection/config.yml
```

or

```text
/content/drive/MyDrive/TreeCanopyProject/config.yml
```

### 1.1 config.yml paths

Configure `config.yml` so it points to both the original Solafune files and the derived folders.

Example:

```yaml
paths:
  # original Solafune files (top-level data/ — not src/data, which is code only)
  train_images_zip: "data/train_images.zip"
  annotations: "data/train_annotations.json"
  eval_images_zip: "data/evaluation_images.zip"
  template: "data/sample_answer.json"

  # extracted and generated data
  train_images: "data/train_images"
  train_masks: "data/train_masks"
  eval_images: "data/evaluation_images"
  eval_masks: "data/evaluation_masks"

  # project structure
  models: "checkpoints"
  notebooks: "notebooks"
  data: "data"
```

With this setup, once you set `PROJECT_ROOT`, an internal helper can join `PROJECT_ROOT` and these
relative paths.

Note: all notebooks read these values from `config.yml` unless manually overwritten, as in the
`13` / `14` master execution plan notebooks.

---

## 2. .env configuration

You can use a `.env` file so code can detect the project root and other values without hardcoding.

In Colab:

```python
env_path = "/content/drive/MyDrive/TreeCanopyProject/.env"
```

In Windows:

```python
env_path = "C:/github/Tree-Canopy-Detection/.env"
```

Example `.env` content for Windows:

```env
PROJECT_ROOT=C:\github\Tree-Canopy-Detection
PYTHONPATH=C:\github\Tree-Canopy-Detection
TOKEN=enter_your_github_token_here
```

Example `.env` content for Colab:

```env
PROJECT_ROOT=/content/drive/MyDrive/TreeCanopyProject
PYTHONPATH=/content/drive/MyDrive/TreeCanopyProject
TOKEN=enter_your_github_token_here
```

`TOKEN` is an optional GitHub personal access token used only by Colab setup cells to clone a
private copy of the repo. Keep the real value in `.env` (gitignored) — never commit it.

Make sure your environment loader reads this file before importing project modules.

---

## 3. Dataset download

You must obtain the dataset directly from the Solafune competition page:

Solafune Tree Canopy Detection data:
https://community.solafune.com/competitions/26ff758c-7422-4cd1-bfe0-daecfc40db70?menu=data

Download all four required files:

- `train_images.zip`
- `train_annotations.json`
- `evaluation_images.zip`
- `sample_answer.json`

Place them under top-level `data/` so they match `config.yaml`. For example:

```text
<PROJECT_ROOT>/data/train_images.zip
<PROJECT_ROOT>/data/train_annotations.json
<PROJECT_ROOT>/data/evaluation_images.zip
<PROJECT_ROOT>/data/sample_answer.json
```

Or place the already-unzipped folders there:

```text
<PROJECT_ROOT>/data/train_images/
<PROJECT_ROOT>/data/evaluation_images/
```

The preprocess notebooks then generate masks into:

```text
<PROJECT_ROOT>/data/train_masks
<PROJECT_ROOT>/data/evaluation_masks
```

The model checkpoints, notebooks, and dataset folders use:

```text
<PROJECT_ROOT>/checkpoints
<PROJECT_ROOT>/notebooks
<PROJECT_ROOT>/data
```

---

## 4. Local Jupyter setup

### 4.1 Prerequisites

- Python 3.8 or newer
- CUDA 11.8 or compatible GPU drivers if you want GPU acceleration
- Git

### 4.2 Clone and environment

```bash
git clone https://github.com/bluefate/cap6415-tree-canopy-detection.git
cd cap6415-tree-canopy-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your `.env` file as described earlier, then start Jupyter:

```bash
jupyter notebook
```

Open the `notebooks` folder and run `01_preflight_check.ipynb` to confirm the environment.

---

## 5. Google Colab setup

### 5.1 Basic project setup

1. Mount Google Drive
2. Point `PROJECT_ROOT` to a folder inside Drive
3. Clone the GitHub repository into that folder
4. Run `01_preflight_check.ipynb` to validate the environment

Place your files in the MyDrive directory as shown below:

```text
/content/drive/MyDrive/TreeCanopyProject/config.yml
/content/drive/MyDrive/TreeCanopyProject/.env
```

In Colab notebooks, set:

```python
env_path = "/content/drive/MyDrive/TreeCanopyProject/.env"
```

and make sure the config loader uses:

```python
root = Path("/content/drive/MyDrive/TreeCanopyProject")
```

Then you can open any notebook from:

```text
/content/drive/MyDrive/TreeCanopyProject/cap6415-tree-canopy-detection/notebooks
```

Select GPU in Colab:

1. Runtime -> Change runtime type
2. Hardware accelerator -> GPU

---

## 6. Preprocessing and data preparation

After you have the dataset and configuration in place, run the preprocessing notebooks in this
order:

1. `01_preflight_check.ipynb`
    - Verifies Python installation, libraries, and GPU.

2. `02_image_extract.ipynb`
    - Reads the paths from `config.yml`.
    - Extracts `train_images.zip` and `evaluation_images.zip` into the `train_images` and
      `evaluation_images` folders.
    - Validates that images and annotations align.

3. `03_preprocess_images.ipynb`
    - Converts TIFF images to a consistent RGB PNG format.
    - Generates masks from polygon annotations for training and evaluation.
    - Populates `train_masks` and `evaluation_masks`.

Once these three steps complete, the data is ready for model experiments.

---

## 7. Running notebooks for quick review

To quickly review individual parts of the pipeline, you can run the main notebooks one by one. Each
notebook reads paths from `config.yml` and assumes the preprocessing above has completed.

Recommended order:

1. `04_starter_sample.ipynb`
    - Educational starter: dataset + sample model sanity checks.

2. `05_class.ipynb`
    - Mask/class/bbox exploration utilities.

3. `06_exploration.ipynb`
    - Explores data statistics, filters, and visualizations.

4. `07_training.ipynb`
    - Configures training loops and runs one or more experiments.

5. `08_evaluation.ipynb`
    - Evaluates trained models and computes metrics.

6. `09_prediction.ipynb`
    - Runs inference on evaluation images and generates submission predictions.

7. `10_benchmark_models.ipynb`
    - Benchmarks multiple architectures and compares results.

8. `11_data_analysis.ipynb`
    - Does deeper error analysis and data insights.

9. `12_filter_experimentation.ipynb`
    - Tests filter pipelines and image transforms.

Each of these notebooks uses `config.yml` to find data, checkpoints, and output paths. You can run
them in Jupyter or Colab.

---

## 8. Running master execution plans (end to end)

If you want to run the full pipeline from preprocessing to training to submission in one pass, use
one of the master execution plan notebooks:

- `13_master_execution_plan_512_rgb.ipynb` (models on RGB / no-filter set)
- `14_master_execution_plan_512_all.ipynb` (includes filter experiment variants)


In these filenames, `512` is the tile/image size used when resizing.

These master notebooks:

- Load configuration and environment values.
- Set or override paths for Colab or local use.
- Build datasets and dataloaders.
- Train selected models and save checkpoints under `checkpoints`.
- Run evaluation and prediction.
- Generate submission JSON files under a submissions folder.

Note:

- Unlike the other notebooks, the master execution notebooks can contain explicit path overrides.
- Check the top of each master notebook for any `root = Path(...)` lines and confirm they match your
  `PROJECT_ROOT` location.
- Every `13_*` / `14_*` master execution plan notebook contains a section labeled “Setup experiments to run”
  that controls which experiment combinations are executed.
- For example, to select all models without filters you can do the following:
    ```python
    experiments = [
        exp for exp in all_experiments if exp[1] == "rgb"
    ]
    ```
- To run all filters on just one model:
    ```python
    experiments = [
        exp for exp in all_experiments if exp[0] == "simple_cnn"
    ]
    ```
- To run all experiments:
    ```python
    experiments = all_experiments
    ```
- To run one experiment:
    ```python
    experiments = [all_experiments[0]]
    ```

Here is a list of all the available experiments. You will also see this in the notebook when
executed

```text
=== Available experiments ===
All experiments: 84 items
  0: ('simple_cnn', 'rgb', None)
  1: ('simple_cnn', 'concat', None)
  2: ('simple_cnn', 'filtered', ['laplacian', 'sobel', 'clahe'])
  3: ('simple_cnn', 'filtered', ['gaussian_3x3', 'gaussian_5x5', 'gaussian_7x7'])
  4: ('simple_cnn', 'filtered', ['sharpen_basic', 'high_pass_3x3', 'edge_enhance'])
  5: ('simple_cnn', 'filtered', ['sobel_x', 'sobel_y', 'laplacian_3x3'])
  6: ('simple_cnn', 'filtered', ['laplacian', 'gaussian_5x5', 'clahe'])
  7: ('unet', 'rgb', None)
  8: ('unet', 'concat', None)
  9: ('unet', 'filtered', ['laplacian', 'sobel', 'clahe'])
  10: ('unet', 'filtered', ['gaussian_3x3', 'gaussian_5x5', 'gaussian_7x7'])
  11: ('unet', 'filtered', ['sharpen_basic', 'high_pass_3x3', 'edge_enhance'])
  12: ('unet', 'filtered', ['sobel_x', 'sobel_y', 'laplacian_3x3'])
  13: ('unet', 'filtered', ['laplacian', 'gaussian_5x5', 'clahe'])
  14: ('smp_unet', 'rgb', None)
  15: ('smp_unet', 'concat', None)
  16: ('smp_unet', 'filtered', ['laplacian', 'sobel', 'clahe'])
  17: ('smp_unet', 'filtered', ['gaussian_3x3', 'gaussian_5x5', 'gaussian_7x7'])
  18: ('smp_unet', 'filtered', ['sharpen_basic', 'high_pass_3x3', 'edge_enhance'])
  19: ('smp_unet', 'filtered', ['sobel_x', 'sobel_y', 'laplacian_3x3'])
  20: ('smp_unet', 'filtered', ['laplacian', 'gaussian_5x5', 'clahe'])
  21: ('smp_fpn', 'rgb', None)
  22: ('smp_fpn', 'concat', None)
  23: ('smp_fpn', 'filtered', ['laplacian', 'sobel', 'clahe'])
  24: ('smp_fpn', 'filtered', ['gaussian_3x3', 'gaussian_5x5', 'gaussian_7x7'])
  25: ('smp_fpn', 'filtered', ['sharpen_basic', 'high_pass_3x3', 'edge_enhance'])
  26: ('smp_fpn', 'filtered', ['sobel_x', 'sobel_y', 'laplacian_3x3'])
  27: ('smp_fpn', 'filtered', ['laplacian', 'gaussian_5x5', 'clahe'])
  28: ('smp_linknet', 'rgb', None)
  29: ('smp_linknet', 'concat', None)
  30: ('smp_linknet', 'filtered', ['laplacian', 'sobel', 'clahe'])
  31: ('smp_linknet', 'filtered', ['gaussian_3x3', 'gaussian_5x5', 'gaussian_7x7'])
  32: ('smp_linknet', 'filtered', ['sharpen_basic', 'high_pass_3x3', 'edge_enhance'])
  33: ('smp_linknet', 'filtered', ['sobel_x', 'sobel_y', 'laplacian_3x3'])
  34: ('smp_linknet', 'filtered', ['laplacian', 'gaussian_5x5', 'clahe'])
  35: ('smp_deeplabv3', 'rgb', None)
  36: ('smp_deeplabv3', 'concat', None)
  37: ('smp_deeplabv3', 'filtered', ['laplacian', 'sobel', 'clahe'])
  38: ('smp_deeplabv3', 'filtered', ['gaussian_3x3', 'gaussian_5x5', 'gaussian_7x7'])
  39: ('smp_deeplabv3', 'filtered', ['sharpen_basic', 'high_pass_3x3', 'edge_enhance'])
  40: ('smp_deeplabv3', 'filtered', ['sobel_x', 'sobel_y', 'laplacian_3x3'])
  41: ('smp_deeplabv3', 'filtered', ['laplacian', 'gaussian_5x5', 'clahe'])
  42: ('smp_deeplabv3plus', 'rgb', None)
  43: ('smp_deeplabv3plus', 'concat', None)
  44: ('smp_deeplabv3plus', 'filtered', ['laplacian', 'sobel', 'clahe'])
  45: ('smp_deeplabv3plus', 'filtered', ['gaussian_3x3', 'gaussian_5x5', 'gaussian_7x7'])
  46: ('smp_deeplabv3plus', 'filtered', ['sharpen_basic', 'high_pass_3x3', 'edge_enhance'])
  47: ('smp_deeplabv3plus', 'filtered', ['sobel_x', 'sobel_y', 'laplacian_3x3'])
  48: ('smp_deeplabv3plus', 'filtered', ['laplacian', 'gaussian_5x5', 'clahe'])
  49: ('segformer', 'rgb', None)
  50: ('segformer', 'concat', None)
  51: ('segformer', 'filtered', ['laplacian', 'sobel', 'clahe'])
  52: ('segformer', 'filtered', ['gaussian_3x3', 'gaussian_5x5', 'gaussian_7x7'])
  53: ('segformer', 'filtered', ['sharpen_basic', 'high_pass_3x3', 'edge_enhance'])
  54: ('segformer', 'filtered', ['sobel_x', 'sobel_y', 'laplacian_3x3'])
  55: ('segformer', 'filtered', ['laplacian', 'gaussian_5x5', 'clahe'])
  56: ('yolov8n', 'rgb', None)
  57: ('yolov8n', 'concat', None)
  58: ('yolov8n', 'filtered', ['laplacian', 'sobel', 'clahe'])
  59: ('yolov8n', 'filtered', ['gaussian_3x3', 'gaussian_5x5', 'gaussian_7x7'])
  60: ('yolov8n', 'filtered', ['sharpen_basic', 'high_pass_3x3', 'edge_enhance'])
  61: ('yolov8n', 'filtered', ['sobel_x', 'sobel_y', 'laplacian_3x3'])
  62: ('yolov8n', 'filtered', ['laplacian', 'gaussian_5x5', 'clahe'])
  63: ('yolov8s', 'rgb', None)
  64: ('yolov8s', 'concat', None)
  65: ('yolov8s', 'filtered', ['laplacian', 'sobel', 'clahe'])
  66: ('yolov8s', 'filtered', ['gaussian_3x3', 'gaussian_5x5', 'gaussian_7x7'])
  67: ('yolov8s', 'filtered', ['sharpen_basic', 'high_pass_3x3', 'edge_enhance'])
  68: ('yolov8s', 'filtered', ['sobel_x', 'sobel_y', 'laplacian_3x3'])
  69: ('yolov8s', 'filtered', ['laplacian', 'gaussian_5x5', 'clahe'])
  70: ('yolov8m', 'rgb', None)
  71: ('yolov8m', 'concat', None)
  72: ('yolov8m', 'filtered', ['laplacian', 'sobel', 'clahe'])
  73: ('yolov8m', 'filtered', ['gaussian_3x3', 'gaussian_5x5', 'gaussian_7x7'])
  74: ('yolov8m', 'filtered', ['sharpen_basic', 'high_pass_3x3', 'edge_enhance'])
  75: ('yolov8m', 'filtered', ['sobel_x', 'sobel_y', 'laplacian_3x3'])
  76: ('yolov8m', 'filtered', ['laplacian', 'gaussian_5x5', 'clahe'])
  77: ('yolov8l', 'rgb', None)
  78: ('yolov8l', 'concat', None)
  79: ('yolov8l', 'filtered', ['laplacian', 'sobel', 'clahe'])
  80: ('yolov8l', 'filtered', ['gaussian_3x3', 'gaussian_5x5', 'gaussian_7x7'])
  81: ('yolov8l', 'filtered', ['sharpen_basic', 'high_pass_3x3', 'edge_enhance'])
  82: ('yolov8l', 'filtered', ['sobel_x', 'sobel_y', 'laplacian_3x3'])
  83: ('yolov8l', 'filtered', ['laplacian', 'gaussian_5x5', 'clahe'])
```

## 9. Reproducibility notes

To keep your runs reproducible:

- Use `config.yml` as the single source of truth for paths and core settings.
- Keep `PROJECT_ROOT` and `PYTHONPATH` in `.env` in sync with the actual root.
- Use fixed random seeds where possible.
- Record GPU type and Colab runtime details when you produce final results.

With `config.yml` and `.env` set up correctly, you can switch between local Jupyter and Google Colab
without changing code in most notebooks. Only the master execution plans may need a quick check for
path overrides at the top of the file.
