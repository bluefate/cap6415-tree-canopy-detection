# Project Notebooks and Methods Documentation

Notebooks live under `notebooks/` as `.ipynb` files.

## Pipeline (keep these for day-to-day work)

### `00 preflight check.ipynb`

- Environment and dependency verification

### `00_image_extract.ipynb`

- Unpack training/evaluation archives
- Clean folders (e.g. `__MACOSX`)
- Build masks from polygon annotations

### `00_preprocess_images.ipynb`

- Convert TIFF → RGB PNG
- Populate `train_masks` / `evaluation_masks`

### `03_training.ipynb`

- Train/val splits, augmentations, Trainer, versioned checkpoints

### `04_evaluation.ipynb`

- Metrics (IoU, Dice, Accuracy), overlays, summaries

### `05_prediction.ipynb`

- Load a checkpoint, run inference, optional submission JSON

### `06_benchmark_models.ipynb`

- Standalone architecture compare (quality + speed) on a shared validation set

### `07_data_analysis.ipynb`

- Dataset stats, class distributions, image properties

### `08_filter_experimentation.ipynb`

- Systematic filter trials for canopy boundary enhancement

### `10_master_execution_plan_512_rgb.ipynb`

- End-to-end pipeline at tile size 512 (RGB / no-filter experiment set)

### `10_master_execution_plan_512_all.ipynb`

- End-to-end pipeline at tile size 512 including filter experiment variants

### `11 simple model filter test.ipynb`

- Focused model × filter combo experiments

### `12_model_tracker_submission_manager.ipynb`

- Scan checkpoints, rank models, build/refresh submission JSON files

### `_run_512.ipynb`

- Convenience runner that executes the 512 master plan + tracker

## Educational / exploration (`notebooks/explore/`)

### `explore/01_starter_sample.ipynb`

- Early prototyping walkthrough (dataset + sample model sanity checks)
  ![img.png](assets/img.png)

### `explore/02_class.ipynb`

- Mask/class/bbox exploration utilities
  ![img_1.png](assets/img_1.png)

### `explore/02_exploration.ipynb`

- Frequency space, kernels, filters, enhancement visualizations
  ![img_2.png](assets/img_2.png)

## Cross-cutting methodologies

### Data preprocessing

- GeoTIFF → PNG, polygon masks, augmentation policies

### Model development

- U-Net / SimpleCNN / SMP / SegFormer / YOLO via `src.models.zoo`

### Performance tracking

- IoU, Dice, Accuracy; versioned checkpoints; submission export

## Best practices

- Prefer `config.yaml` + `src/` over duplicating logic in notebooks
- Use the 512 master plans for full runs; use `explore/` for teaching and EDA
- Keep configuration-driven, reproducible experiment setups
