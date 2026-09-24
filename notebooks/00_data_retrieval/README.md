# 00 — Data retrieval

Get imagery and labels onto disk in the layout the rest of the pipeline expects.
Run in order when starting from scratch.

| Notebook | Purpose |
|----------|---------|
| `00_build_public_sample.ipynb` | **Public mode only.** Download NEON crowns and convert to Solafune-shaped PNGs + JSON under `data/public_sample/`. Skip when using local competition data. |
| `01_preflight_check.ipynb` | Confirm env, paths, and active dataset from `config.yaml`. Downloads configured files if missing; nudges to run `00` when the public sample is not built. |
| `02_image_extract.ipynb` | Unpack train/eval zips, clean junk folders, build masks from polygon annotations. |
| `03_preprocess_images.ipynb` | TIFF → PNG (competition imagery). No-op-ish when public sample already shipped PNGs. |

Paths come from `config.yaml` (`dataset.active`: `public` or `competition`). After this folder, continue with `04_starter_sample.ipynb` and the rest under `notebooks/`.
