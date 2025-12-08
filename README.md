# Tree Canopy Detection

Project for the **Solafune Tree Canopy Detection** competition.

Competition
URL: [Solafune – Tree Canopy Detection](https://solafune.com/competitions/26ff758c-7422-4cd1-bfe0-daecfc40db70)

## Project Abstract

I worked on the Solafune Tree Canopy Detection challenge, where the goal is to detect tree canopy
from high resolution RGB aerial and satellite imagery. Each training image is provided with polygon
annotations for tree objects, and the competition evaluates how accurately a single JSON submission
recovers both individual and grouped tree canopies.

In this project I treated the task as a semantic segmentation problem and built an end to end
pipeline around it. I converted the raw TIFF images into a consistent RGB PNG format, read the
polygon labels from JSON, tiled large scenes into patches, and trained several models using PyTorch.
I experimented with both simple baselines and advanced encoder-decoder architectures, used
configurable data augmentation and loss functions, and tracked validation metrics through a
structured experiment
manager.

To turn trained checkpoints into usable models, I added a model tracker and a unified Predictor
interface. The tracker scans all experiment folders, records metrics and paths, ranks models, and
writes a best model report. The predictor can load any selected checkpoint, run sliding window
inference over full scenes, aggregate probabilities, convert multi class masks into polygons, filter
out small artifacts, and write competition-ready JSON files that preserve the required metadata. I
also provide Colab-friendly scripts so that anyone can start from the raw data in Google Drive and
reproduce training, model selection, and final submission generation in a single, predictable
workflow.

For running, setup, and documentation, please see below:

- Project [Setup Guide](README_setup.md)
- Notebooks [Documentation](README_DOCS.md)

------

### Problem Statement

Tree canopy detection from aerial and satellite imagery is a critical challenge in environmental
monitoring, urban planning, and climate research. The Solafune Tree Canopy Detection competition
aims to develop robust machine learning models capable of accurately segmenting and identifying tree
canopies in high-resolution RGB imagery.

### Technical Challenge

Here is a list of issues I ran into, expanded and cleaned up.

**Limited GPU access**

- Training segmentation models on high-resolution imagery is very compute heavy. I often had to use
  mid-range or shared GPUs, which limited batch size, model size, and experimentation speed.

**Image resolution tradeoffs**

- If I resized images or tiles too small, the model lost fine tree detail, which is critical for
  aerial and satellite imagery.
- If I kept images or tiles too large, memory usage and training time went up and required
  more expensive hardware.
- I had to balance tile size, stride, and batch size to avoid out-of-memory errors while still
  keeping
  enough spatial detail.

**Long training times**

- Stronger encoder-decoder architectures with large input sizes took many epochs to converge.
- Running full experiment grids across models, image sizes, and augmentations was slow under
  hardware limits.

**Sliding window inference cost**

- For full scene inference I needed sliding windows to handle large images.
- This increased prediction time because many overlapping patches had to be processed and
  aggregated.

**Memory and storage pressure**

- High-resolution tiles, masks, and model checkpoints used a lot of disk space in Google Drive and
  Colab.
- Keeping multiple versions per model and per experiment run required careful cleanup and
  organization.

**Data preprocessing complexity**

- Converting between TIFF and PNG while keeping consistent normalization and channels required care.
- Polygon-to-mask and mask-to-polygon conversions introduced corner cases around boundaries and tiny
  objects.

**Small object sensitivity**

- Many of the tree crowns were small relative to the full image.
- Small objects are easy to lose through resizing, pooling, or aggressive post-processing filters.

**Class and label noise**

- Real-world annotations can have boundary noise and label uncertainty around tree edges and
  overlapping canopies.
- Small errors in annotation or rasterization can hurt metric scores even if predictions look
  visually reasonable.

**Reproducibility in Colab and local environments**

- Colab and Jupyter sessions can disconnect or reset, which risked losing environment
  state or partial results.
- I needed extra coding to make sure that if a timeout occurred, rerunning the notebook could
  recover
  gracefully.

**Submission format**

- The competition submission required a very specific JSON structure and metadata.
- Small formatting mistakes or missing fields could invalidate an otherwise correct prediction set.

### Proposed Solution

Developed a comprehensive machine learning approach using:

- Advanced image preprocessing and augmentation techniques
- Semantic segmentation architectures (U-Net, SimpleCNN)
- Sophisticated filtering and enhancement strategies
- Robust model training and evaluation infrastructure

### Methodology

- **Data Preprocessing**: Converted GeoTIFF images to consistent PNG format
- **Annotation Handling**: Processed polygon-based JSON annotations
- **Model Development**:
    * Implement multiple segmentation model architectures
    * Create flexible training and validation workflows
    * Build sophisticated model tracking and benchmarking systems
- **Inference**: Develop a unified prediction pipeline for competition submission

### Key Techniques

- Used dynamic image enhancement techniques
- Added modular model and experiment management
- Implemented advanced visualization and performance tracking tools

### Potential Impact

This research contributes to:

- Improved techniques for automated tree canopy mapping
- Enhanced environmental monitoring capabilities
- Demonstrating machine learning's potential in geospatial analysis

## Summary

This repository contains code and notebooks to train and evaluate models that segment tree canopy
from RGB TIFF aerial and satellite imagery.

The competition uses **polygon-based JSON annotations**. Submissions are a **single JSON file** in
the prescribed schema.

**Source summary from the competition page:**

- Images are **RGB TIFFs** (3-band).
- Training annotations contain **polygon segmentations** with `class` and `confidence_score` fields.
- Submissions must be **one JSON file** that matches the sample format.
- See the competition overview for details.

## Notes from Discussion Board

- **Environment**
    - Must provide a **Dockerfile** describing the environment used.
    - If using NVIDIA GPUs, ensure support for **CUDA 11.8+**.
- **Models**
    - The **YOLO series models from Ultralytics** are explicitly allowed.
    - Semantic segmentation models can be used, but you must adapt them for **instance segmentation** or apply **post-processing** (e.g., watershed with distance maps + Gaussian smoothing +
      local peak detection).
- **Evaluation**
    - You do **not** need to train a classification model for `scene_type` or `cm_resolution`.
      These are **internal weighting factors** used only during evaluation.
    - For submissions:
        1. `scene_type` and `cm_resolution` are not prediction targets.
        2. Match the **sample submission schema** exactly.
        3. If unsure, copy the structure from the provided sample.
- **Licensing**
    - Ultralytics YOLO models are allowed.
    - **GPL/AGPL-licensed software is prohibited** (due to copyleft restrictions).
- **AI Assistant Usage**
    - Using AI assistants (e.g., for code generation/review) is allowed.
    - **Do not upload raw datasets** to external AI services — this counts as releasing the data.
    - Do not publish your **final solution** publicly (e.g., GitHub) during the competition period.

**Helpful Discussion Links:**

- [Data Dictionary](https://solafune.com/competitions/26ff758c-7422-4cd1-bfe0-daecfc40db70?menu=discussion&tab=&id=&topicId=fb298f0f-7ca5-426a-9fd8-c41efa0de87e)
- [Converting to COCO format annotation](https://solafune.com/competitions/26ff758c-7422-4cd1-bfe0-daecfc40db70?tab=&menu=discussion&page=2&topicId=d7a7a13d-e8e7-489d-b8d9-7dca36ae30b7)
- [Submission preparation code](https://solafune.com/competitions/26ff758c-7422-4cd1-bfe0-daecfc40db70?menu=discussion&tab=&id=&page=2&topicId=05a2491b-2094-4f9d-b4b6-e7d38d3f13e0)

## License

- This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.



