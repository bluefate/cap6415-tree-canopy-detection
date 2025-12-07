# Tree Canopy Detection

Project for the **Solafune Tree Canopy Detection** competition.

Competition
URL: [Solafune – Tree Canopy Detection](https://solafune.com/competitions/26ff758c-7422-4cd1-bfe0-daecfc40db70)

## Project Abstract

### Problem Statement

Tree canopy detection from aerial and satellite imagery is a critical challenge in environmental
monitoring, urban planning, and climate research. The Solafune Tree Canopy Detection competition
aims to develop robust machine learning models capable of accurately segmenting and identifying tree
canopies in high-resolution RGB imagery.

### Technical Challenge

The project tackles several key technical challenges:

- Precise segmentation of tree canopies in complex, multi-resolution satellite and aerial images
- Handling diverse geographical landscapes with varying tree densities and types
- Developing a flexible deep learning pipeline that can generalize across different image
  characteristics

### Proposed Solution

Developed a comprehensive machine learning approach using:

- Advanced image preprocessing and augmentation techniques
- Semantic segmentation architectures (U-Net, SimpleCNN)
- Sophisticated filtering and enhancement strategies
- Robust model training and evaluation infrastructure

### Methodology

- **Data Preprocessing**: Convert GeoTIFF images to consistent PNG format
- **Annotation Handling**: Process polygon-based JSON annotations
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
    - Semantic segmentation models can be used, but you must adapt them for **instance segmentation
      ** or apply **post-processing** (e.g., watershed with distance maps + Gaussian smoothing +
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

## Setup

- **Structure**

```
mkdir -p {data/{raw/{images,annotations},processed/{tiles,masks},splits,sample_submission},notebooks,models,src/{data,models,scripts,utils,submission},submission}

tree /mnt/c/github/Tree-Canopy-Detection -I ".ipynb_checkpoints|*.tif" --prune

tree /mnt/c/github/Tree-Canopy-Detection -d

cd "C:\github\Tree-Canopy-Detection"
```

- **Build Docker Environment**

```
docker build -t tree-canopy-detection .
docker run --gpus all -it tree-canopy-detection
```

- **Local Development**

```
pip install -r requirements.txt
jupyter lab /mnt/c/github/Tree-Canopy-Detection
```

- **Tree**

```
/Tree-Canopy-Detection
├── LICENSE
├── README.md
└── src
    └── data
        ├── evaluation_images.zip
        ├── sample_answer.json
        ├── train_annotations.json
        └── train_images.zip
```


