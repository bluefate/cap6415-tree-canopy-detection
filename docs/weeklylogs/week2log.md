Solafune Tree Canopy Detection challenge - Week 2 Project Summary
==================================================================

Overview
----------------------------------------

Work this week expanded the project from initial dataset preparation into annotation analysis, mask
generation, exploratory filtering, enhancement pipelines, and early segmentation model prototyping.
The environment now supports structured experimentation and visual diagnostics across multiple
stages of the workflow.

Annotation Analysis & Mask Generation
----------------------------------------

- Added utilities to load, inspect, and validate JSON polygon annotations.
- Implemented conversion of polygons into bounding boxes for individual and group tree classes.
- Built functions to generate binary and color coded masks from the annotation data.
- Added functions to visualize masks and bounding boxes on images for quality checks.

Exploration Tools & Notebooks
----------------------------------------

- Began refactoring notebooks into different sections for better item exploration.
- Implemented side by side visualization utilities.
- Added frequency domain exploration using FFT magnitude visualizations.
- Created a kernel library.
- Added functions to visualize kernels as heatmaps and 3D surfaces.
- Implemented parameterized kernel generators for controlled experiments.
- Added visualization functions to inspect each enhancement during stage enhancement.
- Added unified exploration functions for kernels, filters, transforms, and enhancement steps.
- Verified behavior of kernel responses, enhancement outputs, and sampling changes on random images.

Continue Model Prototyping
----------------------------------------

- Build more on a lightweight SimpleCNN encoder/decoder segmentation model for CPU based
  experiments.
- Created utilities to train separate models for individual tree masks and group of trees masks.
- Added Zenodo (for testing purposes) dataset loader to support cross dataset experimentation and
  normalization.
