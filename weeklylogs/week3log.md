Solafune Tree Canopy Detection Challenge - Week 3 Project Summary
==================================================================

Overview
----------------------------------------

Week 3 focused on resolving critical system stability issues and establishing data pipeline. The
primary achievement was identifying and solving persistent training crashes caused by TIFF image
handling. Work included implementing format conversion utilities, enhancing data loading mechanisms,
and building comprehensive validation tools to ensure reliable experimentation.

Image Format Conversion & System Stability
----------------------------------------

- Identified TIFF files as the root cause of system crashes during training loops.
- Analyzed issues related to GeoTIFF compression, memory spikes, and inconsistent library support
  across OpenCV versions.
- Implemented comprehensive ImageConverter utility to batch convert TIFF files to PNG format.
- Added automatic annotation file updates with timestamped backup creation and restoration
  capabilities.
- Created image validation tools to verify file integrity before and after conversion.
- Documented technical rationale for PNG preference: consistent memory footprint, faster loading,
  universal library support.
- Maintained original TIFF archives for geospatial metadata preservation while training exclusively
  on PNG.

Enhanced Data Loading Infrastructure
----------------------------------------

- Built image loading utilities with automatic format detection and fallback mechanisms.
- Implemented graceful degradation: OpenCV for speed, automatic fallback to PIL for problematic
  files. (still researching)
- Enhanced ImageMaskDataset with improved tensor conversion handling and consistent shape
  normalization.

Training & Evaluation Pipeline Improvements
----------------------------------------

- Implemented evaluation utilities with batch metrics computation and visualization.
- Continue to Enhanced and Integrated versioning system for checkpoint management and experiment
  reproducibility.
- Expanded model registry to include SimpleCNN, UNet, and segmentation_models_pytorch integrations.
- Implemented benchmarking infrastructure to compare models on shared validation sets.
- Added tracking for IoU, Dice, Accuracy, inference speed, and GPU memory usage.
- Created visualization and CSV export for performance analysis across architectures.

Results & Impact
----------------------------------------

- Eliminated all training crashes, enabling reliable multi-epoch training runs.
- Achieved faster image loading during training with reduced CPU overhead.
- Established stable DataLoader worker processes with consistent memory usage.
- Improved development velocity through reproducible experimentation environment.

Why PNG may be better for training:
================================

**Memory & Stability:**

- TIFFs (especially GeoTIFFs used in aerial imagery) can have complex compression, multiple layers,
  and metadata that causes unpredictable memory spikes
- PNG has simple, well-supported compression that loads consistently

**Speed:**

PNG loads faster in OpenCV/PIL during training
TIFF decompression is CPU-intensive, slowing down your data pipeline
For iterative training over thousands of epochs, this adds up significantly

**Compatibility:**

Every library (OpenCV, PIL, PyTorch, etc.) handles PNG perfectly
TIFF support varies wildly between libraries and versions

**TIFF is better when**

TIFFs contain critical geospatial metadata (coordinate systems, projection info) that you need for
final deployment, keep the original TIFFs archived but still train on PNGs.
