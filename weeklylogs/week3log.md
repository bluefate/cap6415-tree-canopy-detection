Solafune Tree Canopy Detection challenge - Week 3 Project Summary
==================================================================

Overview
----------------------------------------

duet to continue amount of crashes instead of using the TIF images, implemented functioanlty to
change images to PNG from TIF due to issues with cv2.imread, likely causing system crashes due to
memory issues with large compressed files.

Why PNG is better for training:

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
