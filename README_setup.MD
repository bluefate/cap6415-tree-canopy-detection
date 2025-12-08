# Project Setup and Execution Guide

## Local Development Environment Setup

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU support)
- Conda or Virtual Environment recommended

### Installation Steps

1. Clone the Repository

```bash
git clone https://github.com/bluefate/CAP6415_F25_project-Tree-Canopy-Detection.git
cd CAP6415_F25_project-Tree-Canopy-Detection
```

2. Create Virtual Environment

```bash
# Using Conda
conda create -n treecanopy python=3.8
conda activate treecanopy

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install Dependencies

```bash
pip install -r requirements.txt
```

4. Verify Installation

```bash
python notebooks/00_preflight_check.py
```

## Google Colab Setup

### Option 1: Direct Notebook Import

1. Open Google Colab
2. Go to File > Open Notebook
3. Select "GitHub" tab
4. Enter repository URL: `bluefate/CAP6415_F25_project-Tree-Canopy-Detection`
5. Select desired notebook

### Option 2: Manual Setup

```python
# Mount Google Drive
from google.colab import drive


drive.mount('/content/drive')

# Clone Repository
!git
clone
https: // github.com / bluefate / CAP6415_F25_project - Tree - Canopy - Detection.git

# Install Dependencies
!pip
install - r / content / CAP6415_F25_project - Tree - Canopy - Detection / requirements.txt

# Set Working Directory
import os


os.chdir('/content/CAP6415_F25_project-Tree-Canopy-Detection')
```

## Execution Workflow

### Local Execution

```bash
# Run specific notebook
jupyter notebook notebooks/03_training.py

# Or use Python directly
python notebooks/05_prediction.py
```

### Colab Execution

- Open notebook in Colab
- Runtime > Run All
- Select GPU runtime for optimal performance

## Troubleshooting

### Common Issues

1. **CUDA Version Mismatch**
    - Ensure CUDA versions match your GPU drivers
    - Update NVIDIA drivers if necessary

2. **Missing Dependencies**
    - Check `requirements.txt`
    - Install any missing packages manually

3. **Data Path Issues**
    - Verify data paths in configuration files
    - Use absolute or relative path settings

### Performance Optimization

- Use GPU runtime in Colab
- Limit batch sizes for memory constraints
- Monitor GPU memory usage

## Reproducibility

### Recommended Settings

- Use fixed random seeds
- Specify exact package versions
- Document hardware specifications

## Competition Submission Preparation

1. Train your best model
2. Run prediction notebooks
3. Generate submission JSON
4. Validate against competition guidelines

## Additional Resources

- [Solafune Competition Page](https://solafune.com/competitions/26ff758c-7422-4cd1-bfe0-daecfc40db70)
- Project [README.md](README.md)
- Notebooks [Documentation](README_DOCS.MD)

### Note

Always refer to the most recent project documentation for the latest setup instructions.