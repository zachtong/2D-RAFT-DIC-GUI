# 2D-RAFT-DIC-GUI

A graphical user interface for Digital Image Correlation (DIC) using RAFT neural network.

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Overview

2D-RAFT-DIC-GUI is a powerful tool that combines the RAFT (Recurrent All-Pairs Field Transforms) neural network with Digital Image Correlation techniques. It provides an intuitive graphical interface for analyzing displacement fields between image pairs, making it particularly useful for material deformation analysis and strain measurement.

## Features

- 🖼️ Interactive ROI (Region of Interest) selection and editing
- 🔄 Support for both accumulative and incremental DIC modes
- 🎯 Customizable processing parameters (crop size, stride, etc.)
- 🔍 Real-time preview of displacement fields
- 📊 Visualization of U/V displacement components
- 🎬 Animation playback of displacement results
- 🔧 Gaussian smoothing for noise reduction
- 💾 Multiple export formats (NPY, MAT)

## Getting Started

### Prerequisites

To ensure 2D-RAFT-DIC-GUI runs smoothly, please make sure your system meets the following requirements:

* **Python 3.11 or higher**
* **NVIDIA CUDA-capable GPU** (RTX 40 series or newer recommended)
* **CUDA Toolkit 12.8 and corresponding cuDNN** must be installed.

Above configurations are the writer's, we haven't test 

### Installation Guide

We highly recommend using **Conda** to manage project dependencies. It effectively handles complex compatibility issues between PyTorch and CUDA, providing the most convenient installation experience.

#### 1. Install Miniconda (if you don't have it)

Visit the [Miniconda Official Website](https://docs.conda.io/en/latest/miniconda.html) to download and install Miniconda for your operating system. After installation, you may need to close and reopen your terminal (or command prompt) for `conda` commands to be available.

#### 2. Clone the Repository

First, clone the 2D-RAFT-DIC-GUI repository to your local machine:

```bash
git clone https://github.com/zachtong/2D-RAFT-DIC-GUI.git
```
#### 4. Create environment 
```bash
conda env create -f environment.yml
conda activate your_project_env # 'your_project_env' is the environment name defined in environment.yml
```

### Model Setup

1. Download the RAFT model weights from [TBD]
2. Place the model file (`raft-dic_v1.pth`) in the `models` directory

## Usage

1. Launch the GUI:
```bash
python main_GUI.py
```

2. Basic workflow:
   - Select input image directory
   - Choose output directory
   - Draw/edit ROI on reference image
   - Adjust processing parameters
   - Run analysis
   - View and export results

### Parameter Configuration

- **Processing Mode**:
  - Accumulative: Compare all images to first frame
  - Incremental: Compare each image to previous frame

- **Crop Parameters**:
  - Crop Size: Size of processing windows (H×W)
  - Stride: Step size between windows
  - Max Displacement: Maximum expected displacement

- **Smoothing Options**:
  - Enable/disable Gaussian smoothing
  - Adjust sigma value (0.5-5.0)

## Results

The program generates:
- NPY files containing raw displacement fields
- MAT files for MATLAB compatibility
- Visualization of displacement components
- Window layout plots

## Contributing

We welcome contributions! Please feel free to submit pull requests, report bugs, and suggest features.

## Authors

- Zixiang (Zach) Tong @ UT-Austin
- Lehu Bu @ UT-Austin

## Citation

If you use this software in your research, please cite:

```bibtex
[TBD]
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- RAFT implementation based on the original work by [Zachary Teed and Jia Deng](https://github.com/princeton-vl/RAFT)
