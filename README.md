# RAFTcorr

An interactive desktop GUI for 2D Digital Image Correlation (DIC) powered by the RAFT (Recurrent All-Pairs Field Transforms) optical flow network. This tool provides a robust, CUDA-accelerated workflow for calculating full-field displacement and strain from image sequences.

## Documentation
**[Download User Manual (v1.0 PDF)](RAFTcorr_user_manual_v1.0.pdf)**  
*For a complete guide on installation, workflow, parameters, and troubleshooting.*

## Features
*   **Deep Learning Optical Flow**: Uses RAFT models (Large/Fine) for robust tracking even with large displacements or lighting changes.
*   **CUDA Acceleration**: Optimized for NVIDIA GPUs to handle high-resolution images.
*   **Virtual Strain Gauge and Probe Analysis**: Extract time-series data from Points, Lines, and Areas.
*   **Full-Field Strain**: Green-Lagrange and Engineering strain calculations with customizable virtual strain gauges (VSG).
*   **Scientific Export**: Save results to MATLAB (`.mat`) or Python (`.npz`) formats with full metadata.

## Prerequisites

- **OS**: Windows 10/11
- **Python**: 3.8+
- **GPU**: NVIDIA GPU with CUDA 11.8+ (Required)

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/zachtong/RAFTcorr.git
    cd RAFTcorr
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Verify your GPU setup:
    ```bash
    python verify_installation.py
    ```

## Quick Usage

1.  **Launch**: `python main_GUI.py`
2.  **Input**: Select your image folder in the "Path Settings".
3.  **Model**: Choose `RAFT-Large` (default) or `RAFT-Fine`.
4.  **ROI**: Draw a Region of Interest (Rectangle/Polygon/Circle) on the reference image and click **Confirm ROI**.
5.  **Run**: Click **Run** to start processing.
6.  **Analysis**: 
    *   Switch to the **Post-Processing** tab to calculate Strain.
    *   Use **Probe Analysis** to plot displacement/strain over time.
    *   Export data via the **Data Export** section.

## Configuration (Optional)
Customize branding and defaults via `assets/app_config.json`:
```json
{
  "app_title": "RAFTcorr",
  "appearance_mode": "system",
  "color_theme": "blue"
}
```

## Citation
If this software assists your work, please cite the RAFTcorr repository. (Journal paper in preparation)

## License
MIT License. See `LICENSE.md` for details.

## Acknowledgments
- Original RAFT Official Repo: https://github.com/princeton-vl/RAFT
