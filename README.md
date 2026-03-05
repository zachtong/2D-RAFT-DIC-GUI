# RAFTcorr

GPU-accelerated 2D Digital Image Correlation (DIC) powered by the RAFT optical flow network. Full-field displacement and strain analysis through an interactive web GUI.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zachtong/RAFTcorr/blob/main/notebooks/RAFTcorr_Colab.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.md)

---

## Quick Start (Local — with NVIDIA GPU)

```bash
git clone https://github.com/zachtong/RAFTcorr.git
cd RAFTcorr
pip install -e .
python run_prod.py
```

On Windows, you can double-click **`install.bat`** then **`run.bat`** instead.

The browser opens automatically to `http://localhost:5000`.

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA 11.8+ (required)
- NVIDIA driver installed (`nvidia-smi` should work)

> The installer automatically detects your CUDA version and installs the correct PyTorch build.

## Quick Start (No GPU — Google Colab)

Click the badge above or open this link:

**[Open in Google Colab](https://colab.research.google.com/github/zachtong/RAFTcorr/blob/main/notebooks/RAFTcorr_Colab.ipynb)**

1. Set runtime to **T4 GPU**: Runtime → Change runtime type → T4 GPU
2. **Runtime → Run all** (`Ctrl+F9`)
3. Click the tunnel URL that appears

> Colab provides a free T4 GPU. Visualization may be slower due to network tunneling, but processing is fully GPU-accelerated.

---

## Features

- **Deep Learning Optical Flow** — RAFT models for robust tracking with large displacements or lighting changes
- **CUDA Acceleration** — optimized for NVIDIA GPUs
- **Interactive Web GUI** — browser-based interface, no desktop app needed
- **Full-Field Strain** — Green-Lagrange and engineering strain with customizable virtual strain gauges
- **Virtual Extensometers** — point, line, and area probes with time-series extraction
- **Scientific Export** — MATLAB (`.mat`) and Python (`.npz`) formats with full metadata

## Workflow

1. **Load images** — select a folder of reference + deformed image sequences
2. **Set ROI** — draw a region of interest (rectangle, polygon, or circle)
3. **Process** — run RAFT-DIC displacement tracking
4. **Post-process** — view displacement/strain fields, place probes, analyze time series
5. **Export** — save results for further analysis

## Manual Installation (Linux / Advanced)

```bash
git clone https://github.com/zachtong/RAFTcorr.git
cd RAFTcorr

# Install PyTorch with CUDA (adjust cu124 to your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install RAFTcorr
pip install -e .

# Launch
python run_prod.py
```

## For Developers

To modify the React frontend:

```bash
# Install Node.js 18+, then:
cd frontend
npm install
npm run dev      # dev server with hot reload on port 5173

# In another terminal:
python run_dev.py  # Flask API on port 5000
```

To rebuild the production frontend:

```bash
cd frontend && npm run build
```

## Citation

If this software assists your research, please cite the RAFTcorr repository. (Journal paper in preparation)

## License

MIT License. See [LICENSE.md](LICENSE.md) for details.

## Acknowledgments

- RAFT: [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT)
