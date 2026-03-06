<div align="center">

<!-- TODO: Replace with your custom banner image (recommended: 1280×640px)
     Create in Figma/Canva with dark background + UI screenshots + logo text
     Save to docs/images/banner.png -->
<!-- <img src="docs/images/banner.png" alt="RAFTcorr Banner" width="100%"> -->

# RAFTcorr

### GPU-Accelerated Digital Image Correlation Powered by Deep Learning

Full-field displacement and strain analysis through an interactive web interface,<br>
using the [RAFT](https://github.com/princeton-vl/RAFT) optical flow network with CUDA acceleration.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zachtong/RAFTcorr/blob/main/notebooks/RAFTcorr_Colab.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.md)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

</div>

---

<!-- TODO: Replace with your demo GIF (30-60s workflow recording)
     Record with ScreenToGif or OBS: Load images → Draw ROI → Process → View results → Place probes
     Save to docs/images/demo.gif (keep under 10MB) -->
<!-- <p align="center">
  <img src="docs/images/demo.gif" alt="RAFTcorr Demo" width="800">
</p> -->

## Why RAFTcorr?

Traditional DIC methods rely on subset matching with hand-tuned parameters. RAFTcorr replaces this with **RAFT** (Recurrent All-Pairs Field Transforms), a deep learning optical flow model that:

- Handles **large displacements** without coarse-fine pyramids
- Is **robust to lighting changes**, noise, and low-texture regions
- Runs at **GPU speed** — process a 1000×1000 image pair in under a second

> **No GPU?** Try it free on [Google Colab with a T4 GPU](https://colab.research.google.com/github/zachtong/RAFTcorr/blob/main/notebooks/RAFTcorr_Colab.ipynb).

---

## Features

<table>
<tr>
<td width="50%">

**Analysis**
- Deep-learning optical flow (RAFT-Large / RAFT-Small)
- Accumulative and incremental processing modes
- Green-Lagrange & engineering strain computation
- Principal strain directions with overlay visualization
- Adjustable smoothing (displacement & strain)

</td>
<td width="50%">

**Interactive GUI**
- Browser-based — no desktop app to install
- ROI tools: rectangle, polygon, circle, mask import
- Virtual extensometers: point, line, and area probes
- Real-time colormap visualization with opacity control
- Frame-by-frame playback with pre-render cache

</td>
</tr>
<tr>
<td width="50%">

**Export & Data**
- Scientific export: MATLAB (`.mat`) and NumPy (`.npz`)
- CSV time-series export for probe data
- Batch image export with per-component color range & DPI
- Chart PNG/CSV download for time-series plots
- Physical unit conversion (mm, in, m)

</td>
<td width="50%">

**Performance**
- CUDA-accelerated RAFT inference
- Custom CUDA correlation kernel (`alt_cuda_corr`)
- Memory-aware render cache (auto-sized)
- Concurrent pre-render workers for smooth playback
- One-click install on Windows (`install.bat`)

</td>
</tr>
</table>

---

## Screenshots

<!-- TODO: Replace these with real screenshots showing actual experimental data.
     Take full-window screenshots (16:9) of each tab with data loaded.
     Save to docs/images/ directory. -->

<table>
<tr>
<td align="center"><b>ROI Selection</b></td>
<td align="center"><b>Displacement Field</b></td>
</tr>
<tr>
<td>

<!-- Replace: screenshot with images loaded and ROI drawn -->
![ROI Selection](docs/images/screenshot-roi.png)

</td>
<td>

<!-- Replace: screenshot showing colorful displacement overlay -->
![Displacement](docs/images/screenshot-displacement.png)

</td>
</tr>
<tr>
<td align="center"><b>Strain Analysis</b></td>
<td align="center"><b>Virtual Extensometers</b></td>
</tr>
<tr>
<td>

<!-- Replace: screenshot showing strain field with color bar -->
![Strain](docs/images/screenshot-strain.png)

</td>
<td>

<!-- Replace: screenshot showing probes + time-series chart -->
![Probes](docs/images/screenshot-probes.png)

</td>
</tr>
</table>

---

## Workflow

```mermaid
graph LR
    A["Load Images"] --> B["Draw ROI"]
    B --> C["Run RAFT-DIC"]
    C --> D["Displacement Fields"]
    D --> E["Strain Computation"]
    D --> F["Place Probes"]
    E --> G["Export .mat / .npz / CSV"]
    F --> G

    style A fill:#1e293b,stroke:#3b82f6,color:#e2e8f0
    style B fill:#1e293b,stroke:#3b82f6,color:#e2e8f0
    style C fill:#1e293b,stroke:#8b5cf6,color:#e2e8f0
    style D fill:#1e293b,stroke:#10b981,color:#e2e8f0
    style E fill:#1e293b,stroke:#10b981,color:#e2e8f0
    style F fill:#1e293b,stroke:#10b981,color:#e2e8f0
    style G fill:#1e293b,stroke:#f59e0b,color:#e2e8f0
```

---

## Quick Start

### Local Installation (NVIDIA GPU Required)

```bash
git clone https://github.com/zachtong/RAFTcorr.git
cd RAFTcorr
```

**Windows (one-click):** double-click `install.bat`, then `run.bat`.

**Command line:**

```bash
conda create -n raftcorr python=3.10 -y
conda activate raftcorr
pip install -e .
python run_prod.py
```

The browser opens automatically at `http://localhost:5000`.

<details>
<summary><b>Prerequisites</b></summary>

- [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.anaconda.com/miniconda/)
- NVIDIA GPU with **CUDA 11.8+** (required for RAFT inference)
- NVIDIA driver installed — verify with `nvidia-smi`
- The installer auto-detects your CUDA version and installs the matching PyTorch build

</details>

### No GPU? Use Google Colab (Free)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zachtong/RAFTcorr/blob/main/notebooks/RAFTcorr_Colab.ipynb)

1. Set runtime to **T4 GPU**: *Runtime → Change runtime type → T4 GPU*
2. **Run all** cells (`Ctrl+F9`)
3. Click the tunnel URL that appears

> Colab provides a free T4 GPU. Visualization may be slower due to network tunneling, but processing is fully GPU-accelerated.

---

## Architecture

```mermaid
graph TB
    subgraph Frontend["Frontend (React + TypeScript)"]
        UI["Browser UI<br/>Vite + Tailwind + Zustand"]
    end

    subgraph Backend["Backend (Python)"]
        Flask["Flask + SocketIO"]
        RAFT["RAFT Neural Network"]
        Strain["Strain Engine<br/>(NumPy / Numba)"]
        Cache["Render Cache"]
    end

    subgraph GPU["NVIDIA GPU"]
        CUDA["CUDA Kernels"]
    end

    UI <-->|"REST API + WebSocket"| Flask
    Flask --> RAFT
    Flask --> Strain
    Flask --> Cache
    RAFT --> CUDA

    style Frontend fill:#1e293b,stroke:#3b82f6,color:#e2e8f0
    style Backend fill:#1e293b,stroke:#10b981,color:#e2e8f0
    style GPU fill:#1e293b,stroke:#76b900,color:#e2e8f0
```

<details>
<summary><b>Manual Installation (Linux / Advanced)</b></summary>

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

</details>

<details>
<summary><b>For Developers</b></summary>

To modify the React frontend:

```bash
# Install Node.js 18+, then:
cd frontend
npm install
npm run dev        # dev server with hot reload on port 5173

# In another terminal:
python run_dev.py  # Flask API on port 5000
```

To rebuild the production frontend:

```bash
cd frontend && npm run build
```

</details>

---

## Citation

If RAFTcorr assists your research, please cite:

```bibtex
@software{raftcorr2025,
  author    = {Tong, Zixiang and Bu, Lehu},
  title     = {{RAFTcorr}: GPU-Accelerated Digital Image Correlation Using RAFT Optical Flow},
  year      = {2025},
  url       = {https://github.com/zachtong/RAFTcorr},
  note      = {The University of Texas at Austin}
}
```

<!-- TODO: Update with journal paper citation when published -->

---

## Acknowledgments

- **RAFT**: Teed & Deng, [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT) — *RAFT: Recurrent All-Pairs Field Transforms for Optical Flow* (ECCV 2020)
- Developed at **The University of Texas at Austin**

## License

MIT License — see [LICENSE.md](LICENSE.md) for details.
