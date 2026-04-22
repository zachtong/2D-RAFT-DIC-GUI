<div align="center">

<!-- TODO: Add banner image (recommended: 1280x640px, dark background + UI screenshots + logo)
     Save to docs/images/banner.png -->
<!-- <img src="docs/images/banner.png" alt="RAFTcorr Banner" width="100%"> -->

# RAFTcorr

### GPU-Accelerated Digital Image Correlation Powered by Deep Learning

Full-field displacement and strain analysis through an interactive web interface,<br>
built on the [RAFT](https://github.com/princeton-vl/RAFT) optical flow network with CUDA acceleration.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zachtong/RAFTcorr/blob/main/notebooks/RAFTcorr_Colab.ipynb)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

</div>

---

<!-- TODO: Add demo GIF (30-60s workflow recording)
     Record: Load images -> Draw ROI -> Process -> View displacement -> Place probes
     Tools: ScreenToGif or OBS, save to docs/images/demo.gif, keep under 10MB -->
<!-- <p align="center">
  <img src="docs/images/demo.gif" alt="RAFTcorr Demo" width="800">
</p> -->

## Why RAFTcorr?

Traditional DIC relies on iterative subset matching — users must hand-tune subset size, step size, strain windows, and convergence criteria. Performance degrades on large deformations, low-texture surfaces, and noisy images.

RAFTcorr replaces the entire correlation pipeline with **RAFT** (Recurrent All-Pairs Field Transforms), a deep learning optical flow model trained specifically for DIC.

### How RAFTcorr compares

|  | Ncorr | DICe | OpenCorr | VIC-2D/3D | ZEISS Correlate | **RAFTcorr** |
|---|---|---|---|---|---|---|
| **Algorithm** | Subset (IC-GN) | Subset + Global | Subset (IC-GN/NR) | Subset (proprietary) | Facet-based | ${\color{green}\textsf{Deep learning (RAFT)}}$ |
| **Sub-pixel accuracy** | ${\color{green}\textsf{0.01 px}}$ | 0.01–0.05 px | ${\color{green}\textsf{0.01 px}}$ | ${\color{green}\textsf{0.01 px}}$ | ${\color{green}\textsf{0.01 px}}$ | ${\color{red}\textsf{0.03–0.08 px}}$ ¹ |
| **Dense-field speed** | ${\color{red}\textsf{Minutes (CPU)}}$ | ${\color{red}\textsf{Minutes (MPI/CPU)}}$ | Fast (CPU+GPU) | Fast (CPU) | Fast (CPU) | ${\color{green}\textsf{Seconds (GPU)}}$ |
| **Large displacement** | ${\color{red}\textsf{Half subset}}$ | ${\color{red}\textsf{Half subset}}$ | ${\color{red}\textsf{Half subset}}$ | Moderate | Moderate | ${\color{green}\textsf{50+ px native}}$ |
| **Low-texture** | ${\color{red}\textsf{Poor}}$ | Moderate ² | Moderate | Moderate | Moderate | ${\color{green}\textsf{Strong}}$ |
| **Parameters to tune** | ${\color{red}\textsf{5–8}}$ | ${\color{red}\textsf{Many}}$ | ${\color{red}\textsf{5+}}$ | 3–5 (guided) | 3–5 (guided) | ${\color{green}\textsf{0 — neural network}}$ |
| **GPU acceleration** | ${\color{red}\textsf{No}}$ | ${\color{red}\textsf{No}}$ | ${\color{green}\textsf{CUDA (IC-GN)}}$ | ${\color{red}\textsf{No}}$ | ${\color{red}\textsf{No}}$ | ${\color{green}\textsf{CUDA native}}$ |
| **2D / 3D / DVC** | ${\color{red}\textsf{2D only}}$ | ${\color{green}\textsf{2D + Stereo}}$ | ${\color{green}\textsf{2D + Stereo + DVC}}$ | ${\color{green}\textsf{2D + 3D}}$ | 2D (free) + 3D | ${\color{red}\textsf{2D only}}$ ³ |
| **Strain** | Green-Lagrange | Robust (LSQ) | 2D + 3D surface | ${\color{green}\textsf{Comprehensive}}$ | ${\color{green}\textsf{Full}}$ | Green-Lagrange + eng. |
| **Platform** | ${\color{red}\textsf{MATLAB}}$ | C++ (cross) | C++ (cross) | ${\color{red}\textsf{Windows}}$ | ${\color{red}\textsf{Windows}}$ | ${\color{green}\textsf{Browser (cross-platform)}}$ |
| **Cost** | Free (MATLAB req.) | ${\color{green}\textsf{Free (BSD-3)}}$ | ${\color{green}\textsf{Free (MPL-2.0)}}$ | ${\color{red}\textsf{5K–150K+ USD}}$ | Free (2D) / paid (3D) | ${\color{green}\textsf{Free (UT Research License)}}$ ⁴ |
| **Open source** | ${\color{green}\textsf{Yes}}$ | ${\color{green}\textsf{Yes}}$ | ${\color{green}\textsf{Yes}}$ | ${\color{red}\textsf{No}}$ | ${\color{red}\textsf{No}}$ | ${\color{green}\textsf{Source-available}}$ ⁵ |
| **Development** | ${\color{red}\textsf{Dormant (~2019)}}$ | ${\color{green}\textsf{Active (v3.0)}}$ | ${\color{green}\textsf{Active (2025)}}$ | ${\color{green}\textsf{Active (v11)}}$ | ${\color{green}\textsf{Active (2025)}}$ | ${\color{green}\textsf{Active}}$ |

<sup>¹ DIC-optimized RAFT models achieve ~0.03–0.08 px; standard IC-GN methods reach ~0.01 px. Deep learning DIC trades modest sub-pixel precision for order-of-magnitude gains in speed, robustness, and ease of use. Active research is closing this gap.</sup><br>
<sup>² DICe offers a simplex (gradient-free) optimizer that improves robustness in low-contrast regions.</sup><br>
<sup>³ Stereo 3D DIC support is planned for a future release.</sup><br>
<sup>⁴ Free for academic, research, and internal non-commercial use. Commercial distribution requires a separate license — contact licensing@discoveries.utexas.edu. See [LICENSE.md](LICENSE.md).</sup><br>
<sup>⁵ Full source code is publicly available and modifiable for non-commercial use. The UT Research License restricts commercial distribution, so it is not OSI-approved "open source" in the strictest sense — it is best described as *source-available*.</sup>

<details>
<summary><b>vs. open-source DIC tools (Ncorr, DICe, OpenCorr, muDIC)</b></summary>

- **Algorithm generation gap** — Ncorr, DICe, and OpenCorr all use subset-based IC-GN optimization, an approach from the 2000s. RAFTcorr uses deep learning optical flow, which handles sparse textures and large deformations where subset methods fail.
- **Zero-parameter workflow** — Traditional DIC requires careful tuning of subset size, step size, strain window, seed points, and convergence criteria. RAFTcorr requires none of these — the neural network handles everything automatically.
- **Modern platform** — Browser-based GUI vs. MATLAB (Ncorr) or C++ requiring compilation (DICe, OpenCorr). No MATLAB license, no build toolchain needed.
- **vs. OpenCorr specifically** — OpenCorr is the strongest open-source traditional DIC tool: actively maintained, GPU-accelerated IC-GN, and supports 2D + Stereo 3D + DVC. However, it still requires traditional parameter tuning and is limited by subset-based displacement range. RAFTcorr's deep learning approach is fundamentally different — zero parameters, native large-displacement support, and stronger robustness to degraded speckle patterns.
- **Honest trade-off** — Traditional IC-GN DIC (including OpenCorr) achieves ~0.01 px sub-pixel accuracy, roughly 3–8× better than current deep learning methods. For applications where sub-pixel precision is paramount (e.g., measuring <100 microstrain), traditional DIC may still be the better choice. RAFTcorr excels where speed, large deformation, and robustness matter more than extreme sub-pixel precision.

</details>

<details>
<summary><b>vs. commercial software (VIC-2D/3D, ZEISS Correlate, MatchID)</b></summary>

- **Free** — Eliminates the biggest barrier. Labs that cannot afford $5K–150K+ licenses can run full-field DIC at no cost.
- **Open source & customizable** — Commercial DIC tools are black boxes. RAFTcorr is fully transparent — inspect, modify, and extend every line of code.
- **Algorithm transparency** — Academic users need to understand and cite the methods they use. Commercial software cannot provide this level of reproducibility.
- **Direct access to the authors** — File an issue, get a response from the people who built the algorithm.
- **Honest trade-off** — Commercial tools (especially VIC-2D/3D and MatchID) offer mature 3D stereo DIC, extensive validation against engineering standards (ASTM, iDICs), uncertainty quantification, and decades of industrial trust. RAFTcorr is currently 2D-only and has not yet undergone standardized benchmark validation. Choose the right tool for your application.

</details>

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

<!-- TODO: Replace with real screenshots showing actual experimental data.
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
@software{raftcorr2026,
  author    = {Tong, Zixiang and Bu, Lehu and Shi, Qihang and Du, Runtian and Yang, Jin},
  title     = {{RAFTcorr}: An Open-Source, Deep Learning Digital Image Correlation Framework for Dense Displacement Measurement},
  url       = {https://github.com/zachtong/RAFTcorr},
  note      = {The University of Texas at Austin}
}
```

<!-- TODO: Update with journal paper citation when published -->

---

## Acknowledgments

- **RAFT**: Teed & Deng, [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT) — *RAFT: Recurrent All-Pairs Field Transforms for Optical Flow* (ECCV 2020)
- **RAFT-DIC**: Pan, B. and Liu, Y., *User-independent, accurate and pixel-wise DIC measurements with a task-optimized neural network*, Experimental Mechanics, 2024.
- Developed at **The University of Texas at Austin**

## License

RAFTcorr is distributed under **The University of Texas at Austin Research License, Version 1.0**.

- **Free** for academic, research, experimental, personal, consulting, and internal research & development use.
- **Commercial distribution is not permitted** under this license. For commercial use, please contact **licensing@discoveries.utexas.edu**.
- Derivative works are allowed under the same license terms (see Section 3.2).

See [LICENSE.md](LICENSE.md) for the full license text.
