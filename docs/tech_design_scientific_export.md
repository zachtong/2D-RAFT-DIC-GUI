# Technical Design: Scientific Data Export Module

## Overview
This document outlines the design for the "Scientific Data Export" module in RAFT-DIC. The goal is to provide a robust, lossless, and standard-compliant export format for downstream scientific analysis in MATLAB and Python.

## Design Decisions

### 1. File Formats
*   **Formats**: `.mat` (MATLAB) and `.npz` (Python/NumPy).
*   **Rationale**:
    *   **Efficiency**: Both are binary formats that support compression, significantly smaller than CSV for large 3D/4D arrays.
    *   **Structure**: Both support hierarchical or dictionary-like structures, allowing metadata and multi-dimensional arrays to be stored together.
    *   **Ecosystem**: These are the native formats for the two most dominant scientific computing platforms.

### 2. Data Structure (Hierarchical)
To minimize redundancy and maximize usability, data is organized as follows:

#### A. Static Data (Stored Once)
*   `X_ref`: $(H, W)$ Matrix. Reference X coordinates.
*   `Y_ref`: $(H, W)$ Matrix. Reference Y coordinates.
*   `ROI_mask`: $(H, W)$ Boolean Matrix. 1 for valid ROI pixels, 0 for background.
*   **Rationale**: Since the reference grid is constant (Lagrangian description), storing it once saves significant space compared to per-frame storage.

#### B. Dynamic Data (Time-Series)
*   **Displacement**:
    *   `U`: $(T, H, W)$ Matrix. Horizontal displacement.
    *   `V`: $(T, H, W)$ Matrix. Vertical displacement.
*   **Strain**:
    *   `Exx`, `Eyy`, `Exy`: $(T, H, W)$ Matrices. Strain components.
    *   `E1`, `E2`: $(T, H, W)$ Matrices. Principal strains.
    *   `E_von_Mises`: $(T, H, W)$ Matrix. Equivalent von Mises strain. **(Note: Explicitly named `E_von_Mises`, not `E_eff`)**.
*   **Rationale**: Storing as 3D stacks allows for efficient slicing in time (`data[t,:,:]`) or space (`data[:,y,x]`). Separating components (U vs V) is often more convenient for plotting than interleaved arrays.

#### C. Metadata
*   `VSG_size`: (int) Virtual Strain Gauge window size.
*   `Strain_Method`: (str) e.g., "Green-Lagrange".
*   `Poly_Order`: (int) Polynomial order used in VSG.
*   `Weighting`: (str) "Gaussian" or "Uniform".
*   `Pixel_Ratio`: (float) User-defined pixel-to-physical unit ratio.
*   `Physical_Unit`: (str) e.g., "mm", "um".
*   **Rationale**: Ensures the data is self-describing. A user opening the file years later will know exactly how the results were calculated and what the physical scale is.

### 3. Step Parameter Handling
*   **Policy**: Always export at **Full Resolution** (Step=1).
*   **Rationale**: Even if internal calculation allowed skipping pixels, the exported data must be aligned with the `X_ref/Y_ref` grid to allow direct superposition and analysis. (Note: Step is currently disabled in UI anyway).

## Exclusions
*   **CSV Export**: Full-field CSV export is excluded due to file size and performance concerns. Users needing CSV should use the "Probe Export" feature for specific points of interest.
