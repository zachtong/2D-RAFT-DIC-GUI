# Technical Decision Record: Strain Calculation Step Parameter

## Context
The application implements a Virtual Strain Gauge (VSG) method for strain calculation using `scipy.ndimage.correlate` to construct the least-squares matrices ($M$ and $b$). The user interface provided a `Step` parameter to allow downsampling the strain calculation grid, theoretically to improve performance.

## Observation
User testing revealed that increasing the `Step` size (e.g., from 1 to 50) resulted in negligible performance gains (e.g., 50s vs 40s).

## Analysis
The strain calculation algorithm consists of two stages:
1.  **Matrix Construction (Convolution)**: Computing the elements of the least-squares system using full-field convolution. This is performed on the **entire image** regardless of the `Step` size to ensure robust noise handling. This involves ~21 convolutions for a 2nd-order polynomial.
2.  **System Solving**: Solving the linear system $M \cdot a = b$ for each point. This is the only step affected by `Step` downsampling.

The testing confirms that **Stage 1 (Convolution) is the dominant bottleneck**. Since `Step` does not reduce the workload of Stage 1, it provides minimal speedup while significantly reducing output resolution.

## Decision
**Disable the `Step` parameter in the UI.**
*   **Action**: Force `Step = 1` for all calculations.
*   **UI Change**: Set the `Step` input field to `state='disabled'` (or readonly) but keep it visible for transparency.
*   **Benefit**: Ensures users always get full-resolution strain fields ($H \times W$) without a false expectation of performance improvement.
*   **Future Work**: True performance optimization would require replacing the full-field convolution with a sparse kernel approach (e.g., using Numba or C++), at which point `Step` could be re-enabled.
