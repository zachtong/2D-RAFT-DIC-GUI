# Post-Processing Enhancements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade the RAFT-DIC post-processing pipeline from research-grade to commercial-grade: modular code, numerical tests, richer algorithms, and better UX.

**Architecture:** Backend-first approach — refactor `processing.py` into modules, add analytic-solution tests, then layer on new computations (strain rate, confidence, quality metrics) and wire them through existing Flask routes to the React frontend.

**Tech Stack:** Python (NumPy, SciPy, Numba, OpenCV), Flask, React + TypeScript + Zustand + Tailwind

---

## Phase 1: Modular Refactor + Numerical Tests (Foundation)

> Split the 1700-line `processing.py` into focused modules and add analytic-solution tests that verify numerical correctness. Everything else builds on this.

### Task 1.1: Extract strain module

**Files:**
- Create: `raft_dic_gui/strain.py`
- Modify: `raft_dic_gui/processing.py:822-1181`
- Modify: `server/routes/strain.py:9` (import path)

**Step 1: Create `raft_dic_gui/strain.py`**

Move from `processing.py`:
- `_compute_rotation_field_numba()` (lines 822-923)
- `calculate_strain_field()` (lines 925-1181)
- Required imports: `numpy`, `numba.jit`, `numba.prange`, `scipy.signal.fftconvolve`

```python
"""Strain field calculation via Vectorized Weighted Least Squares (VWLS)."""

import numpy as np
from scipy.signal import fftconvolve

try:
    from numba import jit, prange
except ImportError:
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

# Paste _compute_rotation_field_numba and calculate_strain_field here verbatim
```

**Step 2: Update imports in `processing.py`**

Replace the moved functions with re-exports:

```python
# At the location where the functions were removed:
from raft_dic_gui.strain import calculate_strain_field, _compute_rotation_field_numba
```

**Step 3: Update `server/routes/strain.py` line 9**

```python
# Change:
from raft_dic_gui.processing import calculate_strain_field
# To:
from raft_dic_gui.strain import calculate_strain_field
```

**Step 4: Run existing tests**

Run: `cd server && python -m pytest tests/test_strain.py -v`
Expected: All 4 tests PASS (no behavior change)

**Step 5: Commit**

```bash
git add raft_dic_gui/strain.py raft_dic_gui/processing.py server/routes/strain.py
git commit -m "refactor: extract strain module from processing.py"
```

---

### Task 1.2: Extract velocity and smoothing modules

**Files:**
- Create: `raft_dic_gui/velocity.py`
- Create: `raft_dic_gui/smoothing.py`
- Modify: `raft_dic_gui/processing.py:90-103,1675-1706`
- Modify: `server/routes/displacement.py:8-12` (import paths)

**Step 1: Create `raft_dic_gui/velocity.py`**

Move from `processing.py`:
- `calculate_displacement_magnitude()` (lines 1675-1686)
- `calculate_velocity_field()` (lines 1689-1706)

```python
"""Velocity and displacement magnitude calculations."""

import numpy as np


def calculate_displacement_magnitude(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Calculate displacement magnitude: M = sqrt(u^2 + v^2)"""
    return np.sqrt(u**2 + v**2)


def calculate_velocity_field(u_curr, v_curr, u_prev, v_prev, fps=1.0):
    """Calculate velocity magnitude from frame-to-frame displacement difference."""
    du = u_curr - u_prev
    dv = v_curr - v_prev
    return np.sqrt(du**2 + dv**2) * fps
```

**Step 2: Create `raft_dic_gui/smoothing.py`**

Move from `processing.py`:
- `smooth_displacement_field()` (lines 90-103)

```python
"""Displacement field smoothing utilities."""

import numpy as np
from scipy.ndimage import gaussian_filter


def smooth_displacement_field(displacement_field: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    # ... paste existing implementation
```

**Step 3: Update re-exports in `processing.py`**

```python
from raft_dic_gui.velocity import calculate_displacement_magnitude, calculate_velocity_field
from raft_dic_gui.smoothing import smooth_displacement_field
```

**Step 4: Update `server/routes/displacement.py` imports**

```python
from raft_dic_gui.velocity import calculate_displacement_magnitude, calculate_velocity_field
from raft_dic_gui.processing import load_and_convert_image
```

**Step 5: Run all tests**

Run: `cd server && python -m pytest tests/ -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add raft_dic_gui/velocity.py raft_dic_gui/smoothing.py raft_dic_gui/processing.py server/routes/displacement.py
git commit -m "refactor: extract velocity and smoothing modules"
```

---

### Task 1.3: Numerical correctness tests — analytic solutions

**Files:**
- Create: `server/tests/test_strain_numerical.py`

These tests use analytically known displacement fields where the exact strain is computable by hand. They verify the VWLS implementation produces correct results within tolerance.

**Step 1: Write test file**

```python
"""Numerical correctness tests for strain calculation using analytic solutions.

Each test creates a known displacement field, computes strain via calculate_strain_field,
and compares against the analytically expected result.
"""

import numpy as np
import pytest
from raft_dic_gui.strain import calculate_strain_field


def _make_displacement(H, W, u_func, v_func):
    """Build (H, W, 2) displacement field from functions u(x,y), v(x,y)."""
    yy, xx = np.mgrid[0:H, 0:W]
    u = u_func(xx.astype(float), yy.astype(float))
    v = v_func(xx.astype(float), yy.astype(float))
    return np.stack([u, v], axis=-1).astype(np.float64)


class TestUniformExtension:
    """Uniform uniaxial stretch: u = eps_x * x, v = 0.

    Engineering: exx = eps_x, eyy = 0, exy = 0
    Green-Lagrange: exx = eps_x + 0.5*eps_x^2, eyy = 0, exy = 0
    """

    @pytest.mark.parametrize("eps", [0.01, 0.05, 0.10])
    def test_engineering_exx(self, eps):
        H, W = 128, 128
        disp = _make_displacement(H, W, lambda x, y: eps * x, lambda x, y: 0.0 * x)
        result = calculate_strain_field(disp, method="engineering", vsg_size=31, step=1)
        # Interior region (avoid boundary effects)
        interior = result["exx"][20:-20, 20:-20]
        assert np.nanstd(interior) < 1e-6, "exx should be uniform"
        assert abs(np.nanmean(interior) - eps) < 1e-4, f"exx should be {eps}"

    @pytest.mark.parametrize("eps", [0.01, 0.05, 0.10])
    def test_green_lagrange_exx(self, eps):
        H, W = 128, 128
        disp = _make_displacement(H, W, lambda x, y: eps * x, lambda x, y: 0.0 * x)
        result = calculate_strain_field(disp, method="green_lagrange", vsg_size=31, step=1)
        expected = eps + 0.5 * eps**2
        interior = result["exx"][20:-20, 20:-20]
        assert abs(np.nanmean(interior) - expected) < 1e-4

    def test_zero_eyy_exy(self):
        eps = 0.05
        H, W = 128, 128
        disp = _make_displacement(H, W, lambda x, y: eps * x, lambda x, y: 0.0 * x)
        result = calculate_strain_field(disp, method="engineering", vsg_size=31, step=1)
        interior_eyy = result["eyy"][20:-20, 20:-20]
        interior_exy = result["exy"][20:-20, 20:-20]
        assert np.nanmax(np.abs(interior_eyy)) < 1e-6
        assert np.nanmax(np.abs(interior_exy)) < 1e-6


class TestPureShear:
    """Pure shear: u = gamma * y, v = 0.

    Engineering: exx = 0, eyy = 0, exy = gamma/2
    """

    def test_engineering_exy(self):
        gamma = 0.04
        H, W = 128, 128
        disp = _make_displacement(H, W, lambda x, y: gamma * y, lambda x, y: 0.0 * x)
        result = calculate_strain_field(disp, method="engineering", vsg_size=31, step=1)
        interior = result["exy"][20:-20, 20:-20]
        assert abs(np.nanmean(interior) - gamma / 2) < 1e-4


class TestRigidBodyRotation:
    """Small rigid rotation by angle theta (radians): u = -theta*y, v = theta*x.

    All strains should be ~0. Rotation angle should equal theta (in degrees).
    """

    def test_zero_strain(self):
        theta = 0.02  # ~1.15 degrees
        H, W = 128, 128
        disp = _make_displacement(
            H, W,
            lambda x, y: -theta * y,
            lambda x, y: theta * x,
        )
        result = calculate_strain_field(disp, method="engineering", vsg_size=31, step=1)
        interior_exx = result["exx"][20:-20, 20:-20]
        interior_eyy = result["eyy"][20:-20, 20:-20]
        assert np.nanmax(np.abs(interior_exx)) < 1e-4
        assert np.nanmax(np.abs(interior_eyy)) < 1e-4

    def test_rotation_angle(self):
        theta = 0.02
        H, W = 128, 128
        disp = _make_displacement(
            H, W,
            lambda x, y: -theta * y,
            lambda x, y: theta * x,
        )
        result = calculate_strain_field(disp, method="engineering", vsg_size=31, step=1)
        interior = result["rotation"][20:-20, 20:-20]
        expected_deg = np.degrees(theta)
        assert abs(np.nanmean(interior) - expected_deg) < 0.1, \
            f"Rotation should be ~{expected_deg:.2f} deg, got {np.nanmean(interior):.2f}"


class TestPrincipalStrains:
    """Biaxial state: u = eps1*x, v = eps2*y.

    e1 = max(eps1, eps2), e2 = min(eps1, eps2).
    von_mises = sqrt(e1^2 - e1*e2 + e2^2)
    """

    def test_principal_ordering(self):
        eps1, eps2 = 0.03, -0.01
        H, W = 128, 128
        disp = _make_displacement(H, W, lambda x, y: eps1 * x, lambda x, y: eps2 * y)
        result = calculate_strain_field(disp, method="engineering", vsg_size=31, step=1)
        int_e1 = result["e1"][20:-20, 20:-20]
        int_e2 = result["e2"][20:-20, 20:-20]
        assert np.nanmean(int_e1) > np.nanmean(int_e2), "e1 should be > e2"
        assert abs(np.nanmean(int_e1) - eps1) < 1e-4
        assert abs(np.nanmean(int_e2) - eps2) < 1e-4

    def test_von_mises(self):
        eps1, eps2 = 0.03, -0.01
        H, W = 128, 128
        disp = _make_displacement(H, W, lambda x, y: eps1 * x, lambda x, y: eps2 * y)
        result = calculate_strain_field(disp, method="engineering", vsg_size=31, step=1)
        expected_vm = np.sqrt(eps1**2 - eps1 * eps2 + eps2**2)
        int_vm = result["von_mises"][20:-20, 20:-20]
        assert abs(np.nanmean(int_vm) - expected_vm) < 1e-4


class TestStepDownsampling:
    """Verify that step > 1 produces correct but smaller output."""

    def test_step_reduces_size(self):
        H, W = 128, 128
        disp = _make_displacement(H, W, lambda x, y: 0.02 * x, lambda x, y: 0.0 * x)
        r1 = calculate_strain_field(disp, method="engineering", vsg_size=31, step=1)
        r4 = calculate_strain_field(disp, method="engineering", vsg_size=31, step=4)
        assert r4["exx"].shape[0] < r1["exx"].shape[0]
        assert r4["exx"].shape[1] < r1["exx"].shape[1]
        # Values should still be correct
        int4 = r4["exx"][5:-5, 5:-5]
        assert abs(np.nanmean(int4) - 0.02) < 1e-3
```

**Step 2: Run numerical tests**

Run: `cd server && python -m pytest tests/test_strain_numerical.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add server/tests/test_strain_numerical.py
git commit -m "test: add analytic-solution numerical correctness tests for strain"
```

---

## Phase 2: Algorithm Enhancements (Core Numerical)

### Task 2.1: Central difference velocity + Savitzky-Golay temporal smoothing

**Files:**
- Modify: `raft_dic_gui/velocity.py`
- Modify: `server/routes/displacement.py:21-37`
- Create: `server/tests/test_velocity.py`

**Step 1: Write failing test**

```python
# server/tests/test_velocity.py
"""Tests for velocity calculation methods."""
import numpy as np
import pytest
from raft_dic_gui.velocity import (
    calculate_velocity_field,
    calculate_velocity_central,
)


class TestCentralDifference:
    """Central difference: v[i] = (u[i+1] - u[i-1]) / (2*dt)"""

    def test_linear_ramp(self):
        """Linear u = t should give constant velocity = 1."""
        T, H, W = 10, 4, 4
        frames_u = [np.full((H, W), float(t)) for t in range(T)]
        frames_v = [np.zeros((H, W)) for _ in range(T)]
        vel = calculate_velocity_central(frames_u, frames_v, frame_idx=5, fps=1.0)
        assert abs(vel[0, 0] - 1.0) < 1e-10

    def test_endpoints_fallback_to_forward_backward(self):
        """Frame 0 uses forward diff, last frame uses backward diff."""
        T, H, W = 5, 4, 4
        frames_u = [np.full((H, W), float(t) * 2) for t in range(T)]
        frames_v = [np.zeros((H, W)) for _ in range(T)]
        vel_first = calculate_velocity_central(frames_u, frames_v, frame_idx=0, fps=1.0)
        vel_last = calculate_velocity_central(frames_u, frames_v, frame_idx=T - 1, fps=1.0)
        assert abs(vel_first[0, 0] - 2.0) < 1e-10
        assert abs(vel_last[0, 0] - 2.0) < 1e-10
```

**Step 2: Run to verify failure**

Run: `cd server && python -m pytest tests/test_velocity.py -v`
Expected: FAIL (ImportError — `calculate_velocity_central` doesn't exist yet)

**Step 3: Implement central difference**

Add to `raft_dic_gui/velocity.py`:

```python
def calculate_velocity_central(
    frames_u: list,
    frames_v: list,
    frame_idx: int,
    fps: float = 1.0,
) -> np.ndarray:
    """Central difference velocity: v[i] = |D[i+1] - D[i-1]| / (2*dt).

    Falls back to forward/backward difference at endpoints.
    """
    T = len(frames_u)
    dt = 1.0 / fps if fps > 0 else 1.0

    if T < 2:
        return np.zeros_like(frames_u[0])

    if frame_idx == 0:
        # Forward difference
        du = frames_u[1] - frames_u[0]
        dv = frames_v[1] - frames_v[0]
        return np.sqrt(du**2 + dv**2) / dt
    elif frame_idx >= T - 1:
        # Backward difference
        du = frames_u[T - 1] - frames_u[T - 2]
        dv = frames_v[T - 1] - frames_v[T - 2]
        return np.sqrt(du**2 + dv**2) / dt
    else:
        # Central difference
        du = frames_u[frame_idx + 1] - frames_u[frame_idx - 1]
        dv = frames_v[frame_idx + 1] - frames_v[frame_idx - 1]
        return np.sqrt(du**2 + dv**2) / (2.0 * dt)
```

**Step 4: Run tests**

Run: `cd server && python -m pytest tests/test_velocity.py -v`
Expected: PASS

**Step 5: Wire into displacement route**

In `server/routes/displacement.py`, update `_get_displacement_component()` to use central difference when all frames are available:

```python
elif component == "velocity":
    if frame_idx == 0 and len(session.displacement_results) < 2:
        return np.zeros(disp.shape[:2])
    frames_u = [d[:, :, 0] for d in session.displacement_results]
    frames_v = [d[:, :, 1] for d in session.displacement_results]
    return calculate_velocity_central(frames_u, frames_v, frame_idx)
```

**Step 6: Commit**

```bash
git add raft_dic_gui/velocity.py server/routes/displacement.py server/tests/test_velocity.py
git commit -m "feat: central difference velocity with endpoint fallback"
```

---

### Task 2.2: Temporal smoothing for strain

**Files:**
- Modify: `raft_dic_gui/strain.py`
- Modify: `server/routes/strain.py:23-81`
- Modify: `frontend/src/components/postprocessing/StrainControls.tsx`
- Modify: `frontend/src/api/strain.ts`

**Step 1: Add temporal smoothing function to `raft_dic_gui/strain.py`**

```python
def smooth_strain_temporal(strain_results: list, sigma_t: float = 1.0) -> list:
    """Apply Gaussian smoothing along the time axis for each strain component.

    Args:
        strain_results: List of strain dicts (one per frame), each with keys like 'exx', 'eyy', etc.
        sigma_t: Gaussian sigma in frame units (e.g. 1.0 = smooth over ~3 frames).

    Returns:
        New list of strain dicts with temporally smoothed values.
    """
    if not strain_results or sigma_t <= 0:
        return strain_results

    from scipy.ndimage import gaussian_filter1d

    keys = [k for k in strain_results[0].keys() if strain_results[0][k] is not None]
    T = len(strain_results)

    # Stack each component into (T, H, W), smooth along axis=0, then unstack
    smoothed = [{} for _ in range(T)]
    for key in keys:
        stack = np.array([s[key] for s in strain_results])  # (T, H, W)
        # Handle NaN: fill, smooth, restore
        valid = np.isfinite(stack)
        filled = np.nan_to_num(stack, nan=0.0)
        weight = valid.astype(np.float64)
        smoothed_data = gaussian_filter1d(filled, sigma=sigma_t, axis=0)
        smoothed_weight = gaussian_filter1d(weight, sigma=sigma_t, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(smoothed_weight > 0.01, smoothed_data / smoothed_weight, np.nan)
        for t in range(T):
            smoothed[t][key] = result[t].astype(np.float32)

    return smoothed
```

**Step 2: Add `temporal_sigma` parameter to strain calculate endpoint**

In `server/routes/strain.py`, after computing all frames (line ~63), optionally apply temporal smoothing:

```python
temporal_sigma = float(data.get("temporal_sigma", 0))
# ... after the for loop ...
if temporal_sigma > 0:
    from raft_dic_gui.strain import smooth_strain_temporal
    results = smooth_strain_temporal(results, sigma_t=temporal_sigma)
```

**Step 3: Add UI control in `StrainControls.tsx`**

Add a new state and input field for `temporalSigma`:

```tsx
const [temporalSigma, setTemporalSigma] = useState("0");
// In the calculateStrain call:
temporal_sigma: parseFloat(temporalSigma) || 0,
```

Add a FieldRow after "Weighting":
```tsx
<FieldRow label="Time Smooth">
  <SmallInput value={temporalSigma} onChange={setTemporalSigma} placeholder="0 = off" />
</FieldRow>
```

**Step 4: Update `frontend/src/api/strain.ts` params type to include `temporal_sigma`**

**Step 5: Run tests, rebuild frontend**

Run: `cd server && python -m pytest tests/ -v`
Run: `cd frontend && npm run build`

**Step 6: Commit**

```bash
git commit -am "feat: temporal Gaussian smoothing for strain fields"
```

---

### Task 2.3: Decouple VSG sigma from window size

**Files:**
- Modify: `raft_dic_gui/strain.py` (the `calculate_strain_field` function)
- Modify: `server/routes/strain.py:32-37`
- Modify: `frontend/src/components/postprocessing/StrainControls.tsx`
- Modify: `frontend/src/api/strain.ts`

**Step 1: Add `gaussian_sigma` parameter**

In `calculate_strain_field()`, change the signature:

```python
def calculate_strain_field(displacement_field, method='green_lagrange',
                          vsg_size=31, poly_order=1, weighting='Gaussian',
                          step=1, gaussian_sigma=None):
```

Change the kernel construction (line 973):

```python
if weighting == 'Gaussian':
    sigma = gaussian_sigma if gaussian_sigma is not None else vsg_size / 4.0
    dist_sq = X_grid**2 + Y_grid**2
    G = np.exp(-dist_sq / (2 * sigma**2))
```

**Step 2: Thread parameter through backend route**

In `server/routes/strain.py`:

```python
gaussian_sigma_raw = data.get("gaussian_sigma", None)
gaussian_sigma = float(gaussian_sigma_raw) if gaussian_sigma_raw is not None else None
# Pass to calculate_strain_field(..., gaussian_sigma=gaussian_sigma)
```

**Step 3: Add optional UI input**

In `StrainControls.tsx`, add an optional sigma field shown only when weighting is "gaussian":

```tsx
const [gaussianSigma, setGaussianSigma] = useState("");
// ...
{weighting === "gaussian" && (
  <FieldRow label="Sigma">
    <SmallInput value={gaussianSigma} onChange={setGaussianSigma} placeholder="auto" />
  </FieldRow>
)}
// In the API call:
gaussian_sigma: gaussianSigma ? parseFloat(gaussianSigma) : undefined,
```

**Step 4: Run numerical tests (ensure no regression)**

Run: `cd server && python -m pytest tests/test_strain_numerical.py -v`

**Step 5: Commit**

```bash
git commit -am "feat: decouple Gaussian sigma from VSG window size"
```

---

### Task 2.4: Inverse map convergence check

**Files:**
- Modify: `server/deformed_warp.py:74-185`

**Step 1: Add residual-based convergence**

Replace fixed iteration count with convergence check:

```python
def compute_inverse_map(U, V, roi_rect, image_shape, n_iter=10, tol=1e-3):
    # ... existing setup through Step 3 ...

    # --- Step 4: Fixed-point iteration with convergence check ---
    x_ref = out_cols.copy()
    y_ref = out_rows.copy()

    for iteration in range(n_iter):
        x_ref_old = x_ref.copy()
        y_ref_old = y_ref.copy()

        local_row = y_ref - y0
        local_col = x_ref - x0
        coords = np.array([local_row.ravel(), local_col.ravel()])
        U_sampled = map_coordinates(U_clean, coords, order=1, mode='constant', cval=0.0).reshape(out_h, out_w)
        V_sampled = map_coordinates(V_clean, coords, order=1, mode='constant', cval=0.0).reshape(out_h, out_w)

        x_ref = out_cols - U_sampled
        y_ref = out_rows - V_sampled

        # Check convergence
        residual = np.sqrt((x_ref - x_ref_old)**2 + (y_ref - y_ref_old)**2).max()
        if residual < tol:
            break

    # Mark non-converged pixels as invalid
    final_residual = np.sqrt((x_ref - x_ref_old)**2 + (y_ref - y_ref_old)**2)
    converged = final_residual < tol * 10  # generous final threshold
    # ... combine with existing validity mask ...
```

**Step 2: Run existing tests**

Run: `cd server && python -m pytest tests/ -v`

**Step 3: Commit**

```bash
git commit -am "feat: inverse map convergence check with residual threshold"
```

---

## Phase 3: New Computations

### Task 3.1: Strain rate calculation

**Files:**
- Modify: `raft_dic_gui/strain.py`
- Modify: `server/routes/strain.py`
- Modify: `frontend/src/types/api.ts:101-104`
- Modify: `frontend/src/components/postprocessing/VisualizationControls.tsx:18-27`

**Step 1: Add strain rate computation**

In `raft_dic_gui/strain.py`:

```python
def calculate_strain_rate(strain_results: list, fps: float = 1.0) -> list:
    """Compute strain rate dε/dt using central difference.

    Returns list of dicts with keys: 'dexx_dt', 'deyy_dt', 'dexy_dt'.
    """
    T = len(strain_results)
    if T < 2:
        return [None] * T

    dt = 1.0 / fps if fps > 0 else 1.0
    rate_results = []

    for i in range(T):
        rate = {}
        for comp in ['exx', 'eyy', 'exy']:
            if i == 0:
                # Forward difference
                curr = strain_results[0][comp]
                nxt = strain_results[1][comp]
                rate[f'd{comp}_dt'] = (nxt - curr) / dt
            elif i == T - 1:
                # Backward difference
                prev = strain_results[T - 2][comp]
                curr = strain_results[T - 1][comp]
                rate[f'd{comp}_dt'] = (curr - prev) / dt
            else:
                # Central difference
                prev = strain_results[i - 1][comp]
                nxt = strain_results[i + 1][comp]
                rate[f'd{comp}_dt'] = (nxt - prev) / (2.0 * dt)
        rate_results.append(rate)

    return rate_results
```

**Step 2: Compute and store strain rate in the calculate endpoint**

After strain computation completes in `server/routes/strain.py`, compute strain rate and merge into results:

```python
# After temporal smoothing (if any), compute strain rate
from raft_dic_gui.strain import calculate_strain_rate
rates = calculate_strain_rate(results, fps=float(data.get("fps", 1.0)))
for i, rate in enumerate(rates):
    if rate:
        results[i].update(rate)
```

Update `STRAIN_COMPONENTS` to include rate components:

```python
STRAIN_COMPONENTS = [
    "exx", "eyy", "exy", "e1", "e2", "max_shear", "von_mises", "rotation",
    "dexx_dt", "deyy_dt", "dexy_dt",
]
```

**Step 3: Add strain rate to frontend component list**

In `VisualizationControls.tsx`, add to `STRAIN_COMPONENTS`:

```tsx
{ value: "dexx_dt", label: "dεxx/dt" },
{ value: "deyy_dt", label: "dεyy/dt" },
{ value: "dexy_dt", label: "dεxy/dt" },
```

In `frontend/src/types/api.ts`, update `StrainComponent`:

```typescript
export type StrainComponent =
  | "exx" | "eyy" | "exy"
  | "e1" | "e2"
  | "max_shear" | "von_mises" | "rotation"
  | "dexx_dt" | "deyy_dt" | "dexy_dt";
```

**Step 4: Add `fps` input to StrainControls (for strain rate scaling)**

```tsx
<FieldRow label="FPS (for rate)">
  <SmallInput value={fps} onChange={setFps} placeholder="1.0" />
</FieldRow>
```

**Step 5: Rebuild + test**

Run: `cd frontend && npm run build`
Run: `cd server && python -m pytest tests/ -v`

**Step 6: Commit**

```bash
git commit -am "feat: strain rate calculation (dexx_dt, deyy_dt, dexy_dt)"
```

---

### Task 3.2: VWLS residual confidence field

**Files:**
- Modify: `raft_dic_gui/strain.py` (in `calculate_strain_field`)
- Modify: `server/routes/strain.py` (STRAIN_COMPONENTS)
- Modify: `frontend/src/types/api.ts`
- Modify: `frontend/src/components/postprocessing/VisualizationControls.tsx`

**Step 1: Compute fitting residual**

After solving the linear system in `calculate_strain_field`, compute residual:

```python
# After coefficients are obtained, compute residual = ||Ma - b||^2 / ||b||^2
# This gives a normalized goodness-of-fit per pixel
# Add to the returned dict:
#   'confidence': 1.0 - normalized_residual (clamped to [0, 1])
```

The residual for each pixel at solved locations:
```python
# Predicted b: M @ coeffs
pred_u = np.einsum('nij,nj->ni', M_valid, coeffs_u[valid_solve_mask])
resid_u = np.sum((pred_u - b_u_stack[valid_solve_mask])**2, axis=1)
norm_u = np.sum(b_u_stack[valid_solve_mask]**2, axis=1) + 1e-20

pred_v = np.einsum('nij,nj->ni', M_valid, coeffs_v[valid_solve_mask])
resid_v = np.sum((pred_v - b_v_stack[valid_solve_mask])**2, axis=1)
norm_v = np.sum(b_v_stack[valid_solve_mask]**2, axis=1) + 1e-20

# Combined normalized residual
nresid = np.full(N_pixels, np.nan)
nresid[valid_solve_mask] = (resid_u + resid_v) / (norm_u + norm_v)
confidence = np.full(N_pixels, np.nan)
confidence[valid_solve_mask] = np.clip(1.0 - nresid[valid_solve_mask], 0, 1)
confidence_map = confidence.reshape(H_down, W_down)
```

Add `'confidence': confidence_map` to the returned dict.

**Step 2: Register in STRAIN_COMPONENTS**

```python
STRAIN_COMPONENTS = [
    "exx", "eyy", "exy", "e1", "e2", "max_shear", "von_mises", "rotation",
    "dexx_dt", "deyy_dt", "dexy_dt", "confidence",
]
```

**Step 3: Add to frontend**

```tsx
{ value: "confidence", label: "Confidence" },
```

**Step 4: Commit**

```bash
git commit -am "feat: VWLS residual confidence field"
```

---

### Task 3.3: Data quality metric (NCC proxy)

**Files:**
- Create: `server/routes/quality.py`
- Modify: `server/app.py` (register blueprint)
- Modify: `frontend/src/types/api.ts`
- Modify: `frontend/src/api/displacement.ts`

**Step 1: Create quality endpoint**

```python
# server/routes/quality.py
"""Data quality metrics — NCC between warped reference and deformed image."""

import numpy as np
from flask import Blueprint, jsonify, request
from scipy.ndimage import map_coordinates

from server.session import session
from raft_dic_gui.processing import load_and_convert_image

quality_bp = Blueprint("quality", __name__)


@quality_bp.route("/ncc/<int:idx>", methods=["GET"])
def get_ncc(idx: int):
    """Compute NCC (Normalized Cross-Correlation) between warped reference and deformed frame.

    This is a proxy for DIC accuracy — high NCC means the displacement field
    correctly maps reference to deformed.
    """
    if not session.displacement_results or idx < 0 or idx >= len(session.displacement_results):
        return jsonify({"error": "Invalid frame"}), 400

    import os
    import cv2

    # Load reference and deformed images as grayscale
    ref_img = session.reference_image
    if ref_img is None:
        return jsonify({"error": "No reference image"}), 400

    if idx + 1 >= len(session.image_files):
        return jsonify({"error": "No deformed image"}), 400

    def_path = os.path.join(session.image_dir, session.image_files[idx + 1])
    def_img = load_and_convert_image(def_path)

    # Convert to grayscale
    if ref_img.ndim == 3:
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY).astype(np.float64)
    else:
        ref_gray = ref_img.astype(np.float64)
    if def_img.ndim == 3:
        def_gray = cv2.cvtColor(def_img, cv2.COLOR_RGB2GRAY).astype(np.float64)
    else:
        def_gray = def_img.astype(np.float64)

    disp = session.displacement_results[idx]
    u, v = disp[:, :, 0], disp[:, :, 1]
    h, w = u.shape

    # ROI region
    if session.roi_rect:
        x0, y0, x1, y1 = session.roi_rect
    else:
        x0, y0, x1, y1 = 0, 0, w, h

    # Warp reference using displacement
    yy, xx = np.mgrid[y0:y0 + h, x0:x0 + w]
    map_x = (xx + u).astype(np.float32)
    map_y = (yy + v).astype(np.float32)
    warped_ref = cv2.remap(ref_gray.astype(np.float32), map_x, map_y,
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Crop deformed to same region
    def_crop = def_gray[y0:y0 + h, x0:x0 + w]

    # Compute NCC on valid region
    valid = np.isfinite(u) & np.isfinite(v) & (warped_ref > 0)
    if not np.any(valid):
        return jsonify({"ncc": 0.0})

    a = warped_ref[valid] - warped_ref[valid].mean()
    b = def_crop[valid] - def_crop[valid].mean()
    ncc = float(np.sum(a * b) / (np.sqrt(np.sum(a**2) * np.sum(b**2)) + 1e-20))

    return jsonify({"ncc": round(ncc, 6), "frame": idx})
```

**Step 2: Register blueprint in `server/app.py`**

```python
from server.routes.quality import quality_bp
app.register_blueprint(quality_bp, url_prefix="/api/quality")
```

**Step 3: Commit**

```bash
git commit -am "feat: NCC quality metric endpoint"
```

---

### Task 3.4: Cumulative rotation for incremental mode

**Files:**
- Modify: `raft_dic_gui/strain.py`
- Modify: `server/routes/strain.py`

**Step 1: Add cumulative rotation post-processing**

After all frames are computed, if incremental mode is detected (multiple segments), accumulate rotation:

```python
def accumulate_rotation(strain_results: list) -> list:
    """Replace per-frame rotation with cumulative rotation.

    For each frame, rotation[i] = sum(rotation[0:i+1]).
    This is valid for small rotations (additive approximation).
    For large rotations, use R_total = R_new @ R_prev.
    """
    if not strain_results:
        return strain_results

    cumulative = np.zeros_like(strain_results[0].get('rotation', np.array([])))
    for i, s in enumerate(strain_results):
        if 'rotation' in s and s['rotation'] is not None:
            # Additive for small angles (< ~10 deg)
            cumulative = cumulative + np.nan_to_num(s['rotation'], nan=0.0)
            s['rotation_cumulative'] = cumulative.copy()
        else:
            s['rotation_cumulative'] = cumulative.copy()

    return strain_results
```

**Step 2: Call in strain calculate endpoint, add to components list**

```python
# After strain rate computation:
results = accumulate_rotation(results)
```

Add `"rotation_cumulative"` to STRAIN_COMPONENTS.

**Step 3: Add to frontend component list**

```tsx
{ value: "rotation_cumulative", label: "Rotation (cumul.)" },
```

**Step 4: Commit**

```bash
git commit -am "feat: cumulative rotation field"
```

---

## Phase 4: Frontend UX Enhancements

### Task 4.1: Engineering strain large-deformation warning

**Files:**
- Modify: `server/routes/strain.py` (add max strain to completion event)
- Modify: `frontend/src/components/postprocessing/StrainControls.tsx`

**Step 1: Include max strain in `strain:complete` event**

After strain computation, scan for max absolute strain:

```python
max_strain = 0.0
for r in results:
    for comp in ['exx', 'eyy', 'exy']:
        if comp in r:
            finite = r[comp][np.isfinite(r[comp])]
            if finite.size > 0:
                max_strain = max(max_strain, float(np.max(np.abs(finite))))

socketio.emit("strain:complete", {
    "num_frames": len(results),
    "components": STRAIN_COMPONENTS,
    "max_strain": max_strain,
    "method": method,
})
```

**Step 2: Show toast warning in frontend**

In the SocketIO listener (wherever `strain:complete` is handled), check:

```typescript
if (data.method === "engineering" && data.max_strain > 0.05) {
  toast.warn(
    `Max strain is ${(data.max_strain * 100).toFixed(1)}% — consider using Green-Lagrange for large deformations.`
  );
}
```

**Step 3: Commit**

```bash
git commit -am "feat: warn when engineering strain used with >5% deformation"
```

---

### Task 4.2: Von Mises assumption tooltip + export metadata

**Files:**
- Modify: `frontend/src/components/postprocessing/VisualizationControls.tsx`
- Modify: `raft_dic_gui/strain.py` (in `calculate_strain_field`)

**Step 1: Add tooltip to von Mises label in frontend**

```tsx
{ value: "von_mises", label: "Von Mises", title: "2D plane-stress assumption: σ_z = 0" },
```

Modify the SelectField to render option titles as HTML `title` attributes.

**Step 2: Add assumption to export metadata**

In `save_scientific_results`, add to metadata:

```python
metadata['von_mises_assumption'] = 'plane_stress_2D'
metadata['strain_method'] = method  # 'green_lagrange' or 'engineering'
```

**Step 3: Commit**

```bash
git commit -am "feat: von Mises plane-stress tooltip and export metadata"
```

---

### Task 4.3: Reference frame selector

**Files:**
- Modify: `frontend/src/stores/appStore.ts`
- Modify: `frontend/src/components/postprocessing/VisualizationControls.tsx`
- Modify: `server/routes/displacement.py`
- Modify: `server/routes/strain.py`

**Step 1: Add `referenceFrame` state to store**

```typescript
// In AppState:
referenceFrame: number;  // 0 = default (first frame)
// In initial state:
referenceFrame: 0,
// Action:
setReferenceFrame: (frame: number) => void;
```

**Step 2: Add UI control**

In VisualizationControls, add a "Reference Frame" input inside a new CollapsibleSection or under Physical Units:

```tsx
<FieldRow label="Ref. Frame">
  <SmallInput
    value={String(referenceFrame)}
    onChange={(v) => {
      const n = parseInt(v);
      if (!isNaN(n) && n >= 0) setReferenceFrame(n);
    }}
    placeholder="0"
  />
</FieldRow>
```

**Step 3: Backend support**

When `reference_frame > 0`, displacement relative to frame N is:
```
u_rel[i] = u[i] - u[N]
v_rel[i] = v[i] - v[N]
```

Add a query parameter `ref_frame` to displacement render/frame endpoints. If provided, subtract the reference frame's displacement before rendering.

**Step 4: Commit**

```bash
git commit -am "feat: reference frame selector for relative displacement"
```

---

### Task 4.4: Displacement smoothing sigma adjustable in UI

**Files:**
- Modify: `frontend/src/components/postprocessing/VisualizationControls.tsx`
- Modify: `frontend/src/stores/appStore.ts` (add `smoothSigma` to visSettings)
- Modify: `server/routes/displacement.py` (apply smoothing before render if sigma > 0)

**Step 1: Add `smoothSigma` to VisSettings**

```typescript
interface VisSettings {
  // ... existing ...
  smoothSigma: number; // 0 = no smoothing, default 0
}
// initial: smoothSigma: 0,
```

**Step 2: Add UI slider**

```tsx
<FieldRow label="Smooth sigma">
  <SliderField value={vis.smoothSigma} onChange={(v) => update({ smoothSigma: v })}
    min={0} max={10} step={0.5} />
</FieldRow>
```

**Step 3: Pass to render URL as query param**

`/api/displacement/render/{idx}?...&smooth_sigma=2.0`

**Step 4: In displacement render, apply on-the-fly smoothing**

```python
smooth_sigma = request.args.get("smooth_sigma", 0, type=float)
if smooth_sigma > 0:
    from raft_dic_gui.smoothing import smooth_displacement_field
    # Re-smooth the single component
    from scipy.ndimage import gaussian_filter
    disp_data = gaussian_filter(np.nan_to_num(disp_data), sigma=smooth_sigma)
```

**Step 5: Commit**

```bash
git commit -am "feat: adjustable displacement smoothing sigma in UI"
```

---

### Task 4.5: Principal strain direction overlay

**Files:**
- Create: `server/routes/principal_dirs.py`
- Modify: `server/app.py` (register blueprint)
- Modify: `frontend/src/components/postprocessing/PostProcessingView.tsx`
- Modify: `frontend/src/stores/appStore.ts`

**Step 1: Create principal direction render endpoint**

```python
# server/routes/principal_dirs.py
"""Principal strain direction overlay as transparent PNG with cross marks."""

@principal_bp.route("/render/<int:idx>", methods=["GET"])
def render_principal_directions(idx):
    """Render principal strain direction crosses at grid points.

    Each cross has two arms:
    - Major principal direction (e1): drawn longer, red
    - Minor principal direction (e2): drawn shorter, blue

    Direction angle = 0.5 * atan2(2*exy, exx - eyy)
    """
    # Extract exx, eyy, exy from strain_results[idx]
    # Compute angle = 0.5 * arctan2(2*exy, exx - eyy)
    # Draw crosses at grid spacing using matplotlib
    # Return transparent PNG
```

**Step 2: Add toggle in frontend store**

```typescript
arrowSettings: {
  // ... existing ...
  showPrincipalDirs: boolean; // default false
}
```

**Step 3: Overlay in PostProcessingView**

When `showPrincipalDirs` is true and strain is computed, render an additional `<img>` layer:

```tsx
{arrowSettings.showPrincipalDirs && hasStrain && (
  <img src={principalDirUrl} className="absolute inset-0 max-w-none" ... />
)}
```

**Step 4: Add toggle control**

In VelocityArrowControls or a new section:

```tsx
<FieldRow label="Principal Dirs">
  <Toggle checked={arrowSettings.showPrincipalDirs}
    onChange={(v) => updateArrowSettings({ showPrincipalDirs: v })} />
</FieldRow>
```

**Step 5: Commit**

```bash
git commit -am "feat: principal strain direction overlay"
```

---

### Task 4.6: Extract frontend unit conversion utility

**Files:**
- Create: `frontend/src/utils/unitConversion.ts`
- Modify: `frontend/src/components/postprocessing/PostProcessingView.tsx`

**Step 1: Extract getUnitInfo**

Create a standalone utility:

```typescript
// frontend/src/utils/unitConversion.ts
import type { DisplayComponent } from "@/types/api";

interface UnitInfo {
  scale: number;
  unit: string;
}

const STRAIN_COMPONENTS = new Set([
  "exx", "eyy", "exy", "e1", "e2", "max_shear", "von_mises", "confidence",
]);
const STRAIN_RATE_COMPONENTS = new Set(["dexx_dt", "deyy_dt", "dexy_dt"]);

export function getUnitInfo(
  component: DisplayComponent,
  physicalEnabled: boolean,
  physicalRatio: number,
  physicalUnit: string,
  fps: number,
): UnitInfo {
  if (component === "rotation" || component === "rotation_cumulative") {
    return { scale: 1, unit: "[deg]" };
  }
  if (STRAIN_COMPONENTS.has(component)) {
    return { scale: 1, unit: "[-]" };
  }
  if (STRAIN_RATE_COMPONENTS.has(component)) {
    return { scale: fps, unit: "[1/s]" };
  }
  if (component === "velocity") {
    if (physicalEnabled) {
      return { scale: fps * physicalRatio, unit: `[${physicalUnit}/s]` };
    }
    return { scale: fps, unit: "[px/s]" };
  }
  // displacement: u, v, magnitude
  if (physicalEnabled) {
    return { scale: physicalRatio, unit: `[${physicalUnit}]` };
  }
  return { scale: 1, unit: "[px]" };
}
```

**Step 2: Replace inline logic in PostProcessingView**

Import and use the extracted function.

**Step 3: Commit**

```bash
git commit -am "refactor: extract unit conversion utility"
```

---

### Task 4.7: Area probe average strain

**Files:**
- Modify: `raft_dic_gui/probe_manager.py` (already supports area probes with avg metric)

This already works via existing `extract_area_series(data_list, area_id, metric='avg')`. The area probe can already extract average strain values when the display component is a strain type.

**Verify this works:** Manually test placing an area probe and selecting a strain component. If the time series chart shows values, mark this task as already complete.

If it doesn't work, the fix is ensuring the strain data list is passed to `extract_area_series` in the probes route, just like displacement data is.

**Step 1: Verify in `server/routes/probes.py`**

Check that when `component` is a strain component (like "exx"), the correct data source (`session.strain_results`) is used.

**Step 2: Fix if needed, commit**

```bash
git commit -am "fix: area probe strain extraction"
```

---

## Phase 5: Dynamic Cache Sizing

### Task 5.1: Memory-aware cache sizing

**Files:**
- Modify: `server/render_cache.py`
- Modify: `server/deformed_warp.py:41-67`

**Step 1: Add auto-sizing helper**

```python
# In server/render_cache.py:
import psutil

def auto_cache_size(target_mb: int = 500, avg_item_kb: int = 200) -> int:
    """Compute cache size based on available system memory."""
    try:
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        # Use at most target_mb or 10% of available, whichever is smaller
        budget_mb = min(target_mb, available_mb * 0.1)
        return max(64, int(budget_mb * 1024 / avg_item_kb))
    except Exception:
        return 512  # fallback
```

**Step 2: Use in RenderCache and InverseMapCache initialization**

Replace hardcoded `512` and `5` with `auto_cache_size()` calls where caches are created.

**Step 3: Commit**

```bash
git commit -am "feat: memory-aware cache sizing"
```

---

## Summary: Commit Sequence

| # | Commit | Phase |
|---|--------|-------|
| 1 | `refactor: extract strain module from processing.py` | 1 |
| 2 | `refactor: extract velocity and smoothing modules` | 1 |
| 3 | `test: add analytic-solution numerical correctness tests` | 1 |
| 4 | `feat: central difference velocity with endpoint fallback` | 2 |
| 5 | `feat: temporal Gaussian smoothing for strain fields` | 2 |
| 6 | `feat: decouple Gaussian sigma from VSG window size` | 2 |
| 7 | `feat: inverse map convergence check` | 2 |
| 8 | `feat: strain rate calculation` | 3 |
| 9 | `feat: VWLS residual confidence field` | 3 |
| 10 | `feat: NCC quality metric endpoint` | 3 |
| 11 | `feat: cumulative rotation field` | 3 |
| 12 | `feat: warn when engineering strain >5%` | 4 |
| 13 | `feat: von Mises tooltip and export metadata` | 4 |
| 14 | `feat: reference frame selector` | 4 |
| 15 | `feat: adjustable displacement smoothing sigma` | 4 |
| 16 | `feat: principal strain direction overlay` | 4 |
| 17 | `refactor: extract unit conversion utility` | 4 |
| 18 | `fix: area probe strain extraction` | 4 |
| 19 | `feat: memory-aware cache sizing` | 5 |

**Total: 19 commits across 5 phases.**

Phase 1 is prerequisite for all others. Phases 2-5 can be executed in parallel if using subagents per phase, or sequentially within each phase.
