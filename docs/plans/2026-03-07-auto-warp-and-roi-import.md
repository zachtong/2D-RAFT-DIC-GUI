# Auto-Warp Mask, Full-Image Accumulation & ROI Import Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement stable auto-warp mask for incremental mode (full-image coordinate accumulation + NaN-aware interpolation + dynamic roi_rect), fix and enhance ROI import with noise filtering, and add mask-folder Frame 1 logic.

**Architecture:** Three foundational backend changes (NaN-aware interpolation, full-image accumulation, auto-warp), then ROI import fix/enhancement, then mask-folder Frame 1 integration, then UI polish. Each layer builds on the previous.

**Tech Stack:** Python (NumPy, OpenCV, scipy), Flask, React + TypeScript + Zustand, pytest

---

## Phase 1: NaN-Aware Bilinear Interpolation

Replace `cv2.remap` in `warp_displacement_field` with a custom NaN-aware bilinear sampler that excludes NaN neighbors from interpolation, preventing boundary contamination.

### Task 1: Write NaN-aware bilinear interpolation function + tests

**Files:**
- Create: `raft_dic_gui/nan_interp.py`
- Test: `server/tests/test_nan_interp.py`

**Step 1: Write the tests**

Create `server/tests/test_nan_interp.py`:

```python
"""Tests for NaN-aware bilinear interpolation."""

import numpy as np
import pytest
from raft_dic_gui.nan_interp import bilinear_sample_nan_aware


class TestBilinearNanAware:
    """Tests for bilinear_sample_nan_aware(field, x, y)."""

    def test_integer_coords_exact_lookup(self):
        """Integer coordinates should return exact pixel values."""
        field = np.array([[1.0, 2.0],
                          [3.0, 4.0]])
        x = np.array([0.0, 1.0, 0.0, 1.0])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        result = bilinear_sample_nan_aware(field, x, y)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0])

    def test_fractional_coords_standard_bilinear(self):
        """Fractional coords on all-valid field should match standard bilinear."""
        field = np.array([[0.0, 10.0],
                          [0.0, 10.0]])
        x = np.array([0.5])
        y = np.array([0.5])
        result = bilinear_sample_nan_aware(field, x, y)
        np.testing.assert_allclose(result, [5.0])

    def test_nan_neighbor_excluded_and_renormalized(self):
        """When one neighbor is NaN, remaining neighbors are re-weighted."""
        # 2x2 field: top-left is NaN
        field = np.array([[np.nan, 10.0],
                          [0.0,    10.0]])
        # Sample at (0.5, 0.5): 4 neighbors with weights 0.25 each
        # NaN neighbor excluded, remaining 3 share the weight
        x = np.array([0.5])
        y = np.array([0.5])
        result = bilinear_sample_nan_aware(field, x, y)
        # Expected: (0.25*10 + 0.25*0 + 0.25*10) / 0.75 = 20/3
        np.testing.assert_allclose(result, [20.0 / 3.0], rtol=1e-6)

    def test_all_nan_neighbors_returns_nan(self):
        """If all 4 neighbors are NaN, result is NaN."""
        field = np.full((3, 3), np.nan)
        x = np.array([1.0])
        y = np.array([1.0])
        result = bilinear_sample_nan_aware(field, x, y)
        assert np.isnan(result[0])

    def test_fully_out_of_bounds_returns_nan(self):
        """Coordinates completely outside the image return NaN."""
        field = np.ones((4, 4))
        x = np.array([-2.0, 5.0])
        y = np.array([-2.0, 5.0])
        result = bilinear_sample_nan_aware(field, x, y)
        assert np.all(np.isnan(result))

    def test_edge_partial_out_of_bounds(self):
        """Coords near edge where some neighbors are OOB use valid neighbors only."""
        field = np.array([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0]])
        # x=2.5 means x1=3 is OOB (W=3). Only left neighbors valid.
        x = np.array([2.5])
        y = np.array([0.5])
        result = bilinear_sample_nan_aware(field, x, y)
        # Valid neighbors: (2,0)=3.0 weight (1-0.5)*(1-0.5)=0.25
        #                  (2,1)=6.0 weight (1-0.5)*0.5=0.25
        # OOB neighbors: (3,0), (3,1) weight=0
        np.testing.assert_allclose(result, [4.5], rtol=1e-6)

    def test_large_field_vectorized(self):
        """Works on large arrays efficiently (vectorized, no per-pixel loop)."""
        H, W = 512, 512
        field = np.random.rand(H, W).astype(np.float32)
        x = np.random.uniform(0, W - 1, size=10000).astype(np.float32)
        y = np.random.uniform(0, H - 1, size=10000).astype(np.float32)
        result = bilinear_sample_nan_aware(field, x, y)
        assert result.shape == (10000,)
        assert not np.any(np.isnan(result))

    def test_nan_at_boundary_no_contamination(self):
        """Key scenario: valid region surrounded by NaN. Boundary pixels correct."""
        field = np.full((10, 10), np.nan)
        field[3:7, 3:7] = 5.0  # Valid 4x4 block in center

        # Sample at (3.3, 5.0) — just inside valid region
        # Neighbors: (3,5)=5.0 w=0.7, (4,5)=5.0 w=0.3, (3,5)=same, (4,5)=same
        x = np.array([3.3])
        y = np.array([5.0])
        result = bilinear_sample_nan_aware(field, x, y)
        np.testing.assert_allclose(result, [5.0], rtol=1e-6)

        # Sample at (2.7, 5.0) — straddles boundary
        # Neighbors: (2,5)=NaN w->0, (3,5)=5.0, (2,5)=NaN w->0, (3,5)=5.0
        x2 = np.array([2.7])
        result2 = bilinear_sample_nan_aware(field, x2, y)
        np.testing.assert_allclose(result2, [5.0], rtol=1e-6)
```

**Step 2: Run tests to verify they fail**

Run: `cd server && python -m pytest tests/test_nan_interp.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'raft_dic_gui.nan_interp'`

**Step 3: Implement `nan_interp.py`**

Create `raft_dic_gui/nan_interp.py`:

```python
"""NaN-aware bilinear interpolation for displacement field sampling."""

import numpy as np


def bilinear_sample_nan_aware(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Sample a 2D field at sub-pixel (x, y) positions, ignoring NaN neighbors.

    Standard bilinear interpolation takes a weighted average of 4 neighbors.
    If any neighbor is NaN (or out-of-bounds), its weight is set to zero and
    the remaining weights are renormalized.  If ALL 4 neighbors are
    invalid, the result is NaN.

    Args:
        field: (H, W) float array (may contain NaN).
        x: (N,) float array of x (column) coordinates.
        y: (N,) float array of y (row) coordinates.

    Returns:
        (N,) float array of sampled values.
    """
    H, W = field.shape

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    # Fractional parts
    fx = (x - x0).astype(np.float64)
    fy = (y - y0).astype(np.float64)

    # Bilinear weights for 4 corners
    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx * fy

    # Clamp indices for safe array access
    x0c = np.clip(x0, 0, W - 1)
    x1c = np.clip(x1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1)
    y1c = np.clip(y1, 0, H - 1)

    # Fetch values
    v00 = field[y0c, x0c]
    v10 = field[y0c, x1c]
    v01 = field[y1c, x0c]
    v11 = field[y1c, x1c]

    # Mark out-of-bounds corners as invalid (set weight to 0)
    oob00 = (x0 < 0) | (y0 < 0)
    oob10 = (x1 >= W) | (y0 < 0)
    oob01 = (x0 < 0) | (y1 >= H)
    oob11 = (x1 >= W) | (y1 >= H)

    # Mark NaN corners as invalid
    nan00 = np.isnan(v00) | oob00
    nan10 = np.isnan(v10) | oob10
    nan01 = np.isnan(v01) | oob01
    nan11 = np.isnan(v11) | oob11

    # Zero out invalid weights and values
    w00 = np.where(nan00, 0.0, w00)
    w10 = np.where(nan10, 0.0, w10)
    w01 = np.where(nan01, 0.0, w01)
    w11 = np.where(nan11, 0.0, w11)

    v00 = np.where(nan00, 0.0, v00)
    v10 = np.where(nan10, 0.0, v10)
    v01 = np.where(nan01, 0.0, v01)
    v11 = np.where(nan11, 0.0, v11)

    w_sum = w00 + w10 + w01 + w11

    result = np.where(
        w_sum > 0.0,
        (w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11) / w_sum,
        np.nan,
    )

    return result
```

**Step 4: Run tests to verify they pass**

Run: `cd server && python -m pytest tests/test_nan_interp.py -v`
Expected: all 8 tests PASS

**Step 5: Commit**

```bash
git add raft_dic_gui/nan_interp.py server/tests/test_nan_interp.py
git commit -m "feat: NaN-aware bilinear interpolation for displacement sampling"
```

---

### Task 2: Replace cv2.remap in warp_displacement_field with NaN-aware sampler

**Files:**
- Modify: `raft_dic_gui/incremental.py:113-185` (`warp_displacement_field`)
- Test: `server/tests/test_incremental.py` (existing tests must still pass)
- Create: `server/tests/test_nan_boundary.py` (new boundary-contamination regression test)

**Step 1: Write regression test for boundary contamination**

Create `server/tests/test_nan_boundary.py`:

```python
"""Regression test: boundary contamination in warp_displacement_field.

When the valid region of delta_u is surrounded by NaN, sampling at the
boundary must NOT mix in NaN->0 values.  This test creates a scenario
where the old cv2.remap approach would produce incorrect results.
"""

import numpy as np
from raft_dic_gui.incremental import warp_displacement_field


class TestBoundaryContamination:

    def test_no_contamination_at_valid_boundary(self):
        """Boundary pixels of delta_u should not be pulled toward zero."""
        H, W = 100, 100

        # delta_u: valid block [30:70, 30:70] with uniform value 3.0
        delta_u = np.full((H, W, 2), np.nan)
        delta_u[30:70, 30:70, :] = 3.0

        # accumulated_u: uniform 9.7px rightward shift
        # This causes non-integer sampling coords at the boundary of delta_u
        accumulated_u = np.full((H, W, 2), np.nan)
        accumulated_u[20:60, 20:60, 0] = 9.7   # u (x-shift)
        accumulated_u[20:60, 20:60, 1] = 0.0    # v (no y-shift)

        result = warp_displacement_field(delta_u, accumulated_u)

        # For pixel (x=20, y=40) in accumulated_u:
        #   sample coords = (20 + 9.7, 40 + 0) = (29.7, 40)
        #   delta_u valid from x=30, so x0=29 is NaN, x1=30 is valid
        #   NaN-aware: should return 3.0 (only valid neighbor used)
        #   Old cv2.remap: would return ~0.3*0 + 0.7*3 = 2.1 (WRONG)
        valid = ~np.isnan(result[..., 0])
        valid_values = result[valid, 0]

        # All valid sampled values should be close to 3.0
        assert len(valid_values) > 0, "Should have some valid samples"
        np.testing.assert_allclose(
            valid_values, 3.0, atol=0.05,
            err_msg="Boundary contamination detected: values deviate from 3.0"
        )
```

**Step 2: Run test — expect FAIL with old implementation**

Run: `cd server && python -m pytest tests/test_nan_boundary.py -v`
Expected: FAIL — values near boundary will be ~2.1 instead of 3.0

**Step 3: Rewrite `warp_displacement_field` to use NaN-aware sampler**

Modify `raft_dic_gui/incremental.py`:

```python
def warp_displacement_field(delta_u: np.ndarray,
                            accumulated_u: np.ndarray) -> np.ndarray:
    """
    Sample incremental displacement at deformed coordinates.

    When the reference is updated, the new incremental displacement delta_u
    is measured in the deformed coordinate system. To accumulate it with
    previous displacement, we need to sample delta_u at the deformed positions.

    Uses NaN-aware bilinear interpolation so that NaN regions do not
    contaminate valid boundary pixels.

    Args:
        delta_u: (H, W, 2) incremental displacement (in deformed coordinates)
        accumulated_u: (H, W, 2) accumulated displacement so far (in original coords)

    Returns:
        delta_u_sampled: (H, W, 2) delta_u sampled at deformed positions
    """
    from raft_dic_gui.nan_interp import bilinear_sample_nan_aware

    H, W = delta_u.shape[:2]

    # Create coordinate grids
    x, y = np.meshgrid(np.arange(W, dtype=np.float64),
                        np.arange(H, dtype=np.float64))

    # Deformed coordinates
    u_prev_x = accumulated_u[..., 0].astype(np.float64)
    u_prev_y = accumulated_u[..., 1].astype(np.float64)

    x_def = x + u_prev_x
    y_def = y + u_prev_y

    # Mask: accumulated_u itself is NaN → result is NaN
    u_prev_nan = np.isnan(u_prev_x)

    # Flatten for vectorized sampling
    x_flat = x_def.ravel()
    y_flat = y_def.ravel()

    # Replace NaN coords with -999 (will be OOB → NaN result)
    x_flat = np.where(np.isnan(x_flat), -999.0, x_flat)
    y_flat = np.where(np.isnan(y_flat), -999.0, y_flat)

    # Sample each channel with NaN-aware interpolation
    delta_u_sampled = np.empty_like(delta_u)
    for ch in range(2):
        sampled = bilinear_sample_nan_aware(delta_u[..., ch], x_flat, y_flat)
        delta_u_sampled[..., ch] = sampled.reshape(H, W)

    # Also mark NaN where accumulated_u was NaN
    delta_u_sampled[u_prev_nan, 0] = np.nan
    delta_u_sampled[u_prev_nan, 1] = np.nan

    return delta_u_sampled
```

**Step 4: Run all tests**

Run: `cd server && python -m pytest tests/test_nan_boundary.py tests/test_incremental.py -v`
Expected: ALL PASS (including existing incremental tests + new boundary test)

**Step 5: Commit**

```bash
git add raft_dic_gui/incremental.py server/tests/test_nan_boundary.py
git commit -m "fix: replace cv2.remap with NaN-aware bilinear in warp_displacement_field"
```

---

## Phase 2: Full-Image Coordinate Accumulation + Dynamic roi_rect

Change `kf_accumulated` from crop-sized to full-image-sized arrays. Introduce an envelope rect that grows as warped masks expand. All per-frame results are stored in envelope coordinates.

### Task 3: Refactor controller to use full-image accumulation

**Files:**
- Modify: `raft_dic_gui/controller.py:41-290` (`run` method)
- Test: `server/tests/test_incremental.py` (existing tests must still pass)
- Create: `server/tests/test_full_image_accum.py`

**Step 1: Write test for full-image accumulation with shifted mask**

Create `server/tests/test_full_image_accum.py`:

```python
"""Tests for full-image coordinate accumulation in DICProcessor."""

import numpy as np
from raft_dic_gui.incremental import accumulate_displacement


class TestFullImageAccumulation:

    def test_accumulate_in_full_image_coords(self):
        """Accumulation in full-image coords: arrays with different valid regions."""
        H, W = 100, 100

        # u_prev: valid in [20:60, 20:60], uniform 10px rightward
        u_prev = np.full((H, W, 2), np.nan)
        u_prev[20:60, 20:60, 0] = 10.0
        u_prev[20:60, 20:60, 1] = 0.0

        # delta_u: valid in [20:60, 30:70] (shifted right by 10px)
        # Represents DIC output using warped mask
        delta_u = np.full((H, W, 2), np.nan)
        delta_u[20:60, 30:70, 0] = 5.0
        delta_u[20:60, 30:70, 1] = 0.0

        result = accumulate_displacement(u_prev, delta_u, debug=False)

        # For pixel (x=20, y=30): deformed coord = (30, 30)
        # delta_u[30, 30] = 5.0 → total = 10 + 5 = 15
        assert not np.isnan(result[30, 20, 0]), "Should be valid"
        np.testing.assert_allclose(result[30, 20, 0], 15.0, atol=0.1)

    def test_envelope_rect_union(self):
        """Helper: compute union of two bounding boxes."""
        from raft_dic_gui.controller import _envelope_union
        r1 = (10, 20, 50, 60)  # (xmin, ymin, xmax, ymax)
        r2 = (30, 10, 70, 55)
        union = _envelope_union(r1, r2)
        assert union == (10, 10, 70, 60)

    def test_embed_crop_into_envelope(self):
        """Embed a cropped array into a larger envelope array."""
        from raft_dic_gui.controller import _embed_into_envelope
        crop = np.ones((10, 20, 2))
        crop_rect = (5, 5, 25, 15)  # (xmin, ymin, xmax, ymax)
        envelope_rect = (0, 0, 30, 20)
        result = _embed_into_envelope(crop, crop_rect, envelope_rect)
        assert result.shape == (20, 30, 2)
        assert np.isnan(result[0, 0, 0])  # Outside crop
        np.testing.assert_equal(result[5:15, 5:25, :], 1.0)  # Inside crop
```

**Step 2: Run tests — expect FAIL**

Run: `cd server && python -m pytest tests/test_full_image_accum.py -v`
Expected: FAIL — `_envelope_union` and `_embed_into_envelope` don't exist yet

**Step 3: Add helper functions to controller.py**

Add to `raft_dic_gui/controller.py` (module-level, before class):

```python
def _envelope_union(r1, r2):
    """Compute the union bounding box of two (xmin, ymin, xmax, ymax) rects."""
    return (
        min(r1[0], r2[0]),
        min(r1[1], r2[1]),
        max(r1[2], r2[2]),
        max(r1[3], r2[3]),
    )


def _embed_into_envelope(crop, crop_rect, envelope_rect):
    """Embed a crop-sized array into an envelope-sized array (NaN-filled).

    Args:
        crop: (H_crop, W_crop, C) array
        crop_rect: (xmin, ymin, xmax, ymax) of crop in full-image coords
        envelope_rect: (xmin, ymin, xmax, ymax) of envelope in full-image coords

    Returns:
        (H_env, W_env, C) array with crop placed at correct position.
    """
    ex, ey, ex1, ey1 = envelope_rect
    H_env = ey1 - ey
    W_env = ex1 - ex
    C = crop.shape[2] if crop.ndim == 3 else 1

    out = np.full((H_env, W_env, C) if crop.ndim == 3 else (H_env, W_env),
                  np.nan, dtype=crop.dtype)

    cx, cy, cx1, cy1 = crop_rect
    # Offset of crop within envelope
    ox = cx - ex
    oy = cy - ey
    h_crop = cy1 - cy
    w_crop = cx1 - cx

    if crop.ndim == 3:
        out[oy:oy + h_crop, ox:ox + w_crop, :] = crop
    else:
        out[oy:oy + h_crop, ox:ox + w_crop] = crop

    return out
```

**Step 4: Refactor `DICProcessor.run()` for full-image accumulation**

Key changes in `raft_dic_gui/controller.py` `run()` method:

```python
# CHANGE 1: kf_accumulated stores full-image-sized arrays
H_full, W_full = kf_images[1].shape[:2]

kf_accumulated = {
    1: np.full((H_full, W_full, 2), np.nan, dtype=np.float64),
}
# Initialize Frame 1 ROI area to zero displacement
kf_accumulated[1][ymin:ymax, xmin:xmax][roi_mask[ymin:ymax, xmin:xmax]] = 0.0

# Track envelope rect (grows as warped masks expand)
envelope_rect = list(roi_rect)  # mutable copy: [xmin, ymin, xmax, ymax]

# CHANGE 2: In main loop, don't crop delta_disp
# Instead of:  delta_disp = disp_full[ymin:ymax, xmin:xmax, :]
# Use:         delta_disp = disp_full  (full-image)

# CHANGE 3: Accumulate in full-image coords
if is_first_segment:
    total_disp_full = delta_disp_full.copy()
else:
    total_disp_full = accumulate_displacement(
        kf_accumulated[ref_num], delta_disp_full, debug=True
    )

# CHANGE 4: Update envelope if key frame mask extends beyond
if frame_num in key_frame_set:
    kf_accumulated[frame_num] = total_disp_full.copy()
    # Compute warped mask bbox and expand envelope
    valid_mask = ~np.isnan(total_disp_full[..., 0])
    if np.any(valid_mask):
        ys, xs = np.where(valid_mask)
        kf_rect = (int(xs.min()), int(ys.min()),
                   int(xs.max()) + 1, int(ys.max()) + 1)
        envelope_rect = list(_envelope_union(tuple(envelope_rect), kf_rect))

# CHANGE 5: Crop to envelope for storage
ex, ey, ex1, ey1 = envelope_rect
displacement_field = total_disp_full[ey:ey1, ex:ex1, :]
```

**Implementation notes:**
- `save_displacement_results` stores per-frame `.npy` — these now use `envelope_rect` instead of fixed `roi_rect`
- `save_displacement_sequence` at the end uses `envelope_rect` for grid computation
- Session must store `envelope_rect` for downstream visualization (render cache, probes, strain)
- If envelope expands mid-processing, previously saved `.npy` files have the old shape — either re-embed them or defer consolidation to the end. **Recommended:** save full-image crops and consolidate at end.

**Step 5: Run all tests**

Run: `cd server && python -m pytest tests/test_full_image_accum.py tests/test_incremental.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add raft_dic_gui/controller.py server/tests/test_full_image_accum.py
git commit -m "refactor: full-image coordinate accumulation with dynamic envelope rect"
```

---

### Task 4: Update session and downstream to use envelope_rect

**Files:**
- Modify: `server/session.py` — add `envelope_rect` field
- Modify: `server/routes/processing.py` — store `envelope_rect` after processing completes
- Modify: `server/routes/displacement.py` — use `envelope_rect` for rendering
- Modify: `server/routes/export.py` — use `envelope_rect` for export grid

**Step 1: Add envelope_rect to session**

In `server/session.py`, add to `AppSession`:

```python
envelope_rect: Optional[Tuple[int, int, int, int]] = None  # dynamic bounding box
```

And in `reset()`:

```python
self.envelope_rect = None
```

**Step 2: Store envelope_rect after processing**

In `server/routes/processing.py`, in the processing completion callback (where `session.displacement_results` is set), add:

```python
session.envelope_rect = processor.envelope_rect  # set by run()
```

And in `DICProcessor.run()`, store envelope as instance attribute before returning:

```python
self.envelope_rect = tuple(envelope_rect)
```

**Step 3: Update render routes to use envelope_rect**

In displacement/strain rendering routes, replace references to `session.roi_rect` with `session.envelope_rect or session.roi_rect` for the displacement crop region.

**Step 4: Run full test suite**

Run: `cd server && python -m pytest tests/ -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add server/session.py server/routes/processing.py server/routes/displacement.py server/routes/export.py
git commit -m "feat: propagate envelope_rect through session and render pipeline"
```

---

## Phase 3: Auto-Warp Mask for Key Frames

Use `warp_mask_with_holes` from `incremental.py` to auto-generate masks for key frames based on Frame 1's ROI mask + accumulated displacement.

### Task 5: Implement auto-warp in _resolve_mask

**Files:**
- Modify: `raft_dic_gui/controller.py:429-452` (`_resolve_mask`)
- Modify: `raft_dic_gui/config.py` — add `mask_warp_mode` field
- Test: `server/tests/test_auto_warp_integration.py`

**Step 1: Write integration test**

Create `server/tests/test_auto_warp_integration.py`:

```python
"""Integration test: auto-warp produces correct mask for translated specimen."""

import numpy as np
from raft_dic_gui.incremental import warp_mask_with_holes


class TestAutoWarpMask:

    def test_uniform_translation_warp(self):
        """Mask warped by uniform translation should shift by that amount."""
        H, W = 100, 100
        mask = np.zeros((H, W), dtype=bool)
        mask[20:60, 20:60] = True  # 40x40 block

        # Accumulated displacement: 15px right, 10px down
        disp = np.full((H, W, 2), np.nan)
        disp[20:60, 20:60, 0] = 15.0
        disp[20:60, 20:60, 1] = 10.0

        warped = warp_mask_with_holes(mask, disp)

        # Center of mass should shift by (15, 10)
        orig_ys, orig_xs = np.where(mask)
        warp_ys, warp_xs = np.where(warped)

        dx = warp_xs.mean() - orig_xs.mean()
        dy = warp_ys.mean() - orig_ys.mean()

        np.testing.assert_allclose(dx, 15.0, atol=2.0)
        np.testing.assert_allclose(dy, 10.0, atol=2.0)

    def test_warped_mask_preserves_hole(self):
        """A mask with a hole should still have a hole after warping."""
        H, W = 100, 100
        mask = np.zeros((H, W), dtype=bool)
        mask[20:80, 20:80] = True
        mask[40:60, 40:60] = False  # hole

        disp = np.full((H, W, 2), np.nan)
        disp[20:80, 20:80, 0] = 5.0
        disp[20:80, 20:80, 1] = 3.0

        warped = warp_mask_with_holes(mask, disp)

        # Center of hole should have shifted
        center_y, center_x = 50 + 3, 50 + 5
        assert not warped[center_y, center_x], "Hole center should still be empty"

    def test_resolve_mask_auto_warp(self):
        """_resolve_mask with auto-warp uses accumulated displacement."""
        from raft_dic_gui.controller import DICProcessor

        H, W = 100, 100
        roi_mask = np.zeros((H, W), dtype=bool)
        roi_mask[20:60, 20:60] = True

        kf_accumulated = {
            1: np.full((H, W, 2), np.nan, dtype=np.float64),
            50: np.full((H, W, 2), np.nan, dtype=np.float64),
        }
        kf_accumulated[1][20:60, 20:60] = 0.0
        kf_accumulated[50][20:60, 20:60, 0] = 10.0
        kf_accumulated[50][20:60, 20:60, 1] = 0.0

        # Frame 55 references key frame 50 → should get warped mask
        result = DICProcessor._resolve_mask(
            frame_num=55, list_idx=54, ref_num=50,
            roi_mask=roi_mask, kf_accumulated=kf_accumulated,
            user_masks={}, xmin=0, ymin=0, xmax=W, ymax=H,
            auto_warp=True,
        )

        # Warped mask should be shifted right
        warp_ys, warp_xs = np.where(result)
        orig_ys, orig_xs = np.where(roi_mask)
        dx = warp_xs.mean() - orig_xs.mean()
        np.testing.assert_allclose(dx, 10.0, atol=2.0)
```

**Step 2: Run test — expect FAIL**

Run: `cd server && python -m pytest tests/test_auto_warp_integration.py -v`
Expected: FAIL — `_resolve_mask` doesn't accept `auto_warp` parameter

**Step 3: Update `_resolve_mask` to support auto-warp**

Modify `raft_dic_gui/controller.py`:

```python
@staticmethod
def _resolve_mask(
    frame_num, list_idx, ref_num,
    roi_mask, kf_accumulated, user_masks,
    xmin, ymin, xmax, ymax,
    auto_warp=False,
):
    """Determine the mask for the current frame.

    Fallback chain:
    1. Per-frame user mask (from discover_masks, keyed by 0-based index)
    2. Auto-warped mask (Frame 1 ROI + accumulated displacement for ref_num)
    3. Original ROI mask
    """
    # Priority 1: user-provided per-frame mask
    if list_idx in user_masks:
        print(f"[INFO] Frame {frame_num}: using user-provided mask")
        return user_masks[list_idx]

    # Priority 2: auto-warp using accumulated displacement
    if auto_warp and ref_num in kf_accumulated and ref_num != 1:
        acc = kf_accumulated[ref_num]
        if not np.all(np.isnan(acc[..., 0])):
            from raft_dic_gui.incremental import warp_mask_with_holes
            warped = warp_mask_with_holes(roi_mask, acc)
            valid_count = int(np.sum(warped))
            orig_count = int(np.sum(roi_mask))
            print(f"[INFO] Frame {frame_num}: auto-warped mask "
                  f"({valid_count} px, {valid_count/max(1,orig_count)*100:.0f}% of original)")
            return warped

    # Priority 3: original ROI mask
    return roi_mask
```

Also update the call site in `run()`:

```python
current_mask = self._resolve_mask(
    frame_num, list_idx, ref_num,
    roi_mask, kf_accumulated, user_masks,
    xmin, ymin, xmax, ymax,
    auto_warp=(config.mask_dir is None),  # auto-warp when no custom mask folder
)
```

**Step 4: Run all tests**

Run: `cd server && python -m pytest tests/test_auto_warp_integration.py tests/test_incremental.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add raft_dic_gui/controller.py server/tests/test_auto_warp_integration.py
git commit -m "feat: auto-warp mask using accumulated displacement for key frames"
```

---

## Phase 4: Fix and Enhance ROI Import

### Task 6: Fix ROI Import button (frontend wiring)

**Files:**
- Modify: `frontend/src/components/roi/RoiToolbar.tsx:49-52`
- Modify: `frontend/src/stores/roiStore.ts` — add import dialog state

**Step 1: Add import dialog state to roiStore**

```typescript
// Add to roiStore state:
showImportDialog: boolean;
setShowImportDialog: (v: boolean) => void;
```

**Step 2: Wire Import button to open dialog**

In `RoiToolbar.tsx`, replace the empty `if (id === "import")` block:

```typescript
if (id === "import") {
  useRoiStore.getState().setShowImportDialog(true);
  return;
}
```

**Step 3: Create RoiImportDialog component**

Create `frontend/src/components/roi/RoiImportDialog.tsx`:

A modal dialog with:
- Text input for mask image file path (reuse SmallInput)
- "Min Area" slider/input (default: 50 px, range 0-5000) — filter small connected components
- "Smoothing" slider/input (default: 0 px, range 0-10) — Gaussian blur kernel before binarization
- "Preview" button → calls backend preview endpoint
- Preview display: shows imported mask as overlay
- "Apply" button → calls backend import endpoint with parameters
- "Cancel" button → closes dialog

**Step 4: Build and verify in browser**

Run: `cd frontend && npm run build`
Open app, load images, click Import button → dialog should appear.

**Step 5: Commit**

```bash
git add frontend/src/components/roi/RoiImportDialog.tsx frontend/src/components/roi/RoiToolbar.tsx frontend/src/stores/roiStore.ts
git commit -m "feat: ROI import dialog with path input"
```

---

### Task 7: Backend ROI import with noise filtering

**Files:**
- Modify: `server/routes/roi.py:121-145` (`import_mask` endpoint)
- Test: `server/tests/test_roi_import.py`

**Step 1: Write tests**

Create `server/tests/test_roi_import.py`:

```python
"""Tests for enhanced ROI import with noise filtering."""

import numpy as np
import cv2
import os
import pytest


class TestRoiImportFiltering:

    def _create_mask_with_noise(self, tmp_path, H=64, W=64):
        """Create a mask image with a large region + small noise blobs."""
        mask = np.zeros((H, W), dtype=np.uint8)
        # Main region: 30x30 block
        mask[10:40, 10:40] = 255
        # Small noise blobs
        mask[50, 50] = 255        # 1px
        mask[55:57, 55:57] = 255  # 2x2 = 4px
        path = str(tmp_path / "mask.png")
        cv2.imwrite(path, mask)
        return path

    def test_import_with_min_area_removes_noise(self, client, loaded_images, tmp_path):
        """min_area parameter removes connected components smaller than threshold."""
        mask_path = self._create_mask_with_noise(tmp_path)

        resp = client.post("/api/roi/import", json={
            "path": mask_path,
            "min_area": 10,
        })
        assert resp.status_code == 200
        data = resp.get_json()
        # Only the 30x30 block (900px) should remain
        # Noise blobs (1px and 4px) are < 10 → removed
        assert data["area_px"] == 900

    def test_import_with_min_area_zero_keeps_all(self, client, loaded_images, tmp_path):
        """min_area=0 keeps everything including noise."""
        mask_path = self._create_mask_with_noise(tmp_path)

        resp = client.post("/api/roi/import", json={
            "path": mask_path,
            "min_area": 0,
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["area_px"] == 900 + 1 + 4  # main + noise

    def test_import_with_smoothing(self, client, loaded_images, tmp_path):
        """Smoothing applies Gaussian blur before binarization."""
        mask_path = self._create_mask_with_noise(tmp_path)

        resp = client.post("/api/roi/import", json={
            "path": mask_path,
            "smooth_radius": 2,
            "min_area": 0,
        })
        assert resp.status_code == 200
        data = resp.get_json()
        # After smoothing, main region slightly expands, noise might merge or vanish
        # Just verify it succeeds and area is reasonable
        assert data["area_px"] > 0
```

**Step 2: Run tests — expect FAIL**

Run: `cd server && python -m pytest tests/test_roi_import.py -v`
Expected: FAIL — endpoint doesn't accept `min_area` / `smooth_radius`

**Step 3: Enhance import_mask endpoint**

Modify `server/routes/roi.py`:

```python
@roi_bp.route("/import", methods=["POST"])
def import_mask():
    """Import an ROI mask from an image file with optional noise filtering."""
    data = request.get_json(force=True)
    path = data.get("path", "").strip()
    min_area = int(data.get("min_area", 0))
    smooth_radius = int(data.get("smooth_radius", 0))

    if not path:
        return jsonify({"error": "Missing path"}), 400
    if session.reference_image is None:
        return jsonify({"error": "No images loaded"}), 400

    mask_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        return jsonify({"error": "Failed to read mask image"}), 400

    h, w = session.reference_image.shape[:2]
    mask_img = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_NEAREST)

    # Optional Gaussian smoothing before binarization
    if smooth_radius > 0:
        ksize = smooth_radius * 2 + 1
        mask_img = cv2.GaussianBlur(mask_img, (ksize, ksize), 0)

    binary = mask_img > 127

    # Remove small connected components
    if min_area > 0:
        binary_u8 = binary.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_u8, connectivity=8
        )
        for i in range(1, num_labels):  # skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                binary[labels == i] = False

    with session._lock:
        session.roi_mask = binary
        _update_rect()
        session.roi_confirmed = False

    area = int(session.roi_mask.sum())
    return jsonify({"rect": session.roi_rect, "area_px": area})
```

**Step 4: Run tests**

Run: `cd server && python -m pytest tests/test_roi_import.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add server/routes/roi.py server/tests/test_roi_import.py
git commit -m "feat: ROI import with min_area noise filtering and smoothing"
```

---

### Task 8: Connect frontend dialog to enhanced backend

**Files:**
- Modify: `frontend/src/components/roi/RoiImportDialog.tsx` — call API with parameters
- Modify: `frontend/src/api/roi.ts` — update `importMask` signature

**Step 1: Update API function**

In `frontend/src/api/roi.ts`:

```typescript
export async function importMask(
  path: string,
  minArea: number = 0,
  smoothRadius: number = 0,
): Promise<RoiResult> {
  const { data } = await client.post<RoiResult>("/roi/import", {
    path,
    min_area: minArea,
    smooth_radius: smoothRadius,
  });
  return data;
}
```

**Step 2: Wire dialog Apply button**

In `RoiImportDialog.tsx`, the Apply handler:

```typescript
const handleApply = async () => {
  try {
    const result = await importMask(path, minArea, smoothRadius);
    setMaskUrl(`/api/roi/mask?t=${Date.now()}`);
    setShowImportDialog(false);
    toast.success(`Imported mask: ${result.area_px.toLocaleString()} px`);
  } catch (e: any) {
    toast.error(e?.response?.data?.error || "Import failed");
  }
};
```

**Step 3: Build and verify**

Run: `cd frontend && npm run build`
Test: Load images → Import → enter mask path → adjust min area → Apply → mask overlay appears.

**Step 4: Commit**

```bash
git add frontend/src/api/roi.ts frontend/src/components/roi/RoiImportDialog.tsx
git commit -m "feat: connect ROI import dialog to backend with noise filtering"
```

---

## Phase 5: Mask Folder Frame 1 Logic

When the user selects a mask folder for incremental mode, detect whether Frame 1's mask is included. If yes, auto-apply it as the ROI (no manual drawing needed). If no, warn user to draw ROI.

### Task 9: Enhance validate-masks to report has_frame_1

**Files:**
- Modify: `server/routes/processing.py` — `validate_masks` endpoint
- Modify: `frontend/src/api/processing.ts` — update return type
- Test: `server/tests/test_processing.py` (add test)

**Step 1: Write test**

Add to `server/tests/test_processing.py` or create `server/tests/test_mask_frame1.py`:

```python
"""Test: validate-masks reports whether Frame 1 mask is present."""

import numpy as np
import cv2
import os


class TestValidateMasksFrame1:

    def test_reports_has_frame_1_true(self, client, loaded_images, tmp_path):
        """When mask folder contains frame 1 match, has_frame_1=True."""
        # loaded_images uses mock_images which creates 3 images
        # Create mask matching first image filename
        from server.session import session
        first_name = os.path.splitext(session.image_files[0])[0]
        mask = np.ones((64, 64), dtype=np.uint8) * 255
        cv2.imwrite(str(tmp_path / f"{first_name}.png"), mask)

        resp = client.post("/api/processing/validate-masks", json={
            "mask_dir": str(tmp_path),
        })
        data = resp.get_json()
        assert data["has_frame_1"] is True

    def test_reports_has_frame_1_false(self, client, loaded_images, tmp_path):
        """When mask folder does NOT contain frame 1, has_frame_1=False."""
        mask = np.ones((64, 64), dtype=np.uint8) * 255
        cv2.imwrite(str(tmp_path / "frame_099.png"), mask)

        resp = client.post("/api/processing/validate-masks", json={
            "mask_dir": str(tmp_path),
        })
        data = resp.get_json()
        assert data["has_frame_1"] is False
```

**Step 2: Run test — expect FAIL**

Run: `cd server && python -m pytest tests/test_mask_frame1.py -v`
Expected: FAIL — response doesn't contain `has_frame_1`

**Step 3: Update validate-masks endpoint**

In `server/routes/processing.py`, in the validate-masks handler, add:

```python
has_frame_1 = 0 in matched_masks  # 0-based index for frame 1

return jsonify({
    "matched_count": len(matched_masks),
    "total_frames": len(session.image_files),
    "matched_frames": [idx + 1 for idx in sorted(matched_masks.keys())],
    "has_frame_1": has_frame_1,
})
```

**Step 4: Run test**

Run: `cd server && python -m pytest tests/test_mask_frame1.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add server/routes/processing.py server/tests/test_mask_frame1.py
git commit -m "feat: validate-masks reports has_frame_1 for mask folder"
```

---

### Task 10: Apply Frame 1 mask as ROI when present

**Files:**
- Modify: `server/routes/processing.py` — add `/processing/apply-frame1-mask` endpoint
- Modify: `server/routes/roi.py` — (no change, import_mask already handles this)
- Test: `server/tests/test_mask_frame1.py` (extend)

**Step 1: Write test**

```python
def test_apply_frame1_mask_sets_roi(self, client, loaded_images, tmp_path):
    """POST /processing/apply-frame1-mask loads frame 1 mask as ROI."""
    from server.session import session
    first_name = os.path.splitext(session.image_files[0])[0]
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:50, 10:50] = 255
    cv2.imwrite(str(tmp_path / f"{first_name}.png"), mask)

    resp = client.post("/api/processing/apply-frame1-mask", json={
        "mask_dir": str(tmp_path),
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["area_px"] == 40 * 40
    assert session.roi_mask is not None
    assert session.roi_mask.sum() == 40 * 40
```

**Step 2: Implement endpoint**

In `server/routes/processing.py`:

```python
@processing_bp.route("/apply-frame1-mask", methods=["POST"])
def apply_frame1_mask():
    """Load the Frame 1 mask from a mask folder and set it as the ROI."""
    data = request.get_json(force=True)
    mask_dir = data.get("mask_dir", "").strip()

    if not mask_dir or not session.image_files:
        return jsonify({"error": "Missing mask_dir or no images loaded"}), 400

    from raft_dic_gui.mask_loader import discover_masks
    image_shape = (session.image_height, session.image_width)
    matched = discover_masks(mask_dir, session.image_files, image_shape)

    if 0 not in matched:
        return jsonify({"error": "No Frame 1 mask found in folder"}), 404

    with session._lock:
        session.roi_mask = matched[0]
        # Recompute bounding rect
        y_idx, x_idx = np.where(session.roi_mask)
        if len(x_idx) > 0:
            session.roi_rect = (
                int(x_idx.min()), int(y_idx.min()),
                int(x_idx.max()) + 1, int(y_idx.max()) + 1,
            )
        session.roi_confirmed = False

    area = int(session.roi_mask.sum())
    return jsonify({"rect": session.roi_rect, "area_px": area})
```

**Step 3: Run test**

Run: `cd server && python -m pytest tests/test_mask_frame1.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add server/routes/processing.py server/tests/test_mask_frame1.py
git commit -m "feat: apply Frame 1 mask from folder as ROI"
```

---

### Task 11: Frontend — Frame 1 mask auto-apply flow

**Files:**
- Modify: `frontend/src/components/roi/MaskSourceSelector.tsx`
- Modify: `frontend/src/stores/appStore.ts` — add `maskHasFrame1` state
- Modify: `frontend/src/api/processing.ts` — add `applyFrame1Mask`

**Step 1: Update API**

In `frontend/src/api/processing.ts`:

```typescript
export async function applyFrame1Mask(maskDir: string): Promise<RoiResult> {
  const { data } = await client.post<RoiResult>("/processing/apply-frame1-mask", {
    mask_dir: maskDir,
  });
  return data;
}
```

**Step 2: Update appStore**

Add to state:

```typescript
maskHasFrame1: boolean;
setMaskHasFrame1: (v: boolean) => void;
```

**Step 3: Update MaskSourceSelector**

After validation succeeds and `has_frame_1 === true`:
- Show message: "Frame 1 mask found — will be used as ROI"
- Show "Apply as ROI" button → calls `applyFrame1Mask(maskDir)` → refreshes mask overlay on ROI canvas

After validation succeeds and `has_frame_1 === false`:
- Show warning: "No Frame 1 mask found — please draw ROI manually on the canvas"

**Step 4: Build and verify**

Run: `cd frontend && npm run build`
Test full flow in browser.

**Step 5: Commit**

```bash
git add frontend/src/components/roi/MaskSourceSelector.tsx frontend/src/stores/appStore.ts frontend/src/api/processing.ts
git commit -m "feat: auto-apply Frame 1 mask as ROI from mask folder"
```

---

## Phase 6: UI Polish — Auto Warp Label Fix & Key Frame Mask Hints

### Task 12: Fix misleading "Auto Warp" UI label

**Files:**
- Modify: `frontend/src/components/roi/ProcessingParams.tsx:19-22`

**Step 1: Update label to reflect actual behavior**

The "Auto Warp" option now actually works (after Phase 3). But the "auto" / "folder" terminology can be clearer:

```typescript
const maskSourceOptions = [
  { value: "auto", label: "Auto (warp ROI)" },
  { value: "folder", label: "Custom (load folder)" },
];
```

**Step 2: Build and verify**

Run: `cd frontend && npm run build`

**Step 3: Commit**

```bash
git add frontend/src/components/roi/ProcessingParams.tsx
git commit -m "fix: clarify mask source labels in incremental settings"
```

---

### Task 13: Validate enhancement — warn about unmatched key frames

**Files:**
- Modify: `frontend/src/components/roi/MaskSourceSelector.tsx`

**Step 1: Compare matched_frames against keyFrames**

After validation, cross-reference:

```typescript
const keyFrames = useAppStore((s) => s.keyFrames);
// ...
{validationResult && maskSource === "folder" && (
  <>
    {/* Existing matched count message */}
    {/* New: warn about key frames without masks */}
    {(() => {
      const unmatched = keyFrames.filter(
        kf => kf !== 1 && !validationResult.matched_frames.includes(kf)
      );
      if (unmatched.length > 0) {
        return (
          <span className="text-[10px] text-yellow-400">
            Missing masks for key frames: {unmatched.join(", ")}
          </span>
        );
      }
      return null;
    })()}
  </>
)}
```

**Step 2: Build and verify**

Run: `cd frontend && npm run build`

**Step 3: Commit**

```bash
git add frontend/src/components/roi/MaskSourceSelector.tsx
git commit -m "feat: warn about key frames without matching masks"
```

---

## Phase 7: Full Integration Test

### Task 14: End-to-end test with synthetic data

**Files:**
- Create: `server/tests/test_e2e_incremental_warp.py`

**Step 1: Write integration test**

```python
"""End-to-end test: incremental mode with auto-warp on synthetic images.

Creates a sequence of synthetic images with known uniform translation,
runs incremental processing with auto-warp, and verifies:
1. Auto-warped masks shift correctly per key frame
2. Accumulated displacement matches expected total translation
3. Envelope rect grows to contain warped regions
"""

import numpy as np
import os
import cv2
import pytest
from raft_dic_gui.controller import DICProcessor
from raft_dic_gui.config import DICConfig


@pytest.fixture
def synthetic_sequence(tmp_path):
    """Create 5 frames with 10px/frame rightward translation of a textured block."""
    H, W = 128, 256
    np.random.seed(42)
    texture = (np.random.rand(60, 60) * 255).astype(np.uint8)

    for i in range(5):
        img = np.zeros((H, W), dtype=np.uint8)
        x_offset = 40 + i * 10
        img[30:90, x_offset:x_offset+60] = texture
        cv2.imwrite(str(tmp_path / f"frame_{i+1:03d}.png"), img)

    return str(tmp_path), H, W


class TestE2EIncrementalWarp:

    @pytest.mark.skipif(
        not os.path.exists("models"),
        reason="RAFT model not available for integration test"
    )
    def test_auto_warp_accumulation(self, synthetic_sequence, tmp_path):
        """Full pipeline: incremental + auto-warp on synthetic translation."""
        img_dir, H, W = synthetic_sequence
        out_dir = str(tmp_path / "output")

        roi_mask = np.zeros((H, W), dtype=bool)
        roi_mask[25:95, 35:105] = True
        roi_rect = (35, 25, 105, 95)

        config = DICConfig(
            img_dir=img_dir,
            project_root=out_dir,
            model_path="models/RAFTcorr_fine_parameter_more.pth",
            mode="incremental",
            key_frames=[1, 3],
            key_frame_interval=None,
            mask_dir=None,  # triggers auto-warp
        )

        processor = DICProcessor()
        results = processor.run(config, roi_mask, roi_rect)

        assert len(results) == 4  # frames 2-5

        # Envelope should have expanded rightward
        assert processor.envelope_rect[2] > roi_rect[2]
```

**Step 2: Run test**

Run: `cd server && python -m pytest tests/test_e2e_incremental_warp.py -v`
Expected: PASS (or skip if no model available)

**Step 3: Commit**

```bash
git add server/tests/test_e2e_incremental_warp.py
git commit -m "test: end-to-end incremental mode with auto-warp"
```

---

## Dependency Graph

```
Task 1 (NaN-aware interp)
  └→ Task 2 (replace cv2.remap)
       └→ Task 3 (full-image accumulation)
            ├→ Task 4 (session/render pipeline update)
            └→ Task 5 (auto-warp in _resolve_mask)
                 └→ Task 12 (UI label fix)
                      └→ Task 14 (E2E test)

Task 6 (ROI import frontend fix) ──────────────────────┐
  └→ Task 8 (connect dialog to backend)                 │
                                                        │
Task 7 (ROI import backend filtering)                   │── can run in parallel
  └→ Task 8 (connect dialog to backend)                 │   with Phase 1-3
                                                        │
Task 9 (validate-masks has_frame_1)                     │
  └→ Task 10 (apply-frame1-mask endpoint)               │
       └→ Task 11 (frontend Frame 1 flow)               │
            └→ Task 13 (key frame mask warnings) ───────┘
```

**Parallel opportunities:**
- Phase 1-3 (backend core) and Phase 4 (ROI import) are independent
- Task 6+7 can be done in parallel
- Task 9+10 can start before Phase 3 finishes

---

## Summary

| Phase | Tasks | Description |
|-------|-------|-------------|
| 1 | 1-2 | NaN-aware bilinear interpolation |
| 2 | 3-4 | Full-image coordinate accumulation + dynamic envelope |
| 3 | 5 | Auto-warp mask for key frames |
| 4 | 6-8 | Fix + enhance ROI import (noise filter, smoothing) |
| 5 | 9-11 | Mask folder Frame 1 auto-apply as ROI |
| 6 | 12-13 | UI polish (labels, key frame warnings) |
| 7 | 14 | End-to-end integration test |

Total: **14 tasks**, ~7 commits.
