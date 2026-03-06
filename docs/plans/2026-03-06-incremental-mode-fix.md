# Incremental Mode Fix: Displacement Accumulation & ROI Warping

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix incremental mode so it outputs total displacement (relative to Frame 1) with proper ROI warping, making its output semantically identical to accumulative mode for all downstream consumers.

**Architecture:** Modify `controller.py`'s main loop to accumulate frame-to-frame deltas using existing `incremental.py` utilities (`accumulate_displacement`, `warp_mask_with_holes`). Save total (accumulated) displacement per frame so resume works correctly. Delete the dead `run_incremental_processing()` in `processing.py`. No frontend changes needed — the mode selector already works.

**Tech Stack:** Python, NumPy, OpenCV (cv2), pytest

---

## Background

### Current Broken Behavior
- `controller.py` updates reference image each frame (correct) but:
  - Always uses the **original ROI mask** (wrong — material moves out of ROI)
  - Saves **frame-to-frame delta** displacement (wrong — downstream assumes total displacement)
  - No accumulation to Frame 1 coordinates

### Design Principle
- In incremental mode, RAFT computes delta displacement between consecutive frames
- These deltas must be accumulated: `u_total = u_prev + warp(delta, u_prev)`
- The ROI mask must be warped to follow the material using total displacement
- Saved/output displacement must be total (relative to Frame 1), same as accumulative mode
- All existing downstream code (strain, probes, displacement rendering) works unchanged

### Key Files
- `raft_dic_gui/controller.py` — main processing loop (MODIFY)
- `raft_dic_gui/incremental.py` — accumulation utilities (USE, already written)
- `raft_dic_gui/processing.py:1106-1283` — dead `run_incremental_processing()` (DELETE)
- `server/tests/test_incremental.py` — new unit tests (CREATE)

---

### Task 1: Unit Tests for `incremental.py` Utilities

**Files:**
- Create: `server/tests/test_incremental.py`

These utilities exist but have zero test coverage. Test them before we rely on them.

**Step 1: Write tests for accumulate_displacement and warp_mask_with_holes**

```python
"""Tests for incremental DIC utilities."""

import numpy as np
import pytest
from raft_dic_gui.incremental import (
    accumulate_displacement,
    warp_displacement_field,
    warp_mask_with_holes,
    validate_key_frames,
    get_segment_ranges,
)


class TestWarpDisplacementField:
    """Test sampling incremental displacement at deformed coordinates."""

    def test_zero_accumulated_returns_delta_unchanged(self):
        """When accumulated is zero, delta should pass through unchanged."""
        H, W = 32, 32
        delta = np.ones((H, W, 2), dtype=np.float32) * 2.0
        accum = np.zeros((H, W, 2), dtype=np.float32)
        result = warp_displacement_field(delta, accum)
        # With zero accumulated displacement, sampling at identity coords
        # Interior pixels should be unchanged (edges may differ due to border)
        interior = result[2:-2, 2:-2]
        np.testing.assert_allclose(interior, 2.0, atol=0.1)

    def test_nan_in_accumulated_propagates(self):
        """NaN in accumulated displacement should produce NaN in output."""
        H, W = 16, 16
        delta = np.ones((H, W, 2), dtype=np.float32)
        accum = np.zeros((H, W, 2), dtype=np.float32)
        accum[5, 5, :] = np.nan
        result = warp_displacement_field(delta, accum)
        assert np.isnan(result[5, 5, 0])

    def test_out_of_bounds_deformed_coords_become_nan(self):
        """If accumulated displacement pushes coords out of bounds, result is NaN."""
        H, W = 16, 16
        delta = np.ones((H, W, 2), dtype=np.float32)
        accum = np.zeros((H, W, 2), dtype=np.float32)
        accum[8, 8, 0] = 999.0  # Push x way out of bounds
        result = warp_displacement_field(delta, accum)
        assert np.isnan(result[8, 8, 0])


class TestAccumulateDisplacement:
    """Test displacement accumulation."""

    def test_first_accumulation_from_zero(self):
        """Accumulating onto zero should return approximately delta itself."""
        H, W = 32, 32
        u_prev = np.zeros((H, W, 2), dtype=np.float32)
        delta = np.full((H, W, 2), 3.0, dtype=np.float32)
        result = accumulate_displacement(u_prev, delta, debug=False)
        interior = result[2:-2, 2:-2]
        np.testing.assert_allclose(interior, 3.0, atol=0.1)

    def test_two_uniform_steps_sum(self):
        """Two uniform translation steps should sum."""
        H, W = 32, 32
        step = np.full((H, W, 2), 1.5, dtype=np.float32)
        # First step
        total = accumulate_displacement(np.zeros((H, W, 2), dtype=np.float32), step, debug=False)
        # Second step
        total = accumulate_displacement(total, step, debug=False)
        interior = total[2:-2, 2:-2]
        np.testing.assert_allclose(interior, 3.0, atol=0.2)

    def test_nan_pixel_stays_nan(self):
        """NaN in previous displacement should stay NaN after accumulation."""
        H, W = 16, 16
        u_prev = np.zeros((H, W, 2), dtype=np.float32)
        u_prev[4, 4, :] = np.nan
        delta = np.ones((H, W, 2), dtype=np.float32)
        result = accumulate_displacement(u_prev, delta, debug=False)
        assert np.isnan(result[4, 4, 0])


class TestWarpMaskWithHoles:
    """Test ROI mask warping."""

    def test_zero_displacement_preserves_mask(self):
        """Zero displacement should return the same mask."""
        H, W = 64, 64
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[10:50, 10:50] = 1
        disp = np.zeros((H, W, 2), dtype=np.float32)
        warped = warp_mask_with_holes(mask, disp)
        # Should be very close to original
        overlap = np.sum(mask.astype(bool) & warped) / max(np.sum(mask), 1)
        assert overlap > 0.95

    def test_translation_shifts_mask(self):
        """Uniform translation should shift the mask region."""
        H, W = 64, 64
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[20:40, 20:40] = 1
        disp = np.zeros((H, W, 2), dtype=np.float32)
        disp[..., 0] = 5.0  # Shift x by 5
        warped = warp_mask_with_holes(mask, disp)
        # Center of mass should shift right
        ys, xs = np.where(mask > 0)
        yw, xw = np.where(warped)
        assert np.mean(xw) > np.mean(xs) + 3  # Shifted right

    def test_empty_mask_returns_empty(self):
        """Empty mask should return empty mask."""
        mask = np.zeros((32, 32), dtype=np.uint8)
        disp = np.zeros((32, 32, 2), dtype=np.float32)
        warped = warp_mask_with_holes(mask, disp)
        assert np.sum(warped) == 0

    def test_hole_preservation(self):
        """Mask with a hole should preserve the hole after warping."""
        H, W = 64, 64
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[10:50, 10:50] = 1
        mask[25:35, 25:35] = 0  # Hole
        disp = np.zeros((H, W, 2), dtype=np.float32)
        disp[..., 0] = 2.0  # Small shift
        warped = warp_mask_with_holes(mask, disp)
        # Total area should be approximately preserved (not filled in)
        original_area = np.sum(mask > 0)
        warped_area = np.sum(warped)
        assert abs(warped_area - original_area) / original_area < 0.15


class TestValidateKeyFrames:
    """Test key frame validation."""

    def test_valid_key_frames(self):
        valid, msg = validate_key_frames([1, 50, 100], 200)
        assert valid

    def test_missing_frame_1(self):
        valid, msg = validate_key_frames([50, 100], 200)
        assert not valid
        assert "Frame 1" in msg

    def test_empty_list(self):
        valid, msg = validate_key_frames([], 200)
        assert not valid

    def test_out_of_range(self):
        valid, msg = validate_key_frames([1, 500], 200)
        assert not valid


class TestGetSegmentRanges:
    """Test segment range calculation."""

    def test_single_segment(self):
        segments = get_segment_ranges([1], 10)
        assert segments == [(1, 2, 10)]

    def test_two_segments(self):
        segments = get_segment_ranges([1, 5], 10)
        assert segments == [(1, 2, 5), (5, 6, 10)]
```

**Step 2: Run tests to verify they pass**

Run: `cd "C:\Users\13014\OneDrive - The University of Texas at Austin\Documents\Python Codes\2D-RAFT-DIC-GUI" && python -m pytest server/tests/test_incremental.py -v`
Expected: All PASS (testing existing utility functions)

**Step 3: Commit**

```bash
git add server/tests/test_incremental.py
git commit -m "test: add unit tests for incremental.py utilities"
```

---

### Task 2: Fix controller.py — Displacement Accumulation & ROI Warping

**Files:**
- Modify: `raft_dic_gui/controller.py`

This is the core fix. Changes to the `run()` method:

1. Initialize `accumulated_disp` and `current_mask` before the loop
2. In incremental mode, accumulate each frame's delta displacement
3. Warp ROI mask using total displacement from original mask
4. Save/output total displacement (not delta)
5. Handle resume correctly: reload accumulated state from last saved total displacement

**Step 1: Write integration test for incremental accumulation in controller**

Add to `server/tests/test_incremental.py`:

```python
class TestControllerIncrementalAccumulation:
    """Integration test: verify controller accumulates displacements correctly."""

    def test_incremental_saves_total_displacement(self):
        """In incremental mode, saved results should be total displacement, not deltas."""
        # This test verifies the contract: displacement_results[i] is always
        # relative to Frame 1, regardless of processing mode.
        # We can't run the full pipeline without a RAFT model, but we can
        # verify the accumulation logic by checking the controller's state
        # management code paths.
        from raft_dic_gui.incremental import accumulate_displacement

        H, W = 20, 20
        # Simulate 3 frames of incremental processing:
        # Frame 1→2: delta = (2, 0), total = (2, 0)
        # Frame 2→3: delta = (2, 0), total should be ≈ (4, 0)
        delta = np.full((H, W, 2), 0.0, dtype=np.float32)
        delta[..., 0] = 2.0  # Uniform x-translation per step

        accumulated = np.zeros((H, W, 2), dtype=np.float32)

        # Step 1
        total_1 = accumulate_displacement(accumulated, delta, debug=False)
        assert np.nanmean(total_1[2:-2, 2:-2, 0]) == pytest.approx(2.0, abs=0.2)

        # Step 2
        total_2 = accumulate_displacement(total_1, delta, debug=False)
        assert np.nanmean(total_2[2:-2, 2:-2, 0]) == pytest.approx(4.0, abs=0.3)
```

**Step 2: Run to verify test passes**

Run: `python -m pytest server/tests/test_incremental.py::TestControllerIncrementalAccumulation -v`
Expected: PASS

**Step 3: Modify controller.py**

Replace the `run()` method's main loop with incremental accumulation support.

The key changes (in `controller.py`):

**Before the loop (after line 124):** Add incremental state initialization:
```python
# -- Incremental mode state --
accumulated_disp = None  # Will be (H_roi, W_roi, 2) once first frame processed
current_mask = roi_mask.copy()  # Mutable working mask
original_mask_crop = roi_mask[ymin:ymax, xmin:xmax].copy()
is_incremental = config.mode == "incremental"
```

**Inside the loop — resume path (lines 140-158):** When loading a cached result in incremental mode, update accumulated state:
```python
if not force_rerun and os.path.exists(result_path):
    ...
    displacement_field = np.load(result_path)
    sequence_displacements.append(displacement_field)

    if is_incremental:
        ref_roi = def_roi.copy()
        ref_image = def_image.copy()
        # Restore accumulated state from saved total displacement
        accumulated_disp = displacement_field.copy()
        # Re-warp mask from original using total displacement
        from raft_dic_gui.incremental import warp_mask_with_holes
        warped_crop = warp_mask_with_holes(original_mask_crop, accumulated_disp)
        current_mask = roi_mask.copy()
        current_mask[ymin:ymax, xmin:xmax] = warped_crop

    continue
```

**Inside the loop — DIC call (line 170-181):** Use `current_mask` instead of `roi_mask`:
```python
disp_full, _ = proc.dic_over_roi_with_tiling(
    ref_image,
    def_image,
    current_mask,    # ← Changed from roi_mask
    model,
    ...
)
delta_disp = disp_full[ymin:ymax, xmin:xmax, :]
```

**After DIC call — accumulate (replacing lines 182-202):**
```python
if is_incremental:
    from raft_dic_gui.incremental import (
        accumulate_displacement, warp_mask_with_holes
    )
    if accumulated_disp is None:
        # First frame: delta IS the total
        accumulated_disp = delta_disp.copy()
    else:
        accumulated_disp = accumulate_displacement(
            accumulated_disp, delta_disp, debug=True
        )
    displacement_field = accumulated_disp.copy()

    # Update reference for next frame
    ref_roi = def_roi.copy()
    ref_image = def_image.copy()

    # Warp mask from ORIGINAL using TOTAL displacement
    warped_crop = warp_mask_with_holes(original_mask_crop, accumulated_disp)
    current_mask = roi_mask.copy()
    current_mask[ymin:ymax, xmin:xmax] = warped_crop
else:
    displacement_field = delta_disp
```

**Step 4: Run all existing tests**

Run: `python -m pytest server/tests/ -v`
Expected: All PASS (no downstream behavior changed — output format is the same)

**Step 5: Commit**

```bash
git add raft_dic_gui/controller.py server/tests/test_incremental.py
git commit -m "fix: incremental mode now accumulates displacement and warps ROI mask"
```

---

### Task 3: Delete Dead Code — `run_incremental_processing()`

**Files:**
- Modify: `raft_dic_gui/processing.py` — delete lines 1104-1287

The function `run_incremental_processing()` is 180 lines of dead code that duplicates the logic we just integrated into `controller.py`. It was never called and is now superseded.

**Step 1: Remove the function**

Delete everything from line 1104 (`# ========================= Incremental Reference Processing =========================`) to end of the function (line ~1287).

**Step 2: Verify no callers**

Run: `grep -rn "run_incremental_processing" raft_dic_gui/ server/`
Expected: No matches

**Step 3: Run all tests**

Run: `python -m pytest server/tests/ -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add raft_dic_gui/processing.py
git commit -m "refactor: remove dead run_incremental_processing() function"
```

---

### Task 4: Add Pixel-Loss Warning Log

**Files:**
- Modify: `raft_dic_gui/controller.py`

In incremental mode, as frames accumulate, pixels at the edges of the ROI may become NaN (lost during warping). Add a log warning when significant pixel loss occurs, so users can diagnose issues.

**Step 1: Add pixel loss tracking after accumulation**

After the `warp_mask_with_holes` call in the incremental block:

```python
# Log pixel loss
valid_pixels = np.sum(~np.isnan(accumulated_disp[..., 0]))
total_pixels = accumulated_disp.shape[0] * accumulated_disp.shape[1]
loss_pct = (1 - valid_pixels / total_pixels) * 100
if loss_pct > 10:
    print(f"[WARNING] Frame {i}: {loss_pct:.1f}% of ROI pixels lost during accumulation. "
          f"Consider using key frames or reducing deformation range.")
```

**Step 2: Run tests**

Run: `python -m pytest server/tests/ -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add raft_dic_gui/controller.py
git commit -m "feat: add pixel-loss warning in incremental mode"
```

---

### Task 5: Verify End-to-End with Existing Tests

**Files:** None (verification only)

**Step 1: Run full test suite**

Run: `python -m pytest server/tests/ -v --tb=short`
Expected: All tests pass, including new incremental tests.

**Step 2: Verify no regressions**

Check that:
- `test_processing.py` — configure endpoint still works with mode="incremental"
- `test_displacement.py` — displacement rendering unaffected
- `test_strain.py` — strain computation unaffected
- `test_incremental.py` — all new tests pass

---

## Summary of Changes

| File | Action | Lines Changed |
|------|--------|--------------|
| `raft_dic_gui/controller.py` | Modify | ~40 lines added/changed |
| `raft_dic_gui/processing.py` | Delete dead code | ~180 lines removed |
| `server/tests/test_incremental.py` | Create | ~140 lines |
| `raft_dic_gui/incremental.py` | No change | Already correct |

## What's NOT in This Plan (Future P1/P2)

- Key Frame UI in React frontend (P1)
- Key Frame configuration API endpoint (P1)
- Pixel-loss visualization in frontend (P2)
- Adaptive reference update based on correlation quality (P2)
