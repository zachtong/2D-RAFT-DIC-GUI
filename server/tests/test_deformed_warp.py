"""Tests for deformed-view inverse mapping (server/deformed_warp.py)."""

import numpy as np
import pytest

from server.deformed_warp import compute_inverse_map, warp_data_inverse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROI_RECT = (100, 100, 400, 400)  # x0, y0, x1, y1
ROI_H, ROI_W = 300, 300
IMAGE_SHAPE = (512, 512)


def _uniform_disp(u_val: float, v_val: float = 0.0):
    U = np.full((ROI_H, ROI_W), u_val, dtype=np.float64)
    V = np.full((ROI_H, ROI_W), v_val, dtype=np.float64)
    return U, V


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestComputeInverseMap:
    """Test that the fixed-point iteration converges for various displacements."""

    def test_zero_displacement(self):
        """Zero displacement: deformed ROI = reference ROI."""
        U, V = _uniform_disp(0.0)
        inv = compute_inverse_map(U, V, ROI_RECT, IMAGE_SHAPE)

        assert inv.validity_mask.sum() >= ROI_H * ROI_W * 0.99

    def test_small_translation(self):
        """Small uniform translation should produce ~100% coverage."""
        U, V = _uniform_disp(-5.0, 3.0)
        inv = compute_inverse_map(U, V, ROI_RECT, IMAGE_SHAPE)

        coverage = inv.validity_mask.sum() / (ROI_H * ROI_W)
        assert coverage > 0.99, f"Coverage {coverage:.1%} too low for small translation"

    def test_large_translation_full_coverage(self):
        """Large uniform translation (-60px) must still produce full coverage.

        This is the exact scenario that triggered the original bug: the
        fixed-point iteration's initial guess landed outside the ROI,
        causing the leading edge to be marked invalid.
        """
        U, V = _uniform_disp(-60.0)
        inv = compute_inverse_map(U, V, ROI_RECT, IMAGE_SHAPE)

        valid = int(inv.validity_mask.sum())
        expected = ROI_H * ROI_W
        assert valid == expected, (
            f"Expected {expected} valid pixels, got {valid} "
            f"({valid / expected:.1%} coverage)"
        )

    def test_large_positive_translation(self):
        """Large rightward translation (+60px)."""
        U, V = _uniform_disp(60.0)
        inv = compute_inverse_map(U, V, ROI_RECT, IMAGE_SHAPE)

        coverage = inv.validity_mask.sum() / (ROI_H * ROI_W)
        assert coverage > 0.99

    def test_diagonal_translation(self):
        """Large diagonal translation."""
        U, V = _uniform_disp(-40.0, -30.0)
        inv = compute_inverse_map(U, V, ROI_RECT, IMAGE_SHAPE)

        coverage = inv.validity_mask.sum() / (ROI_H * ROI_W)
        assert coverage > 0.99

    def test_deformed_bbox_position(self):
        """Output bounding box should follow the displacement."""
        U, V = _uniform_disp(-60.0)
        inv = compute_inverse_map(U, V, ROI_RECT, IMAGE_SHAPE)

        # Deformed left edge ≈ x0 + U = 100 + (-60) = 40
        assert inv.out_x0 <= 40
        # Deformed right edge ≈ x1 + U = 400 + (-60) = 340
        assert inv.out_x1 >= 340


class TestWarpDataInverse:
    """Test that data is correctly warped to deformed coordinates."""

    def test_uniform_value_preserved(self):
        """Uniform data should remain uniform after warping."""
        U, V = _uniform_disp(-60.0)
        inv = compute_inverse_map(U, V, ROI_RECT, IMAGE_SHAPE)

        data = np.full((ROI_H, ROI_W), 42.0)
        warped = warp_data_inverse(data, inv, IMAGE_SHAPE)

        valid = ~np.isnan(warped)
        assert valid.sum() == ROI_H * ROI_W
        np.testing.assert_allclose(warped[valid], 42.0, atol=1e-6)

    def test_gradient_field_warped_correctly(self):
        """A horizontal gradient should shift spatially after warping."""
        U, V = _uniform_disp(-20.0)
        inv = compute_inverse_map(U, V, ROI_RECT, IMAGE_SHAPE)

        # Create gradient: value = column index
        data = np.tile(np.arange(ROI_W, dtype=np.float64), (ROI_H, 1))
        warped = warp_data_inverse(data, inv, IMAGE_SHAPE)

        # Check that the warped data at the center of the deformed ROI
        # corresponds to the correct reference column
        x0, y0 = ROI_RECT[0], ROI_RECT[1]
        center_def_x = x0 + ROI_W // 2 + int(-20)  # deformed center
        center_def_y = y0 + ROI_H // 2
        val = warped[center_def_y, center_def_x]
        expected_col = ROI_W // 2  # reference column index
        assert abs(val - expected_col) < 2.0, (
            f"Expected ~{expected_col}, got {val}"
        )

    def test_nan_in_data_preserved(self):
        """NaN regions in source data should remain NaN after warping."""
        U, V = _uniform_disp(-10.0)
        inv = compute_inverse_map(U, V, ROI_RECT, IMAGE_SHAPE)

        data = np.ones((ROI_H, ROI_W))
        data[100:200, 100:200] = np.nan  # hole in the middle

        warped = warp_data_inverse(data, inv, IMAGE_SHAPE)

        # The NaN hole should appear in the warped output (shifted by U)
        total_valid = np.sum(~np.isnan(warped))
        # Original valid: 300*300 - 100*100 = 80000
        assert total_valid < ROI_H * ROI_W
        assert total_valid > 0
