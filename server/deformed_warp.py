"""Inverse mapping (backward warping) for deformed-mode visualization.

When background == "deformed", overlay data (displacement/strain) must be
transformed from the reference coordinate system to the deformed coordinate
system so geometry matches the deformed background image.

Core algorithm — for each output pixel (x', y') in deformed space, solve for
its reference coordinate (x, y) via fixed-point iteration:
    x_ref^(k+1) = x_def - U_interp(x_ref^k, y_ref^k)
    y_ref^(k+1) = y_def - V_interp(x_ref^k, y_ref^k)
Then sample the data at (x, y) using bilinear interpolation.
"""

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import map_coordinates


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class InverseMapResult:
    """Precomputed inverse mapping from deformed to reference coordinates."""
    frame_idx: int
    ref_row_coords: np.ndarray   # (out_h, out_w) float64 — ROI-local y coords
    ref_col_coords: np.ndarray   # (out_h, out_w) float64 — ROI-local x coords
    validity_mask: np.ndarray    # (out_h, out_w) bool
    out_x0: int                  # output bounding box in full-image coords
    out_y0: int
    out_x1: int
    out_y1: int


class InverseMapCache:
    """Thread-safe LRU cache for InverseMapResult (keyed by frame index)."""

    def __init__(self, max_size: int = 5):
        self._max_size = max_size
        self._cache: OrderedDict[int, InverseMapResult] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, frame_idx: int) -> Optional[InverseMapResult]:
        with self._lock:
            if frame_idx in self._cache:
                self._cache.move_to_end(frame_idx)
                return self._cache[frame_idx]
            return None

    def put(self, frame_idx: int, result: InverseMapResult) -> None:
        with self._lock:
            if frame_idx in self._cache:
                self._cache.move_to_end(frame_idx)
            else:
                if len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)
            self._cache[frame_idx] = result

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


# ---------------------------------------------------------------------------
# Core algorithms
# ---------------------------------------------------------------------------

def compute_inverse_map(
    U: np.ndarray,
    V: np.ndarray,
    roi_rect: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
    n_iter: int = 5,
) -> InverseMapResult:
    """Compute the inverse mapping from deformed to reference coordinates.

    Parameters
    ----------
    U, V : (roi_h, roi_w) displacement fields (may contain NaN)
    roi_rect : (x0, y0, x1, y1) ROI bounding box in full-image coords
    image_shape : (H, W) of the full image
    n_iter : number of fixed-point iterations

    Returns
    -------
    InverseMapResult with precomputed reference coordinates for each output pixel.
    """
    x0, y0, x1, y1 = roi_rect
    roi_h, roi_w = y1 - y0, x1 - x0
    img_h, img_w = image_shape

    # --- Step 1: Compute deformed bounding box via forward mapping ---
    valid_mask = ~(np.isnan(U) | np.isnan(V))
    row_grid, col_grid = np.mgrid[0:roi_h, 0:roi_w]

    # Absolute positions of valid ROI pixels in deformed frame
    if valid_mask.any():
        def_x = col_grid[valid_mask] + x0 + U[valid_mask]
        def_y = row_grid[valid_mask] + y0 + V[valid_mask]
        out_x0 = max(0, int(np.floor(def_x.min())) - 1)
        out_y0 = max(0, int(np.floor(def_y.min())) - 1)
        out_x1 = min(img_w, int(np.ceil(def_x.max())) + 2)
        out_y1 = min(img_h, int(np.ceil(def_y.max())) + 2)
    else:
        # No valid data — return empty mapping
        return InverseMapResult(
            frame_idx=-1,
            ref_row_coords=np.empty((0, 0)),
            ref_col_coords=np.empty((0, 0)),
            validity_mask=np.empty((0, 0), dtype=bool),
            out_x0=x0, out_y0=y0, out_x1=x1, out_y1=y1,
        )

    out_h = out_y1 - out_y0
    out_w = out_x1 - out_x0

    # --- Step 2: Create output meshgrid (deformed coordinates) ---
    out_rows, out_cols = np.mgrid[out_y0:out_y1, out_x0:out_x1]
    out_rows = out_rows.astype(np.float64)
    out_cols = out_cols.astype(np.float64)

    # --- Step 3: Prepare displacement fields for interpolation ---
    U_clean = np.nan_to_num(U, nan=0.0).astype(np.float64)
    V_clean = np.nan_to_num(V, nan=0.0).astype(np.float64)

    # Validity mask for interpolation (1.0 where data exists, 0.0 where NaN)
    data_valid = valid_mask.astype(np.float64)

    # --- Step 4: Fixed-point iteration ---
    # Initial guess: reference coords = deformed coords
    x_ref = out_cols.copy()
    y_ref = out_rows.copy()

    for _ in range(n_iter):
        # Convert to ROI-local coordinates for map_coordinates
        local_row = y_ref - y0
        local_col = x_ref - x0

        # Sample U and V at current reference estimate
        coords = np.array([local_row.ravel(), local_col.ravel()])
        U_sampled = map_coordinates(
            U_clean, coords, order=1, mode='constant', cval=0.0
        ).reshape(out_h, out_w)
        V_sampled = map_coordinates(
            V_clean, coords, order=1, mode='constant', cval=0.0
        ).reshape(out_h, out_w)

        # Update: x_ref = x_def - U, y_ref = y_def - V
        x_ref = out_cols - U_sampled
        y_ref = out_rows - V_sampled

    # --- Step 5: Compute validity mask ---
    # Final ROI-local coordinates
    final_local_row = y_ref - y0
    final_local_col = x_ref - x0

    # Interpolate the data validity mask
    coords_final = np.array([final_local_row.ravel(), final_local_col.ravel()])
    valid_interp = map_coordinates(
        data_valid, coords_final, order=1, mode='constant', cval=0.0
    ).reshape(out_h, out_w)

    # Check bounds: reference coords must land within ROI
    in_bounds = (
        (final_local_row >= 0) & (final_local_row <= roi_h - 1) &
        (final_local_col >= 0) & (final_local_col <= roi_w - 1)
    )

    # Combined validity: interpolated source is valid AND within bounds
    validity = (valid_interp > 0.5) & in_bounds

    return InverseMapResult(
        frame_idx=-1,  # caller sets this
        ref_row_coords=final_local_row,
        ref_col_coords=final_local_col,
        validity_mask=validity,
        out_x0=out_x0, out_y0=out_y0,
        out_x1=out_x1, out_y1=out_y1,
    )


def warp_data_inverse(
    data: np.ndarray,
    inv_map: InverseMapResult,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Warp ROI-sized data to deformed coordinates using a precomputed inverse map.

    Parameters
    ----------
    data : (roi_h, roi_w) array (may contain NaN)
    inv_map : precomputed InverseMapResult
    image_shape : (H, W) of the full output image

    Returns
    -------
    (H, W) array with warped data placed at the correct position; NaN elsewhere.
    """
    if inv_map.ref_row_coords.size == 0:
        return np.full(image_shape, np.nan)

    # Clean NaN for interpolation
    valid_src = ~np.isnan(data)
    data_clean = np.nan_to_num(data, nan=0.0).astype(np.float64)
    src_valid = valid_src.astype(np.float64)

    out_h = inv_map.out_y1 - inv_map.out_y0
    out_w = inv_map.out_x1 - inv_map.out_x0

    # Interpolate data values
    coords = np.array([
        inv_map.ref_row_coords.ravel(),
        inv_map.ref_col_coords.ravel(),
    ])
    warped = map_coordinates(
        data_clean, coords, order=1, mode='constant', cval=0.0
    ).reshape(out_h, out_w)

    # Interpolate validity mask
    valid_warped = map_coordinates(
        src_valid, coords, order=1, mode='constant', cval=0.0
    ).reshape(out_h, out_w)

    # Combined validity
    final_valid = (valid_warped > 0.5) & inv_map.validity_mask

    # Place into full image
    full = np.full(image_shape, np.nan)
    patch = np.where(final_valid, warped, np.nan)
    full[inv_map.out_y0:inv_map.out_y1, inv_map.out_x0:inv_map.out_x1] = patch

    return full


# ---------------------------------------------------------------------------
# Strain upsampling utility
# ---------------------------------------------------------------------------

def upsample_strain_to_roi(
    strain_data: np.ndarray,
    roi_h: int,
    roi_w: int,
) -> np.ndarray:
    """Upsample (potentially downsampled) strain data to ROI dimensions.

    NaN-safe: uses INTER_LINEAR for values and INTER_NEAREST for the mask.
    """
    mask_valid = ~np.isnan(strain_data)
    data_clean = np.nan_to_num(strain_data, nan=0.0)
    data_resized = cv2.resize(
        data_clean, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR
    )
    mask_resized = cv2.resize(
        mask_valid.astype(np.uint8), (roi_w, roi_h),
        interpolation=cv2.INTER_NEAREST,
    )
    data_resized[mask_resized == 0] = np.nan
    return data_resized


# ---------------------------------------------------------------------------
# Unified entry point for render endpoints
# ---------------------------------------------------------------------------

def get_warped_full_data(
    data: np.ndarray,
    frame_idx: int,
    U: np.ndarray,
    V: np.ndarray,
    roi_rect: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
    cache: InverseMapCache,
    needs_upsample: bool = False,
    roi_h: int = 0,
    roi_w: int = 0,
) -> np.ndarray:
    """Convenience function for render endpoints: upsample if needed, compute
    or retrieve cached inverse map, then warp data to deformed coordinates.

    Parameters
    ----------
    data : ROI-sized (or downsampled) data array
    frame_idx : displacement frame index (for caching)
    U, V : (roi_h, roi_w) displacement fields
    roi_rect : (x0, y0, x1, y1)
    image_shape : (H, W)
    cache : InverseMapCache instance
    needs_upsample : if True, upsample data to (roi_h, roi_w) first
    roi_h, roi_w : target ROI dimensions (required if needs_upsample)

    Returns
    -------
    (H, W) array with warped data; NaN outside valid region.
    """
    # Step 1: Upsample if needed (strain data may be downsampled)
    if needs_upsample and roi_h > 0 and roi_w > 0:
        data = upsample_strain_to_roi(data, roi_h, roi_w)

    # Step 2: Get or compute inverse map
    inv_map = cache.get(frame_idx)
    if inv_map is None:
        inv_map = compute_inverse_map(U, V, roi_rect, image_shape)
        inv_map.frame_idx = frame_idx
        cache.put(frame_idx, inv_map)

    # Step 3: Warp data
    return warp_data_inverse(data, inv_map, image_shape)
