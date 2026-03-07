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


def auto_inverse_cache_size() -> int:
    """Compute inverse map cache size. Each entry is ~2x image size in float64."""
    try:
        import psutil
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        # Each inverse map is ~2 * H * W * 8 bytes. For 1024x1024: ~16MB
        # Use 2% of available RAM for inverse maps
        budget_mb = min(100, available_mb * 0.02)
        return max(3, int(budget_mb / 16))
    except (ImportError, Exception):
        return 5


class InverseMapCache:
    """Thread-safe LRU cache for InverseMapResult (keyed by frame index).

    Automatically invalidates when the quality preset changes.
    """

    def __init__(self, max_size: int = None):
        if max_size is None:
            max_size = auto_inverse_cache_size()
        self._max_size = max_size
        self._cache: OrderedDict[int, InverseMapResult] = OrderedDict()
        self._lock = threading.Lock()
        self._quality: str = "balanced"

    def get(self, frame_idx: int, quality: str = "balanced") -> Optional[InverseMapResult]:
        with self._lock:
            if quality != self._quality:
                self._cache.clear()
                self._quality = quality
                return None
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

# Quality presets: name -> K (query grid subsample factor)
# Higher K = fewer Delaunay points + coarser query grid = faster but less detail
WARP_QUALITY_PRESETS = {
    "fine": 3,       # every 3rd pixel — ROI < 500px
    "balanced": 6,   # every 6th pixel — ROI 500–1500px
    "fast": 12,      # every 12th pixel — ROI 1500–3000px
    "draft": 20,     # every 20th pixel — ROI > 3000px
}


def compute_inverse_map(
    U: np.ndarray,
    V: np.ndarray,
    roi_rect: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
    quality: str = "balanced",
) -> InverseMapResult:
    """Compute the inverse mapping from deformed to reference coordinates.

    Uses the forward displacement map to build scattered correspondences
    (ref_pixel -> deformed_pixel), then interpolates the inverse via Delaunay
    triangulation on a coarse grid and upsamples to full resolution.

    The ``quality`` parameter controls the coarse grid spacing (K):
      - "fine"     (K=4):  highest accuracy, ~1s for 500×500
      - "balanced" (K=8):  good balance,     ~0.2s for 500×500
      - "fast"     (K=16): fastest,          ~0.05s for 500×500

    Parameters
    ----------
    U, V : (roi_h, roi_w) displacement fields (may contain NaN)
    roi_rect : (x0, y0, x1, y1) ROI bounding box in full-image coords
    image_shape : (H, W) of the full image
    quality : "fine", "balanced", or "fast"

    Returns
    -------
    InverseMapResult with precomputed reference coordinates for each output pixel.
    """
    from scipy.interpolate import LinearNDInterpolator

    K = WARP_QUALITY_PRESETS.get(quality, 8)

    x0, y0, x1, y1 = roi_rect
    roi_h, roi_w = y1 - y0, x1 - x0
    img_h, img_w = image_shape

    # Auto-cap K so the coarse grid has at least ~30 points per dimension,
    # preventing excessive coarseness on small ROIs.
    K = min(K, max(1, roi_h // 30), max(1, roi_w // 30))

    # --- Step 1: Forward map — subsample reference grid by K ---
    valid_mask = ~(np.isnan(U) | np.isnan(V))
    if not valid_mask.any():
        return InverseMapResult(
            frame_idx=-1,
            ref_row_coords=np.empty((0, 0)),
            ref_col_coords=np.empty((0, 0)),
            validity_mask=np.empty((0, 0), dtype=bool),
            out_x0=x0, out_y0=y0, out_x1=x1, out_y1=y1,
        )

    # Subsample the displacement field on a regular coarse grid.
    # Always include boundary rows/columns so the Delaunay convex hull
    # covers the full ROI extent.
    row_idx = np.unique(np.append(np.arange(0, roi_h, K), roi_h - 1))
    col_idx = np.unique(np.append(np.arange(0, roi_w, K), roi_w - 1))
    row_grid, col_grid = np.meshgrid(row_idx, col_idx, indexing='ij')
    U_coarse = U[np.ix_(row_idx, col_idx)]
    V_coarse = V[np.ix_(row_idx, col_idx)]
    valid_coarse = ~(np.isnan(U_coarse) | np.isnan(V_coarse))

    if not valid_coarse.any():
        return InverseMapResult(
            frame_idx=-1,
            ref_row_coords=np.empty((0, 0)),
            ref_col_coords=np.empty((0, 0)),
            validity_mask=np.empty((0, 0), dtype=bool),
            out_x0=x0, out_y0=y0, out_x1=x1, out_y1=y1,
        )

    ref_rows = row_grid[valid_coarse].astype(np.float64)
    ref_cols = col_grid[valid_coarse].astype(np.float64)
    def_y = ref_rows + y0 + V_coarse[valid_coarse]
    def_x = ref_cols + x0 + U_coarse[valid_coarse]

    # --- Step 2: Deformed bounding box ---
    # Use coarse scatter points (tight bounds prevent NaN border amplification
    # during upsampling — extra margin pixels outside the Delaunay convex hull
    # would become K-wide NaN strips after cv2.resize).
    out_x0 = max(0, int(np.floor(def_x.min())))
    out_y0 = max(0, int(np.floor(def_y.min())))
    out_x1 = min(img_w, int(np.ceil(def_x.max())) + 1)
    out_y1 = min(img_h, int(np.ceil(def_y.max())) + 1)
    out_h = out_y1 - out_y0
    out_w = out_x1 - out_x0

    # --- Step 3: Build Delaunay on coarse scatter points ---
    points = np.column_stack([def_y, def_x])
    interp_row = LinearNDInterpolator(points, ref_rows, fill_value=np.nan)
    interp_col = LinearNDInterpolator(points, ref_cols, fill_value=np.nan)

    # --- Step 4: Query on coarse output grid, then upsample ---
    coarse_h = max(1, (out_h + K - 1) // K)
    coarse_w = max(1, (out_w + K - 1) // K)
    coarse_y = np.linspace(out_y0, out_y1 - 1, coarse_h)
    coarse_x = np.linspace(out_x0, out_x1 - 1, coarse_w)
    cg_y, cg_x = np.meshgrid(coarse_y, coarse_x, indexing='ij')
    query = np.column_stack([cg_y.ravel(), cg_x.ravel()])

    ref_row_coarse = interp_row(query).reshape(coarse_h, coarse_w)
    ref_col_coarse = interp_col(query).reshape(coarse_h, coarse_w)

    # Upsample to full output resolution
    if coarse_h < out_h or coarse_w < out_w:
        # NaN-safe upsampling: interpolate values and validity separately
        valid_coarse_map = np.isfinite(ref_row_coarse) & np.isfinite(ref_col_coarse)
        rc_clean = np.nan_to_num(ref_row_coarse, nan=0.0)
        cc_clean = np.nan_to_num(ref_col_coarse, nan=0.0)
        vc_float = valid_coarse_map.astype(np.float64)

        ref_row_interp = cv2.resize(rc_clean, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        ref_col_interp = cv2.resize(cc_clean, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        valid_up = cv2.resize(vc_float, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

        # Pixels where the upsampled validity is low came from NaN regions
        up_valid = valid_up > 0.5
        ref_row_interp[~up_valid] = np.nan
        ref_col_interp[~up_valid] = np.nan
    else:
        ref_row_interp = ref_row_coarse
        ref_col_interp = ref_col_coarse

    # --- Step 5: Validity mask ---
    in_bounds = (
        np.isfinite(ref_row_interp) & np.isfinite(ref_col_interp) &
        (ref_row_interp >= 0) & (ref_row_interp <= roi_h - 1) &
        (ref_col_interp >= 0) & (ref_col_interp <= roi_w - 1)
    )

    return InverseMapResult(
        frame_idx=-1,  # caller sets this
        ref_row_coords=np.nan_to_num(ref_row_interp, nan=0.0),
        ref_col_coords=np.nan_to_num(ref_col_interp, nan=0.0),
        validity_mask=in_bounds,
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
    quality: str = "balanced",
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
    inv_map = cache.get(frame_idx, quality=quality)
    if inv_map is None:
        inv_map = compute_inverse_map(U, V, roi_rect, image_shape, quality=quality)
        inv_map.frame_idx = frame_idx
        cache.put(frame_idx, inv_map)

    # Step 3: Warp data
    return warp_data_inverse(data, inv_map, image_shape)
