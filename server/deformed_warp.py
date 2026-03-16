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

    # Auto-raise K to cap total Delaunay points on large images.
    # 10K scatter points gives ~0.2 px interpolation error at 1% strain
    # while keeping the Delaunay build + query under 0.5s.
    max_coarse_points = 10000
    total_coarse_est = (roi_h / K) * (roi_w / K)
    if total_coarse_est > max_coarse_points:
        K = max(K, int(np.ceil(np.sqrt(roi_h * roi_w / max_coarse_points))))

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

    # --- Step 2: Forward-mapped contour for precise deformed boundary ---
    # Extract contour of the valid displacement region at full pixel
    # resolution, then forward-map it using the displacement field.
    # This gives pixel-accurate deformed boundaries instead of the coarse
    # Delaunay convex hull which caused staircase artifacts on curved shapes.
    valid_mask_u8 = valid_mask.astype(np.uint8)
    contours, _ = cv2.findContours(
        valid_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    deformed_contours = []
    all_def_x = [def_x]  # include coarse points for bbox fallback
    all_def_y = [def_y]
    for contour in contours:
        pts = contour.reshape(-1, 2)  # (N, 2) as (col, row)
        c_cols, c_rows = pts[:, 0], pts[:, 1]
        u_vals = U[c_rows, c_cols]
        v_vals = V[c_rows, c_cols]
        ok = np.isfinite(u_vals) & np.isfinite(v_vals)
        if ok.sum() < 3:
            continue
        dx = (c_cols[ok] + x0 + u_vals[ok]).astype(np.float64)
        dy = (c_rows[ok] + y0 + v_vals[ok]).astype(np.float64)
        all_def_x.append(dx)
        all_def_y.append(dy)
        deformed_contours.append(
            np.column_stack([dx, dy]).astype(np.int32)
        )

    # --- Step 2b: Deformed bounding box (from full-res contour) ---
    all_dx = np.concatenate(all_def_x)
    all_dy = np.concatenate(all_def_y)
    out_x0 = max(0, int(np.floor(all_dx.min())))
    out_y0 = max(0, int(np.floor(all_dy.min())))
    out_x1 = min(img_w, int(np.ceil(all_dx.max())) + 1)
    out_y1 = min(img_h, int(np.ceil(all_dy.max())) + 1)
    out_h = out_y1 - out_y0
    out_w = out_x1 - out_x0

    # --- Step 3: Build interpolators ---
    # LinearND for accurate interior interpolation + NearestND to
    # extrapolate coordinates beyond the Delaunay convex hull (boundary
    # pixels that are inside the true deformed contour but outside the
    # coarse-grid convex hull).
    from scipy.interpolate import NearestNDInterpolator

    points = np.column_stack([def_y, def_x])
    use_linear = True
    try:
        from scipy.spatial import Delaunay
        tri = Delaunay(points)
        interp_row_lin = LinearNDInterpolator(tri, ref_rows, fill_value=np.nan)
        interp_col_lin = LinearNDInterpolator(tri, ref_cols, fill_value=np.nan)
    except Exception:
        use_linear = False

    interp_row_nn = NearestNDInterpolator(points, ref_rows)
    interp_col_nn = NearestNDInterpolator(points, ref_cols)

    # --- Step 4: Query on coarse output grid, then upsample ---
    MAX_QUERY_PER_DIM = 50
    query_step_h = max(K, out_h // MAX_QUERY_PER_DIM) if out_h > MAX_QUERY_PER_DIM else K
    query_step_w = max(K, out_w // MAX_QUERY_PER_DIM) if out_w > MAX_QUERY_PER_DIM else K
    coarse_h = max(1, (out_h + query_step_h - 1) // query_step_h)
    coarse_w = max(1, (out_w + query_step_w - 1) // query_step_w)
    coarse_y = np.linspace(out_y0, out_y1 - 1, coarse_h)
    coarse_x = np.linspace(out_x0, out_x1 - 1, coarse_w)
    cg_y, cg_x = np.meshgrid(coarse_y, coarse_x, indexing='ij')
    query = np.column_stack([cg_y.ravel(), cg_x.ravel()])

    if use_linear:
        ref_row_coarse = interp_row_lin(query).reshape(coarse_h, coarse_w)
        ref_col_coarse = interp_col_lin(query).reshape(coarse_h, coarse_w)
        # Fill Delaunay-hull-exterior NaN with nearest-neighbor extrapolation
        nan_mask = np.isnan(ref_row_coarse)
        if nan_mask.any():
            nan_q = query[nan_mask.ravel()]
            ref_row_coarse[nan_mask] = interp_row_nn(nan_q)
            ref_col_coarse[nan_mask] = interp_col_nn(nan_q)
    else:
        ref_row_coarse = interp_row_nn(query).reshape(coarse_h, coarse_w)
        ref_col_coarse = interp_col_nn(query).reshape(coarse_h, coarse_w)

    # Upsample to full output resolution (all coarse cells now have values)
    if coarse_h < out_h or coarse_w < out_w:
        ref_row_interp = cv2.resize(
            ref_row_coarse.astype(np.float64), (out_w, out_h),
            interpolation=cv2.INTER_LINEAR,
        )
        ref_col_interp = cv2.resize(
            ref_col_coarse.astype(np.float64), (out_w, out_h),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        ref_row_interp = ref_row_coarse
        ref_col_interp = ref_col_coarse

    # --- Step 5: Forward-mapped contour validity mask ---
    # Use the pixel-accurate deformed contour instead of the coarse
    # Delaunay hull for the validity boundary.
    if deformed_contours:
        contour_mask = np.zeros((out_h, out_w), dtype=np.uint8)
        shifted = [dc - np.array([[out_x0, out_y0]]) for dc in deformed_contours]
        cv2.fillPoly(contour_mask, shifted, 1)
        validity = contour_mask.astype(bool)
    else:
        # Fallback: coordinate bounds check
        validity = (
            (ref_row_interp >= 0) & (ref_row_interp <= roi_h - 1) &
            (ref_col_interp >= 0) & (ref_col_interp <= roi_w - 1)
        )

    return InverseMapResult(
        frame_idx=-1,  # caller sets this
        ref_row_coords=ref_row_interp,
        ref_col_coords=ref_col_interp,
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

def _nan_safe_resize(arr: np.ndarray, target_wh: Tuple[int, int]) -> np.ndarray:
    """NaN-safe downsample a 2D array using weighted averaging.

    Uses INTER_AREA for proper block averaging; normalizes by the valid-pixel
    weight so NaN regions don't bias neighboring averages toward zero.
    """
    mask = np.isfinite(arr).astype(np.float32)
    clean = np.nan_to_num(arr, nan=0.0).astype(np.float32)
    data_r = cv2.resize(clean, target_wh, interpolation=cv2.INTER_AREA)
    mask_r = cv2.resize(mask, target_wh, interpolation=cv2.INTER_AREA)
    valid = mask_r > 0.5
    result = np.full(data_r.shape, np.nan, dtype=np.float64)
    result[valid] = data_r[valid].astype(np.float64) / mask_r[valid]
    return result


def _warp_at_viewport_scale(
    data: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    roi_rect: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
    scale: float,
    needs_upsample: bool = False,
    roi_h: int = 0,
    roi_w: int = 0,
    quality: str = "balanced",
) -> np.ndarray:
    """Fast path: downsample everything to viewport resolution before warping.

    For a 5000×4000 image viewed in a 1200×800 viewport (scale ≈ 0.2), this
    reduces Delaunay input from ~550K to ~22K points and map_coordinates from
    20M to ~800K pixels — typically 10–50× faster than full-resolution warping.

    The displacement values U, V are scaled by ``scale`` because they represent
    pixel-unit offsets: a 10 px displacement in the original equals 2 px in a
    0.2× downsampled coordinate system.
    """
    img_h, img_w = image_shape
    ds_w = max(1, round(img_w * scale))
    ds_h = max(1, round(img_h * scale))

    x0, y0, x1, y1 = roi_rect
    ds_x0, ds_y0 = round(x0 * scale), round(y0 * scale)
    ds_x1 = max(ds_x0 + 2, round(x1 * scale))
    ds_y1 = max(ds_y0 + 2, round(y1 * scale))
    ds_roi_w, ds_roi_h = ds_x1 - ds_x0, ds_y1 - ds_y0

    # Upsample strain data to ROI size first (if needed), then downsample
    if needs_upsample and roi_h > 0 and roi_w > 0:
        data = upsample_strain_to_roi(data, roi_h, roi_w)

    # Downsample data, U, V to viewport-proportional ROI.
    # Fast path when no NaN (common for DIC displacement fields).
    has_nan = np.isnan(data).any() or np.isnan(U).any() or np.isnan(V).any()

    if not has_nan:
        # No NaN: direct resize (avoids expensive mask computation on large arrays)
        data_ds = cv2.resize(
            data.astype(np.float32), (ds_roi_w, ds_roi_h), interpolation=cv2.INTER_AREA,
        ).astype(np.float64)
        U_ds = cv2.resize(
            U.astype(np.float32), (ds_roi_w, ds_roi_h), interpolation=cv2.INTER_AREA,
        ).astype(np.float64) * scale
        V_ds = cv2.resize(
            V.astype(np.float32), (ds_roi_w, ds_roi_h), interpolation=cv2.INTER_AREA,
        ).astype(np.float64) * scale
    else:
        # NaN-safe: weighted average with mask normalization
        combined_mask = (
            np.isfinite(data) & np.isfinite(U) & np.isfinite(V)
        ).astype(np.float32)
        data_f = np.where(combined_mask > 0, data, 0.0).astype(np.float32)
        U_f = np.where(combined_mask > 0, U, 0.0).astype(np.float32)
        V_f = np.where(combined_mask > 0, V, 0.0).astype(np.float32)

        mask_ds = cv2.resize(combined_mask, (ds_roi_w, ds_roi_h), interpolation=cv2.INTER_AREA)
        data_r = cv2.resize(data_f, (ds_roi_w, ds_roi_h), interpolation=cv2.INTER_AREA)
        U_r = cv2.resize(U_f, (ds_roi_w, ds_roi_h), interpolation=cv2.INTER_AREA)
        V_r = cv2.resize(V_f, (ds_roi_w, ds_roi_h), interpolation=cv2.INTER_AREA)

        valid = mask_ds > 0.5
        inv_w = np.where(valid, 1.0 / mask_ds, 0.0)
        data_ds = np.where(valid, data_r * inv_w, np.nan).astype(np.float64)
        U_ds = np.where(valid, U_r * inv_w * scale, np.nan).astype(np.float64)
        V_ds = np.where(valid, V_r * inv_w * scale, np.nan).astype(np.float64)

    # Auto-select coarser quality for viewport rendering — the output is
    # already low-resolution so fine Delaunay detail is wasted.
    ds_max_dim = max(ds_roi_h, ds_roi_w)
    if ds_max_dim < 500:
        vp_quality = "draft"     # K=20 → ~625 Delaunay points
    elif ds_max_dim < 1500:
        vp_quality = "fast"      # K=12 → ~5K Delaunay points
    else:
        vp_quality = quality

    # Compute inverse map at viewport resolution (no caching — it's fast)
    ds_roi_rect = (ds_x0, ds_y0, ds_x1, ds_y1)
    inv_map = compute_inverse_map(U_ds, V_ds, ds_roi_rect, (ds_h, ds_w), vp_quality)

    return warp_data_inverse(data_ds, inv_map, (ds_h, ds_w))


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
    vw: int = 0,
    vh: int = 0,
) -> np.ndarray:
    """Convenience function for render endpoints: upsample if needed, compute
    or retrieve cached inverse map, then warp data to deformed coordinates.

    When ``vw``/``vh`` indicate a viewport smaller than the full image, all
    inputs are downsampled to viewport resolution before warping, which is
    dramatically faster for large images (e.g. 5000×4000 → 1200×800).

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
    vw, vh : viewport dimensions (0 = full resolution)

    Returns
    -------
    (out_H, out_W) array with warped data; NaN outside valid region.
    When viewport downsampling is active, out_H/out_W are viewport-proportional
    (smaller than H/W), which is what render_overlay_png expects.
    """
    img_h, img_w = image_shape
    scale = min(vw / img_w, vh / img_h, 1.0) if vw > 0 and vh > 0 else 1.0

    # Fast path: warp at viewport resolution
    if scale < 1.0:
        return _warp_at_viewport_scale(
            data, U, V, roi_rect, image_shape, scale,
            needs_upsample, roi_h, roi_w, quality,
        )

    # Full-resolution path (for downloads and when viewport >= image)
    if needs_upsample and roi_h > 0 and roi_w > 0:
        data = upsample_strain_to_roi(data, roi_h, roi_w)

    inv_map = cache.get(frame_idx, quality=quality)
    if inv_map is None:
        inv_map = compute_inverse_map(U, V, roi_rect, image_shape, quality=quality)
        inv_map.frame_idx = frame_idx
        cache.put(frame_idx, inv_map)

    return warp_data_inverse(data, inv_map, image_shape)
