"""
Processing helpers for RAFT-DIC
- Robust image loading/normalization
- Windowed and full-image processing
- Smoothing and result saving
"""

import os
import time
import numpy as np
import cv2
import tifffile
import scipy.io as sio
import io
from PIL import Image
from scipy.interpolate import griddata, Rbf
from scipy.ndimage import gaussian_filter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as collections
import matplotlib.cm as cm

from .model import inference


def load_and_convert_image(img_path: str) -> np.ndarray:
    """Load image and convert to 8-bit RGB with sensible handling for TIFF/float/uint16.

    Robustness improvements:
    - If a file has .tif/.tiff extension but is actually PNG (or another format),
      gracefully fall back to OpenCV instead of raising a hard error from tifffile.
    """
    ext = os.path.splitext(img_path)[1].lower()

    frame = None
    if ext in ['.tif', '.tiff']:
        try:
            frame = tifffile.imread(img_path)
        except Exception as e:
            # Fallback if mislabeled TIFF or unsupported TIFF variant; try generic reader
            print(f"Warning: TIFF reader failed for '{img_path}' ({e}); falling back to cv2.")
            frame = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    else:
        frame = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    if frame is None:
        raise Exception(f"Failed to load image from {img_path}")

    if frame.dtype == np.float32:
        min_val = np.nanmin(frame)
        max_val = np.nanmax(frame)
        if max_val != min_val:
            frame = np.clip((frame - min_val) / (max_val - min_val), 0, 1)
        else:
            frame = np.zeros_like(frame)
        frame = (frame * 255).astype(np.uint8)
    elif frame.dtype == np.uint16:
        frame = (frame / 256).astype(np.uint8)
    elif frame.dtype != np.uint8:
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif len(frame.shape) == 3 and frame.shape[2] > 3:
        frame = frame[:, :, :3]

    if ext in ['.tif', '.tiff'] and len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb


def calculate_window_positions(image_size: int, crop_size: int, shift: int):
    positions = []
    current = 0
    while current <= image_size - crop_size:
        positions.append(current)
        current += shift
    if current < image_size - crop_size:
        positions.append(image_size - crop_size)
    elif positions and positions[-1] + crop_size < image_size:
        positions.append(image_size - crop_size)
    return positions


def smooth_displacement_field(displacement_field: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    smoothed = np.zeros_like(displacement_field)
    for i in range(2):
        comp = displacement_field[..., i]
        valid_mask = ~np.isnan(comp)
        if not np.any(valid_mask):
            smoothed[..., i] = comp
            continue
        filled = np.where(valid_mask, comp, 0)
        smoothed_data = gaussian_filter(filled, sigma)
        weight = gaussian_filter(valid_mask.astype(float), sigma)
        with np.errstate(divide='ignore', invalid='ignore'):
            smoothed[..., i] = np.where(weight > 0.01, smoothed_data / weight, np.nan)
    return smoothed


def cut_image_pair_with_flow(ref_img: np.ndarray, def_img: np.ndarray, project_root: str, model, device: str,
                             crop_size=(128, 128), shift=64, plot_windows=False,
                             roi_mask=None, use_smooth=True, sigma=2.0):
    start_total = time.time()

    windows_dir = os.path.join(project_root, "windows")
    os.makedirs(windows_dir, exist_ok=True)

    H, W, _ = ref_img.shape
    crop_h, crop_w = crop_size

    x_positions = calculate_window_positions(W, crop_w, shift)
    y_positions = calculate_window_positions(H, crop_h, shift)

    windows = []
    global_flow = np.zeros((H, W, 2), dtype=np.float64)
    count_field = np.zeros((H, W), dtype=np.float64)

    if roi_mask is not None:
        roi_mask = roi_mask.astype(bool)
    else:
        roi_mask = np.ones((H, W), dtype=bool)

    # Smooth fusion weights: raised-cosine (Hanning) window to taper borders
    wy = np.hanning(max(crop_h, 2))  # ensure length >= 2
    wx = np.hanning(max(crop_w, 2))
    weight2d = np.outer(wy, wx).astype(np.float64)
    # Avoid extremely small weights causing numerical issues
    weight2d = np.clip(weight2d, 1e-6, None)

    count = 0
    inference_time = 0
    for y in y_positions:
        for x in x_positions:
            ref_window = ref_img[y:y+crop_h, x:x+crop_w, :]
            def_window = def_img[y:y+crop_h, x:x+crop_w, :]

            windows.append({'index': count, 'position': (x, y, x+crop_w, y+crop_h)})

            t0 = time.time()
            flow_low, flow_up = inference(model, ref_window, def_window, device, test_mode=True)
            flow_up = flow_up.squeeze(0)
            inference_time += time.time() - t0

            u = flow_up[0].cpu().numpy()
            v = flow_up[1].cpu().numpy()
            window_flow = np.stack((u, v), axis=-1)

            # Local ROI weighting
            if roi_mask is not None:
                local_roi = roi_mask[y:y+crop_h, x:x+crop_w].astype(np.float64)
                local_weight = weight2d * local_roi
            else:
                local_weight = weight2d
            global_flow[y:y+crop_h, x:x+crop_w, :] += window_flow * local_weight[..., None]
            count_field[y:y+crop_h, x:x+crop_w] += local_weight

            count += 1

    final_flow = np.where(count_field[..., None] > 0,
                          global_flow / count_field[..., None],
                          np.nan)
    final_flow[~roi_mask] = np.nan

    displacement_field = smooth_displacement_field(final_flow, sigma=sigma) if use_smooth else final_flow

    if plot_windows:
        try:
            ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)
            fig, ax = plt.subplots(1, figsize=(8, 8))
            ax.imshow(ref_gray, cmap='gray')
            ax.axis('off')
            colormap = cm.get_cmap('hsv', len(windows))
            patches_list = []
            for w in windows:
                x0, y0, x1, y1 = w['position']
                rect = patches.Rectangle((x0, y0), x1-x0, y1-y0, linewidth=1,
                                         edgecolor=colormap(w['index']), facecolor='none')
                patches_list.append(rect)
            ax.add_collection(collections.PatchCollection(patches_list, match_original=True))
            plt.title("Sliding Windows Layout")
            plt.savefig(os.path.join(windows_dir, "windows_layout.png"), bbox_inches='tight', dpi=300)
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Failed to save windows layout: {str(e)}")

    total_time = time.time() - start_total
    _emit_status("\nTime statistics:")
    _emit_status(f"Total processing time: {total_time:.2f} seconds")
    _emit_status(f"RAFT inference time: {inference_time:.2f} seconds")
    if total_time > 0:
        _emit_status(f"RAFT inference percentage: {(inference_time/total_time*100):.1f}%")
        _emit_status(f"Other operations time: {(total_time-inference_time):.2f} seconds")

    return displacement_field, windows


def process_image_pair(ref_img: np.ndarray, def_img: np.ndarray, project_root: str, model, device: str,
                       roi_mask=None, use_smooth: bool = True, sigma: float = 2.0, iters: int = 12):
    start_total = time.time()
    H, W, _ = ref_img.shape
    if roi_mask is not None:
        roi_mask = roi_mask.astype(bool)
    else:
        roi_mask = np.ones((H, W), dtype=bool)

    t0 = time.time()
    try:
        flow_low, flow_up = inference(model, ref_img, def_img, device, test_mode=True, iters=iters)
    except torch.cuda.OutOfMemoryError:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        # Retry with fewer iterations to reduce memory footprint
        flow_low, flow_up = inference(model, ref_img, def_img, device, test_mode=True, iters=max(4, iters // 2))
    flow_up = flow_up.squeeze(0)
    inference_time = time.time() - t0

    u = flow_up[0].cpu().numpy()
    v = flow_up[1].cpu().numpy()
    displacement_field = np.stack((u, v), axis=-1)
    displacement_field[~roi_mask] = np.nan

    if use_smooth:
        displacement_field = smooth_displacement_field(displacement_field, sigma=sigma)

    total_time = time.time() - start_total
    _emit_status("\nTime statistics:")
    _emit_status(f"Total processing time: {total_time:.2f} seconds")
    _emit_status(f"RAFT inference time: {inference_time:.2f} seconds")
    if total_time > 0:
        _emit_status(f"RAFT inference percentage: {(inference_time/total_time*100):.1f}%")
        _emit_status(f"Other operations time: {(total_time-inference_time):.2f} seconds")

    return displacement_field, None


# ------------------------- New tiled ROI processing -------------------------
import torch

_status_logger = None

def set_status_logger(callback):
    global _status_logger
    _status_logger = callback

def _emit_status(message: str):
    print(message, flush=True)
    if _status_logger:
        try:
            _status_logger(message)
        except Exception:
            pass



_status_logger = None

def set_status_logger(callback):
    global _status_logger
    _status_logger = callback

def _emit_status(message: str):
    print(message, flush=True)
    if _status_logger:
        try:
            _status_logger(message)
        except Exception:
            pass



def _compute_min_bounding_box(roi_mask_full: np.ndarray):
    ys, xs = np.where(roi_mask_full)
    if xs.size == 0 or ys.size == 0:
        return None
    x0 = int(xs.min())
    y0 = int(ys.min())
    m = int(xs.max() - x0 + 1)
    n = int(ys.max() - y0 + 1)
    return x0, y0, m, n


def _expand_bbox(bbox, pad, width, height):
    x0, y0, m, n = bbox
    xC = max(0, x0 - pad)
    yC = max(0, y0 - pad)
    xR = min(width, x0 + m + pad)
    yB = min(height, y0 + n + pad)
    wC = int(xR - xC)
    hC = int(yB - yC)
    return xC, yC, wC, hC


def _generate_tile_starts(total_len, tile_len, overlap):
    """Generate start positions for sliding window with fixed overlap."""
    if total_len <= tile_len:
        return [0]
    
    stride = tile_len - overlap
    if stride <= 0:
        stride = 1 # Fallback to avoid infinite loop
        
    starts = []
    pos = 0
    while pos + tile_len < total_len:
        starts.append(pos)
        pos += stride
        
    # Ensure the last tile covers the end
    if starts[-1] + tile_len < total_len:
        starts.append(total_len - tile_len)
        
    # If the last step was too small (huge overlap), we might want to merge?
    # But for simplicity, just keeping it is fine, fusion handles it.
    return sorted(list(set(starts)))


def dic_over_roi_with_tiling(ref_img: np.ndarray,
                             def_img: np.ndarray,
                             roi_mask_full: np.ndarray,
                             model,
                             device: str,
                             context_padding: int = 32,
                             tile_overlap: int = 32,
                             p_max_pixels: int = 1100*1100,
                             use_smooth: bool = True,
                             sigma: float = 2.0,
                             iters: int = 12):
    """
    Perform DIC over the ROI using an adaptive tiling strategy with weighted fusion.
    
    Args:
        context_padding: Extra pixels around the ROI bounding box to include (default 32).
        tile_overlap: Overlap between tiles in pixels (default 32).
        p_max_pixels: Maximum pixels allowed per inference pass (VRAM limit).
    """
    H, W, _ = ref_img.shape
    start_total = time.time()
    if roi_mask_full is None or roi_mask_full.shape[:2] != (H, W):
        raise ValueError("roi_mask_full must be provided and match image size")

    bbox = _compute_min_bounding_box(roi_mask_full)
    if bbox is None:
        raise ValueError("ROI mask is empty")

    # 1. Define Work Area (ROI + Padding)
    xC, yC, wC, hC = _expand_bbox(bbox, int(context_padding), W, H)

    # 2. Prepare Accumulators
    accum = np.zeros((hC, wC, 2), dtype=np.float64)
    wsum = np.zeros((hC, wC), dtype=np.float64)
    
    # ROI mask cropped to Work Area
    roiC = roi_mask_full[yC:yC+hC, xC:xC+wC]

    # 3. Determine Tiling Strategy
    # Max safe square tile dimension
    max_T = int(np.sqrt(p_max_pixels))
    
    # Check if single shot is possible
    if wC * hC <= p_max_pixels:
        tiles = [(0, 0, wC, hC)]
        is_tiled = False
    else:
        is_tiled = True
        # Ensure tile size is at least larger than overlap
        T = max(max_T, tile_overlap * 2 + 16)
        # But must not exceed budget (if possible, otherwise we clamp T to max_T)
        T = min(T, max_T)
        
        # If T is too small relative to overlap, reduce overlap
        if T <= tile_overlap:
             tile_overlap = T // 4
        
        starts_x = _generate_tile_starts(wC, T, int(tile_overlap))
        starts_y = _generate_tile_starts(hC, T, int(tile_overlap))
        
        tiles = []
        for sy in starts_y:
            for sx in starts_x:
                # Clip tile size if at edges (though _generate_tile_starts usually handles this by shifting start)
                # But here we use fixed size T unless total size is smaller
                tw = min(T, wC - sx)
                th = min(T, hC - sy)
                tiles.append((sx, sy, tw, th))

    # 4. Execute Inference
    inference_time = 0.0
    
    # Pre-compute Hanning window cache
    window_cache = {}

    for (sx, sy, tw, th) in tiles:
        x0, y0 = xC + sx, yC + sy
        x1, y1 = x0 + tw, y0 + th
        
        ref_tile = ref_img[y0:y1, x0:x1, :]
        def_tile = def_img[y0:y1, x0:x1, :]
        
        t0 = time.time()
        try:
            _, flow_up = inference(model, ref_tile, def_tile, device, test_mode=True, iters=iters)
        except torch.cuda.OutOfMemoryError:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            # Simple retry with fewer iterations
            _, flow_up = inference(model, ref_tile, def_tile, device, test_mode=True, iters=max(4, iters // 2))
        
        flow_up = flow_up.squeeze(0)
        inference_time += (time.time() - t0)
        
        u = flow_up[0].cpu().numpy()
        v = flow_up[1].cpu().numpy()
        flow = np.stack((u, v), axis=-1)
        
        # 5. Weighted Fusion
        # Generate or retrieve window
        k = (tw, th)
        if k not in window_cache:
            # Hanning window for smooth blending
            # If single shot, window is all 1s (or we can still use Hanning to suppress edge artifacts if we wanted, but usually 1s)
            if not is_tiled:
                window_cache[k] = np.ones((th, tw), dtype=np.float64)
            else:
                wy = np.hanning(max(th, 4))
                wx = np.hanning(max(tw, 4))
                window_cache[k] = np.outer(wy, wx).astype(np.float64)
        
        weight = window_cache[k]
        
        accum[sy:sy+th, sx:sx+tw, :] += flow * weight[..., None]
        wsum[sy:sy+th, sx:sx+tw] += weight

    # 6. Normalize
    with np.errstate(divide='ignore', invalid='ignore'):
        result_C = np.where(wsum[..., None] > 0, accum / wsum[..., None], np.nan)

    # 7. Post-processing (Smoothing & Masking)
    # Mask out non-ROI areas in the work rectangle
    invalid_mask = ~roiC.astype(bool)
    result_C[invalid_mask] = np.nan
    
    if use_smooth:
        result_C = smooth_displacement_field(result_C, sigma=sigma)

    # 8. Paste back to full image
    disp_full = np.full((H, W, 2), np.nan, dtype=np.float64)
    disp_full[yC:yC+hC, xC:xC+wC, :] = result_C

    total_time = time.time() - start_total
    _emit_status("\nTime statistics:")
    _emit_status(f"Total processing time: {total_time:.2f} seconds")
    _emit_status(f"RAFT inference time: {inference_time:.2f} seconds")
    if total_time > 0:
        _emit_status(f"RAFT inference percentage: {(inference_time/total_time*100):.1f}%")
        _emit_status(f"Other operations time: {(total_time-inference_time):.2f} seconds")

    return disp_full, (xC, yC, wC, hC)


def save_displacement_results(displacement_field: np.ndarray, output_dir: str, index: int,
                               roi_rect=None, roi_mask=None):
    npy_dir = os.path.join(output_dir, "displacement_results_temp_npy")
    os.makedirs(npy_dir, exist_ok=True)

    npy_file = os.path.join(npy_dir, f"displacement_field_{index}.npy")
    np.save(npy_file, displacement_field)


def save_displacement_sequence(displacements, output_dir: str, roi_rect=None, roi_mask=None,
                               save_numpy: bool = True, save_matlab: bool = True,
                               numpy_filename: str = "displacement_sequence.npz",
                               matlab_filename: str = "displacement_sequence.mat"):
    """
    Save an entire sequence of displacement fields into single Python (.npz) and/or MATLAB (.mat) files.

    Args:
        displacements: list of np.ndarray, each H x W x 2 (u,v), NaNs for invalid
        output_dir: destination directory (typically project root)
        roi_rect: (xmin, ymin, xmax, ymax) used to compute reference/current coordinates
        roi_mask: full-image boolean mask; cropped based on roi_rect
        save_numpy: whether to write consolidated .npz
        save_matlab: whether to write consolidated .mat with cell arrays
        numpy_filename: filename for npz
        matlab_filename: filename for mat
    """
    if not displacements:
        return

    os.makedirs(output_dir, exist_ok=True)

    h, w = displacements[0].shape[:2]

    # Compute reference grid and cropped mask once
    if roi_rect is not None:
        xmin, ymin, xmax, ymax = roi_rect
        y_coords, x_coords = np.mgrid[ymin:ymin+h, xmin:xmin+w]
    else:
        y_coords, x_coords = np.mgrid[0:h, 0:w]

    X_ref = x_coords.astype(np.float64)
    Y_ref = y_coords.astype(np.float64)

    roi_mask_crop = None
    if roi_mask is not None:
        if roi_rect is not None:
            xmin, ymin, xmax, ymax = roi_rect
            roi_mask_crop = roi_mask[ymin:ymax, xmin:xmax]
        else:
            roi_mask_crop = roi_mask

    # Prepare Python-friendly consolidated NPZ (stack frames)
    if save_numpy:
        # Stack displacements into (T, H, W, 2)
        disp_stack = np.stack(displacements, axis=0)
        out_npz = os.path.join(output_dir, numpy_filename)
        np.savez_compressed(
            out_npz,
            displacements=disp_stack,
            X_ref=X_ref,
            Y_ref=Y_ref,
            roi_rect=np.array(roi_rect if roi_rect is not None else [0, 0, w, h], dtype=np.int64),
            roi_mask=roi_mask_crop if roi_mask_crop is not None else np.ones((h, w), dtype=bool)
        )

    # Prepare MATLAB-friendly consolidated MAT with cell arrays
    if save_matlab:
        T = len(displacements)
        U_cells = np.empty((1, T), dtype=object)
        V_cells = np.empty((1, T), dtype=object)
        Xref_cells = np.empty((1, T), dtype=object)
        Yref_cells = np.empty((1, T), dtype=object)
        Xcur_cells = np.empty((1, T), dtype=object)
        Ycur_cells = np.empty((1, T), dtype=object)

        for i, disp in enumerate(displacements):
            u = disp[:, :, 0]
            v = disp[:, :, 1]

            Xc = X_ref + u
            Yc = Y_ref + v

            if roi_mask_crop is not None:
                invalid = ~roi_mask_crop
                Xr_i = X_ref.copy()
                Yr_i = Y_ref.copy()
                Xc_i = Xc.copy()
                Yc_i = Yc.copy()
                Xr_i[invalid] = np.nan
                Yr_i[invalid] = np.nan
                Xc_i[invalid] = np.nan
                Yc_i[invalid] = np.nan
            else:
                Xr_i, Yr_i, Xc_i, Yc_i = X_ref, Y_ref, Xc, Yc

            U_cells[0, i] = u
            V_cells[0, i] = v
            Xref_cells[0, i] = Xr_i
            Yref_cells[0, i] = Yr_i
            Xcur_cells[0, i] = Xc_i
            Ycur_cells[0, i] = Yc_i

        out_mat = os.path.join(output_dir, matlab_filename)
        sio.savemat(out_mat, {
            'U': U_cells,
            'V': V_cells,
            'X_ref': Xref_cells,
            'Y_ref': Yref_cells,
            'X_current': Xcur_cells,
            'Y_current': Ycur_cells,
        })


def fig2img(fig):
    """Convert matplotlib image to PIL Image"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img = Image.open(buf)
    # Set uniform display size
    display_size = (400, 400)
    img = img.resize(display_size, Image.LANCZOS)
    plt.close(fig)
    return img


def create_reference_displacement_visualization(displacement, background_image, alpha, current_colormap, vmin_u, vmax_u, vmin_v, vmax_v, roi_rect):
    """Create displacement field visualization on reference image background"""
    xmin, ymin, xmax, ymax = roi_rect
    
    fig, (ax_u, ax_v) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=False)
    
    u = displacement[:, :, 0]
    v = displacement[:, :, 1]
    
    h, w = background_image.shape[:2]
    u_full = np.full((h, w), np.nan)
    v_full = np.full((h, w), np.nan)
    
    u_full[ymin:ymax, xmin:xmax] = u
    v_full[ymin:ymax, xmin:xmax] = v
    
    ax_u.imshow(background_image, cmap='gray')
    mask_u = ~np.isnan(u_full)
    u_masked = np.ma.array(u_full, mask=~mask_u)
    
    im_u = ax_u.imshow(u_masked, cmap=current_colormap, alpha=alpha * mask_u, vmin=vmin_u, vmax=vmax_u)
    ax_u.set_title("U Component on Reference Image")
    fig.colorbar(im_u, ax=ax_u, fraction=0.046, pad=0.04)
    
    ax_v.imshow(background_image, cmap='gray')
    mask_v = ~np.isnan(v_full)
    v_masked = np.ma.array(v_full, mask=~mask_v)
    
    im_v = ax_v.imshow(v_masked, cmap=current_colormap, alpha=alpha * mask_v, vmin=vmin_v, vmax=vmax_v)
    ax_v.set_title("V Component on Reference Image")
    fig.colorbar(im_v, ax=ax_v, fraction=0.046, pad=0.04)
    
    ax_u.set_axis_off()
    ax_v.set_axis_off()
    ax_u.set_aspect('equal')
    ax_v.set_aspect('equal')
    
    return fig


def _create_deformed_displacement_image_fast(img_crop, u_full, v_full, deformed_mask, alpha, current_colormap, vmin_u, vmax_u, vmin_v, vmax_v):
    import matplotlib.colors as mcolors
    bg = img_crop
    if bg.ndim == 2:
        bg_rgb = np.stack([bg, bg, bg], axis=-1)
    else:
        bg_rgb = bg[:, :, :3]

    def blend_component(comp, vmin, vmax):
        norm = (comp - vmin) / max(1e-8, (vmax - vmin))
        norm = np.clip(norm, 0.0, 1.0)
        cmap = cm.get_cmap(current_colormap)
        rgba = cmap(norm)  # float 0-1
        rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
        out = bg_rgb.copy()
        m = deformed_mask
        out[m] = (out[m].astype(np.float32) * (1.0 - alpha) + rgb[m].astype(np.float32) * alpha).astype(np.uint8)
        return out

    img_u = blend_component(u_full, vmin_u, vmax_u)
    img_v = blend_component(v_full, vmin_v, vmax_v)
    combined = np.concatenate([img_u, img_v], axis=1)
    return Image.fromarray(combined)


def prepare_visualization_data(displacement, reference_image, deformed_image, roi_rect, roi_mask,
                               preview_scale, interp_sample_step, background_mode,
                               deform_display_mode, deform_interp, show_deformed_boundary, quiver_step,
                               deformed_view_cache=None, frame_idx=None):
    """Prepare data for visualization without creating a figure"""
    print(f"[DEBUG] prepare_visualization_data: Mode={background_mode}, ROI={roi_rect}")
    data = {
        'background_mode': background_mode,
        'img_crop': None,
        'u_masked': None,
        'v_masked': None,
        'boundary_points': None,
        'quiver_data': None,
        'error': None
    }

    if background_mode == 'reference':
        # Reference mode logic
        try:
            xmin, ymin, xmax, ymax = roi_rect
            print(f"[DEBUG] Reference mode ROI: {xmin},{ymin},{xmax},{ymax}")
            u = displacement[:, :, 0]
            v = displacement[:, :, 1]
            h, w = reference_image.shape[:2]
            u_full = np.full((h, w), np.nan)
            v_full = np.full((h, w), np.nan)
            u_full[ymin:ymax, xmin:xmax] = u
            v_full[ymin:ymax, xmin:xmax] = v
            
            data['img_crop'] = reference_image
            data['u_masked'] = np.ma.array(u_full, mask=np.isnan(u_full))
            data['v_masked'] = np.ma.array(v_full, mask=np.isnan(v_full))
            print(f"[DEBUG] Reference mode prepared. Masked U valid count: {data['u_masked'].count()}")
            return data
        except Exception as e:
            print(f"[DEBUG] Error in reference mode: {e}")
            data['error'] = str(e)
            return data

    # Deformed mode: Show displacement WARPED to deformed frame coordinates
    # Uses forward scatter: for each reference pixel, place its value at the deformed position
    try:
        h, w = deformed_image.shape[:2]
        
        # Check cache first
        if deformed_view_cache is not None and frame_idx is not None:
            cached = deformed_view_cache.get_warped_displacement(frame_idx)
            if cached is not None:
                print(f"[DeformedView] Using cached warped displacement for frame {frame_idx}")
                data['img_crop'] = deformed_image
                data['u_masked'] = np.ma.array(cached['u_warped'], mask=np.isnan(cached['u_warped']))
                data['v_masked'] = np.ma.array(cached['v_warped'], mask=np.isnan(cached['v_warped']))
                return data
        
        print(f"[DeformedView] Computing inverse warp for frame {frame_idx}...")
        xmin, ymin, xmax, ymax = roi_rect
        u_crop = displacement[:, :, 0]
        v_crop = displacement[:, :, 1]
        roi_h, roi_w = u_crop.shape
        
        print(f"[DeformedView] ROI: ({xmin},{ymin}) to ({xmax},{ymax}), shape=({roi_h},{roi_w})")
        
        # Create coordinate grids for the ROI region
        y_ref, x_ref = np.mgrid[ymin:ymax, xmin:xmax]
        
        # Compute deformed coordinates for each pixel in ROI
        # x_def = x_ref + u, y_def = y_ref + v
        x_def = (x_ref + u_crop).astype(np.float32)
        y_def = (y_ref + v_crop).astype(np.float32)
        
        # Create output arrays (full image size) - initialize to 0 for accumulation
        # (np.add.at cannot accumulate on NaN values)
        u_warped = np.zeros((h, w), dtype=np.float32)
        v_warped = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)
        
        # Valid mask: not NaN
        valid_mask = ~np.isnan(u_crop) & ~np.isnan(v_crop)
        
        # Get valid positions
        x_def_valid = x_def[valid_mask]
        y_def_valid = y_def[valid_mask]
        u_valid = u_crop[valid_mask]
        v_valid = v_crop[valid_mask]
        
        print(f"[DeformedView] Valid pixels: {len(u_valid)}")
        
        # Round to nearest integer coordinates
        x_idx = np.round(x_def_valid).astype(np.int32)
        y_idx = np.round(y_def_valid).astype(np.int32)
        
        # Filter to within bounds
        in_bounds = (x_idx >= 0) & (x_idx < w) & (y_idx >= 0) & (y_idx < h)
        x_idx = x_idx[in_bounds]
        y_idx = y_idx[in_bounds]
        u_valid = u_valid[in_bounds]
        v_valid = v_valid[in_bounds]
        
        print(f"[DeformedView] In bounds: {len(u_valid)}")
        
        # Forward scatter: place values at deformed positions
        # Use accumulation for averaging when multiple points land on same pixel
        np.add.at(u_warped, (y_idx, x_idx), u_valid)
        np.add.at(v_warped, (y_idx, x_idx), v_valid)
        np.add.at(count_map, (y_idx, x_idx), 1.0)
        
        # Average where multiple values accumulated
        nonzero = count_map > 0
        u_warped[nonzero] /= count_map[nonzero]
        v_warped[nonzero] /= count_map[nonzero]
        u_warped[~nonzero] = np.nan
        v_warped[~nonzero] = np.nan
        
        # Close small holes using morphological dilation on the mask
        valid_warped = ~np.isnan(u_warped)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(valid_warped.astype(np.uint8), kernel, iterations=1)
        
        # For newly-filled pixels, interpolate from neighbors (simple approach: just keep as is)
        # More sophisticated: use cv2.inpaint or scipy interpolation
        
        valid_count = np.sum(~np.isnan(u_warped))
        print(f"[DeformedView] Warped valid pixels: {valid_count}")
        
        # Store in cache for future use
        if deformed_view_cache is not None and frame_idx is not None:
            deformed_view_cache.set_warped_displacement(frame_idx, u_warped, v_warped)
        
        data['img_crop'] = deformed_image
        data['u_masked'] = np.ma.array(u_warped, mask=np.isnan(u_warped))
        data['v_masked'] = np.ma.array(v_warped, mask=np.isnan(v_warped))
        
        print(f"[DeformedView] Deformed mode with inverse warp prepared.")
        return data
        
    except Exception as e:
        print(f"[DeformedView] Error in deformed mode: {e}")
        import traceback
        traceback.print_exc()
        data['error'] = str(e)
        return data




from scipy.ndimage import correlate
from scipy.signal import fftconvolve  # FFT-accelerated convolution for faster strain calculation

# Numba-optimized rotation angle calculation
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

@jit(nopython=True, parallel=True, cache=True)
def _compute_rotation_field_numba(du_dx, du_dy, dv_dx, dv_dy):
    """
    Compute rotation angle field using polar decomposition with analytical 2x2 solutions.
    Numba JIT-compiled for high performance.
    
    For each pixel:
    1. Build deformation gradient: F = I + grad(u)
    2. Compute right Cauchy-Green tensor: C = F^T * F
    3. Analytical eigendecomposition of 2x2 symmetric matrix C
    4. Compute U = sqrt(C) and R = F * U^(-1)
    5. Extract rotation angle: theta = atan2(R21, R11)
    
    Returns rotation angle in degrees.
    """
    H, W = du_dx.shape
    rotation = np.full((H, W), np.nan)
    
    for i in prange(H):
        for j in range(W):
            # Skip invalid pixels
            ux = du_dx[i, j]
            uy = du_dy[i, j]
            vx = dv_dx[i, j]
            vy = dv_dy[i, j]
            
            if np.isnan(ux) or np.isnan(uy) or np.isnan(vx) or np.isnan(vy):
                continue
            
            # Deformation gradient tensor: F = I + grad(u)
            # F = [[F11, F12], [F21, F22]]
            F11 = 1.0 + ux
            F12 = uy
            F21 = vx
            F22 = 1.0 + vy
            
            # Right Cauchy-Green tensor: C = F^T * F
            # C = [[C11, C12], [C12, C22]] (symmetric)
            C11 = F11 * F11 + F21 * F21
            C12 = F11 * F12 + F21 * F22
            C22 = F12 * F12 + F22 * F22
            
            # Analytical eigenvalue decomposition for 2x2 symmetric matrix
            # eigenvalues: lambda = (trace/2) +/- sqrt((trace/2)^2 - det)
            trace = C11 + C22
            det = C11 * C22 - C12 * C12
            
            half_trace = trace * 0.5
            discriminant = half_trace * half_trace - det
            
            if discriminant < 0:
                continue
            
            sqrt_disc = np.sqrt(discriminant)
            lambda1 = half_trace + sqrt_disc
            lambda2 = half_trace - sqrt_disc
            
            if lambda1 < 0 or lambda2 < 0:
                continue
            
            # Compute sqrt of eigenvalues
            sqrt_l1 = np.sqrt(lambda1)
            sqrt_l2 = np.sqrt(lambda2)
            
            # For 2x2 symmetric matrix, we can compute U = sqrt(C) directly
            # Using the formula: U = (C + sqrt(det)*I) / (sqrt(lambda1) + sqrt(lambda2))
            # This avoids explicit eigenvector computation
            sqrt_det = np.sqrt(det)
            denom = sqrt_l1 + sqrt_l2
            
            if abs(denom) < 1e-12:
                continue
            
            U11 = (C11 + sqrt_det) / denom
            U12 = C12 / denom
            U22 = (C22 + sqrt_det) / denom
            
            # Compute U^(-1) for 2x2 matrix
            det_U = U11 * U22 - U12 * U12
            if abs(det_U) < 1e-12:
                continue
            
            inv_det_U = 1.0 / det_U
            U_inv11 = U22 * inv_det_U
            U_inv12 = -U12 * inv_det_U
            U_inv22 = U11 * inv_det_U
            
            # Rotation matrix: R = F * U^(-1)
            R11 = F11 * U_inv11 + F12 * U_inv12
            R12 = F11 * U_inv12 + F12 * U_inv22
            R21 = F21 * U_inv11 + F22 * U_inv12
            R22 = F21 * U_inv12 + F22 * U_inv22
            
            # Ensure proper rotation (det(R) = 1)
            # For 2D, we can just use atan2 directly if R is close to orthogonal
            # The SVD orthogonalization is replaced by direct angle extraction
            # since the analytical solution should give a proper rotation matrix
            
            # Extract rotation angle (2D) in degrees
            rotation[i, j] = np.degrees(np.arctan2(R21, R11))
    
    return rotation

def calculate_strain_field(displacement_field: np.ndarray, method: str = 'green_lagrange', 
                         vsg_size: int = 31, poly_order: int = 1, weighting: str = 'Gaussian', step: int = 1):
    """
    Calculate strain field from displacement using VSG method with robust boundary handling.
    Implements "Vectorized Weighted Least Squares" to handle invalid pixels (Strategy 2).
    
    Args:
        displacement_field: (H, W, 2) array with (u, v) displacements.
        method: 'green_lagrange' (default) or 'engineering'.
        vsg_size: Size of the local window (odd int).
        poly_order: Polynomial order (1 or 2).
        weighting: 'Uniform' or 'Gaussian'.
        step: Calculation stride (downsampling factor).
        
    Returns:
        strain_dict: Dictionary containing strain components.
    """
    import time
    t_start = time.perf_counter()
    
    u = displacement_field[..., 0]
    v = displacement_field[..., 1]
    
    H, W = u.shape
    print(f"[TIMING] Strain calc: input size {H}x{W}, VSG={vsg_size}, step={step}")
    
    # 1. Create Validity Mask (1 for valid, 0 for invalid)
    mask = (~np.isnan(u)) & (~np.isnan(v))
    mask_float = mask.astype(np.float64)
    
    if not np.any(mask):
        return None
        
    # Fill NaNs with 0 for correlation (they will be weighted by 0 via mask_float)
    u_filled = np.nan_to_num(u)
    v_filled = np.nan_to_num(v)
    
    # 2. Generate Basis Kernels for the Window
    if vsg_size % 2 == 0:
        raise ValueError("VSG Size must be odd.")
        
    half = vsg_size // 2
    x_range = np.arange(-half, half + 1)
    y_range = np.arange(-half, half + 1)
    X_grid, Y_grid = np.meshgrid(x_range, y_range) # Local coordinates
    
    # Weighting Kernel
    if weighting == 'Gaussian':
        sigma = vsg_size / 4.0
        dist_sq = X_grid**2 + Y_grid**2
        G = np.exp(-dist_sq / (2 * sigma**2))
    else:
        G = np.ones((vsg_size, vsg_size))
        
    # Define Basis Functions
    # Order 1: [1, x, y]
    # Order 2: [1, x, y, x^2, xy, y^2]
    basis_funcs = [np.ones_like(X_grid), X_grid, Y_grid]
    if poly_order == 2:
        basis_funcs.extend([X_grid**2, X_grid*Y_grid, Y_grid**2])
        
    num_params = len(basis_funcs)
    
    # 3. Construct Linear System M * a = b for each pixel
    # M_kl = sum(w * phi_k * phi_l) -> Correlate(mask_float, G * phi_k * phi_l)
    # b_k  = sum(w * u * phi_k)     -> Correlate(mask_float * u_filled, G * phi_k)
    
    # Pre-compute weighted basis kernels for M
    # We need to compute upper triangular part of M (symmetric)
    # M is (H, W, num_params, num_params)
    
    # To save memory, we can compute and downsample immediately if step > 1?
    # No, correlation needs full grid. But we can slice result immediately.
    
    # Output grid coordinates
    # If step > 1, we only solve for pixels on the grid
    # But correlation must run on full image to capture neighbors correctly.
    
    # Initialize M and b containers (downsampled size)
    # We use lists to store columns/elements to avoid huge 4D arrays
    
    # Slicing for downsampling
    s_slice = slice(None, None, step)
    
    t1 = time.perf_counter()
    print(f"[TIMING] Strain calc - setup: {t1-t_start:.3f}s")
    
    # Build M (Symmetric) - Using FFT convolution for ~40x speedup
    M_elements = [[None for _ in range(num_params)] for _ in range(num_params)]
    
    for i in range(num_params):
        for j in range(i, num_params):
            # Kernel for M_ij: G * phi_i * phi_j
            kernel_M = G * basis_funcs[i] * basis_funcs[j]
            # Use FFT convolution - flip kernel to match correlate behavior
            kernel_flipped = kernel_M[::-1, ::-1]
            res = fftconvolve(mask_float, kernel_flipped, mode='same')
            
            # Downsample
            res_down = res[s_slice, s_slice]
            M_elements[i][j] = res_down
            if i != j:
                M_elements[j][i] = res_down # Symmetry
    
    t2 = time.perf_counter()
    print(f"[TIMING] Strain calc - M matrix FFT ({6 if poly_order==1 else 21} convolutions): {t2-t1:.3f}s")
                
    # Build b for u and v
    # b_u_k = sum(w * u * phi_k)
    b_u_elements = []
    b_v_elements = []
    
    # Pre-multiply data by mask
    u_masked = u_filled * mask_float
    v_masked = v_filled * mask_float
    
    for k in range(num_params):
        # Kernel for b_k: G * phi_k
        kernel_b = G * basis_funcs[k]
        kernel_b_flipped = kernel_b[::-1, ::-1]
        
        res_u = fftconvolve(u_masked, kernel_b_flipped, mode='same')
        res_v = fftconvolve(v_masked, kernel_b_flipped, mode='same')
        
        b_u_elements.append(res_u[s_slice, s_slice])
        b_v_elements.append(res_v[s_slice, s_slice])
    
    t3 = time.perf_counter()
    print(f"[TIMING] Strain calc - b vector FFT ({3 if poly_order==1 else 6}x2 convolutions): {t3-t2:.3f}s")
        
    # 4. Solve Linear Systems
    # Stack into arrays
    # M_stack: (N_pixels, num_params, num_params)
    # b_u_stack: (N_pixels, num_params)
    
    # Flatten spatial dimensions for batch solving
    H_down, W_down = M_elements[0][0].shape
    N_pixels = H_down * W_down
    
    M_stack = np.empty((N_pixels, num_params, num_params))
    b_u_stack = np.empty((N_pixels, num_params))
    b_v_stack = np.empty((N_pixels, num_params))
    
    for i in range(num_params):
        b_u_stack[:, i] = b_u_elements[i].flatten()
        b_v_stack[:, i] = b_v_elements[i].flatten()
        for j in range(num_params):
            M_stack[:, i, j] = M_elements[i][j].flatten()
            
    # Solve
    # Check for singular matrices (too few points)
    # We can check condition number or determinant, but simplest is to try solve and catch,
    # or rely on pseudoinverse. Pinv is safer for ill-conditioned boundaries.
    
    # Using pinv is slower but robust.
    # M_inv = np.linalg.pinv(M_stack)
    # coeffs_u = M_inv @ b_u_stack[..., None]
    
    # Let's use solve for speed, but mask out bad pixels?
    # A pixel is "bad" if sum of weights (M_00) is too small.
    min_weight = 1e-6
    valid_solve_mask = M_stack[:, 0, 0] > min_weight
    
    coeffs_u = np.full((N_pixels, num_params), np.nan)
    coeffs_v = np.full((N_pixels, num_params), np.nan)
    
    t4 = time.perf_counter()
    print(f"[TIMING] Strain calc - stack arrays: {t4-t3:.3f}s, solving {np.sum(valid_solve_mask)}/{N_pixels} pixels")
    
    # Only solve for valid pixels
    if np.any(valid_solve_mask):
        try:
            # Standard solve
            M_valid = M_stack[valid_solve_mask]
            # Reshape b to (N, 3, 1) to ensure correct broadcasting
            b_u_valid = b_u_stack[valid_solve_mask][..., None]
            b_v_valid = b_v_stack[valid_solve_mask][..., None]
            
            # Result will be (N, 3, 1), squeeze back to (N, 3)
            coeffs_u[valid_solve_mask] = np.linalg.solve(M_valid, b_u_valid).squeeze(-1)
            coeffs_v[valid_solve_mask] = np.linalg.solve(M_valid, b_v_valid).squeeze(-1)
        except np.linalg.LinAlgError:
            # Fallback to lstsq or pinv if singular
            # This is slow, maybe just iterate or use pinv on the batch
            # For now, let's use pinv on the valid subset
            M_valid = M_stack[valid_solve_mask]
            b_u_valid = b_u_stack[valid_solve_mask][..., None]
            b_v_valid = b_v_stack[valid_solve_mask][..., None]
            
            M_inv = np.linalg.pinv(M_valid)
            coeffs_u[valid_solve_mask] = (M_inv @ b_u_valid).squeeze(-1)
            coeffs_v[valid_solve_mask] = (M_inv @ b_v_valid).squeeze(-1)

    t5 = time.perf_counter()
    print(f"[TIMING] Strain calc - linear solve: {t5-t4:.3f}s")

    # 5. Extract Gradients
    # a0, a1(x), a2(y), ...
    # du/dx = a1
    # du/dy = a2
    
    du_dx_flat = coeffs_u[:, 1]
    du_dy_flat = coeffs_u[:, 2]
    dv_dx_flat = coeffs_v[:, 1]
    dv_dy_flat = coeffs_v[:, 2]
    
    # Reshape back to grid
    du_dx = du_dx_flat.reshape(H_down, W_down)
    du_dy = du_dy_flat.reshape(H_down, W_down)
    dv_dx = dv_dx_flat.reshape(H_down, W_down)
    dv_dy = dv_dy_flat.reshape(H_down, W_down)
    
    # 6. Strain Calculation
    if method == 'green_lagrange':
        exx = du_dx + 0.5 * (du_dx**2 + dv_dx**2)
        eyy = dv_dy + 0.5 * (du_dy**2 + dv_dy**2)
        exy = 0.5 * (du_dy + dv_dx + du_dx*du_dy + dv_dx*dv_dy)
        
    elif method == 'engineering':
        exx = du_dx
        eyy = dv_dy
        exy = 0.5 * (du_dy + dv_dx)
        
    else:
        raise ValueError(f"Unknown strain method: {method}")
        
    # Principal Strains
    center = (exx + eyy) / 2.0
    radius = np.sqrt(((exx - eyy) / 2.0)**2 + exy**2)
    e1 = center + radius
    e2 = center - radius
    max_shear = radius 
    von_mises = np.sqrt(e1**2 - e1*e2 + e2**2)
    
    # Rotation Angle via Polar Decomposition (Numba-optimized)
    # F = I + grad(u) -> C = F^T*F -> U = sqrt(C) -> R = F*U^(-1) -> theta = atan2(R21, R11)
    rotation = _compute_rotation_field_numba(du_dx, du_dy, dv_dx, dv_dy)
    
    # Mask out pixels that were originally invalid (to prevent ROI expansion)
    # Downsample mask
    mask_down = mask[s_slice, s_slice]
    
    # Apply mask to all components
    for key, val in locals().items():
        if key in ['exx', 'eyy', 'exy', 'e1', 'e2', 'max_shear', 'von_mises', 'rotation']:
            val[~mask_down] = np.nan

    return {
        'exx': exx,
        'eyy': eyy,
        'exy': exy,
        'e1': e1,
        'e2': e2,
        'max_shear': max_shear,
        'von_mises': von_mises,
        'rotation': rotation
    }




def update_displacement_plot(ax_u, ax_v, data, colormap, alpha, vmin_u, vmax_u, vmin_v, vmax_v):
    """Update existing axes with new data"""
    if data.get('error'):
        return

    # Clear axes but keep them alive
    ax_u.clear()
    ax_v.clear()
    
    img_crop = data['img_crop']
    if img_crop is not None:
        ax_u.imshow(img_crop, cmap='gray')
        ax_v.imshow(img_crop, cmap='gray')

    u_masked = data['u_masked']
    v_masked = data['v_masked']

    if u_masked is not None:
        im_u = ax_u.imshow(u_masked, cmap=colormap, alpha=alpha, vmin=vmin_u, vmax=vmax_u)
        # Note: Colorbar is tricky to update if it's already there. 
        # For robustness, we assume the caller handles colorbar or we just don't update it if it's fixed.
        # But to be safe, we can let the caller handle colorbar creation/update or just re-create it.
        # However, re-creating colorbar every time might shrink the axis.
        # Ideally, we should update the mappable of the existing colorbar.
        pass

    if v_masked is not None:
        im_v = ax_v.imshow(v_masked, cmap=colormap, alpha=alpha, vmin=vmin_v, vmax=vmax_v)

    ax_u.set_title("U Component")
    ax_v.set_title("V Component")

    boundary = data.get('boundary_points')
    if boundary:
        xb, yb = boundary
        ax_u.plot(xb, yb, color='white', linewidth=1.0, alpha=0.8)
        ax_v.plot(xb, yb, color='white', linewidth=1.0, alpha=0.8)

    quiver = data.get('quiver_data')
    if quiver:
        xq, yq, uq, vq = quiver
        ax_u.quiver(xq, yq, uq, vq, color='white', angles='xy', scale_units='xy', scale=1, width=0.002, alpha=0.9)
        ax_v.quiver(xq, yq, uq, vq, color='white', angles='xy', scale_units='xy', scale=1, width=0.002, alpha=0.9)

    ax_u.set_axis_off()
    ax_v.set_axis_off()
    ax_u.set_aspect('equal')
    ax_v.set_aspect('equal')
    





def update_displacement_plot(ax_u, ax_v, data, colormap, alpha, vmin_u, vmax_u, vmin_v, vmax_v):
    """Update existing axes with new data"""
    if data.get('error'):
        return

    # Clear axes but keep them alive
    ax_u.clear()
    ax_v.clear()
    
    img_crop = data['img_crop']
    if img_crop is not None:
        ax_u.imshow(img_crop, cmap='gray')
        ax_v.imshow(img_crop, cmap='gray')

    u_masked = data['u_masked']
    v_masked = data['v_masked']

    if u_masked is not None:
        im_u = ax_u.imshow(u_masked, cmap=colormap, alpha=alpha, vmin=vmin_u, vmax=vmax_u)
        # Note: Colorbar is tricky to update if it's already there. 
        # For robustness, we assume the caller handles colorbar or we just don't update it if it's fixed.
        # But to be safe, we can let the caller handle colorbar creation/update or just re-create it.
        # However, re-creating colorbar every time might shrink the axis.
        # Ideally, we should update the mappable of the existing colorbar.
        pass

    if v_masked is not None:
        im_v = ax_v.imshow(v_masked, cmap=colormap, alpha=alpha, vmin=vmin_v, vmax=vmax_v)

    ax_u.set_title("U Component")
    ax_v.set_title("V Component")

    boundary = data.get('boundary_points')
    if boundary:
        xb, yb = boundary
        ax_u.plot(xb, yb, color='white', linewidth=1.0, alpha=0.8)
        ax_v.plot(xb, yb, color='white', linewidth=1.0, alpha=0.8)

    quiver = data.get('quiver_data')
    if quiver:
        xq, yq, uq, vq = quiver
        ax_u.quiver(xq, yq, uq, vq, color='white', angles='xy', scale_units='xy', scale=1, width=0.002, alpha=0.9)
        ax_v.quiver(xq, yq, uq, vq, color='white', angles='xy', scale_units='xy', scale=1, width=0.002, alpha=0.9)

    ax_u.set_axis_off()
    ax_v.set_axis_off()
    ax_u.set_aspect('equal')
    ax_v.set_aspect('equal')
    
    return im_u, im_v


import scipy.io
import os

def save_scientific_results(displacement_results, strain_results, roi_mask, roi_rect, metadata, file_path, image_files=None, upsample_strain=True, warped_masks=None):
    """
    Save full-field results (Displacement + Strain) to .npz or .mat format.
    
    Args:
        displacement_results: List of (H, W, 2) arrays or paths to .npy files.
        strain_results: List of dicts containing strain components.
        roi_mask: (H, W) boolean mask.
        roi_rect: (xmin, ymin, xmax, ymax).
        metadata: Dict containing analysis parameters.
        file_path: Full path to the output file (including extension .npz or .mat).
        image_files: List of absolute paths to the images corresponding to each frame.
        upsample_strain: If True, upsample strain to ROI resolution. If False, save native resolution.
        warped_masks: Optional list of warped masks from incremental processing (for debugging).
    """
    try:
        # 1. Prepare Static Data (Coordinates)
        xmin, ymin, xmax, ymax = roi_rect
        h, w = roi_mask.shape
        
        # Create meshgrid for ROI
        x_range = np.arange(xmin, xmax)
        y_range = np.arange(ymin, ymax)
        X_ref, Y_ref = np.meshgrid(x_range, y_range)
        
        # Ensure mask matches ROI size (it should be full image size usually, need to crop)
        # roi_mask passed here is usually the full image mask. Crop it to ROI.
        mask_crop = roi_mask[ymin:ymax, xmin:xmax]
        
        # 2. Prepare Dynamic Data (Time-Series)
        T = len(displacement_results)
        H_roi, W_roi = mask_crop.shape
        
        # Initialize arrays with NaNs
        U_stack = np.full((T, H_roi, W_roi), np.nan, dtype=np.float32)
        V_stack = np.full((T, H_roi, W_roi), np.nan, dtype=np.float32)
        
        # Strain stacks (must match keys returned by calculate_strain_field)
        strain_keys = ['exx', 'eyy', 'exy', 'e1', 'e2', 'max_shear', 'von_mises', 'rotation']
        strain_stacks = {k: np.full((T, H_roi, W_roi), np.nan, dtype=np.float32) for k in strain_keys}
        
        # Loop through frames
        for t in range(T):
            try:
                # Load Displacement
                d = displacement_results[t]
                if isinstance(d, str):
                    d = np.load(d)
                
                # Validation
                if d is None or d.size == 0:
                    print(f"[Warning] Frame {t} is empty or None. Skipping.")
                    continue
            
                h_d, w_d = d.shape[:2]
                
                # Debug logging for first frame/mismatches
                if t == 0:
                    print(f"[DEBUG] Export Frame 0: shape={d.shape}, expected ROI={H_roi}x{W_roi}")

                # Determine Crop Logic
                if h_d == H_roi and w_d == W_roi:
                    # Already cropped to ROI
                    d_crop = d
                elif h_d == h and w_d == w:
                    # Full image, need crop
                    d_crop = d[ymin:ymax, xmin:xmax]
                else:
                    print(f"[Warning] Frame {t} shape {d.shape} incompatible with ROI {H_roi}x{W_roi} or Full {h}x{w}. using raw.")
                    d_crop = d

                # Safety check before indexing
                if d_crop.shape[0] != H_roi or d_crop.shape[1] != W_roi:
                     print(f"[Error] Frame {t}: d_crop shape {d_crop.shape} != Mask shape {H_roi}x{W_roi}. Skipping.")
                     continue

            
                # Apply mask (set invalid to NaN)
                valid = mask_crop > 0
                
                u_frame = d_crop[..., 0]
                v_frame = d_crop[..., 1]
                
                # Store
                U_stack[t, valid] = u_frame[valid]
                V_stack[t, valid] = v_frame[valid]
                
                # Load Strain
                if t < len(strain_results):
                    s = strain_results[t]
                    if s:
                        for k in strain_keys:
                            if k in s:
                                val = s[k]
                                h_s, w_s = val.shape
                                
                                # Handle strain dimension mismatch (step > 1 case)
                                if h_s == H_roi and w_s == W_roi:
                                    # Already correct size
                                    val_crop = val
                                elif h_s == h and w_s == w:
                                    # Full image size, crop to ROI
                                    val_crop = val[ymin:ymax, xmin:xmax]
                                elif upsample_strain:
                                    # Different size (likely step > 1), upsample if requested
                                    # Use INTER_NEAREST to preserve exact values (no interpolation)
                                    val_upsampled = cv2.resize(val.astype(np.float32), 
                                                               (W_roi, H_roi), 
                                                               interpolation=cv2.INTER_NEAREST)
                                    val_crop = val_upsampled
                                else:
                                    # Don't upsample - will be handled separately
                                    continue

                                strain_stacks[k][t, valid] = val_crop[valid]
            
            except Exception as e:
                print(f"[Error] Failed processing frame {t}: {str(e)}")
                continue

        # If not upsampling strain, replace strain_stacks with native resolution data
        if not upsample_strain and strain_results:
            # Get native strain size from first valid frame
            first_strain = next((s for s in strain_results if s), None)
            if first_strain:
                first_key = next(iter(first_strain))
                native_h, native_w = first_strain[first_key].shape
                metadata['strain_native_size'] = (native_h, native_w)
                
                # Reinitialize at native size
                strain_stacks = {k: np.full((T, native_h, native_w), np.nan, dtype=np.float32) for k in strain_keys}
                
                for t, s in enumerate(strain_results):
                    if s:
                        for k in strain_keys:
                            if k in s:
                                strain_stacks[k][t] = s[k].astype(np.float32)


        # 3. Save based on extension
        ext = os.path.splitext(file_path)[1].lower()
        
        save_dict = {
            'U': U_stack,
            'V': V_stack,
            'X_ref': X_ref,
            'Y_ref': Y_ref,
            'ROI_mask': mask_crop,
            **strain_stacks,
            **metadata
        }
        
        if image_files is not None:
             save_dict['image_files'] = image_files
        
        # Save warped masks from incremental processing (for debugging)
        if warped_masks is not None and len(warped_masks) > 0:
            for i, wm in enumerate(warped_masks):
                save_dict[f'warped_mask_{i}'] = wm.astype(np.uint8)
            save_dict['num_warped_masks'] = len(warped_masks)

        if ext == '.npz':
            np.savez_compressed(file_path, **save_dict)
            return file_path
        elif ext == '.mat':
            scipy.io.savemat(file_path, save_dict)
            return file_path
        else:
            # Default to npz if unknown
            new_path = file_path + ".npz"
            np.savez_compressed(new_path, **save_dict)
            return new_path
        
    except Exception as e:
        print(f"Error saving scientific results: {e}")
        import traceback
        traceback.print_exc()
        raise e


# ========================= Incremental Reference Processing =========================

def run_incremental_processing(
    model,
    images: list,
    key_frames: list,
    roi_mask: np.ndarray,
    roi_rect: tuple,
    device: str,
    context_padding: int = 32,
    tile_overlap: int = 32,
    p_max_pixels: int = 1100*1100,
    use_smooth: bool = True,
    sigma: float = 2.0,
    iters: int = 12,
    progress_callback=None
) -> tuple:
    """
    Process image sequence with reference updates at key frames (incremental mode).
    
    This function handles large deformations by updating the reference image
    at user-specified key frames and accumulating displacements correctly.
    
    Args:
        model: Loaded RAFT model
        images: List of image file paths
        key_frames: List of frame indices (1-indexed) to use as references
        roi_mask: Full-image boolean mask
        roi_rect: (xmin, ymin, xmax, ymax) ROI bounding box
        device: 'cuda' or 'cpu'
        context_padding: Extra pixels around ROI for context
        tile_overlap: Overlap between tiles
        safety_factor: VRAM safety factor
        p_max_pixels: Maximum pixels per tile
        use_smooth: Whether to apply Gaussian smoothing
        sigma: Smoothing sigma
        iters: RAFT iterations
        progress_callback: Optional callback(frame_idx, total, message)
        
    Returns:
        Tuple of:
        - displacement_results: List of displacements (all relative to Frame 1)
        - warped_masks: List of warped ROI masks at each key frame
        - segment_info: Dict with processing metadata
    """
    from .incremental import (
        warp_mask_with_holes, 
        accumulate_displacement, 
        validate_key_frames,
        get_segment_ranges
    )
    
    num_frames = len(images)
    
    # Validate key frames
    is_valid, msg = validate_key_frames(key_frames, num_frames)
    if not is_valid:
        raise ValueError(f"Invalid key frames: {msg}")
    
    key_frames_sorted = sorted(key_frames)
    segments = get_segment_ranges(key_frames_sorted, num_frames)
    
    # Initialize outputs
    # Index 0 = Frame 1, so we have num_frames - 1 displacement results
    displacement_results = [None] * (num_frames - 1)
    warped_masks = []
    
    # Current accumulated displacement (starts as zeros)
    # Shape matches ROI cropped size
    xmin, ymin, xmax, ymax = roi_rect
    H_roi, W_roi = ymax - ymin, xmax - xmin
    accumulated_disp = np.zeros((H_roi, W_roi, 2), dtype=np.float32)
    
    # Keep reference to ORIGINAL mask (for correct warping)
    original_mask = roi_mask.copy()
    original_mask_crop = original_mask[ymin:ymax, xmin:xmax]
    
    # Current working mask for DIC (starts as original)
    current_mask = roi_mask.copy()
    warped_masks.append(current_mask.copy())  # Store initial mask
    
    print(f"[Incremental] Processing {num_frames} frames with {len(segments)} segments")
    print(f"[Incremental] Key frames: {key_frames_sorted}")
    
    for seg_idx, (ref_frame, start_frame, end_frame) in enumerate(segments):
        print(f"[Incremental] Segment {seg_idx+1}: ref={ref_frame}, frames {start_frame}-{end_frame}")
        
        # Load reference image
        ref_img = load_and_convert_image(images[ref_frame - 1])  # 0-indexed
        
        # Process each frame in this segment
        for frame_idx in range(start_frame, end_frame + 1):
            # 0-indexed for arrays
            result_idx = frame_idx - 2  # Frame 2 -> result[0], Frame 3 -> result[1], etc.
            if result_idx < 0:
                continue  # Skip Frame 1 (reference, no displacement)
            
            if progress_callback:
                progress_callback(frame_idx, num_frames, f"Processing frame {frame_idx}/{num_frames}")
            
            # Load deformed image
            def_img = load_and_convert_image(images[frame_idx - 1])  # 0-indexed
            
            # Run DIC with current mask and reference
            try:
                disp_delta, _ = dic_over_roi_with_tiling(
                    ref_img=ref_img,
                    def_img=def_img,
                    roi_mask_full=current_mask,
                    device=device,
                    model=model,
                    context_padding=context_padding,
                    tile_overlap=tile_overlap,
                    p_max_pixels=p_max_pixels,
                    use_smooth=use_smooth,
                    sigma=sigma,
                    iters=iters
                )
            except Exception as e:
                print(f"[Incremental] Error processing frame {frame_idx}: {e}")
                displacement_results[result_idx] = None
                continue
            
            # Crop to ROI size
            disp_crop = disp_delta[ymin:ymax, xmin:xmax]
            
            # Debug: show raw delta displacement stats
            valid_delta = ~np.isnan(disp_crop[..., 0])
            print(f"[Incremental] Frame {frame_idx} raw delta: valid={np.sum(valid_delta)}, "
                  f"u=[{np.nanmin(disp_crop[...,0]):.2f}, {np.nanmax(disp_crop[...,0]):.2f}], "
                  f"v=[{np.nanmin(disp_crop[...,1]):.2f}, {np.nanmax(disp_crop[...,1]):.2f}]")
            
            # Accumulate displacement
            if seg_idx == 0:
                # First segment: displacement is directly relative to Frame 1
                disp_total = disp_crop
                print(f"[Incremental] Frame {frame_idx}: Segment 0, using delta directly as total")
            else:
                # Subsequent segments: need to accumulate
                print(f"[Incremental] Frame {frame_idx}: Segment {seg_idx}, accumulating with previous displacement")
                disp_total = accumulate_displacement(accumulated_disp, disp_crop)
            
            # Debug: show total displacement stats
            valid_total = ~np.isnan(disp_total[..., 0])
            print(f"[Incremental] Frame {frame_idx} total: valid={np.sum(valid_total)}, "
                  f"u=[{np.nanmin(disp_total[...,0]):.2f}, {np.nanmax(disp_total[...,0]):.2f}], "
                  f"v=[{np.nanmin(disp_total[...,1]):.2f}, {np.nanmax(disp_total[...,1]):.2f}]")
            
            # Store result (ROI-cropped, same format as accumulative mode)
            displacement_results[result_idx] = disp_total
            
            # Update accumulated displacement for next frame
            if frame_idx == end_frame and frame_idx != num_frames:
                # End of segment - this becomes the reference for next segment
                accumulated_disp = disp_total.copy()
                
                # Warp ORIGINAL mask with TOTAL displacement for next segment
                # (Don't use already-warped mask - that causes double deformation!)
                print(f"[Incremental] Warping ORIGINAL mask with total displacement for next segment")
                valid_before = np.sum(original_mask_crop > 0)
                warped_crop = warp_mask_with_holes(original_mask_crop, disp_total)
                valid_after = np.sum(warped_crop)
                print(f"[Incremental] Mask warping: original={valid_before} pixels, warped={valid_after} pixels")
                
                # Reconstruct full mask
                new_mask = original_mask.copy()
                new_mask[ymin:ymax, xmin:xmax] = warped_crop
                current_mask = new_mask
                warped_masks.append(current_mask.copy())
    
    # Segment info for debugging
    segment_info = {
        'key_frames': key_frames_sorted,
        'segments': segments,
        'num_warped_masks': len(warped_masks)
    }
    
    print(f"[Incremental] Processing complete. {sum(1 for d in displacement_results if d is not None)} frames processed.")
    
    return displacement_results, warped_masks, segment_info
