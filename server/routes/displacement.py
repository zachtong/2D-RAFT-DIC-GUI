"""Displacement frame data and rendered visualization endpoints."""

import io

import numpy as np
from flask import Blueprint, jsonify, request

from raft_dic_gui.processing import load_and_convert_image
from raft_dic_gui.velocity import (
    calculate_displacement_magnitude,
    calculate_velocity_central,
    calculate_velocity_field,
)
from server.render_cache import RenderCache, auto_cache_size
from server.serializers import frame_data_to_json, png_response
from server.session import session

displacement_bp = Blueprint("displacement", __name__)
_render_cache = RenderCache(auto_cache_size())


def _active_rect():
    """Return envelope_rect if available, else roi_rect."""
    return session.envelope_rect or session.roi_rect


def _get_displacement_component(
    frame_idx: int, component: str, ref_frame: int = 0
) -> np.ndarray:
    """Extract a displacement component for a given frame.

    When *ref_frame* > 0 the returned data is the relative displacement
    ``disp[frame_idx] - disp[ref_frame]``.  The subtraction is applied to the
    raw (H, W, 2) array **before** extracting the requested component so that
    magnitude / velocity calculations see the re-referenced field.
    """
    disp = session.displacement_results[frame_idx]  # shape (H, W, 2)

    # Subtract reference frame displacement when requested
    if ref_frame > 0 and ref_frame < len(session.displacement_results):
        ref_disp = session.displacement_results[ref_frame]
        disp = disp - ref_disp  # creates a copy — does NOT mutate session data

    if component == "u":
        return disp[:, :, 0]
    elif component == "v":
        return disp[:, :, 1]
    elif component == "magnitude":
        return calculate_displacement_magnitude(disp[:, :, 0], disp[:, :, 1])
    elif component == "velocity":
        frames_u = [d[:, :, 0] for d in session.displacement_results]
        frames_v = [d[:, :, 1] for d in session.displacement_results]
        return calculate_velocity_central(frames_u, frames_v, frame_idx)
    else:
        return disp[:, :, 0]


@displacement_bp.route("/info", methods=["GET"])
def get_info():
    """Return displacement result metadata."""
    if not session.displacement_results:
        return jsonify({"error": "No displacement results"}), 404

    sample = session.displacement_results[0]
    return jsonify({
        "num_frames": len(session.displacement_results),
        "roi_rect": session.roi_rect,
        "roi_shape": list(sample.shape[:2]),
    })


@displacement_bp.route("/frame/<int:idx>", methods=["GET"])
def get_frame_data(idx: int):
    """Return raw displacement data for a frame as JSON."""
    if not session.displacement_results:
        return jsonify({"error": "No displacement results"}), 404

    if idx < 0 or idx >= len(session.displacement_results):
        return jsonify({"error": "Frame index out of range"}), 400

    component = request.args.get("component", "u")
    ref_frame = request.args.get("ref_frame", 0, type=int)
    data = _get_displacement_component(idx, component, ref_frame=ref_frame)

    vmin = request.args.get("vmin", type=float)
    vmax = request.args.get("vmax", type=float)

    return jsonify(frame_data_to_json(data, vmin=vmin, vmax=vmax))


@displacement_bp.route("/range/<int:idx>", methods=["GET"])
def get_range(idx: int):
    """Return auto vmin/vmax for a displacement component (lightweight)."""
    if not session.displacement_results:
        return jsonify({"error": "No displacement results"}), 404
    if idx < 0 or idx >= len(session.displacement_results):
        return jsonify({"error": "Frame index out of range"}), 400

    component = request.args.get("component", "u")
    ref_frame = request.args.get("ref_frame", 0, type=int)
    data = _get_displacement_component(idx, component, ref_frame=ref_frame)
    finite = data[np.isfinite(data)]
    vmin = float(finite.min()) if finite.size > 0 else 0.0
    vmax = float(finite.max()) if finite.size > 0 else 1.0
    return jsonify({"vmin": vmin, "vmax": vmax})


@displacement_bp.route("/render/<int:idx>", methods=["GET"])
def render_frame(idx: int):
    """Render a displacement overlay as PNG using PIL compositing."""
    from PIL import Image
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import os

    if not session.displacement_results:
        return jsonify({"error": "No displacement results"}), 404

    if idx < 0 or idx >= len(session.displacement_results):
        return jsonify({"error": "Frame index out of range"}), 400

    component = request.args.get("component", "u")
    colormap = request.args.get("colormap", "turbo")
    alpha = request.args.get("alpha", 0.7, type=float)
    vmin = request.args.get("vmin", type=float)
    vmax = request.args.get("vmax", type=float)
    background = request.args.get("background", "reference")
    warp_quality = request.args.get("warp_quality", "balanced")
    log_scale = request.args.get("log_scale", "false").lower() in ("true", "1", "yes")
    ref_frame = request.args.get("ref_frame", 0, type=int)
    vw = request.args.get("vw", 0, type=int)
    vh = request.args.get("vh", 0, type=int)
    overlay_only = request.args.get("overlay_only", "false").lower() in ("true", "1")

    # Check cache — exclude alpha when overlay_only (applied client-side)
    cache_params = {k: v for k, v in request.args.items() if k != "_t"}
    if overlay_only:
        cache_params.pop("alpha", None)
    cache_key = (session.result_version, idx, tuple(sorted(cache_params.items())))
    cached = _render_cache.get(cache_key)
    if cached is not None:
        return png_response(cached)

    disp_data = _get_displacement_component(idx, component, ref_frame=ref_frame)

    # Place displacement data in full image coordinates
    h, w = session.image_height, session.image_width
    if h == 0 or w == 0:
        if session.reference_image is not None:
            h, w = session.reference_image.shape[:2]
        else:
            return jsonify({"error": "No image dimensions"}), 500

    rect = _active_rect()
    if background == "deformed" and rect:
        from server.deformed_warp import get_warped_full_data
        disp = session.displacement_results[idx]
        U, V = disp[:, :, 0], disp[:, :, 1]
        full_data = get_warped_full_data(
            data=disp_data, frame_idx=idx,
            U=U, V=V,
            roi_rect=rect,
            image_shape=(h, w),
            cache=session.inverse_map_cache,
            quality=warp_quality,
        )
    else:
        full_data = np.full((h, w), np.nan)
        if rect:
            x0, y0, x1, y1 = rect
            dh, dw = disp_data.shape
            sh = min(dh, y1 - y0)
            sw = min(dw, x1 - x0)
            full_data[y0:y0 + sh, x0:x0 + sw] = disp_data[:sh, :sw]

    if overlay_only:
        # --- Overlay-only mode: return RGBA overlay, no background ---
        # Viewport downsampling (data only, skip bg)
        if vw > 0 and vh > 0:
            import cv2
            scale = min(vw / w, vh / h, 1.0)
            if scale < 1.0:
                out_w, out_h = max(1, int(w * scale)), max(1, int(h * scale))
                mask_f = np.isfinite(full_data).astype(np.float32)
                filled = np.nan_to_num(full_data, nan=0.0).astype(np.float32)
                full_data = cv2.resize(filled, (out_w, out_h), interpolation=cv2.INTER_AREA).astype(np.float64)
                mask_s = cv2.resize(mask_f, (out_w, out_h), interpolation=cv2.INTER_AREA)
                full_data[mask_s < 0.5] = np.nan
                h, w = out_h, out_w

        mask = ~np.isnan(full_data)
        valid_vals = full_data[mask]
        if vmin is None:
            vmin = float(valid_vals.min()) if valid_vals.size > 0 else 0.0
        if vmax is None:
            vmax = float(valid_vals.max()) if valid_vals.size > 0 else 1.0
        if vmin >= vmax:
            vmax = vmin + 1e-10

        cmap = cm.get_cmap(colormap)
        if log_scale:
            log_vmin = vmin if vmin > 0 else 1e-10
            log_vmax = vmax if vmax > 0 else 1.0
            if log_vmin >= log_vmax:
                log_vmin = log_vmax / 1000
            norm = mcolors.LogNorm(vmin=log_vmin, vmax=log_vmax)
        else:
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        normalized = norm(np.nan_to_num(full_data, nan=0.0))
        colored = cmap(normalized)
        colored[:, :, 3] = mask.astype(np.float64)  # 1.0 valid, 0.0 NaN
        overlay_rgba = (colored * 255).astype(np.uint8)
        overlay_pil = Image.fromarray(overlay_rgba, "RGBA")

        buf = io.BytesIO()
        overlay_pil.save(buf, format="PNG")
        buf.seek(0)
        png_bytes = buf.read()
        _render_cache.put(cache_key, png_bytes)
        return png_response(png_bytes)

    # --- Composited mode (legacy) ---
    # Load background image
    if background == "deformed" and idx + 1 < len(session.image_files):
        bg_img = session.deformed_view_cache.get_deformed_image(idx + 1)
        if bg_img is None:
            bg_path = os.path.join(session.image_dir, session.image_files[idx + 1])
            bg_img = load_and_convert_image(bg_path)
    else:
        bg_img = session.reference_image

    if bg_img is None:
        return jsonify({"error": "No reference image"}), 500

    h, w = bg_img.shape[:2]

    # Viewport downsampling
    from server.viewport import downsample_for_viewport
    full_data, bg_img, h, w = downsample_for_viewport(full_data, bg_img, vw, vh)

    # Build valid-data mask
    mask = ~np.isnan(full_data)
    valid_vals = full_data[mask]

    # Auto-determine vmin/vmax from data if not specified
    if vmin is None:
        vmin = float(valid_vals.min()) if valid_vals.size > 0 else 0.0
    if vmax is None:
        vmax = float(valid_vals.max()) if valid_vals.size > 0 else 1.0
    if vmin >= vmax:
        vmax = vmin + 1e-10

    # Convert background to RGB PIL image
    if bg_img.ndim == 2:
        bg_pil = Image.fromarray(bg_img).convert("RGB")
    elif bg_img.shape[2] == 4:
        bg_pil = Image.fromarray(bg_img[:, :, :3])
    else:
        bg_pil = Image.fromarray(bg_img)

    # Apply colormap: normalize data → colormap → RGBA
    cmap = cm.get_cmap(colormap)
    if log_scale:
        log_vmin = vmin if vmin > 0 else 1e-10
        log_vmax = vmax if vmax > 0 else 1.0
        if log_vmin >= log_vmax:
            log_vmin = log_vmax / 1000
        norm = mcolors.LogNorm(vmin=log_vmin, vmax=log_vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    normalized = norm(np.nan_to_num(full_data, nan=0.0))
    colored = cmap(normalized)  # (h, w, 4) float [0, 1]

    # Set alpha: overlay_alpha * data_valid_mask
    colored[:, :, 3] = alpha * mask.astype(np.float64)

    # Convert to uint8 RGBA
    overlay_rgba = (colored * 255).astype(np.uint8)
    overlay_pil = Image.fromarray(overlay_rgba, "RGBA")

    # Composite overlay onto background
    result = bg_pil.copy()
    result.paste(overlay_pil, (0, 0), overlay_pil)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)
    png_bytes = buf.read()
    _render_cache.put(cache_key, png_bytes)

    return png_response(png_bytes)


@displacement_bp.route("/download/<int:idx>", methods=["GET"])
def download_frame(idx):
    """Download a single rendered displacement frame as a high-res PNG."""
    import os
    from io import BytesIO
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from flask import send_file

    if not session.displacement_results:
        return jsonify({"error": "No displacement results"}), 404
    if idx < 0 or idx >= len(session.displacement_results):
        return jsonify({"error": "Frame index out of range"}), 400

    component = request.args.get("component", "u")
    colormap = request.args.get("colormap", "turbo")
    alpha = request.args.get("alpha", 0.7, type=float)
    vmin = request.args.get("vmin", type=float)
    vmax = request.args.get("vmax", type=float)
    background = request.args.get("background", "reference")
    log_scale = request.args.get("log_scale", "false").lower() in ("true", "1")
    dpi = request.args.get("dpi", 150, type=int)
    ref_frame = request.args.get("ref_frame", 0, type=int)

    disp_data = _get_displacement_component(idx, component, ref_frame=ref_frame)

    # Background image
    if background == "deformed" and idx + 1 < len(session.image_files):
        bg_path = os.path.join(session.image_dir, session.image_files[idx + 1])
        bg_img = load_and_convert_image(bg_path)
    else:
        bg_img = session.reference_image

    if bg_img is None:
        return jsonify({"error": "No background image"}), 500

    h, w = bg_img.shape[:2]

    # Place data in full image coordinates
    rect = _active_rect()
    full_data = np.full((h, w), np.nan)
    if rect:
        x0, y0, x1, y1 = rect
        dh, dw = disp_data.shape
        sh = min(dh, y1 - y0)
        sw = min(dw, x1 - x0)
        full_data[y0:y0 + sh, x0:x0 + sw] = disp_data[:sh, :sw]
    else:
        full_data[:disp_data.shape[0], :disp_data.shape[1]] = disp_data

    # Render with matplotlib
    fig_w = w / dpi
    fig_h = h / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    ax.imshow(bg_img, cmap="gray" if bg_img.ndim == 2 else None)

    masked = np.ma.array(full_data, mask=np.isnan(full_data))

    if vmin is None:
        valid = full_data[~np.isnan(full_data)]
        vmin = float(valid.min()) if valid.size > 0 else 0.0
    if vmax is None:
        valid = full_data[~np.isnan(full_data)]
        vmax = float(valid.max()) if valid.size > 0 else 1.0

    norm = None
    if log_scale and vmin > 0 and vmax > 0:
        norm = mcolors.LogNorm(vmin=max(vmin, 1e-10), vmax=vmax)
        im = ax.imshow(masked, cmap=colormap, alpha=alpha, norm=norm)
    else:
        im = ax.imshow(masked, cmap=colormap, alpha=alpha, vmin=vmin, vmax=vmax)

    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"{component.upper()} — Frame {idx + 1}", fontsize=10)
    ax.axis("off")
    fig.tight_layout(pad=0.5)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return send_file(
        buf,
        mimetype="image/png",
        as_attachment=True,
        download_name=f"{component}_frame_{idx + 1:04d}.png",
    )
