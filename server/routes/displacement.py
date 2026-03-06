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
from server.render_cache import RenderCache
from server.serializers import frame_data_to_json, png_response
from server.session import session

displacement_bp = Blueprint("displacement", __name__)
_render_cache = RenderCache(512)


def _get_displacement_component(frame_idx: int, component: str) -> np.ndarray:
    """Extract a displacement component for a given frame."""
    disp = session.displacement_results[frame_idx]  # shape (H, W, 2)
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
    data = _get_displacement_component(idx, component)

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
    data = _get_displacement_component(idx, component)
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
    log_scale = request.args.get("log_scale", "false").lower() in ("true", "1", "yes")

    # Check cache
    cache_params = {k: v for k, v in request.args.items() if k != "_t"}
    cache_key = (session.result_version, idx, tuple(sorted(cache_params.items())))
    cached = _render_cache.get(cache_key)
    if cached is not None:
        return png_response(cached)

    disp_data = _get_displacement_component(idx, component)

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

    # Place displacement data in full image coordinates
    h, w = bg_img.shape[:2]

    if background == "deformed" and session.roi_rect:
        from server.deformed_warp import get_warped_full_data
        disp = session.displacement_results[idx]
        U, V = disp[:, :, 0], disp[:, :, 1]
        full_data = get_warped_full_data(
            data=disp_data, frame_idx=idx,
            U=U, V=V,
            roi_rect=session.roi_rect,
            image_shape=(h, w),
            cache=session.inverse_map_cache,
        )
    else:
        full_data = np.full((h, w), np.nan)
        if session.roi_rect:
            x0, y0, x1, y1 = session.roi_rect
            dh, dw = disp_data.shape
            # Clamp slice to avoid shape mismatch
            sh = min(dh, y1 - y0)
            sw = min(dw, x1 - x0)
            full_data[y0:y0 + sh, x0:x0 + sw] = disp_data[:sh, :sw]

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

    disp_data = _get_displacement_component(idx, component)

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
    full_data = np.full((h, w), np.nan)
    if session.roi_rect:
        x0, y0, x1, y1 = session.roi_rect
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
