"""Strain calculation and data endpoints."""

import io
import threading

import numpy as np
from flask import Blueprint, jsonify, request

from raft_dic_gui.processing import calculate_strain_field
from server.app import socketio
from server.render_cache import RenderCache
from server.serializers import frame_data_to_json, png_response
from server.session import session

strain_bp = Blueprint("strain", __name__)
_render_cache = RenderCache(512)

STRAIN_COMPONENTS = [
    "exx", "eyy", "exy", "e1", "e2", "max_shear", "von_mises", "rotation"
]


@strain_bp.route("/calculate", methods=["POST"])
def calculate():
    """Calculate strain fields for all displacement frames (async)."""
    if not session.displacement_results:
        return jsonify({"error": "No displacement results available"}), 400

    if session.strain_computing:
        return jsonify({"error": "Strain calculation already in progress"}), 409

    data = request.get_json(force=True)
    method = data.get("method", "green_lagrange")
    vsg_size = int(data.get("vsg_size", 31))
    poly_order = int(data.get("poly_order", 1))
    weighting = data.get("weighting", "Gaussian")
    step = int(data.get("step", 1))

    def compute():
        try:
            session.strain_computing = True
            results = []
            total = len(session.displacement_results)

            for i, disp in enumerate(session.displacement_results):
                strain_dict = calculate_strain_field(
                    disp,
                    method=method,
                    vsg_size=vsg_size,
                    poly_order=poly_order,
                    weighting=weighting,
                    step=step,
                )
                results.append(strain_dict)

                socketio.emit("strain:progress", {
                    "percent": round((i + 1) / total * 100, 1),
                    "current": i + 1,
                    "total": total,
                })

            with session._lock:
                session.strain_results = results
                session.result_version += 1
                session.strain_components = STRAIN_COMPONENTS
                session.strain_computing = False
                _render_cache.clear()

            socketio.emit("strain:complete", {
                "num_frames": len(results),
                "components": STRAIN_COMPONENTS,
            })

        except Exception as e:
            session.strain_computing = False
            socketio.emit("strain:error", {"error": str(e)})

    thread = threading.Thread(target=compute, daemon=True)
    thread.start()

    return jsonify({"ok": True})


@strain_bp.route("/status", methods=["GET"])
def get_status():
    """Return strain calculation status."""
    return jsonify({
        "computed": len(session.strain_results) > 0,
        "computing": session.strain_computing,
        "components": session.strain_components,
        "num_frames": len(session.strain_results),
    })


@strain_bp.route("/frame/<int:idx>", methods=["GET"])
def get_frame_data(idx: int):
    """Return raw strain data for a frame as JSON."""
    if not session.strain_results:
        return jsonify({"error": "No strain results"}), 404

    if idx < 0 or idx >= len(session.strain_results):
        return jsonify({"error": "Frame index out of range"}), 400

    component = request.args.get("component", "exx")
    strain_dict = session.strain_results[idx]

    if strain_dict is None or component not in strain_dict:
        return jsonify({"error": f"Component '{component}' not available"}), 400

    data = strain_dict[component]
    vmin = request.args.get("vmin", type=float)
    vmax = request.args.get("vmax", type=float)

    return jsonify(frame_data_to_json(data, vmin=vmin, vmax=vmax))


@strain_bp.route("/range/<int:idx>", methods=["GET"])
def get_range(idx: int):
    """Return auto vmin/vmax for a strain component (lightweight)."""
    if not session.strain_results:
        return jsonify({"error": "No strain results"}), 404
    if idx < 0 or idx >= len(session.strain_results):
        return jsonify({"error": "Frame index out of range"}), 400

    component = request.args.get("component", "exx")
    strain_dict = session.strain_results[idx]
    if strain_dict is None or component not in strain_dict:
        return jsonify({"error": f"Component '{component}' not available"}), 400

    data = strain_dict[component]
    finite = data[np.isfinite(data)]
    vmin = float(finite.min()) if finite.size > 0 else 0.0
    vmax = float(finite.max()) if finite.size > 0 else 1.0
    return jsonify({"vmin": vmin, "vmax": vmax})


@strain_bp.route("/render/<int:idx>", methods=["GET"])
def render_frame(idx: int):
    """Render a strain field overlay as PNG using PIL compositing."""
    from PIL import Image
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import os

    if not session.strain_results:
        return jsonify({"error": "No strain results"}), 404

    if idx < 0 or idx >= len(session.strain_results):
        return jsonify({"error": "Frame index out of range"}), 400

    component = request.args.get("component", "exx")
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

    strain_dict = session.strain_results[idx]
    if strain_dict is None or component not in strain_dict:
        return jsonify({"error": f"Component '{component}' not available"}), 400

    strain_data = strain_dict[component]

    # Load background
    if background == "deformed" and idx + 1 < len(session.image_files):
        bg_path = os.path.join(session.image_dir, session.image_files[idx + 1])
        from raft_dic_gui.processing import load_and_convert_image
        bg_img = load_and_convert_image(bg_path)
    else:
        bg_img = session.reference_image

    if bg_img is None:
        return jsonify({"error": "No reference image"}), 500

    h, w = bg_img.shape[:2]

    if background == "deformed" and session.roi_rect:
        from server.deformed_warp import get_warped_full_data
        x0, y0, x1, y1 = session.roi_rect
        roi_h, roi_w = y1 - y0, x1 - x0
        sh, sw = strain_data.shape

        disp = session.displacement_results[idx]
        U, V = disp[:, :, 0], disp[:, :, 1]
        full_data = get_warped_full_data(
            data=strain_data, frame_idx=idx,
            U=U, V=V,
            roi_rect=session.roi_rect,
            image_shape=(h, w),
            needs_upsample=(sh != roi_h or sw != roi_w),
            roi_h=roi_h, roi_w=roi_w,
            cache=session.inverse_map_cache,
        )
    else:
        full_data = np.full((h, w), np.nan)

        # Strain may be downsampled — upsample to ROI size, then place
        if session.roi_rect:
            x0, y0, x1, y1 = session.roi_rect
            roi_h, roi_w = y1 - y0, x1 - x0
            sh, sw = strain_data.shape

            if sh != roi_h or sw != roi_w:
                import cv2
                mask_valid = ~np.isnan(strain_data)
                data_clean = np.nan_to_num(strain_data, nan=0.0)
                data_resized = cv2.resize(data_clean, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
                mask_resized = cv2.resize(mask_valid.astype(np.uint8), (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
                data_resized[mask_resized == 0] = np.nan
                full_data[y0:y1, x0:x1] = data_resized
            else:
                full_data[y0:y1, x0:x1] = strain_data

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

    # Apply colormap: normalize → colormap → RGBA
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
