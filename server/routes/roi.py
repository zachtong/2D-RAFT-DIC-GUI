"""ROI mask creation endpoints."""

import cv2
import numpy as np
from flask import Blueprint, jsonify, request

from raft_dic_gui.processing import load_and_convert_image
from server.serializers import image_to_png_bytes, png_response
from server.session import session

roi_bp = Blueprint("roi", __name__)


def _ensure_mask():
    """Ensure roi_mask is initialized to the reference image size."""
    if session.reference_image is None:
        return False
    h, w = session.reference_image.shape[:2]
    if session.roi_mask is None or session.roi_mask.shape != (h, w):
        session.roi_mask = np.zeros((h, w), dtype=bool)
    return True


def _update_rect():
    """Recompute bounding rect from current mask."""
    if session.roi_mask is None:
        session.roi_rect = None
        return
    y_idx, x_idx = np.where(session.roi_mask)
    if len(x_idx) > 0:
        session.roi_rect = (
            int(x_idx.min()),
            int(y_idx.min()),
            int(x_idx.max()) + 1,
            int(y_idx.max()) + 1,
        )
    else:
        h, w = session.roi_mask.shape
        session.roi_rect = (0, 0, w, h)


def _apply_shape(new_mask_bool: np.ndarray, mode: str):
    """Apply a shape mask using add or cut mode."""
    with session._lock:
        if not _ensure_mask():
            return jsonify({"error": "No reference image loaded"}), 400
        if mode == "cut":
            session.roi_mask = session.roi_mask & ~new_mask_bool
        else:
            session.roi_mask = session.roi_mask | new_mask_bool
        _update_rect()

    area = int(session.roi_mask.sum())
    return jsonify({"rect": session.roi_rect, "area_px": area})


@roi_bp.route("/polygon", methods=["POST"])
def add_polygon():
    """Add or cut a polygon from the ROI mask."""
    data = request.get_json(force=True)
    points = data.get("points", [])
    mode = data.get("mode", "add")

    if not points or len(points) < 3:
        return jsonify({"error": "Need at least 3 points"}), 400
    if session.reference_image is None:
        return jsonify({"error": "No images loaded"}), 400

    h, w = session.reference_image.shape[:2]
    pts = np.array(points, np.int32)
    new_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(new_mask, [pts], 1)

    return _apply_shape(new_mask.astype(bool), mode)


@roi_bp.route("/rectangle", methods=["POST"])
def add_rectangle():
    """Add or cut a rectangle from the ROI mask."""
    data = request.get_json(force=True)
    mode = data.get("mode", "add")

    if session.reference_image is None:
        return jsonify({"error": "No images loaded"}), 400

    h, w = session.reference_image.shape[:2]
    x0 = int(data.get("x0", 0))
    y0 = int(data.get("y0", 0))
    x1 = int(data.get("x1", w))
    y1 = int(data.get("y1", h))

    new_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(new_mask, (x0, y0), (x1, y1), 1, -1)

    return _apply_shape(new_mask.astype(bool), mode)


@roi_bp.route("/circle", methods=["POST"])
def add_circle():
    """Add or cut a circle from the ROI mask."""
    data = request.get_json(force=True)
    mode = data.get("mode", "add")

    if session.reference_image is None:
        return jsonify({"error": "No images loaded"}), 400

    h, w = session.reference_image.shape[:2]
    cx = int(data.get("cx", 0))
    cy = int(data.get("cy", 0))
    r = int(data.get("r", 0))

    if r <= 0:
        return jsonify({"error": "Radius must be positive"}), 400

    new_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(new_mask, (cx, cy), r, 1, -1)

    return _apply_shape(new_mask.astype(bool), mode)


@roi_bp.route("/import", methods=["POST"])
def import_mask():
    """Import an ROI mask from an image file.

    Optional parameters:
    - min_area (int, default 0): remove connected components smaller than this.
    - smooth_radius (int, default 0): Gaussian blur radius before binarization.
      Kernel size = smooth_radius * 2 + 1.
    """
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

    # 1. Resize to match reference image dimensions
    mask_img = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_NEAREST)

    # 2. Optional smoothing (Gaussian blur before binarization)
    if smooth_radius > 0:
        ksize = smooth_radius * 2 + 1
        mask_img = cv2.GaussianBlur(mask_img, (ksize, ksize), 0)

    # 3. Binarize
    binary = (mask_img > 127).astype(np.uint8)

    # 4. Optional small-component removal
    if min_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        for label_id in range(1, num_labels):
            if stats[label_id, cv2.CC_STAT_AREA] < min_area:
                binary[labels == label_id] = 0

    with session._lock:
        session.roi_mask = binary.astype(bool)
        _update_rect()
        session.roi_confirmed = True

    area = int(session.roi_mask.sum())
    return jsonify({"rect": session.roi_rect, "area_px": area})


@roi_bp.route("/invert", methods=["POST"])
def invert_mask():
    """Invert the current ROI mask."""
    if session.roi_mask is None:
        return jsonify({"error": "No ROI mask exists"}), 400

    with session._lock:
        session.roi_mask = ~session.roi_mask
        _update_rect()

    area = int(session.roi_mask.sum())
    return jsonify({"rect": session.roi_rect, "area_px": area})


@roi_bp.route("/mask/binary", methods=["GET"])
def export_mask_binary():
    """Export the current ROI mask as a binary (white-on-black) PNG for saving."""
    from PIL import Image
    import io

    if session.roi_mask is None:
        return jsonify({"error": "No ROI mask exists"}), 404

    mask_uint8 = session.roi_mask.astype(np.uint8) * 255
    pil_img = Image.fromarray(mask_uint8, "L")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    resp = png_response(buf.read())
    resp.headers["Content-Disposition"] = 'attachment; filename="roi_mask.png"'
    return resp


@roi_bp.route("/mask", methods=["GET"])
def get_mask():
    """Return the current ROI mask as a colored RGBA PNG overlay.

    ROI area: semi-transparent blue highlight.
    Non-ROI area: fully transparent.
    """
    from PIL import Image
    import io

    if session.roi_mask is None:
        return jsonify({"error": "No ROI mask exists"}), 404

    h, w = session.roi_mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    # Blue highlight where mask is True
    rgba[session.roi_mask, 0] = 74   # R
    rgba[session.roi_mask, 1] = 108  # G
    rgba[session.roi_mask, 2] = 247  # B (matches --primary)
    rgba[session.roi_mask, 3] = 80   # Alpha

    pil_img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return png_response(buf.read())


@roi_bp.route("/confirm", methods=["POST"])
def confirm_roi():
    """Finalize the ROI for processing."""
    if session.roi_mask is None or session.roi_mask.sum() == 0:
        return jsonify({"error": "No ROI mask to confirm"}), 400

    with session._lock:
        _update_rect()
        session.roi_confirmed = True

    area = int(session.roi_mask.sum())
    return jsonify({
        "ok": True,
        "rect": session.roi_rect,
        "area_px": area,
    })


@roi_bp.route("", methods=["DELETE"])
def clear_roi():
    """Clear the ROI mask."""
    with session._lock:
        session.roi_mask = None
        session.roi_rect = None
        session.roi_confirmed = False

    return jsonify({"ok": True})
