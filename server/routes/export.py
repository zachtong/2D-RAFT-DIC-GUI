"""Scientific and batch image export endpoints."""

import os
import threading

from flask import Blueprint, jsonify, request

from raft_dic_gui.processing import load_and_convert_image, save_scientific_results
from raft_dic_gui.export_images import export_batch_images
from server.app import socketio
from server.session import session

export_bp = Blueprint("export", __name__)


@export_bp.route("/scientific", methods=["POST"])
def export_scientific():
    """Export displacement + strain data to .mat or .npz."""
    if not session.displacement_results:
        return jsonify({"error": "No displacement results"}), 400

    data = request.get_json(force=True)
    file_path = data.get("file_path", "").strip()
    upsample_strain = data.get("upsample_strain", True)
    metadata = data.get("metadata", {})

    if not file_path:
        return jsonify({"error": "Missing file_path"}), 400

    # Build metadata dict
    export_metadata = {
        "model_path": session.config.model_path,
        "model_label": session.config.model_label,
        "mode": session.config.mode,
        "img_dir": session.image_dir,
    }
    export_metadata.update(metadata)

    # Collect image paths
    image_files = None
    if session.image_files and session.image_dir:
        image_files = [
            os.path.join(session.image_dir, f) for f in session.image_files
        ]

    try:
        save_scientific_results(
            displacement_results=session.displacement_results,
            strain_results=session.strain_results,
            roi_mask=session.roi_mask,
            roi_rect=session.roi_rect,
            metadata=export_metadata,
            file_path=file_path,
            image_files=image_files,
            upsample_strain=upsample_strain,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"ok": True, "path": file_path})


@export_bp.route("/images", methods=["POST"])
def export_images():
    """Start batch image export (async)."""
    if not session.displacement_results:
        return jsonify({"error": "No displacement results"}), 400

    if session.export_active:
        return jsonify({"error": "Export already in progress"}), 409

    data = request.get_json(force=True)
    output_dir = data.get("output_dir", "").strip()
    components = data.get("components", {})
    frame_range = tuple(data.get("frame_range", [1, len(session.displacement_results)]))
    settings = data.get("settings", {})

    if not output_dir:
        return jsonify({"error": "Missing output_dir"}), 400

    def image_loader(frame_idx):
        if frame_idx < len(session.image_files):
            img_path = os.path.join(session.image_dir, session.image_files[frame_idx])
            return load_and_convert_image(img_path)
        return session.reference_image

    def progress_callback(current, total, message):
        session.export_progress = current
        session.export_total = total
        socketio.emit("export:progress", {
            "current": current,
            "total": total,
            "percent": round(current / max(1, total) * 100, 1),
            "message": message,
        })

    def run_export():
        try:
            session.export_active = True
            session.export_progress = 0

            result_dir = export_batch_images(
                output_dir=output_dir,
                components=components,
                frame_range=frame_range,
                displacement_results=session.displacement_results,
                strain_results=session.strain_results,
                roi_rect=session.roi_rect,
                roi_mask=session.roi_mask,
                image_loader=image_loader,
                settings=settings,
                deformed_view_cache=session.deformed_view_cache,
                progress_callback=progress_callback,
            )

            session.export_active = False
            socketio.emit("export:complete", {"path": result_dir})

        except Exception as e:
            session.export_active = False
            socketio.emit("export:error", {"error": str(e)})

    thread = threading.Thread(target=run_export, daemon=True)
    thread.start()

    return jsonify({"ok": True})


@export_bp.route("/images/status", methods=["GET"])
def export_status():
    """Return batch image export progress."""
    return jsonify({
        "active": session.export_active,
        "progress": session.export_progress,
        "total": session.export_total,
        "percent": round(
            session.export_progress / max(1, session.export_total) * 100, 1
        ) if session.export_total > 0 else 0,
    })
