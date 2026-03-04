"""DIC processing run/stop/pause/resume endpoints with WebSocket progress."""

import threading

from flask import Blueprint, jsonify, request

from raft_dic_gui.config import DICConfig
from raft_dic_gui.controller import DICProcessor
from server.app import socketio
from server.session import session

processing_bp = Blueprint("processing", __name__)


@processing_bp.route("/configure", methods=["POST"])
def configure():
    """Update DIC processing configuration."""
    data = request.get_json(force=True)

    with session._lock:
        cfg = session.config
        # Apply provided fields
        if "mode" in data:
            cfg.mode = data["mode"]
        if "context_padding" in data:
            cfg.context_padding = int(data["context_padding"])
        if "tile_overlap" in data:
            cfg.tile_overlap = int(data["tile_overlap"])
        if "use_smooth" in data:
            cfg.use_smooth = bool(data["use_smooth"])
        if "sigma" in data:
            cfg.sigma = float(data["sigma"])
        if "safety_factor" in data:
            cfg.safety_factor = float(data["safety_factor"])
        if "p_max_pixels" in data:
            cfg.p_max_pixels = int(data["p_max_pixels"])
        if "device" in data:
            cfg.device = data["device"]
        if "project_root" in data:
            cfg.project_root = data["project_root"]

        valid, err = cfg.validate()

    if not valid:
        return jsonify({"error": err}), 400

    return jsonify({"ok": True})


@processing_bp.route("/run", methods=["POST"])
def run_processing():
    """Start DIC processing in a background thread."""
    if session.processing_active:
        return jsonify({"error": "Processing already active"}), 409

    # Validate prerequisites
    cfg = session.config
    valid, err = cfg.validate()
    if not valid:
        return jsonify({"error": err}), 400

    if session.roi_mask is None or not session.roi_confirmed:
        return jsonify({"error": "ROI must be confirmed before processing"}), 400

    # Set project_root to a temp output dir if not already set
    if not cfg.project_root:
        import os
        import tempfile
        cfg.project_root = os.path.join(
            tempfile.gettempdir(), "raft_dic_output"
        )

    def progress_callback(percent, current, total):
        socketio.emit("processing:progress", {
            "percent": round(percent, 1),
            "current": current,
            "total": total,
        })

    def check_stop():
        """Check stop/pause. Blocks while paused."""
        if session.stop_requested:
            return True
        if session.pause_requested:
            socketio.emit("processing:paused", {})
            # Block until resumed or cancelled
            session.pause_event.wait()
            session.pause_requested = False
            return session.stop_requested
        return False

    def run_in_thread():
        try:
            session.processing_active = True
            session.stop_requested = False
            session.pause_requested = False
            session.pause_event.set()  # Start unblocked

            processor = DICProcessor(
                update_progress_callback=progress_callback,
                check_stop_callback=check_stop,
            )

            results = processor.run(
                session.config,
                session.roi_mask,
                session.roi_rect,
            )

            with session._lock:
                session.displacement_results = results
                session.processor = processor
                session.processing_active = False
                # New displacement results invalidate cached inverse maps
                session.inverse_map_cache.clear()

                # Set up deformed view cache with results
                if results:
                    import os
                    image_paths = [
                        os.path.join(session.image_dir, f)
                        for f in session.image_files
                    ]
                    session.deformed_view_cache.set_image_paths(image_paths)

            if session.stop_requested:
                socketio.emit("processing:complete", {
                    "stopped": True,
                    "num_frames": len(results),
                })
            else:
                socketio.emit("processing:complete", {
                    "stopped": False,
                    "num_frames": len(results),
                })

        except Exception as e:
            session.processing_active = False
            socketio.emit("processing:error", {"error": str(e)})

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()

    return jsonify({"ok": True})


@processing_bp.route("/pause", methods=["POST"])
def pause_processing():
    """Pause processing — blocks the worker thread between frames."""
    if not session.processing_active:
        return jsonify({"error": "No processing active"}), 400

    session.pause_requested = True
    session.pause_event.clear()  # Block the worker
    return jsonify({"ok": True})


@processing_bp.route("/resume", methods=["POST"])
def resume_processing():
    """Resume paused processing."""
    if not session.processing_active:
        return jsonify({"error": "No processing active"}), 400

    session.pause_requested = False
    session.pause_event.set()  # Unblock the worker
    socketio.emit("processing:resumed", {})
    return jsonify({"ok": True})


@processing_bp.route("/stop", methods=["POST"])
def stop_processing():
    """Cancel processing — stop and keep partial results."""
    if not session.processing_active:
        return jsonify({"error": "No processing active"}), 400

    session.stop_requested = True
    session.pause_event.set()  # Unblock if paused, so thread can exit
    return jsonify({"ok": True})


@processing_bp.route("/status", methods=["GET"])
def get_status():
    """Return current processing status."""
    return jsonify({
        "active": session.processing_active,
        "paused": session.pause_requested,
        "has_results": len(session.displacement_results) > 0,
        "num_frames": len(session.displacement_results),
    })
