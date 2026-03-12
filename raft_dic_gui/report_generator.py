"""
Report generator for RAFT-DIC GUI.

Generates self-contained HTML analysis reports with inline CSS and base64 images.
"""

import base64
import io
import logging
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jinja2 import Template

from raft_dic_gui.export_images import _get_component_data

logger = logging.getLogger(__name__)

# All available sections in order
ALL_SECTIONS = [
    "header", "experiment", "parameters", "roi",
    "key_frames", "statistics", "probes",
]


def generate_report(
    session,
    output_path: str,
    sections: Optional[List[str]] = None,
) -> str:
    """
    Generate an HTML analysis report.

    Args:
        session: AppSession object
        output_path: Path to write the HTML file
        sections: List of section names to include (None = all)

    Returns:
        Absolute path of the generated report.
    """
    abs_path = os.path.abspath(output_path)
    active_sections = sections or ALL_SECTIONS

    context = {"sections": active_sections}

    if "header" in active_sections:
        context["header"] = _render_header_section(session)
    if "experiment" in active_sections:
        context["experiment"] = _render_experiment_section(session)
    if "parameters" in active_sections:
        context["parameters"] = _render_parameters_section(session)
    if "roi" in active_sections:
        context["roi"] = _render_roi_section(session)
    if "key_frames" in active_sections:
        context["key_frames"] = _render_key_frames_section(session)
    if "statistics" in active_sections:
        context["statistics"] = _render_statistics_section(session)
    if "probes" in active_sections:
        context["probes"] = _render_probes_section(session)

    html = _get_template().render(**context)

    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info("Report generated: %s", abs_path)
    return abs_path


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _render_header_section(session) -> dict:
    return {
        "title": "RAFT-DIC Analysis Report",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "software": "RAFT-DIC GUI",
        "model": session.config.model_label or os.path.basename(session.config.model_path),
    }


def _render_experiment_section(session) -> dict:
    return {
        "image_dir": session.image_dir,
        "num_images": len(session.image_files),
        "image_width": session.image_width,
        "image_height": session.image_height,
        "mode": session.config.mode,
        "num_displacement_frames": len(session.displacement_results),
        "num_strain_frames": len(session.strain_results),
    }


def _render_parameters_section(session) -> dict:
    config = session.config
    params = [
        ("Processing Mode", config.mode),
        ("Model", config.model_label or os.path.basename(config.model_path)),
        ("Smoothing", f"{'Enabled' if config.use_smooth else 'Disabled'} (sigma={config.sigma})"),
        ("Context Padding", f"{config.context_padding} px"),
        ("Tile Overlap", f"{config.tile_overlap} px"),
        ("Max Pixels", f"{config.p_max_pixels:,}"),
        ("Device", config.device),
    ]
    if config.mode == "incremental":
        params.append(("Key Frame Interval", str(config.key_frame_interval or "N/A")))
        params.append(("Median Filter", "Yes" if config.use_median_filter else "No"))
    return {"params": params}


def _render_roi_section(session) -> dict:
    result = {
        "has_roi": session.roi_rect is not None,
        "roi_rect": session.roi_rect,
        "image": None,
    }
    if session.roi_rect and session.reference_image is not None:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        import cv2
        rgb = cv2.cvtColor(session.reference_image, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        xmin, ymin, xmax, ymax = session.roi_rect
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                              linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
        ax.set_title("Region of Interest")
        ax.axis("off")
        plt.tight_layout()
        result["image"] = _fig_to_base64(fig)
    return result


def _render_key_frames_section(session) -> dict:
    """Render first frame, peak frame, and last frame snapshots."""
    frames = []
    n = len(session.displacement_results)
    if n == 0:
        return {"frames": frames}

    # Find peak magnitude frame
    peak_idx = 0
    peak_val = 0
    for i in range(n):
        data = _get_component_data("magnitude", i, session.displacement_results, session.strain_results)
        if data is not None:
            fmax = np.nanmax(data)
            if fmax > peak_val:
                peak_val = fmax
                peak_idx = i

    indices = [0, peak_idx, n - 1]
    labels = ["First Frame", f"Peak Frame (#{peak_idx + 1})", "Last Frame"]

    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for idx, lbl in zip(indices, labels):
        if idx not in seen:
            seen.add(idx)
            unique.append((idx, lbl))

    for idx, label in unique:
        data = _get_component_data("magnitude", idx, session.displacement_results, session.strain_results)
        if data is None:
            continue

        fig, ax = plt.subplots(figsize=(5, 3.5), dpi=100)
        masked = np.ma.array(data, mask=np.isnan(data))
        cmap = plt.get_cmap("turbo")
        cmap.set_bad(alpha=0)
        im = ax.imshow(masked, cmap=cmap)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"{label} — Magnitude", fontsize=10)
        ax.axis("off")
        plt.tight_layout()
        frames.append({
            "label": label,
            "frame_idx": idx + 1,
            "image": _fig_to_base64(fig),
        })

    return {"frames": frames}


def _render_statistics_section(session) -> dict:
    """Compute min/max/mean/std for displacement components."""
    stats = []
    components = ["u", "v", "magnitude"]
    labels = ["U (horizontal)", "V (vertical)", "Magnitude"]

    n = len(session.displacement_results)
    for comp, label in zip(components, labels):
        all_vals = []
        for i in range(n):
            data = _get_component_data(comp, i, session.displacement_results, session.strain_results)
            if data is not None:
                valid = data[~np.isnan(data)]
                if valid.size > 0:
                    all_vals.append(valid)

        if all_vals:
            combined = np.concatenate(all_vals)
            stats.append({
                "component": label,
                "min": f"{np.min(combined):.4f}",
                "max": f"{np.max(combined):.4f}",
                "mean": f"{np.mean(combined):.4f}",
                "std": f"{np.std(combined):.4f}",
            })

    # Strain stats if available
    for comp_name in session.strain_components[:5]:  # Limit to first 5
        all_vals = []
        for i in range(len(session.strain_results)):
            data = _get_component_data(comp_name, i, session.displacement_results, session.strain_results)
            if data is not None:
                valid = data[~np.isnan(data)]
                if valid.size > 0:
                    all_vals.append(valid)
        if all_vals:
            combined = np.concatenate(all_vals)
            stats.append({
                "component": comp_name,
                "min": f"{np.min(combined):.6f}",
                "max": f"{np.max(combined):.6f}",
                "mean": f"{np.mean(combined):.6f}",
                "std": f"{np.std(combined):.6f}",
            })

    return {"stats": stats}


def _render_probes_section(session) -> dict:
    """Render time series charts for each probe."""
    probes = session.probe_manager.probes
    if not probes:
        return {"probe_charts": []}

    charts = []
    n = len(session.displacement_results)
    if n == 0:
        return {"probe_charts": []}

    # Build u and v data lists
    u_list = []
    v_list = []
    for i in range(n):
        u_data = _get_component_data("u", i, session.displacement_results, session.strain_results)
        v_data = _get_component_data("v", i, session.displacement_results, session.strain_results)
        u_list.append(u_data)
        v_list.append(v_data)

    # Point probes
    from scipy.ndimage import map_coordinates
    roi_rect = session.roi_rect
    offset = (roi_rect[0], roi_rect[1]) if roi_rect else (0, 0)

    for p in probes:
        if p.type != "point":
            continue
        px = p.coords[0] - offset[0]
        py = p.coords[1] - offset[1]
        coords = np.array([[py], [px]])  # (2, 1) for map_coordinates

        u_series = []
        v_series = []
        for i in range(n):
            if u_list[i] is not None:
                u_val = map_coordinates(u_list[i], coords, order=1, mode="constant", cval=np.nan)[0]
                u_series.append(float(u_val))
            else:
                u_series.append(np.nan)
            if v_list[i] is not None:
                v_val = map_coordinates(v_list[i], coords, order=1, mode="constant", cval=np.nan)[0]
                v_series.append(float(v_val))
            else:
                v_series.append(np.nan)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), dpi=100)
        frames_axis = list(range(1, n + 1))
        ax1.plot(frames_axis, u_series, color=p.color, linewidth=1.5)
        ax1.set_xlabel("Frame", fontsize=9)
        ax1.set_ylabel("U displacement", fontsize=9)
        ax1.set_title(f"{p.label} — U", fontsize=10)
        ax1.grid(True, alpha=0.3)

        ax2.plot(frames_axis, v_series, color=p.color, linewidth=1.5)
        ax2.set_xlabel("Frame", fontsize=9)
        ax2.set_ylabel("V displacement", fontsize=9)
        ax2.set_title(f"{p.label} — V", fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        charts.append({
            "label": p.label,
            "type": p.type,
            "coords": f"({p.coords[0]:.0f}, {p.coords[1]:.0f})",
            "image": _fig_to_base64(fig),
        })

    return {"probe_charts": charts}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _fig_to_base64(fig) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ---------------------------------------------------------------------------
# HTML Template
# ---------------------------------------------------------------------------

def _get_template() -> Template:
    template_path = os.path.join(os.path.dirname(__file__), "report_template.html")
    with open(template_path, "r", encoding="utf-8") as f:
        return Template(f.read())
