"""Velocity and displacement magnitude calculations for RAFT-DIC."""

import numpy as np


def calculate_displacement_magnitude(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Calculate displacement magnitude: M = sqrt(u^2 + v^2)

    Args:
        u: Horizontal displacement component (H, W)
        v: Vertical displacement component (H, W)

    Returns:
        Magnitude array (H, W)
    """
    return np.sqrt(u**2 + v**2)


def calculate_velocity_field(u_curr: np.ndarray, v_curr: np.ndarray,
                            u_prev: np.ndarray, v_prev: np.ndarray,
                            fps: float = 1.0) -> np.ndarray:
    """
    Calculate velocity magnitude from frame-to-frame displacement difference.

    Args:
        u_curr, v_curr: displacement at current frame (H, W)
        u_prev, v_prev: displacement at previous frame (H, W)
        fps: frame rate (Hz), used to convert to physical velocity units

    Returns:
        Velocity magnitude = sqrt(du^2 + dv^2) * fps (H, W)
    """
    du = u_curr - u_prev
    dv = v_curr - v_prev
    return np.sqrt(du**2 + dv**2) * fps
