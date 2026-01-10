"""
Incremental Reference Processing Utilities

This module provides functions for incremental DIC processing where the 
reference image is updated at user-specified key frames to handle large
deformations.

Key functions:
- warp_mask_with_holes: Warp ROI mask preserving internal holes
- accumulate_displacement: Combine segment displacements in original coordinates
- validate_key_frames: Validate key frame selection
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional


def warp_mask_with_holes(mask: np.ndarray, displacement: np.ndarray) -> np.ndarray:
    """
    Warp a binary mask according to a displacement field, preserving holes.
    
    Uses contour-based warping to maintain the topology of complex masks
    with internal holes (e.g., masks with cutouts or complex geometry).
    
    Args:
        mask: (H, W) boolean or uint8 mask, True/1 = valid region
        displacement: (H, W, 2) displacement field (u, v)
        
    Returns:
        warped_mask: (H, W) boolean mask after applying displacement
    """
    H, W = mask.shape
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    
    # Find contours with hierarchy (RETR_TREE captures holes)
    contours, hierarchy = cv2.findContours(
        mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    
    if len(contours) == 0:
        return np.zeros((H, W), dtype=bool)
    
    # Warp each contour point
    def warp_contour(contour: np.ndarray) -> np.ndarray:
        """Warp contour points using displacement field."""
        warped_pts = []
        for pt in contour:
            x, y = int(pt[0][0]), int(pt[0][1])
            
            # Bounds check
            if 0 <= x < W and 0 <= y < H:
                dx = displacement[y, x, 0]
                dy = displacement[y, x, 1]
                
                if not np.isnan(dx) and not np.isnan(dy):
                    new_x = x + dx
                    new_y = y + dy
                else:
                    # Keep original position if displacement is NaN
                    new_x, new_y = x, y
            else:
                new_x, new_y = x, y
            
            warped_pts.append([[new_x, new_y]])
        
        return np.array(warped_pts, dtype=np.float32)
    
    # Separate outer contours and holes
    outer_contours = []
    hole_contours = []
    
    if hierarchy is not None:
        for i, cnt in enumerate(contours):
            warped_cnt = warp_contour(cnt)
            # Clip to image bounds
            warped_cnt = np.clip(warped_cnt, 0, [[W-1, H-1]])
            warped_cnt = warped_cnt.astype(np.int32)
            
            # hierarchy: [Next, Previous, First_Child, Parent]
            parent = hierarchy[0][i][3]
            if parent == -1:
                # This is an outer contour
                outer_contours.append(warped_cnt)
            else:
                # This is a hole (has parent)
                hole_contours.append(warped_cnt)
    
    # Draw result
    new_mask = np.zeros((H, W), dtype=np.uint8)
    
    # Fill outer contours
    for cnt in outer_contours:
        if len(cnt) >= 3:  # Need at least 3 points for polygon
            cv2.drawContours(new_mask, [cnt], 0, 255, -1)
    
    # Subtract holes
    for cnt in hole_contours:
        if len(cnt) >= 3:
            cv2.drawContours(new_mask, [cnt], 0, 0, -1)
    
    # Morphological cleanup to fill small gaps from warping
    kernel = np.ones((3, 3), np.uint8)
    new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, kernel)
    
    return new_mask > 0


def warp_displacement_field(delta_u: np.ndarray, 
                            accumulated_u: np.ndarray) -> np.ndarray:
    """
    Sample incremental displacement at deformed coordinates.
    
    When the reference is updated, the new incremental displacement delta_u
    is measured in the deformed coordinate system. To accumulate it with
    previous displacement, we need to sample delta_u at the deformed positions.
    
    Args:
        delta_u: (H, W, 2) incremental displacement (in deformed coordinates)
        accumulated_u: (H, W, 2) accumulated displacement so far (in original coords)
        
    Returns:
        delta_u_sampled: (H, W, 2) delta_u sampled at deformed positions
    """
    H, W = delta_u.shape[:2]
    
    # Create coordinate grids
    x, y = np.meshgrid(np.arange(W, dtype=np.float32), 
                        np.arange(H, dtype=np.float32))
    
    # Compute deformed coordinates
    x_def = x + accumulated_u[..., 0].astype(np.float32)
    y_def = y + accumulated_u[..., 1].astype(np.float32)
    
    # Replace NaN in coordinates with -1 (will be out of bounds)
    x_def_safe = np.where(np.isnan(x_def), -1, x_def)
    y_def_safe = np.where(np.isnan(y_def), -1, y_def)
    
    # Create validity mask for delta_u (1 = valid, 0 = NaN)
    delta_valid = (~np.isnan(delta_u[..., 0])).astype(np.float32)
    
    # Sample delta_u at deformed positions using bilinear interpolation
    delta_u_sampled = np.zeros_like(delta_u)
    
    for i in range(2):
        # Replace NaN with 0 for remap
        channel = delta_u[..., i].astype(np.float32)
        channel = np.where(np.isnan(channel), 0, channel)
        
        sampled = cv2.remap(
            channel,
            x_def_safe.astype(np.float32),
            y_def_safe.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        delta_u_sampled[..., i] = sampled
    
    # Also remap the validity mask to know which sampled values are reliable
    valid_sampled = cv2.remap(
        delta_valid,
        x_def_safe.astype(np.float32),
        y_def_safe.astype(np.float32),
        interpolation=cv2.INTER_NEAREST,  # Use nearest to avoid blending valid/invalid
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    # Mark invalid: out-of-bounds, or sampled from NaN region of delta_u
    out_of_bounds = (x_def < 0) | (x_def >= W) | (y_def < 0) | (y_def >= H)
    invalid_sample = (valid_sampled < 0.5)  # Sampled from NaN region
    u_prev_nan = np.isnan(accumulated_u[..., 0])  # Previous displacement was NaN
    
    invalid_mask = out_of_bounds | invalid_sample | u_prev_nan
    
    delta_u_sampled[invalid_mask, 0] = np.nan
    delta_u_sampled[invalid_mask, 1] = np.nan
    
    return delta_u_sampled


def accumulate_displacement(u_prev: np.ndarray, 
                           delta_u: np.ndarray,
                           debug: bool = True) -> np.ndarray:
    """
    Accumulate incremental displacement onto previous accumulated displacement.
    
    Handles coordinate transformation: delta_u is sampled at deformed positions
    before being added to the accumulated displacement.
    
    Args:
        u_prev: (H, W, 2) previous accumulated displacement (in original coords)
        delta_u: (H, W, 2) incremental displacement from new reference
        debug: Whether to print debug info
        
    Returns:
        u_total: (H, W, 2) total displacement in original coordinates
    """
    if debug:
        print(f"[Accumulate] Input shapes: u_prev={u_prev.shape}, delta_u={delta_u.shape}")
        
        # Stats for u_prev
        valid_prev = ~np.isnan(u_prev[..., 0])
        if np.any(valid_prev):
            print(f"[Accumulate] u_prev stats: valid_pixels={np.sum(valid_prev)}, "
                  f"u_range=[{np.nanmin(u_prev[...,0]):.2f}, {np.nanmax(u_prev[...,0]):.2f}], "
                  f"v_range=[{np.nanmin(u_prev[...,1]):.2f}, {np.nanmax(u_prev[...,1]):.2f}]")
        else:
            print(f"[Accumulate] u_prev: all NaN!")
        
        # Stats for delta_u
        valid_delta = ~np.isnan(delta_u[..., 0])
        if np.any(valid_delta):
            print(f"[Accumulate] delta_u stats: valid_pixels={np.sum(valid_delta)}, "
                  f"u_range=[{np.nanmin(delta_u[...,0]):.2f}, {np.nanmax(delta_u[...,0]):.2f}], "
                  f"v_range=[{np.nanmin(delta_u[...,1]):.2f}, {np.nanmax(delta_u[...,1]):.2f}]")
        else:
            print(f"[Accumulate] delta_u: all NaN!")
    
    # Sample delta_u at deformed positions
    delta_u_sampled = warp_displacement_field(delta_u, u_prev)
    
    if debug:
        valid_sampled = ~np.isnan(delta_u_sampled[..., 0])
        if np.any(valid_sampled):
            print(f"[Accumulate] After warp: valid_pixels={np.sum(valid_sampled)} (lost {np.sum(valid_delta)-np.sum(valid_sampled)} pixels), "
                  f"u_range=[{np.nanmin(delta_u_sampled[...,0]):.2f}, {np.nanmax(delta_u_sampled[...,0]):.2f}], "
                  f"v_range=[{np.nanmin(delta_u_sampled[...,1]):.2f}, {np.nanmax(delta_u_sampled[...,1]):.2f}]")
        else:
            print(f"[Accumulate] After warp: all NaN! (all pixels lost)")
    
    # Accumulate
    u_total = u_prev + delta_u_sampled
    
    if debug:
        valid_total = ~np.isnan(u_total[..., 0])
        if np.any(valid_total):
            print(f"[Accumulate] Result: valid_pixels={np.sum(valid_total)}, "
                  f"u_range=[{np.nanmin(u_total[...,0]):.2f}, {np.nanmax(u_total[...,0]):.2f}], "
                  f"v_range=[{np.nanmin(u_total[...,1]):.2f}, {np.nanmax(u_total[...,1]):.2f}]")
        else:
            print(f"[Accumulate] Result: all NaN!")
    
    return u_total


def validate_key_frames(key_frames: List[int], 
                        total_frames: int) -> Tuple[bool, str]:
    """
    Validate key frame selection for incremental processing.
    
    Requirements:
    - Must include frame 1 (or 0 if 0-indexed)
    - All frames must be within valid range
    - Must be sorted in ascending order
    
    Args:
        key_frames: List of frame indices (1-indexed, frame 1 = reference)
        total_frames: Total number of frames in sequence
        
    Returns:
        (is_valid, error_message): Tuple of validation result and message
    """
    if not key_frames:
        return False, "No key frames selected. At least frame 1 must be selected."
    
    # Sort key frames
    key_frames_sorted = sorted(key_frames)
    
    # Check frame 1 is included
    if key_frames_sorted[0] != 1:
        return False, "Frame 1 must be selected as the first key frame (initial reference)."
    
    # Check all frames are in valid range
    for f in key_frames_sorted:
        if f < 1 or f > total_frames:
            return False, f"Frame {f} is out of range (1-{total_frames})."
    
    # Check for duplicates
    if len(key_frames_sorted) != len(set(key_frames_sorted)):
        return False, "Duplicate key frames detected."
    
    return True, "Key frames valid."


def get_segment_ranges(key_frames: List[int], 
                       total_frames: int) -> List[Tuple[int, int, int]]:
    """
    Get processing segment ranges from key frames.
    
    Args:
        key_frames: List of key frame indices (1-indexed, sorted)
        total_frames: Total number of frames
        
    Returns:
        List of (ref_frame, start_frame, end_frame) tuples for each segment
    """
    key_frames_sorted = sorted(key_frames)
    segments = []
    
    for i, ref_frame in enumerate(key_frames_sorted):
        start_frame = ref_frame + 1  # First frame to process in this segment
        
        if i < len(key_frames_sorted) - 1:
            end_frame = key_frames_sorted[i + 1]  # Up to (and including) next key frame
        else:
            end_frame = total_frames  # Last segment goes to end
        
        if start_frame <= end_frame:
            segments.append((ref_frame, start_frame, end_frame))
    
    return segments
