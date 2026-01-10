"""
Batch Image Export Module

Provides functions for exporting visualization images of displacement and strain fields.
This module contains pure rendering logic with no UI dependencies.
"""

import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable
import cv2


def _log(msg: str):
    """Print debug message with standard prefix."""
    print(f"[ImageExport] {msg}")


def render_single_frame(
    data: np.ndarray,
    background_image: Optional[np.ndarray],
    roi_rect: Optional[Tuple[int, int, int, int]],
    colormap: str,
    alpha: float,
    vmin: Optional[float],
    vmax: Optional[float],
    include_colorbar: bool,
    include_title: bool,
    title: str,
    colorbar_label: str,
    output_path: str,
    format: str
) -> bool:
    """
    Render a single visualization frame and save to file.
    
    Args:
        data: 2D array of values to visualize (can be full image size or ROI-warped)
        background_image: Optional background image (RGB)
        roi_rect: (xmin, ymin, xmax, ymax) for positioning data on background
        colormap: Matplotlib colormap name
        alpha: Overlay transparency (0-1)
        vmin, vmax: Color range (None for auto)
        include_colorbar: Whether to add colorbar
        include_title: Whether to add title
        title: Title text
        colorbar_label: Label for colorbar
        output_path: Full path to save file
        format: 'png', 'jpg', 'pdf', 'svg'
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create figure
        if background_image is not None:
            h, w = background_image.shape[:2]
            dpi = 100
            fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)
            ax.imshow(background_image)
        else:
            h, w = data.shape
            dpi = 100
            fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)
        
        # Create masked array for overlay
        masked_data = np.ma.array(data, mask=np.isnan(data))
        
        # Determine extent based on roi_rect or data shape
        if roi_rect is not None and background_image is not None:
            xmin, ymin, xmax, ymax = roi_rect
            extent = [xmin, xmax, ymax, ymin]  # Note: ymax, ymin for correct orientation
        else:
            extent = None
        
        # Determine color range
        if vmin is None:
            vmin = np.nanmin(data)
        if vmax is None:
            vmax = np.nanmax(data)
        
        # Plot data overlay
        cmap = plt.get_cmap(colormap)
        cmap.set_bad(alpha=0)  # Transparent for NaN
        
        im = ax.imshow(masked_data, cmap=cmap, alpha=alpha, 
                       vmin=vmin, vmax=vmax, extent=extent)
        
        # Add colorbar
        if include_colorbar:
            cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            if colorbar_label:
                cbar.set_label(colorbar_label)
        
        # Add title
        if include_title and title:
            ax.set_title(title)
        
        # Clean up axes
        ax.axis('off')
        
        # Adjust layout
        plt.tight_layout(pad=0.5)
        
        # Save
        fig.savefig(output_path, format=format, 
                   bbox_inches='tight', 
                   facecolor='white' if format in ['jpg', 'jpeg'] else 'auto',
                   dpi=dpi)
        plt.close(fig)
        
        return True
        
    except Exception as e:
        _log(f"ERROR rendering frame: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')
        return False


def export_batch_images(
    output_dir: str,
    components: Dict[str, Dict],  # {'u': {'vmin': -100, 'vmax': 100}, ...}
    frame_range: Tuple[int, int],  # (start, end) 1-indexed inclusive
    displacement_results: List,
    strain_results: List,
    roi_rect: Tuple[int, int, int, int],
    roi_mask: np.ndarray,
    image_loader: Callable[[int], np.ndarray],  # fn(frame_idx) -> image
    settings: Dict,  # colormap, alpha, display_mode, format, include_colorbar, include_title
    deformed_view_cache,  # For deformed mode warp
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> str:
    """
    Export visualization images for selected components and frames.
    
    Args:
        output_dir: Base output directory (will create timestamped subfolder)
        components: Dict of component settings {name: {'vmin': float, 'vmax': float}}
        frame_range: (start, end) 1-indexed inclusive
        displacement_results: List of displacement arrays or paths
        strain_results: List of strain dicts
        roi_rect: (xmin, ymin, xmax, ymax)
        roi_mask: ROI mask array
        image_loader: Function to load background image by frame index
        settings: General export settings
        deformed_view_cache: Cache for deformed mode operations
        progress_callback: Optional callback(current, total, message)
        
    Returns:
        Path to the export directory
    """
    start_time = time.time()
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = os.path.join(output_dir, f"export_{timestamp}")
    os.makedirs(export_dir, exist_ok=True)
    
    _log(f"Export started: {export_dir}")
    _log(f"Settings: {settings}")
    _log(f"Components: {list(components.keys())}")
    _log(f"Frame range: {frame_range[0]} to {frame_range[1]}")
    
    # Extract settings
    colormap = settings.get('colormap', 'turbo')
    alpha = settings.get('alpha', 0.7)
    display_mode = settings.get('display_mode', 'reference')
    format = settings.get('format', 'png')
    include_colorbar = settings.get('include_colorbar', True)
    include_title = settings.get('include_title', False)
    
    # Calculate total operations for progress
    enabled_components = [name for name, cfg in components.items() if cfg.get('enabled', True)]
    frame_start, frame_end = frame_range
    total_frames = frame_end - frame_start + 1
    total_operations = len(enabled_components) * total_frames
    current_op = 0
    saved_count = 0
    
    _log(f"Total operations: {total_operations} ({len(enabled_components)} components x {total_frames} frames)")
    
    # Process each component
    for comp_name in enabled_components:
        comp_config = components[comp_name]
        vmin = comp_config.get('vmin')
        vmax = comp_config.get('vmax')
        
        _log(f"Processing component: {comp_name}")
        _log(f"  Color range: vmin={vmin}, vmax={vmax}")
        
        # Create component subdirectory
        comp_dir = os.path.join(export_dir, comp_name)
        os.makedirs(comp_dir, exist_ok=True)
        
        # Process each frame
        for frame_idx in range(frame_start - 1, frame_end):  # Convert to 0-indexed
            current_op += 1
            
            if progress_callback:
                progress_callback(current_op, total_operations, 
                                f"Exporting {comp_name} frame {frame_idx + 1}...")
            
            try:
                # Get data for this component and frame
                data = _get_component_data(comp_name, frame_idx, 
                                          displacement_results, strain_results)
                if data is None:
                    _log(f"  Frame {frame_idx + 1}: No data available, skipping")
                    continue
                
                _log(f"  Frame {frame_idx + 1}: data shape = {data.shape}")
                
                # Get background image
                if display_mode == 'deformed':
                    bg_frame_idx = frame_idx + 1  # Deformed uses next frame
                    bg_img = image_loader(bg_frame_idx)
                else:
                    bg_img = image_loader(0)  # Reference frame
                
                # Handle deformed mode warping
                display_data = data
                display_roi = roi_rect
                
                if display_mode == 'deformed' and deformed_view_cache is not None:
                    # Need to warp data to deformed coordinates
                    display_data, display_roi = _warp_for_deformed(
                        data, comp_name, frame_idx,
                        displacement_results, roi_rect, 
                        bg_img.shape[:2] if bg_img is not None else None,
                        deformed_view_cache
                    )
                
                # Build output filename
                filename = f"{comp_name}_frame_{frame_idx + 1:03d}.{format}"
                output_path = os.path.join(comp_dir, filename)
                
                # Build title
                title = f"{comp_name.upper()} - Frame {frame_idx + 1}" if include_title else ""
                
                # Determine colorbar label
                if comp_name in ['u', 'v']:
                    colorbar_label = f"{comp_name} [px]"
                else:
                    colorbar_label = comp_name
                
                # Render and save
                success = render_single_frame(
                    data=display_data,
                    background_image=bg_img,
                    roi_rect=display_roi,
                    colormap=colormap,
                    alpha=alpha,
                    vmin=vmin,
                    vmax=vmax,
                    include_colorbar=include_colorbar,
                    include_title=include_title,
                    title=title,
                    colorbar_label=colorbar_label,
                    output_path=output_path,
                    format=format
                )
                
                if success:
                    saved_count += 1
                    _log(f"  Saved: {filename}")
                    
            except Exception as e:
                _log(f"  ERROR on frame {frame_idx + 1}: {e}")
                import traceback
                traceback.print_exc()
    
    # Save export settings
    settings_path = os.path.join(export_dir, "export_settings.json")
    _save_export_settings(settings_path, components, frame_range, settings)
    
    elapsed = time.time() - start_time
    _log(f"Export complete: {saved_count} images in {elapsed:.1f}s")
    _log(f"Output directory: {export_dir}")
    
    if progress_callback:
        progress_callback(total_operations, total_operations, "Export complete!")
    
    return export_dir


def _get_component_data(
    comp_name: str,
    frame_idx: int,
    displacement_results: List,
    strain_results: List
) -> Optional[np.ndarray]:
    """Get data array for a specific component and frame."""
    try:
        if comp_name in ['u', 'v']:
            # Displacement component
            if frame_idx >= len(displacement_results):
                return None
            d = displacement_results[frame_idx]
            if isinstance(d, str):
                d = np.load(d)
            idx = 0 if comp_name == 'u' else 1
            return d[..., idx]
        else:
            # Strain component
            if frame_idx >= len(strain_results):
                return None
            s = strain_results[frame_idx]
            if s is None or comp_name not in s:
                return None
            return s[comp_name]
    except Exception as e:
        _log(f"ERROR getting component data: {e}")
        return None


def _warp_for_deformed(
    data: np.ndarray,
    comp_name: str,
    frame_idx: int,
    displacement_results: List,
    roi_rect: Tuple,
    output_shape: Tuple,
    deformed_view_cache
) -> Tuple[np.ndarray, Optional[Tuple]]:
    """
    Warp data to deformed coordinates if needed.
    
    Returns:
        (warped_data, display_roi) - roi is None for full-image warped data
    """
    try:
        if frame_idx >= len(displacement_results):
            return data, roi_rect
        
        # Get displacement for this frame
        disp = displacement_results[frame_idx]
        if isinstance(disp, str):
            disp = np.load(disp)
        u_crop = disp[..., 0]
        v_crop = disp[..., 1]
        
        # Check if data shape matches displacement
        data_to_warp = data
        if data.shape != u_crop.shape:
            # Upsample (strain may be downsampled)
            target_h, target_w = u_crop.shape
            _log(f"  Upsampling {comp_name}: {data.shape} -> ({target_h}, {target_w})")
            
            nan_mask = np.isnan(data)
            data_filled = np.nan_to_num(data, nan=0.0)
            data_upsampled = cv2.resize(data_filled.astype(np.float32), 
                                       (target_w, target_h),
                                       interpolation=cv2.INTER_LINEAR)
            nan_mask_upsampled = cv2.resize(nan_mask.astype(np.uint8), 
                                           (target_w, target_h),
                                           interpolation=cv2.INTER_NEAREST) > 0
            data_upsampled[nan_mask_upsampled] = np.nan
            data_to_warp = data_upsampled
        
        # Warp using cache
        if output_shape is None:
            output_shape = (3648, 5472)  # Fallback
        
        _log(f"  Warping to deformed coordinates...")
        warped = deformed_view_cache.warp_data_forward(
            data_to_warp, u_crop, v_crop, roi_rect, output_shape
        )
        
        return warped, None  # None roi means full image
        
    except Exception as e:
        _log(f"  ERROR in warp: {e}")
        return data, roi_rect


def _save_export_settings(
    path: str,
    components: Dict,
    frame_range: Tuple,
    settings: Dict
):
    """Save export settings to JSON for reproducibility."""
    export_info = {
        'timestamp': datetime.now().isoformat(),
        'frame_range': list(frame_range),
        'components': components,
        'settings': settings
    }
    
    try:
        with open(path, 'w') as f:
            json.dump(export_info, f, indent=2)
        _log(f"Settings saved: {path}")
    except Exception as e:
        _log(f"ERROR saving settings: {e}")
