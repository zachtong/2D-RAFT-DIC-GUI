# Phase 6: Advanced DIC Features & Performance Enhancements

## Overview
Following the v1.0 release with Phase 5, Phase 6 introduced several advanced features for handling complex DIC workflows, including large deformations, visualization options, and result export capabilities.

## New Features

### 1. Incremental Reference Mode (Large Deformations)
**Module**: `raft_dic_gui/incremental.py`

A new processing mode for specimens undergoing large deformations where the optical flow network may lose tracking fidelity over extended deformation ranges.

#### Concept
Instead of tracking all frames relative to a single reference (Frame 1), the incremental mode allows users to specify **key frames** where the reference is updated. The displacements are then accumulated in original coordinates.

#### Key Functions
- `warp_mask_with_holes()`: Warps ROI masks while preserving internal holes using contour-based warping
- `warp_displacement_field()`: Samples incremental displacement at deformed coordinates
- `accumulate_displacement()`: Combines segment displacements, handling coordinate transformations
- `validate_key_frames()`: Validates user-specified key frame selection
- `get_segment_ranges()`: Generates processing segments from key frames

#### Mathematical Basis
When the reference is updated at key frame K:
```
Total displacement = u_accumulated + δu_sampled_at_deformed_positions
```
This ensures accumulated displacements remain in the original coordinate system.

---

### 2. Deformed Frame Visualization
**Module**: `raft_dic_gui/deformed_view_cache.py`

Adds the ability to visualize displacement and strain fields overlaid on the **deformed image** (where material points actually are) rather than the reference image.

#### Features
- **Coordinate Mapping Cache**: Caches reference → deformed coordinate mappings
- **LRU Image Cache**: Efficiently caches loaded deformed frame images (configurable max size)
- **Forward Warping**: Maps data from reference to deformed coordinates using scatter approach
- **Post-Processing Component Cache**: Caches warped strain/displacement data

#### Key Methods in `DeformedViewCache`
- `get_deformed_coords(frame_idx, U, V)`: Compute/retrieve deformed coordinate mapping
- `get_warped_mask(frame_idx, mask, U, V)`: Warp ROI mask to deformed space
- `warp_data_forward(data, U, V, roi_rect, output_shape)`: Warp any data field to deformed coordinates
- `get_deformed_image(frame_idx)`: Load deformed frame with LRU caching
- `invalidate_all()` / `invalidate_coords_and_masks()`: Cache management

#### UI Integration
- Toggle between "Reference" and "Deformed" display modes in the visualization panel
- Preview shows data on top of the corresponding image state
- Useful for validating that displacement tracking is physically correct

---

### 3. Batch Image Export
**Module**: `raft_dic_gui/export_images.py`  
**Dialog**: `raft_dic_gui/views/export_dialog.py`

Export visualization images of displacement/strain fields for all frames to create figures for publications or animations.

#### Features
- **Component Selection**: Export any combination of displacement (U, V, magnitude) and strain components (εxx, εyy, εxy, γmax, von Mises)
- **Frame Range Selection**: Export all frames or a custom range
- **Format Options**: PNG, SVG, PDF
- **Display Modes**: Reference view or Deformed view
- **Visual Customization**: Colormap, opacity, color range, colorbar toggle, title toggle

#### Export Workflow
1. Open "Export Images" dialog from Post-Processing panel
2. Select components to export
3. Configure frame range and visual settings
4. Click Export → Images saved to timestamped folder with metadata JSON

#### Key Functions
- `render_single_frame()`: Render and save a single visualization
- `export_batch_images()`: Main export loop with progress callback
- `_warp_for_deformed()`: Warp data for deformed mode export
- `_save_export_settings()`: Save settings JSON for reproducibility

---

### 4. Performance Optimizations
**Commit**: `e36ce10`

#### Strain Calculation Improvements
- Added **Numba JIT** acceleration for rotation angle calculation
- FFT-accelerated convolution (`scipy.signal.fftconvolve`) for faster strain computation
- Optimized memory usage during large ROI processing

#### UI Responsiveness
- Non-blocking strain calculation with progress updates
- Improved preview panel refresh performance

---

## File Changes Summary

| Commit | Files Modified |
|--------|---------------|
| `8532944` Batch Export | `main_GUI.py`, `export_images.py`, `export_dialog.py`, `post_processing_panel.py` |
| `caa1ff5` Deformed View | `main_GUI.py`, `deformed_view_cache.py`, `processing.py`, `control_panel.py`, `post_processing_panel.py`, `preview_panel.py` |
| `dd7b056` Incremental Mode | `main_GUI.py`, `incremental.py`, `processing.py`, `control_panel.py` |
| `e36ce10` Performance | `main_GUI.py`, `processing.py`, `post_processing_panel.py`, `preview_panel.py`, `setup.py` |

---

## Current Status
These features complete the core DIC workflow, enabling:
1. ✅ Standard fixed-reference DIC
2. ✅ Incremental reference updates for large deformations
3. ✅ Deformed frame visualization
4. ✅ Full-field result export (MAT/NPZ)
5. ✅ Batch image export for publications
6. ✅ Interactive probe analysis (Point/Line/Area)

## Next Steps
- **Probe Data CSV Export**: Export time-series data from virtual extensometers
- **Batch Processing**: Process multiple specimen datasets in sequence
- **Report Generation**: Automated PDF report with key figures
