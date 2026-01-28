# Technical Note: Deformed Frame Visualization

## Purpose
Overlay displacement/strain data on the **deformed image** instead of the reference image, showing where material points actually move to during deformation.

## Coordinate Systems

### Reference Coordinates
- Original pixel grid (x, y) on reference image
- Displacement U(x,y), V(x,y) are naturally defined here

### Deformed Coordinates  
- Where each reference point ends up: (x', y') = (x + U, y + V)
- Data must be warped via forward mapping to display here

## Forward Warping Algorithm

```python
def warp_data_forward(data, U, V, roi_rect, output_shape):
    """Scatter reference data to deformed positions."""
    
    # For each valid reference pixel
    for (x, y) in valid_roi_pixels:
        # Compute deformed position
        x_def = x + U[y, x]
        y_def = y + V[y, x]
        
        # Round to nearest output pixel
        i, j = round(y_def), round(x_def)
        
        # Accumulate (handle multiple points mapping to same pixel)
        output[i, j] += data[y, x]
        count[i, j] += 1
    
    # Average where multiple values accumulated
    output /= count  # (where count > 0)
```

## Caching Strategy

### Cache Types
| Cache | Key | Value | Purpose |
|-------|-----|-------|---------|
| `coord_cache` | frame_idx | {x_def, y_def} | Avoid recomputing coordinate grids |
| `mask_cache` | frame_idx | warped_mask | Cached warped ROI boundary |
| `warped_disp_cache` | frame_idx | {u_warped, v_warped} | Warped displacement fields |
| `post_component_cache` | (frame_idx, comp) | warped_data | LRU cache for strain components |
| `image_cache` | frame_idx | RGB array | LRU cache for loaded images |

### Invalidation Rules
- **New images loaded**: `invalidate_all()`
- **Processing re-run**: `invalidate_all()` 
- **ROI changed**: `invalidate_all()`
- **Strain recalculated**: `invalidate_coords_and_masks()` (keep image cache)

## UI Behavior
- Display mode toggle: "Reference" | "Deformed"
- Seamless switching between modes (cached data reused)
- Deformed mode shows data overlaid on corresponding frame image
