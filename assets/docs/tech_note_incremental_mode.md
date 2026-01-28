# Technical Note: Incremental Reference Mode

## Purpose
Handle large deformations where RAFT tracking may lose accuracy over extended deformation ranges. By updating the reference image at user-defined key frames, tracking fidelity is maintained.

## Algorithm

### Processing Segments
Given key frames [1, K₁, K₂, ...] and N total frames:
- **Segment 1**: Frames 1 → K₁ (reference = Frame 1)
- **Segment 2**: Frames K₁ → K₂ (reference = Frame K₁)
- **Segment n**: Frames Kₙ₋₁ → N (reference = Frame Kₙ₋₁)

### Displacement Accumulation
For each segment after the first:

```python
# Previous accumulated displacement (in original coords)
u_prev = accumulated_displacement[key_frame]

# Incremental displacement from new reference
# IMPORTANT: δu is measured in deformed coordinates of the key frame
delta_u = raft_inference(key_frame_image, target_image)

# Sample δu at deformed positions (warp to original coordinates)
delta_u_sampled = sample_at_deformed_positions(delta_u, u_prev)

# Accumulate
u_total = u_prev + delta_u_sampled
```

### Coordinate Transformation
The incremental displacement `δu` is in the deformed coordinate system. To add it to the accumulated displacement, we must sample it at the correct positions:

```
x'(x,y) = x + u_prev(x,y)
y'(x,y) = y + v_prev(x,y)

δu_sampled(x,y) = δu(x', y')  # Bilinear interpolation
```

### Mask Warping
ROI masks must be warped to track with the deformation:
1. Extract contours (including internal holes)
2. Warp contour vertices using displacement field
3. Reconstruct mask via `cv2.drawContours`

## UI Integration
- **Control Panel**: Key frame input field (comma-separated indices)
- **Validation**: Frame 1 must be included; frames sorted ascending

## Use Cases
1. **Tensile tests with necking**: Update reference when strain localizes
2. **Fatigue testing**: Update reference every N cycles
3. **High-strain materials** (rubbers, polymers): Multiple key frames
