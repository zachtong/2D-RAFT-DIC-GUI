# Per-Frame ROI System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Allow users to assign ROI masks to any frame (not just Frame 1), with required masks for reference frames and optional masks for non-reference frames. Deformed view rendering uses per-frame masks as ground-truth geometry boundaries.

**Architecture:** Extend `AppSession` with `per_frame_rois: Dict[int, np.ndarray]` (0-indexed). All existing ROI endpoints gain an optional `frame_idx` parameter (default 0 = backward compatible). Deformed warp pipeline accepts an optional `deformed_mask` to clip validity. Frontend adds a frame selector to the ROI page and per-frame status indicators.

**Tech Stack:** Python/Flask (backend), React/TypeScript/Zustand (frontend), OpenCV, numpy

**Key Constraint:** Zero regression — all existing single-ROI workflows must continue working identically.

---

## Phase 1: Backend — Per-Frame ROI Storage

### Task 1: Add `per_frame_rois` to AppSession

**Files:**
- Modify: `server/session.py`

**Step 1: Add field and sync logic**

In `AppSession` dataclass, add after line 55 (`roi_confirmed`):

```python
# Per-frame ROI masks: 0-based frame index -> bool mask (H, W)
# per_frame_rois[0] is always kept in sync with roi_mask (Frame 1)
per_frame_rois: Dict[int, np.ndarray] = field(default_factory=dict)
```

**Step 2: Update `reset()` method**

In `reset()` (line 148), add after `self.roi_confirmed = False` (line 157):

```python
self.per_frame_rois = {}
```

**Step 3: Verify no regression**

Run: `cd server && python -m pytest tests/ -x -q`
Expected: All existing tests pass unchanged.

---

### Task 2: Extend ROI endpoints with `frame_idx` support

**Files:**
- Modify: `server/routes/roi.py`

The core change: every shape/import/clear/mask endpoint accepts an optional `frame_idx` query param or JSON field. When `frame_idx=0` (default), behavior is identical to current code (operates on `session.roi_mask`). When `frame_idx>0`, operates on `session.per_frame_rois[frame_idx]` instead.

**Step 1: Add helper to get/set per-frame mask**

Add after `_apply_shape()` (after line 54):

```python
def _get_frame_mask(frame_idx: int) -> Optional[np.ndarray]:
    """Get the ROI mask for a specific frame index (0-based)."""
    if frame_idx == 0:
        return session.roi_mask
    return session.per_frame_rois.get(frame_idx)


def _set_frame_mask(frame_idx: int, mask: np.ndarray):
    """Set the ROI mask for a specific frame (and sync frame 0 with roi_mask)."""
    session.per_frame_rois[frame_idx] = mask
    if frame_idx == 0:
        session.roi_mask = mask
        _update_rect()


def _ensure_frame_mask(frame_idx: int) -> bool:
    """Ensure a mask buffer exists for the given frame. Returns False if no image loaded."""
    if session.image_height == 0 or session.image_width == 0:
        if session.reference_image is None:
            return False
    h = session.image_height or session.reference_image.shape[0]
    w = session.image_width or session.reference_image.shape[1]
    if frame_idx == 0:
        if session.roi_mask is None or session.roi_mask.shape != (h, w):
            session.roi_mask = np.zeros((h, w), dtype=bool)
        return True
    if frame_idx not in session.per_frame_rois:
        session.per_frame_rois[frame_idx] = np.zeros((h, w), dtype=bool)
    return True


def _apply_shape_for_frame(new_mask_bool: np.ndarray, mode: str, frame_idx: int):
    """Apply a shape to a specific frame's mask."""
    with session._lock:
        if not _ensure_frame_mask(frame_idx):
            return jsonify({"error": "No reference image loaded"}), 400

        if frame_idx == 0:
            mask = session.roi_mask
        else:
            mask = session.per_frame_rois[frame_idx]

        if mode == "cut":
            result = mask & ~new_mask_bool
        else:
            result = mask | new_mask_bool

        _set_frame_mask(frame_idx, result)

    area = int(result.sum())
    return jsonify({"rect": session.roi_rect, "area_px": area, "frame_idx": frame_idx})
```

**Step 2: Update existing endpoints to accept `frame_idx`**

For `/polygon`, `/rectangle`, `/circle`: extract `frame_idx` from JSON body, default 0.

Example for polygon (apply same pattern to rectangle and circle):

```python
@roi_bp.route("/polygon", methods=["POST"])
def add_polygon():
    data = request.get_json(force=True)
    points = data.get("points", [])
    mode = data.get("mode", "add")
    frame_idx = int(data.get("frame_idx", 0))

    if not points or len(points) < 3:
        return jsonify({"error": "Need at least 3 points"}), 400
    if session.reference_image is None and session.image_height == 0:
        return jsonify({"error": "No images loaded"}), 400

    h = session.image_height or session.reference_image.shape[0]
    w = session.image_width or session.reference_image.shape[1]
    pts = np.array(points, np.int32)
    new_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(new_mask, [pts], 1)

    return _apply_shape_for_frame(new_mask.astype(bool), mode, frame_idx)
```

**Step 3: Update `/import` to accept `frame_idx`**

Add `frame_idx = int(data.get("frame_idx", 0))` to `import_mask()`.
Use `_set_frame_mask(frame_idx, binary.astype(bool))` instead of directly setting `session.roi_mask`.

**Step 4: Update `/mask` GET to accept `frame_idx` query param**

```python
@roi_bp.route("/mask", methods=["GET"])
def get_mask():
    frame_idx = request.args.get("frame_idx", 0, type=int)
    mask = _get_frame_mask(frame_idx)
    if mask is None:
        # Fallback: frame 0 for reference frames
        if frame_idx > 0:
            mask = session.roi_mask
        if mask is None:
            return jsonify({"error": "No ROI mask exists"}), 404
    # ... rest of rendering same as before but using `mask` variable
```

**Step 5: Add per-frame clear endpoint**

```python
@roi_bp.route("/frame/<int:frame_idx>", methods=["DELETE"])
def clear_frame_roi(frame_idx: int):
    """Clear ROI for a specific frame."""
    with session._lock:
        if frame_idx == 0:
            session.roi_mask = None
            session.roi_rect = None
            session.roi_confirmed = False
        session.per_frame_rois.pop(frame_idx, None)
    return jsonify({"ok": True})
```

**Step 6: Add ROI status endpoint**

```python
@roi_bp.route("/frames/status", methods=["GET"])
def frames_roi_status():
    """Return which frames have ROI masks."""
    return jsonify({
        "total_frames": len(session.image_files),
        "frames_with_roi": sorted(session.per_frame_rois.keys()),
        "frame_0_confirmed": session.roi_confirmed,
    })
```

**Step 7: Verify backward compatibility**

When no `frame_idx` is provided (or `frame_idx=0`), all endpoints behave exactly as before. The existing `_ensure_mask()`, `_update_rect()`, `_apply_shape()` functions remain untouched for backward compatibility — new code paths use the new `_*_for_frame` variants.

Run: `cd server && python -m pytest tests/ -x -q`

---

### Task 3: Batch import endpoint

**Files:**
- Modify: `server/routes/roi.py`

**Step 1: Add batch import route**

```python
@roi_bp.route("/frames/batch-import", methods=["POST"])
def batch_import_rois():
    """Batch import ROI masks from a folder.

    JSON body:
    - folder: str — path to folder containing mask images
    - strategy: "auto_match" | "sequential" — matching strategy
      auto_match: extract last number from filename -> frame index
      sequential: 1st file -> frame 0, 2nd -> frame 1, ...

    Returns: {imported: int, total_files: int, assignments: {frame_idx: filename}}
    """
    data = request.get_json(force=True)
    folder = data.get("folder", "").strip()
    strategy = data.get("strategy", "sequential")

    if not folder or not os.path.isdir(folder):
        return jsonify({"error": "Invalid folder path"}), 400

    h = session.image_height or (session.reference_image.shape[0] if session.reference_image is not None else 0)
    w = session.image_width or (session.reference_image.shape[1] if session.reference_image is not None else 0)
    if h == 0 or w == 0:
        return jsonify({"error": "No images loaded"}), 400

    from raft_dic_gui.mask_loader import discover_masks, _natural_sort_key, MASK_EXTENSIONS
    from pathlib import Path

    mask_path = Path(folder)
    mask_files = sorted(
        (f for f in mask_path.iterdir()
         if f.is_file() and f.suffix.lower() in MASK_EXTENSIONS),
        key=lambda f: _natural_sort_key(f.name),
    )

    assignments = {}
    imported = 0

    for file_idx, mask_file in enumerate(mask_files):
        # Determine target frame index
        if strategy == "auto_match":
            import re
            numbers = re.findall(r"\d+", mask_file.stem)
            if not numbers:
                continue
            target_idx = int(numbers[-1])
            # Convert 1-based to 0-based if number > 0
            if target_idx > 0:
                target_idx -= 1
        else:  # sequential
            target_idx = file_idx

        if target_idx >= len(session.image_files):
            continue

        # Load mask
        mask_img = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            continue
        mask_img = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_bool = (mask_img > 127).astype(bool)

        with session._lock:
            _set_frame_mask(target_idx, mask_bool)

        assignments[target_idx] = mask_file.name
        imported += 1

    # If frame 0 was imported, confirm ROI
    if 0 in assignments:
        with session._lock:
            session.roi_confirmed = True

    return jsonify({
        "imported": imported,
        "total_files": len(mask_files),
        "assignments": {str(k): v for k, v in sorted(assignments.items())},
    })
```

**Step 2: Add necessary imports at top of roi.py**

```python
import os
```

Run: `cd server && python -m pytest tests/ -x -q`

---

## Phase 2: Backend — Deformed View Mask Application

### Task 4: Extend deformed_warp to accept `deformed_mask`

**Files:**
- Modify: `server/deformed_warp.py`

**Step 1: Update `compute_inverse_map()` signature and logic**

Add optional parameter `deformed_frame_mask: Optional[np.ndarray] = None`.

After the validity computation (around line 292), before the return statement (line 294):

```python
    # Apply user-provided deformed-frame mask (ground-truth geometry)
    if deformed_frame_mask is not None:
        mask_crop = deformed_frame_mask[out_y0:out_y1, out_x0:out_x1]
        if mask_crop.shape == validity.shape:
            validity = validity & mask_crop
```

**Step 2: Update `_warp_at_viewport_scale()` to accept and pass `deformed_mask`**

Add `deformed_mask=None` parameter. When `deformed_mask` is not None and scale < 1.0, downsample the mask before passing:

```python
def _warp_at_viewport_scale(
    data, U, V, roi_rect, image_shape, scale,
    needs_upsample=False, roi_h=0, roi_w=0,
    deformed_mask=None,  # NEW
):
    # ... existing downsampling code ...

    # Downsample deformed mask if provided
    ds_deformed_mask = None
    if deformed_mask is not None:
        ds_deformed_mask = cv2.resize(
            deformed_mask.astype(np.uint8), (ds_w, ds_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    inv_map = compute_inverse_map(
        U_ds, V_ds, ds_roi_rect, (ds_h, ds_w),
        deformed_frame_mask=ds_deformed_mask,  # NEW
    )
    return warp_data_inverse(data_ds, inv_map, (ds_h, ds_w))
```

**Step 3: Update `get_warped_full_data()` to accept and pass `deformed_mask`**

```python
def get_warped_full_data(
    data, frame_idx, U, V, roi_rect, image_shape, cache,
    needs_upsample=False, roi_h=0, roi_w=0,
    vw=0, vh=0,
    deformed_mask=None,  # NEW
):
    # ... existing code ...

    # Fast path: viewport scale
    if scale < 1.0:
        return _warp_at_viewport_scale(
            data, U, V, roi_rect, image_shape, scale,
            needs_upsample, roi_h, roi_w,
            deformed_mask=deformed_mask,  # NEW
        )

    # Full-resolution path
    # ... existing upsample code ...

    inv_map = cache.get(frame_idx)
    if inv_map is None:
        inv_map = compute_inverse_map(
            U, V, roi_rect, image_shape,
            deformed_frame_mask=deformed_mask,  # NEW
        )
        inv_map.frame_idx = frame_idx
        cache.put(frame_idx, inv_map)

    return warp_data_inverse(data, inv_map, image_shape)
```

**Important cache note:** When `deformed_mask` is provided, the cached inverse map already includes the mask clipping. If the user changes masks between requests for the same frame, the cache must be invalidated. The simplest approach: if `deformed_mask is not None`, skip cache lookup (compute fresh each time). This is safe because mask-clipped renders are less common and already fast at viewport scale.

Revised cache logic:

```python
    inv_map = None if deformed_mask is not None else cache.get(frame_idx)
    if inv_map is None:
        inv_map = compute_inverse_map(
            U, V, roi_rect, image_shape,
            deformed_frame_mask=deformed_mask,
        )
        inv_map.frame_idx = frame_idx
        if deformed_mask is None:
            cache.put(frame_idx, inv_map)
```

Run: `cd server && python -m pytest tests/ -x -q`

---

### Task 5: Render routes pass per-frame masks

**Files:**
- Modify: `server/routes/displacement.py` (lines 162-173)
- Modify: `server/routes/strain.py` (`_place_strain_in_full_image`, lines 310-318)
- Modify: `server/routes/arrows.py` (no change needed — arrows don't use `get_warped_full_data`)

**Step 1: displacement.py — pass deformed_mask**

In `render_frame()` at line 162-173, change to:

```python
    if background == "deformed" and rect:
        from server.deformed_warp import get_warped_full_data
        disp = disp_results[idx]
        U, V = disp[:, :, 0], disp[:, :, 1]
        # User mask for the deformed frame (0-based: displacement idx 0 = frame 2 = index 1)
        deformed_frame_idx = idx + 1
        deformed_mask = session.per_frame_rois.get(deformed_frame_idx)
        full_data = get_warped_full_data(
            data=disp_data, frame_idx=idx,
            U=U, V=V,
            roi_rect=rect,
            image_shape=(h, w),
            cache=session.inverse_map_cache,
            vw=vw, vh=vh,
            deformed_mask=deformed_mask,
        )
```

Apply same pattern in `download_frame()` (around line 265-275).

**Step 2: strain.py — pass deformed_mask in `_place_strain_in_full_image()`**

```python
def _place_strain_in_full_image(
    strain_data, idx, h, w, background="reference", vw=0, vh=0,
):
    rect = _active_rect()
    if background == "deformed" and rect:
        from server.deformed_warp import get_warped_full_data
        x0, y0, x1, y1 = rect
        roi_h, roi_w = y1 - y0, x1 - x0
        sh, sw = strain_data.shape

        disp = session.displacement_results[idx]
        U, V = disp[:, :, 0], disp[:, :, 1]
        deformed_frame_idx = idx + 1
        deformed_mask = session.per_frame_rois.get(deformed_frame_idx)
        return get_warped_full_data(
            data=strain_data, frame_idx=idx,
            U=U, V=V,
            roi_rect=rect,
            image_shape=(h, w),
            needs_upsample=(sh != roi_h or sw != roi_w),
            roi_h=roi_h, roi_w=roi_w,
            cache=session.inverse_map_cache,
            vw=vw, vh=vh,
            deformed_mask=deformed_mask,
        )
    # ... rest unchanged
```

Run: `cd server && python -m pytest tests/ -x -q`

---

### Task 6: Controller uses session.per_frame_rois instead of discover_masks

**Files:**
- Modify: `raft_dic_gui/controller.py` (lines 118-126, 215-220, 271-274)
- Modify: `server/routes/processing.py` (processing run route)

**Step 1: Controller — accept pre-loaded user_masks**

In `controller.py`, modify `run()` to accept an optional `user_masks` parameter instead of always calling `discover_masks()`:

```python
def run(self, config, roi_mask, roi_rect, image_files=None, user_masks=None):
    # ... existing setup code ...

    # Use provided user_masks or discover from mask_dir
    if user_masks is None:
        user_masks = {}
        if config.mask_dir:
            from raft_dic_gui.mask_loader import discover_masks
            ref_image_tmp = proc.load_and_convert_image(
                os.path.join(img_dir, image_files[0])
            )
            image_shape = (ref_image_tmp.shape[0], ref_image_tmp.shape[1])
            user_masks = discover_masks(config.mask_dir, image_files, image_shape)
            del ref_image_tmp
```

**Step 2: processing.py — pass session.per_frame_rois to controller**

In the `/run` route, when starting the processing thread, pass `session.per_frame_rois`:

```python
# In the run route, inside the processing thread function:
per_frame = dict(session.per_frame_rois)  # snapshot
processor.run(
    config, roi_mask, roi_rect,
    image_files=image_files,
    user_masks=per_frame if per_frame else None,
)
```

This way:
- If user drew/imported per-frame ROIs in the UI → they are used directly
- If user only set mask_dir → controller falls back to `discover_masks()`
- If neither → no user masks (auto-warp or original ROI)

Run: `cd server && python -m pytest tests/ -x -q`

---

## Phase 3: Frontend — ROI Page Frame Switching

### Task 7: Frontend API updates

**Files:**
- Modify: `frontend/src/api/roi.ts`

**Step 1: Add frame_idx parameter to existing functions**

```typescript
export async function addPolygon(
  points: [number, number][],
  mode: "add" | "cut" = "add",
  frameIdx: number = 0,
): Promise<RoiResult> {
  const { data } = await client.post<RoiResult>("/roi/polygon", {
    points, mode, frame_idx: frameIdx,
  });
  return data;
}

export async function addRectangle(
  x0: number, y0: number, x1: number, y1: number,
  mode: "add" | "cut" = "add",
  frameIdx: number = 0,
): Promise<RoiResult> {
  const { data } = await client.post<RoiResult>("/roi/rectangle", {
    x0, y0, x1, y1, mode, frame_idx: frameIdx,
  });
  return data;
}

export async function addCircle(
  cx: number, cy: number, r: number,
  mode: "add" | "cut" = "add",
  frameIdx: number = 0,
): Promise<RoiResult> {
  const { data } = await client.post<RoiResult>("/roi/circle", {
    cx, cy, r, mode, frame_idx: frameIdx,
  });
  return data;
}

export async function importMask(
  path: string,
  minArea: number = 0,
  smoothRadius: number = 0,
  frameIdx: number = 0,
): Promise<RoiResult> {
  const { data } = await client.post<RoiResult>("/roi/import", {
    path, min_area: minArea, smooth_radius: smoothRadius,
    frame_idx: frameIdx,
  });
  return data;
}

export function maskImageUrl(frameIdx: number = 0): string {
  return `/api/roi/mask?frame_idx=${frameIdx}`;
}
```

**Step 2: Add new API functions**

```typescript
export async function clearFrameRoi(frameIdx: number): Promise<void> {
  await client.delete(`/roi/frame/${frameIdx}`);
}

export interface FramesRoiStatus {
  total_frames: number;
  frames_with_roi: number[];
  frame_0_confirmed: boolean;
}

export async function getFramesRoiStatus(): Promise<FramesRoiStatus> {
  const { data } = await client.get<FramesRoiStatus>("/roi/frames/status");
  return data;
}

export interface BatchImportResult {
  imported: number;
  total_files: number;
  assignments: Record<string, string>;
}

export async function batchImportRois(
  folder: string,
  strategy: "auto_match" | "sequential" = "sequential",
): Promise<BatchImportResult> {
  const { data } = await client.post<BatchImportResult>("/roi/frames/batch-import", {
    folder, strategy,
  });
  return data;
}
```

---

### Task 8: ROI Store updates

**Files:**
- Modify: `frontend/src/stores/roiStore.ts`

**Step 1: Add per-frame state**

```typescript
interface RoiState {
  // Existing fields (keep all)
  drawingMode: "polygon" | "rectangle" | "circle" | null;
  cutMode: boolean;
  currentPoints: [number, number][];
  maskUrl: string | null;
  showImportDialog: boolean;

  // New fields
  editingFrameIdx: number;  // which frame is being edited (0-based)
  framesWithRoi: number[];  // which frames have ROI

  // New actions
  setEditingFrameIdx: (idx: number) => void;
  setFramesWithRoi: (frames: number[]) => void;
  refreshMaskUrl: () => void;
}
```

The `refreshMaskUrl` action updates `maskUrl` to `/api/roi/mask?frame_idx=${editingFrameIdx}&t=${Date.now()}`.

---

### Task 9: ROI Page frame selector

**Files:**
- Modify: `frontend/src/pages/RoiPage.tsx`
- Create: `frontend/src/components/roi/FrameRoiSelector.tsx`

**Step 1: Create FrameRoiSelector component**

A compact frame navigation bar that shows:
- Current frame number / total frames
- Previous/Next buttons
- A mini status strip showing which frames have ROI (colored dots)
- Current frame's image loads via `/api/images/<frame_idx>`

```
[◀ Prev] Frame 3 / 50 [Next ▶]
ROI: ●○○○●○○○○● (dots for frames with ROI)
     1         5         10
```

**Step 2: Update RoiPage layout**

Add `<FrameRoiSelector />` above or inside the sidebar area. When user switches frame:
1. Update `roiStore.editingFrameIdx`
2. Fetch the corresponding frame's image for canvas display
3. Update `maskUrl` to show that frame's ROI overlay

**Step 3: Update RoiCanvas to use editingFrameIdx**

When `editingFrameIdx > 0`, the canvas shows a different frame's image (not the reference image). Need to load the frame image via the image cache API.

All drawing operations (polygon, rectangle, circle) pass `editingFrameIdx` to the API calls.

---

### Task 10: Update RoiCanvas drawing to pass frame_idx

**Files:**
- Modify: `frontend/src/components/roi/RoiCanvas.tsx`

In each `addShape()` call (around line 128-192), pass the current `editingFrameIdx`:

```typescript
const frameIdx = useRoiStore((s) => s.editingFrameIdx);

// In polygon completion handler:
await addPolygon(points, cutMode ? "cut" : "add", frameIdx);

// In rectangle completion handler:
await addRectangle(x0, y0, x1, y1, cutMode ? "cut" : "add", frameIdx);

// In circle completion handler:
await addCircle(cx, cy, r, cutMode ? "cut" : "add", frameIdx);
```

After each drawing operation, refresh `maskUrl` with the correct frame_idx.

---

## Phase 4: Frontend — Status Indicators & Batch Import

### Task 11: ROI status indicators in frame selector

**Files:**
- Modify: `frontend/src/components/roi/FrameRoiSelector.tsx`

Show per-frame ROI status using colored indicators:
- **Green dot** (●): Frame has user-provided ROI → `[Edit]`
- **Red dot** (✕): Frame is a reference/key frame but has NO ROI → `[Need]`
- **Gray dot** (○): Frame has no ROI and is not a reference frame → `[Add]`

Reference frames are determined from `appStore.keyFrames` + `appStore.keyFrameMode`.

Clicking a dot/indicator jumps to that frame for editing.

---

### Task 12: Batch import dialog

**Files:**
- Create: `frontend/src/components/roi/BatchRoiImportDialog.tsx`
- Modify: `frontend/src/components/roi/RoiToolbar.tsx` (add "Batch Import" button)

**Step 1: Create dialog component**

A modal dialog with:
- Folder path input + Browse (same as current MaskSourceSelector)
- Strategy selector: "Auto-match by name" / "Sequential"
- Preview of assignments (after validation)
- Import button
- Result summary

**Step 2: Add button to toolbar**

Add a "Batch Import" button next to existing Import button in `RoiToolbar.tsx`.

---

### Task 13: Processing validation for reference frame ROI

**Files:**
- Modify: `frontend/src/components/roi/ProcessingParams.tsx`

Before starting processing, check if all reference frames have ROI:

```typescript
const handleStartProcessing = async () => {
  // Check reference frames have ROI
  if (mode === "incremental") {
    const status = await getFramesRoiStatus();
    const missingRefs = keyFrames.filter(kf => !status.frames_with_roi.includes(kf - 1));
    if (missingRefs.length > 0) {
      // Show warning: "Reference frames X, Y need ROI. Use Frame 1 ROI? [Continue] [Cancel]"
      // If continue: proceed (controller will inherit Frame 0 ROI)
      // If cancel: abort
    }
  }
  // ... existing start logic
};
```

---

## Phase 5: Integration & Testing

### Task 14: Integration test — per-frame ROI round-trip

**Files:**
- Create: `server/tests/test_per_frame_roi.py`

```python
def test_per_frame_roi_draw_and_retrieve():
    """Draw ROI on frame 0 and frame 5, verify both stored."""

def test_per_frame_roi_batch_import():
    """Batch import masks, verify assignments."""

def test_per_frame_roi_status():
    """Check /frames/status returns correct frame list."""

def test_per_frame_roi_backward_compat():
    """Drawing with no frame_idx still works on frame 0."""

def test_deformed_view_uses_frame_mask():
    """When frame has user mask, deformed view validity is clipped."""
```

### Task 15: Manual smoke test checklist

- [ ] Load images → draw ROI on Frame 1 → run processing → deformed view works (no regression)
- [ ] Load images → draw ROI on Frame 1 → switch to Frame 5 → draw different ROI → verify both saved
- [ ] Batch import masks → verify status dots update
- [ ] Run incremental processing with per-frame masks → deformed view uses correct masks
- [ ] Accumulative mode with per-frame masks → deformed view uses masks for all frames
- [ ] Clear Frame 5 ROI → verify it's removed, Frame 1 still intact

---

## Implementation Order Summary

```
Phase 1 (Backend storage):    Task 1 → Task 2 → Task 3
Phase 2 (Deformed warp):      Task 4 → Task 5 → Task 6
Phase 3 (Frontend ROI edit):  Task 7 → Task 8 → Task 9 → Task 10
Phase 4 (UI polish):          Task 11 → Task 12 → Task 13
Phase 5 (Testing):            Task 14 → Task 15
```

Each phase is independently testable. Phase 1-2 can be verified with backend tests alone. Phase 3-4 adds UI. Phase 5 ensures integration.
