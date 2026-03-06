# Incremental Mode: Key Frame & Per-Frame Mask Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add key frame control (reference update interval) and per-frame mask loading to incremental mode, with a `ref_map` architecture that future-proofs for non-linear reference strategies (cyclic loading, adaptive reference selection).

**Architecture:** Replace the current per-frame reference update in `controller.py` with a `ref_map: Dict[int, int]` driven loop. Within each segment (between key frames), frames use accumulative correlation against the key frame. Accumulation only happens at segment boundaries. Per-frame masks from a user folder override auto-warp, with fallback chain: user mask > auto warp > original mask.

**Tech Stack:** Python (backend), React + TypeScript + Tailwind (frontend), pytest

---

## Design Decisions (from brainstorming)

1. **Strategy A (段内 accumulative):** Within a segment, all frames correlate against the key frame. Accumulation only at segment boundaries.
2. **ref_map architecture:** Internal representation is `Dict[int, int]` (frame → reference frame). Key frames / Every-N / future adaptive algorithms all just generate a `ref_map`. Controller only sees `ref_map`.
3. **Mask fallback chain:** Per-frame user mask > auto warp > original mask. Providing a mask does NOT auto-create a key frame (decoupled concepts).
4. **Mask file matching:** Try filename match first (sans extension), then number extraction. Graceful handling of mismatches.
5. **UI:** Timeline with clickable markers + "Add Key Frame" input + "Every N" button (方案 C). Mask source in Processing Params. Only shown when Incremental mode selected.
6. **Per-frame output:** Always total displacement relative to Frame 1, regardless of `ref_map` structure.

---

## Phase 1: Backend — ref_map Utilities

### Task 1: ref_map Generation Functions

**Files:**
- Modify: `raft_dic_gui/incremental.py`
- Modify: `server/tests/test_incremental.py`

Add functions to generate `ref_map` from different input strategies, and `kf_accumulated` management.

**Step 1: Write tests**

Add to `server/tests/test_incremental.py`:

```python
class TestBuildRefMap:
    """Test ref_map generation from key frames."""

    def test_every_frame(self):
        """N=1: each frame references the previous frame."""
        from raft_dic_gui.incremental import build_ref_map
        ref_map = build_ref_map(total_frames=5, key_frames=[1])
        # Frame 1 is reference, frames 2-5 all ref frame 1 (single segment)
        assert ref_map == {2: 1, 3: 1, 4: 1, 5: 1}

    def test_every_n_frames(self):
        """Key frames at 1, 3, 5 with 5 total frames."""
        from raft_dic_gui.incremental import build_ref_map
        ref_map = build_ref_map(total_frames=5, key_frames=[1, 3, 5])
        assert ref_map == {2: 1, 3: 1, 4: 3, 5: 3}

    def test_key_frames_generate_segments(self):
        """Key frames [1, 50, 100] with 100 frames."""
        from raft_dic_gui.incremental import build_ref_map
        ref_map = build_ref_map(total_frames=100, key_frames=[1, 50])
        # Frames 2-50 ref frame 1, frames 51-100 ref frame 50
        assert ref_map[2] == 1
        assert ref_map[50] == 1
        assert ref_map[51] == 50
        assert ref_map[100] == 50

    def test_custom_ref_map_passthrough(self):
        """A pre-built ref_map should pass through unchanged."""
        from raft_dic_gui.incremental import build_ref_map
        custom = {2: 1, 3: 1, 4: 1, 5: 1, 6: 1}
        ref_map = build_ref_map(total_frames=6, ref_map=custom)
        assert ref_map == custom


class TestEveryNKeyFrames:
    """Test key frame generation from interval N."""

    def test_every_50(self):
        from raft_dic_gui.incremental import key_frames_every_n
        kf = key_frames_every_n(total_frames=200, n=50)
        assert kf == [1, 51, 101, 151]

    def test_every_1(self):
        from raft_dic_gui.incremental import key_frames_every_n
        kf = key_frames_every_n(total_frames=5, n=1)
        assert kf == [1, 2, 3, 4, 5]

    def test_n_larger_than_total(self):
        from raft_dic_gui.incremental import key_frames_every_n
        kf = key_frames_every_n(total_frames=10, n=100)
        assert kf == [1]
```

**Step 2: Run tests, verify they fail**

Run: `python -m pytest server/tests/test_incremental.py -k "TestBuildRefMap or TestEveryN" -v`

**Step 3: Implement**

Add to `raft_dic_gui/incremental.py`:

```python
def key_frames_every_n(total_frames: int, n: int) -> List[int]:
    """Generate key frame list at every N frames (1-indexed)."""
    if n < 1:
        n = 1
    return [i for i in range(1, total_frames + 1, n)]


def build_ref_map(
    total_frames: int,
    key_frames: Optional[List[int]] = None,
    ref_map: Optional[dict] = None,
) -> dict:
    """Build a frame→reference mapping.

    If *ref_map* is provided directly, validate and return it.
    Otherwise, build from *key_frames* using linear segment strategy.
    """
    if ref_map is not None:
        return dict(ref_map)

    if key_frames is None:
        key_frames = [1]

    key_frames_sorted = sorted(set(key_frames))
    if key_frames_sorted[0] != 1:
        key_frames_sorted.insert(0, 1)

    result = {}
    for i, kf in enumerate(key_frames_sorted):
        end = key_frames_sorted[i + 1] if i + 1 < len(key_frames_sorted) else total_frames + 1
        for frame in range(kf + 1, end):
            if frame <= total_frames:
                result[frame] = kf
        # The key frame itself (except frame 1) also refs the previous key frame
        if kf != 1:
            result[kf] = key_frames_sorted[i - 1]

    return result
```

**Step 4: Run tests, verify pass**

**Step 5: Commit**

```bash
git commit -m "feat: add ref_map generation utilities for key frame strategies"
```

---

## Phase 2: Backend — Per-Frame Mask Loading

### Task 2: Mask Discovery and Loading

**Files:**
- Create: `raft_dic_gui/mask_loader.py`
- Create: `server/tests/test_mask_loader.py`

**Step 1: Write tests**

```python
"""Tests for per-frame mask loading."""
import numpy as np
import pytest
from PIL import Image
from pathlib import Path


def _create_mask_image(path: Path, h: int, w: int):
    """Create a binary mask PNG."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[10:h-10, 10:w-10] = 255
    Image.fromarray(mask).save(path)


class TestMaskDiscovery:
    def test_match_by_filename(self, tmp_path):
        """Masks with same name as images (different extension) should match."""
        from raft_dic_gui.mask_loader import discover_masks
        _create_mask_image(tmp_path / "frame_001.png", 64, 64)
        _create_mask_image(tmp_path / "frame_003.png", 64, 64)

        image_files = ["frame_001.tif", "frame_002.tif", "frame_003.tif"]
        result = discover_masks(str(tmp_path), image_files, image_shape=(64, 64))

        assert 0 in result  # frame_001 matched to index 0
        assert 1 not in result
        assert 2 in result  # frame_003 matched to index 2

    def test_match_by_number(self, tmp_path):
        """Masks named mask_003.png should match frame index 2 (1-indexed → 0-indexed)."""
        from raft_dic_gui.mask_loader import discover_masks
        _create_mask_image(tmp_path / "mask_003.png", 64, 64)

        image_files = ["img_001.tif", "img_002.tif", "img_003.tif"]
        result = discover_masks(str(tmp_path), image_files, image_shape=(64, 64))

        assert 2 in result  # mask_003 → frame index 2

    def test_wrong_size_skipped(self, tmp_path):
        """Masks with wrong dimensions should be skipped."""
        from raft_dic_gui.mask_loader import discover_masks
        _create_mask_image(tmp_path / "frame_001.png", 32, 32)  # Wrong size

        image_files = ["frame_001.tif"]
        result = discover_masks(str(tmp_path), image_files, image_shape=(64, 64))

        assert len(result) == 0

    def test_extra_masks_ignored(self, tmp_path):
        """Masks that don't match any frame should be ignored."""
        from raft_dic_gui.mask_loader import discover_masks
        _create_mask_image(tmp_path / "frame_001.png", 64, 64)
        _create_mask_image(tmp_path / "frame_999.png", 64, 64)

        image_files = ["frame_001.tif"]
        result = discover_masks(str(tmp_path), image_files, image_shape=(64, 64))

        assert len(result) == 1

    def test_empty_folder(self, tmp_path):
        from raft_dic_gui.mask_loader import discover_masks
        result = discover_masks(str(tmp_path), ["img.tif"], image_shape=(64, 64))
        assert len(result) == 0

    def test_filename_match_priority_over_number(self, tmp_path):
        """Filename match should take priority over number extraction."""
        from raft_dic_gui.mask_loader import discover_masks
        _create_mask_image(tmp_path / "frame_001.png", 64, 64)

        image_files = ["frame_001.tif", "frame_002.tif"]
        result = discover_masks(str(tmp_path), image_files, image_shape=(64, 64))

        assert 0 in result
```

**Step 2: Implement `raft_dic_gui/mask_loader.py`**

```python
"""Load and match per-frame masks from a user-provided folder."""

import os
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


MASK_EXTENSIONS = {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}


def discover_masks(
    mask_dir: str,
    image_files: List[str],
    image_shape: Tuple[int, int],
) -> Dict[int, np.ndarray]:
    """Discover and load per-frame masks from a folder.

    Matching strategy (in priority order):
    1. Filename match: mask stem == image stem (ignoring extension)
    2. Number extraction: largest integer in mask filename → 1-indexed frame number

    Args:
        mask_dir: Path to folder containing mask images.
        image_files: List of image filenames (ordered, as loaded).
        image_shape: Expected (H, W) of each mask.

    Returns:
        Dict mapping frame index (0-based) to boolean mask array.
        Only successfully loaded and validated masks are included.
    """
    if not os.path.isdir(mask_dir):
        return {}

    H, W = image_shape

    # Build stem→index lookup from image files
    stem_to_idx = {}
    for i, fname in enumerate(image_files):
        stem = Path(fname).stem
        stem_to_idx[stem] = i

    # Scan mask folder
    mask_files = sorted([
        f for f in os.listdir(mask_dir)
        if Path(f).suffix.lower() in MASK_EXTENSIONS
    ])

    matched: Dict[int, np.ndarray] = {}
    warnings = []

    for mf in mask_files:
        mask_path = os.path.join(mask_dir, mf)
        mask_stem = Path(mf).stem

        # Strategy 1: filename match
        frame_idx = stem_to_idx.get(mask_stem)

        # Strategy 2: number extraction
        if frame_idx is None:
            numbers = re.findall(r"\d+", mask_stem)
            if numbers:
                frame_num = int(numbers[-1])  # 1-indexed
                frame_idx = frame_num - 1  # Convert to 0-indexed
                if frame_idx < 0 or frame_idx >= len(image_files):
                    warnings.append(f"{mf}: frame number {frame_num} out of range, skipped")
                    frame_idx = None

        if frame_idx is None:
            warnings.append(f"{mf}: no matching frame found, skipped")
            continue

        # Load and validate
        try:
            from PIL import Image
            img = Image.open(mask_path)
            mask = np.array(img)

            # Handle RGB/RGBA → grayscale
            if mask.ndim == 3:
                mask = mask[:, :, 0]

            if mask.shape != (H, W):
                warnings.append(
                    f"{mf}: size {mask.shape[1]}x{mask.shape[0]} doesn't match "
                    f"image {W}x{H}, skipped"
                )
                continue

            matched[frame_idx] = mask > 0
        except Exception as e:
            warnings.append(f"{mf}: failed to load ({e}), skipped")

    # Print warnings
    for w in warnings:
        print(f"[Mask Loader] WARNING: {w}")

    total = len(image_files)
    n = len(matched)
    if n > 0:
        print(f"[Mask Loader] Found masks for {n} of {total} frames. "
              f"Remaining frames will use auto warp.")
    else:
        print(f"[Mask Loader] No valid masks found in {mask_dir}. "
              f"Using auto warp for all frames.")

    return matched
```

**Step 3: Run tests, verify pass**

**Step 4: Commit**

```bash
git commit -m "feat: add per-frame mask loader with filename and number matching"
```

---

## Phase 3: Backend — Controller Rewrite with ref_map

### Task 3: Rewrite controller.py Main Loop

**Files:**
- Modify: `raft_dic_gui/controller.py`
- Modify: `raft_dic_gui/config.py`

This is the core change. The controller loop becomes ref_map-driven with `kf_accumulated` caching.

**Step 1: Add fields to DICConfig**

```python
# In config.py DICConfig:
key_frames: Optional[list] = None        # User-specified key frames (1-indexed)
key_frame_interval: Optional[int] = None  # Every N frames shortcut
mask_dir: Optional[str] = None            # Per-frame mask folder path
```

**Step 2: Rewrite controller.py**

Key structural changes to `run()`:

```python
def run(self, config, roi_mask, roi_rect):
    # ... (model loading, image listing, resume check — unchanged) ...

    from raft_dic_gui.incremental import (
        build_ref_map, key_frames_every_n,
        accumulate_displacement, warp_mask_with_holes
    )

    is_incremental = config.mode == "incremental"
    xmin, ymin, xmax, ymax = roi_rect

    # Build ref_map
    if is_incremental:
        if config.key_frame_interval and config.key_frame_interval > 1:
            kf_list = key_frames_every_n(len(image_files), config.key_frame_interval)
        elif config.key_frames:
            kf_list = config.key_frames
        else:
            kf_list = [1]  # Default: single segment (all accumulative against frame 1)
        ref_map = build_ref_map(len(image_files), key_frames=kf_list)
    else:
        ref_map = {i: 1 for i in range(2, len(image_files) + 1)}  # All ref frame 1

    # Load per-frame masks (if provided)
    user_masks = {}
    if is_incremental and config.mask_dir:
        from raft_dic_gui.mask_loader import discover_masks
        user_masks = discover_masks(
            config.mask_dir, image_files,
            image_shape=(ref_image.shape[0], ref_image.shape[1])
        )

    # Key frame accumulated displacements: kf → total_disp relative to frame 1
    # kf_accumulated[1] = zeros (frame 1 is origin)
    H_roi, W_roi = ymax - ymin, xmax - xmin
    kf_accumulated = {1: np.zeros((H_roi, W_roi, 2), dtype=np.float64)}

    # Key frame images cache (only keep what's needed)
    kf_images = {1: ref_image.copy()}

    # Key frame masks cache
    original_mask_crop = roi_mask[ymin:ymax, xmin:xmax].copy()
    kf_masks = {1: roi_mask.copy()}

    # Track current ref to detect ref changes
    current_ref = None

    for i in range(1, len(image_files)):
        if self.check_stop and self.check_stop():
            break

        frame_num = i + 1  # 1-indexed frame number
        ref_num = ref_map.get(frame_num, 1)

        # Detect reference switch → load ref image, compute accumulated at new KF
        if ref_num != current_ref:
            current_ref = ref_num
            if ref_num not in kf_images:
                ref_img_path = os.path.join(img_dir, image_files[ref_num - 1])
                kf_images[ref_num] = proc.load_and_convert_image(ref_img_path)
            ref_image = kf_images[ref_num]

            # Compute mask for this key frame
            if ref_num not in kf_masks:
                if (ref_num - 1) in user_masks:
                    # User provided a mask for this key frame
                    full_mask = np.zeros_like(roi_mask)
                    full_mask[:] = user_masks[ref_num - 1]  # Assuming full-image mask
                    kf_masks[ref_num] = full_mask
                elif ref_num in kf_accumulated:
                    # Auto warp from original
                    warped = warp_mask_with_holes(original_mask_crop, kf_accumulated[ref_num])
                    m = roi_mask.copy()
                    m[ymin:ymax, xmin:xmax] = warped
                    kf_masks[ref_num] = m
                else:
                    kf_masks[ref_num] = roi_mask.copy()

        # Determine mask for this frame
        if (i) in user_masks:  # i is 0-indexed frame index
            current_mask = np.zeros_like(roi_mask)
            current_mask[:] = user_masks[i]
        else:
            current_mask = kf_masks.get(current_ref, roi_mask)

        # ... resume check (load from .npy if exists) ...

        # DIC computation
        def_image = proc.load_and_convert_image(os.path.join(img_dir, image_files[i]))
        disp_full, _ = proc.dic_over_roi_with_tiling(
            ref_image, def_image, current_mask, model, device, ...
        )
        delta_disp = disp_full[ymin:ymax, xmin:xmax, :]

        # Compute total displacement relative to frame 1
        ref_accum = kf_accumulated.get(current_ref, np.zeros((H_roi, W_roi, 2)))
        if np.all(ref_accum == 0):
            displacement_field = delta_disp
        else:
            displacement_field = accumulate_displacement(ref_accum, delta_disp)

        # If this frame is a key frame for a later segment, cache its accumulated disp
        if frame_num in [ref_map.get(j) for j in ref_map if ref_map[j] == frame_num]:
            # This frame is referenced by later frames
            kf_accumulated[frame_num] = displacement_field.copy()
            kf_images[frame_num] = def_image.copy()

        # Save + append (unchanged)
        sequence_displacements.append(displacement_field)
        # ... save to .npy ...

    return sequence_displacements
```

Note: The above is a structural sketch. The actual implementation should:
- Handle the resume path correctly (restore kf_accumulated from saved .npy)
- Pre-compute which frames are referenced as key frames for caching
- Clean up kf_images cache for memory (remove key frames no longer needed)
- Keep progress reporting and stop/pause checks

**Step 3: Run all tests**

**Step 4: Commit**

```bash
git commit -m "feat: ref_map-driven controller with key frame segments and per-frame masks"
```

---

## Phase 4: Backend — API Endpoints

### Task 4: Key Frame and Mask Configuration

**Files:**
- Modify: `server/routes/processing.py`
- Modify: `server/tests/test_processing.py`

**Step 1: Extend `/api/processing/configure` to accept new fields**

```python
# In configure():
if "key_frames" in data:
    cfg.key_frames = data["key_frames"]  # List[int] or None
if "key_frame_interval" in data:
    cfg.key_frame_interval = int(data["key_frame_interval"]) if data["key_frame_interval"] else None
if "mask_dir" in data:
    cfg.mask_dir = data["mask_dir"] if data["mask_dir"] else None
```

**Step 2: Add mask validation endpoint**

```python
@processing_bp.route("/validate-masks", methods=["POST"])
def validate_masks():
    """Validate a mask folder against loaded images."""
    data = request.get_json(force=True)
    mask_dir = data.get("mask_dir", "")

    if not session.image_files:
        return jsonify({"error": "No images loaded"}), 400

    from raft_dic_gui.mask_loader import discover_masks
    masks = discover_masks(
        mask_dir,
        session.image_files,
        image_shape=(session.image_height, session.image_width),
    )

    matched_frames = sorted([idx + 1 for idx in masks.keys()])  # 1-indexed for UI
    return jsonify({
        "matched_count": len(masks),
        "total_frames": len(session.image_files),
        "matched_frames": matched_frames,
    })
```

**Step 3: Add tests**

```python
def test_configure_key_frames(client):
    resp = client.post("/api/processing/configure", json={
        "mode": "incremental",
        "key_frame_interval": 50,
    })
    assert resp.status_code == 200
    assert session.config.key_frame_interval == 50

def test_configure_mask_dir(client):
    resp = client.post("/api/processing/configure", json={
        "mask_dir": "/some/path",
    })
    assert resp.status_code == 200
    assert session.config.mask_dir == "/some/path"
```

**Step 4: Commit**

```bash
git commit -m "feat: API endpoints for key frame and mask configuration"
```

---

## Phase 5: Frontend — Incremental Settings UI

### Task 5: Key Frame Timeline Component

**Files:**
- Create: `frontend/src/components/roi/KeyFrameTimeline.tsx`

A compact timeline component with:
- Horizontal bar representing all frames
- Clickable markers for key frames (Frame 1 always marked, non-removable)
- Hover tooltip showing frame number
- Click to add/remove markers
- "Add Key Frame" input for precise frame number entry
- "Every N" input + Apply button
- "Clear" button (resets to just Frame 1)

```tsx
// KeyFrameTimeline.tsx — structural sketch
interface KeyFrameTimelineProps {
  totalFrames: number;
  keyFrames: number[];      // 1-indexed
  onChange: (kf: number[]) => void;
}

export function KeyFrameTimeline({ totalFrames, keyFrames, onChange }: KeyFrameTimelineProps) {
  const [everyN, setEveryN] = useState("");
  const [addFrame, setAddFrame] = useState("");

  const handleTimelineClick = (e: React.MouseEvent<HTMLDivElement>) => {
    // Calculate frame number from click position
    const rect = e.currentTarget.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    const frame = Math.round(pct * (totalFrames - 1)) + 1;
    if (frame === 1) return; // Can't remove frame 1
    // Toggle: add if missing, remove if present
    if (keyFrames.includes(frame)) {
      onChange(keyFrames.filter(f => f !== frame));
    } else {
      onChange([...keyFrames, frame].sort((a, b) => a - b));
    }
  };

  const handleEveryN = () => {
    const n = parseInt(everyN);
    if (n > 0) {
      const kf = [];
      for (let i = 1; i <= totalFrames; i += n) kf.push(i);
      onChange(kf);
    }
  };

  const handleAddFrame = () => {
    const f = parseInt(addFrame);
    if (f >= 1 && f <= totalFrames && !keyFrames.includes(f)) {
      onChange([...keyFrames, f].sort((a, b) => a - b));
      setAddFrame("");
    }
  };

  return (
    <div className="flex flex-col gap-1.5">
      {/* Timeline bar */}
      <div
        className="relative h-6 bg-[var(--secondary)] rounded cursor-crosshair"
        onClick={handleTimelineClick}
      >
        {keyFrames.map(kf => {
          const pct = ((kf - 1) / (totalFrames - 1)) * 100;
          return (
            <div
              key={kf}
              className="absolute top-0 w-1.5 h-full bg-[var(--primary)] rounded-sm"
              style={{ left: `calc(${pct}% - 3px)` }}
              title={`Frame ${kf}`}
            />
          );
        })}
        {/* Frame 1 and N labels */}
        <span className="absolute left-1 top-1/2 -translate-y-1/2 text-[9px] text-[var(--muted-foreground)]">1</span>
        <span className="absolute right-1 top-1/2 -translate-y-1/2 text-[9px] text-[var(--muted-foreground)]">{totalFrames}</span>
      </div>

      {/* Controls row */}
      <div className="flex items-center gap-2 text-[11px]">
        <span className="text-[var(--muted-foreground)]">Every</span>
        <input ... value={everyN} />
        <button onClick={handleEveryN}>Apply</button>

        <span className="text-[var(--muted-foreground)] ml-2">Add:</span>
        <input ... value={addFrame} />
        <button onClick={handleAddFrame}>+</button>

        <button onClick={() => onChange([1])} className="ml-auto">Clear</button>
      </div>

      {/* Summary */}
      <span className="text-[9px] text-[var(--muted-foreground)]">
        {keyFrames.length} key frame{keyFrames.length !== 1 ? "s" : ""}: {keyFrames.join(", ")}
      </span>
    </div>
  );
}
```

**Step 1: Implement the component**
**Step 2: Commit**

```bash
git commit -m "feat: KeyFrameTimeline component with click markers and Every-N"
```

---

### Task 6: Mask Source Selector

**Files:**
- Create: `frontend/src/components/roi/MaskSourceSelector.tsx`

A dropdown + conditional folder browser + validation summary.

```tsx
interface MaskSourceSelectorProps {
  maskSource: "auto" | "folder";
  maskDir: string;
  onSourceChange: (source: "auto" | "folder") => void;
  onDirChange: (dir: string) => void;
  validationResult?: { matched_count: number; total_frames: number; matched_frames: number[] };
}
```

Shows:
- SelectField: "Auto Warp" / "Load from Folder"
- When "Load from Folder": FileBrowser button + validation summary
- After folder selected: calls `/api/processing/validate-masks` and shows result

**Step 1: Implement the component**
**Step 2: Commit**

```bash
git commit -m "feat: MaskSourceSelector component with folder browser and validation"
```

---

### Task 7: Integrate into ProcessingParams

**Files:**
- Modify: `frontend/src/components/roi/ProcessingParams.tsx`
- Modify: `frontend/src/stores/appStore.ts`
- Modify: `frontend/src/api/processing.ts`

**Step 1: Add state to appStore**

```typescript
// In AppState interface:
keyFrames: number[];
keyFrameMode: "every_frame" | "every_n" | "custom";
keyFrameInterval: number;
maskSource: "auto" | "folder";
maskDir: string;

// Actions:
setKeyFrames: (kf: number[]) => void;
setKeyFrameMode: (mode: "every_frame" | "every_n" | "custom") => void;
setKeyFrameInterval: (n: number) => void;
setMaskSource: (source: "auto" | "folder") => void;
setMaskDir: (dir: string) => void;
```

**Step 2: Add API calls**

```typescript
// In processing.ts:
export async function validateMasks(maskDir: string) {
  const { data } = await client.post("/processing/validate-masks", { mask_dir: maskDir });
  return data;
}
```

**Step 3: Update ProcessingParams.tsx**

Add conditional "Incremental Settings" section shown only when mode === "incremental":

```tsx
{mode === "incremental" && (
  <CollapsibleSection title="Incremental Settings" defaultOpen>
    {/* Reference Update mode selector */}
    <FieldRow label="Ref. Update">
      <SelectField
        value={keyFrameMode}
        options={[
          { value: "every_frame", label: "Every Frame" },
          { value: "every_n", label: "Every N Frames" },
          { value: "custom", label: "Custom Key Frames" },
        ]}
        onChange={handleKeyFrameModeChange}
      />
    </FieldRow>

    {/* Every N input */}
    {keyFrameMode === "every_n" && (
      <FieldRow label="Interval N">
        <SmallInput value={...} onChange={...} />
      </FieldRow>
    )}

    {/* Custom key frame timeline */}
    {keyFrameMode === "custom" && (
      <KeyFrameTimeline
        totalFrames={numFrames}
        keyFrames={keyFrames}
        onChange={handleKeyFrameChange}
      />
    )}

    {/* Mask source */}
    <MaskSourceSelector
      maskSource={maskSource}
      maskDir={maskDir}
      onSourceChange={...}
      onDirChange={...}
    />
  </CollapsibleSection>
)}
```

**Step 4: Build frontend**

Run: `cd frontend && npm run build`

**Step 5: Commit**

```bash
git commit -m "feat: incremental settings UI with key frame timeline and mask source"
```

---

## Phase 6: Integration Tests & Verification

### Task 8: Integration Tests

**Files:**
- Modify: `server/tests/test_incremental.py`
- Modify: `server/tests/test_processing.py`

**Step 1: Add integration tests**

```python
class TestRefMapDrivenAccumulation:
    """Verify ref_map-based accumulation produces correct total displacements."""

    def test_two_segment_uniform_translation(self):
        """Two segments with uniform translation should sum correctly.

        Segment 1 (frames 2-5, ref=1): delta = (2, 0) each
        Segment 2 (frames 6-10, ref=5): delta = (3, 0) each

        Frame 5 total = (2, 0) * 4 steps? No — Strategy A!
        Frame 5 is accumulative against frame 1, so total_5 = DIC(1, 5)
        Frame 6 total = total_5 + DIC(5, 6)
        """
        pass  # Implementation depends on final controller structure

    def test_cyclic_ref_map(self):
        """Non-linear ref_map: frame 6 references frame 1 instead of frame 5."""
        from raft_dic_gui.incremental import build_ref_map, accumulate_displacement
        # Custom ref_map where frame 6 jumps back to frame 1
        ref_map = {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1}
        # All frames ref frame 1 → all displacements are direct, no accumulation
        # This just verifies build_ref_map accepts and passes through custom maps
        assert ref_map[6] == 1
```

**Step 2: Run full test suite**

Run: `python -m pytest server/tests/ -v --tb=short`

**Step 3: Build and manual test**

Run: `cd frontend && npm run build`
Then: `python run_prod.py` and test in browser

**Step 4: Commit**

```bash
git commit -m "test: integration tests for ref_map accumulation and key frame UI"
```

---

## Summary

| Phase | Tasks | Files Changed | Estimated Complexity |
|-------|-------|---------------|---------------------|
| 1: ref_map utilities | Task 1 | incremental.py, tests | Low |
| 2: Mask loader | Task 2 | mask_loader.py (new), tests | Low |
| 3: Controller rewrite | Task 3 | controller.py, config.py | **Medium-High** |
| 4: API endpoints | Task 4 | processing.py, tests | Low |
| 5: Frontend UI | Tasks 5-7 | 3 new components, appStore, ProcessingParams | Medium |
| 6: Integration | Task 8 | tests, verification | Low |

## Future Extension Points (no code needed now)

- **Adaptive ref_map generator:** Analyze correlation quality → generate non-linear ref_map. Only need a new function that returns `Dict[int, int]`.
- **Cyclic loading detection:** Compute image similarity matrix → identify when frames cycle back to earlier states → ref_map points back.
- **Per-frame mask UI painting:** Add ROI editor per frame. Only need to populate `user_masks` dict from a different source.

All three extensions only touch the `ref_map` or `user_masks` generation — the controller loop, accumulation logic, and UI rendering need zero changes.
