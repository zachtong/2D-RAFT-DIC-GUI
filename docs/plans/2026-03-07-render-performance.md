# Render Performance Optimization Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce render memory/time by 90%+ for large images via viewport downsampling, byte-limited cache, and CSS alpha separation.

**Architecture:** Three independent optimizations: (1) RenderCache tracks total bytes to prevent OOM, (2) backend downsamples rendered output to match browser viewport before the expensive colormap step, (3) render endpoints return overlay-only RGBA PNG so frontend can apply alpha via CSS opacity — eliminating re-renders on alpha changes.

**Tech Stack:** Python/Flask (PIL, cv2, matplotlib, numpy), React/TypeScript (Zustand, CSS)

---

### Task 1: RenderCache byte limit

**Files:**
- Modify: `server/render_cache.py`
- Test: `server/tests/test_render_cache.py` (create)

**Step 1: Write test**

Create `server/tests/test_render_cache.py`:

```python
"""Tests for RenderCache byte-limited eviction."""
from server.render_cache import RenderCache


def test_evicts_by_entry_count():
    cache = RenderCache(max_entries=3, max_bytes=10_000_000)
    for i in range(5):
        cache.put(("k", i), b"x" * 100)
    assert len(cache) == 3


def test_evicts_by_total_bytes():
    cache = RenderCache(max_entries=1000, max_bytes=500)
    cache.put(("a",), b"x" * 200)
    cache.put(("b",), b"x" * 200)
    cache.put(("c",), b"x" * 200)
    # 600 bytes > 500 limit, oldest should be evicted
    assert cache.get(("a",)) is None
    assert cache.get(("c",)) is not None
    assert cache.total_bytes <= 500


def test_overwrite_updates_bytes():
    cache = RenderCache(max_entries=100, max_bytes=10_000)
    cache.put(("k",), b"x" * 100)
    assert cache.total_bytes == 100
    cache.put(("k",), b"y" * 300)
    assert cache.total_bytes == 300
    assert cache.get(("k",)) == b"y" * 300


def test_clear_resets_bytes():
    cache = RenderCache(max_entries=100, max_bytes=10_000)
    cache.put(("a",), b"x" * 500)
    cache.put(("b",), b"x" * 500)
    cache.clear()
    assert len(cache) == 0
    assert cache.total_bytes == 0
```

**Step 2: Run test to verify it fails**

Run: `cd server && python -m pytest tests/test_render_cache.py -v`
Expected: FAIL (max_bytes param doesn't exist yet, no total_bytes property)

**Step 3: Implement**

Rewrite `server/render_cache.py`:

```python
"""Simple LRU cache for rendered PNG bytes, shared across route modules."""

from collections import OrderedDict


def auto_cache_size(target_mb: int = 500, avg_item_kb: int = 200) -> int:
    """Compute max entry count. Used as fallback; byte limit is primary."""
    try:
        import psutil
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        budget_mb = min(target_mb, available_mb * 0.1)
        return max(64, int(budget_mb * 1024 / avg_item_kb))
    except (ImportError, Exception):
        return 512


def auto_max_bytes(target_mb: int = 300) -> int:
    """Compute byte budget: min(target_mb, 5% available RAM)."""
    try:
        import psutil
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        budget_mb = min(target_mb, available_mb * 0.05)
        return max(50, int(budget_mb)) * 1024 * 1024
    except (ImportError, Exception):
        return target_mb * 1024 * 1024


class RenderCache:
    """Bounded LRU cache mapping (result_id, frame_idx, params) -> PNG bytes.

    Evicts when either entry count OR total byte size is exceeded.
    """

    def __init__(self, max_entries: int = 512, max_bytes: int = 0):
        self._cache: OrderedDict[tuple, bytes] = OrderedDict()
        self._max = max_entries
        self._max_bytes = max_bytes if max_bytes > 0 else auto_max_bytes()
        self._total_bytes = 0

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def get(self, key: tuple) -> bytes | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: tuple, data: bytes) -> None:
        if key in self._cache:
            self._total_bytes -= len(self._cache[key])
        self._cache[key] = data
        self._cache.move_to_end(key)
        self._total_bytes += len(data)
        self._evict()

    def _evict(self) -> None:
        while self._cache and (
            len(self._cache) > self._max
            or self._total_bytes > self._max_bytes
        ):
            _, evicted = self._cache.popitem(last=False)
            self._total_bytes -= len(evicted)

    def clear(self) -> None:
        self._cache.clear()
        self._total_bytes = 0

    def __len__(self) -> int:
        return len(self._cache)
```

**Step 4: Run test to verify it passes**

Run: `cd server && python -m pytest tests/test_render_cache.py -v`
Expected: 4 passed

**Step 5: Run all existing tests**

Run: `cd server && python -m pytest tests/ -v`
Expected: All pass (API unchanged, existing callers pass max_entries positionally)

---

### Task 2: Viewport downsampling — Backend helper

**Files:**
- Create: `server/viewport.py`
- Test: `server/tests/test_viewport.py` (create)

**Step 1: Write test**

Create `server/tests/test_viewport.py`:

```python
"""Tests for viewport downsampling helper."""
import numpy as np
from server.viewport import downsample_for_viewport


def test_no_downsample_when_no_viewport():
    data = np.ones((100, 200), dtype=np.float64)
    bg = np.zeros((100, 200, 3), dtype=np.uint8)
    d2, b2, h2, w2 = downsample_for_viewport(data, bg, 0, 0)
    assert d2.shape == (100, 200)
    assert b2.shape == (100, 200, 3)


def test_no_downsample_when_viewport_larger():
    data = np.ones((100, 200), dtype=np.float64)
    bg = np.zeros((100, 200, 3), dtype=np.uint8)
    d2, b2, h2, w2 = downsample_for_viewport(data, bg, 400, 300)
    assert d2.shape == (100, 200)


def test_downsample_halves():
    data = np.random.rand(200, 400).astype(np.float64)
    bg = np.random.randint(0, 255, (200, 400, 3), dtype=np.uint8)
    d2, b2, h2, w2 = downsample_for_viewport(data, bg, 200, 100)
    assert h2 == 100
    assert w2 == 200
    assert d2.shape == (100, 200)
    assert b2.shape == (100, 200, 3)


def test_nan_preserved():
    data = np.full((200, 400), np.nan, dtype=np.float64)
    data[50:150, 100:300] = 1.0
    bg = np.zeros((200, 400, 3), dtype=np.uint8)
    d2, b2, h2, w2 = downsample_for_viewport(data, bg, 200, 100)
    # Corners should still be NaN
    assert np.isnan(d2[0, 0])
    # Center should have data
    center = d2[d2.shape[0] // 2, d2.shape[1] // 2]
    assert np.isfinite(center)


def test_grayscale_bg():
    data = np.ones((100, 200), dtype=np.float64)
    bg = np.zeros((100, 200), dtype=np.uint8)  # 2D grayscale
    d2, b2, h2, w2 = downsample_for_viewport(data, bg, 100, 50)
    assert b2.ndim == 2
    assert b2.shape == (50, 100)
```

**Step 2: Run test to verify it fails**

Run: `cd server && python -m pytest tests/test_viewport.py -v`
Expected: FAIL (module doesn't exist)

**Step 3: Implement**

Create `server/viewport.py`:

```python
"""Viewport-aware downsampling for render endpoints."""

from typing import Tuple

import cv2
import numpy as np


def downsample_for_viewport(
    full_data: np.ndarray,
    bg_img: np.ndarray,
    vw: int,
    vh: int,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Downsample data and background to fit the browser viewport.

    Parameters
    ----------
    full_data : (H, W) scalar field (may contain NaN)
    bg_img : (H, W, 3) or (H, W) background image
    vw, vh : viewport width and height in CSS pixels

    Returns
    -------
    (data_out, bg_out, out_h, out_w) — downsampled arrays and their dimensions.
    If viewport is 0 or larger than image, returns inputs unchanged.
    """
    h, w = full_data.shape[:2]

    if vw <= 0 or vh <= 0:
        return full_data, bg_img, h, w

    scale = min(vw / w, vh / h, 1.0)
    if scale >= 1.0:
        return full_data, bg_img, h, w

    out_w = max(1, int(w * scale))
    out_h = max(1, int(h * scale))

    # Downsample scalar data (NaN-safe)
    mask = np.isfinite(full_data)
    filled = np.nan_to_num(full_data, nan=0.0).astype(np.float32)
    data_small = cv2.resize(filled, (out_w, out_h), interpolation=cv2.INTER_AREA)
    mask_small = cv2.resize(
        mask.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_AREA
    )
    data_out = data_small.astype(np.float64)
    data_out[mask_small < 0.5] = np.nan

    # Downsample background
    bg_out = cv2.resize(bg_img, (out_w, out_h), interpolation=cv2.INTER_AREA)

    return data_out, bg_out, out_h, out_w
```

**Step 4: Run test**

Run: `cd server && python -m pytest tests/test_viewport.py -v`
Expected: 5 passed

---

### Task 3: Viewport downsampling — displacement render

**Files:**
- Modify: `server/routes/displacement.py` (render_frame function, ~L108-227)

**Step 1: Modify render_frame**

In `displacement.py`, add viewport downsampling after `full_data` and `bg_img` are ready, before the colormap step. Key changes:

1. Parse `vw` and `vh` from request args (after L129)
2. After `full_data` and `bg_img` are constructed (after L175, before L178), call `downsample_for_viewport()`
3. Import `downsample_for_viewport` from `server.viewport`

The modified `render_frame` function should have these additions:

After line 129 (parsing ref_frame), add:
```python
    vw = request.args.get("vw", 0, type=int)
    vh = request.args.get("vh", 0, type=int)
```

After line 175 (full_data is complete), before line 178 (mask), add:
```python
    from server.viewport import downsample_for_viewport
    full_data, bg_img, h, w = downsample_for_viewport(full_data, bg_img, vw, vh)
```

Exclude `vw`, `vh` from the cache key (they define output resolution — include them):
Actually, `vw`/`vh` SHOULD be in the cache key because different viewports produce different images. Since they're already in `request.args`, they'll be included in `cache_params` automatically. No change needed for cache key.

**Step 2: Run existing tests**

Run: `cd server && python -m pytest tests/test_displacement.py -v`
Expected: All pass (tests don't pass vw/vh, so downsample is skipped)

---

### Task 4: Viewport downsampling — strain render

**Files:**
- Modify: `server/routes/strain.py` (render_frame function, ~L186-321)

**Step 1: Modify render_frame**

Same pattern as displacement. Add after parsing params (~L206):
```python
    vw = request.args.get("vw", 0, type=int)
    vh = request.args.get("vh", 0, type=int)
```

After `full_data` is complete (after L269, before L272), add:
```python
    from server.viewport import downsample_for_viewport
    full_data, bg_img, h, w = downsample_for_viewport(full_data, bg_img, vw, vh)
```

**Step 2: Run existing tests**

Run: `cd server && python -m pytest tests/test_strain.py -v`
Expected: All pass

---

### Task 5: Viewport downsampling — arrows render

**Files:**
- Modify: `server/routes/arrows.py` (~L92-97)

**Step 1: Modify render_arrows**

Parse viewport params after L40:
```python
    vw = request.args.get("vw", 0, type=int)
    vh = request.args.get("vh", 0, type=int)
```

Replace the figure creation at L96 with viewport-aware sizing:
```python
    # Determine render resolution
    if vw > 0 and vh > 0:
        render_scale = min(vw / img_w, vh / img_h, 1.0)
    else:
        render_scale = 1.0
    render_w = max(1, int(img_w * render_scale))
    render_h = max(1, int(img_h * render_scale))

    fig, ax = plt.subplots(figsize=(render_w / dpi, render_h / dpi), dpi=dpi)
```

The `ax.set_xlim(0, img_w)` and `ax.set_ylim(img_h, 0)` remain unchanged — matplotlib maps full-image coordinates to the smaller figure automatically.

**Step 2: Run existing tests**

Run: `cd server && python -m pytest tests/ -v`
Expected: All pass

---

### Task 6: Viewport downsampling — principal dirs render

**Files:**
- Modify: `server/routes/principal_dirs.py` (~L79-82)

**Step 1: Modify render_principal**

Parse viewport params after L29:
```python
    vw = request.args.get("vw", 0, type=int)
    vh = request.args.get("vh", 0, type=int)
```

Replace figure creation at L82 with viewport-aware sizing (same pattern as arrows):
```python
    if vw > 0 and vh > 0:
        render_scale = min(vw / img_w, vh / img_h, 1.0)
    else:
        render_scale = 1.0
    render_w = max(1, int(img_w * render_scale))
    render_h = max(1, int(img_h * render_scale))

    fig, ax = plt.subplots(figsize=(render_w / dpi, render_h / dpi), dpi=dpi)
```

**Step 2: Run all backend tests**

Run: `cd server && python -m pytest tests/ -v`
Expected: All pass

---

### Task 7: CSS Alpha separation — Backend overlay-only mode

**Files:**
- Modify: `server/routes/displacement.py` (render_frame)
- Modify: `server/routes/strain.py` (render_frame)

**Step 1: displacement.py changes**

In `render_frame()`, add `overlay_only` param parsing after line 129:
```python
    overlay_only = request.args.get("overlay_only", "false").lower() in ("true", "1")
```

When `overlay_only=true`:
- Skip loading background image (lines 141-150) — use a dummy shape source
- Skip compositing (lines 217-219) — return RGBA overlay directly
- Exclude `alpha` from cache_params

Replace the section from background loading through PNG encoding (~L140-224) with logic that branches on `overlay_only`:

```python
    if overlay_only:
        # Get image dimensions without loading the full background
        h = session.image_height
        w = session.image_width

        # Place data in full image coordinates
        rect = _active_rect()
        if background == "deformed" and rect:
            from server.deformed_warp import get_warped_full_data
            disp = session.displacement_results[idx]
            U, V = disp[:, :, 0], disp[:, :, 1]
            full_data = get_warped_full_data(
                data=disp_data, frame_idx=idx, U=U, V=V,
                roi_rect=rect, image_shape=(h, w),
                cache=session.inverse_map_cache,
            )
        else:
            full_data = np.full((h, w), np.nan)
            if rect:
                x0, y0, x1, y1 = rect
                dh, dw = disp_data.shape
                sh, sw = min(dh, y1 - y0), min(dw, x1 - x0)
                full_data[y0:y0 + sh, x0:x0 + sw] = disp_data[:sh, :sw]

        # Viewport downsampling (data only, no bg needed)
        if vw > 0 and vh > 0:
            scale = min(vw / w, vh / h, 1.0)
            if scale < 1.0:
                import cv2
                out_w, out_h = max(1, int(w * scale)), max(1, int(h * scale))
                mask = np.isfinite(full_data)
                filled = np.nan_to_num(full_data, nan=0.0).astype(np.float32)
                full_data = cv2.resize(filled, (out_w, out_h), interpolation=cv2.INTER_AREA).astype(np.float64)
                mask_small = cv2.resize(mask.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_AREA)
                full_data[mask_small < 0.5] = np.nan
                h, w = out_h, out_w

        mask = ~np.isnan(full_data)
        valid_vals = full_data[mask]
        # ... (auto vmin/vmax logic unchanged) ...
        # ... (colormap logic unchanged) ...

        # Alpha: 1.0 for valid, 0.0 for NaN (frontend applies user alpha via CSS)
        colored[:, :, 3] = mask.astype(np.float64)
        overlay_rgba = (colored * 255).astype(np.uint8)
        overlay_pil = Image.fromarray(overlay_rgba, "RGBA")

        buf = io.BytesIO()
        overlay_pil.save(buf, format="PNG")
        buf.seek(0)
        png_bytes = buf.read()
    else:
        # ... existing composited rendering (unchanged) ...
```

For the cache key, exclude `alpha` when `overlay_only`:
```python
    cache_params = {k: v for k, v in request.args.items() if k != "_t"}
    if overlay_only:
        cache_params.pop("alpha", None)
```

**Step 2: strain.py — same changes**

Mirror the same `overlay_only` branching in strain's `render_frame()`.

**Step 3: Run tests**

Run: `cd server && python -m pytest tests/ -v`
Expected: All pass (tests don't use overlay_only)

---

### Task 8: Viewport downsampling + Alpha — Frontend URL builders

**Files:**
- Modify: `frontend/src/api/displacement.ts`
- Modify: `frontend/src/api/strain.ts`
- Modify: `frontend/src/api/arrows.ts`
- Modify: `frontend/src/api/principal.ts`

No API signature changes needed — `params` is already `Record<string, string | number>`. The view components will simply pass `vw`, `vh`, and `overlay_only` in the params object. URL builders are generic and don't filter keys.

This task is a **no-op** — the URL builders already forward all params. No changes needed.

---

### Task 9: Frontend — usePreRenderCache changes

**Files:**
- Modify: `frontend/src/hooks/usePreRenderCache.ts`

**Changes:**

1. Accept `viewportSize` parameter: `{ vw: number; vh: number }`
2. Add `vw`, `vh`, `overlay_only: "true"` to `params` in `startPreRender()`
3. Remove `vis.alpha` from the dependency arrays (lines 158-159, 165-166)
4. Remove `alpha` from the `params` object in `startPreRender()`

In `buildFrameUrl` (L28-41), add `vw`/`vh` support — already handled since params are passed through.

In `startPreRender` (L83-159):
```typescript
const params: Record<string, string | number> = {
    component: displayComponent,
    colormap: vis.colormap,
    // alpha removed — applied via CSS
    background: vis.background,
    overlay_only: "true",
    ...(viewportSize.vw > 0 ? { vw: viewportSize.vw } : {}),
    ...(viewportSize.vh > 0 ? { vh: viewportSize.vh } : {}),
    // ... rest unchanged
};
```

Remove `vis.alpha` from the `useEffect` dependency arrays for invalidation (L162-166).

Update function signature:
```typescript
export function usePreRenderCache(
    componentOverride?: string,
    viewportSize?: { vw: number; vh: number },
): PreRenderState {
```

---

### Task 10: Frontend — DisplacementView layer separation

**Files:**
- Modify: `frontend/src/components/displacement/DisplacementView.tsx`

**Changes to DisplacementPanel:**

1. Add `containerSize` state with ResizeObserver (same pattern as PostProcessingView L63-72)
2. Render URL: use `overlay_only: "true"`, add `vw`/`vh`, remove `alpha`
3. Split image into two layers: background `<img>` + overlay `<img style="opacity: vis.alpha">`
4. Background src: `/api/images/reference` (for reference mode) or `/api/images/frame/${currentFrame + 1}` (for deformed mode)

Key JSX structure:
```tsx
<div ref={panelRef} className="relative flex-1 overflow-hidden ...">
  {/* Background layer */}
  <img src={bgSrc} className="max-w-full max-h-full object-contain" style={{transform}} />
  {/* Overlay layer */}
  <img src={overlaySrc} className="absolute inset-0 w-full h-full object-contain"
       style={{transform, opacity: vis.alpha}} />
</div>
```

Wait — the two images must be exactly aligned. Since both are the same dimensions (rendered at viewport size), and both have `object-contain` in the same container, they'll align. The overlay is positioned absolutely on top.

---

### Task 11: Frontend — PostProcessingView layer separation

**Files:**
- Modify: `frontend/src/components/postprocessing/PostProcessingView.tsx`

**Changes:**

1. Add background image `<img>` before the overlay `<img>` (inside the transform div, L391-402)
2. Overlay render URL: add `overlay_only: "true"`, `vw`, `vh`; remove `alpha`
3. Apply `style={{ opacity: vis.alpha }}` on the overlay `<img>` only
4. Change SVG viewBox from `naturalWidth/naturalHeight` to `imageWidth/imageHeight` from store
5. Change `screenToImage` to use `imageWidth/imageHeight` instead of `naturalWidth/naturalHeight`

Background src logic:
```tsx
const bgSrc = vis.background === "deformed" && currentFrame + 1 < numFrames
    ? `/api/images/frame/${currentFrame + 1}`
    : `/api/images/reference`;
```

Arrow and principal overlay images: also add `vw`/`vh` params.

SVG viewBox fix (L437):
```tsx
viewBox={`0 0 ${imageWidth} ${imageHeight}`}
```

screenToImage fix (L88-91):
```tsx
const scaleX = imageWidth / rect.width;
const scaleY = imageHeight / rect.height;
```

---

### Task 12: Frontend build and integration test

**Step 1: Build frontend**

Run: `cd frontend && npm run build`
Expected: Clean build, no TypeScript errors

**Step 2: Run all backend tests**

Run: `cd server && python -m pytest tests/ -v`
Expected: All pass

**Step 3: Manual smoke test**

Run: `python run_prod.py`
Then verify in browser:
- Load images, run displacement
- Verify overlay renders correctly with semi-transparent colormap over speckle background
- Drag alpha slider — should update instantly (no loading spinner)
- Change colormap — should re-render (with loading spinner, but fast)
- Check arrows/streamlines/principal directions render and align correctly
- Zoom in/out — verify probe coordinates still correct
- Download single frame — verify full resolution (not viewport-downsampled)

---

## Summary of all files changed

**Backend (Python):**
| File | Change |
|------|--------|
| `server/render_cache.py` | Add byte-based eviction |
| `server/viewport.py` | New: downsample helper |
| `server/routes/displacement.py` | Add viewport ds + overlay_only mode |
| `server/routes/strain.py` | Add viewport ds + overlay_only mode |
| `server/routes/arrows.py` | Viewport-sized figure |
| `server/routes/principal_dirs.py` | Viewport-sized figure |
| `server/tests/test_render_cache.py` | New: cache byte limit tests |
| `server/tests/test_viewport.py` | New: downsample tests |

**Frontend (TypeScript):**
| File | Change |
|------|--------|
| `frontend/src/hooks/usePreRenderCache.ts` | Add viewport params, remove alpha dep |
| `frontend/src/components/displacement/DisplacementView.tsx` | Layer separation, viewport params |
| `frontend/src/components/postprocessing/PostProcessingView.tsx` | Layer separation, viewport params, SVG viewBox fix |

**NOT changed (no impact):**
- Download/export endpoints (keep full resolution)
- Data endpoints (/frame, /range, /info)
- ROI, processing, probe endpoints
- appStore, page components, other hooks
