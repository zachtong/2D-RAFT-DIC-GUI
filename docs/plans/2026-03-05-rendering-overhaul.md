# Rendering Overhaul: Pre-render + Decoupled Playback

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Decouple frame rendering from playback so all frames are pre-rendered at maximum speed, and playback becomes instant scrubbing through a local cache.

**Architecture:** Backend gains a `/api/prerender/start` endpoint that renders all frames in a background thread and emits SocketIO progress. Frontend adds a `PreRenderCache` (Map of frame index to blob URL) that eagerly fetches all frames. Playback switches from "request on display" to "display from cache". Secondary improvements: pan support for displacement/post-processing views, safer cache keys, and image prefetch.

**Tech Stack:** Flask + SocketIO (backend), React + Zustand + TypeScript (frontend), existing PIL/matplotlib rendering pipeline.

---

## Problem Statement

Current architecture couples **rendering** and **playback**:

```
setInterval(1000/fps) -> setCurrentFrame(n+1) -> React render -> <img src=NEW_URL>
                                                                      |
                                                            browser HTTP request
                                                                      |
                                                            Flask renders PNG (20-300ms)
                                                                      |
                                                            image displayed
```

Issues:
1. At low playback speed (0.5fps), user waits 2s between frames even if render takes 100ms
2. At high playback speed (10fps), render can't keep up, frames pile up
3. Every frame change triggers a fresh HTTP request (no browser caching due to `_t=Date.now()`)
4. No prefetching — each frame starts rendering only when displayed

Target architecture:

```
User clicks "Play" or loads results
        |
   [Pre-render phase]                    [Playback phase]
   Backend renders all frames            Frontend cycles cached blob URLs
   at max speed in background     ->     at user-chosen speed
   Progress bar shown                    Instant frame switching
        |                                      |
   All frames cached as                  setInterval only changes
   blob URLs in browser memory           which cached image to show
```

---

## Phase 1: Backend Pre-render Endpoint (server-side)

### Task 1.1: Add pre-render route module

**Files:**
- Create: `server/routes/prerender.py`
- Modify: `server/app.py` (register blueprint)

**Step 1: Create the pre-render blueprint**

```python
"""Pre-render all frames for a given component + settings, emitting progress via SocketIO."""

import threading
import io

import numpy as np
from flask import Blueprint, jsonify, request
from PIL import Image
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from raft_dic_gui.processing import load_and_convert_image
from server.session import session
from server.render_cache import RenderCache

prerender_bp = Blueprint("prerender", __name__)

# Shared state for the background pre-render job
_prerender_lock = threading.Lock()
_prerender_job = {
    "active": False,
    "cancel": False,
    "progress": 0,
    "total": 0,
    "done_frames": [],  # list of frame indices that are ready
}


def _render_single_frame(idx: int, component: str, params: dict) -> bytes:
    """Render one displacement/strain frame to PNG bytes. Mirrors displacement.py/strain.py render logic."""
    from server.routes.displacement import _get_displacement_component
    import os

    colormap_name = params.get("colormap", "turbo")
    alpha = float(params.get("alpha", 0.7))
    vmin = params.get("vmin")
    vmax = params.get("vmax")
    background = params.get("background", "reference")
    log_scale = str(params.get("log_scale", "false")).lower() in ("true", "1", "yes")

    STRAIN_COMPONENTS = {"exx", "eyy", "exy", "e1", "e2", "max_shear", "von_mises", "rotation"}
    is_strain = component in STRAIN_COMPONENTS

    if is_strain:
        if not session.strain_results or idx >= len(session.strain_results):
            return b""
        comp_data = session.strain_results[idx].get(component)
        if comp_data is None:
            return b""
        disp_data = comp_data
        # Upsample if needed
        if session.roi_rect:
            x0, y0, x1, y1 = session.roi_rect
            target_h, target_w = y1 - y0, x1 - x0
            if disp_data.shape[0] != target_h or disp_data.shape[1] != target_w:
                import cv2
                disp_data = cv2.resize(disp_data.astype(np.float64),
                                       (target_w, target_h),
                                       interpolation=cv2.INTER_LINEAR)
    else:
        if not session.displacement_results or idx >= len(session.displacement_results):
            return b""
        disp_data = _get_displacement_component(idx, component)

    # Load background
    if background == "deformed" and idx + 1 < len(session.image_files):
        bg_img = session.deformed_view_cache.get_deformed_image(idx + 1)
        if bg_img is None:
            bg_path = os.path.join(session.image_dir, session.image_files[idx + 1])
            bg_img = load_and_convert_image(bg_path)
    else:
        bg_img = session.reference_image

    if bg_img is None:
        return b""

    h, w = bg_img.shape[:2]

    if background == "deformed" and session.roi_rect:
        from server.deformed_warp import get_warped_full_data
        disp = session.displacement_results[idx]
        U, V = disp[:, :, 0], disp[:, :, 1]
        full_data = get_warped_full_data(
            data=disp_data, frame_idx=idx, U=U, V=V,
            roi_rect=session.roi_rect, image_shape=(h, w),
            cache=session.inverse_map_cache,
        )
    else:
        full_data = np.full((h, w), np.nan)
        if session.roi_rect:
            rx0, ry0, rx1, ry1 = session.roi_rect
            dh, dw = disp_data.shape
            sh = min(dh, ry1 - ry0)
            sw = min(dw, rx1 - rx0)
            full_data[ry0:ry0 + sh, rx0:rx0 + sw] = disp_data[:sh, :sw]

    mask = ~np.isnan(full_data)
    valid = full_data[mask]

    # vmin/vmax
    v_lo = float(vmin) if vmin is not None else (float(valid.min()) if valid.size > 0 else 0.0)
    v_hi = float(vmax) if vmax is not None else (float(valid.max()) if valid.size > 0 else 1.0)
    if v_lo >= v_hi:
        v_hi = v_lo + 1e-10

    # Background PIL
    if bg_img.ndim == 2:
        bg_pil = Image.fromarray(bg_img).convert("RGB")
    elif bg_img.shape[2] == 4:
        bg_pil = Image.fromarray(bg_img[:, :, :3])
    else:
        bg_pil = Image.fromarray(bg_img)

    # Colormap
    cmap_obj = cm.get_cmap(colormap_name)
    if log_scale:
        lmin = v_lo if v_lo > 0 else 1e-10
        lmax = v_hi if v_hi > 0 else 1.0
        if lmin >= lmax:
            lmin = lmax / 1000
        norm = mcolors.LogNorm(vmin=lmin, vmax=lmax)
    else:
        norm = mcolors.Normalize(vmin=v_lo, vmax=v_hi)

    normalized = norm(np.nan_to_num(full_data, nan=0.0))
    colored = cmap_obj(normalized)
    colored[:, :, 3] = alpha * mask.astype(np.float64)
    overlay = Image.fromarray((colored * 255).astype(np.uint8), "RGBA")

    result = bg_pil.copy()
    result.paste(overlay, (0, 0), overlay)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()


@prerender_bp.route("/start", methods=["POST"])
def start_prerender():
    """Start background pre-rendering of all frames for given component + params."""
    from server.app import socketio

    with _prerender_lock:
        if _prerender_job["active"]:
            return jsonify({"error": "Pre-render already in progress"}), 409

    body = request.get_json(silent=True) or {}
    component = body.get("component", "u")
    params = body.get("params", {})

    num_frames = len(session.displacement_results) if session.displacement_results else 0
    if num_frames == 0:
        return jsonify({"error": "No results to pre-render"}), 404

    with _prerender_lock:
        _prerender_job.update({
            "active": True,
            "cancel": False,
            "progress": 0,
            "total": num_frames,
            "done_frames": [],
        })

    def run():
        try:
            for i in range(num_frames):
                if _prerender_job["cancel"]:
                    break
                _render_single_frame(i, component, params)
                with _prerender_lock:
                    _prerender_job["progress"] = i + 1
                    _prerender_job["done_frames"].append(i)
                socketio.emit("prerender:progress", {
                    "current": i + 1,
                    "total": num_frames,
                    "percent": round((i + 1) / num_frames * 100),
                })
        finally:
            with _prerender_lock:
                _prerender_job["active"] = False
            socketio.emit("prerender:complete", {
                "total": num_frames,
                "cancelled": _prerender_job["cancel"],
            })

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"ok": True, "total": num_frames})


@prerender_bp.route("/stop", methods=["POST"])
def stop_prerender():
    """Cancel in-progress pre-render."""
    with _prerender_lock:
        _prerender_job["cancel"] = True
    return jsonify({"ok": True})


@prerender_bp.route("/status", methods=["GET"])
def prerender_status():
    """Return current pre-render job status."""
    with _prerender_lock:
        return jsonify({
            "active": _prerender_job["active"],
            "progress": _prerender_job["progress"],
            "total": _prerender_job["total"],
        })
```

**Step 2: Register the blueprint in `server/app.py`**

Add to the blueprint registration block:
```python
from server.routes.prerender import prerender_bp
app.register_blueprint(prerender_bp, url_prefix="/api/prerender")
```

**Step 3: Test manually**

```bash
cd server && python -m pytest tests/ -v
```

**Step 4: Commit**

```bash
git add server/routes/prerender.py server/app.py
git commit -m "feat: add /api/prerender endpoint for background frame pre-rendering"
```

---

### Task 1.2: Fix cache key safety (replace `id()` with hash)

**Files:**
- Modify: `server/routes/displacement.py:114`
- Modify: `server/routes/strain.py` (same pattern)
- Modify: `server/routes/arrows.py` (same pattern)

**Step 1: Replace `id()` with a stable session version counter**

In `server/session.py`, add a version counter that increments on each new result load:

```python
# Add to AppSession.__init__:
self.result_version: int = 0

# In the method that sets displacement_results (or wherever results are loaded):
# Increment: self.result_version += 1
```

**Step 2: Update cache key in displacement.py**

```python
# displacement.py line 114 — change:
cache_key = (id(session.displacement_results), idx, tuple(sorted(cache_params.items())))
# to:
cache_key = (session.result_version, idx, tuple(sorted(cache_params.items())))
```

Apply the same pattern in `strain.py` and `arrows.py`.

**Step 3: Clear render caches on new result load**

In `server/routes/processing.py` where `session.displacement_results` is set, add:
```python
from server.routes.displacement import _render_cache as disp_cache
from server.routes.strain import _render_cache as strain_cache
from server.routes.arrows import _render_cache as arrow_cache

disp_cache.clear()
strain_cache.clear()
arrow_cache.clear()
session.result_version += 1
```

**Step 4: Commit**

```bash
git commit -m "fix: replace fragile id()-based cache keys with stable version counter"
```

---

## Phase 2: Frontend Pre-render Cache + Decoupled Playback

### Task 2.1: Create `usePreRenderCache` hook

**Files:**
- Create: `frontend/src/hooks/usePreRenderCache.ts`

This hook manages a Map of `frame_index -> blob_url` and drives the pre-fetch pipeline.

**Step 1: Implement the hook**

```typescript
import { useState, useEffect, useRef, useCallback } from "react";
import { useAppStore } from "@/stores/appStore";

export interface PreRenderState {
  /** Blob URL for a given frame, or undefined if not yet cached */
  getFrame: (idx: number) => string | undefined;
  /** Whether pre-rendering is in progress */
  isPreRendering: boolean;
  /** Progress: 0-100 */
  progress: number;
  /** Number of frames cached */
  cachedCount: number;
  /** Total frames to cache */
  totalFrames: number;
  /** Start pre-rendering with current vis settings */
  startPreRender: () => void;
  /** Cancel in-progress pre-render */
  cancelPreRender: () => void;
  /** Invalidate cache (e.g. when settings change) */
  invalidate: () => void;
}

/**
 * Build the render URL for a single frame (same logic as renderUrl/strainRenderUrl
 * but WITHOUT the _t cache-buster so we can reuse browser cache).
 */
function buildFrameUrl(
  idx: number,
  component: string,
  params: Record<string, string | number>,
  isStrain: boolean
): string {
  const base = isStrain ? "/api/strain/render" : "/api/displacement/render";
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
  }
  qs.set("component", component);
  return `${base}/${idx}?${qs}`;
}

const STRAIN_COMPONENTS = new Set([
  "exx", "eyy", "exy", "e1", "e2", "max_shear", "von_mises", "rotation",
]);

export function usePreRenderCache(): PreRenderState {
  const numFrames = useAppStore((s) => s.numFrames);
  const hasResults = useAppStore((s) => s.hasResults);
  const displayComponent = useAppStore((s) => s.displayComponent);
  const vis = useAppStore((s) => s.visSettings);

  const [isPreRendering, setIsPreRendering] = useState(false);
  const [progress, setProgress] = useState(0);
  const [cachedCount, setCachedCount] = useState(0);

  // Map<frameIndex, blobUrl>
  const cacheRef = useRef<Map<number, string>>(new Map());
  const abortRef = useRef<AbortController | null>(null);
  // Track what settings the cache was built for
  const cacheKeyRef = useRef<string>("");

  const getFrame = useCallback((idx: number) => {
    return cacheRef.current.get(idx);
  }, []);

  const invalidate = useCallback(() => {
    // Revoke all blob URLs to free memory
    for (const url of cacheRef.current.values()) {
      URL.revokeObjectURL(url);
    }
    cacheRef.current.clear();
    setCachedCount(0);
    setProgress(0);
    cacheKeyRef.current = "";
  }, []);

  const cancelPreRender = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    setIsPreRendering(false);
  }, []);

  const startPreRender = useCallback(() => {
    if (!hasResults || numFrames === 0) return;

    // Build a key representing current settings
    const isStrain = STRAIN_COMPONENTS.has(displayComponent);
    const params: Record<string, string | number> = {
      colormap: vis.colormap,
      alpha: vis.alpha,
      background: vis.background,
      ...(vis.fixedRange && vis.vminU ? { vmin: vis.vminU } : {}),
      ...(vis.fixedRange && vis.vmaxU ? { vmax: vis.vmaxU } : {}),
      ...(vis.logScale ? { log_scale: "true" } : {}),
    };
    const settingsKey = `${displayComponent}|${JSON.stringify(params)}`;

    // If cache already matches, skip
    if (cacheKeyRef.current === settingsKey && cacheRef.current.size === numFrames) {
      return;
    }

    // Invalidate old cache
    invalidate();
    cacheKeyRef.current = settingsKey;

    // Abort any in-flight pre-render
    if (abortRef.current) abortRef.current.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setIsPreRendering(true);
    setProgress(0);

    // Fetch frames concurrently with limited parallelism
    const CONCURRENCY = 3;
    let nextIdx = 0;
    let doneCount = 0;

    const fetchFrame = async () => {
      while (nextIdx < numFrames) {
        if (controller.signal.aborted) return;
        const idx = nextIdx++;
        const url = buildFrameUrl(idx, displayComponent, params, isStrain);
        try {
          const resp = await fetch(url, { signal: controller.signal });
          if (!resp.ok) continue;
          const blob = await resp.blob();
          if (controller.signal.aborted) return;
          const blobUrl = URL.createObjectURL(blob);
          cacheRef.current.set(idx, blobUrl);
          doneCount++;
          setCachedCount(doneCount);
          setProgress(Math.round((doneCount / numFrames) * 100));
        } catch {
          // Aborted or network error — skip
          if (controller.signal.aborted) return;
        }
      }
    };

    // Launch concurrent workers
    const workers = Array.from({ length: Math.min(CONCURRENCY, numFrames) }, fetchFrame);
    Promise.all(workers).then(() => {
      if (!controller.signal.aborted) {
        setIsPreRendering(false);
      }
    });
  }, [hasResults, numFrames, displayComponent, vis, invalidate]);

  // Invalidate cache when component or key vis settings change
  useEffect(() => {
    invalidate();
  }, [displayComponent, vis.colormap, vis.alpha, vis.background,
      vis.fixedRange, vis.vminU, vis.vmaxU, vis.logScale, invalidate]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cancelPreRender();
      // eslint-disable-next-line react-hooks/exhaustive-deps
      for (const url of cacheRef.current.values()) {
        URL.revokeObjectURL(url);
      }
    };
  }, [cancelPreRender]);

  return {
    getFrame,
    isPreRendering,
    progress,
    cachedCount,
    totalFrames: numFrames,
    startPreRender,
    cancelPreRender,
    invalidate,
  };
}
```

**Step 2: Commit**

```bash
git add frontend/src/hooks/usePreRenderCache.ts
git commit -m "feat: add usePreRenderCache hook for eager frame caching"
```

---

### Task 2.2: Rewrite `useFrameNav` — wait-for-ready playback

**Files:**
- Modify: `frontend/src/hooks/useFrameNav.ts`

The key change: playback no longer uses blind `setInterval`. Instead it uses `requestAnimationFrame` + timestamp tracking, and only advances when the frame is ready in cache.

**Step 1: Rewrite `useFrameNav`**

```typescript
import { useCallback, useEffect, useRef, useState } from "react";
import { useAppStore } from "@/stores/appStore";

interface FrameNavOptions {
  /** Check if a frame is ready to display (e.g. cached). If not provided, always ready. */
  isFrameReady?: (idx: number) => boolean;
}

export function useFrameNav(options?: FrameNavOptions) {
  const numFrames = useAppStore((s) => s.numFrames);
  const currentFrame = useAppStore((s) => s.currentFrame);
  const setCurrentFrame = useAppStore((s) => s.setCurrentFrame);

  const playRef = useRef(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const fpsRef = useRef(5);
  const rafRef = useRef<number | null>(null);
  const lastFrameTimeRef = useRef(0);
  const isFrameReadyRef = useRef(options?.isFrameReady);

  // Keep ref in sync
  useEffect(() => {
    isFrameReadyRef.current = options?.isFrameReady;
  }, [options?.isFrameReady]);

  const next = useCallback(() => {
    setCurrentFrame(Math.min(currentFrame + 1, numFrames - 1));
  }, [currentFrame, numFrames, setCurrentFrame]);

  const prev = useCallback(() => {
    setCurrentFrame(Math.max(currentFrame - 1, 0));
  }, [currentFrame, setCurrentFrame]);

  const first = useCallback(() => setCurrentFrame(0), [setCurrentFrame]);
  const last = useCallback(
    () => setCurrentFrame(numFrames - 1),
    [numFrames, setCurrentFrame]
  );

  const goTo = useCallback(
    (f: number) => setCurrentFrame(Math.max(0, Math.min(f, numFrames - 1))),
    [numFrames, setCurrentFrame]
  );

  const stopLoop = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, []);

  const pause = useCallback(() => {
    playRef.current = false;
    setIsPlaying(false);
    stopLoop();
  }, [stopLoop]);

  const play = useCallback(() => {
    if (playRef.current) return;

    const state = useAppStore.getState();
    if (state.currentFrame >= state.numFrames - 1) {
      useAppStore.setState({ currentFrame: 0 });
    }

    playRef.current = true;
    setIsPlaying(true);
    lastFrameTimeRef.current = performance.now();

    const tick = (now: number) => {
      if (!playRef.current) return;

      const interval = 1000 / fpsRef.current;
      const elapsed = now - lastFrameTimeRef.current;

      if (elapsed >= interval) {
        const s = useAppStore.getState();
        const nextFrame = s.currentFrame + 1;

        if (nextFrame >= s.numFrames) {
          // Reached end
          playRef.current = false;
          setIsPlaying(false);
          return;
        }

        // Check if next frame is ready (cached)
        const ready = isFrameReadyRef.current
          ? isFrameReadyRef.current(nextFrame)
          : true;

        if (ready) {
          useAppStore.setState({ currentFrame: nextFrame });
          lastFrameTimeRef.current = now - (elapsed - interval); // drift correction
        }
        // If not ready, skip this tick — will retry on next rAF
      }

      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
  }, []);

  const setFps = useCallback((fps: number) => {
    fpsRef.current = fps;
    // No need to restart — rAF loop adapts automatically
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === "ArrowRight") next();
      else if (e.key === "ArrowLeft") prev();
      else if (e.key === " ") {
        e.preventDefault();
        playRef.current ? pause() : play();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [next, prev, play, pause]);

  // Cleanup on unmount
  useEffect(() => {
    return () => stopLoop();
  }, [stopLoop]);

  return {
    currentFrame, numFrames, next, prev, first, last, goTo,
    play, pause, isPlaying, setFps,
  };
}
```

**Step 2: Commit**

```bash
git commit -m "feat: rewrite useFrameNav with rAF loop and frame-ready gating"
```

---

### Task 2.3: Integrate pre-render cache into PostProcessingView

**Files:**
- Modify: `frontend/src/components/postprocessing/PostProcessingView.tsx`
- Modify: `frontend/src/pages/PostProcessingPage.tsx` (to pass cache down or lift hook)

**Step 1: Wire up the pre-render cache**

In `PostProcessingPage.tsx` (or in `PostProcessingView` directly):

```typescript
import { usePreRenderCache } from "@/hooks/usePreRenderCache";
import { useFrameNav } from "@/hooks/useFrameNav";

// Inside component:
const cache = usePreRenderCache();
const isFrameReady = useCallback((idx: number) => !!cache.getFrame(idx), [cache]);
const frameNav = useFrameNav({ isFrameReady });
```

In `PostProcessingView.tsx`, change the image `src` logic:

```typescript
// Before:
const src = isStrain ? strainRenderUrl(currentFrame, renderParams) : renderUrl(currentFrame, renderParams);

// After:
const cachedSrc = cache.getFrame(currentFrame);
const liveSrc = isStrain
  ? strainRenderUrl(currentFrame, renderParams)
  : renderUrl(currentFrame, renderParams);
const src = cachedSrc ?? liveSrc;
```

**Step 2: Add pre-render trigger**

Auto-start pre-rendering when results become available or when the user navigates to the post-processing tab:

```typescript
useEffect(() => {
  if (hasResults && numFrames > 0) {
    cache.startPreRender();
  }
}, [hasResults, numFrames, displayComponent]); // re-start when component changes
```

**Step 3: Add progress indicator in FramePlayback**

```typescript
// In FramePlayback component, show a thin progress bar under the controls:
{cache.isPreRendering && (
  <div className="h-0.5 bg-[var(--border)]">
    <div
      className="h-full bg-[var(--primary)] transition-all duration-200"
      style={{ width: `${cache.progress}%` }}
    />
  </div>
)}
```

**Step 4: Commit**

```bash
git commit -m "feat: integrate pre-render cache into PostProcessingView"
```

---

### Task 2.4: Integrate pre-render cache into DisplacementView

**Files:**
- Modify: `frontend/src/components/displacement/DisplacementView.tsx`
- Modify: `frontend/src/pages/DisplacementPage.tsx`

**Step 1: Apply same pattern to DisplacementPanel**

DisplacementView renders TWO panels (U and V). Each needs its own cache. The approach:

```typescript
// In DisplacementPage or DisplacementView:
// Option A: Two caches (one per component) — simple but 2x memory
// Option B: Single cache for the currently active pair — less memory

// Recommendation: Option A for simplicity. Memory is ~2 * N * PNG_size.
// For 100 frames of 1024x1024 PNG: ~100MB per component = 200MB total. Acceptable.
```

Each `DisplacementPanel` creates its own `usePreRenderCache` scoped to its component. This requires a small refactor to accept `component` as prop to the hook, or to have the hook read from a parameter rather than from `displayComponent` in the store.

**Alternative (simpler):** Add a `componentOverride` parameter to `usePreRenderCache`:

```typescript
export function usePreRenderCache(componentOverride?: string): PreRenderState {
  const storeComponent = useAppStore((s) => s.displayComponent);
  const displayComponent = componentOverride ?? storeComponent;
  // ... rest of hook uses displayComponent
}
```

**Step 2: Commit**

```bash
git commit -m "feat: integrate pre-render cache into DisplacementView"
```

---

## Phase 3: Remove `_t=Date.now()` Cache Busting

### Task 3.1: Switch to version-based cache keys in frontend

**Files:**
- Modify: `frontend/src/api/displacement.ts`
- Modify: `frontend/src/api/strain.ts`
- Modify: `frontend/src/api/arrows.ts`

**Step 1: Replace `_t=Date.now()` with result version**

Add a `resultVersion` field to `appStore` (set when results load or vis settings change):

```typescript
// In appStore:
resultVersion: number;  // Incremented when results load
visVersion: number;     // Incremented when vis settings change

// setResults action:
setResults: (numFrames) => set((s) => ({
  hasResults: numFrames > 0,
  numFrames,
  currentFrame: 0,
  resultVersion: s.resultVersion + 1,
})),

// updateVisSettings action:
updateVisSettings: (partial) => set((s) => ({
  visSettings: { ...s.visSettings, ...partial },
  visVersion: s.visVersion + 1,
})),
```

Then in renderUrl:
```typescript
export function renderUrl(idx: number, params: Record<string, string | number>): string {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
  }
  // No more _t=Date.now() — the server-side LRU cache handles freshness,
  // and the pre-render cache handles client-side freshness via invalidation.
  return `/api/displacement/render/${idx}?${qs}`;
}
```

**Why this is safe:** The pre-render cache hook already invalidates when settings change. For non-cached (live) requests, the URL changes naturally when parameters change because the query string changes.

**Step 2: Update server Cache-Control headers**

In `server/serializers.py`, change:
```python
# From:
resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
# To:
resp.headers["Cache-Control"] = "private, max-age=60"
```

This allows the browser to cache rendered frames for 60 seconds. The pre-render cache handles invalidation explicitly.

**Step 3: Commit**

```bash
git commit -m "feat: replace _t cache-busting with version-based cache invalidation"
```

---

## Phase 4: Pan Support for Displacement/PostProcessing Views

### Task 4.1: Add pan + zoom to PostProcessingView

**Files:**
- Modify: `frontend/src/stores/appStore.ts` (add pan state)
- Modify: `frontend/src/components/postprocessing/PostProcessingView.tsx`

**Step 1: Add pan offset to appStore**

```typescript
// In AppState interface:
viewOffset: { x: number; y: number };

// Initial value:
viewOffset: { x: 0, y: 0 },

// Action:
setViewOffset: (offset: { x: number; y: number }) => set({ viewOffset: offset }),

// In resetSession:
viewOffset: { x: 0, y: 0 },

// In zoomReset:
zoomReset: () => set({ viewZoom: 1, viewOffset: { x: 0, y: 0 } }),
```

**Step 2: Apply transform in PostProcessingView**

```typescript
const viewOffset = useAppStore((s) => s.viewOffset);
const setViewOffset = useAppStore((s) => s.setViewOffset);

// In the container div style:
style={{
  transform: `translate(${viewOffset.x}px, ${viewOffset.y}px) scale(${viewZoom})`,
  transformOrigin: "0 0",
}}
```

**Step 3: Add middle-button pan handler**

```typescript
const panRef = useRef<{ startX: number; startY: number; origX: number; origY: number } | null>(null);

const handleMouseDown = useCallback((e: React.MouseEvent) => {
  if (e.button === 1) { // Middle mouse
    e.preventDefault();
    panRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      origX: viewOffset.x,
      origY: viewOffset.y,
    };
  }
}, [viewOffset]);

const handleMouseMoveWithPan = useCallback((e: React.MouseEvent) => {
  if (panRef.current) {
    const dx = e.clientX - panRef.current.startX;
    const dy = e.clientY - panRef.current.startY;
    setViewOffset({
      x: panRef.current.origX + dx,
      y: panRef.current.origY + dy,
    });
    return; // Don't handle probe placement while panning
  }
  handleMouseMove(e); // existing probe placement handler
}, [handleMouseMove, setViewOffset]);

const handleMouseUp = useCallback(() => {
  panRef.current = null;
}, []);
```

**Step 4: Add scroll-wheel zoom (matching ROI Canvas)**

```typescript
const handleWheel = useCallback((e: React.WheelEvent) => {
  e.preventDefault();
  const factor = e.deltaY < 0 ? 1.1 : 0.9;
  const state = useAppStore.getState();
  const newZoom = Math.max(0.2, Math.min(5, state.viewZoom * factor));

  // Zoom toward mouse position
  const rect = imageAreaRef.current?.getBoundingClientRect();
  if (rect) {
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const newX = mx - ((mx - state.viewOffset.x) / state.viewZoom) * newZoom;
    const newY = my - ((my - state.viewOffset.y) / state.viewZoom) * newZoom;
    useAppStore.setState({ viewZoom: newZoom, viewOffset: { x: newX, y: newY } });
  } else {
    useAppStore.setState({ viewZoom: newZoom });
  }
}, []);
```

**Step 5: Commit**

```bash
git commit -m "feat: add pan + scroll-wheel zoom to displacement/postprocessing views"
```

---

## Phase 5: Supplementary Improvements

### Task 5.1: Matplotlib figure cleanup (prevent memory leak)

**Files:**
- Modify: `server/routes/arrows.py`

**Step 1: Wrap figure creation in try/finally**

```python
fig = None
try:
    fig, ax = plt.subplots(...)
    # ... render quiver/streamlines ...
    buf = io.BytesIO()
    fig.savefig(buf, ...)
    png_bytes = buf.getvalue()
finally:
    if fig is not None:
        plt.close(fig)
```

**Step 2: Commit**

```bash
git commit -m "fix: ensure matplotlib figures are closed in arrows.py to prevent memory leak"
```

---

### Task 5.2: Add image memory cache for frame serving

**Files:**
- Modify: `server/session.py` (add image LRU cache)
- Modify: `server/routes/images.py` (use cache)

**Step 1: Add image cache to session**

```python
# In session.py, add:
from functools import lru_cache

class AppSession:
    def __init__(self):
        # ... existing ...
        self._image_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._image_cache_max = 50  # Keep last 50 frames in memory

    def get_cached_image(self, idx: int) -> np.ndarray | None:
        if idx in self._image_cache:
            self._image_cache.move_to_end(idx)
            return self._image_cache[idx]
        return None

    def cache_image(self, idx: int, img: np.ndarray):
        self._image_cache[idx] = img
        self._image_cache.move_to_end(idx)
        while len(self._image_cache) > self._image_cache_max:
            self._image_cache.popitem(last=False)
```

**Step 2: Use cache in images.py frame route**

```python
@images_bp.route("/frame/<int:index>")
def get_frame(index):
    cached = session.get_cached_image(index)
    if cached is not None:
        return png_response(image_to_png_bytes(cached))
    img = load_and_convert_image(path)
    session.cache_image(index, img)
    return png_response(image_to_png_bytes(img))
```

**Step 3: Commit**

```bash
git commit -m "feat: add in-memory image cache for frame serving (LRU, 50 frames)"
```

---

## Phase 6: Testing & Verification

### Task 6.1: Manual integration test checklist

After all phases are complete, verify:

- [ ] Load a dataset (100+ frames)
- [ ] Navigate to Displacement tab — pre-render starts automatically
- [ ] Progress bar fills up under playback controls
- [ ] Press Play at 5x — playback is smooth, no loading spinners
- [ ] Change colormap — cache invalidates, pre-render restarts
- [ ] Switch to PostProcessing tab — pre-render starts for current component
- [ ] Place a probe while pre-rendering in background
- [ ] Middle-click drag to pan in PostProcessing view
- [ ] Scroll-wheel zoom maintains mouse focus
- [ ] Press Esc during probe placement — returns to normal mode
- [ ] Load new dataset — all caches clear properly
- [ ] Check browser DevTools Network tab: no redundant requests during playback

---

## Implementation Order Summary

| Phase | Task | Priority | Estimated Effort | Dependencies |
|-------|------|----------|------------------|--------------|
| 1.1 | Backend pre-render endpoint | P0 | Medium | None |
| 1.2 | Fix cache key safety | P0 | Small | None |
| 2.1 | `usePreRenderCache` hook | P0 | Medium | None |
| 2.2 | Rewrite `useFrameNav` with rAF | P0 | Medium | None |
| 2.3 | Integrate into PostProcessingView | P0 | Medium | 2.1, 2.2 |
| 2.4 | Integrate into DisplacementView | P1 | Small | 2.1, 2.2 |
| 3.1 | Remove `_t` cache busting | P1 | Small | 2.1 |
| 4.1 | Pan + zoom for post-processing | P1 | Medium | None |
| 5.1 | Matplotlib figure cleanup | P2 | Small | None |
| 5.2 | Image memory cache | P2 | Small | None |
| 6.1 | Integration testing | P0 | Medium | All above |

**Critical path:** 1.2 → 2.1 → 2.2 → 2.3 → 6.1 (this is the minimum for decoupled playback)

**Can be parallelized:** Tasks 1.1, 1.2, 2.1, 2.2, 4.1, 5.1, 5.2 have no mutual dependencies.
