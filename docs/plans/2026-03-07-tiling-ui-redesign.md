# Tiling UI Redesign & Status Dashboard

**Date:** 2026-03-07
**Status:** Design Complete, Ready for Implementation

## Problem Statement

Current tiling parameters (Context Pad, Tile Overlap, Smooth σ, Safety Factor) are:
- Cryptically named — users don't understand what they control
- Missing tooltips and explanations
- Lacking real-time visual feedback (no tile grid preview)
- Missing status information (ROI size, GPU info, memory estimates, timing)
- Safety Factor is disconnected from actual behavior (only applied at model selection)
- Backend has redundant T computation logic and insufficient OOM recovery

## Design Decisions

| Decision | Choice | Alternatives Considered |
|----------|--------|------------------------|
| Control approach | Mixed: main sliders + auto-computed values with click-to-override | A) Simplified 1-2 controls, B) Keep 4 params with better names |
| Tile visualization | Overlay on ROI canvas with toggle | B) Separate minimap, C) Both |
| Status info location | All in parameter panel | B) Split static/dynamic across panel and bottom bar |
| Main sliders | Two independent: Memory Budget + Blend Quality | A) Single Memory slider, B) Quality↔Speed tradeoff |
| Expert mode | Click-to-edit computed values with reset icon | B) Traditional expand panel, C) No expert mode |

---

## Part 1: Two Main Sliders

### Memory Budget Slider

**Semantic:** Controls `p_max_pixels` (max pixels per single inference call).

- **Range:** `64×64 = 4,096` → GPU-allowed maximum (from `estimate_safe_pmax()` at model selection)
- **Scale labels:** Show equivalent square side lengths: `256×256`, `512×512`, `1024×1024`, `2048×2048`
- **Default position:** Auto-set to `safe_pmax` (recommended value), marked "Recommended"
- **Over-budget warning:** Slider can go beyond recommended value; area beyond turns yellow/orange with tooltip "May cause OOM"
- **Live update:** Dragging updates status panel in real-time (tile size, count, estimated memory)

### Blend Quality Slider

**Semantic:** Controls tile overlap ratio and smoothing sigma.

- **Range:** 3 ticks or continuous — `Fast` (overlap=16, σ=1.0) → `Balanced` (overlap=64, σ=1.5) → `High` (overlap=128, σ=2.0)
- **Default position:** `Balanced`
- **Live update:** Dragging updates overlap ratio and tile count in status panel

**Indirect coupling:** Larger overlap increases tile count (smaller stride) but doesn't change per-tile memory. Status panel reflects this.

---

## Part 2: Auto-Computed Values with Click-to-Override

### Computed Parameters Panel

Displayed below the two sliders. Compact two-column grid:

```
┌─────────────────────────────────────┐
│  Tile Size        1024 × 1024  [↺]  │
│  Tile Overlap          64 px   [↺]  │
│  ROI Margin            64 px   [↺]  │
│  Smoothing Radius      1.5     [↺]  │
└─────────────────────────────────────┘
```

### Renamed Parameters

| Old Name | New Name | Tooltip |
|----------|----------|---------|
| Tile Size | **Tile Size** | (keep) |
| Tile Overlap | **Tile Overlap** | "Number of overlapping pixels between adjacent tiles. Larger = smoother blending but more computation" |
| Context Padding | **ROI Margin** | "Extra pixels beyond ROI boundary for model context. Improves accuracy at ROI edges" |
| Smooth σ | **Smoothing Radius** | "Gaussian smoothing radius. Larger = smoother displacement field but less detail. Set to 0 to disable" |
| Safety Factor | **(removed)** | Redundant with Memory Budget slider directly controlling p_max_pixels |

### Interaction Rules

- **Default state:** Gray text, no icon. Value auto-computed from slider position, follows slider changes.
- **Click value:** Becomes editable input. User types custom value, presses Enter to confirm.
- **Overridden state:** White text (emphasis color), `↺` reset icon appears. Value decoupled from slider — no longer follows slider changes.
- **Click `↺`:** Revert to auto-computed value, re-link to slider.
- **Independent:** Each value manages its own override state independently.

### Computation Logic

| Slider | Computes |
|--------|----------|
| Memory Budget | `Tile Size = √p_max_pixels` |
| Blend Quality | `Tile Overlap = f(quality_level)`, `Smoothing Radius = g(quality_level)` |
| Independent default | `ROI Margin = 64` (not linked to any slider, but can be manually overridden) |

### Tooltip Strategy

Each parameter name has a `ⓘ` icon. Hover shows one-sentence explanation: "what it controls" + "effect of increasing/decreasing".

---

## Part 3: Tiling Status Panel

Below computed parameters. Always visible. Two sections: static (updates on param/ROI change) and runtime (appears during processing).

### Static Information

```
┌─ Tiling Status ─────────────────────┐
│  Image          1920 × 1080 px      │
│  ROI Area       812 × 645 px        │
│  Work Area      940 × 773 px        │
│  Tiling         3 × 2 = 6 tiles     │
│  Overlap Ratio  6.7%                │
│  ─────────────────────────────────  │
│  GPU            RTX 3060 12GB       │
│  VRAM Free      8.2 / 12.0 GB      │
│  Est. Memory    ~1.8 GB per tile    │
│  Est. Time      ~42s (6 tiles ×     │
│                 180 frames)         │
└─────────────────────────────────────┘
```

**Update triggers:**
- ROI change → ROI Area, Work Area, Tiling grid, Est. Time
- Slider change → Tiling grid, Overlap Ratio, Est. Memory, Est. Time
- Model change → GPU info, VRAM, all computed values

**Est. Time logic:**
- Model selection runs a small dummy inference to benchmark per-pixel time
- `Est. Time ≈ per_pixel_time × tile_pixels × num_tiles × num_frames`
- Displayed with `~` prefix to indicate approximation

### Runtime Information (appears when processing starts)

```
│  ─── Running ───────────────────── │
│  Frame           23 / 180          │
│  Tile            4 / 6             │
│  VRAM Usage      4.1 / 12.0 GB    │
│  Elapsed         12.4s             │
│  Remaining       ~29.6s            │
└─────────────────────────────────────┘
```

Delivered via existing WebSocket `processing:progress` events with extended payload.

---

## Part 4: ROI Canvas Tile Grid Overlay

### Trigger Conditions

- ROI confirmed + model selected → tile grid can be computed
- "Show Tile Grid" toggle in parameter panel, **default off**

### Visual Design

- **Tile boundaries:** White dashed lines, 1px, `opacity: 0.6`
- **Overlap regions:** Blue semi-transparent fill `rgba(59,130,246,0.15)`
- **Work Area boundary** (ROI bbox + ROI Margin): Orange dashed line, distinguishes ROI from expanded computation area
- **ROI outline:** Existing green ROI boundary unchanged
- **Tile numbers:** Small font `#1`, `#2`... at each tile center, `opacity: 0.5`

### Implementation

- **Frontend-only computation:** Replicate `_generate_tile_starts` logic as a pure TypeScript function
- **Rendering:** SVG `<rect>` elements on existing ROI overlay layer
- **Coordinates:** Follow canvas zoom/pan transform
- **No backend call needed:** Only requires `wC`, `hC` (work area size), `T` (tile side), `overlap` — all available from current parameters

---

## Part 5: Backend Bug Fixes & API Changes

### Bug Fixes

**1. T computation logic** (`processing.py:361-363`)

Current: `T = max(max_T, ...); T = min(T, max_T)` — T always equals max_T.

Fix:
```python
T = max_T
if T < tile_overlap * 2 + 16:
    tile_overlap = max(4, T // 4)
```

**2. Safety Factor disconnect — eliminated**

Safety Factor removed from UI. `p_max_pixels` controlled directly by Memory Budget slider. `estimate_safe_pmax()` only called at model selection for slider default/recommendation.

**3. OOM recovery enhancement**

Current: Only halves iteration count (minimal memory impact).

Fix: OOM → clear CUDA cache → halve `p_max_pixels` → recompute tile grid → retry current frame. Notify frontend via WebSocket of grid change.

**4. Degenerate stride guard**

`_generate_tile_starts`: `stride ≤ 0` → raise explicit error instead of fallback to `stride=1`. Upstream overlap clamping prevents this, but defensive error is clearer.

### New API Endpoint

**`GET /api/processing/tiling-preview`**

Returns all status panel data from current session state:

```json
{
  "image_size": [1920, 1080],
  "roi_bbox": [120, 80, 812, 645],
  "work_area": { "x": 56, "y": 16, "w": 940, "h": 773 },
  "tile_size": 1024,
  "tiles": [[0,0,1024,773], [916,0,1024,773]],
  "tile_count": 2,
  "overlap_ratio": 0.067,
  "gpu": { "name": "RTX 3060", "total_mb": 12288, "free_mb": 8400 },
  "est_memory_per_tile_mb": 1800,
  "est_time_seconds": 42.0,
  "is_single_shot": false
}
```

### Extended WebSocket Payload

`processing:progress` event:

```json
{
  "percent": 12.8,
  "current": 23,
  "total": 180,
  "tile_current": 4,
  "tile_total": 6,
  "vram_used_mb": 4100,
  "vram_total_mb": 12288,
  "elapsed_seconds": 12.4,
  "est_remaining_seconds": 29.6
}
```

### Frontend/Backend Default Sync

Update `DICConfig` defaults to match new UI defaults. Remove discrepancy between frontend (64/64) and backend (32/32).

---

## Part 6: Legacy Code Cleanup

### Delete

| File | Content | Reason |
|------|---------|--------|
| `processing.py` | `cut_image_pair_with_flow()` | Old tiling implementation, unused |
| `processing.py` | `process_image_pair()` | Old single-shot function, replaced by `dic_over_roi_with_tiling` |
| `config.py` | `use_crop`, `crop_size`, `shift` | Old crop system params, unrelated to current tiling |
| `config.py` | `safety_factor` | Replaced by direct `p_max_pixels` control |
| `config.py` | `allow_shared_memory` | Never referenced by any code |
| `processing.py:configure()` | `safety_factor` handling | Field deleted |

### Retain

| File | Content | Reason |
|------|---------|--------|
| `processing.py` | `calculate_window_positions()` | Still used by `preview.py` (Tk GUI) |
| `model.py` | `estimate_safe_pmax()` | Needed for slider default/recommendation at model selection |

### Frontend Cleanup

| File | Change |
|------|--------|
| `ProcessingParams.tsx` | Remove `safetyFactor` state, `showAdvanced` toggle; replace with new sliders + panels |
| `api/processing.ts` | Remove `safety_factor` sending |
| `stores/appStore.ts` | Clean up any `safetyFactor` state |

---

## Implementation Order

1. **Backend bug fixes** — T logic, OOM recovery, stride guard, default sync
2. **Legacy cleanup** — Delete dead code and fields
3. **New API endpoint** — `/api/processing/tiling-preview`
4. **WebSocket payload extension** — Add tile/VRAM/timing fields
5. **Frontend: Sliders** — Memory Budget + Blend Quality with linking logic
6. **Frontend: Computed Parameters** — Click-to-edit with override/reset
7. **Frontend: Status Panel** — Static + runtime info display
8. **Frontend: Tile Grid Overlay** — SVG rendering on ROI canvas
9. **Integration testing** — Verify slider↔backend↔canvas↔status pipeline
