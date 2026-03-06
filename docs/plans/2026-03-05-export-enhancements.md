# Export System Enhancements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add probe data CSV export, chart image save, per-component vmin/vmax, single-frame save, export cancel, DPI control, path validation, and FileBrowser integration to the export system.

**Architecture:** Backend-driven export with new REST endpoints for probe CSV and single-frame downloads. Frontend gains download buttons (browser blob download pattern) for probes/charts/frames, plus UI enhancements to ExportDialog. Cancel uses a threading.Event flag checked per-frame in the export loop.

**Tech Stack:** Flask (backend), React + TypeScript + Zustand + Recharts (frontend), matplotlib (rendering), SocketIO (progress)

---

## Phase 1: Probe Data CSV Export (P0)

### Task 1: Backend — Probe time series CSV export endpoint

**Files:**
- Modify: `server/routes/probes.py` (after line 298)

**Step 1: Add CSV export endpoint**

Add to `server/routes/probes.py` after the kymograph endpoint (line 298):

```python
import csv
import io

@probes_bp.route("/export/csv", methods=["GET"])
def export_probe_csv():
    """Export probe time series data as CSV file download."""
    component = request.args.get("component", "u")
    probe_type = request.args.get("type", "point")
    metric = request.args.get("metric", "avg")
    include_extensometer = request.args.get("extensometer", "false") == "true"

    if not session.displacement_results:
        return jsonify({"error": "No displacement results"}), 400

    probes_of_type = [p for p in session.probes if p["type"] == probe_type]
    if not probes_of_type:
        return jsonify({"error": f"No {probe_type} probes placed"}), 400

    output = io.StringIO()
    writer = csv.writer(output)

    # Build time series data using existing _build_data_list helper
    num_frames = len(session.displacement_results)
    data_list = _build_data_list(component)

    if data_list is None:
        return jsonify({"error": f"Invalid component: {component}"}), 400

    # Header row
    header = ["frame"]
    for p in probes_of_type:
        pid = p["id"]
        header.append(f"{probe_type}_{pid}_{component}")
    writer.writerow(header)

    # Data rows
    scale, offset = _compute_scale_offset()
    for frame_idx in range(num_frames):
        row = [frame_idx + 1]
        data = data_list[frame_idx]
        for p in probes_of_type:
            value = _extract_probe_value(p, data, metric, scale, offset)
            row.append(value)
        writer.writerow(row)

    # If extensometer requested and probe_type is "line", append strain columns
    if include_extensometer and probe_type == "line":
        output.write("\n")
        ext_header = ["frame"]
        for p in probes_of_type:
            ext_header.append(f"line_{p['id']}_strain")
            ext_header.append(f"line_{p['id']}_length")
        writer.writerow(ext_header)

        for frame_idx in range(num_frames):
            row = [frame_idx + 1]
            for p in probes_of_type:
                strain_data = _compute_extensometer(p, frame_idx)
                row.append(strain_data["strain"])
                row.append(strain_data["length"])
            writer.writerow(row)

    csv_content = output.getvalue()
    from flask import Response
    return Response(
        csv_content,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=probes_{probe_type}_{component}.csv"},
    )
```

**Step 2: Add `_extract_probe_value` helper**

Add helper function before the endpoint (after existing helpers around line 88):

```python
def _extract_probe_value(probe, data_2d, metric, scale, offset):
    """Extract a single value from a probe on a 2D data field."""
    if data_2d is None:
        return None

    ptype = probe["type"]
    coords = probe["coords"]

    if ptype == "point":
        x, y = int(round(coords[0])), int(round(coords[1]))
        roi = session.roi_rect
        if roi:
            x -= roi[0]
            y -= roi[1]
        h, w = data_2d.shape
        if 0 <= y < h and 0 <= x < w:
            val = float(data_2d[y, x])
            return round(val * scale + offset, 6) if not (val != val) else None
        return None

    elif ptype == "line":
        import numpy as np
        p1, p2 = coords
        roi = session.roi_rect
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        if roi:
            x1 -= roi[0]; y1 -= roi[1]
            x2 -= roi[0]; y2 -= roi[1]
        num_samples = max(int(np.hypot(x2 - x1, y2 - y1)), 2)
        xs = np.linspace(x1, x2, num_samples).astype(int)
        ys = np.linspace(y1, y2, num_samples).astype(int)
        h, w = data_2d.shape
        mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        vals = data_2d[ys[mask], xs[mask]]
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return None
        if metric == "max":
            v = float(np.max(vals))
        elif metric == "min":
            v = float(np.min(vals))
        else:
            v = float(np.mean(vals))
        return round(v * scale + offset, 6)

    elif ptype == "area":
        import numpy as np
        area_mask = _get_area_mask(probe, data_2d.shape)
        if area_mask is None:
            return None
        vals = data_2d[area_mask]
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return None
        if metric == "max":
            v = float(np.max(vals))
        elif metric == "min":
            v = float(np.min(vals))
        else:
            v = float(np.mean(vals))
        return round(v * scale + offset, 6)

    return None


def _compute_extensometer(probe, frame_idx):
    """Compute extensometer strain and length for a line probe at a specific frame."""
    import numpy as np
    coords = probe["coords"]
    p1, p2 = np.array(coords[0], dtype=float), np.array(coords[1], dtype=float)
    initial_length = float(np.linalg.norm(p2 - p1))

    disp = session.displacement_results[frame_idx]
    roi = session.roi_rect

    def get_disp_at(pt):
        x, y = int(round(pt[0])), int(round(pt[1]))
        if roi:
            x -= roi[0]; y -= roi[1]
        h, w = disp.shape[:2]
        if 0 <= y < h and 0 <= x < w:
            return disp[y, x]  # (2,) — [u, v]
        return np.array([0.0, 0.0])

    d1 = get_disp_at(p1)
    d2 = get_disp_at(p2)
    new_p1 = p1 + d1
    new_p2 = p2 + d2
    current_length = float(np.linalg.norm(new_p2 - new_p1))
    strain = (current_length - initial_length) / initial_length if initial_length > 0 else 0.0

    return {"strain": round(strain, 8), "length": round(current_length, 4)}
```

**Step 3: Run server to verify no import errors**

Run: `cd server && python -c "from routes.probes import probes_bp; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add server/routes/probes.py
git commit -m "feat: add probe CSV export endpoint"
```

---

### Task 2: Frontend — Probe CSV download button

**Files:**
- Modify: `frontend/src/api/probes.ts` (add new function)
- Modify: `frontend/src/components/postprocessing/TimeSeriesChart.tsx` (add download button)

**Step 1: Add API function for CSV download**

Add to `frontend/src/api/probes.ts` (after the existing functions):

```typescript
export async function downloadProbeCSV(
  component: string,
  type: string,
  metric: string,
  includeExtensometer: boolean = false
): Promise<void> {
  const params = new URLSearchParams({
    component,
    type,
    metric,
    extensometer: includeExtensometer ? "true" : "false",
  });
  const response = await client.get(`/probes/export/csv?${params}`, {
    responseType: "blob",
  });
  const blob = new Blob([response.data], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `probes_${type}_${component}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
```

**Step 2: Add CSV download button to TimeSeriesChart**

Modify `frontend/src/components/postprocessing/TimeSeriesChart.tsx`:

Add import at top (after existing imports):
```typescript
import { downloadProbeCSV } from "@/api/probes";
import { Download } from "lucide-react";
import { useToast } from "@/components/shared/Toast";
```

Add download handler inside the component (after `const [metric, setMetric]` line 28):
```typescript
const { toast } = useToast();

const handleDownloadCSV = async () => {
  try {
    const probeType = showPointData ? "point" : showLineData ? "line" : "area";
    const includeExt = showLineData && lineMode === "strain";
    await downloadProbeCSV(displayComponent, probeType, metric, includeExt);
    toast("success", "CSV downloaded");
  } catch (e) {
    toast("error", "CSV download failed");
  }
};
```

Add download button in the header bar (after the metric `<select>` element, around line 167):
```tsx
{/* CSV Download button */}
<button
  onClick={handleDownloadCSV}
  className="p-0.5 rounded hover:bg-[var(--secondary)] text-[var(--muted-foreground)] hover:text-[var(--foreground)]"
  title="Download CSV"
>
  <Download size={12} />
</button>
```

**Step 3: Build and verify**

Run: `cd frontend && npm run build`
Expected: Build succeeds with no errors

**Step 4: Commit**

```bash
git add frontend/src/api/probes.ts frontend/src/components/postprocessing/TimeSeriesChart.tsx
git commit -m "feat: add probe CSV download button to time series chart"
```

---

## Phase 2: Chart Image Save (P0)

### Task 3: Chart PNG save via SVG-to-Canvas

**Files:**
- Modify: `frontend/src/components/postprocessing/TimeSeriesChart.tsx`

**Step 1: Add chart container ref and save handler**

In `TimeSeriesChart.tsx`, add ref and save function:

Add import:
```typescript
import { useRef, useCallback } from "react";
import { Camera } from "lucide-react";
```

Add ref (inside component, before `useEffect`):
```typescript
const chartRef = useRef<HTMLDivElement>(null);
```

Add save handler (after `handleDownloadCSV`):
```typescript
const handleSaveChartPNG = useCallback(async () => {
  const container = chartRef.current;
  if (!container) return;
  const svgElement = container.querySelector("svg.recharts-surface");
  if (!svgElement) {
    toast("error", "Chart not found");
    return;
  }

  try {
    const svgData = new XMLSerializer().serializeToString(svgElement);
    const svgBlob = new Blob([svgData], { type: "image/svg+xml;charset=utf-8" });
    const url = URL.createObjectURL(svgBlob);

    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      const scale = 2; // 2x for retina
      canvas.width = img.width * scale;
      canvas.height = img.height * scale;
      const ctx = canvas.getContext("2d")!;
      ctx.scale(scale, scale);
      ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--card").trim() || "#1e1e2e";
      ctx.fillRect(0, 0, img.width, img.height);
      ctx.drawImage(img, 0, 0);
      canvas.toBlob((blob) => {
        if (!blob) return;
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = `chart_${displayComponent}.png`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        toast("success", "Chart saved as PNG");
      }, "image/png");
    };
    img.src = url;
  } catch (e) {
    toast("error", "Failed to save chart");
  }
}, [displayComponent, toast]);
```

**Step 2: Wrap chart in ref container and add save button**

Wrap the chart's outer `<div>` with `ref={chartRef}`.

Add camera button next to the CSV download button:
```tsx
<button
  onClick={handleSaveChartPNG}
  className="p-0.5 rounded hover:bg-[var(--secondary)] text-[var(--muted-foreground)] hover:text-[var(--foreground)]"
  title="Save chart as PNG"
>
  <Camera size={12} />
</button>
```

**Step 3: Build and verify**

Run: `cd frontend && npm run build`
Expected: Build succeeds

**Step 4: Commit**

```bash
git add frontend/src/components/postprocessing/TimeSeriesChart.tsx
git commit -m "feat: add chart PNG save button to time series chart"
```

---

## Phase 3: Per-Component vmin/vmax in Image Export (P0)

### Task 4: Per-component range controls in ExportDialog

**Files:**
- Modify: `frontend/src/components/postprocessing/ExportDialog.tsx`

**Step 1: Replace single vmin/vmax with per-component ranges**

In `ExportDialog.tsx`, replace the `selectedComponents` state (line 58-60) with a richer structure:

```typescript
const [selectedComponents, setSelectedComponents] = useState<
  Record<string, { enabled: boolean; vmin: string; vmax: string; auto: boolean }>
>({});
```

Update `toggleComponent` (line 90-92):
```typescript
const toggleComponent = (key: string) => {
  setSelectedComponents((prev) => ({
    ...prev,
    [key]: prev[key]
      ? { ...prev[key], enabled: !prev[key].enabled }
      : { enabled: true, vmin: "", vmax: "", auto: true },
  }));
};

const updateComponentRange = (key: string, field: "vmin" | "vmax" | "auto", value: string | boolean) => {
  setSelectedComponents((prev) => ({
    ...prev,
    [key]: { ...prev[key] || { enabled: true, vmin: "", vmax: "", auto: true }, [field]: value },
  }));
};
```

Update `handleImageExport` (line 110-163) to build per-component ranges:
```typescript
const components: Record<string, { vmin?: number; vmax?: number }> = {};
for (const [key, comp] of Object.entries(selectedComponents)) {
  if (!comp?.enabled) continue;
  const entry: { vmin?: number; vmax?: number } = {};
  if (!comp.auto) {
    const vmin = parseFloat(comp.vmin);
    const vmax = parseFloat(comp.vmax);
    if (!isNaN(vmin)) entry.vmin = vmin;
    if (!isNaN(vmax)) entry.vmax = vmax;
  }
  components[key] = entry;
}
```

**Step 2: Update component checkbox UI to include inline range inputs**

Replace the checkbox grid (lines 236-265) with per-component rows:

```tsx
<div className="space-y-0.5">
  {[...DISPLACEMENT_COMPONENTS, ...availableStrain].map((c) => {
    const comp = selectedComponents[c.value];
    const enabled = !!comp?.enabled;
    return (
      <div key={c.value} className="space-y-0.5">
        <label className="flex items-center gap-1 text-[10px] text-[var(--foreground)] cursor-pointer">
          <input
            type="checkbox"
            checked={enabled}
            onChange={() => toggleComponent(c.value)}
            className="accent-[var(--primary)] w-3 h-3"
          />
          {c.label}
          {enabled && (
            <label className="ml-auto flex items-center gap-0.5 text-[9px] text-[var(--muted-foreground)]">
              <input
                type="checkbox"
                checked={comp?.auto ?? true}
                onChange={() => updateComponentRange(c.value, "auto", !(comp?.auto ?? true))}
                className="accent-[var(--primary)] w-2.5 h-2.5"
              />
              auto
            </label>
          )}
        </label>
        {enabled && !(comp?.auto ?? true) && (
          <div className="flex items-center gap-1 ml-4">
            <input
              type="text"
              value={comp?.vmin ?? ""}
              onChange={(e) => updateComponentRange(c.value, "vmin", e.target.value)}
              placeholder="min"
              className="w-14 h-5 bg-[var(--input)] border border-[#3a3d45] rounded px-1 text-[9px] text-[var(--foreground)]"
            />
            <span className="text-[9px] text-[var(--muted-foreground)]">~</span>
            <input
              type="text"
              value={comp?.vmax ?? ""}
              onChange={(e) => updateComponentRange(c.value, "vmax", e.target.value)}
              placeholder="max"
              className="w-14 h-5 bg-[var(--input)] border border-[#3a3d45] rounded px-1 text-[9px] text-[var(--foreground)]"
            />
          </div>
        )}
      </div>
    );
  })}
</div>
```

**Step 3: Build and verify**

Run: `cd frontend && npm run build`
Expected: Build succeeds

**Step 4: Commit**

```bash
git add frontend/src/components/postprocessing/ExportDialog.tsx
git commit -m "feat: per-component vmin/vmax in image export dialog"
```

---

## Phase 4: Single-Frame Save (P1)

### Task 5: Backend — Single-frame download endpoint

**Files:**
- Modify: `server/routes/displacement.py` (add new endpoint)
- Modify: `server/routes/strain.py` (add new endpoint)

**Step 1: Add download endpoint to displacement.py**

Add after the existing render endpoint:

```python
@displacement_bp.route("/download/<int:idx>", methods=["GET"])
def download_frame(idx):
    """Download a single rendered frame as PNG file (browser download)."""
    if not session.displacement_results or idx >= len(session.displacement_results):
        return jsonify({"error": "Invalid frame index"}), 400

    component = request.args.get("component", "u")
    colormap = request.args.get("colormap", "turbo")
    alpha = float(request.args.get("alpha", 0.7))
    vmin = request.args.get("vmin", type=float)
    vmax = request.args.get("vmax", type=float)
    background = request.args.get("background", "reference")
    log_scale = request.args.get("log_scale", "false") == "true"
    dpi = int(request.args.get("dpi", 150))

    # Reuse existing render logic to get data + background
    disp = session.displacement_results[idx]
    comp_map = {"u": 0, "v": 1}

    if component in comp_map:
        data = disp[:, :, comp_map[component]]
    elif component == "magnitude":
        import numpy as np
        data = np.sqrt(disp[:, :, 0] ** 2 + disp[:, :, 1] ** 2)
    elif component == "velocity" and idx > 0:
        import numpy as np
        prev = session.displacement_results[idx - 1]
        diff = disp - prev
        data = np.sqrt(diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2)
    else:
        import numpy as np
        data = np.zeros(disp.shape[:2])

    # Render with matplotlib at specified DPI
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from io import BytesIO

    bg_img = session.reference_image
    if background == "deformed" and idx + 1 < len(session.image_files):
        img_path = os.path.join(session.image_dir, session.image_files[idx + 1])
        bg_img = load_and_convert_image(img_path)

    h, w = data.shape
    roi = session.roi_rect
    fig_w = session.image_width / dpi
    fig_h = session.image_height / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    if bg_img is not None:
        ax.imshow(bg_img, cmap="gray")

    full = np.full((session.image_height, session.image_width), np.nan)
    if roi:
        x0, y0, x1, y1 = roi
        full[y0 : y0 + h, x0 : x0 + w] = data
    else:
        full[:h, :w] = data

    norm = None
    if log_scale:
        from matplotlib.colors import LogNorm
        abs_data = np.abs(full[np.isfinite(full)])
        if len(abs_data) > 0 and abs_data.max() > 0:
            norm = LogNorm(vmin=max(abs_data.min(), 1e-10), vmax=abs_data.max())

    im = ax.imshow(full, cmap=colormap, alpha=alpha, vmin=vmin, vmax=vmax, norm=norm)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"{component.upper()} — Frame {idx + 1}", fontsize=10)
    ax.axis("off")
    fig.tight_layout(pad=0.5)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    from flask import send_file
    return send_file(
        buf,
        mimetype="image/png",
        as_attachment=True,
        download_name=f"{component}_frame_{idx + 1:04d}.png",
    )
```

**Step 2: Add similar endpoint to strain.py**

Follow the same pattern but extract strain component data instead.

**Step 3: Commit**

```bash
git add server/routes/displacement.py server/routes/strain.py
git commit -m "feat: add single-frame download endpoints with DPI control"
```

---

### Task 6: Frontend — Save frame button on visualization views

**Files:**
- Modify: `frontend/src/api/export.ts` (add download function)
- Modify: `frontend/src/components/postprocessing/PostProcessingView.tsx` (add save button)

**Step 1: Add download API function**

Add to `frontend/src/api/export.ts`:

```typescript
export async function downloadSingleFrame(params: {
  idx: number;
  component: string;
  colormap: string;
  alpha: number;
  vmin?: number;
  vmax?: number;
  background: string;
  log_scale: boolean;
  dpi: number;
  isStrain?: boolean;
}): Promise<void> {
  const { idx, isStrain, ...rest } = params;
  const base = isStrain ? "/strain/download" : "/displacement/download";
  const queryParams = new URLSearchParams();
  for (const [k, v] of Object.entries(rest)) {
    if (v !== undefined && v !== null) queryParams.set(k, String(v));
  }
  const response = await client.get(`${base}/${idx}?${queryParams}`, {
    responseType: "blob",
  });
  const blob = new Blob([response.data], { type: "image/png" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${rest.component}_frame_${idx + 1}.png`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
```

**Step 2: Add save button to PostProcessingView header**

In `PostProcessingView.tsx`, add a "Save Frame" button in the header section (around line 346-357):

```tsx
import { Download } from "lucide-react";
import { downloadSingleFrame } from "@/api/export";
import { useToast } from "@/components/shared/Toast";
```

Add handler inside the component:
```typescript
const handleSaveFrame = async () => {
  try {
    const isStrain = !["u", "v", "magnitude", "velocity"].includes(displayComponent);
    await downloadSingleFrame({
      idx: currentFrame,
      component: displayComponent,
      colormap: vis.colormap,
      alpha: vis.alpha,
      background: vis.background,
      log_scale: vis.logScale,
      dpi: 300,
      isStrain,
    });
    toast("success", "Frame saved");
  } catch (e) {
    toast("error", "Failed to save frame");
  }
};
```

Add button next to the frame info in the header:
```tsx
<button
  onClick={handleSaveFrame}
  className="p-1 rounded hover:bg-[var(--secondary)] text-[var(--muted-foreground)] hover:text-[var(--foreground)]"
  title="Save current frame (300 DPI)"
>
  <Download size={14} />
</button>
```

**Step 3: Build and verify**

Run: `cd frontend && npm run build`
Expected: Build succeeds

**Step 4: Commit**

```bash
git add frontend/src/api/export.ts frontend/src/components/postprocessing/PostProcessingView.tsx
git commit -m "feat: add single-frame save button to post-processing view"
```

---

## Phase 5: Export Cancel + DPI Control (P1)

### Task 7: Backend — Cancellable batch export

**Files:**
- Modify: `server/session.py` (add cancel event)
- Modify: `server/routes/export.py` (add cancel endpoint + pass event)
- Modify: `raft_dic_gui/export_images.py` (check cancel flag per-frame)

**Step 1: Add cancel event to session**

In `server/session.py`, add to `AppSession.__init__` (or dataclass fields):

```python
import threading

# In the class fields:
export_cancel: threading.Event = field(default_factory=threading.Event)
```

**Step 2: Add cancel endpoint**

Add to `server/routes/export.py`:

```python
@export_bp.route("/images/cancel", methods=["POST"])
def cancel_export():
    """Cancel an in-progress batch image export."""
    if not session.export_active:
        return jsonify({"error": "No export in progress"}), 400
    session.export_cancel.set()
    return jsonify({"ok": True})
```

**Step 3: Pass cancel event to export_batch_images**

In `run_export()` inside `export.py`, pass the cancel event:

```python
result_dir = export_batch_images(
    ...,
    cancel_event=session.export_cancel,
)
```

And reset it before starting:
```python
session.export_cancel.clear()
```

**Step 4: Check cancel in export loop**

In `raft_dic_gui/export_images.py`, in the main frame loop of `export_batch_images()`, add:

```python
if cancel_event and cancel_event.is_set():
    if progress_callback:
        progress_callback(current, total, "Export cancelled")
    return output_dir  # Return partial results
```

**Step 5: Commit**

```bash
git add server/session.py server/routes/export.py raft_dic_gui/export_images.py
git commit -m "feat: add export cancel support with threading.Event"
```

---

### Task 8: DPI control in ExportDialog + backend

**Files:**
- Modify: `frontend/src/components/postprocessing/ExportDialog.tsx` (add DPI input)
- Modify: `raft_dic_gui/export_images.py` (use DPI from settings)

**Step 1: Add DPI input to ExportDialog**

In `ExportDialog.tsx`, add DPI field after frame range (around line 229):

```tsx
{/* DPI */}
<div className="flex items-center gap-1">
  <span className="text-[10px] text-[var(--muted-foreground)]">DPI:</span>
  <select
    value={dpi}
    onChange={(e) => setDpi(e.target.value)}
    className="px-1.5 py-0.5 rounded text-[10px] bg-[var(--input)] border border-[var(--border)] text-[var(--foreground)]"
  >
    <option value="100">100 (fast)</option>
    <option value="150">150 (default)</option>
    <option value="300">300 (publication)</option>
    <option value="600">600 (high-res)</option>
  </select>
</div>
```

Add state: `const [dpi, setDpi] = useState("150");`

Include in settings: `dpi: parseInt(dpi),`

**Step 2: Use DPI in backend render**

In `raft_dic_gui/export_images.py`, in `render_single_frame()`, replace hardcoded DPI:

```python
dpi = settings.get("dpi", 150)
fig.savefig(output_path, format=fmt, dpi=dpi, bbox_inches="tight")
```

**Step 3: Build and verify**

Run: `cd frontend && npm run build`
Expected: Build succeeds

**Step 4: Commit**

```bash
git add frontend/src/components/postprocessing/ExportDialog.tsx raft_dic_gui/export_images.py
git commit -m "feat: add DPI control to image export"
```

---

### Task 9: Frontend — Cancel button + progress bar

**Files:**
- Modify: `frontend/src/api/export.ts` (add cancel function)
- Modify: `frontend/src/components/postprocessing/ExportDialog.tsx` (cancel button + progress bar)

**Step 1: Add cancel API function**

Add to `frontend/src/api/export.ts`:

```typescript
export async function cancelExport(): Promise<void> {
  await client.post("/export/images/cancel");
}
```

**Step 2: Add cancel button and progress bar to ExportDialog**

Replace the export button area (lines 273-283) with:

```tsx
{exportActive ? (
  <div className="space-y-1">
    {/* Progress bar */}
    <div className="h-1.5 bg-[var(--input)] rounded-full overflow-hidden">
      <div
        className="h-full bg-[var(--primary)] transition-all duration-300"
        style={{ width: `${exportProgress}%` }}
      />
    </div>
    <div className="flex gap-1">
      <span className="text-[10px] text-[var(--muted-foreground)] flex-1">
        Exporting... {exportProgress.toFixed(0)}%
      </span>
      <button
        onClick={async () => {
          try { await cancelExport(); } catch {}
        }}
        className="px-2 py-1 bg-red-900/30 hover:bg-red-900/50 rounded text-[10px] text-red-400"
      >
        Cancel
      </button>
    </div>
  </div>
) : (
  <button onClick={handleImageExport} disabled={!imgDir.trim()} className={btnClass}>
    <Download size={12} /> Export Images
  </button>
)}
```

**Step 3: Build and verify**

Run: `cd frontend && npm run build`
Expected: Build succeeds

**Step 4: Commit**

```bash
git add frontend/src/api/export.ts frontend/src/components/postprocessing/ExportDialog.tsx
git commit -m "feat: add export cancel button and progress bar"
```

---

## Phase 6: Path Validation + FileBrowser Integration (P1)

### Task 10: Backend — Path validation

**Files:**
- Modify: `server/routes/export.py` (add validation)

**Step 1: Add path validation to scientific export**

In `export_scientific()` (line 16-60), add validation after `file_path` extraction:

```python
# Validate directory exists
export_dir = os.path.dirname(file_path)
if export_dir and not os.path.isdir(export_dir):
    return jsonify({"error": f"Directory does not exist: {export_dir}"}), 400

# Check for overwrite
if os.path.exists(file_path):
    overwrite = data.get("overwrite", False)
    if not overwrite:
        return jsonify({
            "error": "File already exists",
            "exists": True,
            "path": file_path,
        }), 409
```

**Step 2: Add path validation to image export**

In `export_images()` (line 63-126):

```python
# Validate and create output directory
parent_dir = os.path.dirname(output_dir.rstrip("/\\"))
if parent_dir and not os.path.isdir(parent_dir):
    return jsonify({"error": f"Parent directory does not exist: {parent_dir}"}), 400
```

**Step 3: Commit**

```bash
git add server/routes/export.py
git commit -m "feat: add path validation and overwrite check to export endpoints"
```

---

### Task 11: Frontend — Overwrite confirmation + FileBrowser for export paths

**Files:**
- Modify: `frontend/src/components/postprocessing/ExportDialog.tsx`

**Step 1: Add overwrite confirmation to scientific export**

Update `handleScientificExport`:

```typescript
const handleScientificExport = async () => {
  if (!sciPath.trim()) return;
  try {
    const result = await exportScientific({
      file_path: sciPath.trim(),
      upsample_strain: true,
    });
    toast("success", "Scientific data exported successfully");
  } catch (e: any) {
    if (e?.response?.status === 409 && e?.response?.data?.exists) {
      // File exists — ask to overwrite
      if (window.confirm(`File already exists:\n${sciPath}\n\nOverwrite?`)) {
        try {
          await exportScientific({
            file_path: sciPath.trim(),
            upsample_strain: true,
            metadata: {},
            overwrite: true,
          });
          toast("success", "Scientific data exported (overwritten)");
        } catch (e2) {
          const msg = e2 instanceof Error ? e2.message : "Unknown error";
          toast("error", `Export failed: ${msg}`);
        }
      }
    } else {
      const msg = e instanceof Error ? e.message : e?.response?.data?.error || "Unknown error";
      toast("error", `Scientific export failed: ${msg}`);
    }
  }
};
```

**Step 2: Add FileBrowser integration for export paths**

Add the existing `FileBrowser` component to let users browse directories:

```tsx
import { FileBrowser } from "@/components/shared/FileBrowser";

// State
const [browseTarget, setBrowseTarget] = useState<"sci" | "img" | null>(null);
```

Add browse buttons next to each path input:

```tsx
<div className="flex gap-1">
  <input type="text" value={sciPath} onChange={...} className={inputClass + " flex-1"} />
  <button
    onClick={() => setBrowseTarget("sci")}
    className="px-1.5 py-0.5 bg-[var(--secondary)] hover:bg-[var(--secondary)]/80 rounded text-[10px]"
  >
    ...
  </button>
</div>

{browseTarget && (
  <FileBrowser
    open
    onClose={() => setBrowseTarget(null)}
    onSelect={(path) => {
      if (browseTarget === "sci") { sciTouched.current = true; setSciPath(path); }
      else { imgTouched.current = true; setImgDir(path); }
      setBrowseTarget(null);
    }}
    initialPath={browseTarget === "sci" ? sciPath : imgDir}
  />
)}
```

**Step 3: Build and verify**

Run: `cd frontend && npm run build`
Expected: Build succeeds

**Step 4: Commit**

```bash
git add frontend/src/components/postprocessing/ExportDialog.tsx frontend/src/api/export.ts
git commit -m "feat: add overwrite confirmation and FileBrowser to export paths"
```

---

## Task Summary

| Task | Phase | Description | Priority |
|------|-------|-------------|----------|
| 1 | P1 | Backend probe CSV export endpoint | P0 |
| 2 | P1 | Frontend probe CSV download button | P0 |
| 3 | P2 | Chart PNG save via SVG-to-Canvas | P0 |
| 4 | P3 | Per-component vmin/vmax in ExportDialog | P0 |
| 5 | P4 | Backend single-frame download endpoint | P1 |
| 6 | P4 | Frontend save frame button | P1 |
| 7 | P5 | Cancellable batch export | P1 |
| 8 | P5 | DPI control | P1 |
| 9 | P5 | Cancel button + progress bar UI | P1 |
| 10 | P6 | Backend path validation | P1 |
| 11 | P6 | Overwrite confirm + FileBrowser | P1 |
