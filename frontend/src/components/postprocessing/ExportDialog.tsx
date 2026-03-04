import { useState, useEffect, useRef } from "react";
import { CollapsibleSection } from "@/components/shared/CollapsibleSection";
import { SmallInput } from "@/components/shared/SmallInput";
import { useAppStore } from "@/stores/appStore";
import { useToast } from "@/components/shared/Toast";
import { exportScientific, exportImages } from "@/api/export";
import { Download } from "lucide-react";
import type { DisplayComponent } from "@/types/api";

const DISPLACEMENT_COMPONENTS: { value: DisplayComponent; label: string }[] = [
  { value: "u", label: "U (horizontal)" },
  { value: "v", label: "V (vertical)" },
  { value: "magnitude", label: "Magnitude" },
  { value: "velocity", label: "Velocity" },
];

const STRAIN_COMPONENTS: { value: DisplayComponent; label: string }[] = [
  { value: "exx", label: "εxx" },
  { value: "eyy", label: "εyy" },
  { value: "exy", label: "εxy" },
  { value: "e1", label: "ε1" },
  { value: "e2", label: "ε2" },
  { value: "max_shear", label: "Max Shear" },
  { value: "von_mises", label: "Von Mises" },
  { value: "rotation", label: "Rotation" },
];

const inputClass =
  "w-full h-6 bg-[var(--input)] border border-[#3a3d45] rounded px-1.5 text-[10px] text-[var(--foreground)] focus:border-[var(--primary)] focus:outline-none";

const btnClass =
  "w-full flex items-center justify-center gap-1 py-1.5 bg-[var(--secondary)] hover:bg-[var(--secondary)]/80 rounded text-[11px] text-[var(--foreground)] disabled:opacity-40";

export function ExportDialog() {
  const { toast } = useToast();

  const imageDir = useAppStore((s) => s.imageDir);
  const numFrames = useAppStore((s) => s.numFrames);
  const exportActive = useAppStore((s) => s.exportActive);
  const exportProgress = useAppStore((s) => s.exportProgress);
  const displayComponent = useAppStore((s) => s.displayComponent);
  const hasStrain = useAppStore((s) => s.hasStrain);
  const strainComponents = useAppStore((s) => s.strainComponents);
  const vis = useAppStore((s) => s.visSettings);
  const arrows = useAppStore((s) => s.arrowSettings);

  // --- Scientific export state ---
  const [sciPath, setSciPath] = useState("");
  const sciTouched = useRef(false);

  // --- Image export state ---
  const [imgDir, setImgDir] = useState("");
  const imgTouched = useRef(false);
  const [frameStart, setFrameStart] = useState("1");
  const [frameEnd, setFrameEnd] = useState(String(Math.max(1, numFrames)));

  // --- Component selection ---
  const [selectedComponents, setSelectedComponents] = useState<
    Record<string, boolean>
  >({});

  // Default paths based on imageDir (only if user hasn't manually edited)
  useEffect(() => {
    if (imageDir) {
      const normalized = imageDir.replace(/\\/g, "/").replace(/\/+$/, "");
      if (!sciTouched.current) setSciPath(`${normalized}/results.mat`);
      if (!imgTouched.current) setImgDir(`${normalized}/export/`);
    }
  }, [imageDir]);

  // Update frameEnd when numFrames changes
  useEffect(() => {
    setFrameEnd(String(Math.max(1, numFrames)));
  }, [numFrames]);

  // Initialize selected components with current displayComponent
  useEffect(() => {
    setSelectedComponents((prev) => {
      // Only set initial default if nothing is selected yet
      if (Object.values(prev).some(Boolean)) return prev;
      return { [displayComponent]: true };
    });
  }, [displayComponent]);

  // Available strain components (intersection of computed and known)
  const availableStrain = hasStrain
    ? STRAIN_COMPONENTS.filter((c) => strainComponents.includes(c.value))
    : [];

  const toggleComponent = (key: string) => {
    setSelectedComponents((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  // --- Handlers ---

  const handleScientificExport = async () => {
    if (!sciPath.trim()) return;
    try {
      await exportScientific({
        file_path: sciPath.trim(),
        upsample_strain: true,
      });
      toast("success", "Scientific data exported successfully");
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Unknown error";
      toast("error", `Scientific export failed: ${msg}`);
    }
  };

  const handleImageExport = async () => {
    if (!imgDir.trim()) return;

    // Build components dict from selected checkboxes
    const components: Record<string, { vmin?: number; vmax?: number }> = {};
    for (const [key, checked] of Object.entries(selectedComponents)) {
      if (!checked) continue;
      const entry: { vmin?: number; vmax?: number } = {};
      if (vis.fixedRange) {
        const vmin = parseFloat(vis.vminU);
        const vmax = parseFloat(vis.vmaxU);
        if (!isNaN(vmin)) entry.vmin = vmin;
        if (!isNaN(vmax)) entry.vmax = vmax;
      }
      components[key] = entry;
    }

    if (Object.keys(components).length === 0) {
      toast("error", "Select at least one component to export");
      return;
    }

    // Build WYSIWYG settings from current visualization state
    const settings: Record<string, unknown> = {
      colormap: vis.colormap,
      alpha: vis.alpha,
      display_mode: vis.background,
      use_physical_units: vis.physicalEnabled,
      physical_ratio: vis.physicalRatio,
      physical_unit: vis.physicalUnit,
      fps: vis.fps,
      use_log_scale: vis.logScale,
      show_quiver: arrows.showQuiver,
      show_streamlines: arrows.showStreamlines,
      arrow_spacing: arrows.spacing,
      arrow_scale: arrows.scale,
      arrow_color: arrows.color,
      arrow_width: arrows.lineWidth,
      include_colorbar: true,
      format: "png",
    };

    try {
      await exportImages({
        output_dir: imgDir.trim(),
        components,
        frame_range: [parseInt(frameStart), parseInt(frameEnd)],
        settings,
      });
      toast("success", "Image export started");
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Unknown error";
      toast("error", `Image export failed: ${msg}`);
    }
  };

  return (
    <CollapsibleSection title="Export" defaultOpen={false}>
      {/* --- Scientific Data Export --- */}
      <div className="space-y-1">
        <span className="text-[10px] font-medium text-[var(--muted-foreground)] uppercase tracking-wide">
          Scientific Data
        </span>
        <input
          type="text"
          value={sciPath}
          onChange={(e) => {
            sciTouched.current = true;
            setSciPath(e.target.value);
          }}
          placeholder="Export path (.mat or .npz)"
          className={inputClass}
        />
        <button
          onClick={handleScientificExport}
          disabled={!sciPath.trim()}
          className={btnClass}
        >
          <Download size={12} /> Export .mat / .npz
        </button>
      </div>

      {/* --- Divider --- */}
      <div className="h-px bg-[var(--border)] my-2" />

      {/* --- Visualization Image Export --- */}
      <div className="space-y-1.5">
        <span className="text-[10px] font-medium text-[var(--muted-foreground)] uppercase tracking-wide">
          Visualization Images
        </span>

        {/* Output directory */}
        <input
          type="text"
          value={imgDir}
          onChange={(e) => {
            imgTouched.current = true;
            setImgDir(e.target.value);
          }}
          placeholder="Output directory"
          className={inputClass}
        />

        {/* Frame range */}
        <div className="flex items-center gap-1">
          <span className="text-[10px] text-[var(--muted-foreground)]">
            Frames:
          </span>
          <SmallInput
            value={frameStart}
            onChange={setFrameStart}
            className="w-10"
          />
          <span className="text-[10px] text-[var(--muted-foreground)]">—</span>
          <SmallInput
            value={frameEnd}
            onChange={setFrameEnd}
            className="w-10"
          />
        </div>

        {/* Component checkboxes */}
        <div className="space-y-1">
          <span className="text-[10px] text-[var(--muted-foreground)]">
            Components:
          </span>
          <div className="grid grid-cols-2 gap-x-2 gap-y-0.5">
            {DISPLACEMENT_COMPONENTS.map((c) => (
              <label
                key={c.value}
                className="flex items-center gap-1 text-[10px] text-[var(--foreground)] cursor-pointer"
              >
                <input
                  type="checkbox"
                  checked={!!selectedComponents[c.value]}
                  onChange={() => toggleComponent(c.value)}
                  className="accent-[var(--primary)] w-3 h-3"
                />
                {c.label}
              </label>
            ))}
            {availableStrain.map((c) => (
              <label
                key={c.value}
                className="flex items-center gap-1 text-[10px] text-[var(--foreground)] cursor-pointer"
              >
                <input
                  type="checkbox"
                  checked={!!selectedComponents[c.value]}
                  onChange={() => toggleComponent(c.value)}
                  className="accent-[var(--primary)] w-3 h-3"
                />
                {c.label}
              </label>
            ))}
          </div>
        </div>

        {/* WYSIWYG note */}
        <p className="text-[9px] text-[var(--muted-foreground)] italic">
          Uses current visualization settings (colormap, overlay, arrows)
        </p>

        {/* Export button */}
        <button
          onClick={handleImageExport}
          disabled={!imgDir.trim() || exportActive}
          className={btnClass}
        >
          <Download size={12} />
          {exportActive
            ? `Exporting... ${exportProgress.toFixed(0)}%`
            : "Export Images"}
        </button>
      </div>
    </CollapsibleSection>
  );
}
