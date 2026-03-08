import { useState, useEffect, useRef } from "react";
import { CollapsibleSection } from "@/components/shared/CollapsibleSection";
import { SmallInput } from "@/components/shared/SmallInput";
import { FileBrowser } from "@/components/shared/FileBrowser";
import { useAppStore } from "@/stores/appStore";
import { useToast } from "@/components/shared/Toast";
import { exportScientific, exportImages, cancelExport } from "@/api/export";
import { Download, FolderOpen } from "lucide-react";
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
  { value: "dexx_dt", label: "dεxx/dt" },
  { value: "deyy_dt", label: "dεyy/dt" },
  { value: "dexy_dt", label: "dεxy/dt" },
];

interface CompEntry {
  enabled: boolean;
  vmin: string;
  vmax: string;
  auto: boolean;
}

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
  const [dpi, setDpi] = useState("150");

  // --- Component selection with per-component ranges ---
  const [selectedComponents, setSelectedComponents] = useState<
    Record<string, CompEntry>
  >({});

  // --- FileBrowser state ---
  const [browseTarget, setBrowseTarget] = useState<"sci" | "img" | null>(null);

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
      if (Object.values(prev).some((c) => c.enabled)) return prev;
      return { [displayComponent]: { enabled: true, vmin: "", vmax: "", auto: true } };
    });
  }, [displayComponent]);

  // Available strain components (intersection of computed and known)
  const availableStrain = hasStrain
    ? STRAIN_COMPONENTS.filter((c) => strainComponents.includes(c.value))
    : [];

  const toggleComponent = (key: string) => {
    setSelectedComponents((prev) => ({
      ...prev,
      [key]: prev[key]
        ? { ...prev[key], enabled: !prev[key].enabled }
        : { enabled: true, vmin: "", vmax: "", auto: true },
    }));
  };

  const updateComponentRange = (
    key: string,
    field: "vmin" | "vmax" | "auto",
    value: string | boolean
  ) => {
    setSelectedComponents((prev) => ({
      ...prev,
      [key]: {
        ...(prev[key] || { enabled: true, vmin: "", vmax: "", auto: true }),
        [field]: value,
      },
    }));
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
    } catch (e: any) {
      if (e?.response?.status === 409 && e?.response?.data?.exists) {
        if (window.confirm(`File already exists:\n${sciPath}\n\nOverwrite?`)) {
          try {
            await exportScientific({
              file_path: sciPath.trim(),
              upsample_strain: true,
              overwrite: true,
            });
            toast("success", "Scientific data exported (overwritten)");
          } catch (e2) {
            const msg = e2 instanceof Error ? e2.message : "Unknown error";
            toast("error", `Export failed: ${msg}`);
          }
        }
      } else {
        const msg = e?.response?.data?.error || (e instanceof Error ? e.message : "Unknown error");
        toast("error", `Scientific export failed: ${msg}`);
      }
    }
  };

  const handleImageExport = async () => {
    if (!imgDir.trim()) return;

    // Build components dict with per-component ranges
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
      dpi: parseInt(dpi),
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

  const handleCancel = async () => {
    try {
      await cancelExport();
    } catch {
      // ignore — export may have already completed
    }
  };

  const allComponents = [...DISPLACEMENT_COMPONENTS, ...availableStrain];

  return (
    <CollapsibleSection title="Export" defaultOpen={false}>
      {/* --- Scientific Data Export --- */}
      <div className="space-y-1">
        <span className="text-[10px] font-medium text-[var(--muted-foreground)] uppercase tracking-wide">
          Scientific Data
        </span>
        <div className="flex gap-1">
          <input
            type="text"
            value={sciPath}
            onChange={(e) => {
              sciTouched.current = true;
              setSciPath(e.target.value);
            }}
            placeholder="Export path (.mat or .npz)"
            className={inputClass + " flex-1"}
          />
          <button
            onClick={() => setBrowseTarget("sci")}
            className="px-1.5 shrink-0 bg-[var(--secondary)] hover:bg-[var(--secondary)]/80 rounded text-[var(--muted-foreground)]"
            title="Browse"
          >
            <FolderOpen size={12} />
          </button>
        </div>
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
        <div className="flex gap-1">
          <input
            type="text"
            value={imgDir}
            onChange={(e) => {
              imgTouched.current = true;
              setImgDir(e.target.value);
            }}
            placeholder="Output directory"
            className={inputClass + " flex-1"}
          />
          <button
            onClick={() => setBrowseTarget("img")}
            className="px-1.5 shrink-0 bg-[var(--secondary)] hover:bg-[var(--secondary)]/80 rounded text-[var(--muted-foreground)]"
            title="Browse"
          >
            <FolderOpen size={12} />
          </button>
        </div>

        {/* Frame range + DPI */}
        <div className="flex items-center gap-1">
          <span className="text-[10px] text-[var(--muted-foreground)]">Frames:</span>
          <SmallInput value={frameStart} onChange={setFrameStart} className="w-10" />
          <span className="text-[10px] text-[var(--muted-foreground)]">—</span>
          <SmallInput value={frameEnd} onChange={setFrameEnd} className="w-10" />
          <div className="w-px h-3 bg-[var(--border)] mx-0.5" />
          <span className="text-[10px] text-[var(--muted-foreground)]">DPI:</span>
          <select
            value={dpi}
            onChange={(e) => setDpi(e.target.value)}
            className="px-1 py-0.5 rounded text-[9px] bg-[var(--input)] border border-[var(--border)] text-[var(--foreground)]"
          >
            <option value="100">100</option>
            <option value="150">150</option>
            <option value="300">300</option>
            <option value="600">600</option>
          </select>
        </div>

        {/* Component checkboxes with per-component range */}
        <div className="space-y-0.5">
          <span className="text-[10px] text-[var(--muted-foreground)]">Components:</span>
          {allComponents.map((c) => {
            const comp = selectedComponents[c.value];
            const enabled = !!comp?.enabled;
            return (
              <div key={c.value}>
                <div className="flex items-center gap-1">
                  <label className="flex items-center gap-1 text-[10px] text-[var(--foreground)] cursor-pointer flex-1">
                    <input
                      type="checkbox"
                      checked={enabled}
                      onChange={() => toggleComponent(c.value)}
                      className="accent-[var(--primary)] w-3 h-3"
                    />
                    {c.label}
                  </label>
                  {enabled && (
                    <label className="flex items-center gap-0.5 text-[9px] text-[var(--muted-foreground)] cursor-pointer">
                      <input
                        type="checkbox"
                        checked={comp?.auto ?? true}
                        onChange={() =>
                          updateComponentRange(c.value, "auto", !(comp?.auto ?? true))
                        }
                        className="accent-[var(--primary)] w-2.5 h-2.5"
                      />
                      auto
                    </label>
                  )}
                </div>
                {enabled && !(comp?.auto ?? true) && (
                  <div className="flex items-center gap-1 ml-4 mt-0.5">
                    <input
                      type="text"
                      value={comp?.vmin ?? ""}
                      onChange={(e) =>
                        updateComponentRange(c.value, "vmin", e.target.value)
                      }
                      placeholder="min"
                      className="w-14 h-5 bg-[var(--input)] border border-[#3a3d45] rounded px-1 text-[9px] text-[var(--foreground)]"
                    />
                    <span className="text-[9px] text-[var(--muted-foreground)]">~</span>
                    <input
                      type="text"
                      value={comp?.vmax ?? ""}
                      onChange={(e) =>
                        updateComponentRange(c.value, "vmax", e.target.value)
                      }
                      placeholder="max"
                      className="w-14 h-5 bg-[var(--input)] border border-[#3a3d45] rounded px-1 text-[9px] text-[var(--foreground)]"
                    />
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* WYSIWYG note */}
        <p className="text-[9px] text-[var(--muted-foreground)] italic">
          Uses current visualization settings (colormap, overlay, arrows)
        </p>

        {/* Export button / progress */}
        {exportActive ? (
          <div className="space-y-1">
            <div className="h-1.5 bg-[var(--input)] rounded-full overflow-hidden">
              <div
                className="h-full bg-[var(--primary)] transition-all duration-300"
                style={{ width: `${exportProgress}%` }}
              />
            </div>
            <div className="flex items-center gap-1">
              <span className="text-[10px] text-[var(--muted-foreground)] flex-1">
                Exporting... {exportProgress.toFixed(0)}%
              </span>
              <button
                onClick={handleCancel}
                className="px-2 py-1 bg-red-900/30 hover:bg-red-900/50 rounded text-[10px] text-red-400"
              >
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <button
            onClick={handleImageExport}
            disabled={!imgDir.trim()}
            className={btnClass}
          >
            <Download size={12} /> Export Images
          </button>
        )}
      </div>

      {/* FileBrowser modal */}
      {browseTarget && (
        <FileBrowser
          open
          onClose={() => setBrowseTarget(null)}
          onSelect={(path) => {
            if (browseTarget === "sci") {
              sciTouched.current = true;
              setSciPath(path);
            } else {
              imgTouched.current = true;
              setImgDir(path);
            }
            setBrowseTarget(null);
          }}
          initialPath={
            browseTarget === "sci"
              ? sciPath.replace(/\/[^/]*$/, "")
              : imgDir
          }
        />
      )}
    </CollapsibleSection>
  );
}
