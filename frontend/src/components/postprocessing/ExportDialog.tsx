import { useState } from "react";
import { CollapsibleSection } from "@/components/shared/CollapsibleSection";
import { FieldRow } from "@/components/shared/FieldRow";
import { Toggle } from "@/components/shared/Toggle";
import { SmallInput } from "@/components/shared/SmallInput";
import { useAppStore } from "@/stores/appStore";
import { exportScientific, exportImages } from "@/api/export";
import { Download } from "lucide-react";

export function ExportDialog() {
  const numFrames = useAppStore((s) => s.numFrames);
  const exportActive = useAppStore((s) => s.exportActive);
  const exportProgress = useAppStore((s) => s.exportProgress);

  // Scientific export
  const [sciPath, setSciPath] = useState("");
  const [upsample, setUpsample] = useState(true);

  // Image export
  const [imgDir, setImgDir] = useState("");
  const [frameStart, setFrameStart] = useState("0");
  const [frameEnd, setFrameEnd] = useState(String(Math.max(0, numFrames - 1)));

  const handleScientificExport = async () => {
    if (!sciPath.trim()) return;
    try {
      await exportScientific({
        file_path: sciPath.trim(),
        upsample_strain: upsample,
      });
    } catch (e) {
      console.error("Scientific export failed:", e);
    }
  };

  const handleImageExport = async () => {
    if (!imgDir.trim()) return;
    try {
      await exportImages({
        output_dir: imgDir.trim(),
        components: { u: {}, v: {} },
        frame_range: [parseInt(frameStart), parseInt(frameEnd)],
        settings: {},
      });
    } catch (e) {
      console.error("Image export failed:", e);
    }
  };

  return (
    <CollapsibleSection title="Data Export" defaultOpen={false}>
      <FieldRow label="Upsample Strain">
        <Toggle checked={upsample} onChange={setUpsample} />
      </FieldRow>

      <div className="space-y-1">
        <input
          type="text"
          value={sciPath}
          onChange={(e) => setSciPath(e.target.value)}
          placeholder="Export path (.mat or .npz)"
          className="w-full h-6 bg-[var(--input)] border border-[#3a3d45] rounded px-1.5 text-[10px] text-[var(--foreground)] focus:border-[var(--primary)] focus:outline-none"
        />
        <button
          onClick={handleScientificExport}
          disabled={!sciPath.trim()}
          className="w-full flex items-center justify-center gap-1 py-1.5 bg-[var(--secondary)] hover:bg-[var(--secondary)]/80 rounded text-[11px] text-[var(--foreground)] disabled:opacity-40"
        >
          <Download size={12} /> Export Scientific Data
        </button>
      </div>

      <div className="h-px bg-[var(--border)] my-1" />

      <div className="space-y-1">
        <input
          type="text"
          value={imgDir}
          onChange={(e) => setImgDir(e.target.value)}
          placeholder="Output directory"
          className="w-full h-6 bg-[var(--input)] border border-[#3a3d45] rounded px-1.5 text-[10px] text-[var(--foreground)] focus:border-[var(--primary)] focus:outline-none"
        />
        <div className="flex items-center gap-1">
          <span className="text-[10px] text-[var(--muted-foreground)]">Frames:</span>
          <SmallInput value={frameStart} onChange={setFrameStart} className="w-10" />
          <span className="text-[10px] text-[var(--muted-foreground)]">—</span>
          <SmallInput value={frameEnd} onChange={setFrameEnd} className="w-10" />
        </div>
        <button
          onClick={handleImageExport}
          disabled={!imgDir.trim() || exportActive}
          className="w-full flex items-center justify-center gap-1 py-1.5 bg-[var(--secondary)] hover:bg-[var(--secondary)]/80 rounded text-[11px] text-[var(--foreground)] disabled:opacity-40"
        >
          <Download size={12} />
          {exportActive
            ? `Exporting... ${exportProgress.toFixed(0)}%`
            : "Export Visualization Images"}
        </button>
      </div>
    </CollapsibleSection>
  );
}
