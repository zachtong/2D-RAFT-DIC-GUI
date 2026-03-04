import { useRoiStore, type DrawingMode } from "@/stores/roiStore";
import { useAppStore } from "@/stores/appStore";
import { clearRoi, invertMask } from "@/api/roi";
import {
  Pentagon,
  RectangleHorizontal,
  Circle,
  Scissors,
  Upload,
  RotateCcw,
  Trash2,
} from "lucide-react";

const ROI_TOOLS: {
  id: DrawingMode | "import" | "invert" | "clear";
  label: string;
  icon: React.ComponentType<{ className?: string }>;
}[] = [
  { id: "polygon", label: "Polygon", icon: Pentagon },
  { id: "rectangle", label: "Rectangle", icon: RectangleHorizontal },
  { id: "circle", label: "Circle", icon: Circle },
  { id: "cut", label: "Cut", icon: Scissors },
  { id: "import", label: "Import", icon: Upload },
  { id: "invert", label: "Invert", icon: RotateCcw },
  { id: "clear", label: "Clear", icon: Trash2 },
];

export function RoiToolbar() {
  const drawingMode = useRoiStore((s) => s.drawingMode);
  const setDrawingMode = useRoiStore((s) => s.setDrawingMode);
  const clearPoints = useRoiStore((s) => s.clearPoints);
  const setMaskUrl = useRoiStore((s) => s.setMaskUrl);
  const setRoiConfirmed = useAppStore((s) => s.setRoiConfirmed);
  const imageFiles = useAppStore((s) => s.imageFiles);

  const handleToolClick = async (id: string) => {
    if (id === "clear") {
      await clearRoi();
      clearPoints();
      setMaskUrl(null);
      setRoiConfirmed(false);
      return;
    }
    if (id === "invert") {
      const result = await invertMask();
      if (result) setMaskUrl(`/api/roi/mask?t=${Date.now()}`);
      return;
    }
    if (id === "import") {
      // For now, prompt isn't implemented in browser — would need file dialog
      return;
    }
    // Toggle drawing mode
    const mode = id as DrawingMode;
    setDrawingMode(drawingMode === mode ? null : mode);
  };

  const disabled = imageFiles.length === 0;

  return (
    <div className="flex items-center justify-between px-3 py-1.5 bg-[var(--card)] border-t border-[var(--border)]">
      <div className="flex items-center gap-1">
        <span className="text-[10px] text-[var(--muted-foreground)] mr-2 uppercase tracking-wider font-semibold">
          ROI Tools
        </span>
        {ROI_TOOLS.map((tool) => (
          <button
            key={tool.id}
            onClick={() => handleToolClick(tool.id!)}
            disabled={disabled}
            className={`flex items-center gap-1 px-2 py-1 rounded text-[10px] transition-colors disabled:opacity-30 ${
              drawingMode === tool.id
                ? "bg-[var(--primary)] text-white"
                : "text-[var(--muted-foreground)] hover:bg-[var(--secondary)] hover:text-[var(--foreground)]"
            }`}
          >
            <tool.icon className="w-3 h-3" />
            {tool.label}
          </button>
        ))}
      </div>
    </div>
  );
}
