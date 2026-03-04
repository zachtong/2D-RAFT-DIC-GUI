import { useAppStore } from "@/stores/appStore";
import { renderUrl } from "@/api/displacement";
import { ColorbarOverlay } from "@/components/shared/ColorbarOverlay";
import { useColorRange } from "@/hooks/useColorRange";
import { ImageIcon } from "lucide-react";

function DisplacementPanel({
  label,
  component,
}: {
  label: string;
  component: "u" | "v";
}) {
  const currentFrame = useAppStore((s) => s.currentFrame);
  const hasResults = useAppStore((s) => s.hasResults);
  const vis = useAppStore((s) => s.visSettings);
  const viewZoom = useAppStore((s) => s.viewZoom);

  const vmin = component === "u" ? vis.vminU : vis.vminV;
  const vmax = component === "u" ? vis.vmaxU : vis.vmaxV;
  const autoRange = useColorRange(currentFrame, component, hasResults);

  const src = renderUrl(currentFrame, {
    component,
    colormap: vis.colormap,
    alpha: vis.alpha,
    background: vis.background,
    ...(vis.fixedRange && vmin ? { vmin } : {}),
    ...(vis.fixedRange && vmax ? { vmax } : {}),
  });

  return (
    <div className="relative flex-1 overflow-hidden flex items-center justify-center bg-[var(--background)]">
      <div className="absolute top-3 left-1/2 -translate-x-1/2 z-10 bg-[var(--card)]/80 px-3 py-1 rounded text-[11px] text-[var(--foreground)]">
        {label}
      </div>
      <img
        src={src}
        alt={label}
        className="max-w-full max-h-full object-contain transition-transform"
        style={viewZoom !== 1 ? { transform: `scale(${viewZoom})` } : undefined}
        draggable={false}
      />
      <ColorbarOverlay
        colormap={vis.colormap}
        vmin={vis.fixedRange && vmin ? parseFloat(vmin) : autoRange?.vmin}
        vmax={vis.fixedRange && vmax ? parseFloat(vmax) : autoRange?.vmax}
        unit={
          vis.physicalEnabled
            ? `[${vis.physicalUnit}]`
            : "[px]"
        }
        scaleFactor={vis.physicalEnabled ? vis.physicalRatio : 1}
      />
    </div>
  );
}

export function DisplacementView() {
  const hasResults = useAppStore((s) => s.hasResults);
  const currentFrame = useAppStore((s) => s.currentFrame);
  const numFrames = useAppStore((s) => s.numFrames);

  if (!hasResults) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-[var(--muted-foreground)]">
        <ImageIcon className="w-12 h-12 mb-3 opacity-40" />
        <p className="text-[13px]">No displacement results</p>
        <p className="text-[11px] opacity-60">
          Run processing from the ROI Selection page first
        </p>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col">
      {/* Header info */}
      <div className="flex items-center justify-between px-4 py-1.5 bg-[var(--card)] border-b border-[var(--border)]">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-[var(--primary)]" />
          <span className="text-[11px] text-[var(--foreground)]">
            Displacement Field Overlay
          </span>
        </div>
        <span className="text-[11px] text-[var(--muted-foreground)]">
          Frame: {currentFrame + 1} / {numFrames}
        </span>
      </div>

      {/* Side-by-side U/V panels */}
      <div className="flex-1 flex">
        <DisplacementPanel label="U COMPONENT" component="u" />
        <div className="w-px bg-[var(--border)]" />
        <DisplacementPanel label="V COMPONENT" component="v" />
      </div>
    </div>
  );
}
