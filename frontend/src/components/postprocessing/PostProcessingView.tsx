import { useState, useEffect, useRef, useMemo } from "react";
import { useAppStore } from "@/stores/appStore";
import { renderUrl } from "@/api/displacement";
import { strainRenderUrl } from "@/api/strain";
import { arrowRenderUrl } from "@/api/arrows";
import { ColorbarOverlay } from "@/components/shared/ColorbarOverlay";
import { useColorRange } from "@/hooks/useColorRange";
import type { StrainComponent } from "@/types/api";
import { ImageIcon, Loader2 } from "lucide-react";

const STRAIN_COMPONENTS: string[] = [
  "exx", "eyy", "exy", "e1", "e2", "max_shear", "von_mises", "rotation",
];

/** Compute the colorbar unit string and display scale factor. */
function getUnitInfo(
  component: string,
  physEnabled: boolean,
  physUnit: string,
  physRatio: number,
  fps: number
): { unit: string; scale: number } {
  if (STRAIN_COMPONENTS.includes(component)) {
    return component === "rotation"
      ? { unit: "[rad]", scale: 1 }
      : { unit: "[-]", scale: 1 };
  }
  if (component === "velocity") {
    if (physEnabled) {
      return fps !== 1
        ? { unit: `[${physUnit}/s]`, scale: physRatio * fps }
        : { unit: `[${physUnit}/frame]`, scale: physRatio };
    }
    return fps !== 1
      ? { unit: "[px/s]", scale: fps }
      : { unit: "[px/frame]", scale: 1 };
  }
  // u, v, magnitude
  if (physEnabled) {
    return { unit: `[${physUnit}]`, scale: physRatio };
  }
  return { unit: "[px]", scale: 1 };
}

export function PostProcessingView() {
  const hasResults = useAppStore((s) => s.hasResults);
  const currentFrame = useAppStore((s) => s.currentFrame);
  const numFrames = useAppStore((s) => s.numFrames);
  const displayComponent = useAppStore((s) => s.displayComponent);
  const vis = useAppStore((s) => s.visSettings);
  const arrows = useAppStore((s) => s.arrowSettings);
  const probes = useAppStore((s) => s.probes);
  const viewZoom = useAppStore((s) => s.viewZoom);

  const showArrows =
    displayComponent === "velocity" &&
    (arrows.showQuiver || arrows.showStreamlines);

  const autoRange = useColorRange(currentFrame, displayComponent, hasResults);

  const { unit: colorbarUnit, scale: colorbarScale } = useMemo(
    () =>
      getUnitInfo(
        displayComponent,
        vis.physicalEnabled,
        vis.physicalUnit,
        vis.physicalRatio,
        vis.fps
      ),
    [displayComponent, vis.physicalEnabled, vis.physicalUnit, vis.physicalRatio, vis.fps]
  );

  // Loading indicator: only shows if image takes >150ms to load (cold cache).
  // Cached frames load instantly, so the timer is cleared before it fires.
  const [imgLoading, setImgLoading] = useState(false);
  const loadTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    loadTimerRef.current = setTimeout(() => setImgLoading(true), 150);
    return () => {
      if (loadTimerRef.current) clearTimeout(loadTimerRef.current);
    };
  }, [currentFrame, displayComponent]);

  if (!hasResults) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-[var(--muted-foreground)]">
        <ImageIcon className="w-12 h-12 mb-3 opacity-40" />
        <p className="text-[13px]">No results available</p>
        <p className="text-[11px] opacity-60">
          Run processing and navigate here to analyze results
        </p>
      </div>
    );
  }

  const isStrain = STRAIN_COMPONENTS.includes(displayComponent);
  const renderParams = {
    component: displayComponent,
    colormap: vis.colormap,
    alpha: vis.alpha,
    background: vis.background,
    ...(vis.fixedRange && vis.vminU ? { vmin: vis.vminU } : {}),
    ...(vis.fixedRange && vis.vmaxU ? { vmax: vis.vmaxU } : {}),
    ...(vis.logScale ? { log_scale: "true" } : {}),
  };

  const src = isStrain
    ? strainRenderUrl(currentFrame, renderParams)
    : renderUrl(currentFrame, renderParams);

  return (
    <div className="flex-1 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-1.5 bg-[var(--card)] border-b border-[var(--border)]">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-[var(--primary)]" />
          <span className="text-[11px] text-[var(--foreground)]">
            {isStrain ? "Strain Field" : "Displacement Field"} — {displayComponent.toUpperCase()}
          </span>
        </div>
        <span className="text-[11px] text-[var(--muted-foreground)]">
          Frame: {currentFrame + 1} / {numFrames}
        </span>
      </div>

      {/* Main visualization */}
      <div className="flex-1 relative flex items-center justify-center bg-[var(--background)]">
        {/* Shared wrapper — main image sizes the box, overlay fills it */}
        <div
          className="relative max-w-full max-h-full"
          style={viewZoom !== 1 ? { transform: `scale(${viewZoom})` } : undefined}
        >
          <img
            src={src}
            alt={`${displayComponent} frame ${currentFrame}`}
            className="block max-w-full max-h-full"
            draggable={false}
            onLoad={() => {
              if (loadTimerRef.current) clearTimeout(loadTimerRef.current);
              setImgLoading(false);
            }}
          />
          {showArrows && currentFrame > 0 && (
            <img
              src={arrowRenderUrl(currentFrame, {
                show_quiver: arrows.showQuiver,
                show_streamlines: arrows.showStreamlines,
                spacing: arrows.spacing,
                scale: arrows.scale,
                color: arrows.color,
                line_width: arrows.lineWidth,
                background: vis.background,
                stream_ds: arrows.streamQuality,
              })}
              alt="velocity arrows"
              className="absolute inset-0 w-full h-full pointer-events-none"
              draggable={false}
            />
          )}
        </div>

        {/* Loading overlay — centered on image area */}
        {imgLoading && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <div className="flex items-center gap-2 bg-black/60 backdrop-blur-sm text-white px-4 py-2 rounded-lg">
              <Loader2 size={16} className="animate-spin" />
              <span className="text-[12px] font-medium">Rendering frame...</span>
            </div>
          </div>
        )}

        <ColorbarOverlay
          colormap={vis.colormap}
          vmin={vis.fixedRange && vis.vminU ? parseFloat(vis.vminU) : autoRange?.vmin}
          vmax={vis.fixedRange && vis.vmaxU ? parseFloat(vis.vmaxU) : autoRange?.vmax}
          unit={colorbarUnit}
          scaleFactor={colorbarScale}
          logScale={vis.logScale}
        />

        {/* SVG overlay for probe markers */}
        <svg className="absolute inset-0 w-full h-full pointer-events-none">
          {probes.map((probe) => {
            if (probe.type === "point") {
              const coords = probe.coords as [number, number];
              return (
                <circle
                  key={`p-${probe.id}`}
                  cx="50%"
                  cy="50%"
                  r={5}
                  fill={probe.color}
                  stroke="white"
                  strokeWidth={1}
                  opacity={0.8}
                />
              );
            }
            return null;
          })}
        </svg>
      </div>
    </div>
  );
}
