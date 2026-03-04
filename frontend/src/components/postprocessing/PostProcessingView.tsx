import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { useAppStore } from "@/stores/appStore";
import { renderUrl } from "@/api/displacement";
import { strainRenderUrl } from "@/api/strain";
import { arrowRenderUrl } from "@/api/arrows";
import { addPoint, addLine, listProbes } from "@/api/probes";
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

  const placingMode = useAppStore((s) => s.probePlacingMode);
  const placingFirst = useAppStore((s) => s.probePlacingFirst);
  const setPlacingMode = useAppStore((s) => s.setProbePlacingMode);
  const setPlacingFirst = useAppStore((s) => s.setProbePlacingFirst);
  const setProbes = useAppStore((s) => s.setProbes);
  const updateVis = useAppStore((s) => s.updateVisSettings);

  const imgRef = useRef<HTMLImageElement>(null);
  const imageAreaRef = useRef<HTMLDivElement>(null);
  // Remember the background before entering placement mode so we can restore it
  const savedBgRef = useRef<"reference" | "deformed" | null>(null);

  // Track container size via ResizeObserver for explicit pixel constraints on the image.
  // CSS percentage-based max-height doesn't resolve correctly through flex/auto-height chains,
  // so we measure the container and apply pixel values directly.
  const [containerSize, setContainerSize] = useState({ w: 0, h: 0 });
  useEffect(() => {
    const el = imageAreaRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setContainerSize({ w: Math.round(width), h: Math.round(height) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const showArrows =
    displayComponent === "velocity" &&
    (arrows.showQuiver || arrows.showStreamlines);

  const autoRange = useColorRange(currentFrame, displayComponent, hasResults);

  // Convert screen click coordinates to image pixel coordinates
  const screenToImage = useCallback(
    (clientX: number, clientY: number): [number, number] | null => {
      const img = imgRef.current;
      if (!img) return null;
      const rect = img.getBoundingClientRect();
      const sx = clientX - rect.left;
      const sy = clientY - rect.top;
      const scaleX = img.naturalWidth / rect.width;
      const scaleY = img.naturalHeight / rect.height;
      const ix = Math.round(sx * scaleX);
      const iy = Math.round(sy * scaleY);
      if (ix < 0 || ix >= img.naturalWidth || iy < 0 || iy >= img.naturalHeight) return null;
      return [ix, iy];
    }, []
  );

  // Handle click on image area for probe placement
  const handleImageClick = useCallback(async (e: React.MouseEvent) => {
    if (!placingMode) return;
    const pt = screenToImage(e.clientX, e.clientY);
    if (!pt) return;

    try {
      if (placingMode === "point") {
        await addPoint(pt[0], pt[1]);
        const updated = await listProbes();
        setProbes(updated);
        setPlacingMode(null);
      } else if (placingMode === "line") {
        if (!placingFirst) {
          setPlacingFirst(pt);
        } else {
          await addLine(placingFirst, pt);
          const updated = await listProbes();
          setProbes(updated);
          setPlacingMode(null);
        }
      }
    } catch (err) {
      console.error("Failed to place probe:", err);
    }
  }, [placingMode, placingFirst, screenToImage, setProbes, setPlacingMode, setPlacingFirst]);

  // Force reference background during probe placement so click coordinates
  // are always in the reference frame (displacement field coordinate system).
  useEffect(() => {
    if (placingMode) {
      // Entering placement mode — save current bg and force reference
      if (vis.background !== "reference") {
        savedBgRef.current = vis.background;
        updateVis({ background: "reference" });
      }
    } else {
      // Exiting placement mode — restore previous bg
      if (savedBgRef.current) {
        updateVis({ background: savedBgRef.current });
        savedBgRef.current = null;
      }
    }
  }, [placingMode, updateVis]); // deliberately omit vis.background to avoid loop

  // Escape key cancels placement mode
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && placingMode) {
        setPlacingMode(null);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [placingMode, setPlacingMode]);

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
    <div className="flex-1 flex flex-col min-h-0">
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

      {/* Main visualization — relative container with absolute inner to break percentage chain */}
      <div ref={imageAreaRef} className="flex-1 relative bg-[var(--background)] overflow-hidden min-h-0">
        {/* Absolute fill layer — gives children explicit pixel dimensions */}
        <div className="absolute inset-0 flex items-center justify-center">
          {/* Shared wrapper — explicit pixel max-height from ResizeObserver */}
          <div
            className="relative"
            style={{
              maxWidth: containerSize.w > 0 ? `${containerSize.w}px` : "100%",
              maxHeight: containerSize.h > 0 ? `${containerSize.h}px` : "100%",
              ...(viewZoom !== 1 ? { transform: `scale(${viewZoom})` } : {}),
              cursor: placingMode ? "crosshair" : undefined,
            }}
            onClick={handleImageClick}
          >
            <img
              ref={imgRef}
              src={src}
              alt={`${displayComponent} frame ${currentFrame}`}
              className="block max-w-full"
              style={{ maxHeight: containerSize.h > 0 ? `${containerSize.h}px` : undefined }}
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

            {/* SVG overlay for probe markers — viewBox matches image natural size */}
            {imgRef.current && (
              <svg
                className="absolute inset-0 w-full h-full pointer-events-none"
                viewBox={`0 0 ${imgRef.current.naturalWidth} ${imgRef.current.naturalHeight}`}
                preserveAspectRatio="xMidYMid meet"
              >
                {probes.map((probe) => {
                  if (probe.type === "point") {
                    const [x, y] = probe.coords as [number, number];
                    return (
                      <circle
                        key={`p-${probe.id}`}
                        cx={x}
                        cy={y}
                        r={5}
                        fill={probe.color}
                        stroke="white"
                        strokeWidth={1.5}
                        opacity={0.9}
                      />
                    );
                  }
                  if (probe.type === "line") {
                    const [[x1, y1], [x2, y2]] = probe.coords as [[number, number], [number, number]];
                    return (
                      <g key={`l-${probe.id}`}>
                        <line x1={x1} y1={y1} x2={x2} y2={y2}
                          stroke={probe.color} strokeWidth={2} opacity={0.9} />
                        <circle cx={x1} cy={y1} r={4} fill={probe.color} stroke="white" strokeWidth={1} />
                        <circle cx={x2} cy={y2} r={4} fill={probe.color} stroke="white" strokeWidth={1} />
                      </g>
                    );
                  }
                  return null;
                })}
                {/* Pending first endpoint for line placement */}
                {placingFirst && (
                  <circle cx={placingFirst[0]} cy={placingFirst[1]} r={5}
                    fill="none" stroke="white" strokeWidth={2} strokeDasharray="3,3" />
                )}
              </svg>
            )}
          </div>
        </div>

        {/* Placement mode indicator */}
        {placingMode && (
          <div className="absolute top-3 left-1/2 -translate-x-1/2 z-10 bg-[var(--primary)]/90 px-3 py-1 rounded text-[11px] text-white">
            {placingMode === "point"
              ? "Click to place point probe"
              : placingFirst
                ? "Click to place second endpoint"
                : "Click to place first endpoint"}
            <span className="ml-2 opacity-70">(Esc to cancel)</span>
          </div>
        )}

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
      </div>
    </div>
  );
}
