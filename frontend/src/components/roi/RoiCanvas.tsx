import { useRef, useState, useCallback, useEffect } from "react";
import { useRoiStore } from "@/stores/roiStore";
import { useAppStore } from "@/stores/appStore";
import { referenceImageUrl } from "@/api/images";
import { addPolygon, addRectangle, addCircle, confirmRoi } from "@/api/roi";
import { ImageIcon } from "lucide-react";

export function RoiCanvas() {
  const containerRef = useRef<HTMLDivElement>(null);
  const imageFiles = useAppStore((s) => s.imageFiles);
  const imageWidth = useAppStore((s) => s.imageWidth);
  const imageHeight = useAppStore((s) => s.imageHeight);
  const setRoiConfirmed = useAppStore((s) => s.setRoiConfirmed);

  const drawingMode = useRoiStore((s) => s.drawingMode);
  const currentPoints = useRoiStore((s) => s.currentPoints);
  const addPoint = useRoiStore((s) => s.addPoint);
  const clearPoints = useRoiStore((s) => s.clearPoints);
  const maskUrl = useRoiStore((s) => s.maskUrl);
  const setMaskUrl = useRoiStore((s) => s.setMaskUrl);

  // Pan and zoom
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });

  // Temporary shape drawing state (for rect/circle second click)
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null);

  // Fit image to container on load
  useEffect(() => {
    if (!containerRef.current || !imageWidth || !imageHeight) return;
    const cw = containerRef.current.clientWidth;
    const ch = containerRef.current.clientHeight;
    const s = Math.min(cw / imageWidth, ch / imageHeight, 1) * 0.9;
    setScale(s);
    setOffset({
      x: (cw - imageWidth * s) / 2,
      y: (ch - imageHeight * s) / 2,
    });
  }, [imageWidth, imageHeight, imageFiles.length]);

  // Escape key cancels current drawing
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape" && drawingMode) {
        clearPoints();
        setMousePos(null);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [drawingMode, clearPoints]);

  // Convert screen coords to image pixel coords
  const screenToImage = useCallback(
    (clientX: number, clientY: number): [number, number] => {
      if (!containerRef.current) return [0, 0];
      const rect = containerRef.current.getBoundingClientRect();
      const sx = clientX - rect.left;
      const sy = clientY - rect.top;
      const ix = (sx - offset.x) / scale;
      const iy = (sy - offset.y) / scale;
      return [Math.round(ix), Math.round(iy)];
    },
    [scale, offset]
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      // Middle mouse = pan
      if (e.button === 1) {
        e.preventDefault();
        setIsPanning(true);
        setPanStart({ x: e.clientX - offset.x, y: e.clientY - offset.y });
        return;
      }
    },
    [offset]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (isPanning) {
        setOffset({ x: e.clientX - panStart.x, y: e.clientY - panStart.y });
        return;
      }
      if (drawingMode && currentPoints.length > 0) {
        const [ix, iy] = screenToImage(e.clientX, e.clientY);
        setMousePos({ x: ix, y: iy });
      }
    },
    [isPanning, panStart, drawingMode, currentPoints.length, screenToImage]
  );

  const handleMouseUp = useCallback(() => {
    setIsPanning(false);
  }, []);

  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.1 : 0.9;
      const rect = containerRef.current!.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const newScale = Math.min(Math.max(scale * factor, 0.1), 10);
      setScale(newScale);
      setOffset({
        x: mx - ((mx - offset.x) / scale) * newScale,
        y: my - ((my - offset.y) / scale) * newScale,
      });
    },
    [scale, offset]
  );

  const handleClick = useCallback(
    async (e: React.MouseEvent) => {
      if (e.button !== 0 || !drawingMode || imageFiles.length === 0) return;
      const [ix, iy] = screenToImage(e.clientX, e.clientY);

      if (drawingMode === "polygon" || drawingMode === "cut") {
        addPoint(ix, iy);
      } else if (drawingMode === "rectangle") {
        if (currentPoints.length === 0) {
          addPoint(ix, iy);
        } else {
          // Complete rectangle
          const [x0, y0] = currentPoints[0];
          try {
            await addRectangle(
              Math.min(x0, ix), Math.min(y0, iy),
              Math.max(x0, ix), Math.max(y0, iy),
              "add"
            );
            setMaskUrl(`/api/roi/mask?t=${Date.now()}`);
            await confirmRoi();
            setRoiConfirmed(true);
          } catch (err) {
            console.error("Failed to add rectangle:", err);
          }
          clearPoints();
          setMousePos(null);
        }
      } else if (drawingMode === "circle") {
        if (currentPoints.length === 0) {
          addPoint(ix, iy);
        } else {
          const [cx, cy] = currentPoints[0];
          const r = Math.sqrt((ix - cx) ** 2 + (iy - cy) ** 2);
          try {
            await addCircle(cx, cy, r);
            setMaskUrl(`/api/roi/mask?t=${Date.now()}`);
            await confirmRoi();
            setRoiConfirmed(true);
          } catch (err) {
            console.error("Failed to add circle:", err);
          }
          clearPoints();
          setMousePos(null);
        }
      }
    },
    [drawingMode, imageFiles.length, currentPoints, screenToImage, addPoint, clearPoints, setMaskUrl, setRoiConfirmed]
  );

  const handleDoubleClick = useCallback(async () => {
    if (
      (drawingMode === "polygon" || drawingMode === "cut") &&
      currentPoints.length >= 3
    ) {
      try {
        const mode = drawingMode === "cut" ? "cut" : "add";
        await addPolygon(currentPoints, mode);
        setMaskUrl(`/api/roi/mask?t=${Date.now()}`);
        await confirmRoi();
        setRoiConfirmed(true);
      } catch (err) {
        console.error("Failed to add polygon:", err);
      }
      clearPoints();
      setMousePos(null);
    }
  }, [drawingMode, currentPoints, clearPoints, setMaskUrl, setRoiConfirmed]);

  // Convert image coords to screen for SVG
  const imgToScreen = (ix: number, iy: number) => ({
    x: ix * scale + offset.x,
    y: iy * scale + offset.y,
  });

  const hasImage = imageFiles.length > 0;

  return (
    <div
      ref={containerRef}
      className="flex-1 relative overflow-hidden bg-[var(--background)]"
      style={{ cursor: drawingMode ? "crosshair" : "default" }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onClick={handleClick}
      onDoubleClick={handleDoubleClick}
      onWheel={handleWheel}
      onContextMenu={(e) => e.preventDefault()}
    >
      {hasImage ? (
        <>
          {/* Reference image */}
          <img
            key={imageFiles[0] ?? "ref"}
            src={`${referenceImageUrl()}?v=${encodeURIComponent(imageFiles[0] ?? "")}`}
            alt="Reference"
            draggable={false}
            className="absolute top-0 left-0 select-none max-w-none"
            style={{
              transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})`,
              transformOrigin: "0 0",
              imageRendering: scale > 2 ? "pixelated" : "auto",
            }}
          />

          {/* ROI mask overlay */}
          {maskUrl && (
            <img
              src={maskUrl}
              alt="ROI Mask"
              draggable={false}
              className="absolute top-0 left-0 select-none pointer-events-none max-w-none"
              style={{
                transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})`,
                transformOrigin: "0 0",
              }}
            />
          )}

          {/* SVG overlay for active drawing */}
          <svg className="absolute inset-0 w-full h-full pointer-events-none">
            {/* In-progress polygon/cut lines */}
            {(drawingMode === "polygon" || drawingMode === "cut") &&
              currentPoints.length > 0 && (
                <>
                  <polyline
                    points={[
                      ...currentPoints.map((p) => {
                        const s = imgToScreen(p[0], p[1]);
                        return `${s.x},${s.y}`;
                      }),
                      ...(mousePos
                        ? [
                            (() => {
                              const s = imgToScreen(mousePos.x, mousePos.y);
                              return `${s.x},${s.y}`;
                            })(),
                          ]
                        : []),
                    ].join(" ")}
                    fill="none"
                    stroke={drawingMode === "cut" ? "#ef4444" : "#3b82f6"}
                    strokeWidth="1.5"
                    strokeDasharray="4 2"
                  />
                  {currentPoints.map((p, i) => {
                    const s = imgToScreen(p[0], p[1]);
                    return (
                      <circle
                        key={i}
                        cx={s.x}
                        cy={s.y}
                        r={3}
                        fill={drawingMode === "cut" ? "#ef4444" : "#3b82f6"}
                      />
                    );
                  })}
                </>
              )}

            {/* In-progress rectangle preview */}
            {drawingMode === "rectangle" &&
              currentPoints.length === 1 &&
              mousePos && (() => {
                const s0 = imgToScreen(currentPoints[0][0], currentPoints[0][1]);
                const s1 = imgToScreen(mousePos.x, mousePos.y);
                return (
                  <rect
                    x={Math.min(s0.x, s1.x)}
                    y={Math.min(s0.y, s1.y)}
                    width={Math.abs(s1.x - s0.x)}
                    height={Math.abs(s1.y - s0.y)}
                    fill="rgba(59,130,246,0.15)"
                    stroke="#3b82f6"
                    strokeWidth="1.5"
                    strokeDasharray="4 2"
                  />
                );
              })()}

            {/* In-progress circle preview */}
            {drawingMode === "circle" &&
              currentPoints.length === 1 &&
              mousePos && (() => {
                const center = imgToScreen(currentPoints[0][0], currentPoints[0][1]);
                const edge = imgToScreen(mousePos.x, mousePos.y);
                const r = Math.sqrt(
                  (edge.x - center.x) ** 2 + (edge.y - center.y) ** 2
                );
                return (
                  <circle
                    cx={center.x}
                    cy={center.y}
                    r={r}
                    fill="rgba(59,130,246,0.15)"
                    stroke="#3b82f6"
                    strokeWidth="1.5"
                    strokeDasharray="4 2"
                  />
                );
              })()}
          </svg>
        </>
      ) : (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-[var(--muted-foreground)]">
          <ImageIcon className="w-12 h-12 mb-3 opacity-40" />
          <p className="text-[13px]">No images loaded</p>
          <p className="text-[11px] opacity-60">
            Set input directory and load images to begin
          </p>
        </div>
      )}
    </div>
  );
}
