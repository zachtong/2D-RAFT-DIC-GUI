import { useState } from "react";
import { useFrameNav } from "@/hooks/useFrameNav";
import { useAppStore } from "@/stores/appStore";
import {
  SkipBack,
  ChevronLeft,
  Play,
  Pause,
  ChevronRight,
  SkipForward,
  ZoomIn,
  ZoomOut,
  Maximize,
} from "lucide-react";

const SPEED_OPTIONS = [
  { value: "0.5", label: "0.5x" },
  { value: "1", label: "1x" },
  { value: "2", label: "2x" },
  { value: "5", label: "5x" },
  { value: "10", label: "10x" },
];

interface PreRenderCacheInfo {
  isPreRendering: boolean;
  progress: number;
  cachedCount: number;
  totalFrames: number;
  isFrameReady: (idx: number) => boolean;
}

interface FramePlaybackProps {
  preRenderCache?: PreRenderCacheInfo;
}

export function FramePlayback({ preRenderCache }: FramePlaybackProps) {
  const { currentFrame, numFrames, next, prev, first, last, goTo, play, pause, isPlaying, setFps } =
    useFrameNav(preRenderCache ? { isFrameReady: preRenderCache.isFrameReady } : undefined);
  const zoomIn = useAppStore((s) => s.zoomIn);
  const zoomOut = useAppStore((s) => s.zoomOut);
  const zoomReset = useAppStore((s) => s.zoomReset);
  const [frameInput, setFrameInput] = useState("");
  const [speed, setSpeed] = useState("5");

  const handleSpeedChange = (val: string) => {
    setSpeed(val);
    setFps(parseFloat(val));
  };

  const handleGoTo = () => {
    const n = parseInt(frameInput);
    if (!isNaN(n)) goTo(n - 1); // UI shows 1-indexed
    setFrameInput("");
  };

  if (numFrames === 0) return null;

  return (
    <div className="relative flex items-center justify-between px-3 py-2 bg-[var(--card)] border-t border-[var(--border)]">
      {/* Navigation buttons */}
      <div className="flex items-center gap-1">
        <button onClick={first} className="p-1 hover:bg-[var(--secondary)] rounded text-[var(--muted-foreground)]">
          <SkipBack size={14} />
        </button>
        <button onClick={prev} className="p-1 hover:bg-[var(--secondary)] rounded text-[var(--muted-foreground)]">
          <ChevronLeft size={14} />
        </button>
        <button
          onClick={() => (isPlaying ? pause() : play())}
          className="p-1.5 bg-[var(--primary)] hover:bg-[var(--primary)]/90 rounded-full text-white"
        >
          {isPlaying ? <Pause size={14} /> : <Play size={14} />}
        </button>
        <button onClick={next} className="p-1 hover:bg-[var(--secondary)] rounded text-[var(--muted-foreground)]">
          <ChevronRight size={14} />
        </button>
        <button onClick={last} className="p-1 hover:bg-[var(--secondary)] rounded text-[var(--muted-foreground)]">
          <SkipForward size={14} />
        </button>

        <div className="w-px h-4 bg-[var(--border)] mx-1" />

        <button onClick={zoomIn} className="p-1 hover:bg-[var(--secondary)] rounded text-[var(--muted-foreground)]">
          <ZoomIn size={14} />
        </button>
        <button onClick={zoomOut} className="p-1 hover:bg-[var(--secondary)] rounded text-[var(--muted-foreground)]">
          <ZoomOut size={14} />
        </button>
        <button onClick={zoomReset} className="p-1 hover:bg-[var(--secondary)] rounded text-[var(--muted-foreground)]">
          <Maximize size={14} />
        </button>
      </div>

      {/* Speed control */}
      <div className="flex items-center gap-2">
        <span className="text-[10px] text-[var(--muted-foreground)]">Speed</span>
        <select
          value={speed}
          onChange={(e) => handleSpeedChange(e.target.value)}
          className="bg-[var(--input)] border border-[#3a3d45] rounded px-1.5 py-0.5 text-[10px] text-[var(--foreground)]"
        >
          {SPEED_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      </div>

      {/* Frame input */}
      <div className="flex items-center gap-2">
        <span className="text-[10px] text-[var(--muted-foreground)]">Frame</span>
        <input
          type="number"
          value={frameInput || currentFrame + 1}
          onChange={(e) => setFrameInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleGoTo()}
          className="w-12 bg-[var(--input)] border border-[#3a3d45] rounded px-1.5 py-0.5 text-[10px] text-[var(--foreground)] text-center"
        />
        <span className="text-[10px] text-[var(--muted-foreground)]">/ {numFrames}</span>
        <button
          onClick={handleGoTo}
          className="px-2 py-0.5 bg-[var(--secondary)] hover:bg-[var(--secondary)]/80 rounded text-[10px] text-[var(--foreground)]"
        >
          Go
        </button>
      </div>

      {/* Pre-render progress bar */}
      {preRenderCache?.isPreRendering && (
        <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-[var(--border)]">
          <div
            className="h-full bg-[var(--primary)] transition-all duration-300"
            style={{ width: `${preRenderCache.progress}%` }}
          />
        </div>
      )}
    </div>
  );
}
