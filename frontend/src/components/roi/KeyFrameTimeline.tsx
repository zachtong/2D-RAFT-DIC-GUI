import { useState, useRef, useCallback } from "react";
import { SmallInput } from "@/components/shared/SmallInput";

interface KeyFrameTimelineProps {
  totalFrames: number;
  keyFrames: number[]; // 1-indexed
  onChange: (kf: number[]) => void;
}

export function KeyFrameTimeline({
  totalFrames,
  keyFrames,
  onChange,
}: KeyFrameTimelineProps) {
  const barRef = useRef<HTMLDivElement>(null);
  const [everyN, setEveryN] = useState("10");
  const [addFrame, setAddFrame] = useState("");

  const sorted = [...keyFrames].sort((a, b) => a - b);

  const handleBarClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (!barRef.current || totalFrames <= 1) return;
      const rect = barRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const fraction = x / rect.width;
      const frame = Math.max(1, Math.min(totalFrames, Math.round(fraction * (totalFrames - 1) + 1)));

      // Check if clicking near an existing key frame
      const tolerance = Math.max(1, Math.round(totalFrames * 0.02));
      const existing = keyFrames.find(
        (kf) => Math.abs(kf - frame) <= tolerance
      );

      if (existing !== undefined && existing !== 1) {
        // Remove it (but never frame 1)
        onChange(keyFrames.filter((kf) => kf !== existing));
      } else if (!keyFrames.includes(frame)) {
        onChange([...keyFrames, frame].sort((a, b) => a - b));
      }
    },
    [totalFrames, keyFrames, onChange]
  );

  const handleApplyEveryN = () => {
    const n = parseInt(everyN, 10);
    if (isNaN(n) || n < 1) return;
    const frames: number[] = [];
    for (let f = 1; f <= totalFrames; f += n) {
      frames.push(f);
    }
    onChange(frames);
  };

  const handleAddFrame = () => {
    const f = parseInt(addFrame, 10);
    if (isNaN(f) || f < 1 || f > totalFrames) return;
    if (!keyFrames.includes(f)) {
      onChange([...keyFrames, f].sort((a, b) => a - b));
    }
    setAddFrame("");
  };

  const handleClear = () => {
    onChange([1]);
  };

  return (
    <div className="flex flex-col gap-1.5 w-full">
      {/* Timeline bar */}
      <div
        ref={barRef}
        onClick={handleBarClick}
        className="relative h-4 bg-[var(--input)] border border-[#3a3d45] rounded cursor-crosshair overflow-hidden"
        title="Click to add/remove key frames"
      >
        {totalFrames > 1 &&
          sorted.map((kf) => {
            const pct = ((kf - 1) / (totalFrames - 1)) * 100;
            return (
              <div
                key={kf}
                className="absolute top-0 bottom-0 w-[3px] rounded-sm"
                style={{
                  left: `calc(${pct}% - 1px)`,
                  backgroundColor:
                    kf === 1 ? "var(--primary)" : "var(--foreground)",
                  opacity: kf === 1 ? 1 : 0.8,
                }}
              />
            );
          })}
      </div>

      {/* Summary */}
      <span className="text-[10px] text-[var(--muted-foreground)]">
        {sorted.length} key frame{sorted.length !== 1 ? "s" : ""}
        {sorted.length <= 6
          ? `: ${sorted.join(", ")}`
          : `: ${sorted.slice(0, 5).join(", ")}...${sorted[sorted.length - 1]}`}
      </span>

      {/* Every N row */}
      <div className="flex items-center gap-1">
        <span className="text-[10px] text-[var(--muted-foreground)] shrink-0">
          Every
        </span>
        <SmallInput
          value={everyN}
          onChange={setEveryN}
          className="w-10"
        />
        <button
          onClick={handleApplyEveryN}
          className="h-6 px-2 text-[11px] bg-[var(--input)] border border-[#3a3d45] rounded hover:bg-[var(--secondary)] text-[var(--foreground)]"
        >
          Apply
        </button>
      </div>

      {/* Add specific + Clear row */}
      <div className="flex items-center gap-1">
        <span className="text-[10px] text-[var(--muted-foreground)] shrink-0">
          Frame
        </span>
        <SmallInput
          value={addFrame}
          onChange={setAddFrame}
          className="w-10"
          placeholder="#"
        />
        <button
          onClick={handleAddFrame}
          className="h-6 px-2 text-[11px] bg-[var(--input)] border border-[#3a3d45] rounded hover:bg-[var(--secondary)] text-[var(--foreground)]"
        >
          Add
        </button>
        <button
          onClick={handleClear}
          className="h-6 px-2 text-[11px] bg-[var(--input)] border border-[#3a3d45] rounded hover:bg-[var(--secondary)] text-[var(--muted-foreground)]"
        >
          Clear
        </button>
      </div>
    </div>
  );
}
