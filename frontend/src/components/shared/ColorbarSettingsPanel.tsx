import { useEffect, useRef } from "react";
import { useAppStore } from "@/stores/appStore";
import type { ColorbarSettings } from "@/stores/appStore";

const DISCRETE_OPTIONS = [
  { value: 0, label: "Continuous" },
  { value: 5, label: "5 levels" },
  { value: 8, label: "8 levels" },
  { value: 10, label: "10 levels" },
  { value: 16, label: "16 levels" },
  { value: 32, label: "32 levels" },
];

const DEFAULT_SETTINGS: ColorbarSettings = {
  labelText: "",
  labelFontSize: 12,
  tickCount: 0,
  tickFontSize: 10,
  shrink: 0.8,
  hideOutline: false,
  discreteLevels: 0,
};

const inputClass =
  "w-14 h-5 bg-[var(--input)] border border-[#3a3d45] rounded px-1 text-[9px] text-[var(--foreground)] text-center focus:border-[var(--primary)] focus:outline-none";

interface Props {
  onClose: () => void;
}

export function ColorbarSettingsPanel({ onClose }: Props) {
  const panelRef = useRef<HTMLDivElement>(null);
  const cbs = useAppStore((s) => s.colorbarSettings);
  const update = useAppStore((s) => s.updateColorbarSettings);

  // Close on click outside
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [onClose]);

  // Close on Escape
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [onClose]);

  const handleReset = () => update(DEFAULT_SETTINGS);

  const setNum = (key: keyof ColorbarSettings, raw: string, fallback: number) => {
    const v = parseFloat(raw);
    update({ [key]: isNaN(v) ? fallback : v } as Partial<ColorbarSettings>);
  };

  return (
    <div
      ref={panelRef}
      className="absolute left-full top-0 ml-2 z-50 w-[260px]
        bg-[var(--card)] border border-[var(--border)] rounded-lg shadow-xl
        p-3 space-y-3"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-semibold text-[var(--foreground)]">
          Colorbar Settings
        </span>
        <button
          onClick={onClose}
          className="text-[var(--muted-foreground)] hover:text-[var(--foreground)] text-[14px] leading-none"
        >
          ×
        </button>
      </div>

      {/* Label section */}
      <div className="space-y-1">
        <span className="text-[9px] font-medium text-[var(--muted-foreground)] uppercase tracking-wide">
          Label
        </span>
        <input
          type="text"
          value={cbs.labelText}
          onChange={(e) => update({ labelText: e.target.value })}
          placeholder="Auto (from component)"
          className="w-full h-5 bg-[var(--input)] border border-[#3a3d45] rounded px-1.5
            text-[9px] text-[var(--foreground)] placeholder:text-[var(--muted-foreground)]
            focus:border-[var(--primary)] focus:outline-none"
        />
        <div className="flex items-center gap-1">
          <span className="text-[9px] text-[var(--muted-foreground)]">Size:</span>
          <input
            type="number"
            value={cbs.labelFontSize}
            onChange={(e) => setNum("labelFontSize", e.target.value, 12)}
            min={6}
            max={24}
            className={inputClass}
          />
          <span className="text-[8px] text-[var(--muted-foreground)]">px</span>
        </div>
      </div>

      {/* Ticks section */}
      <div className="space-y-1">
        <span className="text-[9px] font-medium text-[var(--muted-foreground)] uppercase tracking-wide">
          Ticks
        </span>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1">
            <span className="text-[9px] text-[var(--muted-foreground)]">Count:</span>
            <input
              type="number"
              value={cbs.tickCount || ""}
              onChange={(e) => setNum("tickCount", e.target.value, 0)}
              placeholder="auto"
              min={0}
              max={20}
              className={inputClass}
            />
          </div>
          <div className="flex items-center gap-1">
            <span className="text-[9px] text-[var(--muted-foreground)]">Size:</span>
            <input
              type="number"
              value={cbs.tickFontSize}
              onChange={(e) => setNum("tickFontSize", e.target.value, 10)}
              min={6}
              max={20}
              className={inputClass}
            />
          </div>
        </div>
      </div>

      {/* Style section */}
      <div className="space-y-1">
        <span className="text-[9px] font-medium text-[var(--muted-foreground)] uppercase tracking-wide">
          Style
        </span>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1">
            <span className="text-[9px] text-[var(--muted-foreground)]">Shrink:</span>
            <input
              type="number"
              value={cbs.shrink}
              onChange={(e) => setNum("shrink", e.target.value, 0.8)}
              step={0.05}
              min={0.3}
              max={1.0}
              className={inputClass}
            />
          </div>
          <label className="flex items-center gap-0.5 text-[9px] text-[var(--foreground)] cursor-pointer">
            <input
              type="checkbox"
              checked={cbs.hideOutline}
              onChange={() => update({ hideOutline: !cbs.hideOutline })}
              className="accent-[var(--primary)] w-2.5 h-2.5"
            />
            No outline
          </label>
        </div>
        <div className="flex items-center gap-1">
          <span className="text-[9px] text-[var(--muted-foreground)]">Levels:</span>
          <select
            value={cbs.discreteLevels}
            onChange={(e) => update({ discreteLevels: parseInt(e.target.value) })}
            className="flex-1 px-1 py-0.5 rounded text-[9px] bg-[var(--input)] border border-[var(--border)] text-[var(--foreground)]"
          >
            {DISCRETE_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </div>
        <p className="text-[8px] text-[var(--muted-foreground)] italic">
          Shrink only affects exported images/animations
        </p>
      </div>

      {/* Reset */}
      <button
        onClick={handleReset}
        className="w-full py-1 bg-[var(--secondary)] hover:bg-[var(--secondary)]/80
          rounded text-[9px] text-[var(--muted-foreground)] hover:text-[var(--foreground)]
          transition-colors"
      >
        Reset Defaults
      </button>
    </div>
  );
}
