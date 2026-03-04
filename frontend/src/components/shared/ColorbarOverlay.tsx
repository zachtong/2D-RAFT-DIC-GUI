const COLORMAPS: Record<string, string> = {
  turbo:
    "linear-gradient(to top, #30123b, #4662d7, #36aaf9, #1ae4b6, #72fe5e, #c8ef34, #faba39, #f66b19, #ca2a04, #7a0403)",
  viridis:
    "linear-gradient(to top, #440154, #482878, #3e4989, #31688e, #26828e, #1f9e89, #35b779, #6ece58, #b5de2b, #fde725)",
  jet:
    "linear-gradient(to top, #0000ff, #00ffff, #00ff00, #ffff00, #ff8800, #ff0000)",
  coolwarm:
    "linear-gradient(to top, #3b4cc0, #6788ee, #9abbff, #c9d7ef, #edd1c2, #f7a789, #e26952, #b40426)",
  plasma:
    "linear-gradient(to top, #0d0887, #5b02a3, #9c179e, #cb4679, #ed7953, #fdb42f, #f0f921)",
  inferno:
    "linear-gradient(to top, #000004, #1b0c41, #4a0c6b, #781c6d, #a52c60, #cf4446, #ed6925, #fb9b06, #f7d13d, #fcffa4)",
};

function fmt(v: number): string {
  const abs = Math.abs(v);
  if (abs === 0) return "0";
  if (abs >= 1000 || abs < 0.01) return v.toExponential(1);
  return parseFloat(v.toPrecision(3)).toString();
}

function linearTicks(vmin: number, vmax: number): number[] {
  const ticks: number[] = [];
  for (let i = 0; i < 5; i++) ticks.push(vmin + (vmax - vmin) * (i / 4));
  return ticks;
}

function logTicks(vmin: number, vmax: number): number[] {
  const safeMin = Math.max(vmin, 1e-20);
  const safeMax = Math.max(vmax, 1e-20);
  const logMin = Math.log10(safeMin);
  const logMax = Math.log10(safeMax);

  const ticks: number[] = [vmin];
  const minPow = Math.ceil(logMin);
  const maxPow = Math.floor(logMax);
  for (let p = minPow; p <= maxPow; p++) {
    const val = Math.pow(10, p);
    if (val > safeMin * 1.01 && val < safeMax * 0.99) ticks.push(val);
  }
  ticks.push(vmax);

  // If fewer than 3 intermediate ticks, add log-spaced intermediates
  if (ticks.length < 4) {
    const extra: number[] = [];
    for (let i = 1; i <= 3; i++) {
      const logVal = logMin + (logMax - logMin) * (i / 4);
      extra.push(Math.pow(10, logVal));
    }
    ticks.splice(1, 0, ...extra);
  }
  return ticks;
}

function tickPct(
  value: number,
  vmin: number,
  vmax: number,
  isLog: boolean
): number {
  if (isLog) {
    const logMin = Math.log10(Math.max(vmin, 1e-20));
    const logMax = Math.log10(Math.max(vmax, 1e-20));
    if (logMax === logMin) return 50;
    return (
      ((Math.log10(Math.max(value, 1e-20)) - logMin) / (logMax - logMin)) * 100
    );
  }
  if (vmax === vmin) return 50;
  return ((value - vmin) / (vmax - vmin)) * 100;
}

interface ColorbarOverlayProps {
  colormap: string;
  vmin?: number;
  vmax?: number;
  unit?: string;
  scaleFactor?: number;
  logScale?: boolean;
}

export function ColorbarOverlay({
  colormap,
  vmin,
  vmax,
  unit,
  scaleFactor = 1,
  logScale = false,
}: ColorbarOverlayProps) {
  const hasRange = vmin != null && vmax != null;

  const ticks = hasRange
    ? logScale
      ? logTicks(vmin, vmax)
      : linearTicks(vmin, vmax)
    : [];

  return (
    <div className="absolute right-3 top-8 bottom-8 z-10 pointer-events-none flex flex-col bg-black/40 backdrop-blur-sm rounded-md p-1.5 gap-1">
      {/* Unit label */}
      {unit && (
        <span className="text-[9px] text-white/70 font-mono leading-none text-center shrink-0">
          {unit}
        </span>
      )}
      {/* Gradient bar + tick labels */}
      <div className="flex-1 flex gap-1 min-h-0">
        {/* Gradient bar with tick marks */}
        <div
          className="relative w-4 rounded-sm border border-white/25 shrink-0"
          style={{ background: COLORMAPS[colormap] ?? COLORMAPS.turbo }}
        >
          {hasRange &&
            ticks.map((val, i) => {
              const pct = tickPct(val, vmin, vmax, logScale);
              return (
                <div
                  key={i}
                  className="absolute right-0 w-1.5 border-t border-white/80"
                  style={{ bottom: `${pct}%` }}
                />
              );
            })}
        </div>
        {/* Tick value labels */}
        <div className="relative min-w-[2rem]">
          {hasRange &&
            ticks.map((val, i) => {
              const pct = tickPct(val, vmin, vmax, logScale);
              return (
                <span
                  key={i}
                  className="absolute left-0 text-[9px] text-white font-mono leading-none whitespace-nowrap"
                  style={{
                    bottom: `${pct}%`,
                    transform: "translateY(50%)",
                  }}
                >
                  {fmt(val * scaleFactor)}
                </span>
              );
            })}
        </div>
      </div>
    </div>
  );
}
