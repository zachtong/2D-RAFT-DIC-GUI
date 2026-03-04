import { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import { useAppStore } from "@/stores/appStore";
import { getTimeSeries, getExtensometer } from "@/api/probes";

const COLORS = [
  "#3b82f6", "#ef4444", "#22c55e", "#f59e0b",
  "#8b5cf6", "#ec4899", "#06b6d4", "#f97316",
];

interface ChartPoint {
  frame: number;
  [key: string]: number | null;
}

export function TimeSeriesChart() {
  const probes = useAppStore((s) => s.probes);
  const currentFrame = useAppStore((s) => s.currentFrame);
  const displayComponent = useAppStore((s) => s.displayComponent);
  const setCurrentFrame = useAppStore((s) => s.setCurrentFrame);
  const [data, setData] = useState<ChartPoint[]>([]);
  const [mode, setMode] = useState<"component" | "extensometer">("component");

  const pointProbes = probes.filter((p) => p.type === "point");
  const lineProbes = probes.filter((p) => p.type === "line");
  const hasProbes = pointProbes.length > 0 || lineProbes.length > 0;

  // Fetch time series data
  useEffect(() => {
    if (!hasProbes) {
      setData([]);
      return;
    }

    let cancelled = false;

    const fetchData = async () => {
      try {
        if (mode === "extensometer" && lineProbes.length > 0) {
          // Fetch extensometer data for each line probe
          const results = await Promise.all(
            lineProbes.map((p) => getExtensometer(p.id))
          );
          if (cancelled) return;

          const frames = results[0]?.num_frames ?? 0;
          const chartData: ChartPoint[] = [];
          for (let i = 0; i < frames; i++) {
            const point: ChartPoint = { frame: i + 1 };
            results.forEach((r, idx) => {
              point[`line_${lineProbes[idx].id}_strain`] = r.strains[i];
            });
            chartData.push(point);
          }
          setData(chartData);
        } else {
          // Fetch component time series for point probes
          const probeType = pointProbes.length > 0 ? "point" : "line";
          const result = await getTimeSeries(displayComponent, probeType, "avg");
          if (cancelled) return;

          const frames = Object.values(result.series)[0]?.length ?? 0;
          const chartData: ChartPoint[] = [];
          for (let i = 0; i < frames; i++) {
            const point: ChartPoint = { frame: i + 1 };
            for (const [id, values] of Object.entries(result.series)) {
              point[`probe_${id}`] = values[i];
            }
            chartData.push(point);
          }
          setData(chartData);
        }
      } catch (e) {
        console.error("Failed to fetch time series:", e);
      }
    };

    fetchData();
    return () => { cancelled = true; };
  }, [hasProbes, pointProbes.length, lineProbes.length, displayComponent, mode]);

  if (!hasProbes || data.length === 0) return null;

  const seriesKeys = Object.keys(data[0] || {}).filter((k) => k !== "frame");

  const title = mode === "extensometer"
    ? "Virtual Extensometer — Engineering Strain"
    : `Time Series — ${displayComponent.toUpperCase()}`;

  return (
    <div className="h-[180px] border-t border-[var(--border)] bg-[var(--card)] px-2 pt-1 pb-2">
      <div className="flex items-center justify-between mb-1">
        <span className="text-[10px] text-[var(--muted-foreground)] uppercase tracking-wider font-semibold">
          {title}
        </span>
        {lineProbes.length > 0 && (
          <div className="flex items-center gap-1">
            <button
              onClick={() => setMode("component")}
              className={`px-2 py-0.5 rounded text-[9px] ${
                mode === "component"
                  ? "bg-[var(--primary)] text-white"
                  : "text-[var(--muted-foreground)] hover:bg-[var(--secondary)]"
              }`}
            >
              Component
            </button>
            <button
              onClick={() => setMode("extensometer")}
              className={`px-2 py-0.5 rounded text-[9px] ${
                mode === "extensometer"
                  ? "bg-[var(--primary)] text-white"
                  : "text-[var(--muted-foreground)] hover:bg-[var(--secondary)]"
              }`}
            >
              Extensometer
            </button>
          </div>
        )}
      </div>
      <ResponsiveContainer width="100%" height="85%">
        <LineChart
          data={data}
          onClick={(e: any) => {
            if (e?.activeLabel) setCurrentFrame(Number(e.activeLabel) - 1);
          }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            dataKey="frame"
            tick={{ fontSize: 9, fill: "var(--muted-foreground)" }}
            axisLine={{ stroke: "var(--border)" }}
          />
          <YAxis
            tick={{ fontSize: 9, fill: "var(--muted-foreground)" }}
            axisLine={{ stroke: "var(--border)" }}
            width={50}
          />
          <Tooltip
            contentStyle={{
              background: "var(--card)",
              border: "1px solid var(--border)",
              borderRadius: "4px",
              fontSize: "10px",
            }}
          />
          <ReferenceLine
            x={currentFrame + 1}
            stroke="var(--primary)"
            strokeDasharray="3 3"
          />
          {seriesKeys.map((key, i) => (
            <Line
              key={key}
              type="monotone"
              dataKey={key}
              stroke={COLORS[i % COLORS.length]}
              dot={false}
              strokeWidth={1.5}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
