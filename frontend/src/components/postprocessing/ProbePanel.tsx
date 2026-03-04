import { useState } from "react";
import { CollapsibleSection } from "@/components/shared/CollapsibleSection";
import { SegmentedControl } from "@/components/shared/SegmentedControl";
import { useAppStore } from "@/stores/appStore";
import { listProbes, clearProbes, removeProbe } from "@/api/probes";

type ProbeTab = "Point" | "Line" | "Area";

export function ProbePanel() {
  const probes = useAppStore((s) => s.probes);
  const setProbes = useAppStore((s) => s.setProbes);
  const [tab, setTab] = useState<ProbeTab>("Point");

  const filteredProbes = probes.filter(
    (p) => p.type === tab.toLowerCase()
  );

  const handleClear = async () => {
    try {
      await clearProbes(tab.toLowerCase());
      const updated = await listProbes();
      setProbes(updated);
    } catch (e) {
      console.error("Failed to clear probes:", e);
    }
  };

  const handleRemove = async (id: number) => {
    try {
      await removeProbe(id, tab.toLowerCase());
      const updated = await listProbes();
      setProbes(updated);
    } catch (e) {
      console.error("Failed to remove probe:", e);
    }
  };

  return (
    <CollapsibleSection title="Virtual Extensometers">
      <SegmentedControl
        options={["Point", "Line", "Area"]}
        value={tab}
        onChange={(v) => setTab(v as ProbeTab)}
      />

      <div className="flex gap-1 mt-1">
        <button className="flex-1 px-2 py-1 bg-[var(--secondary)] hover:bg-[var(--secondary)]/80 rounded text-[10px] text-[var(--foreground)]">
          + Add
        </button>
        <button
          onClick={() => {
            if (filteredProbes.length > 0) {
              handleRemove(filteredProbes[filteredProbes.length - 1].id);
            }
          }}
          className="flex-1 px-2 py-1 bg-[var(--secondary)] hover:bg-[var(--secondary)]/80 rounded text-[10px] text-[var(--foreground)]"
        >
          - Remove
        </button>
        <button
          onClick={handleClear}
          className="flex-1 px-2 py-1 bg-[var(--secondary)] hover:bg-red-900/30 rounded text-[10px] text-red-400"
        >
          Clear
        </button>
      </div>

      {/* Probe table */}
      <div className="mt-1 bg-[var(--input)] rounded border border-[var(--border)]">
        <div className="flex text-[9px] text-[var(--muted-foreground)] border-b border-[var(--border)] px-2 py-1">
          <span className="w-6">ID</span>
          <span className="flex-1">COORDS</span>
          <span className="w-10 text-right">COLOR</span>
        </div>
        {filteredProbes.length === 0 ? (
          <div className="flex items-center justify-center py-4 text-[10px] text-[var(--muted-foreground)]">
            No {tab.toLowerCase()} probes placed
          </div>
        ) : (
          filteredProbes.map((probe) => (
            <div
              key={probe.id}
              className="flex items-center text-[10px] text-[var(--foreground)] px-2 py-0.5 border-b border-[var(--border)] last:border-b-0"
            >
              <span className="w-6">{probe.id}</span>
              <span className="flex-1 truncate">
                {JSON.stringify(probe.coords)}
              </span>
              <span
                className="w-3 h-3 rounded-full ml-auto shrink-0"
                style={{ background: probe.color }}
              />
            </div>
          ))
        )}
      </div>
    </CollapsibleSection>
  );
}
