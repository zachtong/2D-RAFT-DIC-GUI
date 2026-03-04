import { useState } from "react";
import { CollapsibleSection } from "@/components/shared/CollapsibleSection";
import { FieldRow } from "@/components/shared/FieldRow";
import { SegmentedControl } from "@/components/shared/SegmentedControl";
import { Toggle } from "@/components/shared/Toggle";
import { SmallInput } from "@/components/shared/SmallInput";
import { useAppStore } from "@/stores/appStore";
import { configureProcessing } from "@/api/processing";

export function ProcessingParams() {
  const mode = useAppStore((s) => s.mode);
  const setMode = useAppStore((s) => s.setMode);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Advanced params with sensible defaults
  const [contextPadding, setContextPadding] = useState("64");
  const [tileOverlap, setTileOverlap] = useState("64");
  const [sigma, setSigma] = useState("1.5");
  const [safetyFactor, setSafetyFactor] = useState("0.75");

  const handleModeChange = (val: string) => {
    const m = val.toLowerCase() as "accumulative" | "incremental";
    setMode(m);
    configureProcessing({ mode: m }).catch(() => {});
  };

  const handleParamChange = (key: string, value: string) => {
    const num = parseFloat(value);
    if (!isNaN(num)) {
      configureProcessing({ [key]: num }).catch(() => {});
    }
  };

  return (
    <CollapsibleSection title="Processing Mode">
      <SegmentedControl
        options={["Accumulative", "Incremental"]}
        value={mode === "accumulative" ? "Accumulative" : "Incremental"}
        onChange={handleModeChange}
      />

      <FieldRow label="Show Advanced">
        <Toggle checked={showAdvanced} onChange={setShowAdvanced} />
      </FieldRow>

      {showAdvanced && (
        <div className="flex flex-col gap-1.5 pt-1">
          <FieldRow label="Context Pad">
            <SmallInput
              value={contextPadding}
              onChange={(v) => {
                setContextPadding(v);
                handleParamChange("context_padding", v);
              }}
            />
          </FieldRow>
          <FieldRow label="Tile Overlap">
            <SmallInput
              value={tileOverlap}
              onChange={(v) => {
                setTileOverlap(v);
                handleParamChange("tile_overlap", v);
              }}
            />
          </FieldRow>
          <FieldRow label="Smooth σ">
            <SmallInput
              value={sigma}
              onChange={(v) => {
                setSigma(v);
                handleParamChange("sigma", v);
              }}
            />
          </FieldRow>
          <FieldRow label="Safety Factor">
            <SmallInput
              value={safetyFactor}
              onChange={(v) => {
                setSafetyFactor(v);
                handleParamChange("safety_factor", v);
              }}
            />
          </FieldRow>
        </div>
      )}
    </CollapsibleSection>
  );
}
