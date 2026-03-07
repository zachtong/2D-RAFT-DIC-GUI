import { useState, useCallback } from "react";
import { CollapsibleSection } from "@/components/shared/CollapsibleSection";
import { FieldRow } from "@/components/shared/FieldRow";
import { SegmentedControl } from "@/components/shared/SegmentedControl";
import { SelectField } from "@/components/shared/SelectField";
import { Toggle } from "@/components/shared/Toggle";
import { SmallInput } from "@/components/shared/SmallInput";
import { useAppStore } from "@/stores/appStore";
import { configureProcessing, validateMasks } from "@/api/processing";
import { KeyFrameTimeline } from "./KeyFrameTimeline";
import { MaskSourceSelector } from "./MaskSourceSelector";

const keyFrameModeOptions = [
  { value: "every_frame", label: "Every Frame" },
  { value: "every_n", label: "Every N" },
  { value: "custom", label: "Custom" },
];

const maskSourceOptions = [
  { value: "auto", label: "Auto (warp ROI)" },
  { value: "folder", label: "Custom (load folder)" },
];

export function ProcessingParams() {
  const mode = useAppStore((s) => s.mode);
  const setMode = useAppStore((s) => s.setMode);
  const totalFrames = useAppStore((s) => s.imageFiles.length);
  const keyFrames = useAppStore((s) => s.keyFrames);
  const keyFrameMode = useAppStore((s) => s.keyFrameMode);
  const keyFrameInterval = useAppStore((s) => s.keyFrameInterval);
  const maskSource = useAppStore((s) => s.maskSource);
  const maskDir = useAppStore((s) => s.maskDir);
  const maskValidation = useAppStore((s) => s.maskValidation);
  const setKeyFrames = useAppStore((s) => s.setKeyFrames);
  const setKeyFrameMode = useAppStore((s) => s.setKeyFrameMode);
  const setKeyFrameInterval = useAppStore((s) => s.setKeyFrameInterval);
  const useMedianFilter = useAppStore((s) => s.useMedianFilter);
  const setUseMedianFilter = useAppStore((s) => s.setUseMedianFilter);
  const setMaskSource = useAppStore((s) => s.setMaskSource);
  const setMaskDir = useAppStore((s) => s.setMaskDir);
  const setMaskValidation = useAppStore((s) => s.setMaskValidation);

  const [showAdvanced, setShowAdvanced] = useState(false);

  // Advanced params with sensible defaults
  const [contextPadding, setContextPadding] = useState("64");
  const [tileOverlap, setTileOverlap] = useState("64");
  const [sigma, setSigma] = useState("1.5");
  const [safetyFactor, setSafetyFactor] = useState("0.75");

  // Local state for interval input (synced on blur / Enter)
  const [intervalInput, setIntervalInput] = useState(String(keyFrameInterval));

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

  // --- Incremental settings handlers ---

  const syncKeyFrameConfig = useCallback(
    (
      newMode: "every_frame" | "every_n" | "custom",
      interval?: number,
      frames?: number[]
    ) => {
      if (newMode === "every_frame") {
        configureProcessing({
          key_frames: null,
          key_frame_interval: 1,
        }).catch(() => {});
      } else if (newMode === "every_n") {
        configureProcessing({
          key_frame_interval: interval ?? keyFrameInterval,
          key_frames: null,
        }).catch(() => {});
      } else {
        // custom
        configureProcessing({
          key_frames: frames ?? keyFrames,
          key_frame_interval: null,
        }).catch(() => {});
      }
    },
    [keyFrameInterval, keyFrames]
  );

  const handleKeyFrameModeChange = (val: string) => {
    const m = val as "every_frame" | "every_n" | "custom";
    setKeyFrameMode(m);
    syncKeyFrameConfig(m);
  };

  const handleIntervalCommit = (val: string) => {
    const n = parseInt(val, 10);
    if (!isNaN(n) && n >= 1) {
      setKeyFrameInterval(n);
      syncKeyFrameConfig("every_n", n);
    }
  };

  const handleKeyFramesChange = (kf: number[]) => {
    setKeyFrames(kf);
    syncKeyFrameConfig("custom", undefined, kf);
  };

  const handleMedianFilterChange = (checked: boolean) => {
    setUseMedianFilter(checked);
    configureProcessing({ use_median_filter: checked }).catch(() => {});
  };

  const handleMaskSourceChange = (source: "auto" | "folder") => {
    setMaskSource(source);
    setMaskValidation(null);
    if (source === "auto") {
      configureProcessing({ mask_dir: null }).catch(() => {});
    } else {
      configureProcessing({ mask_dir: maskDir || null }).catch(() => {});
    }
  };

  const handleMaskDirChange = (dir: string) => {
    setMaskDir(dir);
    setMaskValidation(null);
  };

  const handleValidateMasks = async () => {
    if (!maskDir) return;
    try {
      const result = await validateMasks(maskDir);
      setMaskValidation(result);
      // Sync to backend if valid
      configureProcessing({ mask_dir: maskDir }).catch(() => {});
    } catch {
      setMaskValidation({ matched_count: 0, total_frames: totalFrames, matched_frames: [], has_frame_1: false });
    }
  };

  return (
    <>
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

      {mode === "incremental" && (
        <CollapsibleSection title="Incremental Settings" defaultOpen>
          <FieldRow label="Ref. Update">
            <SelectField
              value={keyFrameMode}
              options={keyFrameModeOptions}
              onChange={handleKeyFrameModeChange}
            />
          </FieldRow>

          {keyFrameMode === "every_n" && (
            <FieldRow label="Interval">
              <SmallInput
                value={intervalInput}
                onChange={(v) => setIntervalInput(v)}
                className="w-14"
              />
              <button
                onClick={() => handleIntervalCommit(intervalInput)}
                className="h-6 px-2 text-[11px] bg-[var(--input)] border border-[#3a3d45] rounded hover:bg-[var(--secondary)] text-[var(--foreground)]"
              >
                Set
              </button>
            </FieldRow>
          )}

          {keyFrameMode === "custom" && (
            <KeyFrameTimeline
              totalFrames={totalFrames}
              keyFrames={keyFrames}
              onChange={handleKeyFramesChange}
            />
          )}

          <FieldRow label="Mask">
            <SelectField
              value={maskSource}
              options={maskSourceOptions}
              onChange={(v) =>
                handleMaskSourceChange(v as "auto" | "folder")
              }
            />
          </FieldRow>

          {maskSource === "folder" && (
            <MaskSourceSelector
              maskSource={maskSource}
              maskDir={maskDir}
              onSourceChange={handleMaskSourceChange}
              onDirChange={handleMaskDirChange}
              validationResult={maskValidation}
              onValidate={handleValidateMasks}
              keyFrames={keyFrames}
              keyFrameMode={keyFrameMode}
            />
          )}

          <FieldRow label="Median Filter">
            <Toggle
              checked={useMedianFilter}
              onChange={handleMedianFilterChange}
            />
          </FieldRow>
          {useMedianFilter && (
            <p className="text-[10px] text-yellow-400/80 leading-tight px-1 -mt-0.5">
              Applies median filter to accumulated displacement.
              Reduces error buildup but modifies raw output data.
            </p>
          )}
        </CollapsibleSection>
      )}
    </>
  );
}
