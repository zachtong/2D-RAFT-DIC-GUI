import { useState } from "react";
import { CollapsibleSection } from "@/components/shared/CollapsibleSection";
import { FieldRow } from "@/components/shared/FieldRow";
import { SmallInput } from "@/components/shared/SmallInput";
import { SelectField } from "@/components/shared/SelectField";
import { useAppStore } from "@/stores/appStore";
import { calculateStrain } from "@/api/strain";

const METHOD_OPTIONS = [
  { value: "green_lagrange", label: "Green-Lagrange" },
  { value: "engineering", label: "Engineering" },
];

const WEIGHTING_OPTIONS = [
  { value: "gaussian", label: "Gaussian" },
  { value: "uniform", label: "Uniform" },
];

export function StrainControls() {
  const hasStrain = useAppStore((s) => s.hasStrain);
  const strainComputing = useAppStore((s) => s.strainComputing);
  const strainProgress = useAppStore((s) => s.strainProgress);
  const setStrainComputing = useAppStore((s) => s.setStrainComputing);

  const [method, setMethod] = useState("green_lagrange");
  const [vsgSize, setVsgSize] = useState("31");
  const [step, setStep] = useState("15");
  const [polyOrder, setPolyOrder] = useState("1");
  const [weighting, setWeighting] = useState("uniform");
  const [spatialSigma, setSpatialSigma] = useState("0");
  const [temporalSigma, setTemporalSigma] = useState("0");
  const [gaussianSigma, setGaussianSigma] = useState("");
  const fps = useAppStore((s) => s.visSettings.fps);

  const handleCalculate = async () => {
    if (strainComputing) return;
    setStrainComputing(true);
    try {
      await calculateStrain({
        method,
        vsg_size: parseInt(vsgSize),
        step: parseInt(step),
        poly_order: parseInt(polyOrder),
        weighting,
        spatial_sigma: parseFloat(spatialSigma) || 0,
        temporal_sigma: parseFloat(temporalSigma) || 0,
        gaussian_sigma: gaussianSigma ? parseFloat(gaussianSigma) : undefined,
        fps: fps || 1,
      });
    } catch (e) {
      console.error("Failed to start strain calculation:", e);
      setStrainComputing(false);
    }
  };

  return (
    <CollapsibleSection title="Strain Calculation">
      <FieldRow label="Method">
        <SelectField
          value={method}
          options={METHOD_OPTIONS}
          onChange={setMethod}
        />
      </FieldRow>
      <FieldRow label="VSG Size">
        <SmallInput value={vsgSize} onChange={setVsgSize} placeholder="odd, e.g. 31" />
      </FieldRow>
      <FieldRow label="Step">
        <SmallInput value={step} onChange={setStep} placeholder="e.g. 15" />
      </FieldRow>
      <FieldRow label="Poly Order">
        <SmallInput
          value={polyOrder}
          onChange={(v) => {
            const n = parseInt(v);
            if (v === "" || (!isNaN(n) && n >= 1 && n <= 2)) setPolyOrder(v);
          }}
          placeholder="1 or 2"
        />
      </FieldRow>
      <FieldRow label="Weighting">
        <SelectField
          value={weighting}
          options={WEIGHTING_OPTIONS}
          onChange={setWeighting}
        />
      </FieldRow>
      {weighting === "gaussian" && (
        <FieldRow label="Gauss σ">
          <SmallInput value={gaussianSigma} onChange={setGaussianSigma} placeholder="auto" />
        </FieldRow>
      )}
      <div className="h-px bg-[var(--border)] my-1.5" />
      <div className="text-[9px] text-[var(--muted-foreground)] mb-1 uppercase tracking-wider">Smoothing</div>
      <FieldRow label="Spatial">
        <div className="flex items-center gap-1">
          <SmallInput value={spatialSigma} onChange={setSpatialSigma} placeholder="0 = off" />
          <span className="text-[9px] text-[var(--muted-foreground)] shrink-0">px</span>
        </div>
      </FieldRow>
      {parseFloat(spatialSigma) > 0 && (
        <div className="text-[9px] text-[var(--muted-foreground)] mt-0.5">
          Gaussian blur on displacement before strain calc
        </div>
      )}
      <FieldRow label="Temporal">
        <div className="flex items-center gap-1">
          <SmallInput value={temporalSigma} onChange={setTemporalSigma} placeholder="0 = off" />
          <span className="text-[9px] text-[var(--muted-foreground)] shrink-0">frames</span>
        </div>
      </FieldRow>
      {parseFloat(temporalSigma) > 0 && (
        <div className="text-[9px] text-[var(--muted-foreground)] mt-0.5">
          Gaussian smoothing of strain across time
        </div>
      )}
      <button
        onClick={handleCalculate}
        disabled={strainComputing}
        className={`w-full flex items-center justify-center gap-1.5 py-1.5 rounded text-[11px] mt-1 ${
          strainComputing
            ? "bg-[var(--secondary)] text-[var(--muted-foreground)] animate-pulse"
            : hasStrain
            ? "bg-green-600/20 text-green-400 border border-green-600/40 hover:bg-green-600/30 cursor-pointer"
            : "bg-[var(--primary)] hover:bg-[var(--primary)]/90 text-white"
        }`}
      >
        {strainComputing
          ? `Computing... ${strainProgress.toFixed(0)}%`
          : hasStrain
          ? "Recalculate Strain"
          : "Calculate Strain"}
      </button>
    </CollapsibleSection>
  );
}
