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
  const [weighting, setWeighting] = useState("gaussian");

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
        <SmallInput value={vsgSize} onChange={setVsgSize} />
      </FieldRow>
      <FieldRow label="Step">
        <SmallInput value={step} onChange={setStep} />
      </FieldRow>
      <FieldRow label="Poly Order">
        <SmallInput value={polyOrder} onChange={setPolyOrder} />
      </FieldRow>
      <FieldRow label="Weighting">
        <SelectField
          value={weighting}
          options={WEIGHTING_OPTIONS}
          onChange={setWeighting}
        />
      </FieldRow>
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
