import { SmallInput } from "@/components/shared/SmallInput";

interface MaskSourceSelectorProps {
  maskSource: "auto" | "folder";
  maskDir: string;
  onSourceChange: (source: "auto" | "folder") => void;
  onDirChange: (dir: string) => void;
  validationResult?: {
    matched_count: number;
    total_frames: number;
    matched_frames: number[];
  } | null;
  onValidate: () => void;
}

export function MaskSourceSelector({
  maskDir,
  onDirChange,
  validationResult,
  onValidate,
}: MaskSourceSelectorProps) {
  return (
    <div className="flex flex-col gap-1.5 w-full">
      {/* Folder path input + Validate */}
      <div className="flex items-center gap-1">
        <SmallInput
          value={maskDir}
          onChange={onDirChange}
          className="flex-1 min-w-0 text-left"
          placeholder="Mask folder path"
        />
        <button
          onClick={onValidate}
          disabled={!maskDir}
          className="h-6 px-2 text-[11px] bg-[var(--input)] border border-[#3a3d45] rounded hover:bg-[var(--secondary)] text-[var(--foreground)] disabled:opacity-40 disabled:cursor-not-allowed shrink-0"
        >
          Validate
        </button>
      </div>

      {/* Validation result */}
      {validationResult && (
        <span
          className={`text-[10px] ${
            validationResult.matched_count === validationResult.total_frames
              ? "text-green-400"
              : validationResult.matched_count > 0
              ? "text-yellow-400"
              : "text-red-400"
          }`}
        >
          Matched {validationResult.matched_count} of{" "}
          {validationResult.total_frames} frames
        </span>
      )}
    </div>
  );
}
