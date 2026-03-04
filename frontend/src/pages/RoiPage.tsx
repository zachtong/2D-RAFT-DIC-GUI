import { Sidebar } from "@/components/layout/Sidebar";
import { PathSettings } from "@/components/roi/PathSettings";
import { ProcessingParams } from "@/components/roi/ProcessingParams";
import { VisualizationSettings } from "@/components/roi/VisualizationSettings";
import { RunControl } from "@/components/roi/RunControl";
import { RoiCanvas } from "@/components/roi/RoiCanvas";
import { RoiToolbar } from "@/components/roi/RoiToolbar";

export function RoiPage() {
  return (
    <div className="flex flex-1 overflow-hidden">
      <Sidebar>
        <div className="flex flex-col flex-1 overflow-y-auto">
          <PathSettings />
          <ProcessingParams />
          <VisualizationSettings />
        </div>
        <RunControl />
      </Sidebar>
      <div className="flex-1 flex flex-col">
        <RoiCanvas />
        <RoiToolbar />
      </div>
    </div>
  );
}
