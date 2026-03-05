import { Sidebar } from "@/components/layout/Sidebar";
import { VisualizationSettings } from "@/components/roi/VisualizationSettings";
import { DisplacementView } from "@/components/displacement/DisplacementView";
import { FramePlayback } from "@/components/displacement/FramePlayback";

export function DisplacementPage() {
  return (
    <div className="flex flex-1 overflow-hidden">
      <Sidebar>
        <div className="flex-1 overflow-y-auto">
          <VisualizationSettings />
        </div>
      </Sidebar>
      <div className="flex-1 flex flex-col overflow-hidden">
        <DisplacementView />
        <FramePlayback />
      </div>
    </div>
  );
}
