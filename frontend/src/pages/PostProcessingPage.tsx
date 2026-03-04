import { Sidebar } from "@/components/layout/Sidebar";
import { StrainControls } from "@/components/postprocessing/StrainControls";
import { VisualizationControls } from "@/components/postprocessing/VisualizationControls";
import { VelocityArrowControls } from "@/components/postprocessing/VelocityArrowControls";
import { ProbePanel } from "@/components/postprocessing/ProbePanel";
import { ExportDialog } from "@/components/postprocessing/ExportDialog";
import { PostProcessingView } from "@/components/postprocessing/PostProcessingView";
import { FramePlayback } from "@/components/displacement/FramePlayback";
import { TimeSeriesChart } from "@/components/postprocessing/TimeSeriesChart";

export function PostProcessingPage() {
  return (
    <div className="flex flex-1 overflow-hidden">
      <Sidebar>
        <div className="flex-1 overflow-y-auto">
          <StrainControls />
          <VisualizationControls />
          <VelocityArrowControls />
          <ProbePanel />
          <ExportDialog />
        </div>
      </Sidebar>
      <div className="flex-1 flex flex-col">
        <PostProcessingView />
        <TimeSeriesChart />
        <FramePlayback />
      </div>
    </div>
  );
}
