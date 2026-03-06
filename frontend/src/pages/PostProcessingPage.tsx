import { useRef, useState, useEffect, useCallback } from "react";
import { Sidebar } from "@/components/layout/Sidebar";
import { StrainControls } from "@/components/postprocessing/StrainControls";
import { VisualizationControls } from "@/components/postprocessing/VisualizationControls";
import { VelocityArrowControls } from "@/components/postprocessing/VelocityArrowControls";
import { ProbePanel } from "@/components/postprocessing/ProbePanel";
import { ExportDialog } from "@/components/postprocessing/ExportDialog";
import { PostProcessingView } from "@/components/postprocessing/PostProcessingView";
import { FramePlayback } from "@/components/displacement/FramePlayback";
import { TimeSeriesChart } from "@/components/postprocessing/TimeSeriesChart";
import { ResizableSplitter } from "@/components/postprocessing/ResizableSplitter";
import { usePreRenderCache } from "@/hooks/usePreRenderCache";
import { useAppStore } from "@/stores/appStore";

export function PostProcessingPage() {
  const splitContainerRef = useRef<HTMLDivElement>(null);
  const [topRatio, setTopRatio] = useState(0.7);

  const cache = usePreRenderCache();
  const hasResults = useAppStore((s) => s.hasResults);
  const numFrames = useAppStore((s) => s.numFrames);

  // Auto-start pre-render when results are available and there are multiple frames
  useEffect(() => {
    if (hasResults && numFrames > 1) {
      cache.startPreRender();
    }
  }, [hasResults, numFrames, cache.startPreRender]);

  const isFrameReady = useCallback(
    (idx: number) => !!cache.getFrame(idx),
    [cache.getFrame]
  );

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
      <div className="flex-1 flex flex-col overflow-hidden">
        <div ref={splitContainerRef} className="flex-1 flex flex-col overflow-hidden">
          <div style={{ flex: `0 0 ${topRatio * 100}%` }} className="overflow-hidden flex flex-col">
            <PostProcessingView cache={cache} />
          </div>
          <ResizableSplitter
            containerRef={splitContainerRef}
            onRatioChange={setTopRatio}
          />
          <div style={{ flex: 1 }} className="overflow-hidden min-h-[120px]">
            <TimeSeriesChart />
          </div>
        </div>
        <FramePlayback preRenderCache={{
          isPreRendering: cache.isPreRendering,
          progress: cache.progress,
          cachedCount: cache.cachedCount,
          totalFrames: cache.totalFrames,
          isFrameReady,
        }} />
      </div>
    </div>
  );
}
