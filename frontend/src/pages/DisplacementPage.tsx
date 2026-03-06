import { useEffect, useCallback } from "react";
import { Sidebar } from "@/components/layout/Sidebar";
import { VisualizationSettings } from "@/components/roi/VisualizationSettings";
import { DisplacementView } from "@/components/displacement/DisplacementView";
import { FramePlayback } from "@/components/displacement/FramePlayback";
import { usePreRenderCache } from "@/hooks/usePreRenderCache";
import { useAppStore } from "@/stores/appStore";

export function DisplacementPage() {
  const cacheU = usePreRenderCache("u");
  const cacheV = usePreRenderCache("v");
  const hasResults = useAppStore((s) => s.hasResults);
  const numFrames = useAppStore((s) => s.numFrames);

  // Auto-start pre-render when results are available and there are multiple frames
  useEffect(() => {
    if (hasResults && numFrames > 1) {
      cacheU.startPreRender();
      cacheV.startPreRender();
    }
  }, [hasResults, numFrames, cacheU.startPreRender, cacheV.startPreRender]);

  const isFrameReady = useCallback(
    (idx: number) => !!cacheU.getFrame(idx) && !!cacheV.getFrame(idx),
    [cacheU.getFrame, cacheV.getFrame]
  );

  const combinedProgress = Math.round(
    ((cacheU.cachedCount + cacheV.cachedCount) /
      (cacheU.totalFrames + cacheV.totalFrames || 1)) *
      100
  );

  return (
    <div className="flex flex-1 overflow-hidden">
      <Sidebar>
        <div className="flex-1 overflow-y-auto">
          <VisualizationSettings />
        </div>
      </Sidebar>
      <div className="flex-1 flex flex-col overflow-hidden">
        <DisplacementView cacheU={cacheU} cacheV={cacheV} />
        <FramePlayback
          preRenderCache={{
            isPreRendering: cacheU.isPreRendering || cacheV.isPreRendering,
            progress: combinedProgress,
            cachedCount: cacheU.cachedCount + cacheV.cachedCount,
            totalFrames: cacheU.totalFrames + cacheV.totalFrames,
            isFrameReady,
          }}
        />
      </div>
    </div>
  );
}
