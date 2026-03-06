import { useState, useEffect, useRef, useCallback } from "react";
import { useAppStore } from "@/stores/appStore.ts";

export interface PreRenderState {
  /** Blob URL for a given frame, or undefined if not yet cached */
  getFrame: (idx: number) => string | undefined;
  /** Whether pre-rendering is in progress */
  isPreRendering: boolean;
  /** Progress: 0-100 */
  progress: number;
  /** Number of frames cached */
  cachedCount: number;
  /** Total frames to cache */
  totalFrames: number;
  /** Start pre-rendering with current vis settings */
  startPreRender: () => void;
  /** Cancel in-progress pre-render */
  cancelPreRender: () => void;
  /** Invalidate cache (e.g. when settings change) */
  invalidate: () => void;
}

const STRAIN_COMPONENTS = new Set([
  "exx", "eyy", "exy", "e1", "e2", "max_shear", "von_mises", "rotation",
]);

function buildFrameUrl(
  idx: number,
  component: string,
  params: Record<string, string | number>,
  isStrain: boolean
): string {
  const base = isStrain ? "/api/strain/render" : "/api/displacement/render";
  const qs = new URLSearchParams();
  qs.set("component", component);
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== "") qs.set(k, String(v));
  }
  return `${base}/${idx}?${qs}`;
}

export function usePreRenderCache(componentOverride?: string): PreRenderState {
  const numFrames = useAppStore((s) => s.numFrames);
  const hasResults = useAppStore((s) => s.hasResults);
  const storeComponent = useAppStore((s) => s.displayComponent);
  const vis = useAppStore((s) => s.visSettings);
  const referenceFrame = useAppStore((s) => s.referenceFrame);

  const displayComponent = componentOverride ?? storeComponent;

  const [isPreRendering, setIsPreRendering] = useState(false);
  const [progress, setProgress] = useState(0);
  const [cachedCount, setCachedCount] = useState(0);

  const cacheRef = useRef<Map<number, string>>(new Map());
  const abortRef = useRef<AbortController | null>(null);
  const cacheKeyRef = useRef<string>("");

  const getFrame = useCallback((idx: number) => {
    return cacheRef.current.get(idx);
  }, []);

  const invalidate = useCallback(() => {
    for (const url of cacheRef.current.values()) {
      URL.revokeObjectURL(url);
    }
    cacheRef.current.clear();
    setCachedCount(0);
    setProgress(0);
    cacheKeyRef.current = "";
  }, []);

  const cancelPreRender = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    setIsPreRendering(false);
  }, []);

  const startPreRender = useCallback(() => {
    if (!hasResults || numFrames === 0) return;

    const isStrain = STRAIN_COMPONENTS.has(displayComponent);
    // Pick correct per-component vmin/vmax (V component uses vminV/vmaxV)
    const vmin = displayComponent === "v" ? vis.vminV : vis.vminU;
    const vmax = displayComponent === "v" ? vis.vmaxV : vis.vmaxU;
    const params: Record<string, string | number> = {
      colormap: vis.colormap,
      alpha: vis.alpha,
      background: vis.background,
      ...(vis.fixedRange && vmin ? { vmin } : {}),
      ...(vis.fixedRange && vmax ? { vmax } : {}),
      ...(vis.logScale ? { log_scale: "true" } : {}),
      ...(referenceFrame > 0 ? { ref_frame: referenceFrame } : {}),
      ...(vis.smoothSigma > 0 && !isStrain ? { smooth_sigma: vis.smoothSigma } : {}),
    };
    const settingsKey = `${displayComponent}|${JSON.stringify(params)}`;

    // If cache already matches and is complete, skip
    if (cacheKeyRef.current === settingsKey && cacheRef.current.size >= numFrames) {
      return;
    }

    // Invalidate old cache if settings changed
    if (cacheKeyRef.current !== settingsKey) {
      invalidate();
    }
    cacheKeyRef.current = settingsKey;

    // Abort any in-flight pre-render
    if (abortRef.current) abortRef.current.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setIsPreRendering(true);
    setProgress(0);

    const CONCURRENCY = 3;
    let nextIdx = 0;

    const fetchFrame = async () => {
      while (nextIdx < numFrames) {
        if (controller.signal.aborted) return;
        const idx = nextIdx++;

        // Skip already-cached frames
        if (cacheRef.current.has(idx)) continue;

        const url = buildFrameUrl(idx, displayComponent, params, isStrain);
        try {
          const resp = await fetch(url, { signal: controller.signal });
          if (!resp.ok) continue;
          const blob = await resp.blob();
          if (controller.signal.aborted) return;
          const blobUrl = URL.createObjectURL(blob);
          cacheRef.current.set(idx, blobUrl);
          const count = cacheRef.current.size;
          setCachedCount(count);
          setProgress(Math.round((count / numFrames) * 100));
        } catch {
          if (controller.signal.aborted) return;
        }
      }
    };

    const workers = Array.from(
      { length: Math.min(CONCURRENCY, numFrames) },
      fetchFrame
    );
    Promise.all(workers).then(() => {
      if (!controller.signal.aborted) {
        setIsPreRendering(false);
      }
    });
  }, [hasResults, numFrames, displayComponent, vis.colormap, vis.alpha,
      vis.background, vis.fixedRange, vis.vminU, vis.vmaxU, vis.vminV, vis.vmaxV, vis.logScale, vis.smoothSigma, referenceFrame, invalidate]);

  // Invalidate cache when relevant settings change
  useEffect(() => {
    invalidate();
    cancelPreRender();
  }, [displayComponent, vis.colormap, vis.alpha, vis.background,
      vis.fixedRange, vis.vminU, vis.vmaxU, vis.vminV, vis.vmaxV, vis.logScale, vis.smoothSigma, referenceFrame, invalidate, cancelPreRender]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cancelPreRender();
      for (const url of cacheRef.current.values()) {
        URL.revokeObjectURL(url);
      }
    };
  }, [cancelPreRender]);

  return {
    getFrame,
    isPreRendering,
    progress,
    cachedCount,
    totalFrames: numFrames,
    startPreRender,
    cancelPreRender,
    invalidate,
  };
}
