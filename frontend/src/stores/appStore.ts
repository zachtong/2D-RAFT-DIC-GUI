import { create } from "zustand";
import type { ModelMetadata, ProbeData, DisplayComponent } from "@/types/api";

interface ArrowSettings {
  showQuiver: boolean;
  showStreamlines: boolean;
  spacing: number;
  scale: number;
  color: string;
  lineWidth: number;
  streamQuality: number; // downsample factor: 1=full, 2=fine, 4=balanced, 8=fast
}

interface VisSettings {
  colormap: string;
  alpha: number;
  background: "reference" | "deformed";
  logScale: boolean;
  smoothSigma: number;
  physicalEnabled: boolean;
  physicalRatio: number;
  physicalUnit: string;
  fps: number;
  fixedRange: boolean;
  vminU: string;
  vmaxU: string;
  vminV: string;
  vmaxV: string;
}

interface AppState {
  // Image state
  imageDir: string;
  imageFiles: string[];
  imageWidth: number;
  imageHeight: number;

  // Model state
  selectedModel: string;
  modelMetadata: ModelMetadata | null;

  // Config
  mode: "accumulative" | "incremental";

  // ROI
  roiConfirmed: boolean;

  // Processing
  processingActive: boolean;
  processingPaused: boolean;
  processingProgress: number;
  processingCurrent: number;
  processingTotal: number;

  // Displacement
  hasResults: boolean;
  numFrames: number;
  currentFrame: number;
  referenceFrame: number;

  // Strain
  hasStrain: boolean;
  strainComputing: boolean;
  strainProgress: number;
  strainComponents: string[];

  // Display
  displayComponent: DisplayComponent;
  visSettings: VisSettings;
  arrowSettings: ArrowSettings;

  // Probes
  probes: ProbeData[];
  activeProbeTab: "point" | "line" | "area";
  probePlacingMode: "point" | "line" | "area-rect" | "area-circle" | "area-poly" | null;
  probePlacingFirst: [number, number] | null;
  areaPolyPoints: [number, number][];

  // View zoom & pan (shared by displacement + postprocessing)
  viewZoom: number;
  viewOffset: { x: number; y: number };

  // Export
  exportActive: boolean;
  exportProgress: number;

  // Actions
  resetSession: () => void;
  setImageDir: (dir: string) => void;
  setImageFiles: (files: string[], w: number, h: number) => void;
  setModel: (path: string, meta: ModelMetadata | null) => void;
  setMode: (mode: "accumulative" | "incremental") => void;
  setRoiConfirmed: (v: boolean) => void;
  setProcessing: (active: boolean, progress?: number, current?: number, total?: number) => void;
  setProcessingPaused: (paused: boolean) => void;
  setResults: (numFrames: number) => void;
  setCurrentFrame: (frame: number) => void;
  setReferenceFrame: (frame: number) => void;
  setStrain: (has: boolean, components?: string[]) => void;
  setStrainComputing: (v: boolean, progress?: number) => void;
  setDisplayComponent: (c: DisplayComponent) => void;
  updateVisSettings: (partial: Partial<VisSettings>) => void;
  updateArrowSettings: (partial: Partial<ArrowSettings>) => void;
  setProbes: (probes: ProbeData[]) => void;
  setActiveProbeTab: (tab: "point" | "line" | "area") => void;
  setProbePlacingMode: (mode: "point" | "line" | "area-rect" | "area-circle" | "area-poly" | null) => void;
  setProbePlacingFirst: (pt: [number, number] | null) => void;
  addAreaPolyPoint: (x: number, y: number) => void;
  clearAreaPolyPoints: () => void;
  setExport: (active: boolean, progress?: number) => void;
  setViewOffset: (offset: { x: number; y: number }) => void;
  zoomIn: () => void;
  zoomOut: () => void;
  zoomReset: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  imageDir: "",
  imageFiles: [],
  imageWidth: 0,
  imageHeight: 0,
  selectedModel: "",
  modelMetadata: null,
  mode: "accumulative",
  roiConfirmed: false,
  processingActive: false,
  processingPaused: false,
  processingProgress: 0,
  processingCurrent: 0,
  processingTotal: 0,
  hasResults: false,
  numFrames: 0,
  currentFrame: 0,
  referenceFrame: 0,
  hasStrain: false,
  strainComputing: false,
  strainProgress: 0,
  strainComponents: [],
  displayComponent: "u",
  visSettings: {
    colormap: "turbo",
    alpha: 0.7,
    background: "reference",
    logScale: false,
    smoothSigma: 0,
    physicalEnabled: false,
    physicalRatio: 1.0,
    physicalUnit: "px",
    fps: 1.0,
    fixedRange: false,
    vminU: "",
    vmaxU: "",
    vminV: "",
    vmaxV: "",
  },
  arrowSettings: {
    showQuiver: false,
    showStreamlines: false,
    spacing: 30,
    scale: 20,
    color: "white",
    lineWidth: 1.5,
    streamQuality: 4,
  },
  probes: [],
  activeProbeTab: "point",
  probePlacingMode: null,
  probePlacingFirst: null,
  areaPolyPoints: [],
  viewZoom: 1,
  viewOffset: { x: 0, y: 0 },
  exportActive: false,
  exportProgress: 0,

  resetSession: () =>
    set({
      imageWidth: 0,
      imageHeight: 0,
      roiConfirmed: false,
      processingActive: false,
      processingPaused: false,
      processingProgress: 0,
      processingCurrent: 0,
      processingTotal: 0,
      hasResults: false,
      numFrames: 0,
      currentFrame: 0,
      referenceFrame: 0,
      hasStrain: false,
      strainComputing: false,
      strainProgress: 0,
      strainComponents: [],
      displayComponent: "u",
      probes: [],
      probePlacingMode: null,
      probePlacingFirst: null,
      areaPolyPoints: [],
      viewZoom: 1,
      viewOffset: { x: 0, y: 0 },
      exportActive: false,
      exportProgress: 0,
    }),
  setImageDir: (dir) => set({ imageDir: dir }),
  setImageFiles: (files, w, h) =>
    set({ imageFiles: files, imageWidth: w, imageHeight: h }),
  setModel: (path, meta) => set({ selectedModel: path, modelMetadata: meta }),
  setMode: (mode) => set({ mode }),
  setRoiConfirmed: (v) => set({ roiConfirmed: v }),
  setProcessing: (active, progress = 0, current = 0, total = 0) =>
    set({
      processingActive: active,
      processingPaused: false,
      processingProgress: progress,
      processingCurrent: current,
      processingTotal: total,
    }),
  setProcessingPaused: (paused) => set({ processingPaused: paused }),
  setResults: (numFrames) =>
    set({ hasResults: numFrames > 0, numFrames, currentFrame: 0 }),
  setCurrentFrame: (frame) => set({ currentFrame: frame }),
  setReferenceFrame: (frame) => set({ referenceFrame: frame }),
  setStrain: (has, components = []) =>
    set({ hasStrain: has, strainComponents: components, strainComputing: false }),
  setStrainComputing: (v, progress = 0) => set({ strainComputing: v, strainProgress: progress }),
  setDisplayComponent: (c) => set({ displayComponent: c }),
  updateVisSettings: (partial) =>
    set((s) => ({ visSettings: { ...s.visSettings, ...partial } })),
  updateArrowSettings: (partial) =>
    set((s) => ({ arrowSettings: { ...s.arrowSettings, ...partial } })),
  setProbes: (probes) => set({ probes }),
  setActiveProbeTab: (tab) => set({ activeProbeTab: tab }),
  setProbePlacingMode: (mode) => set({ probePlacingMode: mode, probePlacingFirst: null, areaPolyPoints: [] }),
  setProbePlacingFirst: (pt) => set({ probePlacingFirst: pt }),
  addAreaPolyPoint: (x, y) => set((s) => ({ areaPolyPoints: [...s.areaPolyPoints, [x, y]] })),
  clearAreaPolyPoints: () => set({ areaPolyPoints: [] }),
  setExport: (active, progress = 0) =>
    set({ exportActive: active, exportProgress: progress }),
  setViewOffset: (offset) => set({ viewOffset: offset }),
  zoomIn: () => set((s) => ({ viewZoom: Math.min(s.viewZoom * 1.25, 5) })),
  zoomOut: () => set((s) => ({ viewZoom: Math.max(s.viewZoom / 1.25, 0.2) })),
  zoomReset: () => set({ viewZoom: 1, viewOffset: { x: 0, y: 0 } }),
}));
