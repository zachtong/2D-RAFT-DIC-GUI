/* ─── Backend API types ─── */

export interface DICConfig {
  imgDir: string;
  modelPath: string;
  mode: "accumulative" | "incremental";
  contextPadding: number;
  tileOverlap: number;
  useSmooth: boolean;
  sigma: number;
  safetyFactor: number;
  pMaxPixels: number;
  device: string;
  projectRoot: string;
}

export interface ModelEntry {
  path: string;
  label: string;
}

export interface ModelMetadata {
  path: string;
  label: string;
  small: boolean;
  mixed_precision: boolean;
  alternate_corr: boolean;
  corr_levels: number | null;
  corr_radius: number | null;
  variant: string;
  full_resolution: boolean;
  summary: string;
  safe_pmax: number;
}

export interface ImageLoadResult {
  files: string[];
  count: number;
  width: number;
  height: number;
  sort_suggestion?: "natural";
}

export interface RoiResult {
  rect: [number, number, number, number] | null;
  area_px: number;
}

export interface ProcessingProgress {
  percent: number;
  current: number;
  total: number;
}

export interface DisplacementInfo {
  num_frames: number;
  roi_rect: [number, number, number, number];
  roi_shape: [number, number];
}

export interface FrameData {
  data: number[][];
  shape: [number, number];
  vmin: number;
  vmax: number;
  mean: number;
}

export interface StrainStatus {
  computed: boolean;
  computing: boolean;
  components: string[];
  num_frames: number;
}

export interface ProbeData {
  id: number;
  type: "point" | "line" | "area";
  coords: unknown;
  color: string;
  label: string;
}

export interface TimeSeriesResult {
  series: Record<string, (number | null)[]>;
}

export interface KymographResult {
  data: number[][];
  dist_axis: number[];
  shape: [number, number];
}

export interface ExportProgress {
  current: number;
  total: number;
  percent: number;
  message: string;
}

export type StrainComponent =
  | "exx" | "eyy" | "exy"
  | "e1" | "e2"
  | "max_shear" | "von_mises" | "rotation"
  | "rotation_cumulative"
  | "confidence"
  | "dexx_dt" | "deyy_dt" | "dexy_dt";

export type DisplacementComponent = "u" | "v" | "magnitude" | "velocity";

export type DisplayComponent = DisplacementComponent | StrainComponent;
