import client from "./client";
import type { RoiResult } from "@/types/api";

export async function addPolygon(
  points: [number, number][],
  mode: "add" | "cut" = "add",
  frameIdx: number = 0,
): Promise<RoiResult> {
  const { data } = await client.post<RoiResult>("/roi/polygon", {
    points, mode, frame_idx: frameIdx,
  });
  return data;
}

export async function addRectangle(
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  mode: "add" | "cut" = "add",
  frameIdx: number = 0,
): Promise<RoiResult> {
  const { data } = await client.post<RoiResult>("/roi/rectangle", {
    x0, y0, x1, y1, mode, frame_idx: frameIdx,
  });
  return data;
}

export async function addCircle(
  cx: number,
  cy: number,
  r: number,
  mode: "add" | "cut" = "add",
  frameIdx: number = 0,
): Promise<RoiResult> {
  const { data } = await client.post<RoiResult>("/roi/circle", {
    cx, cy, r, mode, frame_idx: frameIdx,
  });
  return data;
}

export async function importMask(
  path: string,
  minArea: number = 0,
  smoothRadius: number = 0,
  frameIdx: number = 0,
): Promise<RoiResult> {
  const { data } = await client.post<RoiResult>("/roi/import", {
    path, min_area: minArea, smooth_radius: smoothRadius,
    frame_idx: frameIdx,
  });
  return data;
}

export async function invertMask(frameIdx: number = 0): Promise<RoiResult> {
  const { data } = await client.post<RoiResult>("/roi/invert", {
    frame_idx: frameIdx,
  });
  return data;
}

export function maskImageUrl(frameIdx: number = 0): string {
  return `/api/roi/mask?frame_idx=${frameIdx}`;
}

export async function confirmRoi(): Promise<RoiResult> {
  const { data } = await client.post<RoiResult>("/roi/confirm");
  return data;
}

export async function clearRoi(): Promise<void> {
  await client.delete("/roi");
}

export async function clearFrameRoi(frameIdx: number): Promise<void> {
  await client.delete(`/roi/frame/${frameIdx}`);
}

export interface FramesRoiStatus {
  total_frames: number;
  frames_with_roi: number[];
  frame_0_confirmed: boolean;
}

export async function getFramesRoiStatus(): Promise<FramesRoiStatus> {
  const { data } = await client.get<FramesRoiStatus>("/roi/frames/status");
  return data;
}

export interface BatchImportResult {
  imported: number;
  total_files: number;
  assignments: Record<string, string>;
}

export async function batchImportRois(
  folder: string,
  strategy: "auto_match" | "sequential" = "sequential",
): Promise<BatchImportResult> {
  const { data } = await client.post<BatchImportResult>("/roi/frames/batch-import", {
    folder, strategy,
  });
  return data;
}

export async function exportMask(): Promise<void> {
  const resp = await client.get("/roi/mask/binary", { responseType: "blob" });
  const url = URL.createObjectURL(resp.data);
  const a = document.createElement("a");
  a.href = url;
  a.download = "roi_mask.png";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
