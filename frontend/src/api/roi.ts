import client from "./client";
import type { RoiResult } from "@/types/api";

export async function addPolygon(
  points: [number, number][],
  mode: "add" | "cut" = "add"
): Promise<RoiResult> {
  const { data } = await client.post<RoiResult>("/roi/polygon", { points, mode });
  return data;
}

export async function addRectangle(
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  mode: "add" | "cut" = "add"
): Promise<RoiResult> {
  const { data } = await client.post<RoiResult>("/roi/rectangle", { x0, y0, x1, y1, mode });
  return data;
}

export async function addCircle(
  cx: number,
  cy: number,
  r: number,
  mode: "add" | "cut" = "add"
): Promise<RoiResult> {
  const { data } = await client.post<RoiResult>("/roi/circle", { cx, cy, r, mode });
  return data;
}

export async function importMask(path: string): Promise<RoiResult> {
  const { data } = await client.post<RoiResult>("/roi/import", { path });
  return data;
}

export async function invertMask(): Promise<RoiResult> {
  const { data } = await client.post<RoiResult>("/roi/invert");
  return data;
}

export function maskImageUrl(): string {
  return "/api/roi/mask";
}

export async function confirmRoi(): Promise<RoiResult> {
  const { data } = await client.post<RoiResult>("/roi/confirm");
  return data;
}

export async function clearRoi(): Promise<void> {
  await client.delete("/roi");
}
