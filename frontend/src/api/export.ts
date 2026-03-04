import client from "./client";

export async function exportScientific(params: {
  file_path: string;
  upsample_strain?: boolean;
  metadata?: Record<string, unknown>;
}): Promise<{ ok: boolean; path: string }> {
  const { data } = await client.post("/export/scientific", params);
  return data;
}

export async function exportImages(params: {
  output_dir: string;
  components: Record<string, { vmin?: number; vmax?: number }>;
  frame_range: [number, number];
  settings: Record<string, unknown>;
}): Promise<void> {
  await client.post("/export/images", params);
}

export async function getExportStatus(): Promise<{
  active: boolean;
  progress: number;
  total: number;
  percent: number;
}> {
  const { data } = await client.get("/export/images/status");
  return data;
}
