import client from "./client";
import type { ImageLoadResult } from "@/types/api";

export interface BrowseEntry {
  name: string;
  type: "dir" | "image";
}

export interface BrowseResult {
  path: string;
  parent: string;
  entries: BrowseEntry[];
  image_count: number;
}

export async function browseDirectory(path: string): Promise<BrowseResult> {
  const { data } = await client.post<BrowseResult>("/images/browse", { path });
  return data;
}

export async function loadImages(dir: string, naturalSort = false): Promise<ImageLoadResult> {
  const { data } = await client.post<ImageLoadResult>("/images/load", { dir, natural_sort: naturalSort });
  return data;
}

export function referenceImageUrl(): string {
  return "/api/images/reference";
}

export function frameImageUrl(index: number): string {
  return `/api/images/frame/${index}`;
}
