import { create } from "zustand";

export type DrawingMode = "polygon" | "rectangle" | "circle" | "cut" | null;

interface RoiState {
  drawingMode: DrawingMode;
  currentPoints: [number, number][];
  maskUrl: string | null;

  setDrawingMode: (mode: DrawingMode) => void;
  addPoint: (x: number, y: number) => void;
  clearPoints: () => void;
  setMaskUrl: (url: string | null) => void;
}

export const useRoiStore = create<RoiState>((set) => ({
  drawingMode: null,
  currentPoints: [],
  maskUrl: null,

  setDrawingMode: (mode) => set({ drawingMode: mode, currentPoints: [] }),
  addPoint: (x, y) =>
    set((s) => ({ currentPoints: [...s.currentPoints, [x, y]] })),
  clearPoints: () => set({ currentPoints: [] }),
  setMaskUrl: (url) => set({ maskUrl: url }),
}));
