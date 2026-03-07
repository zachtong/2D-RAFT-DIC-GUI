import { create } from "zustand";

export type DrawingMode = "polygon" | "rectangle" | "circle" | "cut" | null;

interface RoiState {
  drawingMode: DrawingMode;
  currentPoints: [number, number][];
  maskUrl: string | null;
  showImportDialog: boolean;

  setDrawingMode: (mode: DrawingMode) => void;
  addPoint: (x: number, y: number) => void;
  clearPoints: () => void;
  setMaskUrl: (url: string | null) => void;
  setShowImportDialog: (v: boolean) => void;
}

export const useRoiStore = create<RoiState>((set) => ({
  drawingMode: null,
  currentPoints: [],
  maskUrl: null,
  showImportDialog: false,

  setDrawingMode: (mode) => set({ drawingMode: mode, currentPoints: [] }),
  addPoint: (x, y) =>
    set((s) => ({ currentPoints: [...s.currentPoints, [x, y]] })),
  clearPoints: () => set({ currentPoints: [] }),
  setMaskUrl: (url) => set({ maskUrl: url }),
  setShowImportDialog: (v) => set({ showImportDialog: v }),
}));
