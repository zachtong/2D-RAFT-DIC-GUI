import { create } from "zustand";

export type ShapeMode = "polygon" | "rectangle" | "circle";
export type DrawingMode = ShapeMode | null;

interface RoiState {
  drawingMode: DrawingMode;
  cutMode: boolean;
  currentPoints: [number, number][];
  maskUrl: string | null;
  showImportDialog: boolean;

  setDrawingMode: (mode: DrawingMode) => void;
  setCutMode: (cut: boolean) => void;
  activateTool: (shape: ShapeMode, cut: boolean) => void;
  addPoint: (x: number, y: number) => void;
  clearPoints: () => void;
  setMaskUrl: (url: string | null) => void;
  setShowImportDialog: (v: boolean) => void;
}

export const useRoiStore = create<RoiState>((set, get) => ({
  drawingMode: null,
  cutMode: false,
  currentPoints: [],
  maskUrl: null,
  showImportDialog: false,

  setDrawingMode: (mode) => set({ drawingMode: mode, currentPoints: [] }),
  setCutMode: (cut) => set({ cutMode: cut }),
  activateTool: (shape, cut) => {
    const s = get();
    // Toggle off if same tool re-clicked
    if (s.drawingMode === shape && s.cutMode === cut) {
      set({ drawingMode: null, currentPoints: [] });
    } else {
      set({ drawingMode: shape, cutMode: cut, currentPoints: [] });
    }
  },
  addPoint: (x, y) =>
    set((s) => ({ currentPoints: [...s.currentPoints, [x, y]] })),
  clearPoints: () => set({ currentPoints: [] }),
  setMaskUrl: (url) => set({ maskUrl: url }),
  setShowImportDialog: (v) => set({ showImportDialog: v }),
}));
