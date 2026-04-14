import { create } from "zustand";

export type ShapeMode = "polygon" | "rectangle" | "circle";
export type DrawingMode = ShapeMode | null;

interface RoiState {
  drawingMode: DrawingMode;
  cutMode: boolean;
  currentPoints: [number, number][];
  maskUrl: string | null;
  showImportDialog: boolean;
  showBatchImportDialog: boolean;

  // Per-frame ROI editing
  editingFrameIdx: number; // 0-based frame index being edited
  framesWithRoi: number[]; // which frames have ROI

  setDrawingMode: (mode: DrawingMode) => void;
  setCutMode: (cut: boolean) => void;
  activateTool: (shape: ShapeMode, cut: boolean) => void;
  addPoint: (x: number, y: number) => void;
  clearPoints: () => void;
  setMaskUrl: (url: string | null) => void;
  setShowImportDialog: (v: boolean) => void;
  setShowBatchImportDialog: (v: boolean) => void;

  // Per-frame actions
  setEditingFrameIdx: (idx: number) => void;
  setFramesWithRoi: (frames: number[]) => void;
  refreshMaskUrl: () => void;
}

export const useRoiStore = create<RoiState>((set, get) => ({
  drawingMode: null,
  cutMode: false,
  currentPoints: [],
  maskUrl: null,
  showImportDialog: false,
  showBatchImportDialog: false,
  editingFrameIdx: 0,
  framesWithRoi: [],

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
  setShowBatchImportDialog: (v) => set({ showBatchImportDialog: v }),

  setEditingFrameIdx: (idx) => {
    set({ editingFrameIdx: idx, currentPoints: [], drawingMode: null });
    // Update mask URL for the new frame
    get().refreshMaskUrl();
  },
  setFramesWithRoi: (frames) => set({ framesWithRoi: frames }),
  refreshMaskUrl: () => {
    const idx = get().editingFrameIdx;
    set({ maskUrl: `/api/roi/mask?frame_idx=${idx}&t=${Date.now()}` });
  },
}));
