import { useState, useEffect, useCallback } from "react";
import { browseDirectory, type BrowseEntry } from "@/api/images";
import { Folder, Image, ArrowUp, X, Check } from "lucide-react";

interface FileBrowserProps {
  open: boolean;
  onClose: () => void;
  onSelect: (path: string) => void;
  initialPath?: string;
}

export function FileBrowser({ open, onClose, onSelect, initialPath = "" }: FileBrowserProps) {
  const [currentPath, setCurrentPath] = useState(initialPath);
  const [parentPath, setParentPath] = useState("");
  const [entries, setEntries] = useState<BrowseEntry[]>([]);
  const [imageCount, setImageCount] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const browse = useCallback(async (path: string) => {
    setLoading(true);
    setError("");
    try {
      const result = await browseDirectory(path);
      setCurrentPath(result.path);
      setParentPath(result.parent);
      setEntries(result.entries);
      setImageCount(result.image_count);
    } catch (e: any) {
      setError(e.response?.data?.error ?? "Failed to browse directory");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (open) browse(initialPath);
  }, [open, initialPath, browse]);

  if (!open) return null;

  const dirs = entries.filter((e) => e.type === "dir");
  const images = entries.filter((e) => e.type === "image");

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-[var(--card)] border border-[var(--border)] rounded-lg shadow-xl w-[500px] max-h-[70vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--border)]">
          <h3 className="text-[13px] font-semibold text-[var(--foreground)]">Select Image Directory</h3>
          <button onClick={onClose} className="p-1 hover:bg-[var(--secondary)] rounded text-[var(--muted-foreground)]">
            <X size={14} />
          </button>
        </div>

        {/* Path bar */}
        <div className="flex items-center gap-2 px-4 py-2 border-b border-[var(--border)] bg-[var(--background)]">
          <button
            onClick={() => browse(parentPath)}
            disabled={currentPath === parentPath}
            className="p-1 hover:bg-[var(--secondary)] rounded text-[var(--muted-foreground)] disabled:opacity-30"
          >
            <ArrowUp size={14} />
          </button>
          <span className="text-[11px] text-[var(--foreground)] truncate flex-1 font-mono">
            {currentPath}
          </span>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto min-h-0 px-2 py-1">
          {loading && (
            <div className="flex items-center justify-center py-8 text-[11px] text-[var(--muted-foreground)]">
              Loading...
            </div>
          )}
          {error && (
            <div className="text-[11px] text-red-400 py-4 text-center">{error}</div>
          )}
          {!loading && !error && entries.length === 0 && (
            <div className="text-[11px] text-[var(--muted-foreground)] py-4 text-center">
              Empty directory
            </div>
          )}
          {!loading && dirs.map((entry) => (
            <button
              key={entry.name}
              onClick={() => browse(currentPath + "/" + entry.name)}
              className="flex items-center gap-2 w-full px-2 py-1.5 rounded hover:bg-[var(--secondary)] text-left"
            >
              <Folder size={14} className="text-[var(--primary)] shrink-0" />
              <span className="text-[11px] text-[var(--foreground)] truncate">{entry.name}</span>
            </button>
          ))}
          {!loading && images.length > 0 && (
            <div className="mt-1 pt-1 border-t border-[var(--border)]">
              <span className="text-[9px] text-[var(--muted-foreground)] px-2 uppercase tracking-wider">
                {images.length} image{images.length !== 1 ? "s" : ""} in this directory
              </span>
              {images.slice(0, 5).map((entry) => (
                <div key={entry.name} className="flex items-center gap-2 px-2 py-1">
                  <Image size={12} className="text-[var(--muted-foreground)] shrink-0" />
                  <span className="text-[10px] text-[var(--muted-foreground)] truncate">{entry.name}</span>
                </div>
              ))}
              {images.length > 5 && (
                <span className="text-[9px] text-[var(--muted-foreground)] px-2">
                  ... and {images.length - 5} more
                </span>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-4 py-3 border-t border-[var(--border)]">
          <span className="text-[10px] text-[var(--muted-foreground)]">
            {imageCount > 0
              ? `${imageCount} image${imageCount !== 1 ? "s" : ""} found`
              : "No images in this directory"}
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={onClose}
              className="px-3 py-1.5 rounded text-[11px] text-[var(--muted-foreground)] hover:bg-[var(--secondary)]"
            >
              Cancel
            </button>
            <button
              onClick={() => {
                onSelect(currentPath);
                onClose();
              }}
              disabled={imageCount === 0}
              className="flex items-center gap-1 px-3 py-1.5 bg-[var(--primary)] hover:bg-[var(--primary)]/90 text-white rounded text-[11px] disabled:opacity-40"
            >
              <Check size={12} />
              Select ({imageCount})
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
