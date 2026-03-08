import dataclasses
from dataclasses import dataclass, field
from typing import Tuple, Optional, Any

@dataclass
class DICConfig:
    """
    Centralized configuration for RAFT-DIC processing.
    Holds all parameters required for the processing pipeline, decoupling them from UI state.
    """
    # Path Settings
    img_dir: str = ""
    project_root: str = ""
    model_path: str = ""
    model_label: str = ""

    # Processing Mode
    mode: str = "accumulative"  # 'accumulative' or 'incremental'

    # Smoothing Settings
    use_smooth: bool = True
    sigma: float = 2.0

    # Tiling / ROI Settings
    context_padding: int = 64    # Context padding (px)
    tile_overlap: int = 64       # Tile overlap (px)
    p_max_pixels: int = 1100 * 1100

    # Runtime / Hardware
    device: str = "cuda"

    # Incremental mode settings
    key_frames: Optional[list] = None        # User-specified key frames (1-indexed)
    key_frame_interval: Optional[int] = None  # Every N frames shortcut
    mask_dir: Optional[str] = None            # Per-frame mask folder path
    use_median_filter: bool = False           # Median filter accumulated displacement (reduces error accumulation)

    # Internal / Metadata
    model_metadata: Optional[Any] = None

    def validate(self) -> Tuple[bool, str]:
        """Validate the configuration. Returns (is_valid, error_message)."""
        if not self.img_dir:
            return False, "Input directory is not selected."
        # if not self.project_root:
        #     return False, "Output directory is not selected."
        if not self.model_path:
            return False, "Model checkpoint is not selected."

        return True, ""
