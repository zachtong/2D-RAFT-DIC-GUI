"""Load and match per-frame masks from a user-provided folder."""

import os
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

MASK_EXTENSIONS = {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}


def _extract_last_integer(name: str) -> int | None:
    """Extract the last integer from a filename string."""
    matches = re.findall(r"\d+", name)
    if matches:
        return int(matches[-1])
    return None


def _load_mask(path: Path, image_shape: Tuple[int, int]) -> np.ndarray | None:
    """Load a mask image, validate dimensions, and return a boolean array.

    Returns None if the mask cannot be loaded or has wrong dimensions.
    """
    from PIL import Image

    try:
        img = Image.open(path)
        arr = np.array(img)
    except Exception as e:
        print(f"{path.name}: failed to load ({e}), skipped")
        return None

    # Convert RGB/RGBA to grayscale by taking first channel
    if arr.ndim == 3:
        arr = arr[:, :, 0]

    # Validate dimensions
    h, w = image_shape
    if arr.shape[0] != h or arr.shape[1] != w:
        print(
            f"{path.name}: size {arr.shape[1]}x{arr.shape[0]} doesn't match "
            f"image {w}x{h}, skipped"
        )
        return None

    # Binarize
    return arr > 0


def discover_masks(
    mask_dir: str,
    image_files: List[str],
    image_shape: Tuple[int, int],
) -> Dict[int, np.ndarray]:
    """Discover and load per-frame masks from a directory.

    Matching strategy (priority order):
    1. Filename match: mask stem == image stem (ignoring extension)
    2. Number extraction: last integer in mask filename as 1-indexed frame number

    Parameters
    ----------
    mask_dir : str
        Path to folder containing mask images.
    image_files : list of str
        Ordered list of image filenames (basenames) for matching.
    image_shape : (int, int)
        Expected (H, W) dimensions for masks.

    Returns
    -------
    dict
        Mapping of 0-based frame index to boolean numpy array (H, W).
    """
    mask_path = Path(mask_dir)
    if not mask_path.is_dir():
        print(f"Mask directory not found: {mask_dir}")
        return {}

    # Build stem-to-index lookup from image files
    stem_to_index: Dict[str, int] = {}
    for idx, fname in enumerate(image_files):
        stem = Path(fname).stem
        stem_to_index[stem] = idx

    num_images = len(image_files)
    result: Dict[int, np.ndarray] = {}

    # Collect mask files
    mask_files = sorted(
        f for f in mask_path.iterdir()
        if f.is_file() and f.suffix.lower() in MASK_EXTENSIONS
    )

    for mask_file in mask_files:
        mask_stem = mask_file.stem
        frame_idx: int | None = None

        # Strategy 1: filename match
        if mask_stem in stem_to_index:
            frame_idx = stem_to_index[mask_stem]
        else:
            # Strategy 2: number extraction (1-indexed)
            frame_num = _extract_last_integer(mask_stem)
            if frame_num is not None:
                candidate = frame_num - 1  # convert to 0-based
                if 0 <= candidate < num_images:
                    frame_idx = candidate
                else:
                    print(
                        f"{mask_file.name}: frame number {frame_num} out of "
                        f"range (1-{num_images}), skipped"
                    )
                    continue

        if frame_idx is None:
            print(f"{mask_file.name}: no matching frame found, skipped")
            continue

        # Load and validate mask
        mask = _load_mask(mask_file, image_shape)
        if mask is None:
            continue

        result[frame_idx] = mask

    # Summary
    if not result:
        print("No valid masks found, using auto warp for all frames")
    else:
        print(
            f"Found masks for {len(result)} of {num_images} frames. "
            f"Remaining frames will use auto warp."
        )

    return result
