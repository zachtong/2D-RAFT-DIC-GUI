import os
import json
import time
import numpy as np
import raft_dic_gui.processing as proc
import raft_dic_gui.model as mdl
from raft_dic_gui.config import DICConfig
from raft_dic_gui.incremental import (
    build_ref_map,
    key_frames_every_n,
    accumulate_displacement,
    warp_mask_with_holes,
)


class DICProcessor:
    """
    Controller class for RAFT-DIC processing.
    Handles the main processing loop, interacting with the processing module
    and reporting progress back to the GUI via callbacks.

    The processing loop is driven by a **ref_map** (frame -> reference frame,
    both 1-indexed).  Different strategies (accumulative, key-frame intervals,
    explicit key frames) simply produce different ref_maps.
    """

    def __init__(self, update_progress_callback=None, check_stop_callback=None):
        """
        Initialize the processor.
        :param update_progress_callback: Callable(percent, current, total) to update UI progress.
        :param check_stop_callback: Callable() -> bool that returns True if processing should stop.
        """
        self.update_progress = update_progress_callback
        self.check_stop = check_stop_callback
        self.displacement_results = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, config: DICConfig, roi_mask: np.ndarray, roi_rect: tuple):
        """
        Execute the processing pipeline.
        :param config: DICConfig object with all parameters.
        :param roi_mask: Boolean mask of the ROI (full image size).
        :param roi_rect: Tuple (xmin, ymin, xmax, ymax) of the ROI bounding box.
        :return: List of displacement fields (or paths to them).
        """
        img_dir = config.img_dir
        out_dir = config.project_root

        # Load model
        device = config.device
        try:
            model = mdl.load_model(config.model_path, device=device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {config.model_path}: {e}")

        # Get image file list
        image_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'))
        ])

        if len(image_files) < 2:
            raise Exception("At least 2 images are needed")

        # Store absolute paths for export
        self.image_files = [os.path.abspath(os.path.join(img_dir, f)) for f in image_files]

        if roi_mask is None:
            raise Exception("ROI mask is missing")

        xmin, ymin, xmax, ymax = roi_rect
        H_roi = ymax - ymin
        W_roi = xmax - xmin

        # Ensure output directory exists
        os.makedirs(out_dir, exist_ok=True)

        # ----------------------------------------------------------
        # Build ref_map
        # ----------------------------------------------------------
        total_frames = len(image_files)  # 1-indexed: frames 1..total_frames
        ref_map = self._build_ref_map(config, total_frames)

        # Set of frames that are referenced by other frames (for caching)
        key_frame_set = set(ref_map.values())
        key_frame_set.add(1)  # Frame 1 is always a key frame

        print(f"[INFO] ref_map built: {len(ref_map)} entries, "
              f"key frames = {sorted(key_frame_set)}")

        # ----------------------------------------------------------
        # Smart Resume: Check if configuration has changed
        # ----------------------------------------------------------
        force_rerun = self._check_config_changed(config, roi_rect, out_dir)

        # ----------------------------------------------------------
        # Discover per-frame user masks (if mask_dir configured)
        # ----------------------------------------------------------
        user_masks = {}
        if config.mask_dir:
            from raft_dic_gui.mask_loader import discover_masks
            ref_image_tmp = proc.load_and_convert_image(
                os.path.join(img_dir, image_files[0])
            )
            image_shape = (ref_image_tmp.shape[0], ref_image_tmp.shape[1])
            user_masks = discover_masks(config.mask_dir, image_files, image_shape)
            del ref_image_tmp

        # ----------------------------------------------------------
        # State caches
        # ----------------------------------------------------------
        # kf_accumulated[frame_num] = total displacement (H_roi, W_roi, 2)
        # relative to Frame 1 for each key frame.
        kf_accumulated = {
            1: np.zeros((H_roi, W_roi, 2), dtype=np.float64),
        }

        # kf_images[frame_num] = loaded full-size image for reference frames
        kf_images = {}

        # Original ROI mask crop (for auto-warping)
        original_mask_crop = roi_mask[ymin:ymax, xmin:xmax].copy()

        # Load frame 1 image (always needed as a reference)
        ref1_path = os.path.join(img_dir, image_files[0])
        kf_images[1] = proc.load_and_convert_image(ref1_path)

        total_pairs = total_frames - 1

        if self.update_progress:
            self.update_progress(0, 0, total_pairs)

        sequence_displacements = []
        last_update_time = 0

        # ----------------------------------------------------------
        # Main processing loop: frames 2..N  (1-indexed)
        # ----------------------------------------------------------
        for frame_num in range(2, total_frames + 1):
            # frame_num is 1-indexed; list index is frame_num - 1
            list_idx = frame_num - 1
            pair_idx = frame_num - 1  # pair index for progress (1-based pair count)

            # Check for stop request
            if self.check_stop and self.check_stop():
                break

            # ----------------------------------------------------------
            # Resume: try to load cached result
            # ----------------------------------------------------------
            result_filename = f"displacement_field_{list_idx}.npy"
            result_path = os.path.join(
                out_dir, "displacement_results_temp_npy", result_filename
            )

            if not force_rerun and os.path.exists(result_path):
                loaded = self._try_load_cached(
                    result_path, list_idx, frame_num, key_frame_set,
                    kf_accumulated, kf_images, img_dir, image_files,
                )
                if loaded is not None:
                    sequence_displacements.append(loaded)
                    # Update progress even for cached frames
                    last_update_time = self._update_progress_throttled(
                        pair_idx, total_pairs, last_update_time
                    )
                    continue

            # ----------------------------------------------------------
            # Progress update (rate-limited to 10Hz)
            # ----------------------------------------------------------
            last_update_time = self._update_progress_throttled(
                pair_idx, total_pairs, last_update_time
            )

            # ----------------------------------------------------------
            # Determine reference and load images
            # ----------------------------------------------------------
            ref_num = ref_map[frame_num]

            # Ensure reference image is loaded
            if ref_num not in kf_images:
                ref_path = os.path.join(img_dir, image_files[ref_num - 1])
                kf_images[ref_num] = proc.load_and_convert_image(ref_path)

            ref_image = kf_images[ref_num]

            # Load deformed image
            def_img_path = os.path.join(img_dir, image_files[list_idx])
            def_image = proc.load_and_convert_image(def_img_path)

            # ----------------------------------------------------------
            # Determine mask for this frame
            # ----------------------------------------------------------
            current_mask = self._resolve_mask(
                frame_num, list_idx, ref_num,
                roi_mask, original_mask_crop,
                kf_accumulated, user_masks,
                xmin, ymin, xmax, ymax,
            )

            # ----------------------------------------------------------
            # Run DIC
            # ----------------------------------------------------------
            disp_full, _ = proc.dic_over_roi_with_tiling(
                ref_image,
                def_image,
                current_mask,
                model,
                device,
                context_padding=config.context_padding,
                tile_overlap=config.tile_overlap,
                p_max_pixels=config.p_max_pixels,
                use_smooth=config.use_smooth,
                sigma=config.sigma,
            )
            delta_disp = disp_full[ymin:ymax, xmin:xmax, :]

            # ----------------------------------------------------------
            # Compute total displacement relative to Frame 1
            # ----------------------------------------------------------
            kf_acc = kf_accumulated[ref_num]
            is_first_segment = np.allclose(kf_acc, 0, atol=1e-12, equal_nan=False) and not np.any(np.isnan(kf_acc))

            if is_first_segment:
                # First segment (ref == frame 1 with zero accumulated):
                # delta IS the total displacement
                total_disp = delta_disp.copy()
            else:
                # Subsequent segments: accumulate
                total_disp = accumulate_displacement(
                    kf_acc, delta_disp, debug=True
                )

            displacement_field = total_disp

            # ----------------------------------------------------------
            # Cache key frame accumulated displacement
            # ----------------------------------------------------------
            if frame_num in key_frame_set:
                kf_accumulated[frame_num] = total_disp.copy()
                # Load and cache this frame's image for future reference
                if frame_num not in kf_images:
                    kf_images[frame_num] = def_image
                print(f"[INFO] Cached key frame {frame_num} accumulated displacement")

            # ----------------------------------------------------------
            # Evict reference images that are no longer needed
            # ----------------------------------------------------------
            self._evict_unused_ref_images(
                frame_num, ref_map, key_frame_set, kf_images, total_frames
            )

            # ----------------------------------------------------------
            # Warn if significant pixel loss
            # ----------------------------------------------------------
            valid_pixels = int(np.sum(~np.isnan(total_disp[..., 0])))
            total_pixels = H_roi * W_roi
            loss_pct = (1 - valid_pixels / total_pixels) * 100
            if loss_pct > 10:
                print(
                    f"[WARNING] Frame {frame_num}: {loss_pct:.1f}% of ROI pixels "
                    f"lost ({valid_pixels}/{total_pixels} valid)."
                )

            # ----------------------------------------------------------
            # Save immediately (resume support)
            # ----------------------------------------------------------
            try:
                proc.save_displacement_results(
                    displacement_field,
                    out_dir,
                    index=list_idx,
                    roi_rect=roi_rect,
                    roi_mask=roi_mask,
                )
            except Exception as e:
                print(f"[ERROR] Failed to save result for frame {frame_num}: {e}")

            sequence_displacements.append(displacement_field)

        self.displacement_results = sequence_displacements
        return sequence_displacements

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_ref_map(config: DICConfig, total_frames: int):
        """Build the frame -> reference mapping from config settings.

        Strategy priority:
        1. key_frame_interval -> generate key frames via key_frames_every_n()
        2. key_frames -> use directly
        3. Default: key_frames=[1] (all frames reference frame 1)

        For accumulative mode the default is used, which makes every frame
        reference frame 1 -- equivalent to the original accumulative behavior.
        """
        kf = None

        if config.key_frame_interval is not None and config.key_frame_interval >= 1:
            kf = key_frames_every_n(total_frames, config.key_frame_interval)
        elif config.key_frames is not None and len(config.key_frames) > 0:
            kf = list(config.key_frames)

        # Default: single segment referencing frame 1
        if kf is None:
            kf = [1]

        return build_ref_map(total_frames, key_frames=kf)

    def _check_config_changed(self, config: DICConfig, roi_rect: tuple, out_dir: str) -> bool:
        """Compare current config to saved config. Returns True if force rerun needed."""
        config_path = os.path.join(out_dir, "run_config.json")

        # Serialize key_frames as sorted list for stable comparison
        kf_serialized = sorted(config.key_frames) if config.key_frames else None

        current_config_dict = {
            "model_path": config.model_path,
            "tile_overlap": config.tile_overlap,
            "context_padding": config.context_padding,
            "safety_factor": config.safety_factor,
            "use_smooth": config.use_smooth,
            "sigma": config.sigma,
            "p_max_pixels": config.p_max_pixels,
            "mode": config.mode,
            "roi_rect": list(roi_rect),
            # Incremental mode parameters
            "key_frames": kf_serialized,
            "key_frame_interval": config.key_frame_interval,
            "mask_dir": config.mask_dir,
        }

        force_rerun = False
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    saved_config = json.load(f)
                if saved_config != current_config_dict:
                    print("[INFO] Configuration changed. Forcing re-run.")
                    force_rerun = True
                else:
                    print("[INFO] Configuration matches. Resuming...")
            except Exception as e:
                print(f"[WARNING] Failed to read run config: {e}. Forcing re-run.")
                force_rerun = True

        # Save current config
        try:
            with open(config_path, 'w') as f:
                json.dump(current_config_dict, f, indent=4)
        except Exception as e:
            print(f"[WARNING] Failed to save run config: {e}")

        # Cleanup stale results if forcing re-run
        if force_rerun:
            import glob as glob_mod
            npy_dir = os.path.join(out_dir, "displacement_results_temp_npy")
            if os.path.exists(npy_dir):
                print(f"[INFO] Cleaning up stale results in {npy_dir}...")
                files = glob_mod.glob(os.path.join(npy_dir, "displacement_field_*.npy"))
                for f in files:
                    try:
                        os.remove(f)
                    except Exception as e:
                        print(f"[WARNING] Failed to delete stale file {f}: {e}")

        return force_rerun

    def _try_load_cached(
        self, result_path, list_idx, frame_num, key_frame_set,
        kf_accumulated, kf_images, img_dir, image_files,
    ):
        """Try to load a cached .npy result. Returns the array or None on failure.

        If the frame is a key frame, restores kf_accumulated and kf_images.
        """
        print(f"[INFO] Frame {frame_num} already processed. Loading from cache...")
        try:
            displacement_field = np.load(result_path)

            # If this frame is a key frame, restore accumulated state
            if frame_num in key_frame_set:
                kf_accumulated[frame_num] = displacement_field.copy()
                # Load the image for future reference if not already cached
                if frame_num not in kf_images:
                    img_path = os.path.join(img_dir, image_files[frame_num - 1])
                    kf_images[frame_num] = proc.load_and_convert_image(img_path)
                print(f"[INFO] Restored key frame {frame_num} state from cache")

            return displacement_field
        except Exception as e:
            print(
                f"[WARNING] Failed to load existing result for frame "
                f"{frame_num}: {e}. Reprocessing."
            )
            return None

    @staticmethod
    def _resolve_mask(
        frame_num, list_idx, ref_num,
        roi_mask, original_mask_crop,
        kf_accumulated, user_masks,
        xmin, ymin, xmax, ymax,
    ):
        """Determine the mask for the current frame.

        Fallback chain:
        1. Per-frame user mask (from discover_masks, keyed by 0-based index)
        2. Auto-warped mask from the key frame's accumulated displacement
        3. Original ROI mask
        """
        # Check for user-provided per-frame mask (0-indexed)
        if list_idx in user_masks:
            print(f"[INFO] Frame {frame_num}: using user-provided mask")
            return user_masks[list_idx]

        # Auto-warp from key frame if accumulated displacement is non-trivial
        kf_acc = kf_accumulated.get(ref_num)
        if kf_acc is not None:
            has_displacement = not (
                np.allclose(kf_acc, 0, atol=1e-12, equal_nan=False)
                and not np.any(np.isnan(kf_acc))
            )
            if has_displacement:
                warped_crop = warp_mask_with_holes(original_mask_crop, kf_acc)
                current_mask = roi_mask.copy()
                current_mask[ymin:ymax, xmin:xmax] = warped_crop
                return current_mask

        # Default: original mask
        return roi_mask

    def _update_progress_throttled(self, pair_idx, total_pairs, last_update_time):
        """Update progress callback, rate-limited to ~10Hz. Returns new last_update_time."""
        current_time = time.time()
        if self.update_progress and (
            current_time - last_update_time > 0.1 or pair_idx == total_pairs
        ):
            percent = (pair_idx / max(1, total_pairs)) * 100.0
            self.update_progress(percent, pair_idx, total_pairs)
            return current_time
        return last_update_time

    @staticmethod
    def _evict_unused_ref_images(frame_num, ref_map, key_frame_set, kf_images, total_frames):
        """Remove cached reference images that are no longer needed by future frames.

        A reference image is needed only while there are future frames that
        reference it. Once we've processed past the last such frame, the image
        can be evicted to free memory.
        """
        # Check each cached image (except frame 1 which is cheap to keep)
        to_evict = []
        for cached_kf in list(kf_images.keys()):
            if cached_kf == 1:
                continue
            # Find the last frame that references this key frame
            still_needed = False
            for future_frame in range(frame_num + 1, total_frames + 1):
                if future_frame in ref_map and ref_map[future_frame] == cached_kf:
                    still_needed = True
                    break
            if not still_needed:
                to_evict.append(cached_kf)

        for kf in to_evict:
            del kf_images[kf]
            print(f"[INFO] Evicted reference image for key frame {kf} (no longer needed)")
