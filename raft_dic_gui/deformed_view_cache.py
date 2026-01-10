"""
Deformed View Cache Manager

Manages coordinate mappings and image cache for deformed frame visualization.
This module provides caching to avoid redundant computation when switching
between reference and deformed view modes.

Key Features:
- Coordinate mapping cache (reference -> deformed)
- Warped ROI mask cache
- LRU image cache for loaded deformed frames
- Thread-safe invalidation
"""

import numpy as np
import cv2
from collections import OrderedDict
from typing import List, Dict, Optional, Tuple

# Debug toggle - set to False to disable verbose output
DEBUG_DEFORMED_VIEW = True

def _debug(msg: str):
    """Print debug message with standard prefix."""
    if DEBUG_DEFORMED_VIEW:
        print(f"[DeformedView] {msg}")


class DeformedViewCache:
    """
    Manages coordinate mappings and image cache for deformed frame visualization.
    
    This class should be instantiated once in main_GUI and shared across
    both Displacement and Post-processing tabs.
    
    Usage:
        cache = DeformedViewCache(max_images=5)
        cache.set_image_paths(image_file_paths)
        
        # When displaying frame N in deformed mode:
        coords = cache.get_deformed_coords(frame_idx, U, V)
        image = cache.get_deformed_image(frame_idx)
        mask = cache.get_warped_mask(frame_idx, original_mask, U, V)
    """
    
    def __init__(self, max_images: int = 5, max_post_components: int = 3):
        """
        Initialize the deformed view cache.
        
        Args:
            max_images: Maximum number of deformed frame images to keep in LRU cache.
            max_post_components: Maximum number of post-processing components to cache.
        """
        _debug(f"__init__: Creating cache with max_images={max_images}")
        
        # Coordinate mapping cache: frame_idx -> {'x_def': ndarray, 'y_def': ndarray}
        self.coord_cache: Dict[int, Dict[str, np.ndarray]] = {}
        
        # Warped mask cache: frame_idx -> warped_mask
        self.mask_cache: Dict[int, np.ndarray] = {}
        
        # Warped displacement cache: frame_idx -> {'u_warped': ndarray, 'v_warped': ndarray}
        self.warped_disp_cache: Dict[int, Dict[str, np.ndarray]] = {}
        
        # LRU cache for loaded deformed frame images
        self.image_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.max_images = max_images
        
        # LRU cache for post-processing components: (frame_idx, component_name) -> warped_data
        # This caches warped strain/displacement data for post-processing visualization
        self.post_component_cache: OrderedDict[Tuple[int, str], np.ndarray] = OrderedDict()
        self.max_post_components = max_post_components
        
        # Image file paths - set by main_GUI when images are loaded
        self._image_paths: List[str] = []
        
        # Reference to image loading function (to avoid circular imports)
        self._load_image_func = None
    
    def set_image_paths(self, paths: List[str]):
        """
        Set image file paths for loading deformed frames.
        
        Args:
            paths: List of absolute paths to image files.
        """
        _debug(f"set_image_paths: Setting {len(paths)} image paths")
        self._image_paths = paths
        # Clear image cache when paths change
        self.image_cache.clear()
    
    def set_load_function(self, load_func):
        """
        Set the image loading function.
        
        Args:
            load_func: Function that takes a path and returns RGB image array.
        """
        self._load_image_func = load_func
        _debug("set_load_function: Load function configured")
    
    def invalidate_all(self, reason: str = "unknown"):
        """
        Clear all caches.
        
        Should be called when:
        - New images are loaded
        - Processing is re-run
        - ROI is changed
        
        Args:
            reason: Reason for invalidation (for debugging).
        """
        _debug(f"invalidate_all: Clearing all caches, reason='{reason}'")
        _debug(f"  - coord_cache had {len(self.coord_cache)} entries")
        _debug(f"  - mask_cache had {len(self.mask_cache)} entries")
        _debug(f"  - warped_disp_cache had {len(self.warped_disp_cache)} entries")
        _debug(f"  - post_component_cache had {len(self.post_component_cache)} entries")
        _debug(f"  - image_cache had {len(self.image_cache)} entries")
        
        self.coord_cache.clear()
        self.mask_cache.clear()
        self.warped_disp_cache.clear()
        self.post_component_cache.clear()
        self.image_cache.clear()
    
    def invalidate_coords_and_masks(self, reason: str = "unknown"):
        """
        Clear coordinate and mask caches, but keep image cache.
        
        Should be called when displacement results change but images are the same.
        
        Args:
            reason: Reason for invalidation (for debugging).
        """
        _debug(f"invalidate_coords_and_masks: Clearing coord/mask/disp/post caches, reason='{reason}'")
        _debug(f"  - coord_cache had {len(self.coord_cache)} entries")
        _debug(f"  - mask_cache had {len(self.mask_cache)} entries")
        _debug(f"  - warped_disp_cache had {len(self.warped_disp_cache)} entries")
        _debug(f"  - post_component_cache had {len(self.post_component_cache)} entries")
        
        self.coord_cache.clear()
        self.mask_cache.clear()
        self.warped_disp_cache.clear()
        self.post_component_cache.clear()
    
    def get_deformed_coords(self, frame_idx: int, 
                            U: np.ndarray, V: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get or compute deformed coordinate mapping for a frame.
        
        The mapping defines where each reference pixel ends up in deformed space.
        
        Args:
            frame_idx: Frame index (0-based for displacement results).
            U: Horizontal displacement field (H, W).
            V: Vertical displacement field (H, W).
            
        Returns:
            Dictionary with 'x_def' and 'y_def' coordinate arrays.
        """
        cache_hit = frame_idx in self.coord_cache
        
        if cache_hit:
            _debug(f"get_deformed_coords: frame={frame_idx}, CACHE HIT")
            return self.coord_cache[frame_idx]
        
        # Compute deformed coordinates
        _debug(f"get_deformed_coords: frame={frame_idx}, CACHE MISS, computing...")
        _debug(f"  - U shape={U.shape}, NaN count={np.sum(np.isnan(U))}")
        _debug(f"  - V shape={V.shape}, NaN count={np.sum(np.isnan(V))}")
        
        coords = self._compute_deformed_coords(U, V)
        
        # Cache the result
        self.coord_cache[frame_idx] = coords
        
        _debug(f"  - x_def range=[{np.nanmin(coords['x_def']):.1f}, {np.nanmax(coords['x_def']):.1f}]")
        _debug(f"  - y_def range=[{np.nanmin(coords['y_def']):.1f}, {np.nanmax(coords['y_def']):.1f}]")
        
        return coords
    
    def _compute_deformed_coords(self, U: np.ndarray, V: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute the deformed coordinate mapping.
        
        For each reference coordinate (x, y), compute the deformed position (x', y').
        This is the forward mapping: x' = x + U(x,y), y' = y + V(x,y)
        
        Args:
            U: Horizontal displacement (H, W).
            V: Vertical displacement (H, W).
            
        Returns:
            {'x_def': array, 'y_def': array} - deformed coordinates for each pixel.
        """
        H, W = U.shape
        
        # Create reference coordinate grids
        x_ref, y_ref = np.meshgrid(np.arange(W, dtype=np.float64), 
                                    np.arange(H, dtype=np.float64))
        
        # Compute deformed coordinates
        x_def = x_ref + U.astype(np.float64)
        y_def = y_ref + V.astype(np.float64)
        
        # Convert to float32 for cv2.remap compatibility
        return {
            'x_def': x_def.astype(np.float32),
            'y_def': y_def.astype(np.float32)
        }
    
    def get_warped_mask(self, frame_idx: int, 
                        original_mask: np.ndarray,
                        U: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Get or compute warped ROI mask for a frame.
        
        The warped mask shows where the original ROI ends up in deformed space.
        
        Args:
            frame_idx: Frame index.
            original_mask: Original ROI mask (H, W) boolean or uint8.
            U: Horizontal displacement field.
            V: Vertical displacement field.
            
        Returns:
            Warped mask in deformed coordinates.
        """
        cache_hit = frame_idx in self.mask_cache
        
        if cache_hit:
            _debug(f"get_warped_mask: frame={frame_idx}, CACHE HIT")
            return self.mask_cache[frame_idx]
        
        _debug(f"get_warped_mask: frame={frame_idx}, CACHE MISS, computing...")
        valid_before = np.sum(original_mask > 0)
        
        # Compute warped mask
        warped_mask = self._warp_mask_forward(original_mask, U, V)
        
        valid_after = np.sum(warped_mask > 0)
        _debug(f"  - valid pixels: before={valid_before}, after={valid_after}")
        
        # Cache the result
        self.mask_cache[frame_idx] = warped_mask
        
        return warped_mask
    
    def _warp_mask_forward(self, mask: np.ndarray, 
                           U: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Warp mask using forward mapping (reference -> deformed).
        
        Uses scatter approach: for each valid reference pixel, mark
        its deformed position as valid.
        
        Args:
            mask: Original ROI mask (H, W).
            U: Horizontal displacement.
            V: Vertical displacement.
            
        Returns:
            Warped mask in deformed space.
        """
        H, W = mask.shape
        
        # Get deformed coordinates
        x_ref, y_ref = np.meshgrid(np.arange(W), np.arange(H))
        x_def = x_ref + U
        y_def = y_ref + V
        
        # Create output mask
        warped_mask = np.zeros((H, W), dtype=np.uint8)
        
        # Get valid reference pixels
        valid = (mask > 0) & (~np.isnan(U)) & (~np.isnan(V))
        
        # Get their deformed positions
        x_def_valid = x_def[valid].astype(np.int32)
        y_def_valid = y_def[valid].astype(np.int32)
        
        # Filter to within bounds
        in_bounds = (x_def_valid >= 0) & (x_def_valid < W) & \
                    (y_def_valid >= 0) & (y_def_valid < H)
        
        x_def_valid = x_def_valid[in_bounds]
        y_def_valid = y_def_valid[in_bounds]
        
        # Mark deformed positions
        warped_mask[y_def_valid, x_def_valid] = 1
        
        # Morphological closing to fill small gaps
        kernel = np.ones((3, 3), np.uint8)
        warped_mask = cv2.morphologyEx(warped_mask, cv2.MORPH_CLOSE, kernel)
        
        return warped_mask > 0
    
    def warp_data_forward(self, data: np.ndarray, U: np.ndarray, V: np.ndarray, 
                          roi_rect: tuple, output_shape: tuple) -> np.ndarray:
        """
        Warp data from reference to deformed coordinates using forward scatter.
        
        This is used for visualizing any data field (displacement, strain, etc.)
        in deformed coordinates.
        
        Args:
            data: Data to warp (shape matches displacement within ROI).
            U: Horizontal displacement field (ROI-cropped).
            V: Vertical displacement field (ROI-cropped).
            roi_rect: (xmin, ymin, xmax, ymax) of ROI in full image.
            output_shape: (H, W) of output full image.
            
        Returns:
            Warped data in full image coordinates (NaN where no data).
        """
        xmin, ymin, xmax, ymax = roi_rect
        h, w = output_shape
        
        # Create coordinate grids for ROI
        y_ref, x_ref = np.mgrid[ymin:ymax, xmin:xmax]
        
        # Compute deformed coordinates
        x_def = (x_ref + U).astype(np.float32)
        y_def = (y_ref + V).astype(np.float32)
        
        # Create output arrays
        warped_data = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)
        
        # Valid mask
        valid_mask = ~np.isnan(U) & ~np.isnan(V) & ~np.isnan(data)
        
        # Get valid positions
        x_def_valid = x_def[valid_mask]
        y_def_valid = y_def[valid_mask]
        data_valid = data[valid_mask].astype(np.float32)
        
        # Round to nearest integer coordinates
        x_idx = np.round(x_def_valid).astype(np.int32)
        y_idx = np.round(y_def_valid).astype(np.int32)
        
        # Filter to within bounds
        in_bounds = (x_idx >= 0) & (x_idx < w) & (y_idx >= 0) & (y_idx < h)
        x_idx = x_idx[in_bounds]
        y_idx = y_idx[in_bounds]
        data_valid = data_valid[in_bounds]
        
        # Forward scatter with accumulation
        np.add.at(warped_data, (y_idx, x_idx), data_valid)
        np.add.at(count_map, (y_idx, x_idx), 1.0)
        
        # Average where multiple values accumulated
        nonzero = count_map > 0
        warped_data[nonzero] /= count_map[nonzero]
        warped_data[~nonzero] = np.nan
        
        return warped_data
    
    def get_deformed_image(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Load deformed frame image with LRU caching.
        
        Args:
            frame_idx: Frame index (1-based, where frame 1 is reference).
                      For displacement result index i, frame_idx = i + 1.
            
        Returns:
            RGB image array, or None if loading fails.
        """
        # Check cache
        if frame_idx in self.image_cache:
            _debug(f"get_deformed_image: frame={frame_idx}, CACHE HIT")
            # Move to end (most recently used)
            self.image_cache.move_to_end(frame_idx)
            return self.image_cache[frame_idx]
        
        _debug(f"get_deformed_image: frame={frame_idx}, CACHE MISS, loading...")
        
        # Validate
        if not self._image_paths:
            _debug("  - ERROR: No image paths set!")
            return None
        
        if frame_idx < 0 or frame_idx >= len(self._image_paths):
            _debug(f"  - ERROR: frame_idx {frame_idx} out of range [0, {len(self._image_paths)})")
            return None
        
        if self._load_image_func is None:
            _debug("  - ERROR: Load function not set!")
            return None
        
        # Load image
        try:
            image_path = self._image_paths[frame_idx]
            _debug(f"  - Loading from: {image_path}")
            image = self._load_image_func(image_path)
            _debug(f"  - Loaded image shape: {image.shape}")
        except Exception as e:
            _debug(f"  - ERROR loading image: {e}")
            return None
        
        # Evict oldest if at capacity
        if len(self.image_cache) >= self.max_images:
            evicted_key, _ = self.image_cache.popitem(last=False)
            _debug(f"  - LRU eviction: removed frame {evicted_key}")
        
        # Add to cache
        self.image_cache[frame_idx] = image
        _debug(f"  - Cache size now: {len(self.image_cache)}/{self.max_images}")
        
        return image
    
    def has_cached_coords(self, frame_idx: int) -> bool:
        """Check if coordinates are cached for a frame."""
        return frame_idx in self.coord_cache
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics for debugging."""
        return {
            'coord_cache_size': len(self.coord_cache),
            'mask_cache_size': len(self.mask_cache),
            'warped_disp_cache_size': len(self.warped_disp_cache),
            'image_cache_size': len(self.image_cache),
            'image_cache_max': self.max_images,
            'num_image_paths': len(self._image_paths)
        }
    
    def get_warped_displacement(self, frame_idx: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Get cached warped displacement for a frame.
        
        Args:
            frame_idx: Frame index.
            
        Returns:
            {'u_warped': array, 'v_warped': array} or None if not cached.
        """
        if frame_idx in self.warped_disp_cache:
            _debug(f"get_warped_displacement: frame={frame_idx}, CACHE HIT")
            return self.warped_disp_cache[frame_idx]
        return None
    
    def set_warped_displacement(self, frame_idx: int, u_warped: np.ndarray, v_warped: np.ndarray):
        """
        Store warped displacement in cache.
        
        Args:
            frame_idx: Frame index.
            u_warped: Warped U displacement (full image size).
            v_warped: Warped V displacement (full image size).
        """
        _debug(f"set_warped_displacement: frame={frame_idx}, caching warped data")
        self.warped_disp_cache[frame_idx] = {
            'u_warped': u_warped,
            'v_warped': v_warped
        }
    
    def get_post_component(self, frame_idx: int, component_name: str) -> Optional[np.ndarray]:
        """
        Get cached warped post-processing component (LRU).
        
        Args:
            frame_idx: Frame index.
            component_name: Component name (e.g., 'exx', 'u', 'von_mises').
            
        Returns:
            Warped component data, or None if not cached.
        """
        key = (frame_idx, component_name)
        if key in self.post_component_cache:
            _debug(f"get_post_component: frame={frame_idx}, comp={component_name}, CACHE HIT")
            # Move to end (most recently used)
            self.post_component_cache.move_to_end(key)
            return self.post_component_cache[key]
        return None
    
    def set_post_component(self, frame_idx: int, component_name: str, warped_data: np.ndarray):
        """
        Store warped post-processing component in LRU cache.
        
        Args:
            frame_idx: Frame index.
            component_name: Component name.
            warped_data: Warped component data (full image size).
        """
        key = (frame_idx, component_name)
        
        # Evict oldest if at capacity
        while len(self.post_component_cache) >= self.max_post_components:
            evicted_key, _ = self.post_component_cache.popitem(last=False)
            _debug(f"set_post_component: LRU eviction: removed {evicted_key}")
        
        self.post_component_cache[key] = warped_data
        _debug(f"set_post_component: frame={frame_idx}, comp={component_name}, cached. Size: {len(self.post_component_cache)}/{self.max_post_components}")
