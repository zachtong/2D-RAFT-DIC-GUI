"""Tests for per-frame mask loading."""
import numpy as np
import pytest
from PIL import Image
from pathlib import Path


def _create_mask_image(path, h, w, value=255):
    """Create a single-channel mask PNG."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[10:h-10, 10:w-10] = value
    Image.fromarray(mask).save(path)


def _create_rgb_mask(path, h, w):
    """Create an RGB mask image."""
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    mask[10:h-10, 10:w-10, :] = 255
    Image.fromarray(mask).save(path)


class TestMaskDiscovery:
    def test_match_by_filename(self, tmp_path):
        from raft_dic_gui.mask_loader import discover_masks
        _create_mask_image(tmp_path / "frame_001.png", 64, 64)
        _create_mask_image(tmp_path / "frame_003.png", 64, 64)
        image_files = ["frame_001.tif", "frame_002.tif", "frame_003.tif"]
        result = discover_masks(str(tmp_path), image_files, (64, 64))
        assert 0 in result
        assert 1 not in result
        assert 2 in result
        assert result[0].dtype == bool

    def test_match_by_number(self, tmp_path):
        from raft_dic_gui.mask_loader import discover_masks
        _create_mask_image(tmp_path / "mask_003.png", 64, 64)
        image_files = ["img_001.tif", "img_002.tif", "img_003.tif"]
        result = discover_masks(str(tmp_path), image_files, (64, 64))
        assert 2 in result  # mask_003 -> frame 3 -> index 2

    def test_filename_match_priority(self, tmp_path):
        """If a mask matches by filename, don't also try number matching."""
        from raft_dic_gui.mask_loader import discover_masks
        _create_mask_image(tmp_path / "frame_001.png", 64, 64)
        image_files = ["frame_001.tif", "frame_002.tif"]
        result = discover_masks(str(tmp_path), image_files, (64, 64))
        assert 0 in result
        assert len(result) == 1

    def test_wrong_size_skipped(self, tmp_path):
        from raft_dic_gui.mask_loader import discover_masks
        _create_mask_image(tmp_path / "frame_001.png", 32, 32)
        result = discover_masks(str(tmp_path), ["frame_001.tif"], (64, 64))
        assert len(result) == 0

    def test_extra_masks_ignored(self, tmp_path):
        from raft_dic_gui.mask_loader import discover_masks
        _create_mask_image(tmp_path / "frame_001.png", 64, 64)
        _create_mask_image(tmp_path / "frame_999.png", 64, 64)
        result = discover_masks(str(tmp_path), ["frame_001.tif"], (64, 64))
        assert len(result) == 1

    def test_empty_folder(self, tmp_path):
        from raft_dic_gui.mask_loader import discover_masks
        result = discover_masks(str(tmp_path), ["img.tif"], (64, 64))
        assert len(result) == 0

    def test_rgb_mask_handled(self, tmp_path):
        from raft_dic_gui.mask_loader import discover_masks
        _create_rgb_mask(tmp_path / "frame_001.png", 64, 64)
        result = discover_masks(str(tmp_path), ["frame_001.tif"], (64, 64))
        assert 0 in result
        assert result[0].dtype == bool

    def test_nonexistent_dir(self):
        from raft_dic_gui.mask_loader import discover_masks
        result = discover_masks("/nonexistent/path", ["img.tif"], (64, 64))
        assert len(result) == 0

    def test_number_out_of_range(self, tmp_path):
        from raft_dic_gui.mask_loader import discover_masks
        _create_mask_image(tmp_path / "mask_999.png", 64, 64)
        result = discover_masks(str(tmp_path), ["img_001.tif", "img_002.tif"], (64, 64))
        assert len(result) == 0

    def test_mask_values_are_boolean(self, tmp_path):
        from raft_dic_gui.mask_loader import discover_masks
        _create_mask_image(tmp_path / "frame_001.png", 64, 64, value=128)
        result = discover_masks(str(tmp_path), ["frame_001.tif"], (64, 64))
        assert result[0].dtype == bool
        assert np.any(result[0])  # Should have True values
