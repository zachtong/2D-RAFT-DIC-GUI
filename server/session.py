"""Server-side session state — replaces RAFTDICGUI class for state management."""

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from raft_dic_gui.config import DICConfig
from raft_dic_gui.controller import DICProcessor
from raft_dic_gui.probe_manager import ProbeManager
from raft_dic_gui.deformed_view_cache import DeformedViewCache
from server.deformed_warp import InverseMapCache


def _make_set_event() -> threading.Event:
    """Create an Event that starts in the 'set' (unblocked) state."""
    e = threading.Event()
    e.set()
    return e


@dataclass
class AppSession:
    """Singleton holding all application state between requests."""

    # Configuration
    config: DICConfig = field(default_factory=DICConfig)

    # Image state
    image_dir: str = ""
    image_files: List[str] = field(default_factory=list)
    reference_image: Optional[np.ndarray] = None
    image_width: int = 0
    image_height: int = 0

    # ROI state
    roi_mask: Optional[np.ndarray] = None
    roi_rect: Optional[Tuple[int, int, int, int]] = None
    roi_confirmed: bool = False

    # Processing state
    processor: DICProcessor = field(default_factory=DICProcessor)
    processing_active: bool = False
    stop_requested: bool = False
    pause_requested: bool = False
    pause_event: threading.Event = field(default_factory=lambda: _make_set_event(), repr=False)
    displacement_results: List[Any] = field(default_factory=list)
    result_version: int = 0

    # Strain state
    strain_results: List[Any] = field(default_factory=list)
    strain_computing: bool = False
    strain_components: List[str] = field(default_factory=list)

    # Probes
    probe_manager: ProbeManager = field(default_factory=ProbeManager)

    # Cache
    deformed_view_cache: DeformedViewCache = field(
        default_factory=DeformedViewCache
    )
    inverse_map_cache: InverseMapCache = field(
        default_factory=InverseMapCache,
        repr=False,
    )

    # Visualization settings (used by render endpoints)
    vis_settings: Dict[str, Any] = field(default_factory=lambda: {
        "colormap": "turbo",
        "alpha": 0.7,
        "background": "reference",
        "log_scale": False,
        "physical_ratio": 1.0,
        "physical_unit": "px",
        "dt": 1.0,
        "fps": 1.0,
    })

    # Export state
    export_active: bool = False
    export_progress: int = 0
    export_total: int = 0
    export_cancel: threading.Event = field(default_factory=threading.Event)

    # Image frame cache (LRU, up to 50 frames)
    _image_cache: OrderedDict = field(default_factory=OrderedDict, repr=False)
    _image_cache_max: int = 50

    # Lock for thread-safe access
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def get_cached_image(self, idx: int):
        """Get cached image by index, or None."""
        if idx in self._image_cache:
            self._image_cache.move_to_end(idx)
            return self._image_cache[idx]
        return None

    def cache_image(self, idx: int, img):
        """Cache an image array."""
        self._image_cache[idx] = img
        self._image_cache.move_to_end(idx)
        while len(self._image_cache) > self._image_cache_max:
            self._image_cache.popitem(last=False)

    def reset(self):
        """Reset all state to defaults."""
        self.image_dir = ""
        self.image_files = []
        self.reference_image = None
        self.image_width = 0
        self.image_height = 0
        self.roi_mask = None
        self.roi_rect = None
        self.roi_confirmed = False
        self.processing_active = False
        self.stop_requested = False
        self.pause_requested = False
        self.pause_event.set()
        self.displacement_results = []
        self.strain_results = []
        self.strain_computing = False
        self.strain_components = []
        self.probe_manager = ProbeManager()
        self.deformed_view_cache = DeformedViewCache()
        self.inverse_map_cache.clear()
        self._image_cache.clear()
        self.result_version += 1
        self.export_active = False
        self.export_progress = 0
        self.export_total = 0
        self.export_cancel.clear()


# Module-level singleton
session = AppSession()
