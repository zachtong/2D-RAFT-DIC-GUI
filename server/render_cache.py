"""Simple LRU cache for rendered PNG bytes, shared across route modules."""

from collections import OrderedDict


def auto_cache_size(target_mb: int = 500, avg_item_kb: int = 200) -> int:
    """Compute cache size based on available system memory.

    Uses at most target_mb or 10% of available RAM, whichever is smaller.
    Falls back to 512 if psutil is not available.
    """
    try:
        import psutil
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        budget_mb = min(target_mb, available_mb * 0.1)
        return max(64, int(budget_mb * 1024 / avg_item_kb))
    except (ImportError, Exception):
        return 512


class RenderCache:
    """Bounded LRU cache mapping (result_id, frame_idx, params) → PNG bytes."""

    def __init__(self, max_entries: int = 512):
        self._cache: OrderedDict[tuple, bytes] = OrderedDict()
        self._max = max_entries

    def get(self, key: tuple) -> bytes | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: tuple, data: bytes) -> None:
        self._cache[key] = data
        self._cache.move_to_end(key)
        while len(self._cache) > self._max:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)
