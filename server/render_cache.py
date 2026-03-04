"""Simple LRU cache for rendered PNG bytes, shared across route modules."""

from collections import OrderedDict


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
