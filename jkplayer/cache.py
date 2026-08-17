"""
RAM frame cache - LRU with a BYTE budget.

Why bytes and not an item count: 1080p RGBA half = 16.6 MB/frame,
4K = 66 MB/frame. An item count would say nothing and the user would wonder
where the memory went.

Thread-safe (worker threads fill it, the GUI thread reads it).
Key = file path -> the cache survives a range change and a re-wiring of the
node.
"""

import threading
from collections import OrderedDict


class FrameCache(object):

    def __init__(self, budget_mb=4096):
        self._lock = threading.RLock()
        self._items = OrderedDict()          # key -> ndarray
        self._bytes = 0
        self._budget = max(64, int(budget_mb)) * 1024 * 1024
        # statistics (read-only from the outside)
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.stores = 0

    # ---- basic operations ----
    def get(self, key):
        """Returns the frame and refreshes it in the LRU, or None."""
        if key is None:
            return None
        with self._lock:
            arr = self._items.get(key)
            if arr is None:
                self.misses += 1
                return None
            self._items.move_to_end(key)
            self.hits += 1
            return arr

    def peek(self, key):
        """Like get(), but without touching the LRU or the stats (for the bar)."""
        if key is None:
            return None
        with self._lock:
            return self._items.get(key)

    def contains(self, key):
        if key is None:
            return False
        with self._lock:
            return key in self._items

    def put(self, key, arr):
        if key is None or arr is None:
            return
        nbytes = int(arr.nbytes)
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return
            self._items[key] = arr
            self._bytes += nbytes
            self.stores += 1
            self._evict_locked()

    def _evict_locked(self):
        while self._bytes > self._budget and len(self._items) > 1:
            _k, old = self._items.popitem(last=False)     # oldest
            self._bytes -= int(old.nbytes)
            self.evictions += 1

    # ---- management ----
    def clear(self):
        with self._lock:
            self._items.clear()
            self._bytes = 0

    def set_budget_mb(self, mb):
        with self._lock:
            self._budget = max(64, int(mb)) * 1024 * 1024
            self._evict_locked()

    def keys(self):
        with self._lock:
            return set(self._items.keys())

    # ---- information ----
    @property
    def bytes_used(self):
        with self._lock:
            return self._bytes

    @property
    def budget_bytes(self):
        return self._budget

    @property
    def mb_used(self):
        return self.bytes_used / (1024.0 * 1024.0)

    @property
    def budget_mb(self):
        return self._budget / (1024.0 * 1024.0)

    @property
    def fill_ratio(self):
        return self.bytes_used / float(self._budget) if self._budget else 0.0

    def frame_capacity(self, frame_nbytes):
        """How many frames of that size fit into the budget."""
        return int(self._budget // frame_nbytes) if frame_nbytes else 0

    def cached_frames(self, seq, key_fn=None):
        """Sorted list of the sequence's cached frames (for the cache bar).

        `key_fn` (frame -> key) is needed when the key also carries the layer -
        see FrameLoader.key_for. Without it the bare path is used, as before.
        """
        if seq is None:
            return []
        keys = self.keys()
        if key_fn is None:
            key_fn = seq.key_for
        return [f for f in seq.frames() if key_fn(f) in keys]

    def cached_runs(self, seq, key_fn=None):
        """Continuous runs [(from,to), ...] - cheaper to draw than single frames."""
        runs = []
        for f in self.cached_frames(seq, key_fn):
            if runs and f == runs[-1][1] + 1:
                runs[-1][1] = f
            else:
                runs.append([f, f])
        return runs

    def stats(self):
        with self._lock:
            total = self.hits + self.misses
            return {"mb_used": round(self._bytes / 1048576.0, 1),
                    "mb_budget": round(self._budget / 1048576.0, 1),
                    "frames": len(self._items),
                    "hits": self.hits, "misses": self.misses,
                    "hit_rate": round(self.hits / total, 3) if total else 0.0,
                    "evictions": self.evictions}
