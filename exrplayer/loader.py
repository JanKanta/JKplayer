"""
Background frame loading - the heart of the caching logic.

DESIGN (following RV / DJV):
  * Two queues:
      URGENT     - a small window around the playhead in the playback
                   direction (look-ahead). It is REBUILT on every move, so it
                   always matches what the user needs right now. Old requests
                   never hold up new ones.
      BACKGROUND - the "region cache" = the whole range, filled in when idle.
    Workers take from URGENT first, then from BACKGROUND.
  * Generation: on a sequence change the number is bumped and the results of
    old work are dropped (so they cannot pollute the cache with stale data).
  * None of this runs on the GUI thread and none of it calls Nuke.
  * Fill rate (frames/s) and decode time are measured - so we optimise from
    data, not from impressions.
"""

import heapq
import threading
import time
from collections import deque

from . import exrcore, exrread, reader


class FrameLoader(object):

    def __init__(self, cache, workers=8, on_ready=None):
        self.cache = cache
        self.on_ready = on_ready          # callback(frame) - CALLED FROM WORKERS!

        self._lock = threading.Condition(threading.Lock())
        self._urgent = []                 # heap (priority, order, frame)
        self._background = deque()        # frames for the region cache
        self._inflight = set()            # keys being processed right now
        self._counter = 0
        self._generation = 0
        self._seq = None
        self._layer = exrcore.ROOT_LAYER  # which layer (AOV) is being read
        self._stopping = False

        # statistics
        self._decoded = 0                 # total number of decoded frames
        self._decode_time = 0.0           # total decode time (s)
        self._recent = deque(maxlen=64)   # completion times for the fill rate
        self._failures = 0                # how many frames could not be read
        self.last_error = None            # why (so it does not stay silent)

        self._threads = []
        for i in range(max(1, int(workers))):
            t = threading.Thread(target=self._worker, name="exr-loader-%d" % i)
            t.daemon = True
            t.start()
            self._threads.append(t)

    # ------------------------------------------------------------------ API
    def key_for(self, frame, seq=None, layer=None):
        """Cache key for the given frame.

        It is the file path plus the layer. The layer has to be in there
        because of the double view: both inputs may point at THE SAME file and
        differ only by layer - with the bare path they would overwrite each
        other's data in the cache. The plain rgba key stays the bare path, so
        the cache is not invalidated just because a second input appeared.
        """
        seq = self._seq if seq is None else seq
        if seq is None:
            return None
        layer = self._layer if layer is None else layer
        path = seq.key_for(frame)
        return path if layer == exrcore.ROOT_LAYER else "%s\x00%s" % (path, layer)

    def key_fn(self):
        """A frame -> key function, frozen to the current sequence and layer.

        For the cache bar: the caller invokes it many times and should not
        have to take the lock.
        """
        with self._lock:
            seq, layer = self._seq, self._layer
        return lambda f: self.key_for(f, seq, layer)

    def set_sequence(self, seq):
        """Switches the source. Drops the queue and the results of old work."""
        with self._lock:
            if seq == self._seq:
                return
            self._seq = seq
            self._generation += 1
            self._urgent = []
            self._background.clear()
            self._lock.notify_all()

    def set_layer(self, layer):
        """Switches the layer (AOV). Returns True when it really changed.

        The cache is keyed by file path, not by layer - the caller therefore
        has to flush it after a change, otherwise frames from the previous
        layer would stay.
        """
        with self._lock:
            if layer == self._layer:
                return False
            self._layer = layer
            self._generation += 1
            self._urgent = []
            self._background.clear()
            self._lock.notify_all()
            return True

    @property
    def layer(self):
        with self._lock:
            return self._layer

    @property
    def sequence(self):
        with self._lock:
            return self._seq

    def request(self, frame, priority=0):
        """Queues one frame (ignored when already cached or in flight)."""
        with self._lock:
            self._enqueue_locked(frame, priority)
            self._lock.notify()

    def set_playhead(self, frame, direction=1, ahead=24, behind=4,
                     lo=None, hi=None):
        """Rebuilds the URGENT queue around the playhead. Call on every move.

        The current frame has the highest priority, then frames in the
        playback direction, and finally a few frames behind (scrubbing back
        and forth). `lo`/`hi` = mark IN/OUT: nothing outside them is
        pre-fetched (it would be wasted work and wasted memory).
        """
        with self._lock:
            if self._seq is None:
                return
            self._urgent = []
            step = 1 if direction >= 0 else -1

            def want(f):
                if lo is not None and f < lo:
                    return False
                if hi is not None and f > hi:
                    return False
                return True

            if want(frame):
                self._enqueue_locked(frame, 0)
            for i in range(1, int(ahead) + 1):
                f = frame + i * step
                if want(f):
                    self._enqueue_locked(f, i)
            for i in range(1, int(behind) + 1):
                f = frame - i * step
                if want(f):
                    self._enqueue_locked(f, ahead + i)
            self._lock.notify_all()

    def cache_range(self, first=None, last=None, anchor=None, direction=1,
                    limit=None):
        """Background region cache - a ROLLING WINDOW forward from the playhead.

        Why from the playhead and not from the start of the range: on a long
        shot the whole range does not fit in memory. Filling from the start
        would evict (through the LRU) exactly the frames the playhead is
        heading for - the cache would thrash and playback would stutter. This
        way what is needed next gets filled and old frames behind fall out on
        their own (LRU).

        anchor    - where to start (typically the current frame); None = range start
        direction - fill direction (following the playback direction)
        limit     - how many frames to schedule at most (typically cache capacity)
        """
        with self._lock:
            seq = self._seq
            if seq is None:
                return
            a = seq.first if first is None else int(first)
            b = seq.last if last is None else int(last)
            if b < a:
                a, b = b, a
            span = b - a + 1
            start = a if anchor is None else max(a, min(b, int(anchor)))
            step = 1 if direction >= 0 else -1
            count = span if limit is None else max(1, min(span, int(limit)))

            self._background.clear()
            for i in range(count):
                f = a + ((start - a) + i * step) % span      # wrap in the range
                if not self.cache.contains(self.key_for(f, seq)):
                    self._background.append(f)
            self._lock.notify_all()

    def cancel_background(self):
        with self._lock:
            self._background.clear()

    def cancel_all(self):
        with self._lock:
            self._urgent = []
            self._background.clear()

    def stop(self):
        with self._lock:
            self._stopping = True
            self._urgent = []
            self._background.clear()
            self._lock.notify_all()

    # ------------------------------------------------------------- internals
    def _enqueue_locked(self, frame, priority):
        seq = self._seq
        if seq is None:
            return
        f = seq.clamp(frame)
        key = self.key_for(f, seq)
        if key in self._inflight or self.cache.contains(key):
            return
        self._counter += 1
        heapq.heappush(self._urgent, (int(priority), self._counter, f))

    def _next_job_locked(self):
        """(frame, generation, key, path) or None. Urgent comes first."""
        seq = self._seq
        if seq is None:
            return None
        for queue, pop in ((self._urgent, self._pop_urgent_locked),
                           (self._background, self._pop_background_locked)):
            while queue:
                f = pop()
                key = self.key_for(f, seq)
                if key in self._inflight or self.cache.contains(key):
                    continue
                self._inflight.add(key)
                return f, self._generation, key, seq.path_for(f)
        return None

    def _pop_urgent_locked(self):
        _prio, _cnt, f = heapq.heappop(self._urgent)
        return f

    def _pop_background_locked(self):
        return self._background.popleft()

    def _worker(self):
        while True:
            with self._lock:
                while True:
                    if self._stopping:
                        return
                    job = self._next_job_locked()
                    if job is not None:
                        break
                    self._lock.wait(0.25)
                frame, generation, key, path = job
                layer = self._layer

            arr = None
            reason = None
            t0 = time.monotonic()
            try:
                arr = reader.read_frame(path, layer)
            except exrread.ExrUnsupported as exc:
                reason = "we cannot read this EXR: %s" % exc
            except (OSError, IOError):
                reason = "cannot open the file: %s" % path
            except Exception as exc:
                reason = "%s: %s" % (type(exc).__name__, exc)
            dt = time.monotonic() - t0
            if reason is not None:
                with self._lock:
                    self._failures += 1
                    self.last_error = reason

            with self._lock:
                self._inflight.discard(key)
                stale = (generation != self._generation)
                if arr is not None and not stale:
                    self._decoded += 1
                    self._decode_time += dt
                    self._recent.append(time.monotonic())
                self._lock.notify()

            if arr is not None and not stale:
                self.cache.put(key, arr)
                if self.on_ready is not None:
                    try:
                        self.on_ready(frame)
                    except Exception:
                        pass

    # ------------------------------------------------------------ statistics
    @property
    def pending(self):
        with self._lock:
            return len(self._urgent) + len(self._background) + len(self._inflight)

    @property
    def fill_fps(self):
        """Frames/s stored in the last ~2 s (the real cache fill rate)."""
        now = time.monotonic()
        with self._lock:
            recent = [t for t in self._recent if now - t <= 2.0]
        if len(recent) < 2:
            return 0.0
        span = recent[-1] - recent[0]
        return (len(recent) - 1) / span if span > 0 else 0.0

    @property
    def avg_decode_ms(self):
        with self._lock:
            if not self._decoded:
                return 0.0
            return self._decode_time / self._decoded * 1000.0

    def stats(self):
        return {"pending": self.pending,
                "fill_fps": round(self.fill_fps, 1),
                "avg_decode_ms": round(self.avg_decode_ms, 1),
                "decoded": self._decoded,
                "failures": self._failures,
                "last_error": self.last_error,
                "workers": len(self._threads)}
