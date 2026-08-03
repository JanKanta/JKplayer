"""
EXR sequence - translates "frame number -> file path".

No Qt and no Nuke, so it can be tested from the console.
Supported path notations (the same ones a Nuke Read node uses):
    /path/plate.%04d.exr      printf style
    /path/plate.####.exr      hash style
    /path/plate.exr           single frame (still)
"""

import os
import re

_RE_PRINTF = re.compile(r"%0?(\d*)d")
_RE_HASH = re.compile(r"#+")


class ExrSequence(object):
    """Immutable description of a sequence. Equality compares pattern and range."""

    def __init__(self, pattern, first, last):
        self.pattern = pattern.replace("\\", "/")
        self.first = int(first)
        self.last = int(last)
        if self.last < self.first:
            self.last = self.first
        self.is_still = not (_RE_PRINTF.search(self.pattern)
                             or _RE_HASH.search(self.pattern))
        # paths are asked for over and over at runtime (display, look-ahead,
        # cache bar) and regex substitution is not free -> remember them
        self._paths = {}

    # ---- frame mapping ----
    def clamp(self, frame):
        return max(self.first, min(self.last, int(frame)))

    def path_for(self, frame):
        """Path to the file of that frame (the frame is clamped to the range)."""
        f = self.clamp(frame)
        cached = self._paths.get(f)
        if cached is not None:
            return cached
        p = self.pattern
        if _RE_PRINTF.search(p):
            out = _RE_PRINTF.sub(lambda m: ("%0*d" % (int(m.group(1) or 0), f)), p)
        elif _RE_HASH.search(p):
            out = _RE_HASH.sub(lambda m: str(f).zfill(len(m.group(0))), p)
        else:
            out = p
        self._paths[f] = out
        return out

    def key_for(self, frame):
        """Cache key. It is the file path => the cache survives a range change
        and a re-wiring of the node, as long as it points at the same files."""
        return self.path_for(frame)

    @property
    def frame_count(self):
        return self.last - self.first + 1

    def frames(self):
        return range(self.first, self.last + 1)

    # ---- helpers ----
    def exists(self, frame):
        return os.path.isfile(self.path_for(frame))

    def missing_frames(self, limit=None):
        out = []
        for f in self.frames():
            if not self.exists(f):
                out.append(f)
                if limit and len(out) >= limit:
                    break
        return out

    def label(self):
        return os.path.basename(self.pattern)

    def __eq__(self, other):
        return (isinstance(other, ExrSequence)
                and other.pattern == self.pattern
                and other.first == self.first
                and other.last == self.last)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.pattern, self.first, self.last))

    def __repr__(self):
        return "ExrSequence(%r, %d-%d)" % (self.pattern, self.first, self.last)


def is_exr(path):
    return bool(path) and os.path.splitext(path)[1].lower() == ".exr"


def from_read_node(node):
    """ExrSequence from a Nuke Read node, or None when it is not an EXR Read.

    Deliberately strict: we work ONLY with Read nodes and ONLY with EXR
    (that was the scope given for v2).
    """
    if node is None:
        return None
    try:
        if node.Class() != "Read":
            return None
        knobs = node.knobs()
        if "file" not in knobs:
            return None
        pattern = node["file"].value()
        if not is_exr(pattern):
            return None
        return ExrSequence(pattern, int(node.firstFrame()), int(node.lastFrame()))
    except Exception:
        return None
