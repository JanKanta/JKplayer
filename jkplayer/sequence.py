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

    def __init__(self, pattern, first, last, offset=0):
        self.pattern = pattern.replace("\\", "/")
        # first/last are the range ON THE TIMELINE, i.e. already shifted; the
        # offset is kept so path_for can go back to the file. An offset of
        # +5 means the clip starts five frames later and frame 10 shows the
        # file of frame 5.
        self.offset = int(offset)
        self.first = int(first) + self.offset
        self.last = int(last) + self.offset
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
        """Path to the file of that TIMELINE frame (clamped to the range)."""
        key = self.clamp(frame)
        cached = self._paths.get(key)
        if cached is not None:
            return cached
        f = key - self.offset          # back to the frame the file is named by
        p = self.pattern
        if _RE_PRINTF.search(p):
            out = _RE_PRINTF.sub(lambda m: ("%0*d" % (int(m.group(1) or 0), f)), p)
        elif _RE_HASH.search(p):
            out = _RE_HASH.sub(lambda m: str(f).zfill(len(m.group(0))), p)
        else:
            out = p
        self._paths[key] = out
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
        name = os.path.basename(self.pattern)
        # The shift is SAID OUT LOUD. Otherwise the per-input timing is
        # invisible here and the frame numbers simply look wrong.
        if self.offset:
            name += "  (offset %+d)" % self.offset
        return name

    def __eq__(self, other):
        return (isinstance(other, ExrSequence)
                and other.pattern == self.pattern
                and other.first == self.first
                and other.last == self.last
                and other.offset == self.offset)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.pattern, self.first, self.last, self.offset))

    def __repr__(self):
        return "ExrSequence(%r, %d-%d, offset=%d)" % (
            self.pattern, self.first, self.last, self.offset)


def is_exr(path):
    return bool(path) and os.path.splitext(path)[1].lower() == ".exr"


# Nodes that hand the picture straight on. Walked THROUGH, so wiring with a
# dot - which everybody does - does not read as an empty input.
PASSTHROUGH = ("Dot", "NoOp")

# TimeOffset used to be walked through as well, with its shift read off the
# node. It is not any more: the player has its own per-input 'Start at' and
# 'Offset' (see node._add_knobs), which say the same thing in one place, are
# saved with the settings and are visible next to the range they produce. Two
# ways to shift the same clip meant they could disagree, and reading a shift
# out of a node whose knob name we had to guess was never solid.
MAX_WALK = 32                       # a loop in the graph must not hang us


def resolve_source(node):
    """The Read node behind an input, or None.

    Walks up through dots to the Read that actually holds the files. Anything
    else stops the walk: this player reads EXRs off disk rather than pulling
    the image through Nuke, so a node that CHANGES the picture cannot be
    honoured and pretending otherwise would show the wrong thing.
    """
    for _ in range(MAX_WALK):
        if node is None:
            return None
        try:
            cls = node.Class()
        except Exception:
            return None
        if cls == "Read":
            return node
        if cls not in PASSTHROUGH:
            return None
        try:
            node = node.input(0)
        except Exception:
            return None
    return None


def from_read_node(node, start_at=0, nudge=0):
    """ExrSequence for what is wired into an input, or None.

    Only EXR, and only a Read at the end of the walk (see resolve_source).

    `start_at` and `nudge` are the node's own per-input timing, and they are
    now the ONLY way a clip is moved in time:

      * `start_at` is ABSOLUTE placement - the timeline frame the first frame
        lands on. 0 leaves it where the files say it is.
      * `nudge` is added on top, always. That is the "it is one frame out"
        control, and it works whichever way the clip was placed.
    """
    read = resolve_source(node)
    if read is None:
        return None
    try:
        knobs = read.knobs()
        if "file" not in knobs:
            return None
        pattern = read["file"].value()
        if not is_exr(pattern):
            return None
        first, last = int(read.firstFrame()), int(read.lastFrame())
        place = (int(start_at) - first) if start_at else 0
        return ExrSequence(pattern, first, last, place + int(nudge))
    except Exception:
        return None
