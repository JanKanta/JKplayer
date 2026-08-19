"""
Reading movies through ffmpeg, for review deliverables.

LONG-GOP IS ALLOWED, but it is not free and the panel is told which it has.
This started as an all-intra whitelist on the argument that a long-GOP seek
costs decoding back to the last keyframe. True, but the size of it was never
measured, and measuring changed the answer: on 4K, a cold jump costs 341-418
ms in ProRes and 312-512 ms in h264 with a 25-frame GOP - the same. Only a
250-frame GOP shows a difference, and at its worst (204 frames past the
keyframe) it is 757 ms, about twice. Most of that baseline is starting ffmpeg
at all, which every codec pays.

So the split is real but small, and refusing an mp4 outright was the bigger
harm. probe() reports `all_intra` and what a jump costs, and the near-frame
case is handled by reading forward instead of restarting - see SKIP_AHEAD.

ONE PROCESS PER FILE, kept open and read sequentially, because that is how a
movie is built and how it is played. Frames asked for IN ORDER cost nothing
extra; a jump restarts ffmpeg at the new point. The loader's threads all go
through one lock per file - no loss, since ffmpeg threads inside itself and
already decodes 4K ProRes far faster than the disk feeds it.

FRAME NUMBERS TRAVEL IN THE PATH. Everything upstream - the cache, the loader,
the look-ahead - addresses a frame by its path, which works because an image
sequence is one file per frame. A movie is not, so a reference here looks like
"/path/clip.mov|1042" (see SEP). It is a small lie in a string, and it buys
not touching the cache key, the loader queue or the sequence cache at all.
"""

import atexit
import os
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

SEP = "|"                   # illegal in a Windows path, so it cannot collide

EXTENSIONS = (".mov", ".mp4", ".mxf", ".m4v")

# READING FORWARD INSTEAD OF STARTING AGAIN. A jump a few frames ahead is
# served by pulling frames off the pipe and dropping them, which beats
# restarting ffmpeg until the dropped frames cost more than the restart.
#
# Measured, restart against reading forward, at the step where they cross:
#     4K ProRes    410 ms restart, crosses at 12
#     4K long-GOP  520 ms restart, crosses at 16
#     HD ProRes    131 ms restart, crosses at 16
#
# Both halves scale with frame size together, which is why one number covers
# every resolution. Ten sits under the worst of the three with room to spare.
# What it buys: stepping one frame on at 4K went from a 500 ms restart to
# 56 ms, and that is the commonest gesture there is.
SKIP_AHEAD = 10

# Codecs where every frame stands alone. The names are ffmpeg's own.
ALL_INTRA = {
    "prores": "ProRes",
    "dnxhd": "DNxHD/DNxHR",
    "mjpeg": "Motion JPEG",
    "rawvideo": "uncompressed",
    "v210": "v210",
    "cfhd": "CineForm",
    "ffvhuff": "FFVHuff",
    "huffyuv": "HuffYUV",
    "utvideo": "Ut Video",
}

# What ffmpeg is asked to hand over. 16-bit so a 10-bit ProRes is not squashed
# into 8 on the way out; the player stores float16 anyway.
PIX_FMT = "rgb48le"
BYTES_PER_SAMPLE = 2

# Only so many movies stay open at once - each one is a live ffmpeg process.
MAX_OPEN = 4

# HOW MUCH TO ASK THE PIPE FOR AT ONCE. Not the whole frame: asking a Windows
# pipe for 53 MB in one call took 142 ms where the same bytes in 1 MB pieces
# took 46. Three times, for the size of one number.
READ_CHUNK = 1 << 20

# WHERE THE TWO BINARIES COME FROM. Nuke ships neither, so this is the one
# part of format support that is not self-contained. They are looked for in
# this order, first hit wins:
#
#   1. JKPLAYER_FFMPEG - a folder, for a studio that keeps its own copy
#   2. <plugin>/bin - a copy shipped beside the code
#   3. PATH - a machine where somebody installed it
#
# BOTH are needed, not just ffmpeg. ffprobe is the only one that reports the
# frame rate as the exact ratio it is: ffmpeg's own output rounds 24000/1001
# to "23.98", and since a seek is frame/fps that error grows with the frame
# number - about one whole frame by six thousand in. A player that quietly
# shows the neighbouring frame is worse than one that will not open the file.
ENV_DIR = "JKPLAYER_FFMPEG"

_tools = {}
_tools_lock = threading.Lock()


def _plugin_bin():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")


def _find_tool(name):
    """Full path to ffmpeg/ffprobe, or the bare name to let PATH try."""
    with _tools_lock:
        got = _tools.get(name)
    if got:
        return got
    exe = name + (".exe" if os.name == "nt" else "")
    for folder in (os.environ.get(ENV_DIR, ""), _plugin_bin()):
        if folder:
            cand = os.path.join(folder, exe)
            if os.path.isfile(cand):
                with _tools_lock:
                    _tools[name] = cand
                return cand
    with _tools_lock:
        _tools[name] = name            # not found here - PATH gets its turn
    return name


def ffmpeg_path():
    return _find_tool("ffmpeg")


def ffprobe_path():
    return _find_tool("ffprobe")


_TIMEOUT = 30


class MovUnsupported(Exception):
    """A movie we will not read, and why."""


# --------------------------------------------------------------- references
def is_mov(path):
    """Is this path a movie - with or without a frame on the end?"""
    base = split_ref(path)[0] if path else ""
    return bool(base) and os.path.splitext(base)[1].lower() in EXTENSIONS


def make_ref(path, frame):
    """clip.mov + 42 -> "clip.mov|42"."""
    return "%s%s%d" % (path, SEP, int(frame))


def split_ref(ref):
    """"clip.mov|42" -> ("clip.mov", 42). A plain path comes back with None."""
    if not ref:
        return "", None
    cut = ref.rfind(SEP)
    if cut < 0:
        return ref, None
    tail = ref[cut + 1:]
    try:
        return ref[:cut], int(tail)
    except ValueError:
        return ref, None


# ------------------------------------------------------------------ probing
_probe_lock = threading.Lock()
_probe_cache = {}


def _runs(exe):
    try:
        subprocess.run([exe, "-version"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=_TIMEOUT)
        return True
    except Exception:
        return False


def missing():
    """Which of the two tools cannot be run. Answered once, remembered.

    BOTH are checked. A machine with ffmpeg but no ffprobe would open movies
    and then seek by a rounded frame rate - see the note at ENV_DIR - so it
    counts as not having movie support at all rather than as having a
    slightly worse one.
    """
    with _probe_lock:
        got = _probe_cache.get("__tools__")
        if got is None:
            got = tuple(n for n, exe in (("ffmpeg", ffmpeg_path()),
                                         ("ffprobe", ffprobe_path()))
                        if not _runs(exe))
            _probe_cache["__tools__"] = got
        return got


def available():
    """Can movies be read on this machine at all?"""
    return not missing()


def why_unavailable():
    """A sentence for the panel, naming what is missing and where we looked."""
    gone = missing()
    if not gone:
        return ""
    return ("%s not found - movies need both. Looked in $%s, in %s and on "
            "PATH." % (" and ".join(gone), ENV_DIR, _plugin_bin()))


def _run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=_TIMEOUT)


def _ffprobe(path):
    """What ffprobe says about the first video stream. Cached per file."""
    key = os.path.abspath(path)
    with _probe_lock:
        got = _probe_cache.get(key)
    if got is not None:
        return got

    if not available():
        raise MovUnsupported(why_unavailable())
    fields = ("codec_name,width,height,pix_fmt,nb_frames,r_frame_rate,"
              "avg_frame_rate,color_space,color_range,color_transfer,"
              "color_primaries,bits_per_raw_sample,duration,has_b_frames")
    out = _run([ffprobe_path(), "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=" + fields,
                "-of", "default=noprint_wrappers=1", path])
    if out.returncode != 0:
        raise MovUnsupported("ffprobe could not read it: %s"
                             % out.stderr.decode("utf-8", "replace").strip()[:200])
    info = {}
    for line in out.stdout.decode("utf-8", "replace").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()
    with _probe_lock:
        if len(_probe_cache) > 64:
            _probe_cache.clear()
        _probe_cache[key] = info
    return info


def _ratio(text, default=0.0):
    """ffprobe writes rates as "24000/1001"."""
    if not text or text in ("0/0", "N/A"):
        return default
    if "/" in text:
        num, den = text.split("/", 1)
        try:
            den = float(den)
            return float(num) / den if den else default
        except ValueError:
            return default
    try:
        return float(text)
    except ValueError:
        return default


def frame_rate(path):
    info = _ffprobe(split_ref(path)[0])
    return (_ratio(info.get("r_frame_rate"))
            or _ratio(info.get("avg_frame_rate")) or 24.0)


def frame_count(path):
    """How many frames the movie holds. Counted if the container will not say."""
    base = split_ref(path)[0]
    info = _ffprobe(base)
    try:
        n = int(info.get("nb_frames", "0"))
    except ValueError:
        n = 0
    if n > 0:
        return n
    # Some containers leave nb_frames out. Duration times rate is close but
    # not exact, and "close" would put the last frame past the end - so count
    # the packets, which is exact and reads no pixels.
    out = _run([ffprobe_path(), "-v", "error", "-select_streams", "v:0",
                "-count_packets", "-show_entries", "stream=nb_read_packets",
                "-of", "csv=p=0", base])
    try:
        return int(out.stdout.decode("ascii", "replace").strip())
    except ValueError:
        return 0


def _check(info, path):
    """Refuse what we cannot decode. Long-GOP is slower, not refused."""
    try:
        w, h = int(info["width"]), int(info["height"])
    except (KeyError, ValueError):
        raise MovUnsupported("no picture size in the file")
    if w <= 0 or h <= 0:
        raise MovUnsupported("nonsense picture size %sx%s" % (w, h))
    return w, h


def probe(path):
    """Cheap facts, in the shape reader.probe hands on. Never raises."""
    base = split_ref(path)[0]
    try:
        info = _ffprobe(base)
    except Exception as exc:
        return {"supported": False, "reason": str(exc), "backend": "ffmpeg"}
    out = {
        "width": int(info.get("width", 0) or 0),
        "height": int(info.get("height", 0) or 0),
        "compression": ALL_INTRA.get(info.get("codec_name"),
                                     info.get("codec_name", "?")),
        "codec": info.get("codec_name", "?"),
        "pixel_aspect": 1.0,
        "backend": "ffmpeg",
        "pix_fmt": info.get("pix_fmt", ""),
        "frames": frame_count(base),
        "fps": _ratio(info.get("r_frame_rate"), 24.0),
        # what the file says its colour is - a suggestion for the input space,
        # never imposed, exactly as with DPX
        "color_space": info.get("color_space", ""),
        "color_range": info.get("color_range", ""),
        "suggested_space": _suggest(info),
        "chroma": _chroma(info.get("pix_fmt", "")),
        # ALL-INTRA OR NOT is a fact about how much a jump costs, and the
        # panel should be able to say so rather than leave someone wondering
        # why one clip scrubs and another hesitates.
        # The panel prints the channel list beside the format. A movie has no
        # channel names of its own, but "nothing" would read as a broken probe
        # rather than as "it is a picture", so say what actually comes out.
        "channels": (["R", "G", "B", "A"]
                     if "a" in _chroma_letters(info.get("pix_fmt", ""))
                     else ["R", "G", "B"]),
        "all_intra": info.get("codec_name") in ALL_INTRA,
        "seek_cost": ("one frame" if info.get("codec_name") in ALL_INTRA
                      else "up to a whole group of pictures"),
    }
    try:
        _check(info, base)
        out["supported"] = True
    except MovUnsupported as exc:
        out["supported"] = False
        out["reason"] = str(exc)
    return out


def _suggest(info):
    """The input space the file claims. Video is nearly always rec709."""
    tr = (info.get("color_transfer") or "").lower()
    if tr in ("bt709", "smpte170m", "bt470bg", "unknown", "", "n/a"):
        return "rec709"
    if tr in ("iec61966-2-1", "srgb"):
        return "sRGB"
    if tr in ("linear",):
        return "linear"
    return "rec709"


def _chroma_letters(pix_fmt):
    """The part of a pixel format name before the subsampling digits."""
    p = (pix_fmt or "").lower()
    for i, ch in enumerate(p):
        if ch.isdigit():
            return p[:i]
    return p


def _chroma(pix_fmt):
    """How much colour the codec threw away - a QC fact, not a detail."""
    p = (pix_fmt or "").lower()
    if "444" in p or p.startswith("rgb") or p.startswith("gbr"):
        return "4:4:4"
    if "422" in p:
        return "4:2:2"
    if "420" in p:
        return "4:2:0"
    return "?"


# ------------------------------------------------------------------ decoding
class _Movie(object):
    """One open ffmpeg, positioned somewhere in one file."""

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.info = _ffprobe(path)
        self.width, self.height = _check(self.info, path)
        self.fps = _ratio(self.info.get("r_frame_rate"), 24.0) or 24.0
        self.frame_bytes = self.width * self.height * 3 * BYTES_PER_SAMPLE
        self._proc = None
        self._next = None            # the frame the pipe is about to give us

    # ---- the process ----
    def _start(self, frame):
        self._stop()
        args = [ffmpeg_path(), "-v", "error", "-nostdin"]
        if frame > 0:
            # SEEK BEFORE -i, and land a QUARTER OF A FRAME EARLY.
            #
            # ffmpeg gives back the first frame whose timestamp is at or after
            # the seek, so seeking INTO frame f lands on f+1. Measured: with
            # -ss (f+0.5)/fps every single frame came back one too late, and
            # the last frame came back as nothing at all. Seeking to exactly
            # f/fps is right but sits on the boundary, where a float rounding
            # error of one tick tips it into f+1 - which for 24000/1001 rates
            # is not hypothetical.
            #
            # A quarter of a frame early is inside the gap between f-1 and f,
            # so it can only resolve to f, and it has margin either way.
            args += ["-ss", "%.6f" % (max(0.0, frame - 0.25) / self.fps)]
        args += ["-i", self.path, "-f", "rawvideo", "-pix_fmt", PIX_FMT, "-"]
        # NO bufsize HERE. Sizing the buffer to a whole frame looks tidy and
        # is a trap: measured on 4K it dropped the pipe from 1100 MB/s to
        # 212, a fifth of the speed, because every read then waits on a 53 MB
        # buffer to fill. Python's own default is what streams best.
        self._proc = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._next = frame

    def _stop(self):
        proc, self._proc, self._next = self._proc, None, None
        if proc is None:
            return
        try:
            proc.stdout.close()
        except Exception:
            pass
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass

    def _read_one(self):
        """Exactly one frame off the pipe, or None at the end.

        Straight into a buffer with readinto rather than joining chunks: at
        53 MB a 4K frame the join was a second copy of the whole thing, for
        nothing. A FRESH buffer each time, because the conversion afterwards
        happens outside the lock so that decodes overlap - handing the next
        reader the same bytearray would let it overwrite a frame still being
        converted.
        """
        want = self.frame_bytes
        buf = bytearray(want)
        view = memoryview(buf)
        got = 0
        while got < want:
            end = min(got + READ_CHUNK, want)
            n = self._proc.stdout.readinto(view[got:end])
            if not n:
                return None
            got += n
        return buf

    # ---- what the reader asks for ----
    def frame(self, index):
        """(h, w, 4) float16 - the frame at that index, 0-based."""
        with self.lock:
            ahead = -1 if self._proc is None else index - self._next
            if ahead < 0 or ahead > SKIP_AHEAD:
                self._start(index)
            else:
                # a few frames ahead: pull them off and drop them, which is
                # cheaper than starting ffmpeg again (see SKIP_AHEAD)
                while self._next < index:
                    if self._read_one() is None:
                        self._start(index)
                        break
                    self._next += 1
            raw = self._read_one()
            if raw is None:
                # The pipe ran dry. Once more from a fresh process, so a
                # decoder that died halfway is not reported as a short file.
                self._start(index)
                raw = self._read_one()
                if raw is None:
                    err = b""
                    if self._proc is not None:
                        try:
                            err = self._proc.stderr.read() or b""
                        except Exception:
                            err = b""
                    self._stop()
                    raise MovUnsupported(
                        "no frame %d in %s%s" % (index,
                                                 os.path.basename(self.path),
                                                 (": " + err.decode("utf-8", "replace").strip()[:120]) if err else ""))
            self._next = index + 1

        vals = np.frombuffer(bytes(memoryview(raw)), dtype="<u2").reshape(
            self.height, self.width, 3)
        return _to_half(vals)

    def close(self):
        with self.lock:
            self._stop()


# 16-bit code -> float16, the same trick as in dpxread: a lookup instead of a
# divide keeps the whole-image float32 temporary from ever existing.
_LUT = None
_LUT_LOCK = threading.Lock()


def _code_lut():
    global _LUT
    with _LUT_LOCK:
        if _LUT is None:
            _LUT = (np.arange(65536, dtype=np.float32) / 65535.0).astype(np.float16)
            _LUT.flags.writeable = False
        return _LUT


# Spread over the cores, for the same reason as the display path: turning
# 26 million samples into halves is one lookup each and no arithmetic, so one
# core spends it waiting for memory. Measured on 4K: 80 ms down to 17.
_BANDS = max(1, min(8, os.cpu_count() or 4))
_POOL = None
_POOL_LOCK = threading.Lock()


def _pool():
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ThreadPoolExecutor(max_workers=_BANDS,
                                       thread_name_prefix="exr-mov")
        return _POOL


def _to_half(vals):
    """(h, w, 3) uint16 codes -> (h, w, 4) float16, alpha solid."""
    h, w = vals.shape[0], vals.shape[1]
    out = np.empty((h, w, 4), np.float16)
    lut = _code_lut()

    def band(a, b):
        out[a:b, :, 3] = 1.0            # movies carry no alpha we can use
        for c in range(3):
            out[a:b, :, c] = lut[vals[a:b, :, c]]

    if _BANDS <= 1 or h < _BANDS or h * w < 250000:
        band(0, h)
        return out
    edges = [h * i // _BANDS for i in range(_BANDS + 1)]
    list(_pool().map(lambda i: band(edges[i], edges[i + 1]), range(_BANDS)))
    return out


_open = {}
_open_lock = threading.Lock()


def _movie(path):
    key = os.path.abspath(path)
    with _open_lock:
        mov = _open.get(key)
        if mov is None:
            while len(_open) >= MAX_OPEN:
                _, old = _open.popitem()
                old.close()
            mov = _Movie(path)
            _open[key] = mov
        return mov


def read_rgba_half(ref, layer=None):
    """(h, w, 4) float16 for "clip.mov|42". Raises MovUnsupported."""
    if layer not in (None, "", "rgba"):
        raise MovUnsupported("a movie has no layers (asked for %r)" % (layer,))
    path, frame = split_ref(ref)
    if frame is None:
        frame = 0
    return _movie(path).frame(int(frame))


def close_all():
    """Let go of every ffmpeg.

    NOT hooked to the panel closing, deliberately - Nuke closes and reopens the
    panel when it is docked (see the note at the end of panel.py), and tearing
    the decoders down every time would trade a few idle processes for a stall
    on every dock. MAX_OPEN caps how many can be idle, each one blocked writing
    to a full pipe and costing no CPU. atexit below makes sure they go when
    Nuke does rather than being orphaned.
    """
    with _open_lock:
        movies = list(_open.values())
        _open.clear()
    for mov in movies:
        mov.close()


# ----------------------------------------------------------------- metadata
_TAG_RE = re.compile(r"^TAG:(.+)$")


def metadata(ref):
    """[(key, value)] - the container's own tags plus what the stream says."""
    path = split_ref(ref)[0]
    out = []

    def add(key, value):
        if value not in (None, "", "N/A", "unknown", "0/0"):
            out.append((key, str(value)))

    try:
        info = _ffprobe(path)
    except Exception:
        return []

    add("codec", ALL_INTRA.get(info.get("codec_name"), info.get("codec_name")))
    add("size", "%sx%s" % (info.get("width"), info.get("height")))
    add("pixelFormat", info.get("pix_fmt"))
    add("chroma", _chroma(info.get("pix_fmt", "")))
    add("bitDepth", info.get("bits_per_raw_sample"))
    fps = _ratio(info.get("r_frame_rate"))
    if fps:
        add("framesPerSecond", "%g" % fps)
    add("frames", info.get("nb_frames"))
    add("duration", info.get("duration"))
    add("colorSpace", info.get("color_space"))
    add("colorRange", info.get("color_range"))
    add("colorTransfer", info.get("color_transfer"))
    add("colorPrimaries", info.get("color_primaries"))

    # the container's tags - timecode, reel and whatever the writer put there
    try:
        got = _run([ffprobe_path(), "-v", "error", "-show_entries",
                    "format_tags:stream_tags", "-select_streams", "v:0",
                    "-of", "default=noprint_wrappers=1", path])
        for line in got.stdout.decode("utf-8", "replace").splitlines():
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            m = _TAG_RE.match(k.strip())
            if m:
                add(m.group(1), v.strip())
    except Exception:
        pass
    return out


# A child process outlives its parent on Windows unless somebody says
# otherwise, and an orphaned ffmpeg holding a decoded 4K frame is not something
# to leave behind.
atexit.register(close_all)
