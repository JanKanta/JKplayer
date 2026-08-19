"""
EXR file reader - pure Python + numpy + zlib.

WHY OUR OWN READER: the OpenEXR pip package SEGFAULTS inside Nuke's Python
(a DLL clash with Nuke's own OpenEXR/Imath libraries) and would take the whole
of Nuke down with it. This reader uses NO third-party native library -> it
cannot crash.

PERFORMANCE (1920x1080 ZIPS half, measured): 1 thread 11.5 fps, 8 threads
53.6 fps, 12 threads 57.1 fps.  Nuke itself plays the same material at 22 fps.
`zlib.decompress` releases the GIL, which is why this scales across threads.

SUPPORTED: scanline EXR, compression NONE / ZIPS / ZIP, types HALF / FLOAT / UINT.
UNSUPPORTED: tiled, deep, multipart, PIZ / PXR24 / B44 / DWAA
             (for those read_display() returns None -> the caller falls back).
"""

import struct
import zlib

import numpy as np

MAGIC = 20000630

PIX_UINT, PIX_HALF, PIX_FLOAT = 0, 1, 2
_NPTYPE = {PIX_UINT: np.uint32, PIX_HALF: np.float16, PIX_FLOAT: np.float32}
_PIXSIZE = {PIX_UINT: 4, PIX_HALF: 2, PIX_FLOAT: 4}

# compression -> rows per chunk (None = we cannot do it)
_ROWS_PER_CHUNK = {0: 1, 2: 1, 3: 16}          # NONE, ZIPS, ZIP
_ZIPPED = (2, 3)

COMPRESSION_NAMES = {0: "NONE", 1: "RLE", 2: "ZIPS", 3: "ZIP", 4: "PIZ",
                     5: "PXR24", 6: "B44", 7: "B44A", 8: "DWAA", 9: "DWAB"}


class ExrUnsupported(Exception):
    """A valid EXR, but of a kind we cannot read (tiled/deep/exotic compression)."""


# ---------------------------------------------------------------------------
# LUT: half float (16 bits = 65536 values) -> uint8 sRGB.
# It removes the per-pixel pow() tonemapping: 68.7 ms -> 9.5 ms on 1080p.
# ---------------------------------------------------------------------------
def _build_half_to_srgb_lut():
    vals = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)
    vals = np.nan_to_num(vals, nan=0.0, posinf=1.0, neginf=0.0)
    srgb = np.where(vals <= 0.0031308,
                    vals * 12.92,
                    1.055 * np.clip(vals, 0.0, None) ** (1.0 / 2.4) - 0.055)
    return (np.clip(srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


HALF_TO_SRGB8 = _build_half_to_srgb_lut()


def linear_to_srgb8(arr):
    """float32 linear -> uint8 sRGB (for FLOAT channels, where the LUT cannot be used)."""
    srgb = np.where(arr <= 0.0031308,
                    arr * 12.92,
                    1.055 * np.clip(arr, 0.0, None) ** (1.0 / 2.4) - 0.055)
    return (np.clip(srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _read_cstr(buf, i):
    j = buf.index(b"\x00", i)
    return buf[i:j].decode("latin-1"), j + 1


class ExrFile(object):
    """Loads and parses an EXR. Reading is thread-safe (each instance owns its data)."""

    def __init__(self, path):
        with open(path, "rb") as fh:
            self.data = fh.read()
        self.path = path
        self._parse_header()

    # ---- header ----
    def _parse_header(self):
        data = self.data
        if len(data) < 8:
            raise ValueError("file too short")
        magic, version = struct.unpack_from("<Ii", data, 0)
        if magic != MAGIC:
            raise ValueError("not an EXR (magic %d)" % magic)
        if version & 0x200:
            raise ExrUnsupported("tiled EXR")
        if version & 0x800:
            raise ExrUnsupported("deep EXR")
        if version & 0x1000:
            raise ExrUnsupported("multipart EXR")

        self.channels = []          # [(name, pixtype)]
        self.compression = None
        self.x0 = self.y0 = 0
        self.width = self.height = 0
        # 1.0 unless the file says otherwise. Anamorphic and 2K/HD-squeezed
        # plates carry it, and without it a 2:1 squeeze reads as "the two
        # inputs are different shapes" when they are the same picture.
        self.pixel_aspect = 1.0

        i = 8
        while True:
            name, i = _read_cstr(data, i)
            if name == "":
                break
            _atype, i = _read_cstr(data, i)
            size = struct.unpack_from("<i", data, i)[0]
            i += 4
            body = data[i:i + size]
            i += size
            if name == "compression":
                self.compression = body[0]
            elif name == "dataWindow":
                x0, y0, x1, y1 = struct.unpack("<iiii", body)
                self.x0, self.y0 = x0, y0
                self.width, self.height = x1 - x0 + 1, y1 - y0 + 1
            elif name == "channels":
                self.channels = _parse_channels(body)
            elif name == "pixelAspectRatio" and size >= 4:
                par = struct.unpack_from("<f", body, 0)[0]
                if par > 0.0:
                    self.pixel_aspect = float(par)

        for _cname, ptype in self.channels:
            if ptype not in _NPTYPE:
                raise ExrUnsupported("pixel type %d" % ptype)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("missing/invalid dataWindow")
        if self.compression not in _ROWS_PER_CHUNK:
            raise ExrUnsupported("compression %s" %
                                 COMPRESSION_NAMES.get(self.compression,
                                                       self.compression))
        # EXR spec: channels are stored in the data in ALPHABETICAL order
        self.channels.sort(key=lambda c: c[0])
        self.rows_per_chunk = _ROWS_PER_CHUNK[self.compression]
        self.pixel_stride = sum(_PIXSIZE[t] for _n, t in self.channels)

        nchunks = (self.height + self.rows_per_chunk - 1) // self.rows_per_chunk
        self.offsets = struct.unpack_from("<%dQ" % nchunks, data, i)

    # ---- pixels ----
    def _decoded_bytes(self):
        """All chunks decompressed and joined into one buffer (h, row_bytes)."""
        data = self.data
        row_bytes = self.width * self.pixel_stride
        chunk_bytes = self.rows_per_chunk * row_bytes
        zipped = self.compression in _ZIPPED

        chunks = []
        for off in self.offsets:                     # the only Python loop
            size = struct.unpack_from("<i", data, off + 4)[0]
            raw = data[off + 8: off + 8 + size]
            chunks.append(zlib.decompress(raw) if zipped else raw)

        buf = np.frombuffer(b"".join(chunks), dtype=np.uint8)
        if not zipped:
            return buf.reshape(self.height, row_bytes)

        # ZIP post-process: predictor (delta) + deinterleave of the two halves.
        # Vectorised across all chunks at once; uint8 cumsum wraps mod 256,
        # which is exactly the EXR semantics (and is 1.4x faster than int16).
        if buf.size == len(self.offsets) * chunk_bytes:
            a = buf.reshape(len(self.offsets), chunk_bytes).copy()
            a[:, 1:] -= 128
            a = np.cumsum(a, axis=1, dtype=np.uint8)
            half = (chunk_bytes + 1) // 2
            out = np.empty_like(a)
            out[:, 0::2] = a[:, :half]
            out[:, 1::2] = a[:, half:chunk_bytes]
            return out.reshape(self.height, row_bytes)

        # chunks of unequal size (an incomplete last one) -> one at a time
        parts, pos = [], 0
        for chunk in chunks:
            n = len(chunk)
            a = buf[pos:pos + n].copy()
            if n:
                a[1:] -= 128
                a = np.cumsum(a, dtype=np.uint8)
            half = (n + 1) // 2
            o = np.empty(n, dtype=np.uint8)
            o[0::2] = a[:half]
            o[1::2] = a[half:n]
            parts.append(o)
            pos += n
        return np.concatenate(parts).reshape(self.height, row_bytes)

    def _planes(self):
        """{channel name: 2D array (h,w) in its original type}."""
        rows = self._decoded_bytes()
        planes, off = {}, 0
        for cname, ptype in self.channels:
            nbytes = self.width * _PIXSIZE[ptype]
            block = np.ascontiguousarray(rows[:, off:off + nbytes])
            planes[cname] = block.view(_NPTYPE[ptype]).reshape(self.height, self.width)
            off += nbytes
        return planes

    def _pick(self, planes, name):
        """Finds channel 'R' even when it is named 'color.R' and the like."""
        if name in planes:
            return planes[name]
        suffix = "." + name
        for key in planes:
            if key.endswith(suffix):
                return planes[key]
        return None

    def read_rgba_half(self, layer=None):
        """(h,w,4) float16 R,G,B,A scene-linear = THE CACHE FORMAT.

        Half on purpose: it is the native type of most EXRs (no conversion),
        half the memory of float32, and the GPU uploads it directly as
        GL_HALF_FLOAT. Alpha is 1.0 when the file has none.

        `layer` picks a layer (AOV); None or "rgba" = the base channels.
        """
        from . import exrcore                    # only for the layer mapping
        planes = self._planes()
        h, w = self.height, self.width
        out = np.empty((h, w, 4), dtype=np.float16)
        out[:, :, 3] = 1.0

        if layer and layer != exrcore.ROOT_LAYER:
            mapping = exrcore.channel_map(list(planes.keys()), layer)
            if not mapping:
                raise ExrUnsupported("layer '%s' is not in the file" % layer)
            for name, slots in mapping.items():
                data = planes[name].astype(np.float16, copy=False)
                for slot in slots:
                    out[:, :, slot] = data
            return out

        for idx, name in enumerate(("R", "G", "B")):
            p = self._pick(planes, name)
            if p is None:
                p = self._pick(planes, "Y")
            out[:, :, idx] = 0.0 if p is None else p.astype(np.float16, copy=False)
        a = self._pick(planes, "A")
        out[:, :, 3] = 1.0 if a is None else a.astype(np.float16, copy=False)
        return out

    def read_display_rgb8(self):
        """(h,w,3) uint8 sRGB for display. HALF goes through the LUT (fast)."""
        planes = self._planes()
        chans = []
        for name in ("R", "G", "B"):
            p = self._pick(planes, name)
            if p is None:
                p = self._pick(planes, "Y")
            if p is None:
                chans.append(np.zeros((self.height, self.width), np.uint8))
            elif p.dtype == np.float16:
                chans.append(HALF_TO_SRGB8[p.view(np.uint16)])     # LUT
            else:
                chans.append(linear_to_srgb8(p.astype(np.float32)))
        return np.stack(chans, axis=-1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def read_display_rgb8(path):
    """uint8 sRGB (h,w,3) for display, or None when we cannot/it fails."""
    try:
        return ExrFile(path).read_display_rgb8()
    except (ExrUnsupported, ValueError, OSError, zlib.error):
        return None


def _parse_channels(body):
    """List of [(name, pixtype)] from the 'channels' block in the header."""
    out, j = [], 0
    while j < len(body):
        cname, j = _read_cstr(body, j)
        if cname == "":
            break
        ptype = struct.unpack_from("<i", body, j)[0]
        j += 16              # type(4) + pLinear/pad(4) + xSampling,ySampling(8)
        out.append((cname, ptype))
    return out


def channel_names(path):
    """Channel names from the header - even for files we cannot decode ourselves.

    The header is always UNCOMPRESSED, so the channel list can be read even
    from a DWAA or PIZ file. Without this the layer selector would only offer
    rgba for those, because ExrFile raises on unsupported compression long
    before the caller gets to the list.
    """
    try:
        with open(path, "rb") as fh:
            data = fh.read(1 << 16)          # a header always fits in 64 kB
        if len(data) < 8 or struct.unpack_from("<I", data, 0)[0] != MAGIC:
            return []
        i = 8
        while True:
            name, i = _read_cstr(data, i)
            if name == "":
                break
            _atype, i = _read_cstr(data, i)
            size = struct.unpack_from("<i", data, i)[0]
            i += 4
            if name == "channels":
                return [c[0] for c in _parse_channels(data[i:i + size])]
            i += size
    except Exception:
        pass
    return []


def probe(path):
    """File info without decoding pixels (cheap) - for diagnostics."""
    try:
        e = ExrFile(path)
        return {"width": e.width, "height": e.height,
                "pixel_aspect": e.pixel_aspect,
                "compression": COMPRESSION_NAMES.get(e.compression, e.compression),
                "channels": [c[0] for c in e.channels],
                "supported": True}
    except ExrUnsupported as exc:
        # report the channels anyway - Nuke's DLL will read the file, just not us
        return {"supported": False, "reason": str(exc),
                "channels": channel_names(path)}
    except Exception as exc:
        return {"supported": False, "reason": "%s: %s" % (type(exc).__name__, exc),
                "channels": channel_names(path)}


# ---------------------------------------------------------------- metadata
# Every attribute in the header, not just the four the decoder needs.
#
# It costs almost nothing: the loop in ExrFile already walks name, type and
# body of each one and simply drops what it does not recognise. The header is
# also never compressed and sits at the front of the file, so this reads a few
# kilobytes rather than the frame.
#
# Deliberately NOT going through the DLL. exrcore would need another set of
# ctypes struct layouts, which is the part of it that has broken before across
# OpenEXR versions - and there is nothing to gain, because parsing a header is
# not where time goes.

HEADER_CHUNK = 64 * 1024        # read this much first; grow if the header is longer
HEADER_MAX = 4 * 1024 * 1024    # a header past this is not a header any more

# Attributes the rest of the player already shows in its own words. Kept out of
# the metadata list so it does not repeat the status line back at you.
_METADATA_SKIP = ("channels", "dataWindow", "displayWindow", "lineOrder",
                  "tiles", "chunkCount")

_LINE_ORDER = {0: "increasing Y", 1: "decreasing Y", 2: "random Y"}
_ENVMAP = {0: "latlong", 1: "cube"}


def _fmt_floats(body, count, fmt="%g"):
    if len(body) < 4 * count:
        return None
    return " ".join(fmt % v for v in struct.unpack_from("<%df" % count, body))


def _fmt_ints(body, count):
    if len(body) < 4 * count:
        return None
    return " ".join(str(v) for v in struct.unpack_from("<%di" % count, body))


def _fmt_timecode(body):
    """SMPTE timecode. The digits are BCD inside the first word."""
    if len(body) < 8:
        return None
    tf = struct.unpack_from("<I", body)[0]
    def two(units_shift, tens_shift, tens_mask):
        return ((tf >> tens_shift) & tens_mask) * 10 + ((tf >> units_shift) & 0x0F)
    return "%02d:%02d:%02d:%02d" % (two(24, 28, 0x3), two(16, 20, 0x7),
                                    two(8, 12, 0x7), two(0, 4, 0x3))


def _fmt_keycode(body):
    if len(body) < 28:
        return None
    f, m, p, c, pc, ppf, pppc = struct.unpack_from("<7i", body)
    return "mfc %d film %d prefix %d count %d perf %d" % (f, m, p, c, pc)


def _fmt_stringvector(body):
    out, i = [], 0
    while i + 4 <= len(body):
        n = struct.unpack_from("<i", body, i)[0]
        i += 4
        if n < 0 or i + n > len(body):
            break
        out.append(body[i:i + n].decode("latin-1", "replace"))
        i += n
    return ", ".join(out) if out else None


def _decode_attr(atype, body):
    """One header attribute as text, or None when we cannot make sense of it."""
    try:
        if atype == "string":
            return body.decode("utf-8", "replace")
        if atype == "int":
            return str(struct.unpack_from("<i", body)[0])
        if atype == "float":
            return "%g" % struct.unpack_from("<f", body)[0]
        if atype == "double":
            return "%g" % struct.unpack_from("<d", body)[0]
        if atype in ("v2i", "v3i", "box2i", "box2f"):
            n = {"v2i": 2, "v3i": 3, "box2i": 4, "box2f": 4}[atype]
            return (_fmt_ints(body, n) if atype != "box2f"
                    else _fmt_floats(body, n))
        if atype in ("v2f", "v3f", "m33f", "m44f", "chromaticities"):
            n = {"v2f": 2, "v3f": 3, "m33f": 9, "m44f": 16,
                 "chromaticities": 8}[atype]
            return _fmt_floats(body, n)
        if atype == "rational":
            a, b = struct.unpack_from("<iI", body)
            return "%d/%d" % (a, b)
        if atype == "timecode":
            return _fmt_timecode(body)
        if atype == "keycode":
            return _fmt_keycode(body)
        if atype == "stringvector":
            return _fmt_stringvector(body)
        if atype == "compression":
            return COMPRESSION_NAMES.get(body[0], str(body[0]))
        if atype == "lineOrder":
            return _LINE_ORDER.get(body[0], str(body[0]))
        if atype == "envmap":
            return _ENVMAP.get(body[0], str(body[0]))
    except (struct.error, IndexError, UnicodeDecodeError):
        return None
    return None


def header_attributes(path):
    """[(name, type, body)] straight out of the header, in file order.

    Raises ValueError when the file is not an EXR. Reads in chunks: a header is
    normally a few kB, but one carrying a preview image or a long history can
    run to megabytes, and the frame behind it is not worth reading for this.
    """
    size = HEADER_CHUNK
    while True:
        with open(path, "rb") as fh:
            data = fh.read(size)
        if len(data) < 8 or struct.unpack_from("<i", data)[0] != MAGIC:
            raise ValueError("not an EXR file")
        out = []
        i = 8
        complete = False
        try:
            while True:
                name, i = _read_cstr(data, i)
                if name == "":
                    complete = True
                    break
                atype, i = _read_cstr(data, i)
                n = struct.unpack_from("<i", data, i)[0]
                i += 4
                if n < 0 or i + n > len(data):
                    break               # ran off the end of what we read
                out.append((name, atype, data[i:i + n]))
                i += n
        except (ValueError, struct.error):
            pass                        # the same: read more and try again
        if complete or len(data) < size or size >= HEADER_MAX:
            return out
        size *= 4


def metadata(path):
    """[(name, text)] - the header as something a person can read.

    Order is the file's own, which is how the tools that wrote it grouped
    things. Attributes the player already reports in its own words are left
    out, and one it cannot decode is still LISTED, with its type and size:
    knowing something is there beats it quietly not existing.
    """
    try:
        attrs = header_attributes(path)
    except (OSError, ValueError):
        return []
    out = []
    for name, atype, body in attrs:
        if name in _METADATA_SKIP:
            continue
        text = _decode_attr(atype, body)
        if text is None:
            text = "<%s, %d bytes>" % (atype, len(body))
        out.append((name, text))
    return out
