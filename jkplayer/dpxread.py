"""
Reading DPX (SMPTE 268M), in pure Python and numpy.

Far simpler than EXR: fixed fields at fixed offsets, no attribute loop, and in
practice no compression at all - so decoding is reading the bytes and taking
them apart with shifts. The awkward parts are not the layout, they are the
variants, which is why anything not understood is refused loudly rather than
turned into plausible-looking wrong pixels.

WHAT COMES OUT: the code values NORMALISED to 0-1, as float16 - not linear
light. That is deliberate. The player folds the input space into its display
table (see nukelut.display_lut), and Cineon is already in that table expecting
exactly this: an encoded 0-1 where the code is v * 1023. So a 10-bit log DPX
needs no colour code of its own, only the input space set to Cineon - the same
switch an EXR in a log space would need.

THE HEADER MAY BE SHORT. The spec lays out four blocks up to byte 2048, but
the only thing that says where the pixels start is the offset at byte 4;
ffmpeg, for one, writes 1664 and omits the film and television blocks
entirely. Reading those blocks at their nominal offsets would read image data
and report it as metadata, so every field is checked against that offset
before it is touched.
"""

import os
import struct
import threading

import numpy as np

MAGIC_BE = b"SDPX"          # the file's own numbers are big-endian
MAGIC_LE = b"XPDS"          # ... little-endian

# Generic file header
OFF_DATA_OFFSET = 4
OFF_VERSION = 8
OFF_FILE_SIZE = 16
OFF_GENERIC_SIZE = 24
OFF_INDUSTRY_SIZE = 28
OFF_USER_SIZE = 32
OFF_FILE_NAME = 36
OFF_CREATED = 136
OFF_CREATOR = 160
OFF_PROJECT = 260
OFF_COPYRIGHT = 460

# Image header
OFF_ORIENTATION = 768
OFF_ELEMENTS = 770
OFF_WIDTH = 772
OFF_HEIGHT = 776
OFF_ELEMENT_0 = 780
ELEMENT_SIZE = 72

# ... within one image element
EL_DESCRIPTOR = 20
EL_TRANSFER = 21
EL_COLORIMETRIC = 22
EL_BIT_DEPTH = 23
EL_PACKING = 24
EL_ENCODING = 26
EL_DATA_OFFSET = 28
EL_EOL_PADDING = 32
EL_EOI_PADDING = 36
EL_DESCRIPTION = 40

# Image orientation header
OFF_SOURCE_NAME = 1432
OFF_SOURCE_CREATED = 1532
OFF_INPUT_DEVICE = 1556
OFF_INPUT_SERIAL = 1588
OFF_ASPECT = 1628           # two uint32: horizontal, vertical

# Film header
OFF_FILM_PREFIX = 1670
OFF_FILM_COUNT = 1676
OFF_FILM_FORMAT = 1682
OFF_FRAME_POSITION = 1714
OFF_SEQUENCE_LEN = 1718
OFF_FILM_FPS = 1726
OFF_FRAME_ID = 1734
OFF_SLATE = 1766

# Television header
OFF_TIMECODE = 1920
OFF_USER_BITS = 1924
OFF_TV_FPS = 1940
OFF_TV_GAMMA = 1948

HEADER_SIZE = 2048

# Descriptors we can turn into a picture. The rest (single channels, colour
# difference, depth) are legal DPX and simply not what a review player shows.
DESC_R, DESC_G, DESC_B, DESC_A = 1, 2, 3, 4
DESC_LUMA = 6
DESC_RGB, DESC_RGBA, DESC_ABGR = 50, 51, 52

DESCRIPTOR_NAMES = {
    0: "user defined", 1: "red", 2: "green", 3: "blue", 4: "alpha",
    6: "luminance", 7: "colour difference", 8: "depth", 9: "composite video",
    50: "RGB", 51: "RGBA", 52: "ABGR", 100: "CbYCrY", 102: "CbYCr",
}

# Transfer characteristic (element +21). It is what the file CLAIMS about its
# own encoding; plenty of writers leave it at "unspecified", so it is offered
# as a suggestion for the input space and never imposed.
TRANSFER_NAMES = {
    0: "user defined", 1: "printing density", 2: "linear", 3: "logarithmic",
    4: "unspecified video", 5: "SMPTE 240M", 6: "ITU-R 709", 7: "ITU-R 601 B/G",
    8: "ITU-R 601 M", 9: "NTSC", 10: "PAL", 11: "Z linear", 12: "homogeneous",
}

# What the transfer byte means for OUR input-space list (see nukelut.names).
# Only the two that are unambiguous - "unspecified video" is exactly that.
TRANSFER_TO_SPACE = {1: "Cineon", 2: "linear", 3: "Cineon", 11: "linear"}

PACK_TIGHT, PACK_METHOD_A, PACK_METHOD_B = 0, 1, 2
ENCODE_NONE, ENCODE_RLE = 0, 1

# DPX marks a field as "not set" by filling it with ones, per width. Without
# this the metadata panel fills up with 4294967295 and rows of 0xFF.
UNSET_U32 = 0xFFFFFFFF
UNSET_U16 = 0xFFFF
UNSET_U8 = 0xFF


class DpxUnsupported(Exception):
    """A valid DPX we cannot decode - or one we cannot prove we can."""


def is_dpx(path):
    return bool(path) and os.path.splitext(path)[1].lower() == ".dpx"


def _text(raw):
    """A fixed-width DPX string -> str, or "" when it is unset."""
    if not raw:
        return ""
    cut = raw.split(b"\0")[0]
    if not cut or cut[:1] == b"\xff":
        return ""
    try:
        out = cut.decode("ascii", "replace").strip()
    except Exception:
        return ""
    return "" if out.strip("\xff") == "" else out


class DpxHeader(object):
    """Everything the header says, read once."""

    def __init__(self, blob):
        if len(blob) < OFF_ELEMENT_0 + ELEMENT_SIZE:
            raise DpxUnsupported("file is shorter than a DPX header")
        magic = blob[:4]
        if magic == MAGIC_BE:
            self.endian = ">"
        elif magic == MAGIC_LE:
            self.endian = "<"
        else:
            raise DpxUnsupported("not a DPX (magic %r)" % magic[:4])
        self.blob = blob

        self.data_offset = self._u32(OFF_DATA_OFFSET)
        self.version = _text(blob[OFF_VERSION:OFF_VERSION + 8])
        self.orientation = self._u16(OFF_ORIENTATION)
        self.elements = self._u16(OFF_ELEMENTS)
        self.width = self._u32(OFF_WIDTH)
        self.height = self._u32(OFF_HEIGHT)

        e = OFF_ELEMENT_0
        self.descriptor = self._u8(e + EL_DESCRIPTOR)
        self.transfer = self._u8(e + EL_TRANSFER)
        self.colorimetric = self._u8(e + EL_COLORIMETRIC)
        self.bit_depth = self._u8(e + EL_BIT_DEPTH)
        self.packing = self._u16(e + EL_PACKING)
        self.encoding = self._u16(e + EL_ENCODING)
        # The ELEMENT's own offset wins when it is set: with several elements
        # the one at byte 4 points at the first, not at ours.
        el_off = self._u32(e + EL_DATA_OFFSET)
        if el_off not in (0, UNSET_U32):
            self.data_offset = el_off
        self.eol_padding = self._maybe_u32(e + EL_EOL_PADDING)
        self.eoi_padding = self._maybe_u32(e + EL_EOI_PADDING)

        if not (0 < self.width < 1 << 20) or not (0 < self.height < 1 << 20):
            raise DpxUnsupported("nonsense image size %sx%s"
                                 % (self.width, self.height))

    # ---- small readers ---------------------------------------------------
    def _u8(self, off):
        return struct.unpack_from("B", self.blob, off)[0]

    def _u16(self, off):
        return struct.unpack_from(self.endian + "H", self.blob, off)[0]

    def _u32(self, off):
        return struct.unpack_from(self.endian + "I", self.blob, off)[0]

    def _f32(self, off):
        return struct.unpack_from(self.endian + "f", self.blob, off)[0]

    def _maybe_u32(self, off):
        v = self._u32(off)
        return 0 if v == UNSET_U32 else v

    def has(self, off, size):
        """Is that field really in the header, and not already image data?

        See the note at the top: the header can stop well before 2048.
        """
        return (off + size <= len(self.blob)
                and (self.data_offset == 0 or off + size <= self.data_offset))

    # ---- what the pixels are --------------------------------------------
    @property
    def channels(self):
        return {DESC_RGB: 3, DESC_RGBA: 4, DESC_ABGR: 4, DESC_LUMA: 1,
                DESC_R: 1, DESC_G: 1, DESC_B: 1, DESC_A: 1}.get(
                    self.descriptor, 0)

    def check(self):
        """Refuse everything we cannot decode - before any pixel is read."""
        if self.encoding == ENCODE_RLE:
            raise DpxUnsupported("run-length encoded DPX is not supported")
        if self.encoding != ENCODE_NONE:
            raise DpxUnsupported("unknown encoding %d" % self.encoding)
        if self.elements not in (0, 1) and self.descriptor not in (
                DESC_RGB, DESC_RGBA, DESC_ABGR):
            raise DpxUnsupported("%d image elements is not supported"
                                 % self.elements)
        if self.channels == 0:
            raise DpxUnsupported(
                "image type %s (%d) is not a picture this player shows"
                % (DESCRIPTOR_NAMES.get(self.descriptor, "?"), self.descriptor))
        if self.bit_depth not in (8, 10, 12, 16):
            raise DpxUnsupported("%d bits per sample is not supported"
                                 % self.bit_depth)
        if self.orientation not in (0, UNSET_U16):
            # Flipped and rotated scans exist. Showing one the wrong way up is
            # worse than not showing it, so say so instead of guessing.
            raise DpxUnsupported(
                "image orientation %d (not left-to-right, top-to-bottom)"
                % self.orientation)
        if self.bit_depth in (10, 12) and self.packing == PACK_TIGHT:
            raise DpxUnsupported(
                "%d-bit samples packed with no padding are not supported"
                % self.bit_depth)
        if self.packing not in (PACK_TIGHT, PACK_METHOD_A, PACK_METHOD_B):
            raise DpxUnsupported("unknown packing %d" % self.packing)


def _read_header(path):
    with open(path, "rb") as fh:
        blob = fh.read(HEADER_SIZE)
    return DpxHeader(blob)


def _samples(head, raw):
    """The file's bytes -> (h, w, channels) uint16 of code values."""
    w, h, nch = head.width, head.height, head.channels
    order = head.endian

    if head.bit_depth == 8:
        want = w * h * nch
        if raw.size < want:
            raise DpxUnsupported("image data is short (%d of %d bytes)"
                                 % (raw.size, want))
        out = raw[:want].reshape(h, w, nch).astype(np.uint16)
        return out

    if head.bit_depth == 16:
        vals = np.frombuffer(raw, dtype=order + "u2")
        want = w * h * nch
        if vals.size < want:
            raise DpxUnsupported("image data is short (%d of %d samples)"
                                 % (vals.size, want))
        return vals[:want].reshape(h, w, nch)

    if head.bit_depth == 12:
        # Method A puts each 12-bit sample in the TOP of its own 16-bit word.
        vals = np.frombuffer(raw, dtype=order + "u2")
        want = w * h * nch
        if vals.size < want:
            raise DpxUnsupported("image data is short (%d of %d samples)"
                                 % (vals.size, want))
        vals = vals[:want].reshape(h, w, nch)
        shift = 4 if head.packing == PACK_METHOD_A else 0
        return (vals >> shift) & 0x0FFF

    # 10 bit: three samples to a 32-bit word, two bits spare.
    #
    # METHOD A puts the spare bits at the BOTTOM, so the first sample sits in
    # bits 31..22. This is the one everything in the wild writes, and it was
    # settled here by decoding a file from another implementation both ways
    # and seeing which one gave back the values that went in - not by reading
    # the sentence in the spec, which can be read either way.
    words = np.frombuffer(raw, dtype=order + "u4")
    per_row = (w * nch + 2) // 3           # words a row takes
    want = per_row * h
    if words.size < want:
        raise DpxUnsupported("image data is short (%d of %d words)"
                             % (words.size, want))
    words = words[:want].reshape(h, per_row)
    if head.packing == PACK_METHOD_B:
        shifts = (20, 10, 0)
    else:
        shifts = (22, 12, 2)
    triples = np.empty((h, per_row, 3), np.uint16)
    for i, s in enumerate(shifts):
        triples[:, :, i] = (words >> s) & 0x03FF
    flat = triples.reshape(h, per_row * 3)
    return flat[:, :w * nch].reshape(h, w, nch)


# Code value -> float16, one table per bit depth, built once.
#
# THE TABLE IS THE POINT. Doing it in arithmetic means a uint32 temporary and
# then a float32 one, each the size of the image, and only then the cast down
# to half - measured 218 ms a 4K frame. A lookup goes straight from the code
# to the number that gets stored, and the biggest table (16-bit) is 128 KB, so
# it sits in cache. Measured 218 -> 111 ms, and unlike the arithmetic it lets
# go of the GIL, so the loader's threads actually overlap.
_LUTS = {}
_LUT_LOCK = threading.Lock()


def _code_lut(bits):
    with _LUT_LOCK:
        lut = _LUTS.get(bits)
        if lut is None:
            top = float((1 << bits) - 1)
            lut = (np.arange(1 << bits, dtype=np.float32) / top).astype(np.float16)
            lut.flags.writeable = False
            _LUTS[bits] = lut
        return lut


def _fast_rgb10(head, raw, out):
    """The case everything in the wild writes: 10-bit RGB, method A.

    Three samples to a 32-bit word and three channels to a pixel means ONE
    WORD IS ONE PIXEL - no interleaving to undo, no padding to skip, so each
    channel is a shift and a lookup straight into the output. Returns False
    when the file is not that shape and the general path has to run.
    """
    if (head.bit_depth != 10 or head.descriptor != DESC_RGB
            or head.packing != PACK_METHOD_A):
        return False
    w, h = head.width, head.height
    words = np.frombuffer(raw, dtype=head.endian + "u4")
    if words.size < w * h:
        raise DpxUnsupported("image data is short (%d of %d words)"
                             % (words.size, w * h))
    words = words[:w * h].reshape(h, w)
    lut = _code_lut(10)
    for c, shift in enumerate((22, 12, 2)):
        out[:, :, c] = lut[(words >> shift) & 0x03FF]
    return True


def read_rgba_half(path, layer=None):
    """(h, w, 4) float16 - code values normalised to 0-1, alpha 1 when absent.

    `layer` exists so this can stand in for the EXR reader; DPX holds one
    image and nothing else, so anything but the default is an error rather
    than something quietly ignored.
    """
    if layer not in (None, "", "rgba"):
        raise DpxUnsupported("DPX has no layers (asked for %r)" % (layer,))
    head = _read_header(path)
    head.check()
    with open(path, "rb") as fh:
        fh.seek(head.data_offset)
        raw = np.frombuffer(fh.read(), dtype=np.uint8)

    if head.eol_padding or head.eoi_padding:
        raise DpxUnsupported("padded scanlines are not supported")

    h, w = head.height, head.width
    out = np.empty((h, w, 4), np.float16)
    out[:, :, 3] = 1.0                     # solid unless the file says otherwise
    if _fast_rgb10(head, raw, out):
        return out

    codes = _samples(head, raw)
    lut = _code_lut(head.bit_depth)
    if head.descriptor == DESC_ABGR:
        for dst, src in ((0, 3), (1, 2), (2, 1), (3, 0)):
            out[:, :, dst] = lut[codes[:, :, src]]
    elif head.channels == 1:
        # a single channel shows as grey, and keeps a solid alpha
        grey = lut[codes[:, :, 0]]
        for c in (0, 1, 2):
            out[:, :, c] = grey
    else:
        for c in range(head.channels):
            out[:, :, c] = lut[codes[:, :, c]]
    return out


def probe(path):
    """Cheap facts about the file, in the shape reader.probe hands on."""
    head = _read_header(path)
    aspect = 1.0
    if head.has(OFF_ASPECT, 8):
        hor = head._u32(OFF_ASPECT)
        ver = head._u32(OFF_ASPECT + 4)
        if hor not in (0, UNSET_U32) and ver not in (0, UNSET_U32):
            aspect = float(hor) / float(ver)
    info = {
        "width": head.width,
        "height": head.height,
        "compression": "none",            # RLE is refused in check()
        "pixel_aspect": aspect,
        "bit_depth": head.bit_depth,
        "descriptor": DESCRIPTOR_NAMES.get(head.descriptor, str(head.descriptor)),
        "transfer": TRANSFER_NAMES.get(head.transfer, str(head.transfer)),
        # what the file claims it is encoded in, for the input-space menu -
        # a suggestion, see TRANSFER_TO_SPACE
        "suggested_space": TRANSFER_TO_SPACE.get(head.transfer),
    }
    try:
        head.check()
    except DpxUnsupported as exc:
        info["reason"] = str(exc)
    return info


def _timecode(v):
    """The BCD word DPX stores -> HH:MM:SS:FF."""
    if v in (0, UNSET_U32):
        return ""
    parts = [(v >> s) & 0xFF for s in (24, 16, 8, 0)]
    out = []
    for byte in parts:
        hi, lo = byte >> 4, byte & 0x0F
        if hi > 9 or lo > 9:
            return ""                      # not BCD after all
        out.append("%d%d" % (hi, lo))
    return ":".join(out)


def metadata(path):
    """[(key, value)] from the header, in the order the file lays them out.

    Only what is really there: DPX marks an unset field by filling it with
    ones, and a panel full of 4294967295 would be worse than an empty one.
    """
    try:
        head = _read_header(path)
    except (DpxUnsupported, OSError):
        return []

    out = []

    def add(key, value):
        if value not in (None, "", []):
            out.append((key, str(value)))

    def text(key, off, size):
        if head.has(off, size):
            add(key, _text(head.blob[off:off + size]))

    def u32(key, off, scale=None):
        if not head.has(off, 4):
            return
        v = head._u32(off)
        if v != UNSET_U32:
            add(key, v if scale is None else scale(v))

    def f32(key, off, fmt="%g"):
        if not head.has(off, 4):
            return
        v = head._f32(off)
        if v == v and abs(v) < 1e30:       # not NaN, not the unset pattern
            add(key, fmt % v)

    add("dpxVersion", head.version)
    text("fileName", OFF_FILE_NAME, 100)
    text("created", OFF_CREATED, 24)
    text("creator", OFF_CREATOR, 100)
    text("project", OFF_PROJECT, 200)
    text("copyright", OFF_COPYRIGHT, 200)

    add("size", "%dx%d" % (head.width, head.height))
    add("bitDepth", head.bit_depth)
    add("imageType", DESCRIPTOR_NAMES.get(head.descriptor, head.descriptor))
    add("transfer", TRANSFER_NAMES.get(head.transfer, head.transfer))
    add("colorimetric", TRANSFER_NAMES.get(head.colorimetric, head.colorimetric))
    if head.has(OFF_ELEMENT_0 + EL_DESCRIPTION, 32):
        text("elementDescription", OFF_ELEMENT_0 + EL_DESCRIPTION, 32)

    text("sourceFile", OFF_SOURCE_NAME, 100)
    text("sourceCreated", OFF_SOURCE_CREATED, 24)
    text("inputDevice", OFF_INPUT_DEVICE, 32)
    text("inputSerial", OFF_INPUT_SERIAL, 32)

    text("filmPrefix", OFF_FILM_PREFIX, 6)
    text("filmCount", OFF_FILM_COUNT, 4)
    text("filmFormat", OFF_FILM_FORMAT, 32)
    u32("framePosition", OFF_FRAME_POSITION)
    u32("sequenceLength", OFF_SEQUENCE_LEN)
    f32("filmFrameRate", OFF_FILM_FPS)
    text("frameId", OFF_FRAME_ID, 32)
    text("slate", OFF_SLATE, 100)

    if head.has(OFF_TIMECODE, 4):
        add("timeCode", _timecode(head._u32(OFF_TIMECODE)))
    if head.has(OFF_USER_BITS, 4):
        v = head._u32(OFF_USER_BITS)
        if v not in (0, UNSET_U32):
            add("userBits", "0x%08X" % v)
    f32("framesPerSecond", OFF_TV_FPS)
    f32("gamma", OFF_TV_GAMMA)
    return out


def channel_names(path):
    """The channels this file carries, EXR-style names."""
    head = _read_header(path)
    if head.descriptor in (DESC_RGBA, DESC_ABGR):
        return ["R", "G", "B", "A"]
    if head.descriptor == DESC_RGB:
        return ["R", "G", "B"]
    if head.descriptor == DESC_LUMA:
        return ["Y"]
    return [DESCRIPTOR_NAMES.get(head.descriptor, "?")]
