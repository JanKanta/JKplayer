"""
Reading EXR through NUKE'S OWN library (OpenEXRCore) called via ctypes.

WHY: our pure Python reader (exrread.py) only handles NONE/ZIPS/ZIP. This one
does EVERYTHING Nuke does - DWAA, DWAB, PIZ, B44, PXR24, tiled - at C speed.

WHY IT IS SAFE (unlike the OpenEXR pip package, which segfaulted): we use the
library Nuke already has loaded - no version clash. `OpenEXRCore` is a pure C
API, so ctypes is enough, no compilation.

It is found next to the running interpreter (see paths.nuke_dirs), so it works on
Windows, Linux and macOS and on any Nuke version without configuration. When it
is not found, everything falls back to the pure Python reader.

The layout of the pipeline struct was found EMPIRICALLY (see OFFSET_* below)
and is VERIFIED on every open (channel names and sizes have to line up). When
they do not, an exception is raised and the caller uses the pure Python reader.
"""

import ctypes as C
import glob
import os
import sys
import threading

import numpy as np

from .paths import nuke_dirs

# ----------------------------------------------------------------- library
_LIB = None
_LIB_ERROR = None
_LIB_LOCK = threading.Lock()

# What the library is called on each platform. Globs rather than fixed names -
# the version is in the file name (OpenEXRCore_Foundry_3_3) and changes with
# every Nuke, so pinning it would mean a new release for every Nuke release.
_LIB_PATTERNS = {
    "win32": ["OpenEXRCore*.dll"],
    "darwin": ["libOpenEXRCore*.dylib"],
}
_LIB_PATTERNS_DEFAULT = ["libOpenEXRCore*.so*"]        # Linux and the rest


# offsets in exr_decode_pipeline_t (found empirically, verified at runtime)
OFFSET_CHANNELS = 8
OFFSET_CHANNEL_COUNT = 16
PIPELINE_SIZE = 8192              # generous headroom, we do not model the struct


class ExrCoreError(Exception):
    pass


class ChunkInfo(C.Structure):
    _fields_ = [("idx", C.c_int32), ("start_x", C.c_int32),
                ("start_y", C.c_int32), ("height", C.c_int32),
                ("width", C.c_int32), ("level_x", C.c_uint8),
                ("level_y", C.c_uint8), ("type", C.c_uint8),
                ("compression", C.c_uint8), ("data_offset", C.c_uint64),
                ("packed_size", C.c_uint64), ("unpacked_size", C.c_uint64),
                ("sample_count_data_offset", C.c_uint64),
                ("sample_count_table_size", C.c_uint64)]


class ChanInfo(C.Structure):
    _fields_ = [("channel_name", C.c_char_p), ("height", C.c_int32),
                ("width", C.c_int32), ("x_samples", C.c_int32),
                ("y_samples", C.c_int32), ("p_linear", C.c_uint8),
                ("bytes_per_element", C.c_int8), ("data_type", C.c_uint16),
                ("user_bytes_per_element", C.c_int16),
                ("user_data_type", C.c_uint16),
                ("user_pixel_stride", C.c_int32),
                ("user_line_stride", C.c_int32),
                ("decode_to_ptr", C.c_void_p)]


class AttrString(C.Structure):
    """exr_attr_string_t - a string in the header carries its own length."""
    _fields_ = [("length", C.c_int32), ("alloc_size", C.c_int32),
                ("str", C.c_char_p)]


class ChlistEntry(C.Structure):
    """exr_attr_chlist_entry_t"""
    _fields_ = [("name", AttrString), ("pixel_type", C.c_int),
                ("p_linear", C.c_uint8), ("reserved", C.c_uint8 * 3),
                ("x_sampling", C.c_int32), ("y_sampling", C.c_int32)]


class Chlist(C.Structure):
    """exr_attr_chlist_t"""
    _fields_ = [("num_channels", C.c_int), ("num_alloced", C.c_int),
                ("entries", C.POINTER(ChlistEntry))]


_DTYPE = {0: np.uint32, 1: np.float16, 2: np.float32}      # UINT, HALF, FLOAT


def _find_library():
    """(directory, full path) of the newest OpenEXRCore found, or (None, None)."""
    patterns = _LIB_PATTERNS.get(sys.platform, _LIB_PATTERNS_DEFAULT)
    for d in nuke_dirs():
        found = []
        for pattern in patterns:
            found.extend(glob.glob(os.path.join(d, pattern)))
        if found:
            # sorted by name, so 3_3 wins when several sit side by side
            return d, sorted(found)[-1]
    return None, None


def library():
    """Loads the DLL (once). Returns None when it is not available."""
    global _LIB, _LIB_ERROR
    if _LIB is not None or _LIB_ERROR is not None:
        return _LIB
    with _LIB_LOCK:
        if _LIB is not None or _LIB_ERROR is not None:
            return _LIB
        d, path = _find_library()
        if path is None:
            _LIB_ERROR = ("OpenEXRCore not found in %s"
                          % (", ".join(nuke_dirs()) or "any known location"))
            return None
        try:
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(d)
            lib = C.CDLL(path)
            lib.exr_start_read.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p]
            lib.exr_start_read.restype = C.c_int
            lib.exr_finish.argtypes = [C.c_void_p]
            lib.exr_finish.restype = C.c_int
            lib.exr_get_data_window.argtypes = [C.c_void_p, C.c_int, C.c_void_p]
            lib.exr_get_data_window.restype = C.c_int
            lib.exr_get_compression.argtypes = [C.c_void_p, C.c_int,
                                                C.POINTER(C.c_int)]
            lib.exr_get_compression.restype = C.c_int
            lib.exr_get_storage.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_int)]
            lib.exr_get_storage.restype = C.c_int
            lib.exr_get_scanlines_per_chunk.argtypes = [C.c_void_p, C.c_int,
                                                        C.POINTER(C.c_int32)]
            lib.exr_get_scanlines_per_chunk.restype = C.c_int
            lib.exr_read_scanline_chunk_info.argtypes = [
                C.c_void_p, C.c_int, C.c_int, C.POINTER(ChunkInfo)]
            lib.exr_read_scanline_chunk_info.restype = C.c_int
            lib.exr_decoding_initialize.argtypes = [
                C.c_void_p, C.c_int, C.POINTER(ChunkInfo), C.c_void_p]
            lib.exr_decoding_initialize.restype = C.c_int
            lib.exr_decoding_choose_default_routines.argtypes = [
                C.c_void_p, C.c_int, C.c_void_p]
            lib.exr_decoding_choose_default_routines.restype = C.c_int
            lib.exr_decoding_update.argtypes = [C.c_void_p, C.c_int,
                                                C.POINTER(ChunkInfo), C.c_void_p]
            lib.exr_decoding_update.restype = C.c_int
            lib.exr_decoding_run.argtypes = [C.c_void_p, C.c_int, C.c_void_p]
            lib.exr_decoding_run.restype = C.c_int
            lib.exr_decoding_destroy.argtypes = [C.c_void_p, C.c_void_p]
            lib.exr_decoding_destroy.restype = C.c_int
            # multipart: parts and their channels (older DLLs may lack these)
            for fn, args in (
                    ("exr_get_count", [C.c_void_p, C.POINTER(C.c_int)]),
                    ("exr_get_name", [C.c_void_p, C.c_int,
                                      C.POINTER(C.c_char_p)]),
                    ("exr_get_channels", [C.c_void_p, C.c_int,
                                          C.POINTER(C.POINTER(Chlist))])):
                f = getattr(lib, fn, None)
                if f is not None:
                    f.argtypes, f.restype = args, C.c_int
            _LIB = lib
        except Exception as exc:
            _LIB_ERROR = "%s: %s" % (type(exc).__name__, exc)
            return None
    return _LIB


def available():
    return library() is not None


def library_error():
    return _LIB_ERROR


# ------------------------------------------------------------------ reading
def read_planes(path):
    """{channel name: 2D numpy array} in the native type. Raises ExrCoreError."""
    lib = library()
    if lib is None:
        raise ExrCoreError("OpenEXRCore is not available: %s" % _LIB_ERROR)

    ctxt = C.c_void_p()
    rc = lib.exr_start_read(C.byref(ctxt), path.encode("utf-8"), None)
    if rc != 0 or not ctxt:
        raise ExrCoreError("cannot open (rc=%d): %s" % (rc, path))
    try:
        storage = C.c_int()
        if lib.exr_get_storage(ctxt, 0, C.byref(storage)) == 0 and storage.value != 0:
            raise ExrCoreError("scanline EXR only (storage=%d)" % storage.value)

        dw = (C.c_int32 * 4)()
        if lib.exr_get_data_window(ctxt, 0, C.byref(dw)) != 0:
            raise ExrCoreError("cannot read dataWindow")
        x0, y0, x1, y1 = dw[0], dw[1], dw[2], dw[3]
        width, height = x1 - x0 + 1, y1 - y0 + 1
        if width <= 0 or height <= 0:
            raise ExrCoreError("invalid dataWindow")

        ci = ChunkInfo()
        if lib.exr_read_scanline_chunk_info(ctxt, 0, y0, C.byref(ci)) != 0:
            raise ExrCoreError("cannot read chunk info")

        pipe = (C.c_uint8 * PIPELINE_SIZE)()
        if lib.exr_decoding_initialize(ctxt, 0, C.byref(ci), C.byref(pipe)) != 0:
            raise ExrCoreError("decoding_initialize failed")
        try:
            base = C.addressof(pipe)
            chans_ptr = C.cast(base + OFFSET_CHANNELS,
                               C.POINTER(C.c_void_p))[0]
            nchan = C.cast(base + OFFSET_CHANNEL_COUNT,
                           C.POINTER(C.c_int16))[0]
            if not chans_ptr or not (0 < nchan <= 64):
                raise ExrCoreError("unexpected pipeline layout "
                                   "(channels=%s, count=%s)" % (chans_ptr, nchan))
            chans = C.cast(chans_ptr, C.POINTER(ChanInfo))

            # LAYOUT CHECK: channels must have the right width and a known type
            planes, targets = {}, []
            for i in range(nchan):
                c = chans[i]
                if c.width != width or c.data_type not in _DTYPE:
                    raise ExrCoreError(
                        "layout check failed (channel %d: w=%d type=%d)"
                        % (i, c.width, c.data_type))
                name = c.channel_name.decode("latin-1") if c.channel_name else str(i)
                dt = _DTYPE[c.data_type]
                plane = np.empty((height, width), dtype=dt)
                planes[name] = plane
                targets.append((plane, plane.dtype.itemsize))

            scan_per_chunk = C.c_int32()
            lib.exr_get_scanlines_per_chunk(ctxt, 0, C.byref(scan_per_chunk))
            step = max(1, scan_per_chunk.value)

            first = True
            y = y0
            while y <= y1:
                if not first:
                    if lib.exr_read_scanline_chunk_info(ctxt, 0, y,
                                                        C.byref(ci)) != 0:
                        raise ExrCoreError("chunk info failed at y=%d" % y)
                    if lib.exr_decoding_update(ctxt, 0, C.byref(ci),
                                               C.byref(pipe)) != 0:
                        raise ExrCoreError("decoding_update failed at y=%d" % y)
                row = ci.start_y - y0
                for i in range(nchan):
                    plane, itemsize = targets[i]
                    c = chans[i]
                    c.user_data_type = c.data_type
                    c.user_bytes_per_element = c.bytes_per_element
                    c.user_pixel_stride = itemsize
                    c.user_line_stride = width * itemsize
                    c.decode_to_ptr = (plane.ctypes.data
                                       + row * width * itemsize)
                if first:
                    if lib.exr_decoding_choose_default_routines(
                            ctxt, 0, C.byref(pipe)) != 0:
                        raise ExrCoreError("choose_default_routines failed")
                    first = False
                if lib.exr_decoding_run(ctxt, 0, C.byref(pipe)) != 0:
                    raise ExrCoreError("decoding_run failed at y=%d" % y)
                y = ci.start_y + max(1, ci.height)
                if ci.height <= 0:
                    y += step
            return planes
        finally:
            lib.exr_decoding_destroy(ctxt, C.byref(pipe))
    finally:
        lib.exr_finish(C.byref(ctxt))


ROOT_LAYER = "rgba"          # channels without a dot (R, G, B, A)

# Channel suffix -> slot in RGBA. Besides colours it also takes coordinate
# names, so normals and positions can be displayed (normal.X, P.x, ...).
_SUFFIX = {"R": 0, "RED": 0, "X": 0, "U": 0,
           "G": 1, "GREEN": 1, "Y": 1, "V": 1,
           "B": 2, "BLUE": 2, "Z": 2, "W": 2,
           "A": 3, "ALPHA": 3}


def layer_of(channel):
    """Which layer the channel belongs to. No dot = the base RGBA."""
    return channel.rsplit(".", 1)[0] if "." in channel else ROOT_LAYER


def layers(channels):
    """Layers in the file, the base RGBA always first (when it exists)."""
    seen = []
    for name in channels:
        lay = layer_of(name)
        if lay not in seen:
            seen.append(lay)
    root = [lay for lay in seen if lay == ROOT_LAYER]
    return root + sorted(lay for lay in seen if lay != ROOT_LAYER)


def channel_map(channels, layer=ROOT_LAYER):
    """{channel name: [slots in RGBA]} for the given layer.

    A layer with a SINGLE channel (typically depth.Z) is copied into RGB so it
    reads as grey - otherwise it would only light up one channel.
    """
    mine = [c for c in channels if layer_of(c) == layer]
    if not mine:
        return {}
    if len(mine) == 1:
        return {mine[0]: [0, 1, 2]}
    out = {}
    free = [0, 1, 2, 3]
    for name in mine:                       # by suffix first
        idx = _SUFFIX.get(name.rsplit(".", 1)[-1].upper())
        if idx is not None and idx in free:
            out[name] = [idx]
            free.remove(idx)
    for name in mine:                       # the rest in order into free slots
        if name not in out and free:
            out[name] = [free.pop(0)]
    return out


# -------------------------------------------------------------- multipart
# A multipart EXR keeps every AOV in ITS OWN part with its own header. When
# only the channels of part 0 are read (which is how it used to be here), such
# a file looks as if it had a single rgba layer - the rest sits in parts nobody
# asked about.
def _part_count(lib, ctxt):
    fn = getattr(lib, "exr_get_count", None)
    if fn is None:
        return 1
    n = C.c_int()
    if fn(ctxt, C.byref(n)) != 0:
        return 1
    return max(1, n.value)


def _part_name(lib, ctxt, part):
    fn = getattr(lib, "exr_get_name", None)
    if fn is None:
        return ""
    nm = C.c_char_p()
    if fn(ctxt, part, C.byref(nm)) != 0 or not nm.value:
        return ""
    return nm.value.decode("latin-1")


def _part_channels(lib, ctxt, part):
    """Channel names in the part. Empty list when the layout does not fit."""
    fn = getattr(lib, "exr_get_channels", None)
    if fn is None:
        return []
    lst = C.POINTER(Chlist)()
    if fn(ctxt, part, C.byref(lst)) != 0 or not lst:
        return []
    cl = lst.contents
    if not (0 < cl.num_channels <= 1024):
        return []
    out = []
    for i in range(cl.num_channels):
        e = cl.entries[i]
        if not e.name.str:
            return []
        name = e.name.str.decode("latin-1")
        if len(name) != e.name.length:      # struct layout check
            return []
        out.append(name)
    return out


def _strip_view(names):
    """Cuts a common view suffix off the part names.

    Nuke names parts "layer.view" (rgba.main, depth.main). When ALL parts share
    the same last component, it is the view and does not belong in the layer
    name. When they differ (left/right in stereo) it is kept - that is
    information the user needs to see.
    """
    tails = set()
    for name in names:
        if "." not in name:
            return list(names)
        tails.add(name.rsplit(".", 1)[1])
    if len(tails) != 1:
        return list(names)
    return [name.rsplit(".", 1)[0] for name in names]


def _layer_table(lib, ctxt):
    """[(name for the user, part index, layer inside the part)]."""
    nparts = _part_count(lib, ctxt)
    pnames = _strip_view([_part_name(lib, ctxt, p) for p in range(nparts)])

    table, taken = [], set()
    for part in range(nparts):
        chans = _part_channels(lib, ctxt, part)
        if not chans:
            continue
        inner_layers = layers(chans)
        for inner in inner_layers:
            # Some renderers give the channels bare names (R,G,B,A) and tell
            # AOVs apart only by the part name - then the part name is the layer.
            label = inner
            if (inner == ROOT_LAYER and nparts > 1 and pnames[part]
                    and len(inner_layers) == 1):
                label = pnames[part]
            if label in taken:
                label = "%s#%d" % (label, part)
            taken.add(label)
            table.append((label, part, inner))
    return table


def file_layers(path):
    """Layers in the file across all parts. Empty when it cannot be done."""
    lib = library()
    if lib is None:
        return []
    ctxt = C.c_void_p()
    if lib.exr_start_read(C.byref(ctxt), path.encode("utf-8"), None) != 0:
        return []
    try:
        return [label for label, _part, _inner in _layer_table(lib, ctxt)]
    finally:
        lib.exr_finish(C.byref(ctxt))


def _resolve_layer(lib, ctxt, layer):
    """Layer -> (part index, layer name inside the part)."""
    if _part_count(lib, ctxt) == 1:
        return 0, layer         # ordinary file: no searching through parts
    table = _layer_table(lib, ctxt)
    for label, part, inner in table:
        if label == layer:
            return part, inner
    # An unknown layer - typically the default "rgba" in a file where no part
    # is called that (the first AOV may be named Diffuse). Take the first one
    # so at least something shows up.
    if table and layer == ROOT_LAYER:
        return table[0][1], table[0][2]
    return 0, layer         # let channel_map report the error, it is clearer


def read_rgba_half_direct(path, layer=ROOT_LAYER):
    """(h,w,4) float16 RGBA - decodes DIRECTLY into the final buffer.

    Two savings over read_planes(): the library writes straight into the
    interleaved RGBA array (no copy while assembling channels - on 6K that is
    148 MB per frame) and channels outside RGBA are not decoded at all
    (decode_to_ptr = NULL).
    """
    lib = library()
    if lib is None:
        raise ExrCoreError("OpenEXRCore is not available: %s" % _LIB_ERROR)

    ctxt = C.c_void_p()
    rc = lib.exr_start_read(C.byref(ctxt), path.encode("utf-8"), None)
    if rc != 0 or not ctxt:
        raise ExrCoreError("cannot open (rc=%d): %s" % (rc, path))
    try:
        part, inner = _resolve_layer(lib, ctxt, layer)

        storage = C.c_int()
        if (lib.exr_get_storage(ctxt, part, C.byref(storage)) == 0
                and storage.value != 0):
            raise ExrCoreError("scanline EXR only (storage=%d)" % storage.value)

        dw = (C.c_int32 * 4)()
        if lib.exr_get_data_window(ctxt, part, C.byref(dw)) != 0:
            raise ExrCoreError("cannot read dataWindow")
        x0, y0, x1, y1 = dw[0], dw[1], dw[2], dw[3]
        width, height = x1 - x0 + 1, y1 - y0 + 1
        if width <= 0 or height <= 0:
            raise ExrCoreError("invalid dataWindow")

        ci = ChunkInfo()
        if lib.exr_read_scanline_chunk_info(ctxt, part, y0, C.byref(ci)) != 0:
            raise ExrCoreError("cannot read chunk info")

        pipe = (C.c_uint8 * PIPELINE_SIZE)()
        if lib.exr_decoding_initialize(ctxt, part, C.byref(ci),
                                       C.byref(pipe)) != 0:
            raise ExrCoreError("decoding_initialize failed")
        try:
            base = C.addressof(pipe)
            chans_ptr = C.cast(base + OFFSET_CHANNELS, C.POINTER(C.c_void_p))[0]
            nchan = C.cast(base + OFFSET_CHANNEL_COUNT, C.POINTER(C.c_int16))[0]
            if not chans_ptr or not (0 < nchan <= 64):
                raise ExrCoreError("unexpected pipeline layout")
            chans = C.cast(chans_ptr, C.POINTER(ChanInfo))

            out = np.empty((height, width, 4), dtype=np.float16)
            out[:, :, 3] = 1.0                    # alpha, when the file has none
            out_base = out.ctypes.data
            row_bytes = width * 8                 # 4 channels x half
            HALF = 1
            copies = []                           # (source, [targets]) after decode

            names = []
            for i in range(nchan):
                c = chans[i]
                if c.width != width:
                    raise ExrCoreError("layout check failed (channel %d)" % i)
                names.append(c.channel_name.decode("latin-1")
                             if c.channel_name else "")
            wanted_map = channel_map(names, inner)

            wanted = []                           # (channel index, offset in pixel)
            for i, name in enumerate(names):
                slots = wanted_map.get(name)
                if not slots:
                    chans[i].decode_to_ptr = None  # not interested -> do not decode
                    continue
                c = chans[i]
                # the library converts types itself -> ask for half whatever
                # the file holds
                c.user_data_type = HALF
                c.user_bytes_per_element = 2
                c.user_pixel_stride = 8
                c.user_line_stride = row_bytes
                # One channel can go to several slots (a grey layer). The
                # library can only write to one, the rest is copied at the end.
                wanted.append((i, slots[0] * 2))
                if len(slots) > 1:
                    copies.append((slots[0], slots[1:]))
            if not wanted:
                raise ExrCoreError("layer '%s' has no usable channels" % layer)

            if lib.exr_decoding_choose_default_routines(
                    ctxt, part, C.byref(pipe)) != 0:
                raise ExrCoreError("choose_default_routines failed")

            y = y0
            first = True
            while y <= y1:
                if not first:
                    if lib.exr_read_scanline_chunk_info(ctxt, part, y,
                                                        C.byref(ci)) != 0:
                        raise ExrCoreError("chunk info failed at y=%d" % y)
                    if lib.exr_decoding_update(ctxt, part, C.byref(ci),
                                               C.byref(pipe)) != 0:
                        raise ExrCoreError("decoding_update failed at y=%d" % y)
                first = False
                row_ptr = out_base + (ci.start_y - y0) * row_bytes
                for i, off in wanted:              # per chunk only pointers now
                    chans[i].decode_to_ptr = row_ptr + off
                if lib.exr_decoding_run(ctxt, part, C.byref(pipe)) != 0:
                    raise ExrCoreError("decoding_run failed at y=%d" % y)
                y = ci.start_y + max(1, ci.height)
            for src, dsts in copies:              # grey layer into all of RGB
                for dst in dsts:
                    out[:, :, dst] = out[:, :, src]
            return out
        finally:
            lib.exr_decoding_destroy(ctxt, C.byref(pipe))
    finally:
        lib.exr_finish(C.byref(ctxt))


def _pick(planes, name):
    if name in planes:
        return planes[name]
    suffix = "." + name
    for key in planes:
        if key.endswith(suffix):
            return planes[key]
    return None


def read_rgba_half(path):
    """(h,w,4) float16 scene-linear RGBA - the same format as exrread."""
    planes = read_planes(path)
    if not planes:
        raise ExrCoreError("no channels")
    any_plane = next(iter(planes.values()))
    h, w = any_plane.shape
    out = np.empty((h, w, 4), dtype=np.float16)
    for idx, nm in enumerate(("R", "G", "B")):
        p = _pick(planes, nm)
        if p is None:
            p = _pick(planes, "Y")
        out[:, :, idx] = 0.0 if p is None else p.astype(np.float16, copy=False)
    a = _pick(planes, "A")
    out[:, :, 3] = 1.0 if a is None else a.astype(np.float16, copy=False)
    return out


def probe(path):
    """File info (dimensions, compression) through the C API."""
    lib = library()
    if lib is None:
        return {"supported": False, "reason": "OpenEXRCore unavailable"}
    ctxt = C.c_void_p()
    if lib.exr_start_read(C.byref(ctxt), path.encode("utf-8"), None) != 0:
        return {"supported": False, "reason": "cannot open"}
    try:
        dw = (C.c_int32 * 4)()
        lib.exr_get_data_window(ctxt, 0, C.byref(dw))
        comp = C.c_int()
        lib.exr_get_compression(ctxt, 0, C.byref(comp))
        storage = C.c_int()
        lib.exr_get_storage(ctxt, 0, C.byref(storage))
        return {"supported": storage.value == 0,
                "width": dw[2] - dw[0] + 1, "height": dw[3] - dw[1] + 1,
                "compression_id": comp.value, "storage": storage.value,
                "reason": "" if storage.value == 0 else "tiled/deep"}
    finally:
        lib.exr_finish(C.byref(ctxt))
