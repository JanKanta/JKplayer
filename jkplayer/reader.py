"""
Reader selection: which module turns a file into pixels.

EXR:

1) `exrcore` - Nuke's own OpenEXRCore DLL through ctypes.
   Handles EVERYTHING (DWAA, DWAB, PIZ, B44, PXR24...) and is ~2.3x faster.
   Verified: the result is BIT-IDENTICAL to our Python reader.
2) `exrread` - our pure Python reader (NONE/ZIPS/ZIP).
   A fallback in case the DLL is unavailable or changes its layout.

The choice is automatic; the reason for a fallback can be read from
last_note().
"""

import threading

from . import dpxread
from . import exrcore, exrread
from . import movread

_lock = threading.Lock()
_note = None            # latest information about what is going on
_core_disabled = False  # once core fails on the layout, stop trying it


def _set_note(text):
    global _note
    with _lock:
        _note = text


def last_note():
    return _note


def backend_name():
    if _core_disabled or not exrcore.available():
        return "python"
    return "nuke-dll"


def read_frame(path, layer=exrcore.ROOT_LAYER):
    """(h,w,4) float16 RGBA. Raises on failure.

    Scene-linear for EXR, which is what an EXR holds. A DPX holds CODE VALUES
    and comes back normalised to 0-1 instead - the display table folds the
    input space in (nukelut.display_lut), and Cineon there expects exactly
    that, so a log DPX needs the input space set and no code of its own.
    """
    global _core_disabled
    if movread.is_mov(path):
        return movread.read_rgba_half(path, None if layer in
                                      (exrcore.ROOT_LAYER, "") else layer)
    if dpxread.is_dpx(path):
        return dpxread.read_rgba_half(path, None if layer in
                                      (exrcore.ROOT_LAYER, "") else layer)
    if not _core_disabled and exrcore.available():
        try:
            # decodes straight into the final RGBA buffer and skips channels
            # we do not display -> measured 60 -> 116 fps on 4 threads
            return exrcore.read_rgba_half_direct(path, layer)
        except exrcore.ExrCoreError as exc:
            # layout/init failed -> stop trying it, go with Python
            if "layout" in str(exc) or "pipeline" in str(exc):
                _core_disabled = True
                _set_note("OpenEXRCore disabled (%s), using the Python reader"
                          % exc)
            else:
                _set_note("OpenEXRCore: %s" % exc)
        except Exception as exc:
            _core_disabled = True
            _set_note("OpenEXRCore error (%s: %s), using the Python reader"
                      % (type(exc).__name__, exc))
    return exrread.ExrFile(path).read_rgba_half(layer)


def probe(path):
    """File info plus whether we can read it. Never raises."""
    if movread.is_mov(path):
        try:
            return movread.probe(path)
        except Exception as exc:
            return {"supported": False, "reason": str(exc), "backend": "ffmpeg"}
    if dpxread.is_dpx(path):
        try:
            info = dpxread.probe(path)
        except Exception as exc:
            return {"supported": False, "reason": str(exc), "backend": "-"}
        info["backend"] = "dpx"
        info["supported"] = "reason" not in info
        try:
            info["channels"] = dpxread.channel_names(path)
        except Exception:
            info["channels"] = []
        return info
    info = exrread.probe(path)                 # cheap pure Python parser
    if info.get("supported"):
        info["backend"] = backend_name()
        return info
    # our reader cannot do it - can Nuke's DLL?
    if not _core_disabled and exrcore.available():
        try:
            core = exrcore.probe(path)
            if core.get("supported"):
                core["backend"] = "nuke-dll"
                core["channels"] = info.get("channels", [])
                # the DLL probe does not report it; square is the safe answer
                core.setdefault("pixel_aspect", 1.0)
                # we know the compression name even for the ones our Python
                # reader cannot handle - the panel should say "DWAA", not "id 8"
                cid = core.get("compression_id")
                core["compression"] = exrread.COMPRESSION_NAMES.get(
                    cid, "id %s" % cid)
                return core
        except Exception:
            pass
    info["backend"] = "-"
    return info


def layers_of(path):
    """Layers in the file, for the selector at the top of the panel.

    Ask the DLL first - only it sees into every part of a multipart file,
    where each AOV has its own header. The fallback is the cheap Python
    header parser; that one copes even with files it cannot decode afterwards
    (DWAA, PIZ...), because the header is always uncompressed - but it only
    sees the first part.
    """
    if movread.is_mov(path) or dpxread.is_dpx(path):
        return [exrcore.ROOT_LAYER]            # one image, no AOVs
    if not _core_disabled and exrcore.available():
        try:
            found = exrcore.file_layers(path)
            if found:
                return found
        except Exception:
            pass
    try:
        channels = exrread.channel_names(path)
    except Exception:
        channels = []
    found = exrcore.layers(channels)
    return found or [exrcore.ROOT_LAYER]


def metadata(path):
    """[(name, text)] - everything in the EXR header, for the META panel.

    Always the Python parser, whichever backend is decoding pixels. A header is
    uncompressed and sits at the front of the file, so there is no speed to win
    here - and going through the DLL would mean another set of ctypes struct
    layouts, which is exactly the part of exrcore that has broken across
    OpenEXR versions before.
    """
    if movread.is_mov(path):
        return movread.metadata(path)
    if dpxread.is_dpx(path):
        return dpxread.metadata(path)
    return exrread.metadata(path)
