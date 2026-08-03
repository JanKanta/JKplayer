"""
EXR reader selection.

1) `exrcore` - Nuke's own OpenEXRCore DLL through ctypes.
   Handles EVERYTHING (DWAA, DWAB, PIZ, B44, PXR24...) and is ~2.3x faster.
   Verified: the result is BIT-IDENTICAL to our Python reader.
2) `exrread` - our pure Python reader (NONE/ZIPS/ZIP).
   A fallback in case the DLL is unavailable or changes its layout.

The choice is automatic; the reason for a fallback can be read from
last_note().
"""

import threading

from . import exrcore, exrread

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
    """(h,w,4) float16 scene-linear RGBA. Raises on failure."""
    global _core_disabled
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
