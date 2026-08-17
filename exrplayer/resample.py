"""
Fitting one frame onto another's size.

Only ever used to make a COMPARISON possible. The difference checks subtract
two frames pixel by pixel, so a 4K render against a 6K plate has nothing to
subtract and the check simply refused. Everything else in the player shows a
plate at its own size and never comes through here.

THIS IS NOT NUKE'S REFORMAT and does not try to be. Nuke has ten filters and a
box of modes; matching any of them to the last bit is a project of its own, and
a check that quietly resamples with a filter nobody named is a check that
reports its own ringing as if it were in the plate. So there is one filter, it
is written down, and the panel says out loud when it has been used.

No Qt and no Nuke, so it can be tested from the console.
"""

import threading

import numpy as np

try:
    from concurrent.futures import ThreadPoolExecutor
    # numpy releases the GIL on the big array operations, so these really do
    # run at the same time. Made once and kept - starting four threads per
    # repaint would cost more than the work.
    _pool = ThreadPoolExecutor(max_workers=4)
except Exception:                 # pragma: no cover - very old interpreters
    _pool = None

# An exact integer shrink is done as a block average instead: for a 2x or 3x
# reduction that is both cheaper and BETTER than sampling four neighbours -
# every source pixel is counted exactly once, so nothing aliases. It is also
# the same thing the display path already does when it zooms out.
BOX_MIN = 2

FILTER_NAME = "bilinear"          # what the status line calls it

_weights = {}                     # (n_in, n_out) -> (i0, i1, w)
_lock = threading.Lock()

# The one result kept back (see fit). One, not a dictionary: two frames are
# compared at a time, and a cache that grew would hold whole 6K frames alive
# behind the frame cache's back and quietly eat the budget it was given.
_last = (None, None)              # (key, array)
_last_src = None                  # the source, kept alive so its id stays its own


def _set_last(key, out, src):
    """Caller holds _lock."""
    global _last, _last_src
    _last = (key, out)
    _last_src = src


def _axis(n_in, n_out):
    """Source indices and blend for one axis, cached.

    Sampled at PIXEL CENTRES ((i + 0.5) / n), which is what keeps the picture
    where it was: sampling at the edges shifts the whole image by half a pixel
    per axis, and on a difference check a half-pixel shift lights up every edge
    in the frame.
    """
    key = (int(n_in), int(n_out))
    with _lock:
        got = _weights.get(key)
        if got is not None:
            return got
    pos = (np.arange(n_out, dtype=np.float32) + 0.5) * (float(n_in) / n_out) - 0.5
    np.clip(pos, 0.0, n_in - 1.0, out=pos)
    i0 = pos.astype(np.int32)
    i1 = np.minimum(i0 + 1, n_in - 1)
    out = (i0, i1, (pos - i0).astype(np.float32))
    with _lock:
        _weights[key] = out
    return out


def _box_factors(src, oh, ow):
    """(fy, fx) when an exact block average applies, else None."""
    ih, iw = src.shape[0], src.shape[1]
    if oh <= 0 or ow <= 0 or ih % oh or iw % ow:
        return None
    fy, fx = ih // oh, iw // ow
    if fy < BOX_MIN and fx < BOX_MIN:
        return None
    return fy, fx


def _box(src, oh, ow, fy, fx):
    """One band of an exact integer shrink, by averaging blocks."""
    block = src.astype(np.float32).reshape(oh, fy, ow, fx, src.shape[2])
    return block.mean(axis=(1, 3), dtype=np.float32).astype(src.dtype)


def _bilinear(src, oh, ow, r0, r1, rw, c0, c1, cw):
    """One band of the output. Rows first, then columns - separable.

    Two passes over the picture rather than a four-neighbour gather per output
    pixel, which in numpy is several times cheaper. Everything here is memory
    bandwidth, which is why it is worth splitting over threads at all.
    """
    tmp = src[r0].astype(np.float32)
    tmp *= (1.0 - rw)[:, None, None]
    tmp += src[r1].astype(np.float32) * rw[:, None, None]

    out = tmp[:, c0]
    out *= (1.0 - cw)[None, :, None]
    out += tmp[:, c1] * cw[None, :, None]
    return out.astype(src.dtype)


# Rows per thread band. Below this the split costs more than it saves.
BAND_MIN = 96
THREADS = 4


def _fit_uncached(src, oh, ow):
    """Picks the kernel, then runs it over bands of output rows.

    The banding wraps BOTH kernels. It used to wrap only the bilinear one, and
    the box path - the better of the two - then measured slower than the path
    it was there to beat.
    """
    ih, iw = src.shape[0], src.shape[1]
    factors = _box_factors(src, oh, ow)
    if factors is not None:
        fy, fx = factors

        def band(a, b):
            return _box(src[a * fy:b * fy], b - a, ow, fy, fx)
    else:
        r0, r1, rw = _axis(ih, oh)
        c0, c1, cw = _axis(iw, ow)

        def band(a, b):
            return _bilinear(src, b - a, ow, r0[a:b], r1[a:b], rw[a:b],
                             c0, c1, cw)

    bands = max(1, min(THREADS, oh // BAND_MIN))
    if bands == 1 or _pool is None:
        return band(0, oh)

    step = (oh + bands - 1) // bands
    cuts = [(i, min(oh, i + step)) for i in range(0, oh, step)]
    return np.concatenate(list(_pool.map(lambda c: band(*c), cuts)), axis=0)


def fit(src, shape):
    """`src` resampled to (h, w) of `shape`. The SAME array when it already fits.

    Returns the input untouched on an exact match, so the common case - two
    plates of one size - costs nothing at all.

    The last result is REMEMBERED. A held frame is re-rendered on every pan,
    zoom and slider move, and resampling a 6K plate each time is a fifth of a
    second thrown away every repaint. Keyed on the array object itself, so a
    new frame during playback misses it and nothing stale is ever returned.
    """
    if src is None:
        return None
    oh, ow = int(shape[0]), int(shape[1])
    ih, iw = src.shape[0], src.shape[1]
    if (ih, iw) == (oh, ow):
        return src
    if oh <= 0 or ow <= 0 or ih <= 0 or iw <= 0:
        return src

    key = (id(src), ih, iw, oh, ow)
    with _lock:
        held_key, held = _last
        if held_key == key and held is not None:
            return held

    out = _fit_uncached(src, oh, ow)
    with _lock:
        # `src` is kept alive alongside it on purpose: id() is only unique
        # while the object lives, and letting it be collected would let a new
        # array land on the same address and be handed this result.
        _set_last(key, out, src)
    return out


def label(src_shape, dst_shape):
    """What the status line says about it, or "" when nothing was done."""
    if src_shape is None or dst_shape is None:
        return ""
    if tuple(src_shape[:2]) == tuple(dst_shape[:2]):
        return ""
    return "B rescaled %dx%d to %dx%d (%s)" % (
        src_shape[1], src_shape[0], dst_shape[1], dst_shape[0], FILTER_NAME)
