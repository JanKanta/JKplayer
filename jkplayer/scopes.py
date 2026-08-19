"""
Computations for the histogram, vectorscope and waveform.

Deliberately separated from the drawing, so they can be measured and tested
without Qt.

SPEED: everything is computed from SUBSAMPLED data (about 150k pixels). It has
no visible effect on the shape of the histogram or on the vectorscope -
statistically it is more than enough - and instead of 2 Mpx only a fifth is
processed, so the scopes do not slow playback down. Converting a value to a
position goes through a table over the bits of a half float (the same trick as
the display uses), so no powers have to be evaluated.

WHAT THE SCOPES SHOW:

The HISTOGRAM and the WAVEFORM measure LEVEL, so they have a scene-linear axis
from 0 to 55 where the value 1.0 sits at HIST_SPLIT (see value_to_pos). The
data is linearised by the input transform first - otherwise a log recording
would show encoded values and the 1.0 boundary would mean nothing. Thanks to
that everything ABOVE 1 is visible: in the histogram right of the clipping
line, in the waveform above it. In QC mode they measure what is ON THE SCREEN
(0-255) instead - a QC visualisation has no scene-linear equivalent, so a 0-55
axis would lie.

The VECTORSCOPE always measures the FINISHED IMAGE, and deliberately so. It
shows colour, not level: the angle is the hue and the radius the saturation
relative to a full-scale signal - which only means anything in a BOUNDED
domain. A scene-linear axis up to 55 was tried and it lies: above 1.0 the axis
turns logarithmic, so the strong channel gets compressed while the weak ones
are still climbing the gamma curve. The difference between the channels (i.e.
the saturation) peaks at exactly 1.0 and then collapses - measured on a red
1:0.2:0.2 the channel gap went 92.6 at value 1.0, 43.7 at 4.0, 30.7 at 20.0.
The trace folds back towards the centre and a bright saturated colour reads as
LESS saturated than a mid-level one. Off the finished image it behaves the way
it should: the saturation grows up to the clipping point and an over-exposed
area lands in the centre, because on screen it really is white. As a bonus it
follows the whole colour path for free - the display transform, CC gain, gamma
and saturation are all already in that image.

All of them respect the channel selection - with R/G/B only that channel is
drawn, with A the alpha and with Y the luminance.
"""

import math

import numpy as np

# ---------------------------------------------------------------- histogram
HIST_BINS = 128
HIST_MAX = 55.0        # end of the axis (the same ceiling as the value map)
HIST_SPLIT = 0.70      # where the value 1.0 sits (= the clipping boundary)
HIST_ENCODE = 2.2      # gamma below 1.0, so the shadows are visible too

_HALF_VALUES = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)
# no inf/nan - multiplying those by gain would otherwise produce nan (inf * 0)
_HALF_SAFE = np.nan_to_num(_HALF_VALUES, nan=0.0, posinf=HIST_MAX, neginf=0.0)

# the order matches the channel selector in the panel (RGB, R, G, B, A, Luminance)
CHANNEL_KEYS = ("rgb", "r", "g", "b", "a", "y")

CURVE_COLORS = {"r": (255, 60, 60), "g": (60, 255, 60), "b": (80, 130, 255),
                "a": (215, 215, 215), "y": (215, 215, 215)}

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def channel_key(index):
    """Index from the channel selector -> key for the scopes."""
    try:
        return CHANNEL_KEYS[int(index)]
    except (IndexError, ValueError, TypeError):
        return "rgb"


def value_to_pos(v):
    """Scene-linear value -> position on the 0..1 axis (1.0 sits at HIST_SPLIT).

    Below 1 the axis is gamma-encoded (otherwise everything would collapse to
    the left), above 1 logarithmic (otherwise 1-55 would take almost the whole
    width). At the point 1.0 the two halves join exactly.
    """
    v = np.asarray(v, dtype=np.float32)
    v = np.nan_to_num(v, nan=0.0, posinf=HIST_MAX, neginf=0.0)
    low = HIST_SPLIT * np.power(np.clip(v, 0.0, 1.0), 1.0 / HIST_ENCODE)
    hi_t = np.log2(np.clip(v, 1.0, None)) / np.log2(HIST_MAX)
    high = HIST_SPLIT + (1.0 - HIST_SPLIT) * np.clip(hi_t, 0.0, 1.0)
    return np.where(v <= 1.0, low, high).astype(np.float32)


def _pos_to_bin(pos):
    return np.clip(pos * (HIST_BINS - 1) + 0.5, 0, HIST_BINS - 1)


def _srgb_encode(v):
    v = np.clip(v, 0.0, None)
    return np.where(v <= 0.0031308, v * 12.92,
                    1.055 * np.power(v, 1.0 / 2.4) - 0.055)


def _srgb_decode(e):
    e = np.clip(e, 0.0, None)
    return np.where(e <= 0.04045, e / 12.92,
                    np.power((e + 0.055) / 1.055, 2.4))


def apply_cc(v, gain=1.0, gamma=1.0):
    """Scene-linear values -> scene-linear values AFTER CC.

    Gain is a plain multiplier, that belongs in linear. Gamma, however, acts
    in the viewer above the display domain (see imageview.build_lut), so for
    the scopes to show the same thing as the image, the value is converted to
    sRGB, raised to the power and taken back to linear. The axis stays
    scene-linear and 1.0 still means clipping - gamma has a fixed point at 1.0,
    so it does not move the clipping boundary.

    ABOVE 1.0 gamma is not applied at all: in the image there is only white up
    there, which the slider does not touch, and the scope would otherwise
    change the "how far over it is" reading without anything moving on screen.
    Mathematically, raising a value above 1 to that power would even pull it in
    the OPPOSITE direction to values below 1.

    With an OCIO display the display domain is not exactly sRGB (gamma is then
    applied over the finished bytes), so this is an approximation - but the
    direction and size of the shift are right, and the scope axis is
    deliberately fixed anyway, independent of the view.
    """
    v = np.asarray(v, dtype=np.float32) * float(gain)
    if abs(float(gamma) - 1.0) <= 1e-6:
        return v                          # no round trip, keep it bit for bit
    low = _srgb_decode(np.power(_srgb_encode(np.clip(v, 0.0, 1.0)),
                                1.0 / max(float(gamma), 1e-3)))
    return np.where(v > 1.0, v, low).astype(np.float32)


_LUT_CACHE = {}


def bin_lut(gain=1.0, gamma=1.0):
    """Table half float bits -> bin number, CC already included.

    Gain and gamma are baked straight into the table, so the binning itself
    stays a single array lookup - just as fast as before, even though the
    histogram now reacts to CC. We keep a few of the latest tables so dragging
    a slider does not keep building new ones.
    """
    key = (round(float(gain), 4), round(float(gamma), 4))
    lut = _LUT_CACHE.get(key)
    if lut is None:
        lut = _pos_to_bin(value_to_pos(
            apply_cc(_HALF_SAFE, *key))).astype(np.uint8)
        if len(_LUT_CACHE) > 8:
            _LUT_CACHE.clear()
        _LUT_CACHE[key] = lut
    return lut


_POS_CACHE = {}


def pos_lut(gain=1.0, gamma=1.0):
    """Table half float bits -> position on a 0..255 axis, CC included.

    The same as bin_lut, only at a finer scale - the waveform computes its
    rows from it, so the histogram's 128 bins would be too coarse. Without the
    table, converting the whole crop through np.power costs several times the
    rest of the computation (measured 8.9 -> 2.3 ms on 2K).
    """
    key = (round(float(gain), 4), round(float(gamma), 4))
    lut = _POS_CACHE.get(key)
    if lut is None:
        lut = (value_to_pos(apply_cc(_HALF_SAFE, *key)) * 255.0).astype(np.float32)
        if len(_POS_CACHE) > 8:
            _POS_CACHE.clear()
        _POS_CACHE[key] = lut
    return lut


def subsample(arr, budget):
    """A step such that at most `budget` pixels remain (speed, see the header)."""
    h, w = arr.shape[0], arr.shape[1]
    step = 1
    while step < 16 and (h // step) * (w // step) > budget:
        step += 1
    return arr[::step, ::step]


def _as_rgb(sub):
    """Expands a grey image to three channels.

    Displaying a single channel keeps the result single-channel all the way to
    the QImage (see imageview), so an (h,w) array can arrive here. The
    expansion happens AFTER subsampling, when it is a few hundred thousand
    pixels instead of the whole crop.
    """
    return sub if sub.ndim == 3 else np.repeat(sub[:, :, None], 3, axis=2)


def _channel_indices(channel):
    """[(key, channel index)] - the fast path, when saturation changes nothing."""
    if channel == "rgb":
        return [("r", 0), ("g", 1), ("b", 2)]
    if channel in ("r", "g", "b"):
        return [(channel, {"r": 0, "g": 1, "b": 2}[channel])]
    if channel == "a":
        return [("a", 3)]
    return None                       # luminance cannot be taken from the bits


def _planes_float(sub, channel, sat_matrix):
    """[(key, 2D float32)] - when it has to be computed (saturation, luminance)."""
    rgb = sub[:, :, :3].astype(np.float32)
    if sat_matrix is not None:
        rgb = np.clip(rgb.reshape(-1, 3) @ sat_matrix, 0.0, None).reshape(rgb.shape)
    if channel == "rgb":
        return [("r", rgb[:, :, 0]), ("g", rgb[:, :, 1]), ("b", rgb[:, :, 2])]
    if channel in ("r", "g", "b"):
        return [(channel, rgb[:, :, {"r": 0, "g": 1, "b": 2}[channel]])]
    if channel == "a":
        return [("a", sub[:, :, 3].astype(np.float32))]
    return [("y", rgb @ _LUMA)]


def _encoded_rgb(arr, channel, gain=1.0, sat_matrix=None, linearize=None,
                 budget=120000, gamma=1.0):
    """Scene-linear half -> ((h,w,3) float32 0..255, which channels to draw).

    The conversion is the same as the histogram axis (value_to_pos), so the
    value 1.0 lands on HIST_SPLIT * 255 and HIST_MAX on 255. The waveform then
    works exactly as it does from a finished image, only its axis no longer
    ends at 1.0 - above it there is room up to 55.

    The order of operations matches the histogram: linearise the input,
    saturation from CC, gain, and only then the axis. The returned index list
    says which channels are worth drawing (for R/G/B only that one, for A and
    Y the result is grey in all three).
    """
    if arr is None or arr.ndim != 3 or arr.shape[2] < 4:
        return None, ()
    sub = subsample(arr, budget)
    if sub.size == 0:
        return None, ()
    if linearize is not None:
        sub = np.dstack([linearize(sub[:, :, :3]), sub[:, :, 3:4]])

    if sat_matrix is None and channel != "y":
        # The fast path through the table, exactly as in the histogram: ONE
        # contiguous copy and an array lookup instead of powers over the whole
        # crop.
        cont = np.ascontiguousarray(sub)
        bits = cont.view(np.uint16)
        lut = pos_lut(gain, gamma)
        if channel == "rgb":
            return np.ascontiguousarray(lut[bits[:, :, :3]]), (0, 1, 2)
        if channel == "a":
            # alpha is grey in all three channels -> the centre on the
            # vectorscope, a white trace in the waveform
            return np.repeat(lut[bits[:, :, 3]][:, :, None], 3, axis=2), (0, 1, 2)
        i = {"r": 0, "g": 1, "b": 2}[channel]
        out = np.zeros(cont.shape[:2] + (3,), np.float32)
        out[:, :, i] = lut[bits[:, :, i]]     # the others zero, so the trace
        return out, (i,)                      # stays in its own axis

    rgb = sub[:, :, :3].astype(np.float32)
    if sat_matrix is not None:
        rgb = np.clip(rgb.reshape(-1, 3) @ sat_matrix, 0.0,
                      None).reshape(rgb.shape)
    keep = (0, 1, 2)
    if channel in ("r", "g", "b"):
        i = {"r": 0, "g": 1, "b": 2}[channel]
        plane = rgb[:, :, i]
        rgb = np.zeros_like(rgb)
        rgb[:, :, i] = plane
        keep = (i,)
    elif channel == "a":
        rgb = np.repeat(sub[:, :, 3].astype(np.float32)[:, :, None], 3, axis=2)
    elif channel == "y":
        rgb = np.repeat((rgb @ _LUMA)[:, :, None], 3, axis=2)
    return value_to_pos(apply_cc(rgb, gain, gamma)) * 255.0, keep


def _pack(binned, over):
    """The common tail of both histograms: bincount, normalisation, colours.

    EVERY curve is normalised by its own peak. With a shared peak, RGB used to
    squash the weaker channels down to a few percent of the height (on the test
    plate green ended up at 5 %) - a channel then looked completely different
    in RGB than on its own. A histogram is read for WHERE the values sit (black
    point, clipping, a colour cast) anyway, not for how many there are.
    """
    curves = np.empty((len(binned), HIST_BINS), dtype=np.float32)
    colors = []
    for i, (key, b) in enumerate(binned):
        counts = np.bincount(b.ravel(), minlength=HIST_BINS)
        curves[i] = counts[:HIST_BINS]
        colors.append(CURVE_COLORS.get(key, (215, 215, 215)))
    peak = curves.max(axis=1, keepdims=True)
    np.divide(curves, peak, out=curves, where=peak > 0)
    return curves, colors, max(over) if over else 0.0


# Where to put a mark on the scene-linear axis, and which of them get a
# number written by them. Powers of two above 1.0 because the axis is
# logarithmic there, so they come out evenly spaced; 0.18 below it because mid
# grey is the one reference everybody exposes against.
#
# Not every mark is labelled - six numbers across a 230 px strip would collide
# and the strip would read as a row of digits rather than as a scale. The
# unlabelled ones are still drawn, so a value between two labels can be counted
# off instead of guessed.
LINEAR_MARKS = ((0.0, "0"), (0.18, "0.18"), (0.5, ""), (1.0, "1"),
                (2.0, ""), (4.0, "4"), (8.0, ""), (16.0, "16"),
                (32.0, ""), (HIST_MAX, "55"))


def _axis_marks(marks):
    """[(position 0..1, text)] - the values put through value_to_pos.

    Built from the same function the data goes through, so a mark sits exactly
    where that value lands and cannot drift away from the curve above it.
    """
    return [(float(value_to_pos(np.float32(v))), text) for v, text in marks]


# histogram axis: (position of the clipping line 0..1, labels [(position, text)])
AXIS_LINEAR = (HIST_SPLIT, _axis_marks(LINEAR_MARKS))
AXIS_DISPLAY = (1.0, [(0.0, "0"), (0.25, ""), (0.5, "128"),
                      (0.75, ""), (1.0, "255")])


def histogram_display(rgb8, channel="rgb"):
    """Histogram of the FINISHED IMAGE (uint8), axis 0-255.

    Used in QC mode: a grain or value map visualisation has no scene-linear
    equivalent, so it makes sense to measure exactly what is on screen.
    """
    if rgb8 is None or rgb8.ndim not in (2, 3):
        return None
    sub = _as_rgb(subsample(rgb8, 150000))
    if sub.size == 0:
        return None
    if channel == "rgb":
        planes = [("r", sub[:, :, 0]), ("g", sub[:, :, 1]), ("b", sub[:, :, 2])]
    elif channel in ("r", "g", "b"):
        planes = [(channel, sub[:, :, {"r": 0, "g": 1, "b": 2}[channel]])]
    else:
        # A and Y are already grey in the image, so one channel is enough
        planes = [(channel, sub[:, :, 0])]
    scale = (HIST_BINS - 1) / 255.0
    binned = [(k, (p.astype(np.float32) * scale + 0.5).astype(np.int32))
              for k, p in planes]
    over = [float((p >= 255).mean()) for _k, p in planes]
    curves, colors, clipped = _pack(binned, over)
    return curves, colors, clipped, AXIS_DISPLAY


def histogram(arr, channel="rgb", gain=1.0, sat_matrix=None, budget=150000,
              linearize=None, gamma=1.0):
    """(curves, colours, clipped_fraction, axis) or None.

    `curves` is (n, HIST_BINS) float32 normalised to 1, `colours` a list of RGB
    triples. How many curves appear is decided by the selected channel (RGB =
    three, otherwise one). The axis is scene-linear including the whole CC -
    gain, gamma and saturation.

    `linearize` is an optional half (h,w,3) -> half (h,w,3) function that
    straightens the input space into linear (e.g. when the file holds LogC).
    Without it the histogram would measure encoded values on log material and
    the clipping line at 1.0 would mean nothing.
    """
    if arr is None or arr.ndim != 3 or arr.shape[2] < 4:
        return None
    sub = subsample(arr, budget)
    if sub.size == 0:
        return None
    if linearize is not None:
        sub = np.dstack([linearize(sub[:, :, :3]), sub[:, :, 3:4]])

    idxs = None if (sat_matrix is not None or channel == "y") else \
        _channel_indices(channel)
    if idxs is not None:                        # the fast path through the table
        # ONE contiguous copy, not one per channel. The subsampled view is
        # scattered across the whole frame, so every pass costs cache misses -
        # three passes instead of one took 2.81 ms instead of 1.68 ms on 6K.
        # The result is bit-identical.
        cont = np.ascontiguousarray(sub)
        bits = cont.view(np.uint16)
        lut = bin_lut(gain, gamma)
        binned = [(key, lut[bits[:, :, i]]) for key, i in idxs]
        # Gamma does not move the clipping boundary - it acts above the display
        # domain and is fixed at 1.0 (see apply_cc), so a gain threshold is enough.
        limit = float(np.float16(1.0 / max(gain, 1e-6)))
        over = [float((cont[:, :, i].astype(np.float32) > limit).mean())
                for _key, i in idxs]
    else:
        planes = _planes_float(sub, channel, sat_matrix)
        g = float(gain)
        binned = [(key, _pos_to_bin(value_to_pos(
            apply_cc(p, g, gamma))).astype(np.int32)) for key, p in planes]
        over = [float((p * g > 1.0).mean()) for _, p in planes]

    curves, colors, clipped = _pack(binned, over)
    return curves, colors, clipped, AXIS_LINEAR


# -------------------------------------------------------------- vectorscope
VS_SIZE = 128          # grid resolution (VS_SIZE x VS_SIZE bins)

# How much of the radius the full saturation of an 8bit signal takes. Chosen so
# that the OUTERMOST 100 % target still fits inside the circle: green and
# magenta have the largest chrominance magnitude of all the primaries (152
# against the nominal 128), so at 0.90 their boxes landed at 1.07 of the
# radius, i.e. outside the graticule. At 0.80 they sit at 0.95, just inside.
VS_GAIN = 0.80

# Rec.709 conversion to chrominance
_KR, _KG, _KB = 0.2126, 0.7152, 0.0722
_CB_D = 2.0 * (1.0 - _KB)      # 1.8556
_CR_D = 2.0 * (1.0 - _KR)      # 1.5748

# How much the trace is lifted. The density is already log-normalised, but
# ordinary footage still leaves most of the plate barely glowing - the colour
# it carries only reads in the few brightest bins. A gamma under 1 pulls the
# middle up, so the hue is legible across the whole trace instead of just at
# its core.
VS_TRACE_LIFT = 0.65

# Base luminance the hue plate is built around. Lower = more saturated hues,
# because the same chrominance is a bigger share of a darker colour. Measured
# mean saturation of the plate from the centre outwards:
#     160 (the original)  0.22 / 0.52 / 0.78 / 0.87
#     110 (now)           0.30 / 0.68 / 0.88 / 0.93
#      70                 0.44 / 0.84 / 0.95 / 0.98
# 70 is vivid even right next to the centre, which throws away the reading of
# how far from neutral something is. 110 keeps that gradient and the colours
# are still lively.
VS_PLATE_LUMA = 110.0

# angles of the primaries on the vectorscope (for the graticule)
VS_TARGETS = ("R", "Yl", "G", "Cy", "B", "Mg")


def _target_positions(level=0.75):
    """Where a primary at `level` saturation sits - for drawing the graticule.

    A broadcast vectorscope has two rings of them: the 75 % colour bars, which
    is what material is normally graded against, and 100 % further out as the
    absolute limit. VS_TARGETS is in circular order, so joining the points in
    that sequence draws the hexagon between them.
    """
    base = {"R": (255, 0, 0), "Yl": (255, 255, 0), "G": (0, 255, 0),
            "Cy": (0, 255, 255), "B": (0, 0, 255), "Mg": (255, 0, 255)}
    out = {}
    for name in VS_TARGETS:
        r, g, b = [c * level for c in base[name]]
        y = _KR * r + _KG * g + _KB * b
        cb = (b - y) / _CB_D
        cr = (r - y) / _CR_D
        out[name] = (float(cb / 128.0 * VS_GAIN), float(cr / 128.0 * VS_GAIN))
    return out


TARGET_POS = _target_positions(0.75)        # the 75 % bars, and the hexagon
TARGET_POS_100 = _target_positions(1.0)     # the absolute limit


def _vectorscope_grid(rgb, size):
    """The chrominance grid from (h,w,3) float 0..255."""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    y = _KR * r + _KG * g + _KB * b
    cb = (b - y) / _CB_D
    cr = (r - y) / _CR_D

    half = size * 0.5
    scale = half * VS_GAIN / 128.0
    dx = cb * scale
    dy = cr * scale

    # Clip RADIALLY to the circle, not per axis. Clipping x and y separately
    # piles everything over-range into the CORNERS of the square, i.e. outside
    # the round graticule, where it means nothing - and with the multiplier
    # turned up the whole trace ended up out there. This way it stops on the
    # rim, the same as a broadcast vectorscope (and DaVinci).
    lim = half - 1.0
    r = np.hypot(dx, dy)
    over = r > lim
    if over.any():
        shrink = np.where(over, lim / np.maximum(r, 1e-6), 1.0)
        dx = dx * shrink
        dy = dy * shrink

    x = np.clip(half + dx, 0, size - 1).astype(np.int32)
    # vertically flipped: a larger Cr means higher up, i.e. a smaller row index
    yy = np.clip(half - dy, 0, size - 1).astype(np.int32)

    flat = np.bincount((yy * size + x).ravel(), minlength=size * size)
    grid = flat[:size * size].astype(np.float32).reshape(size, size)
    # logarithm - otherwise the neutral centre would outshine everything else
    peak = grid.max()
    if peak > 0:
        grid = np.log1p(grid) / np.log1p(peak)
        grid **= VS_TRACE_LIFT          # and a lift, see VS_TRACE_LIFT
    return grid


def vectorscope(rgb8, channel="rgb", budget=120000, size=VS_SIZE):
    """(size, size) float32 0..1 - pixel density by colour.

    Computed from the FINISHED IMAGE (uint8). Cb horizontally, Cr vertically
    (upwards), the centre is neutral grey.

With a single channel the trace is a STRAIGHT LINE out of the centre, and
    that is deliberate. The image is already grey there (the channel copied
    into RGB), so it is put back into its own axis: what is left is not a hue -
    an isolated channel has none - but the LENGTH of the line, i.e. how much of
    that channel is in the shot. A pixel at full value lands exactly on that
    primary's 100 % target, so the graticule can be read against it.

    Alpha and luminance stay grey, so they draw a dot in the centre - neither
    of them is a colour and neither has any chrominance.

    Weighting the trace by the channel instead of isolating it was tried and
    measured: the log normalisation swallows it, so the picture moved by 1.5 of
    255 on average and only a tenth of the pixels changed at all. Truthful, but
    you could not see it.

    There is no multiplier. The scale is fixed, the same as on a broadcast
    vectorscope: full saturation of an 8bit signal reaches VS_GAIN of the
    radius, so the 75 % and 100 % targets mean something absolute and the trace
    can be read against them. A stretched trace has nothing left to compare to.
    """
    if rgb8 is None or rgb8.ndim not in (2, 3):
        return None
    sub = _as_rgb(subsample(rgb8, budget))
    if sub.size == 0:
        return None
    rgb = sub[:, :, :3].astype(np.float32)
    if channel in ("r", "g", "b"):
        # the image is already grey - put it back into its own axis, so the
        # line points at that primary and its length is the channel's amount
        keep = np.zeros_like(rgb)
        keep[:, :, {"r": 0, "g": 1, "b": 2}[channel]] = rgb[:, :, 0]
        rgb = keep
    return _vectorscope_grid(rgb, size)


def vectorscope_point(rgb):
    """(x, y) in -1..1 for ONE display-space pixel, or None.

    The same arithmetic _vectorscope_grid does, deliberately sharing the
    constants: the probe dot has to land exactly on the trace that pixel made,
    and a second copy of the maths would drift away from it the first time
    either was touched.
    """
    if rgb is None or len(rgb) < 3:
        return None
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    y = _KR * r + _KG * g + _KB * b
    dx = ((b - y) / _CB_D) * (VS_GAIN / 128.0)
    dy = ((r - y) / _CR_D) * (VS_GAIN / 128.0)
    rad = math.hypot(dx, dy)
    if rad > 1.0:                  # radial clip, the same rule as the grid
        dx, dy = dx / rad, dy / rad
    return dx, dy


def target_colors():
    """{target name: (r,g,b)} - each in its own colour, so it reads instantly."""
    base = {"R": (255, 60, 60), "Yl": (235, 235, 70), "G": (70, 235, 90),
            "Cy": (70, 220, 235), "B": (90, 120, 255), "Mg": (235, 80, 220)}
    return {name: base[name] for name in VS_TARGETS}


TARGET_COLORS = target_colors()


def hue_plate(size=VS_SIZE):
    """(size,size,3) uint8 - what colour each place on the vectorscope has.

    Used as a backdrop that gets multiplied by the density: the trace then
    carries the colour of what it represents, instead of a uniform green.

    EVERY hue is normalised to full brightness, not just the ones that would
    overflow. Built around a mid grey the plate came out washed out and the
    trace was a muddy grey-brown; this way the hue is at its most vivid and
    the density alone decides how bright it is. The centre stays neutral -
    there all three channels are equal, so it normalises to white.
    """
    half = size * 0.5
    scale = 128.0 / (half * VS_GAIN)
    j, i = np.meshgrid(np.arange(size, dtype=np.float32),
                       np.arange(size, dtype=np.float32))
    cb = (j - half) * scale
    cr = (half - i) * scale
    y = np.full((size, size), VS_PLATE_LUMA, dtype=np.float32)
    r = y + _CR_D * cr
    b = y + _CB_D * cb
    g = (y - _KR * r - _KB * b) / _KG
    out = np.clip(np.stack([r, g, b], axis=2), 0.0, None)
    mx = out.max(axis=2, keepdims=True)
    out = out * (255.0 / np.maximum(mx, 1e-6))
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Waveform - values along the columns of the image
# ---------------------------------------------------------------------------
WF_W = 256             # how many image columns collapse into one
WF_H = 128             # steps on the vertical axis (0 at the bottom, the top
                       # of the axis at the top)

# Waveform axis: (position of the 1.0 line or None, labels [(position, text)]).
# Positions are measured FROM THE BOTTOM, 0 = value 0, 1 = top of the axis.
WF_AXIS_LINEAR = (HIST_SPLIT, _axis_marks(LINEAR_MARKS))
WF_AXIS_DISPLAY = (None, [(0.0, "0"), (0.25, ""), (0.5, "128"),
                          (0.75, ""), (1.0, "255")])


def _waveform_grid(rgb, keep, width, height, top):
    """The computation shared by both waveforms, from (h,w,3) float 0..`top`."""
    h, w = rgb.shape[0], rgb.shape[1]
    if h == 0 or w == 0:
        return None

    # image column -> waveform column (one bin can take several columns)
    cols = np.minimum((np.arange(w, dtype=np.int64) * width) // max(1, w),
                      width - 1)
    cols = np.repeat(cols[None, :], h, axis=0)

    out = np.zeros((height, width, 3), dtype=np.float32)
    scale = (height - 1) / float(top)
    for c in keep:
        rows = (height - 1) - np.clip(rgb[:, :, c] * scale, 0,
                                      height - 1).astype(np.int64)
        flat = np.bincount((rows * width + cols).ravel(),
                           minlength=height * width)
        out[:, :, c] = flat[:height * width].reshape(height, width)

    # Logarithm as in the vectorscope - otherwise a flat sky outshines
    # everything. EVERY channel is normalised separately, for the same reason
    # as in the histogram (see _pack): otherwise the strongest channel damps
    # the others so much that they look different in RGB than on their own.
    out = np.log1p(out)
    peak = out.max(axis=(0, 1), keepdims=True)
    np.divide(out, peak, out=out, where=peak > 0)
    return out


def waveform(rgb8, channel="rgb", budget=200000, width=WF_W, height=WF_H):
    """(height, width, 3) float32 0..1 - density of values along the columns.

    A classic waveform: horizontally the image columns in their original order,
    vertically the value (0 at the bottom, 255 at the top) and the brightness
    says how many pixels landed there. Each channel has its own plane, so they
    are drawn over each other like a parade - where they overlap, white comes out.

    Computed from the FINISHED IMAGE (uint8) - used in QC mode, the axis is 0-255.
    """
    if rgb8 is None or rgb8.ndim not in (2, 3):
        return None
    sub = _as_rgb(subsample(rgb8, budget))
    if sub.size == 0:
        return None

    keep = (0, 1, 2)
    if channel in ("r", "g", "b"):
        # the image is already grey (the channel copied into RGB) - draw only
        # into its own axis, so the trace carries that channel's colour
        keep = ({"r": 0, "g": 1, "b": 2}[channel],)
    return _waveform_grid(sub[:, :, :3].astype(np.float32), keep,
                          width, height, 255.0)


def waveform_linear(arr, channel="rgb", gain=1.0, sat_matrix=None,
                    budget=200000, width=WF_W, height=WF_H, linearize=None,
                    gamma=1.0):
    """Waveform from SCENE-LINEAR data, with room for values above 1.

    The vertical axis is the same as the histogram's (see _encoded_rgb): the
    value 1.0 sits at HIST_SPLIT, above it there is only what clips, up to
    HIST_MAX. The line at 1.0 is drawn by WaveformCanvas from WF_AXIS_LINEAR.
    """
    rgb, keep = _encoded_rgb(arr, channel, gain, sat_matrix, linearize,
                             budget, gamma)
    if rgb is None:
        return None
    return _waveform_grid(rgb, keep, width, height, 255.0)
