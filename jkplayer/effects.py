"""
QC effects.

The input is always a SCENE-LINEAR half crop (h,w,4) from the cache plus a LUT
for the conversion into display. The output is uint8 RGB (h,w,3), i.e. a
finished image.

Why some effects work in the display domain and others in linear:
  * Grain / Saturation / Canvas are VISUAL checks - they act on what a person
    sees, i.e. after the display transform (same as in v1).
  * ValueMap MEASURES values - it has to work on scene-linear float, otherwise
    it would make no sense (8 bits has neither negative values nor values
    above 1).
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import nukelut

try:
    from scipy import ndimage as _ndi
except Exception:
    _ndi = None

NONE = "none"
LOG = "log"
DIFF = "diff"
HPDIFF = "hpdiff"
GRAIN = "grain"
BANDPASS = "bandpass"
SAT = "sat"
CANVAS = "canvas"
VALUEMAP = "valuemap"
TEMPORAL = "temporal"

# The QC menu of ONE window: detail and its changes (grain, high-pass,
# temporal), then colour and exposure (saturation, value map). Canvas is last
# on purpose - it is not a computation over the image like the others, just a
# shifted crop, and it is the only one displayed through the ordinary display
# transform.
#
# The two-input comparisons are NOT here. They compare A against B, which is
# what Overlay mode is for, so they live in its own selector (OVERLAY_MODES) -
# offering them per window meant picking "difference" on a window and then
# wondering which two things were being differenced.
#
# NONE is NOT in the list either: QC is switched off by its toggle, not by an
# entry in the menu. It stays as a value though - the panel sends it to a
# window when QC is off.
ORDER = [LOG, GRAIN, BANDPASS, TEMPORAL, SAT, VALUEMAP, CANVAS]

# The log curves offered by the log view. Taken from the same table the input
# transforms come from, so the shot is shown in exactly the curve it would be
# read back with - and only the LOG ones: putting sRGB in this list would make
# a "log view" that is not a log view.
LOG_CURVES = ["Cineon", "AlexaV3LogC", "SLog3", "Log3G10"]

# What the Overlay strip offers. NONE first - that is the plain dissolve.
OVERLAY_MODES = [NONE, DIFF, HPDIFF]

LABELS = {NONE: "No effect", LOG: "Log view", DIFF: "Difference",
          GRAIN: "Grain check",
          BANDPASS: "High-pass", SAT: "Saturation check",
          CANVAS: "Canvas check", VALUEMAP: "Value map",
          TEMPORAL: "Temporal check",
          HPDIFF: "High-pass difference"}

# Labels for the Overlay selector - shorter, the strip is narrow, and "off"
# reads better than "no effect" next to a dissolve.
OVERLAY_LABELS = {NONE: "Off", DIFF: "Difference", HPDIFF: "High-pass diff"}

# The modes that need BOTH inputs at once. Because of them the panel decodes
# even the window that is not currently visible (see PlayerPanel._live_slots).
NEEDS_BOTH = (DIFF, HPDIFF)

# Difference: the colours the overlay fills changed places with. All of them
# dark and low in saturation - unmistakable in the image, but they do not pull
# the eye the way a pure colour would and do not read as an error. Which one
# suits depends on the plate: green shows up on a blue background, blue on green.
# DiMatte: every channel is drawn in ITS OWN colour - R red, G green, B blue,
# alpha white. No damping: when I look at a matte I want to see at a glance
# that green is green.
MATTE_COLORS = ((255, 48, 48), (48, 255, 72), (64, 110, 255), (240, 240, 240))

DIFF_COLORS = [(38, 46, 68), (38, 60, 44), (68, 42, 42)]
DIFF_COLOR_NAMES = ["Blue-grey", "Green-grey", "Red-grey"]
DIFF_COLOR = DIFF_COLORS[0]
DIFF_OVERLAY, DIFF_PLAIN = 0, 1
DIFF_MODES = ["Overlay", "Difference"]
DIFF_PLAIN_GAIN = 8.0        # so that an intensity of 1.00 on the plain
                             # difference matches how it looked with the
                             # original gain

# A marker in PARAMS: parameters tagged like this are not drawn as separate
# sliders but as ONE slider with several handles (see overlay._BandSlider).
BANDS = "bands"

LUMA_R, LUMA_G, LUMA_B = 0.2126, 0.7152, 0.0722       # Rec.709

# Effects that blur - they are several times more expensive than the rest and
# therefore have their own, stricter pixel ceiling (see ImageView._effect_step).
BLUR_HEAVY = (GRAIN, BANDPASS)

# Temporal: multiplier on the difference against the previous frame (so small
# changes show up too)
TEMPORAL_GAIN = 8.0
# CAREFUL: no fixed threshold for "almost identical" frames. Measurements
# showed that on a static shot neighbouring frames differ by 0.0001, on motion
# by hundredths - any fixed threshold would cry wolf on one piece of material
# and stay silent on another. Only a BITWISE match is reported, which is
# unambiguous proof of a duplicate; the numeric difference is printed as
# information and you judge it yourself.

# Grain checker. The way grain is pulled in a comp: a high-pass in SCENE-LINEAR
# (subtract a blurred copy - the "background"), take its magnitude like a
# |A-B| difference merge, then let the DISPLAY CURVE show it. The curve is what
# makes it read: it lifts the dark grain (steep near black) and compresses the
# bright edges (flat near white), so the grain is even and fine and the edges
# do not blow out. Doing the high-pass after the display transform (the old
# way) and then a linear gain gets this backwards and the edges clip - see
# _grain. Defaults chosen against a real reference plate.
GRAIN_CONTRAST = 20.0         # exposure applied BEFORE the display curve
GRAIN_SIZE = 1.0              # background blur sigma, in scene-linear pixels;
                              # small = only the finest grain, larger = coarser
GRAIN_FINE = 0.0              # single-pixel emphasis via the [[2,2,2],[2,-15,2],
                              # [2,2,2]] matrix; 0 = plain high-pass


# ---------------------------------------------------------------------------
# Settings of the individual QC modes.
# (key, label, min, max, default, number of decimals)
# ---------------------------------------------------------------------------
PARAMS = {
    LOG: [
        # a seventh element = a menu instead of a slider (see overlay.EffectPanel)
        ("curve", "Log curve", 0, len(LOG_CURVES) - 1, 0, 0, LOG_CURVES),
        ("exposure", "Exposure (stops)", -6.0, 6.0, 0.0, 2),
        ("black", "Black level", 0.0, 0.5, 0.0, 3),
    ],
    DIFF: [
        # a seventh element = a menu instead of a slider (see overlay.EffectPanel)
        ("mode", "Display", 0, len(DIFF_MODES) - 1, DIFF_OVERLAY, 0,
         DIFF_MODES),
        ("color", "Overlay colour", 0, len(DIFF_COLORS) - 1, 0, 0,
         DIFF_COLOR_NAMES),
        ("threshold", "Threshold", 0.0, 0.2, 0.01, 3),
        ("intensity", "Intensity", 0.0, 8.0, 1.0, 2),
    ],
    GRAIN: [
        ("contrast", "Grain contrast", 1.0, 40.0, GRAIN_CONTRAST, 1),
        ("size", "Grain / background size", 0.3, 6.0, GRAIN_SIZE, 1),
        ("fine", "Fineness", 0.0, 1.0, GRAIN_FINE, 2),
    ],
    BANDPASS: [
        # the "to" end stops at 24 px on purpose: the cost of blurring grows
        # with the radius (measured 8 px = 31 ms, 24 px = 75 ms, 48 px =
        # 190 ms) and a wider band shows lighting rather than texture anyway.
        # Sigma is in SCREEN pixels, so when zoomed out it covers a much larger
        # piece of the image.
        ("fine", "From detail (px)", 0.0, 8.0, 0.0, 1),
        ("coarse", "To detail (px)", 1.0, 24.0, 8.0, 1),
        ("gain", "Gain", 1.0, 40.0, 8.0, 0),
        ("mid", "Background level", 0.0, 200.0, 128.0, 0),
        # Colour in a high-pass is mostly chroma noise fighting the texture you
        # are trying to read. Pulled all the way up this is the check in
        # luminance; part way it just calms the colour speckle down without
        # losing which channel a difference sits in.
        ("desat", "Desaturate", 0.0, 1.0, 0.0, 2),
    ],
    HPDIFF: [
        # The same band as the high-pass, applied to BOTH inputs, so the two
        # are directly comparable - that is the whole point of the mode.
        ("fine", "From detail (px)", 0.0, 8.0, 0.0, 1),
        ("coarse", "To detail (px)", 1.0, 24.0, 8.0, 1),
        ("gain", "Gain", 1.0, 60.0, 16.0, 0),
        ("desat", "Desaturate", 0.0, 1.0, 0.0, 2),
    ],
    SAT: [
        # Default 2.0, not 1.0: levelling the brightness COMPRESSES the colour
        # (measured on a real plate 77 -> 49), so at 1.0 you would see less
        # colour than in the original. At 2.0 it is 98 and only 0.3 % of the
        # pixels clip.
        ("boost", "Saturation gain", 1.0, 6.0, 2.0, 1),
        ("level", "Background level", 40.0, 200.0, 128.0, 0),
    ],
    CANVAS: [
        ("shift_x", "Shift X (%)", 0.0, 100.0, 50.0, 0),
        ("shift_y", "Shift Y (%)", 0.0, 100.0, 50.0, 0),
    ],
    VALUEMAP: [
        # Three band boundaries on ONE multi-handle slider (seventh element
        # BANDS). The four grey steps spread out below the first boundary, the
        # other two split the colour bands above it.
        ("b1", "Greys up to", 0.05, 4.0, 1.0, 2, BANDS),
        ("b2", "Blue up to", 1.0, 60.0, 20.0, 1, BANDS),
        ("b3", "Green up to", 5.0, 200.0, 55.0, 1, BANDS),
    ],
    TEMPORAL: [
        ("gain", "Difference gain", 1.0, 40.0, TEMPORAL_GAIN, 0),
        ("offset", "Compare with frame -N", 1.0, 8.0, 1.0, 0),
    ],
}

DESCRIPTION = {
    LOG: ("The shot read back through a LOG curve, whatever the monitor is\n"
          "set to. The toe and the shoulder are stretched out, so what a\n"
          "display transform has already rolled away is visible again.\n"
          "Curve    = which log encoding to read it in.\n"
          "Exposure = stops, for pushing a dark or bright plate into the\n"
          "           part of the curve you want to look at.\n"
          "Black    = lifts the floor away, so the toe is not mistaken for\n"
          "           detail.\n"
          "Look for: crushed blacks, clipped highlights, banding in a\n"
          "gradient, a grade already baked into a plate that should be raw."),
    DIFF: ("Difference between input A and B (not between frames - that is\n"
           "Temporal).\n"
           "Overlay     = the image stays, changed places are covered with a\n"
           "              dark blue-grey; you see WHERE it was touched.\n"
           "Difference  = the classic |A-B| amplified; you also see HOW much.\n"
           "Pick the overlay colour to suit the plate - green shows up on a\n"
           "blue background, blue on green.\n"
           "The threshold decides what counts as a change - it filters out\n"
           "codec noise.\n"
           "Intensity: the strength of the colour in overlay mode, the gain on\n"
           "the difference in the plain mode."),
    GRAIN: ("Pulls the grain the way a comp does: a scene-linear high-pass (a\n"
            "blurred copy subtracted), its magnitude like a |A-B| merge, shown\n"
            "through the display curve - which lifts the fine grain and\n"
            "compresses the edges, so the grain reads even and fine on near-\n"
            "black and the edges stay thin.\n"
            "Contrast = grain brightness; Size = radius of the background\n"
            "subtracted (small = only the finest grain, larger = coarser too);\n"
            "Fineness = single-pixel grain over 2-3px softness (the matrix).\n"
            "Look for: missing or doubled grain, areas with no grain\n"
            "(repainted / blurred), a jump in graininess between shots."),
    HPDIFF: ("The high-pass of A against the high-pass of B - both put through\n"
             "the same band, then subtracted.\n"
             "BLACK = the same detail in both. Anything lit up is texture that\n"
             "differs, with the overall level taken out first, so a grade\n"
             "between the two does not drown the answer the way a plain\n"
             "difference does.\n"
             "Look for: paint fixes, re-grained patches, softened areas,\n"
             "a swapped plate, detail that went missing in a comp."),
    BANDPASS: ("Keeps only detail in the given size range and subtracts the rest.\n"
               "A narrow band = you see just that one layer of detail.\n"
               "Look for: soft spots after a paint fix, clone stamp prints,\n"
               "plate seams, a change of texture, added detail without its own\n"
               "frequency."),
    SAT: ("Levels the brightness (HSV Value -> mid grey), only the colour stays.\n"
          "The channel that is always the largest in the shot comes out flat -\n"
          "it is the one carrying that levelled brightness. The information is\n"
          "then in the other two.\n"
          "Look for: colour casts where neutral is expected, a colour that\n"
          "does not match between elements, exaggerated saturation."),
    CANVAS: ("Swaps parts of the image, so the original edges meet in the middle.\n"
             "Look for: light/dark fringes at the edges, joins that do not\n"
             "match, a badly cleaned plate edge."),
    VALUEMAP: ("False colours from the scene-linear values.\n"
               "red = negative (an error!), four grey steps up to the first\n"
               "boundary, then blue, green and orange (HDR).\n"
               "You move the boundaries with the three-handle slider - its\n"
               "track carries exactly the colours the bands have in the image."),
    TEMPORAL: ("Difference against the previous frame.\n"
               "A black area = the frame did not change at all (a duplicate!).\n"
               "An even lift across the whole image = flicker."),
}


def defaults(effect):
    return {p[0]: p[4] for p in PARAMS.get(effect, [])}


def param(params, key, fallback):
    """Safely reads a parameter (returns the default when it is missing)."""
    try:
        return float(params.get(key, fallback))
    except Exception:
        return fallback


def needs_previous(effect):
    """The temporal check needs the previous frame besides the current one."""
    return effect == TEMPORAL


def matte_overlay(rgb8, matte, channels, lightness=1.0, gain=1.0, gamma=1.0):
    """Overlays the finished image with mattes from the RGBA DiMatte input.

    `rgb8`      the finished image (h,w,3) uint8 - a grey one is converted to
                colour, otherwise there would be nothing to tell the matte apart
    `matte`     (h,w,4) half - the channels carry the mattes
    `channels`  which channels to draw, in R,G,B,A order
    `lightness` lightness of the matte colour
    `gain`      matte gain - a weak matte can be pulled up
    `gamma`     shape of the matte transition (below 1 harder, above 1 softer)

    CAREFUL: NO THRESHOLD. The matte is mixed by its own value, so the
    transitions stay exactly as they are in it - the edge of a blurred matte is
    blurred in the overlay too. With a threshold every matte would become a
    hard blob, and the edges are exactly where you can tell whether a matte is
    well made.

    They are drawn in R, G, B, A order, so on an overlap the later one wins -
    just like stacking mattes in a comp.
    """
    if rgb8 is None or matte is None or not any(channels):
        return rgb8
    if rgb8.ndim == 2:                          # grey image -> colour
        rgb8 = np.repeat(rgb8[:, :, None], 3, axis=2)
    light = max(0.0, min(1.0, float(lightness)))
    gain = max(0.0, float(gain))
    gamma = max(0.01, float(gamma))
    out = rgb8.astype(np.float32)
    for idx, want in enumerate(channels):
        if not want or idx >= matte.shape[2]:
            continue
        m = np.asarray(matte[:, :, idx], np.float32)
        if abs(gain - 1.0) > 1e-3:
            m = m * gain
        m = np.clip(m, 0.0, 1.0)
        if abs(gamma - 1.0) > 1e-3:
            m = np.power(m, 1.0 / gamma)     # the shape of the transition,
        m = m[:, :, None]                    # not its existence
        color = np.asarray(MATTE_COLORS[idx], np.float32) * light
        out *= (1.0 - m)                        # out = out*(1-m) + colour*m
        out += color * m
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def needs_other(effect):
    """Difference needs a frame from the OTHER input (same frame, other plate)."""
    return effect in NEEDS_BOTH


def diff_is_overlay(params):
    """Is an overlay drawn over the image (rather than a plain difference)?"""
    return int(round(param(params, "mode", DIFF_OVERLAY))) == DIFF_OVERLAY


def difference_color(params):
    """The overlay colour including the intensity."""
    idx = max(0, min(len(DIFF_COLORS) - 1, int(round(param(params, "color", 0)))))
    intensity = max(0.0, param(params, "intensity", 1.0))
    return tuple(min(255, int(round(c * intensity))) for c in DIFF_COLORS[idx])


def difference_mask(cur, other, lut, params=None):
    """Where the inputs differ (a bool array h,w).

    Deliberately separated from the drawing: the overlay is drawn over the
    IMAGE, which has already been through the chosen display transform, whereas
    this decision runs on a fixed built-in conversion (`lut`) - so the
    threshold always means the same thing regardless of the monitor setup.
    """
    params = params or {}
    thr = max(0.0, param(params, "threshold", 0.01))
    if thr <= 0.0:
        # A threshold of 0 should mean ANY difference. In bytes everything
        # finer than one display step would be lost (0.5 against 0.505 is the
        # same byte), so this one case is compared straight from the data.
        return np.any(np.asarray(cur[:, :, :3], np.float32)
                      != np.asarray(other[:, :, :3], np.float32), axis=2)
    a = lut[cur[:, :, :3].view(np.uint16)]
    b = lut[other[:, :, :3].view(np.uint16)]
    d = np.maximum(a, b)
    d -= np.minimum(a, b)
    return d.max(axis=2) > int(round(thr * 255.0))


def difference(cur, other, lut, params=None, threads=1):
    """The difference of two inputs. `cur` and `other` are (h,w,4) half scene-linear.

    Computed in the display domain (through `lut`), not in scene-linear: the
    threshold and the gain then mean the same thing a person sees on screen,
    and it does not matter whether the plate is log or linear. Everything runs
    in uint8 - the difference of two bytes is a whole number 0-255, so a table
    is enough for the gain.
    """
    overlay = diff_is_overlay(params)
    color = difference_color(params) if overlay else None
    intensity = max(0.0, param(params, "intensity", 1.0))
    table = np.clip(np.arange(256, dtype=np.float32)
                    * (intensity * DIFF_PLAIN_GAIN), 0, 255).astype(np.uint8)

    def band(c, o):
        a = lut[c[:, :, :3].view(np.uint16)]
        if overlay:
            # Overlay: the image stays, changed places get a solid colour.
            out = np.ascontiguousarray(a)
            out[difference_mask(c, o, lut, params)] = color
            return out
        # The plain difference. Intensity means the gain on the difference
        # here; the DIFF_PLAIN_GAIN multiplier is there so that an intensity of
        # 1.00 looks the same as it used to - small differences would otherwise
        # not be visible at all.
        b = lut[o[:, :, :3].view(np.uint16)]
        d = np.maximum(a, b)
        d -= np.minimum(a, b)                   # |a - b| unsigned, no floats
        return table[d]

    bands = _band_count(cur.shape[0], int(threads), 0)
    if bands > 1:
        return _banded(lambda r0, r1: band(cur[r0:r1], other[r0:r1]),
                       cur.shape[0], bands, 0)
    return band(cur, other)


def temporal(cur, prev, lut, params=None, threads=1):
    """|current - previous| amplified. Black = no change, glowing = motion.

    Everything is computed in uint8: the difference of two bytes is a whole
    number 0-255, so a 256-entry table covers both the gain and the clip.
    Converting to float32 would give exactly the same result, only it would
    cost 2.3x more (24.8 -> 10.8 ms on 1080p).
    """
    gain = param(params or {}, "gain", TEMPORAL_GAIN)
    table = np.clip(np.arange(256, dtype=np.float32) * gain,
                    0, 255).astype(np.uint8)

    def band(c, p):
        a = _to_display(c, lut)
        b = _to_display(p, lut)
        return table[np.maximum(a, b) - np.minimum(a, b)]   # |a-b|, no underflow

    bands = _band_count(cur.shape[0], int(threads), 0)
    if bands > 1:
        return _banded(lambda r0, r1: band(cur[r0:r1], prev[r0:r1]),
                       cur.shape[0], bands, 0)
    return band(cur, prev)


def frame_difference(cur, prev, step=8):
    """(mean difference 0-255, are_they_bit_identical) on subsampled data.

    A bitwise match is unambiguous proof of a duplicate frame. The mean
    difference reveals "almost identical" frames (e.g. a stuck render with
    different noise). Subsampling is enough - it detects a duplicate reliably
    and costs next to nothing.
    """
    if cur is None or prev is None or cur.shape != prev.shape:
        return None, False
    a = cur[::step, ::step, :3]
    b = prev[::step, ::step, :3]
    identical = np.array_equal(a.view(np.uint16), b.view(np.uint16))
    diff = float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())
    return diff, identical


def needs_full_frame(effect):
    """No effect needs the whole image any more.

    Canvas is handled by shifting the coordinates (see canvas_source_index), so
    it is computed from the visible crop only and keeps full detail when
    zoomed in.
    """
    return False


def canvas_source_index(x0, y0, x1, y1, step, full_h, full_w, params=None):
    """(ys, xs) - which source rows/columns to take pixels from.

    Canvas check = shifting the image with wraparound, so the original edges
    meet in the middle. Instead of shuffling the whole image we simply take the
    shifted pixels, so only what is visible gets computed (and zooming in gives
    full detail). The shift is adjustable (half the image by default = the
    classic canvas check).
    """
    params = params or {}
    dy = int(full_h * param(params, "shift_y", 50.0) / 100.0)
    dx = int(full_w * param(params, "shift_x", 50.0) / 100.0)
    ys = (np.arange(y0, y1, step) + dy) % full_h
    xs = (np.arange(x0, x1, step) + dx) % full_w
    return ys, xs


def _wrap_runs(idx, step):
    """Splits a wrapped index list into (src_from, src_to, dst_from, dst_to) runs.

    The shift wraps at most once along each axis, so this gives one or two runs
    per axis - each of them a plain slice of the source.
    """
    out = []
    n = len(idx)
    start = 0
    while start < n:
        end = start + 1
        while end < n and idx[end] == idx[end - 1] + step:
            end += 1
        out.append((int(idx[start]), int(idx[end - 1]) + 1, start, end))
        start = end
    return out


def canvas_crop(arr, x0, y0, x1, y1, step, params=None):
    """The shifted (canvas) crop of `arr`, assembled from CONTIGUOUS blocks.

    The obvious way to do this is arr[np.ix_(ys, xs)], but that is a per-element
    gather and cannot read memory in order: measured on 4K it costs 30 ms a
    frame, against 0 ms for the plain slice ordinary display gets - which is why
    the canvas check stuttered while nothing else did.

    The shift wraps at most once per axis, so the result is really just two to
    four RECTANGLES copied out of the source. Copying them as slices is a
    straight memory copy and is 3x faster (30 -> 10 ms), for a bit-identical
    result.
    """
    full_h, full_w = arr.shape[0], arr.shape[1]
    ys, xs = canvas_source_index(x0, y0, x1, y1, step, full_h, full_w, params)
    if len(ys) == 0 or len(xs) == 0:
        return arr[np.ix_(ys, xs)]              # nothing to copy - let numpy do it
    out = np.empty((len(ys), len(xs)) + arr.shape[2:], dtype=arr.dtype)
    for sy, ey, dy0, dy1 in _wrap_runs(ys, step):
        for sx, ex, dx0, dx1 in _wrap_runs(xs, step):
            out[dy0:dy1, dx0:dx1] = arr[sy:ey:step, sx:ex:step]
    return out


def _to_display(lin, lut):
    """Scene-linear half -> uint8 RGB through the LUT (as in normal display)."""
    return lut[lin[:, :, :3].view(np.uint16)]


def _box(radius):
    """Odd box width for uniform_filter, from a radius in pixels."""
    return 2 * max(1, int(round(radius))) + 1


def _band(disp, fine, coarse):
    """The SIGNED band a high-pass leaves: one blur subtracted from another.

    Shared by the high-pass check and the high-pass comparison, so the two
    inputs of that comparison are put through exactly the same thing - a band
    computed two slightly different ways would show a difference that is not in
    the pictures.

    BOX blurs (running sums), not gaussians. A gaussian costs more the wider it
    gets - measured on 2.2 Mpx it goes 39 ms at radius 2 to 426 ms at radius
    24, so the "to detail" slider made the check unusable at its own top end. A
    box is O(1) in the radius: 31 ms to 40 ms across that whole range. Since
    this is the DIFFERENCE of two blurs, what the band keeps is decided by the
    two radii, and the kernel shape only changes how sharply the band falls off
    at its edges - a box rings very slightly around hard edges where a gaussian
    would not, which on a texture check is no loss.
    """
    coarse = max(coarse, fine + 0.3)        # the wider one really has to be wider
    if _ndi is None:                        # fallback without scipy: a 3x3 box
        b = disp + np.roll(disp, 1, 0) + np.roll(disp, -1, 0)
        b = b + np.roll(b, 1, 1) + np.roll(b, -1, 1)
        return disp - b * (1.0 / 9.0)
    high = _ndi.uniform_filter(disp, size=(_box(coarse), _box(coarse), 1),
                               mode="nearest")
    if fine > 0.05:
        low = _ndi.uniform_filter(disp, size=(_box(fine), _box(fine), 1),
                                  mode="nearest")
    else:
        low = disp
    return low - high


def _desaturate(out, desat):
    """Mixes a signed image towards its own luminance, in place where it can."""
    if desat <= 0.001 or out.ndim != 3 or out.shape[2] != 3:
        return out
    y = (out[:, :, 0] * LUMA_R + out[:, :, 1] * LUMA_G + out[:, :, 2] * LUMA_B)
    out *= (1.0 - desat)
    out += y[:, :, None] * desat
    return out


# Every half float there is, as float32 - for building tables indexed by half
# BITS (the display LUTs are indexed that way, see imageview.build_lut).
_HALF_VALUES = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)

_GRAIN_TABLE = None
_GRAIN_TABLE_KEY = None
_GRAIN_TABLE_LOCK = threading.Lock()


def _grain_table(lut, contrast):
    """A table that does |x| * contrast AND the display curve in one lookup.

    The grain ends with abs, a multiply, a clip and then the display LUT - four
    passes over a 26 MB array. All but the lookup can be baked into the table
    instead, because the table is indexed by the half BITS of the residual and
    there are only 65536 of those: entry i is simply what |half(i)| * contrast
    comes out as on screen. Measured 26.5 -> 19.7 ms on 2.2 Mpx, for a table
    that costs 0.2 ms and is then reused until the contrast changes.

    Quantising the residual to half BEFORE the multiply rather than after moves
    the last bit: measured one display level on 0.7 % of pixels.
    """
    global _GRAIN_TABLE, _GRAIN_TABLE_KEY
    key = (id(lut), float(contrast))
    with _GRAIN_TABLE_LOCK:                 # the bands all ask at once
        if key != _GRAIN_TABLE_KEY:
            v = np.nan_to_num(np.abs(_HALF_VALUES) * float(contrast),
                              nan=0.0, posinf=60000.0, neginf=0.0)
            np.clip(v, 0.0, 60000.0, out=v)
            _GRAIN_TABLE = lut[v.astype(np.float16).view(np.uint16)]
            _GRAIN_TABLE_KEY = key
        return _GRAIN_TABLE


_LOG_TABLE = None
_LOG_TABLE_KEY = None
_LOG_TABLE_LOCK = threading.Lock()


def _log_table(curve, exposure, black):
    """half BITS -> uint8, the whole log view in one lookup.

    The curve itself is a log10 per pixel, which on a 4K frame is the most
    expensive thing in the check by a wide margin. It does not have to be:
    the input is a half, so there are only 65536 answers, and the table is
    built once and reused until a slider moves.
    """
    global _LOG_TABLE, _LOG_TABLE_KEY
    name = LOG_CURVES[max(0, min(len(LOG_CURVES) - 1, int(round(curve))))]
    key = (name, float(exposure), float(black))
    with _LOG_TABLE_LOCK:
        if key != _LOG_TABLE_KEY:
            # The table covers EVERY half there is, so the top of it overflows
            # as soon as exposure is pushed up, and log10 is handed a zero at
            # the bottom. Both are expected and both are dealt with by the
            # clip below - they must not print a warning per slider move.
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                v = np.clip(_HALF_VALUES, 0.0, None) * (2.0 ** float(exposure))
                v = np.asarray(nukelut.encode(name, v), dtype=np.float32)
            # Black level lifts the floor away, so the bottom of the curve is
            # not read as detail when it is only the toe of the encoding.
            lo = float(black)
            if lo > 0.0:
                v = (v - lo) / max(1e-6, 1.0 - lo)
            _LOG_TABLE = (np.clip(np.nan_to_num(v, nan=0.0, posinf=1.0,
                                                neginf=0.0), 0.0, 1.0)
                          * 255.0 + 0.5).astype(np.uint8)
            _LOG_TABLE_KEY = key
        return _LOG_TABLE


def _log(lin, _lut, params=None):
    """The shot as it looks IN LOG, whatever the monitor is set to.

    Not a display transform and not a check over the picture - it is the same
    data read back through a log curve, which is how a shot is judged before it
    is graded: the toe and the shoulder are stretched out, so crushed blacks,
    clipped highlights and banding are all visible where a display transform
    has already rolled them away.

    The window's own display setting is deliberately ignored (`_lut` is unused)
    - a log view that still went through sRGB would be neither one thing nor
    the other.
    """
    params = params or {}
    table = _log_table(param(params, "curve", 0),
                       param(params, "exposure", 0.0),
                       param(params, "black", 0.0))
    return table[lin[:, :, :3].view(np.uint16)]


def _grain(lin, lut, params=None):
    """Shows the GRAIN the way a compositor pulls it: a high-pass in SCENE-LINEAR
    (subtract a blurred copy of the plate - the "background"), its MAGNITUDE
    like a |A-B| difference merge, then shown through the ordinary display
    curve. The curve is the whole trick. Grain lives in the shadows, where the
    curve is steep, so it is lifted and reads evenly across the frame; edges are
    large, where the curve is flat, so they compress instead of blowing out.
    Result: fine, even grain on near-black, thin controlled edges.

    Measured on a real reference: in scene-linear the edges are 58x the flat
    grain, yet through the curve they sit only a little above it - exactly what
    makes the reference look fine. The previous version high-passed AFTER the
    display transform and applied a linear gain, which scales grain and edges
    together, so the edges clipped to white and dominated. This does it in the
    right order.

    The controls each separate grain from content a different way:
      contrast  exposure BEFORE the curve - grain brightness
      size      radius of the background subtracted; small = only the finest
                grain, larger = coarser texture comes through too
      fine      single-pixel grain over 2-3px softness, via the
                [[2,2,2],[2,-15,2],[2,2,2]] matrix (weights sum to 1, so a flat
                area is unchanged and a lone pixel comes out 15x). This is the
                reference's own matrix. 0 = plain high-pass.
    """
    params = params or {}
    contrast = param(params, "contrast", GRAIN_CONTRAST)
    size = max(0.3, param(params, "size", GRAIN_SIZE))
    fine = max(0.0, min(1.0, param(params, "fine", GRAIN_FINE)))

    src = lin[:, :, :3].astype(np.float32)          # scene-linear

    if _ndi is None:                        # fallback without scipy: a 3x3 box
        b = src + np.roll(src, 1, 0) + np.roll(src, -1, 0)
        b = b + np.roll(b, 1, 1) + np.roll(b, -1, 1)
        resid = src - b * (1.0 / 9.0)
    else:
        # A BOX blur (running sum), not a gaussian: uniform_filter is O(1) in
        # the radius and several times cheaper than gaussian_filter, which is
        # the bulk of the frame time. For a high-pass BACKGROUND the exact shape
        # barely matters - the grain is the residual either way. `size` is the
        # radius; k is the box width.
        resid = src - _ndi.uniform_filter(src, size=(_box(size), _box(size), 1),
                                          mode="nearest")

    if fine > 0.0:
        # The matrix on the residual: 2*(8 neighbours) - 15*centre, scaled by
        # -1/15 so a single-pixel spike comes back the right way up at ~1x while
        # coarser detail comes out smaller - that is what emphasises the finest
        # grain.
        #
        # Written as a 3x3 BOX rather than as eight shifted copies. The eight
        # neighbours are the 3x3 sum minus the centre, so the whole matrix is
        # just (17*centre - 2*sum9)/15 - one running-sum pass instead of eight
        # full-size temporaries. Measured 114 -> 38 ms on 2.2 Mpx, and the
        # result matches to 4e-09 (plain float rounding).
        if _ndi is None:                    # no running sums without scipy
            s9 = resid + np.roll(resid, 1, 0) + np.roll(resid, -1, 0)
            s9 = s9 + np.roll(s9, 1, 1) + np.roll(s9, -1, 1)
            s9 *= (1.0 / 9.0)               # the mean, as uniform_filter gives
        else:
            s9 = _ndi.uniform_filter(resid, size=(3, 3, 1), mode="nearest")
        sharp = resid * (17.0 / 15.0)
        sharp -= s9 * (18.0 / 15.0)         # s9 is the MEAN, so sum9 = s9 * 9
        resid = resid * (1.0 - fine) + sharp * fine

    # Magnitude (like |A-B|), exposed, then THROUGH THE DISPLAY CURVE, so the
    # viewer's own transform lifts the grain and compresses the edges. All of
    # that is one lookup: the abs, the contrast and the curve live in the table
    # (see _grain_table), so only the residual has to be walked over.
    return _grain_table(lut, contrast)[
        resid.astype(np.float16).view(np.uint16)]


def _bandpass(lin, lut, params=None):
    """A band pass - keeps only detail in the given size range.

    It blurs the image twice and subtracts: the narrower blur decides how FINE
    a detail still passes, the wider one how COARSE a detail is already
    subtracted. One band of frequencies stays between them. With "from" at 0 it
    is an ordinary high-pass.

    Why it sits next to the grain check: grain has local normalisation and a
    soft clip in order to pull out the grain and suppress the shot. This one
    normalises nothing, so it is an honest subtraction - DIFFERENCES IN TEXTURE
    (a paint fix, cloning, a plate seam) read better in it than the grain itself.
    """
    params = params or {}
    fine = param(params, "fine", 0.0)
    coarse = param(params, "coarse", 8.0)
    gain = param(params, "gain", 8.0)
    mid = param(params, "mid", 128.0)
    desat = max(0.0, min(1.0, param(params, "desat", 0.0)))
    coarse = max(coarse, fine + 0.3)        # the wider one really has to be wider

    out = _band(_to_display(lin, lut).astype(np.float32), fine, coarse)
    out *= gain
    # Towards luminance, BEFORE the pedestal and the clip: what is being mixed
    # is then the signed band, so grey stays grey and nothing has been clipped
    # to white first. At 1.0 the three channels are identical and the check is
    # in luminance. Skipped entirely at 0, which is the default, and on a
    # single-channel crop there is nothing to mix.
    _desaturate(out, desat)
    out += mid
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


def hp_difference(cur, other, lut, params=None, threads=1):
    """The high-pass of A against the high-pass of B.

    Both inputs go through the SAME band (see _band) and the two results are
    subtracted. What is left is where the two differ in TEXTURE - a paint fix,
    a re-grain, a softened patch, a plate swap - with the overall level taken
    out of the picture by the high-pass first. A plain difference lights up
    wherever the two are graded even slightly apart; this one does not care
    about that and only answers "is the same detail there".

    Black means the same detail in both. It sits on black rather than on a grey
    pedestal on purpose: on a comparison, "nothing" should look like nothing.
    """
    params = params or {}
    fine = param(params, "fine", 0.0)
    coarse = param(params, "coarse", 8.0)
    gain = param(params, "gain", 16.0)
    desat = max(0.0, min(1.0, param(params, "desat", 0.0)))

    def band(a, b):
        out = _band(_to_display(a, lut).astype(np.float32), fine, coarse)
        out -= _band(_to_display(b, lut).astype(np.float32), fine, coarse)
        _desaturate(out, desat)
        np.abs(out, out=out)                # direction does not matter, presence does
        out *= gain
        np.clip(out, 0, 255, out=out)
        return out.astype(np.uint8)

    halo = _halo_rows(BANDPASS, params)
    bands = _band_count(cur.shape[0], int(threads), halo)
    if bands > 1:
        return _banded(lambda r0, r1: band(cur[r0:r1], other[r0:r1]),
                       cur.shape[0], bands, halo)
    return band(cur, other)


def _saturation(lin, lut, params=None):
    """Every pixel is levelled to the same brightness. Only the colour stays:
    desaturated areas turn grey, saturated colours glow.

    `boost` additionally amplifies the deviation from grey, so even faint casts
    become visible.
    """
    params = params or {}
    level = param(params, "level", 128.0)
    boost = param(params, "boost", 1.0)

    d8 = _to_display(lin, lut)
    # the maximum of three 2D arrays, still in uint8. disp.max(axis=2) gives
    # the same, but costs 12x more (31.7 -> 2.6 ms on 1080p) - reducing along
    # an axis is expensive here.
    mx8 = np.maximum(np.maximum(d8[:, :, 0], d8[:, :, 1]), d8[:, :, 2])

    # The per-pixel divide level/mx is a TABLE lookup: mx is a byte, so there
    # are only 256 answers. Measured 15.4 ms of dividing gone.
    scale = np.arange(256, dtype=np.float32)
    scale[0] = 1e-6                         # black is overwritten below anyway
    np.divide(level, scale, out=scale)

    out = d8.astype(np.float32)
    if abs(boost - 1.0) > 1e-3:             # amplify the deviation from neutral grey
        # gray + (x - gray) * boost  ==  x * boost + gray * (1 - boost), which
        # is one pass fewer over a 26 MB array. gray is summed by hand rather
        # than with mean(axis=2) for the same reason the maximum above is -
        # measured 14.8 -> 5.3 ms. Folding it in before the scale (instead of
        # after, as it used to be) can move the last bit: measured one display
        # level on 0.06 % of pixels, which no eye finds on a saturation check.
        gray = out[:, :, 0] + out[:, :, 1] + out[:, :, 2]
        gray *= ((1.0 - boost) / 3.0)
        out *= boost
        out += gray[:, :, None]
    out *= scale[mx8][:, :, None]
    out[mx8 == 0] = level                   # black -> grey
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


def _canvas(lin, lut, params=None):
    """Swaps the quadrants diagonally - the original edges meet in the middle."""
    disp = _to_display(lin, lut)
    h, w = disp.shape[0], disp.shape[1]
    return np.ascontiguousarray(np.roll(disp, (h // 2, w // 2), axis=(0, 1)))


# ValueMap: colours by the value of the scene-linear luminance
VM_NEGATIVE = (255, 0, 0)         # below 0
VM_GRAYS = (0, 85, 170, 255)      # 0-0.25-0.5-0.75-1 in four steps
VM_OVER_1 = (0, 0, 255)           # 1 to 20
VM_OVER_20 = (0, 255, 0)          # 20 to 55
VM_OVER_55 = (255, 140, 0)        # above 55


def valuemap_bands(params):
    """(b1, b2, b3) - the band boundaries, always increasing.

    The handles already push each other along while dragging in the UI, but the
    values can also come from the node, so the computation checks the order too.
    """
    params = params or {}
    b1 = max(1e-3, param(params, "b1", 1.0))
    b2 = max(b1 * 1.001, param(params, "b2", 20.0))
    b3 = max(b2 * 1.001, param(params, "b3", 55.0))
    return b1, b2, b3


def _valuemap(lin, _lut, params=None):
    """False colours by SCENE-LINEAR luminance - an exposure and HDR check.

    You set the band boundaries yourself (the three-handle slider): `b1` is the
    end of the greys, above it blue up to `b2`, green up to `b3` and orange
    above that. The four grey steps always spread evenly below b1.
    """
    params = params or {}
    b1, b2, b3 = valuemap_bands(params)
    rgb = lin[:, :, :3].astype(np.float32)
    if rgb.shape[2] == 1:
        # a single isolated channel (see imageview._isolate_channel): ITS value
        # is what the bands are read from - weighting one channel by the three
        # luminance coefficients would only scale it by their sum, which is 1
        lum = rgb[:, :, 0]
    else:
        lum = (rgb[:, :, 0] * LUMA_R + rgb[:, :, 1] * LUMA_G
               + rgb[:, :, 2] * LUMA_B)
    h, w = lum.shape
    out = np.empty((h, w, 3), dtype=np.uint8)
    grays = np.array(VM_GRAYS, dtype=np.uint8)
    idx = np.clip((lum / b1 * 4.0).astype(np.int32), 0, 3)
    gray = grays[idx]
    out[:, :, 0] = gray
    out[:, :, 1] = gray
    out[:, :, 2] = gray
    out[lum < 0.0] = VM_NEGATIVE
    out[(lum >= b1) & (lum < b2)] = VM_OVER_1
    out[(lum >= b2) & (lum < b3)] = VM_OVER_20
    out[lum >= b3] = VM_OVER_55
    return out


_FUNCS = {LOG: _log, GRAIN: _grain, BANDPASS: _bandpass, SAT: _saturation,
          CANVAS: _canvas, VALUEMAP: _valuemap}


# ---------------------------------------------------------------------------
# Computing a check on SEVERAL THREADS.
#
# The blur-heavy checks are the one expensive thing left on the GUI thread
# (measured: one grain frame at 2.2 Mpx is 70 ms single-threaded, i.e. a 13 fps
# ceiling on DISPLAY no matter how fast the decoding is). numpy releases the GIL
# for large arrays, so splitting the picture into horizontal bands and doing one
# band per thread genuinely parallelises: measured 70 ms -> 31 ms on 6 threads.
#
# Each band is computed with a HALO of extra rows above and below, which are
# then thrown away. Without it every band edge would be a seam - the blur inside
# a band cannot see the rows that belong to its neighbour. With the halo the
# result is bit-identical to the single-threaded one (verified; only the two
# rows at the very top and bottom of the image differ, and those are the
# wrap-around of the blur itself, which the single-threaded version has too).
# ---------------------------------------------------------------------------
_POOL = None
_POOL_SIZE = 0
_POOL_LOCK = threading.Lock()


def _pool(threads):
    """One shared pool, rebuilt only when the thread count really changes."""
    global _POOL, _POOL_SIZE
    with _POOL_LOCK:
        if _POOL is None or _POOL_SIZE != threads:
            if _POOL is not None:
                _POOL.shutdown(wait=False)
            _POOL = ThreadPoolExecutor(max_workers=threads,
                                       thread_name_prefix="exr-qc")
            _POOL_SIZE = threads
        return _POOL


def _halo_rows(effect, params):
    """How many rows a band has to overlap its neighbour by.

    It has to cover the reach of the widest filter in the effect, otherwise the
    band edge is computed from data that is not there.
    """
    if effect == GRAIN:
        # box radius (uniform_filter) + 1 row for the 3x3 matrix + slack
        return int(round(max(0.3, param(params, "size", GRAIN_SIZE)))) + 2
    if effect == BANDPASS:
        # a box of that radius reaches exactly it (see _box)
        return int(max(param(params, "coarse", 8.0), 1.0)) + 2
    return 0


def _band_count(height, threads, halo):
    """How many bands are worth using.

    A band has to be a good deal taller than its halo - otherwise the overlap
    is most of the work and the threads cost more than they save. Pointwise
    checks have no halo at all; there the floor is just "big enough that a
    thread earns its keep".
    """
    if threads <= 1:
        return 1
    min_rows = max(64, 4 * halo + 16)
    return max(1, min(int(threads), int(height // min_rows)))


def _banded(make, height, bands, halo):
    """Runs `make(r0, r1)` over horizontal bands in parallel and stitches them.

    `make` gets the row range to COMPUTE (halo included) and returns the result
    for exactly those rows; the halo rows are trimmed off here. Written against
    row ranges rather than arrays so the two-input checks (difference,
    temporal) can slice both of their inputs the same way.

    Only for checks whose output row depends on nearby rows at most - i.e.
    everything except the canvas check, which shifts the whole picture.
    """
    step = -(-height // bands)                  # ceil, so the bands cover it all

    def one(i):
        y0 = i * step
        y1 = min(height, y0 + step)
        r0 = max(0, y0 - halo)                  # compute a bit more...
        r1 = min(height, y1 + halo)
        out = make(r0, r1)
        return out[y0 - r0:y0 - r0 + (y1 - y0)]  # ...and keep only our rows

    return np.concatenate(list(_pool(bands).map(one, range(bands))), axis=0)


# Checks whose rows can be computed independently (with a halo where they
# blur). CANVAS is deliberately absent: it rolls the picture by half its height,
# so a band has no idea what belongs in it.
BANDABLE = (GRAIN, BANDPASS, SAT, VALUEMAP, DIFF, TEMPORAL)


def _es_comp(es):
    """Keeps the grain BRIGHTNESS the same whatever the sampling.

    When the source is point-subsampled by es (the fast/coarse render, or a
    zoomed-out one), the high-pass sees neighbours es pixels apart, so the grain
    magnitude grows. Measured on a real plate it grows as 1 + ln(es) almost
    exactly (es 1..4 -> x1.00, 1.69, 2.07, 2.36). Dividing the contrast by that
    cancels it: the fast render and the full render then match - no jump in
    lightness when the refinement lands, or when you change zoom.
    """
    es = max(1, int(es))
    return 1.0 / (1.0 + float(np.log(es)))


def apply(effect, lin, lut, params=None, es=1, threads=1, lut_f=None):
    """uint8 RGB (h,w,3), or None when the effect is unknown / there is nothing to do.

    `es` is the source subsampling step the crop was taken at (1 = full). Only
    the grain uses it, to keep its brightness constant across samplings.
    `threads` splits the blur-heavy checks over that many bands (see _pool).
    `lut_f` is the same display curve UNCLIPPED (imageview.build_lut_f). The
    band pass is handed that one instead: it subtracts one blur from another,
    and over a highlight already flattened to white there is nothing left to
    subtract - the check would report smooth where the plate has texture.
    """
    fn = _FUNCS.get(effect)
    if fn is None or lin is None or lin.size == 0:
        return None
    params = params or {}
    if effect == BANDPASS and lut_f is not None:
        lut = lut_f
    if effect == GRAIN and es > 1:
        params = dict(params)
        params["contrast"] = param(params, "contrast", GRAIN_CONTRAST) * _es_comp(es)
    if effect in BANDABLE and int(threads) > 1:
        halo = _halo_rows(effect, params)
        bands = _band_count(lin.shape[0], int(threads), halo)
        if bands > 1:
            return _banded(lambda r0, r1: fn(lin[r0:r1], lut, params),
                           lin.shape[0], bands, halo)
    return fn(lin, lut, params)


def legend(effect):
    """A short explanation for the UI (what the colours mean / what to look for)."""
    if effect == VALUEMAP:
        return ("red < 0  |  4 grey steps below the first boundary  |  blue  "
                "|  green  |  orange above the last (boundaries are in the slider)")
    if effect == GRAIN:
        return "fine grain on near-black; edges compressed by the display curve"
    if effect == LOG:
        return "the shot in log, not through the display - toe and shoulder open"
    if effect == HPDIFF:
        return "black = the same detail in A and B; lit = texture that differs"
    if effect == BANDPASS:
        return "only detail between 'from' and 'to'; smooth spots = paint or blur"
    if effect == SAT:
        return "grey = no colour, colour saturation at constant brightness"
    if effect == CANVAS:
        return "quadrants swapped diagonally - the original edges are in the middle"
    if effect == DIFF:
        return ("A against B; overlay = dark blue-grey where the images differ "
                "by more than the threshold")
    if effect == TEMPORAL:
        return ("difference against the previous frame (x%g): black = no change"
                % TEMPORAL_GAIN)
    return ""
