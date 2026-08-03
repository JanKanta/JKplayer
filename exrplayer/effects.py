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

import numpy as np

try:
    from scipy import ndimage as _ndi
except Exception:
    _ndi = None

NONE = "none"
DIFF = "diff"
GRAIN = "grain"
BANDPASS = "bandpass"
SAT = "sat"
CANVAS = "canvas"
VALUEMAP = "valuemap"
TEMPORAL = "temporal"

# Ordered by group: first the difference between inputs, then detail and its
# changes (grain, high-pass, temporal), then colour and exposure (saturation,
# value map). Canvas is last on purpose - it is not a computation over the
# image like the others, just a shifted crop, and it is the only one displayed
# through the ordinary display transform.
#
# NONE is NOT in the list: QC is switched off by its toggle, not by an entry in
# the menu. It stays as a value though - the panel sends it to a window when QC
# is off.
ORDER = [DIFF, GRAIN, BANDPASS, TEMPORAL, SAT, VALUEMAP, CANVAS]
LABELS = {NONE: "No effect", DIFF: "Difference", GRAIN: "Grain check",
          BANDPASS: "High-pass", SAT: "Saturation check",
          CANVAS: "Canvas check", VALUEMAP: "Value map",
          TEMPORAL: "Temporal check"}

# The only mode that needs BOTH inputs at once. Because of it the panel decodes
# even the window that is not currently visible (see PlayerPanel._live_slots).
NEEDS_BOTH = (DIFF,)

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

# Grain checker (all values chosen by measuring on a real plate, see _grain)
GRAIN_SIGMA = 1.0             # high-pass sigma: smaller = only the finest detail
GRAIN_ENERGY_SIGMA = 2.0      # the neighbourhood defining "local activity";
                              # narrower = better edge rejection
                              # (edge/flat ratio 4.3 -> 2.8)
GRAIN_FLOOR = 0.5             # so completely smooth areas do not divide by zero
GRAIN_SOFT_K = 1.3            # steepness of the soft clip (tanh); larger = more grain
GRAIN_SCALE = 60.0            # final grain contrast
GRAIN_MID = 72.0              # background brightness (128 was too light a grey)


# ---------------------------------------------------------------------------
# Settings of the individual QC modes.
# (key, label, min, max, default, number of decimals)
# ---------------------------------------------------------------------------
PARAMS = {
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
        ("boost", "Grain contrast", 0.25, 4.0, 1.0, 2),
        ("sigma", "Grain size", 0.5, 3.0, GRAIN_SIGMA, 1),
        ("edge", "Edge rejection", 1.0, 8.0, GRAIN_ENERGY_SIGMA, 1),
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
    GRAIN: ("Shows the grain and suppresses the content of the shot.\n"
            "Look for: missing or doubled grain, areas with no grain\n"
            "(repainted / blurred), a jump in graininess between shots."),
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


def difference(cur, other, lut, params=None):
    """The difference of two inputs. `cur` and `other` are (h,w,4) half scene-linear.

    Computed in the display domain (through `lut`), not in scene-linear: the
    threshold and the gain then mean the same thing a person sees on screen,
    and it does not matter whether the plate is log or linear. Everything runs
    in uint8 - the difference of two bytes is a whole number 0-255, so a table
    is enough for the gain.
    """
    a = lut[cur[:, :, :3].view(np.uint16)]
    if diff_is_overlay(params):
        # Overlay: the image stays, changed places are covered with a solid colour.
        out = np.ascontiguousarray(a)
        out[difference_mask(cur, other, lut, params)] = difference_color(params)
        return out

    # The plain difference. Intensity means the gain on the difference here;
    # the DIFF_PLAIN_GAIN multiplier is there so that an intensity of 1.00
    # looks the same as it used to - small differences would otherwise not be
    # visible at all.
    b = lut[other[:, :, :3].view(np.uint16)]
    d = np.maximum(a, b)
    d -= np.minimum(a, b)                       # |a - b| without a sign, without floats
    intensity = max(0.0, param(params, "intensity", 1.0))
    table = np.clip(np.arange(256, dtype=np.float32)
                    * (intensity * DIFF_PLAIN_GAIN), 0, 255).astype(np.uint8)
    return table[d]


def temporal(cur, prev, lut, params=None):
    """|current - previous| amplified. Black = no change, glowing = motion.

    Everything is computed in uint8: the difference of two bytes is a whole
    number 0-255, so a 256-entry table covers both the gain and the clip.
    Converting to float32 would give exactly the same result, only it would
    cost 2.3x more (24.8 -> 10.8 ms on 1080p).
    """
    gain = param(params or {}, "gain", TEMPORAL_GAIN)
    a = _to_display(cur, lut)
    b = _to_display(prev, lut)
    diff = np.maximum(a, b) - np.minimum(a, b)      # |a-b| without underflow
    table = np.clip(np.arange(256, dtype=np.float32) * gain,
                    0, 255).astype(np.uint8)
    return table[diff]


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


def _to_display(lin, lut):
    """Scene-linear half -> uint8 RGB through the LUT (as in normal display)."""
    return lut[lin[:, :, :3].view(np.uint16)]


def _grain(lin, lut, params=None):
    """Shows the GRAIN and suppresses the content. The grain swings around dark grey.

    Measured on a real plate: a plain high-pass has 18x the variance on edges
    than on flat areas, so at a gain where the edges do not clip, the grain is
    invisible (hence the original "grey" result). The fix has two parts:
      1) LOCAL NORMALISATION - divide by the local activity in a narrow
         neighbourhood, so edges cancel out and grain in quiet areas is boosted.
      2) A SOFT CLIP (tanh) - the remnants of edges saturate smoothly instead
         of shooting off to white/black.
    Result: edge/flat ratio 13.3x -> 1.9x, grain 6.1 -> 24, no clipping.
    """
    params = params or {}
    boost = param(params, "boost", 1.0)
    sigma = param(params, "sigma", GRAIN_SIGMA)
    edge_sigma = param(params, "edge", GRAIN_ENERGY_SIGMA)

    disp = _to_display(lin, lut).astype(np.float32)
    if _ndi is None:                        # fallback without scipy: a 3x3 box
        b = disp + np.roll(disp, 1, 0) + np.roll(disp, -1, 0)
        b = b + np.roll(b, 1, 1) + np.roll(b, -1, 1)
        blur = (8.0 * b + 52.0 * disp) * (1.0 / 124.0)
        return np.clip((disp - blur) * 16.0 * boost + GRAIN_MID,
                       0, 255).astype(np.uint8)

    hp = disp - _ndi.gaussian_filter(disp, sigma=(sigma, sigma, 0), truncate=3.0)
    # Local activity in a narrow neighbourhood - the narrower, the better the
    # edges cancel. Summing three abs values is the same as abs().mean(axis=2),
    # but without a temporary array across all channels (16.9 -> 7.3 ms).
    # CAREFUL with /3.0 instead of *(1/3): one third is not exact in binary, so
    # multiplying by the reciprocal moves the last bit and the result differs
    # from np.mean by a level.
    energy = _ndi.gaussian_filter(
        (np.abs(hp[:, :, 0]) + np.abs(hp[:, :, 1]) + np.abs(hp[:, :, 2])) / 3.0,
        sigma=edge_sigma, truncate=2.0)
    norm = hp / (energy[:, :, None] + GRAIN_FLOOR)
    # A SOFT clip: grain (small values) stays linear, edges saturate smoothly
    # instead of shooting off -> the shot underneath practically disappears and
    # nothing clips (measured: edge/flat ratio 4.5 -> 1.9, clipping 4.9 -> 0 %)
    out = np.tanh(norm * (GRAIN_SOFT_K * boost)) * GRAIN_SCALE + GRAIN_MID
    return np.clip(out, 0, 255).astype(np.uint8)


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
    coarse = max(coarse, fine + 0.3)        # the wider one really has to be wider

    disp = _to_display(lin, lut).astype(np.float32)
    if _ndi is None:                        # fallback without scipy: a 3x3 box
        b = disp + np.roll(disp, 1, 0) + np.roll(disp, -1, 0)
        b = b + np.roll(b, 1, 1) + np.roll(b, -1, 1)
        return np.clip((disp - b * (1.0 / 9.0)) * gain + mid,
                       0, 255).astype(np.uint8)

    # truncate 2.0 instead of 3.0: on a wide blur it shortens the kernel by a
    # third and it is not noticeable on a visual check (it is the difference of
    # two blurs anyway)
    low = disp if fine <= 0.05 else _ndi.gaussian_filter(
        disp, sigma=(fine, fine, 0), truncate=2.0)
    high = _ndi.gaussian_filter(disp, sigma=(coarse, coarse, 0), truncate=2.0)
    return np.clip((low - high) * gain + mid, 0, 255).astype(np.uint8)


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
    disp = d8.astype(np.float32)
    mx = mx8.astype(np.float32)[:, :, None]
    out = disp * (level / np.maximum(mx, 1e-6))
    if abs(boost - 1.0) > 1e-3:             # amplify the deviation from neutral grey
        gray = out.mean(axis=2, keepdims=True)
        out = gray + (out - gray) * boost
    out[mx8 == 0] = level                   # black -> grey
    return np.clip(out, 0, 255).astype(np.uint8)


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
    lum = rgb[:, :, 0] * 0.2126 + rgb[:, :, 1] * 0.7152 + rgb[:, :, 2] * 0.0722
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


_FUNCS = {GRAIN: _grain, BANDPASS: _bandpass, SAT: _saturation,
          CANVAS: _canvas, VALUEMAP: _valuemap}


def apply(effect, lin, lut, params=None):
    """uint8 RGB (h,w,3), or None when the effect is unknown / there is nothing to do."""
    fn = _FUNCS.get(effect)
    if fn is None or lin is None or lin.size == 0:
        return None
    return fn(lin, lut, params or {})


def legend(effect):
    """A short explanation for the UI (what the colours mean / what to look for)."""
    if effect == VALUEMAP:
        return ("red < 0  |  4 grey steps below the first boundary  |  blue  "
                "|  green  |  orange above the last (boundaries are in the slider)")
    if effect == GRAIN:
        return "grain swings around dark grey; edges suppressed by normalisation"
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
