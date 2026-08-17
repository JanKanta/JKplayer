"""
Safe image display (QPainter, no OpenGL).

WHY: the GL path (a raw pointer into glTexImage/setAttributeArray) crashed
Nuke. This uses no low-level call - it cannot crash.

COLOUR: display ALWAYS goes through OCIO (see ocio.py) - the built-in
conversion and its switch no longer exist. The remaining build_lut table is
only for the QC effects, which need a display-referred domain, and as a
fallback if OCIO were missing.

Everything rests on the fact that half has only 65536 possible values: the
conversion from half, the exposure and the shaper are therefore a single lookup
in a precomputed table, which is rebuilt only when a setting changes, not every
frame.

The cache stays SCENE-LINEAR half - so a channel/exposure change never touches
the cache and decodes nothing, it only recomputes the display from data that is
already loaded.
"""

import time

import numpy as np
from .qtcompat import QtCore, QtGui, QtWidgets, event_pos

from . import annotate
from . import resample
from . import effects as fx
from . import nukelut
from . import ocio as ocio_mod

MODE_NUKE, MODE_OCIO = range(2)

CH_RGB, CH_R, CH_G, CH_B, CH_A, CH_LUMA = range(6)

# Pixel ceiling for the QC effects. Effects are more expensive than ordinary
# display (grain does a blur), so when ZOOMED OUT we compute at a coarser step -
# the detail is not visible anyway. As soon as you zoom in, the crop is small
# and gets computed at full resolution, so the grain check is exact exactly
# where you are looking at it.
EFFECT_PIXEL_BUDGET = 2000000

EFFECT_MARGIN = 0.10        # a smaller margin than ordinary display (0.35):
                            # thanks to that we fit under the ceiling at 100 %
                            # and grain is computed EXACTLY (step 1) where you
                            # are inspecting it

# Grain / high-pass are computed at (near) FULL resolution and the RESULT is
# averaged down to screen size, rather than point-subsampling the source first.
# Point-subsampling a high-pass aliases the grain: it comes out coarse and it
# CRAWLS when you pan or zoom, because the sampling grid slides under it (a 1px
# pan picks a different set of pixels). Averaging a full-res result is what the
# eye expects - fine, even grain that sits still, the same as looking at a baked
# grain pass.
#
# This is the ceiling on that COMPUTE. At screen resolution (zoom >= 100 %) the
# visible crop is at most a screenful, so it sits under this and is computed
# exactly. Zoomed out the source crop is larger; the compute step is then
# raised just enough to fit here, and the result is still averaged down, so it
# stays steady. Deliberately near a screenful (not the whole 6K plate): the
# extra resolution would not be visible and would only cost frames.
GRAIN_SS_BUDGET = 3_000_000

# While PLAYING or moving the view, the blur-heavy checks render at this much
# compute so playback stays smooth; a moment after everything goes quiet they
# re-render at the full budget above. Half the full budget: the fast and full
# renders then match outright from ~125 % up (both full resolution there), and
# where they still differ - around 100 % and when zoomed right out - the grain
# brightness is already matched (see effects._es_comp) and the coarse render is
# drawn smoothed (see paintEvent), so the refinement is a gentle sharpen, not a
# jump. Raise it toward the full budget for an even smaller step (fewer fps),
# lower it for more fps.
GRAIN_FAST_BUDGET = 1_500_000

_HALF_VALUES = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)


def build_lut(gain=1.0, gamma=1.0):
    """65536 -> uint8, conversion through sRGB. The input is half float BITS.

    Display always goes through OCIO now; this table is left for the QC
    effects, which need a sensible display-referred domain, and as a fallback
    if OCIO were unavailable.
    """
    v = np.nan_to_num(_HALF_VALUES, nan=0.0, posinf=1e4, neginf=0.0) * float(gain)
    v = np.clip(v, 0.0, None)
    v = np.where(v <= 0.0031308, v * 12.92,
                 1.055 * np.power(v, 1.0 / 2.4) - 0.055)
    if abs(gamma - 1.0) > 1e-6:
        v = np.power(np.clip(v, 0.0, None), 1.0 / max(gamma, 1e-3))
    return (np.clip(v, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def build_lut_f(gain=1.0, gamma=1.0):
    """The same curve as build_lut, in float32 and WITHOUT the clip.

    build_lut ends in uint8, so everything at or above display white lands on
    255 and everything below black on 0. That is right for a picture and wrong
    for a CHECK: a high pass over a flattened highlight finds no texture there,
    because the flattening happened before it looked. On a plate that goes over
    1.0 - a CG render, a practical, an unclamped grade - the check would be
    blind exactly where one most wants to look.

    So the curve is continued instead of clipped (sRGB's power arm carries on
    perfectly well above 1.0) and mirrored through zero for negatives, keeping
    their sign. Still scaled by 255, so the gain and pedestal of the checks
    mean the same as before.
    """
    v = np.nan_to_num(_HALF_VALUES, nan=0.0, posinf=1e4, neginf=-1e4) * float(gain)
    sign = np.sign(v)
    a = np.abs(v)
    a = np.where(a <= 0.0031308, a * 12.92,
                 1.055 * np.power(a, 1.0 / 2.4) - 0.055)
    if abs(gamma - 1.0) > 1e-6:
        a = np.power(np.maximum(a, 0.0), 1.0 / max(gamma, 1e-3))
    return (sign * a * 255.0).astype(np.float32)


LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)   # Rec.709

# Luminance through tables: every channel has its own 65536-entry table already
# multiplied by its weight, so converting the whole crop to float32 and back
# falls away. Measured on 2.2 Mpx: 27.5 -> 21.6 ms, the result bit-identical.
_HALF_SAFE = np.nan_to_num(_HALF_VALUES, nan=0.0, posinf=65504.0, neginf=0.0)
_LUMA_TABLES = tuple((_HALF_SAFE * w).astype(np.float32) for w in LUMA)


def _luma_bits(bits):
    """RGB half bits -> luminance half bits (ready for a LUT lookup)."""
    y = _LUMA_TABLES[0][bits[:, :, 0]]
    y += _LUMA_TABLES[1][bits[:, :, 1]]
    y += _LUMA_TABLES[2][bits[:, :, 2]]
    return y.astype(np.float16).view(np.uint16)


def _halve(img):
    """Average 2x2 blocks -> (ceil(h/2), ceil(w/2), ...).

    Kept in uint16: four bytes always fit, so nothing is promoted to float and
    the arithmetic happens on the SMALL output rather than on a float copy of
    the whole input. Measured on 2.2 Mpx that is 55 ms down to 7 ms.

    An odd side replicates its last row/column first, so the result is ceil,
    exactly what a point subsample of the same step would have produced - the
    coverage test compares those sizes and would re-render every frame if they
    disagreed by even one row.
    """
    if img.shape[0] % 2:
        img = np.concatenate([img, img[-1:]], axis=0)
    if img.shape[1] % 2:
        img = np.concatenate([img, img[:, -1:]], axis=1)
    s = img[0::2, 0::2].astype(np.uint16)
    s += img[1::2, 0::2]
    s += img[0::2, 1::2]
    s += img[1::2, 1::2]
    s += 2                                  # round to nearest, not down
    s >>= 2
    return s.astype(np.uint8)


def _block_mean(img, f):
    """Average f x f blocks -> (ceil(h/f), ceil(w/f), ...).

    Anchored to pixel 0 of the crop, and every pixel is weighted the same, so
    the shrunk result does not crawl the way a point subsample does.

    Halving repeatedly where it can (see _halve); an odd factor finishes on
    reduceat, which sums whole blocks including a short final one, so dividing
    by the real count per block keeps the edge blocks right.
    """
    if f <= 1:
        return img
    while f > 1 and f % 2 == 0:
        img = _halve(img)
        f //= 2
    if f <= 1:
        return img
    h, w = img.shape[0], img.shape[1]
    src = img.astype(np.float32)
    yi = np.arange(0, h, f)
    xi = np.arange(0, w, f)
    s = np.add.reduceat(np.add.reduceat(src, yi, axis=0), xi, axis=1)
    yc = np.add.reduceat(np.ones(h, np.float32), yi)
    xc = np.add.reduceat(np.ones(w, np.float32), xi)
    cnt = yc[:, None] * xc[None, :]
    if src.ndim == 3:
        cnt = cnt[:, :, None]
    return (s / cnt + 0.5).astype(np.uint8)


def _make_qimage(rgb):
    """A QImage over a numpy array - both colour (h,w,3) and grey (h,w)."""
    if rgb.ndim == 2:
        h, w = rgb.shape
        return QtGui.QImage(rgb.data, w, h, w, QtGui.QImage.Format_Grayscale8)
    h, w = rgb.shape[0], rgb.shape[1]
    return QtGui.QImage(rgb.data, w, h, w * 3, QtGui.QImage.Format_RGB888)

# The neutral conversion for the QC effects. The effects are deliberately NOT
# computed from the CC values: when someone pulls the exposure, the image of
# the check should change, not what the check measures. CC is therefore applied
# ON TOP of the result (see _apply_cc).
NEUTRAL_LUT = build_lut()
NEUTRAL_LUT_F = build_lut_f()    # the same, unclipped, for the band checks


def build_cc_lut(gain, gamma):
    """256 -> 256 uint8: gain and gamma over finished bytes, or None.

    This is the CC path for the QC effects - it tints the result of the check
    but does not touch the data the check was computed from.
    """
    if abs(float(gain) - 1.0) < 1e-6 and abs(float(gamma) - 1.0) < 1e-6:
        return None
    v = np.linspace(0.0, 1.0, 256, dtype=np.float32) * float(gain)
    v = np.clip(v, 0.0, None)
    if abs(float(gamma) - 1.0) > 1e-6:
        v = np.power(v, 1.0 / max(float(gamma), 1e-3))
    return (np.clip(v, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def build_gamma_lut(gamma):
    """256 -> 256 uint8, or None when gamma is 1.0.

    With OCIO the display transform is already in the 3D LUT, so the gamma from
    CC is applied over the finished bytes - one extra table lookup.
    """
    if abs(float(gamma) - 1.0) < 1e-6:
        return None
    v = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    v = np.power(v, 1.0 / max(float(gamma), 1e-3))
    return (np.clip(v, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def build_saturation_matrix(saturation):
    """A 3x3 matrix for saturation, or None when it is 1.0 (nothing to do).

    It is multiplied from the right (pixel @ M), so M is already transposed.
    """
    s = float(saturation)
    if abs(s - 1.0) < 1e-3:
        return None
    m = np.eye(3, dtype=np.float32) * s + (1.0 - s) * LUMA[None, :]
    return np.ascontiguousarray(m.T)


class ImageView(QtWidgets.QWidget):
    """Shows a scene-linear half frame; pan/zoom through QPainter."""

    probeChanged = QtCore.Signal(object)     # pixel values under the cursor
    viewportChanged = QtCore.Signal()        # zoom/pan - the other window follows
    picked = QtCore.Signal()                 # a click = this window is active
    annotated = QtCore.Signal()              # a note was added or taken back
    textWanted = QtCore.Signal(float, float)  # image point a note was asked for

    def __init__(self, parent=None):
        super(ImageView, self).__init__(parent)
        self.setMinimumSize(160, 120)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setAutoFillBackground(False)
        self.setMouseTracking(True)           # probe even without a held button
        self.probe_frozen = False             # the P key freezes the readout

        self._frame = None           # (h,w,4) float16 scene-linear
        self._prev = None            # the previous frame (for the temporal check)
        self._other = None           # the same frame from the other input (difference)
        self._matte = None           # the DiMatte input frame (mattes in RGBA)
        self._diff_mask = None       # where A and B differ (difference overlay)
        self.matte_channels = ()     # which matte channels to draw
        self.matte_shape = (1.0, 1.0, 1.0)   # lightness, gain, gamma
        self._qimage = None          # the finished image to draw
        self._rgb = None             # keeps the buffer alive for the QImage!
        self._dirty = True
        self._step = 1               # subsampling (see _pick_step)
        self._rendered = None        # (x0, y0, cols, rows, step) what is in _qimage
        self.margin = 0.35           # margin around the visible area (share of the window)

        self._lut = build_lut()
        self.gain = 1.0
        self.gamma = 1.0
        self.channels = CH_RGB
        self.saturation = 1.0        # not in the LUT - handled over RGB pixels
        self._sat_matrix = None      # None = saturation 1.0, i.e. do nothing
        self.ocio = None             # ocio.DisplayTransform, or None
        self.nuke_display = nukelut.DEFAULT_DISPLAY   # the Nuke mode
        self.nuke_input = nukelut.DEFAULT_INPUT
        self._gamma_lut = None       # gamma top-up with OCIO (see build_gamma_lut)
        self._cc_lut = None          # CC over a QC effect result (build_cc_lut)
        self._fx_lut = None          # NEUTRAL_LUT + input linearisation
        self._fx_lut_f = None        # ... and its unclipped float twin
        self._fx_lut_src = None      # the table _fx_lut was built from
        self.effect = fx.NONE
        self.effect_params = {}      # settings of the active effect (see overlay.py)
        # Threads the blur-heavy QC checks are computed on (cv_qc_threads).
        # They are the only expensive thing left on the GUI thread, and they
        # are what limits the DISPLAY rate - see effects._apply_banded.
        self.qc_threads = 1
        # Playing? Then the margin around the visible area is dropped - see
        # _visible_box. Set by the panel from its play button.
        self.playing = False
        # Annotation mode. The store is shared by the panel (one set of notes
        # for the shot, not one per window); the tool is None unless a pencil
        # or the text button is armed.
        self.annotations = None
        self.annot_tool = None       # None | "draw" | "text"
        self.annot_color = 0
        self.annot_pen = annotate.LINE_W
        self.annot_text = annotate.TEXT_H
        self.annot_frame = 0
        self._stroke = None          # the stroke being drawn, in image pixels
        self._note_drag = None       # the note being dragged, while it is held
        # What the last redraw cost, for the status line. Covers the WHOLE
        # display path - crop, colour transform and the QC check when one is on
        # - so it is the rate the picture itself can be produced at, in every
        # mode. Which zoom is cheap is genuinely unobvious (the visible area
        # grows as you zoom out), so this is the number to read.
        self.last_render_ms = 0.0
        self.last_render_px = 0
        self.last_render_scale = (1, 1)      # (display step, compute step)
        # Set when a comparison had to fit B onto A. Shown in the status
        # line - a difference over a resampled input is not the same
        # measurement as one over two matching plates, and the person
        # reading it has to know.
        self.last_resample = ""
        # ON = never render a check coarse, not even while playing: a QC check
        # showing an approximation is a check you cannot trust (cv_qc_full_play).
        self.qc_full_play = True

        self._zoom = 0.0             # 0 = fit
        self._pan = [0.0, 0.0]
        self._drag = None
        self._syncing = False        # currently taking the view from the other window

        # Progressive rendering for the blur-heavy checks: coarse (and fast)
        # while playing or moving, then refined to full quality once things go
        # quiet. _fast is the current mode; the timer fires the refinement.
        self._fast = False
        self._refine_timer = QtCore.QTimer(self)
        self._refine_timer.setSingleShot(True)
        self._refine_timer.setInterval(130)
        self._refine_timer.timeout.connect(self._refine_now)
        # In Wipe this window is drawn OVER the other one. Below 1.0 the
        # background must not be filled - otherwise the image would blend with
        # the grey fill instead of with the other input. Siblings draw into one
        # buffer in z order, so whatever we do not draw stays from the window
        # underneath.
        self._opacity = 1.0
        self.last_error = None
        self._note = None
        self._extremes = None        # (min, max) of the frame - frame_extremes
        self._extremes_for = None    # frame AND colour space they were taken in            # e.g. "previous frame missing"

    # ------------------------------------------------------------- content
    def set_frame(self, arr, prev=None, other=None, matte=None):
        """`prev` = the previous frame (temporal check),
        `other` = the same frame from the other input (difference),
        `matte` = the DiMatte input frame (mattes in the RGBA channels)."""
        self._frame = arr
        self._prev = prev
        self._other = other
        self._matte = matte
        self._dirty = True
        self._begin_fast()          # a new frame (playback / scrub) -> render coarse
        self.update()

    def set_matte(self, channels, lightness=1.0, gain=1.0, gamma=1.0):
        """Which matte channels to draw over the image and how (DiMatte mode)."""
        channels = tuple(bool(c) for c in channels)
        shape = (float(lightness), float(gain), float(gamma))
        if (channels, shape) == (self.matte_channels, self.matte_shape):
            return
        self.matte_channels = channels
        self.matte_shape = shape
        self.invalidate()

    def matte_active(self):
        return any(self.matte_channels)

    def set_color(self, gain=None, gamma=None, channels=None, saturation=None):
        rebuild = False
        if saturation is not None and abs(saturation - self.saturation) > 1e-6:
            self.saturation = float(saturation)
            self._sat_matrix = build_saturation_matrix(self.saturation)
            self._dirty = True
        if gain is not None and abs(gain - self.gain) > 1e-9:
            self.gain = float(gain); rebuild = True
        if gamma is not None and abs(gamma - self.gamma) > 1e-9:
            self.gamma = float(gamma); rebuild = True
        if channels is not None and int(channels) != self.channels:
            self.channels = int(channels)
            self._dirty = True
        if rebuild:
            self._rebuild_luts()
            self._dirty = True
        self.update()

    def _rebuild_luts(self):
        self._lut = nukelut.display_lut(self.nuke_display, self.nuke_input,
                                        self.gain, self.gamma)
        self._gamma_lut = build_gamma_lut(self.gamma)
        self._cc_lut = build_cc_lut(self.gain, self.gamma)

    def set_ocio(self, transform):
        """Switches the OCIO path on/off (transform = ocio.DisplayTransform or None)."""
        if transform is self.ocio:
            return
        self.ocio = transform
        self.invalidate()

    def set_nuke_color(self, display, input_space):
        """The built-in mode: display + input space (see nukelut)."""
        if (display, input_space) == (self.nuke_display, self.nuke_input):
            return
        self.nuke_display = display
        self.nuke_input = input_space
        self._rebuild_luts()
        self.invalidate()

    def ocio_active(self):
        return self.ocio is not None and self.ocio.ready()

    def ocio_key(self):
        """Something that changes when the OCIO transform does.

        Used to date a cached measurement (see frame_extremes): the same frame
        read through a different input space has different values, and a cache
        that only knew about the frame would keep answering with the old ones.
        """
        return id(self.ocio) if self.ocio_active() else None

    # ---- input linearisation; works the same in both modes -----------------
    def is_linear_input(self):
        if self.ocio_active():
            return self.ocio.is_linear_input()
        return self.nuke_input == nukelut.DEFAULT_INPUT

    def linear_table(self):
        """65536 -> half. None when the input is already linear or a table is not enough."""
        if self.ocio_active():
            return self.ocio.linear_table()
        return nukelut.linear_table(self.nuke_input)

    def linearize_fn(self):
        """A half (h,w,3) -> half (h,w,3) function, or None. For the scopes."""
        if self.is_linear_input():
            return None
        table = self.linear_table()
        if table is not None:
            return lambda a: table[a.view(np.uint16)]
        if self.ocio_active():
            def exact(a):
                buf = np.ascontiguousarray(a.astype(np.float32))
                return self.ocio.to_linear(buf).astype(np.float16)
            return exact
        return None

    def invalidate(self):
        """Forces a redraw even when the object has not changed.

        Needed after re-baking OCIO: the transform is still THE SAME object,
        only with a different cube inside, so nothing would be redrawn and the
        change would only show up when moving to the next frame.
        """
        self._dirty = True
        self._rendered = None
        self.update()

    def _begin_fast(self):
        """Enter the coarse, fast render and arm the refinement. Only the
        blur-heavy checks are slow enough to bother; the rest already render at
        full rate, so for them this does nothing and no refinement is armed.
        """
        if self.qc_full_play:
            return                          # real data always - see qc_full_play
        if self.effect in fx.BLUR_HEAVY:
            self._fast = True
            self._dirty = True
            self._refine_timer.start()      # restart: refine when it goes quiet

    def _refine_now(self):
        """The view has been still for a moment - re-render at full quality."""
        if self._fast:
            self._fast = False
            self._dirty = True
            self.update()

    def _ss_budget(self):
        """Compute budget for the blur-heavy checks: small while interacting.

        qc_full_play wins outright, so flipping it mid-playback takes effect at
        once rather than waiting for the coarse render to expire.
        """
        if self.qc_full_play:
            return GRAIN_SS_BUDGET
        return GRAIN_FAST_BUDGET if self._fast else GRAIN_SS_BUDGET

    def set_effect(self, effect, params=None):
        """A QC effect (see effects.py). NONE = ordinary display."""
        if effect != self.effect:
            self.effect = effect
            self.effect_params = dict(params or fx.defaults(effect))
            self._dirty = True
            self._rendered = None      # canvas needs a different crop than the rest
            self.update()
        elif params is not None:
            self.set_effect_params(params)

    def set_effect_params(self, params):
        """A slider moved in the overlay - redraw only when it really differs."""
        params = dict(params or {})
        if params == self.effect_params:
            return
        self.effect_params = params
        self._dirty = True
        self._rendered = None          # canvas may want a different crop
        self._begin_fast()             # dragging a slider -> render coarse, then refine
        self.update()

    def _pick_step(self):
        """Which pixel to take every time.

        When the image is scaled down into the window, there is no point
        computing pixels that will not be seen anyway. The step is picked so the
        computed image lands as close to SCREEN RESOLUTION as a whole step can.

        CAREFUL with powers of two: the step used to double, so the real
        overhead swung between 1x and 2x screen resolution. 6K in a window fell
        on the worst end (step 2 = 1.9x, 4.8 Mpx per frame). An arbitrary whole
        step levels it out at ~1.3x (step 3, 2.1 Mpx) - measured 71 -> 32 ms.

        ROUNDED, not truncated. Truncating guarantees the result is never below
        screen resolution, but it also means that just under a whole step the
        picture is computed at up to 4x the pixels that are shown: measured on
        4K, zoom 59 % computed the entire 8.85 Mpx plate to fill a 2.2 Mpx
        window, while zoom 50 % - where the step finally ticked over - computed
        2.2 Mpx and ran fast. Rounding removes that cliff (59 % is now 5.6x
        cheaper) at the cost of at most a 1.5x magnification of the computed
        image, so in that band the picture is a touch softer and fine detail can
        alias. Deliberate trade: the band 50-99 % was the slowest place in the
        player and it is where a whole frame is usually reviewed.
        """
        z = self._effective_zoom()
        if z >= 1.0:
            return 1
        return max(1, min(8, int(round(1.0 / z))))

    def _visible_box(self, z, margin=None):
        """The area of the image (x0,y0,x1,y1) that is visible, plus a margin.

        Thanks to the margin, a small pan needs no recomputation at all - the
        image is already drawn a bit further than what is visible. Pass
        margin=0 for the box that is EXACTLY on screen (the scopes want that -
        see visible_linear).

        WHILE PLAYING there is no margin. It buys nothing there - every frame
        is new, so the whole crop is recomputed anyway - and a 0.35 margin is
        1.35^2 = 1.8x the pixels through the display transform. Measured on 4K
        that is the difference between 25 and ~40 fps at 92 % zoom. The moment
        playback stops the margin is back, so panning stays free.
        """
        w, h = self.image_size
        if not w or not h:
            return 0, 0, 0, 0
        vw, vh = max(1, self.width()), max(1, self.height())
        # the centre of the window corresponds to image point (w/2 + pan) -
        # see paintEvent
        cx = w / 2.0 + self._pan[0]
        cy = h / 2.0 + self._pan[1]
        if margin is None:
            if self.playing:
                margin = 0.0
            else:
                margin = EFFECT_MARGIN if self.effect != fx.NONE else self.margin
        half_w = vw / (2.0 * z) * (1.0 + margin)
        half_h = vh / (2.0 * z) * (1.0 + margin)
        x0 = int(max(0, cx - half_w))
        y0 = int(max(0, cy - half_h))
        x1 = int(min(w, cx + half_w + 1))
        y1 = int(min(h, cy + half_h + 1))
        return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)

    def _covers(self, box, step):
        """Is what we need already drawn?"""
        if self._rendered is None or self._qimage is None:
            return False
        rx0, ry0, cols, rows, rstep = self._rendered
        if rstep != step:
            return False
        x0, y0, x1, y1 = box
        return (rx0 <= x0 and ry0 <= y0
                and rx0 + cols * rstep >= x1 and ry0 + rows * rstep >= y1)

    def _effect_step(self, box, step):
        """Coarsens the step so the effect fits under its pixel ceiling.

        The blurring checks have a lower ceiling and their step is raised BY
        ONE, not doubled. Doubling overshoots the ceiling - depending on the
        zoom, once down to a quarter and another time barely at all, so the
        cost jumped between 37 and 93 ms. Going by one always stops just under
        the ceiling and the cost is flat.
        """
        x0, y0, x1, y1 = box
        s = max(1, step)
        if self.effect in fx.BLUR_HEAVY:
            # Bound the COMPUTE (not the display) by the supersample budget.
            # Pick the coarsest-affordable compute step `es`, then the display
            # step is the screen step rounded UP to a multiple of es, so the
            # result averages down to it cleanly (see _supersample_step /
            # _block_mean). Zoomed in the crop is small, es stays 1 and the
            # display step stays 1 - crisp, no average-down. Zoomed out es rises
            # and the result is averaged, so the grain stays fine and steady
            # instead of crawling. This replaces the old "raise the display step
            # to fit", which smeared the zoomed-in view.
            area = max(1, x1 - x0) * max(1, y1 - y0)
            budget = self._ss_budget()
            es = 1
            while es < 16 and area / float(es * es) > budget:
                es += 1
            return ((s + es - 1) // es) * es
        while s < 16 and ((x1 - x0) // s) * ((y1 - y0) // s) > EFFECT_PIXEL_BUDGET:
            s *= 2
        return s

    # How much MORE than the screen shows a blur-heavy check may compute. One
    # 2x2 average is what stops the grain crawling; beyond that the extra
    # samples are thrown away by the averaging and buy nothing you can see.
    SUPERSAMPLE_MAX = 2

    def _supersample_step(self, box, step):
        """For a blur-heavy check: the finest step (a divisor of `step`) that
        fits the supersample budget, but never finer than SUPERSAMPLE_MAX times
        the display step. The effect is computed at that step and the result
        averaged down by step // estep, so the grain is computed on real
        neighbours (fine, and it does not crawl). estep == step means no
        supersampling - the old point-subsample path.

        The cap matters when zoomed out, where the visible area is the whole
        plate but the screen shows a thumbnail of it: budget alone picked the
        finest affordable step and ended up computing 2.21 Mpx to draw 0.13 Mpx
        - 17x oversampled, and the frame rate went DOWN as you zoomed out, which
        is the opposite of what anyone expects.
        """
        if step <= 1 or self.effect not in fx.BLUR_HEAVY:
            return step
        x0, y0, x1, y1 = box
        area = max(1, x1 - x0) * max(1, y1 - y0)
        budget = self._ss_budget()
        finest = max(1, -(-step // self.SUPERSAMPLE_MAX))    # ceil
        for es in range(finest, step + 1):
            if step % es == 0 and area / float(es * es) <= budget:
                return es
        return step


    def _isolate_channel(self, arr):
        """The selected channel alone, as (h,w,1). With RGB nothing is copied.

        Thanks to this the R/G/B/A/Y keys work in QC modes too: the check is
        computed DIRECTLY FROM THAT CHANNEL (e.g. grain in red only), which is
        exactly what switching channels is for.

        ONE channel, not the same one written into three. Three copies made the
        check cost MORE than plain RGB - it did the identical arithmetic three
        times over and threw two thirds of it away: measured on 2.2 Mpx, grain
        in red was 31 ms against 25 ms for full RGB, and in luminance 48 ms.
        On one channel it is 7 ms, for a bit-identical result.
        """
        if self.channels == CH_RGB:
            return arr
        if self.channels == CH_LUMA:
            src = _luma_bits(arr.view(np.uint16)).view(np.float16)
        elif self.channels in (CH_R, CH_G, CH_B):
            src = arr[:, :, {CH_R: 0, CH_G: 1, CH_B: 2}[self.channels]]
        else:                                     # alpha
            src = arr[:, :, 3]
        return np.ascontiguousarray(src[:, :, None])

    def _render_effect(self, arr, prev_crop, es=1):
        """The QC check. It separates two things that are often confused:

        The INPUT transform (log -> linear) decides WHAT the numbers mean -
        every check needs that. Without it grain would be computed from encoded
        values and the value map would throw log mid grey into a completely
        different band.

        The DISPLAY is NOT used by the checks at all - they always run on a
        fixed built-in sRGB, whether Color management is on Nuke or OCIO. Two
        reasons: a check should give the same result regardless of the monitor
        you picked, and it should not cost 4x more (OCIO 40 ms against 10 ms)
        for colours you cannot see in a false-colour display anyway.

        The canvas check never gets here - it is not a computation, just a
        shifted crop, and it is displayed completely normally including the
        chosen display transform.
        """
        arr, prev_crop, lut, lut_f = self._effect_inputs(arr, prev_crop)

        # The saturation check measures the RATIO BETWEEN channels, so it has
        # to be computed from the whole of RGB - an isolated channel is
        # monochrome and would have nothing to measure. The channel is
        # therefore picked from the FINISHED result: you see how that channel
        # contributes to the resulting colour. The other checks isolate before
        # the computation, where it makes sense (grain and exposure per channel).
        if self.effect == fx.SAT:
            out = fx.apply(fx.SAT, arr, lut, self.effect_params, es,
                           self.qc_threads)
            return None if out is None else self._isolate_result(out, arr)

        src = self._isolate_channel(arr)
        if self.effect == fx.TEMPORAL:
            out = fx.temporal(src, self._isolate_channel(prev_crop),
                              lut, self.effect_params, self.qc_threads)
        elif self.effect == fx.DIFF:
            out = fx.difference(src, self._isolate_channel(prev_crop),
                                lut, self.effect_params, self.qc_threads)
        elif self.effect == fx.HPDIFF:
            out = fx.hp_difference(src, self._isolate_channel(prev_crop),
                                   lut_f, self.effect_params, self.qc_threads)
        else:
            # es lets grain keep a constant brightness across samplings (fast
            # vs full, and across zoom) - see effects._es_comp.
            out = fx.apply(self.effect, src, lut, self.effect_params, es,
                           self.qc_threads, lut_f)
        if out is not None and out.ndim == 3 and out.shape[2] == 1:
            out = out[:, :, 0]      # one channel stays GREY to the QImage
        return out

    def _effect_inputs(self, arr, other):
        """Data and table prepared for the QC computation.

        The value map reads VALUES (it classifies scene-linear bands) and
        luminance has to be computed from linearised channels - these two cases
        genuinely need converted data. Everything else only goes through the
        conversion into display, so baking the linearisation into the table is
        enough - and that is free (9.5 ms against 9.4 ms without it).
        """
        lut, lut_f = NEUTRAL_LUT, NEUTRAL_LUT_F
        if not self.is_linear_input():
            table = self.linear_table()
            if table is None or self.effect == fx.VALUEMAP \
                    or self.channels == CH_LUMA:
                arr = self._linearize(arr, table)
                other = self._linearize(other, table)
            else:
                lut, lut_f = self._effect_lut(table)
        return arr, other, lut, lut_f

    def _isolate_result(self, rgb8, arr):
        """Picks a channel out of an already computed check result (uint8 RGB)."""
        ch = self.channels
        if ch == CH_RGB:
            return rgb8
        if ch == CH_A:
            a = np.clip(arr[:, :, 3].astype(np.float32), 0.0, 1.0)
            g = (a * 255.0 + 0.5).astype(np.uint8)
        elif ch == CH_LUMA:
            g = np.clip(rgb8.astype(np.float32) @ LUMA, 0, 255).astype(np.uint8)
        else:
            g = rgb8[:, :, {CH_R: 0, CH_G: 1, CH_B: 2}[ch]]
        return np.repeat(g[:, :, None], 3, axis=2)

    def _effect_lut(self, table):
        """(display table, unclipped float twin), input linearisation baked in.

        Both are composed the same way, so the band checks see exactly the
        curve everything else does - only without the ends cut off.
        """
        if self._fx_lut_src is not table:
            self._fx_lut_src = table
            bits = np.ascontiguousarray(table).view(np.uint16)
            self._fx_lut = NEUTRAL_LUT[bits]
            self._fx_lut_f = NEUTRAL_LUT_F[bits]
        return self._fx_lut, self._fx_lut_f

    def _linearize(self, arr, table):
        """Straightens the input space into scene-linear (only where necessary)."""
        if arr is None:
            return None
        fn = self.linearize_fn()
        if fn is None:
            return arr
        out = np.empty(arr.shape, dtype=arr.dtype)
        out[:, :, :3] = fn(arr[:, :, :3])
        out[:, :, 3] = arr[:, :, 3]
        return out

    def _render_ocio(self, arr):
        """Display through OCIO. Exposure is inside the shaper, gamma at the end.

        Channel selection is handled by copying the channel into RGB and
        letting it go through the transform as grey - i.e. exactly what you see
        without OCIO too.
        """
        if self.channels == CH_RGB:
            src = arr[:, :, :3]
        elif self.channels == CH_A:
            # alpha is a 0-1 fraction, not a scene-linear colour - a display
            # transform would distort it, so it is shown raw as before
            a = np.clip(arr[:, :, 3].astype(np.float32), 0.0, 1.0)
            g = (a * 255.0 + 0.5).astype(np.uint8)
            return np.repeat(g[:, :, None], 3, axis=2)
        else:
            src = self._isolate_channel(arr)[:, :, :3]
        try:
            rgb = self.ocio.apply(src, self.gain)
        except Exception as exc:
            self.last_error = "OCIO: %s" % exc
            return None
        if self._gamma_lut is not None:
            rgb = self._gamma_lut[rgb]
        if self.channels == CH_RGB:
            rgb = self._apply_saturation(rgb)
        return rgb

    def _apply_cc(self, rgb):
        """CC over the finished image - gain, gamma and saturation over bytes.

        Used for the QC effects: it tints the result of the check without
        reaching into what the check was computed from.
        """
        if self._cc_lut is not None:
            rgb = self._cc_lut[rgb]
        return self._apply_saturation(rgb)

    def display_rgb(self):
        """The last drawn image (h,w,3) uint8, or None.

        The scopes compute from it so they show exactly what is visible.
        """
        return self._rgb

    def visible_linear(self):
        """Scene-linear data of the CURRENTLY VISIBLE area, already stepped down.

        The histogram and the waveform compute from it, so they describe the
        part you are actually looking at.

        The box is taken WITHOUT the margin, unlike the rendered one. The
        drawing keeps a 35 % reserve around the visible area so that a small
        pan needs no re-render - but the scopes are not allowed to measure it:
        with the reserve they described a third more than is on screen, and a
        pan inside it did not change their data at all, so they looked frozen.
        """
        if self._frame is None:
            return None
        x0, y0, x1, y1 = self._visible_box(self._effective_zoom(), margin=0.0)
        if x1 <= x0 or y1 <= y0:
            return self._frame
        step = max(1, self._step)
        return self._frame[y0:y1:step, x0:x1:step]

    def visible_display(self):
        """The drawn image cropped to what is REALLY on screen.

        `_rgb` holds the whole rendered region, i.e. the visible area plus the
        35 % reserve (see _visible_box). The vectorscope reads this, so without
        the crop it measured a third more than is on screen - and a pan inside
        the reserve did not change it at all, so it looked frozen.
        """
        if self._rgb is None or self._rendered is None:
            return self._rgb
        rx0, ry0, cols, rows, step = self._rendered
        x0, y0, x1, y1 = self._visible_box(self._effective_zoom(), margin=0.0)
        cx0 = max(0, (x0 - rx0) // step)
        cy0 = max(0, (y0 - ry0) // step)
        cx1 = min(cols, -(-(x1 - rx0) // step))      # ceil, so nothing is lost
        cy1 = min(rows, -(-(y1 - ry0) // step))
        if cx1 <= cx0 or cy1 <= cy0:
            return self._rgb
        return self._rgb[cy0:cy1, cx0:cx1]

    def scope_source(self):
        """Everything the scopes need - so they do not have to fetch it piecemeal."""
        return {"linear": self.visible_linear(),
                "display": self.visible_display(),
                "qc": self.effect != fx.NONE,
                "channels": self.channels,
                "gain": self.gain,
                "gamma": self.gamma,
                "sat_matrix": self._sat_matrix,
                "linearize": self.linearize_fn()}

    def scope_source_for(self, arr, rgb, look=None):
        """A scope context for a frame that is NOT the one on screen.

        Same shape as scope_source(), but over the WHOLE frame and the look the
        export is rendering - so the scopes written into a JPEG describe that
        picture, not whatever the viewport happened to be showing.
        """
        look = look or self.current_look()
        # ALL FOUR channels, exactly as visible_linear() hands them over - the
        # scopes reach for the alpha themselves (see scopes._planes_float), so
        # an RGB-only slice quietly gives them nothing to measure.
        return {"linear": arr,
                "display": rgb,
                "qc": look.get("effect", fx.NONE) != fx.NONE,
                "channels": int(look.get("channels", CH_RGB)),
                "gain": float(look.get("gain", 1.0)),
                "gamma": float(look.get("gamma", 1.0)),
                "sat_matrix": build_saturation_matrix(
                    float(look.get("saturation", 1.0))),
                "linearize": self.linearize_fn()}

    def _apply_saturation(self, rgb):
        """Saturation in the display domain (like CC in a viewer).

        It does not go into the LUT - that maps one value to one value, whereas
        saturation mixes channels together. We do it with a single matrix
        multiply (BLAS, runs on several cores): measured on 1080p 15 ms against
        44 ms for writing per channel, deviation at most 1/255. At a saturation
        of 1.0 it is skipped entirely, so ordinary display costs nothing extra,
        and it is computed only from the visible crop.
        """
        # A grey image (a single channel, see _isolate_channel) has no colour to
        # mix, so there is nothing for the matrix to do - and it does not have
        # the three columns the matrix expects either.
        if self._sat_matrix is None or rgb.ndim != 3 or rgb.shape[2] != 3:
            return rgb
        flat = rgb.reshape(-1, 3).astype(np.float32) @ self._sat_matrix
        return np.clip(flat, 0, 255).astype(np.uint8).reshape(rgb.shape)

    def _render(self, box, step):
        """Linear half -> uint8 RGB through the LUT, only for the crop `box`."""
        self._dirty = False
        self._qimage = None
        self._rgb = None
        self._rendered = None
        self.last_resample = ""      # re-decided below, per render
        arr = self._frame
        if arr is None or arr.ndim != 3 or arr.shape[2] < 4:
            return
        x0, y0, x1, y1 = box
        prev_crop = None
        if fx.needs_previous(self.effect):
            if self._prev is None or self._prev.shape != arr.shape:
                self.last_error = None
                self._note = "(temporal: previous frame missing)"
                return
            prev_crop = self._prev[y0:y1:step, x0:x1:step]
        if fx.needs_other(self.effect):
            if self._other is None:
                self.last_error = None
                self._note = "(difference: the second input is not connected)"
                return
            # Different sizes used to stop the check outright. Now BOTH inputs
            # are fitted onto the compare size - a 4K render against a 6K plate
            # is a comparison people actually want. It is OUR resample, not a
            # Reformat, so the note says so: on a difference check the filter
            # used is part of what you are looking at.
            #
            # The crop box was worked out against THIS window's frame, so when
            # the compare size is something else the whole thing is computed at
            # the frame's own size and only the comparison inputs are fitted.
            other = resample.fit(self._other, arr.shape)
            self.last_resample = resample.label(self._other.shape, arr.shape)
            prev_crop = other[y0:y1:step, x0:x1:step]
        # Blur-heavy checks compute at a finer step (es) and average the result
        # down by step // es; everything else has es == step (no change).
        es = self._supersample_step(box, step)
        if self.effect == fx.CANVAS:
            # shifting the coordinates with wraparound -> the crop is already
            # "swapped". Assembled from contiguous blocks rather than gathered
            # per pixel - see effects.canvas_crop, it is 3x the difference.
            arr = fx.canvas_crop(arr, x0, y0, x1, y1, step, self.effect_params)
        else:
            arr = arr[y0:y1:es, x0:x1:es]
        if arr.size == 0:
            return
        rows, cols = arr.shape[0], arr.shape[1]
        # how much this redraw is really chewing through, whatever path follows
        self.last_render_px = rows * cols
        self.last_render_scale = (step, es)
        try:
            # Difference in overlay mode shows the REAL image - it therefore
            # has to go through the chosen display transform just like ordinary
            # display, otherwise the whole plate would come out lighter (a
            # fixed sRGB is flatter than, say, rec1886 or an OCIO view). Only
            # the difference mask is computed and the marks are drawn onto the
            # finished image afterwards.
            self._diff_mask = None
            if self.effect == fx.DIFF and fx.diff_is_overlay(self.effect_params):
                if prev_crop is not None:
                    a, b, fx_lut, _f = self._effect_inputs(arr, prev_crop)
                    self._diff_mask = fx.difference_mask(
                        self._isolate_channel(a), self._isolate_channel(b),
                        fx_lut, self.effect_params)
            elif self.effect not in (fx.NONE, fx.CANVAS):
                # A QC effect decides the whole output. The channel selection
                # is applied BEFORE it (see _isolate_channel), so the check
                # runs over the channel you switched to. Canvas never gets
                # here - it is already handled by the shifted crop above and is
                # displayed completely normally.
                rgb = self._render_effect(arr, prev_crop, es)
                if rgb is None:
                    return
                rgb = self._apply_cc(rgb)
                f = step // es                  # >1 only for the blur-heavy path
                if f > 1:                       # full-res grain -> screen res
                    rgb = _block_mean(rgb, f)
                self._finish(rgb, box, step, rgb.shape[1], rgb.shape[0])
                return
            if self.ocio_active():
                rgb = self._render_ocio(arr)
                if rgb is None:
                    return
                self._finish(rgb, box, step, cols, rows)
                return

            # Single channels stay SINGLE-CHANNEL all the way to the QImage.
            # The grey used to be copied into three channels, which is
            # pointless extra work: measured on 2.2 Mpx 10 ms a frame (for the
            # R channel that was most of the whole display). Qt handles a grey
            # image directly.
            bits = arr.view(np.uint16)               # bits of the half values
            if self.channels == CH_RGB:
                rgb = self._lut[bits[:, :, :3]]
                rgb = self._apply_saturation(rgb)    # RGB only, not on greys
            elif self.channels in (CH_R, CH_G, CH_B):
                idx = {CH_R: 0, CH_G: 1, CH_B: 2}[self.channels]
                rgb = self._lut[bits[:, :, idx]]
            elif self.channels == CH_A:
                # alpha raw (0-1 -> 0-255), no exposure and no display transform
                a = np.clip(arr[:, :, 3].astype(np.float32), 0.0, 1.0)
                rgb = (a * 255.0 + 0.5).astype(np.uint8)
            else:                                    # luminance
                rgb = self._lut[_luma_bits(bits)]

            self._finish(rgb, box, step, cols, rows)
        except Exception as exc:
            self.last_error = "render: %s" % exc
            self._qimage = None
            self._rendered = None

    def current_look(self):
        """What the window is showing right now, enough to reproduce it later.

        Stored with an annotation so the export can put the note back on the
        picture it was drawn on - a circle around a grain problem says nothing
        over a plain plate.
        """
        return {"effect": self.effect,
                "params": dict(self.effect_params),
                "channels": self.channels,
                "gain": self.gain,
                "gamma": self.gamma,
                "saturation": self.saturation}

    def render_full(self, arr, look=None):
        """A WHOLE scene-linear frame as uint8 RGB (h,w,3), for the export.

        `look` is a dict from current_look(); without one the window's present
        state is used. Not the crop path: an export must not depend on where
        the viewport happened to be, nor on the coarser step a zoomed-out view
        is computed at.
        """
        if arr is None or arr.ndim != 3 or arr.shape[2] < 4:
            return None
        look = look or self.current_look()
        effect = look.get("effect", fx.NONE)
        channels = int(look.get("channels", CH_RGB))
        gain = float(look.get("gain", 1.0))
        gamma = float(look.get("gamma", 1.0))

        # Its own tables, built from the look - the window is not disturbed,
        # and an export started while someone is dragging a slider still comes
        # out as the note was made.
        if abs(gain - self.gain) > 1e-9 or abs(gamma - self.gamma) > 1e-9:
            lut = nukelut.display_lut(self.nuke_display, self.nuke_input,
                                      gain, gamma)
        else:
            lut = self._lut
        sat = build_saturation_matrix(float(look.get("saturation", 1.0)))

        if effect not in (fx.NONE, fx.CANVAS):
            # A check decides the whole picture. Both inputs are needed for the
            # comparisons; without the other one there is nothing to compare,
            # so the plain image is the honest answer.
            src = self._isolate_for(arr, channels)
            _a, _b, fx_lut, fx_lut_f = self._effect_inputs(arr, None)
            if effect in (fx.DIFF, fx.HPDIFF):
                # fitted the same way the viewer fitted it, so the exported
                # JPEG is the picture the note was written on
                other = resample.fit(self._other, arr.shape)
                if other is None:
                    rgb = None
                else:
                    mate = self._isolate_for(other, channels)
                    # the export must go through the same tables the viewer
                    # did, or a note would point at something else
                    fn = fx.difference if effect == fx.DIFF else fx.hp_difference
                    rgb = fn(src, mate,
                             fx_lut if effect == fx.DIFF else fx_lut_f,
                             look.get("params"), self.qc_threads)
            elif effect == fx.TEMPORAL:
                rgb = None                  # a still frame has no previous one
            elif effect == fx.SAT:
                out = fx.apply(fx.SAT, arr, fx_lut, look.get("params"), 1,
                               self.qc_threads)
                rgb = None if out is None else self._isolate_result(out, arr)
            else:
                rgb = fx.apply(effect, src, fx_lut, look.get("params"), 1,
                               self.qc_threads, fx_lut_f)
            if rgb is not None:
                if rgb.ndim == 3 and rgb.shape[2] == 1:
                    rgb = rgb[:, :, 0]
                return self._as_rgb888(self._apply_cc(rgb))

        if self.ocio_active():
            rgb = self._render_ocio(arr)
        else:
            bits = arr.view(np.uint16)
            if channels == CH_RGB:
                rgb = lut[bits[:, :, :3]]
                if sat is not None:
                    flat = rgb.reshape(-1, 3).astype(np.float32) @ sat
                    rgb = np.clip(flat, 0, 255).astype(np.uint8).reshape(rgb.shape)
            elif channels in (CH_R, CH_G, CH_B):
                rgb = lut[bits[:, :, {CH_R: 0, CH_G: 1, CH_B: 2}[channels]]]
            elif channels == CH_A:
                a = np.clip(arr[:, :, 3].astype(np.float32), 0.0, 1.0)
                rgb = (a * 255.0 + 0.5).astype(np.uint8)
            else:
                rgb = lut[_luma_bits(bits)]
        return self._as_rgb888(rgb)

    @staticmethod
    def _as_rgb888(rgb):
        """(h,w,3) uint8, contiguous - what QImage wants."""
        if rgb is None:
            return None
        if rgb.ndim == 2:                  # grey -> colour
            rgb = np.repeat(rgb[:, :, None], 3, axis=2)
        return np.ascontiguousarray(rgb)

    def _isolate_for(self, arr, channels):
        """_isolate_channel, but for a GIVEN channel choice (see render_full)."""
        was, self.channels = self.channels, channels
        try:
            return self._isolate_channel(arr)
        finally:
            self.channels = was

    def _finish(self, rgb, box, step, cols, rows):
        """The common tail of every path: DiMatte mattes, buffer, QImage.

        The mattes are drawn RIGHT AT THE END, over the finished image - they
        are "there is a matte here" marks, not data that should go through the
        colour path.
        """
        mask = self._diff_mask
        if mask is not None and mask.shape == rgb.shape[:2]:
            if rgb.ndim == 2:                        # grey image -> colour
                rgb = np.repeat(rgb[:, :, None], 3, axis=2)
            rgb = np.ascontiguousarray(rgb)
            rgb[mask] = fx.difference_color(self.effect_params)
        crop = self._matte_crop(box, step, rgb.shape[:2])
        if crop is not None:
            rgb = fx.matte_overlay(rgb, crop, self.matte_channels,
                                   *self.matte_shape)
        rgb = np.ascontiguousarray(rgb)
        self._rgb = rgb                              # KEEPS the buffer alive
        self._qimage = _make_qimage(rgb)
        self._rendered = (box[0], box[1], cols, rows, step)

    def _matte_crop(self, box, step, shape):
        """The same crop out of the matte, or None when it cannot be used."""
        if not self.matte_active() or self._matte is None or self._frame is None:
            return None
        if self._matte.shape[:2] != self._frame.shape[:2]:
            return None                              # other resolution - do not mix
        x0, y0, x1, y1 = box
        crop = self._matte[y0:y1:step, x0:x1:step]
        return crop if crop.shape[:2] == tuple(shape) else None

    # --------------------------------------------------------------- view
    def current_frame_array(self):
        """The current scene-linear frame (to find out its size in memory)."""
        return self._frame

    @property
    def image_size(self):
        if self._frame is None:
            return (0, 0)
        return (self._frame.shape[1], self._frame.shape[0])

    def _fit_zoom(self):
        w, h = self.image_size
        if not w or not h:
            return 1.0
        return min(self.width() / float(w), self.height() / float(h))

    def _effective_zoom(self):
        return self._fit_zoom() if self._zoom <= 0.0 else self._zoom

    def zoom_percent(self):
        return self._effective_zoom() * 100.0

    def fit(self):
        self._zoom = 0.0
        self._pan = [0.0, 0.0]
        self._moved()

    def zoom_1_1(self):
        self._zoom = 1.0
        self._moved()

    def set_zoom_percent(self, percent):
        """Zoom straight to a value, keeping the point in the MIDDLE of the
        window where it is (the same as the wheel does around the cursor, and
        the same as picking a zoom in the Nuke Viewer).
        """
        new = max(0.02, min(64.0, float(percent) / 100.0))
        if abs(new - self._effective_zoom()) < 1e-6:
            return
        self._zoom = new
        self._moved()

    # ---- the shared view in double display --------------------------------
    def _moved(self):
        """Zoom or pan changed - redraw and tell the other view."""
        self._begin_fast()          # moving the view -> render coarse, refine after
        self.update()
        if not self._syncing:
            self.viewportChanged.emit()

    def viewport(self):
        return (self._zoom, self._pan[0], self._pan[1])

    def set_viewport(self, state):
        """Takes the view over from the other window. Deliberately does NOT
        emit - otherwise the two windows would bounce the signal back and forth
        forever."""
        zoom, px, py = state
        if (zoom, px, py) == (self._zoom, self._pan[0], self._pan[1]):
            return
        self._syncing = True
        try:
            self._zoom = zoom
            self._pan = [px, py]
        finally:
            self._syncing = False
        self._begin_fast()          # the synced view moved -> render coarse
        self.update()

    def wheelEvent(self, event):
        if self._frame is None:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        old = self._effective_zoom()
        new = max(0.02, min(64.0, old * (1.25 if delta > 0 else 1 / 1.25)))
        pos = event_pos(event)
        dx = pos.x() - self.width() / 2.0
        dy = pos.y() - self.height() / 2.0
        self._pan[0] += dx * (1.0 / old - 1.0 / new)
        self._pan[1] += dy * (1.0 / old - 1.0 / new)
        self._zoom = new
        self._moved()
        event.accept()

    def mousePressEvent(self, event):
        # even a plain click makes this the active window (scopes, readout)
        self.picked.emit()
        if event.button() == QtCore.Qt.LeftButton and self.annot_tool:
            # A tool is armed, so the left button writes instead of panning -
            # the middle button still pans, which is how you move around while
            # annotating without putting the pencil down.
            at = self._widget_to_image(event_pos(event))
            if at is not None:
                if self.annot_tool == "text":
                    self._press_note(float(at[0]), float(at[1]))
                else:
                    self._stroke = [(float(at[0]), float(at[1]))]
                return
        if event.button() == QtCore.Qt.LeftButton:
            self._drag = event_pos(event)
            self.setCursor(QtCore.Qt.ClosedHandCursor)
        elif event.button() == QtCore.Qt.MiddleButton and self.annot_tool:
            self._drag = event_pos(event)
            self.setCursor(QtCore.Qt.ClosedHandCursor)

    def _press_note(self, x, y):
        """The text tool was pressed on the image.

        Landing ON a note takes hold of it instead of asking for a new one:
        the note is only opened for editing on RELEASE, and only if it was not
        dragged anywhere. That way one press does both jobs - a click edits, a
        drag moves - and neither can happen by accident.
        """
        index = None
        if self.annotations is not None:
            index = self.annotations.text_at(
                self.annot_frame, x, y, self.current_look(),
                self.image_size[0], self.image_size[1])
        if index is None:
            self.textWanted.emit(x, y)      # empty ground - a new note
            return
        ox, oy = self.annotations.text_pos(self.annot_frame, index)
        self._note_drag = {"index": index, "grab": (x, y),
                           "origin": (ox, oy), "moved": False}
        self.setCursor(QtCore.Qt.SizeAllCursor)

    def mouseMoveEvent(self, event):
        p = event_pos(event)
        if self._note_drag is not None:
            at = self._widget_to_image(p, clamp=True)
            if at is not None:
                gx, gy = self._note_drag["grab"]
                ox, oy = self._note_drag["origin"]
                if self.annotations.move_text(
                        self.annot_frame, self._note_drag["index"],
                        ox + (at[0] - gx), oy + (at[1] - gy)):
                    self._note_drag["moved"] = True
                    self.update()
            return
        if self._stroke is not None:
            at = self._widget_to_image(p)
            if at is not None:
                self._stroke.append((float(at[0]), float(at[1])))
                self.update()
            return
        if self._drag is None:
            self._emit_probe(p)
            return
        z = self._effective_zoom()
        self._pan[0] -= (p.x() - self._drag.x()) / z
        self._pan[1] -= (p.y() - self._drag.y()) / z
        self._drag = p
        self._moved()

    def leaveEvent(self, _event):
        if not self.probe_frozen:
            self.probeChanged.emit(None)

    def mouseReleaseEvent(self, _event):
        if self._note_drag is not None:
            drag, self._note_drag = self._note_drag, None
            self.unsetCursor()
            if drag["moved"]:
                self.annotated.emit()       # it was dragged: put it down
            else:
                gx, gy = drag["grab"]       # it was only clicked: open it
                self.textWanted.emit(gx, gy)
            return
        if self._stroke is not None:
            stroke, self._stroke = self._stroke, None
            if (self.annotations is not None
                    and self.annotations.add_stroke(
                        self.annot_frame, stroke, self.annot_color,
                        self.annot_pen, self.current_look())):
                self.annotated.emit()
            self.update()
        self._drag = None
        self.unsetCursor()

    # -------------------------------------------------------------- probe
    def _widget_to_image(self, pos, clamp=False):
        """A point in the window -> image pixel coordinates, or None outside.

        Exactly the inverse of the transform used to draw in paintEvent.
        `clamp` holds the point at the edge instead of dropping it, which is
        what dragging wants: letting go of a note the moment the pointer left
        the picture would leave it wherever it happened to be.
        """
        w, h = self.image_size
        if not w or not h:
            return None
        z = self._effective_zoom()
        ox = self.width() / 2.0 - (self._pan[0] + w / 2.0) * z
        oy = self.height() / 2.0 - (self._pan[1] + h / 2.0) * z
        ix = int((pos.x() - ox) / z)
        iy = int((pos.y() - oy) / z)
        if clamp:
            return max(0, min(w - 1, ix)), max(0, min(h - 1, iy))
        if 0 <= ix < w and 0 <= iy < h:
            return ix, iy
        return None

    def _emit_probe(self, pos):
        if self.probe_frozen:
            return
        self.probeChanged.emit(self.probe_at(self._widget_to_image(pos)))

    def frame_extremes(self):
        """(min, max) scene-linear over the WHOLE frame, or None.

        Not over a subsample: a single stray pixel - one negative, one value
        that should not be up at 60 - is exactly what this is for, and a
        subsample is precisely what would miss it.

        Done on the half BITS rather than on the floats. For a positive half
        the bit pattern sorts in value order, and a negative one has the top
        bit set, so as unsigned it sorts above every positive; that gives both
        ends from two integer passes instead of a float conversion of the whole
        frame. Measured on 6K: 133 ms as floats, 42 ms this way.

        Cached against the frame object, because nothing about it changes until
        a new frame arrives.
        """
        arr = self._frame
        if arr is None or arr.ndim != 3 or arr.shape[2] < 3:
            return None
        key = (id(arr), self.nuke_input, self.ocio_key())
        if self._extremes_for == key:
            return self._extremes

        # WHICH HALF VALUES OCCUR, not what the smallest bit pattern is. A half
        # has only 65536 possible values, so one count over the frame says
        # exactly which of them are in it - and then the input transform can be
        # applied to those 65536 instead of to 28 million pixels.
        #
        # Counting rather than taking min/max of the raw bits also means no
        # assumption about the transform: a curve that is not monotonic would
        # move which value is the largest, and mapping only the two raw
        # extremes would then report a number that is not in the picture.
        bits = np.ascontiguousarray(arr[:, :, :3]).view(np.uint16)
        counts = np.bincount(bits.ravel(), minlength=65536)
        present = np.flatnonzero(counts)
        if present.size == 0:
            return None

        # present holds BIT PATTERNS, so it is viewed as half to get the raw
        # values. The linear table, on the other hand, already holds VALUES
        # (nukelut.linear_table / ocio.linear_table both return float16) and is
        # only INDEXED by the bits - casting it to uint16 turned 0.214 into 0
        # and 33.0 into a denormal, which is how a log input came out with a
        # maximum of a billionth.
        table = None if self.is_linear_input() else self.linear_table()
        if table is not None:
            vals = np.asarray(table)[present]
        else:
            vals = present.astype(np.uint16).view(np.float16)
        vals = np.asarray(vals, dtype=np.float32)
        if table is None and not self.is_linear_input() and self.ocio_active():
            # OCIO with no table of its own - exact, and only ever over the
            # handful of values that are actually in the frame
            try:
                buf = np.ascontiguousarray(
                    vals.reshape(1, -1, 1).repeat(3, axis=2))
                vals = self.ocio.to_linear(buf)[0, :, 0]
            except Exception:
                pass

        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            return None
        out = (float(finite.min()), float(finite.max()))
        self._extremes_for, self._extremes = key, out
        return out

    def probe_at(self, xy):
        """The values of one pixel: raw, linearised and as displayed."""
        if xy is None or self._frame is None:
            return None
        ix, iy = xy
        raw = np.asarray(self._frame[iy, ix], dtype=np.float32)
        px = self._frame[iy:iy + 1, ix:ix + 1]

        lin = raw[:3]
        table = self.linear_table() if not self.is_linear_input() else None
        if table is not None:
            lin = np.asarray(table[px[:, :, :3].view(np.uint16)],
                             dtype=np.float32).reshape(3)

        if self.ocio_active():
            try:
                shown = self.ocio.apply(px[:, :, :3], self.gain).reshape(3)
            except Exception:
                shown = self._lut[px[:, :, :3].view(np.uint16)].reshape(3)
        else:
            shown = self._lut[px[:, :, :3].view(np.uint16)].reshape(3)
        if self._gamma_lut is not None and self.ocio_active():
            shown = self._gamma_lut[shown]

        lum = float(np.dot(lin, LUMA))
        return {"x": ix, "y": iy,
                # the waveform needs it to place the column marker, and only
                # the view knows how wide the picture is
                "image_w": self.image_size[0],
                "raw": raw,                     # what is in the file
                "linear": lin,                  # after the input transform
                "shown": np.asarray(shown, dtype=np.int32),
                "lum": lum,
                # stops above/below mid grey - the fastest read on exposure
                "stops": (float(np.log2(lum / 0.18)) if lum > 1e-6 else None)}

    def mouseDoubleClickEvent(self, _event):
        self.fit()

    def set_opacity(self, value):
        """Opacity of the window (used by Wipe). 1.0 = ordinary, opaque."""
        value = max(0.0, min(1.0, float(value)))
        if abs(value - self._opacity) < 1e-3:
            return
        self._opacity = value
        self.update()

    # -------------------------------------------------------------- drawing
    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        if self._opacity >= 0.999:
            painter.fillRect(self.rect(), QtGui.QColor(28, 28, 28))
        else:
            # the background is not drawn, so the other input shows through
            painter.setOpacity(self._opacity)
        if self._frame is None:
            painter.setPen(QtGui.QColor(150, 150, 150))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "(no frame)")
            return

        z = self._effective_zoom()
        step = self._pick_step()
        if fx.needs_full_frame(self.effect):
            w, h = self.image_size          # canvas swaps quadrants ->
            box = (0, 0, w, h)              # it needs the whole image, not a crop
        else:
            box = self._visible_box(z)
        if self.effect not in (fx.NONE, fx.CANVAS):
            # a pixel ceiling only for the computed effects; canvas is just
            # different addressing and costs the same as ordinary display ->
            # full detail
            step = self._effect_step(box, step)
        # recomputed only when the frame/colour changed, the scale changed, or
        # the user moved outside the already drawn area (hence the margin)
        if self._dirty or step != self._step or not self._covers(box, step):
            self._step = step
            t0 = time.perf_counter()
            self._render(box, step)
            self.last_render_ms = (time.perf_counter() - t0) * 1000.0
        if self._qimage is None or self._rendered is None:
            painter.setPen(QtGui.QColor(150, 150, 150))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter,
                             self._note or "(no frame)")
            return

        w, h = self.image_size            # the FULL image size
        rx0, ry0, cols, rows, rstep = self._rendered
        # Smooth whenever the rendered image is MAGNIFIED on screen, i.e. one
        # rendered pixel covers more than one screen pixel (rstep * z > 1). That
        # is the zoomed-out case (z < 1) as before, but ALSO the coarse fast
        # grain render (rstep > 1 at z = 1): bilinear makes its half-res grain
        # read as slightly soft fine grain instead of hard blocks, so it barely
        # changes when the full-resolution refinement lands.
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform,
                              rstep * z > 1.001)
        # the top left corner of the WHOLE image on screen
        ox = self.width() / 2.0 - (self._pan[0] + w / 2.0) * z
        oy = self.height() / 2.0 - (self._pan[1] + h / 2.0) * z
        target = QtCore.QRectF(ox + rx0 * z, oy + ry0 * z,
                               cols * rstep * z, rows * rstep * z)
        painter.drawImage(target, self._qimage)

        # The notes go on LAST and use the same ox/oy/z the picture was drawn
        # with, so they sit on the pixels they were drawn on at any zoom.
        if self.annotations is not None:
            self.annotations.draw(painter, self.annot_frame, ox, oy, z,
                                  self.current_look(), w, h)
            if self._stroke and len(self._stroke) > 1:
                pen = QtGui.QPen(QtGui.QColor(*annotate.COLORS[
                    self.annot_color % len(annotate.COLORS)]))
                pen.setWidthF(max(1.0, self.annot_pen * z))
                pen.setCapStyle(QtCore.Qt.RoundCap)
                pen.setJoinStyle(QtCore.Qt.RoundJoin)
                painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
                painter.setPen(pen)
                painter.drawPolyline(QtGui.QPolygonF(
                    [QtCore.QPointF(ox + x * z, oy + y * z)
                     for x, y in self._stroke]))
