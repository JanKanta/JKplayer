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

import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from .qtcompat import QtCore, QtGui, QtWidgets, event_pos

from . import annotate
from . import resample
from . import effects as fx
from . import nukelut

MODE_NUKE, MODE_OCIO = range(2)

# ---------------------------------------------------------------------------
# The display gather, spread over the cores
#
# Turning half floats into screen bytes is one table lookup per pixel and no
# arithmetic worth the name, so a single core spends nearly all of it waiting
# for RAM - measured 625 M lookups a second, which IS numpy's ceiling for a
# random gather on one thread. Splitting it by rows is embarrassingly parallel:
# unlike the QC blurs a lookup has no neighbourhood, so there is no halo and no
# seam, and the result is bit-identical (there is a test).
#
# EIGHT BANDS, not one per core. On 4K it takes the display from 26 to 131 fps,
# of a 166 fps ceiling at sixteen - and the cores it leaves alone are the ones
# decoding the next frame. Chasing the last 25 % here would be paid for out of
# the fill rate.
DISPLAY_BANDS = max(1, min(8, os.cpu_count() or 4))

# Under this the pool costs more than the gather saves - handing work to eight
# threads and collecting it again is tens of microseconds, and a thumbnail-sized
# crop is done in less than that.
BAND_FLOOR_PX = 250000

# A POOL OF ITS OWN, not the one in effects. That one is sized by cv_qc_threads
# and rebuilds itself whenever the count changes; asking it for a different
# number here would tear it down and build it again on every single frame.
_DISPLAY_POOL = None
_DISPLAY_POOL_LOCK = threading.Lock()


def _display_pool():
    global _DISPLAY_POOL
    with _DISPLAY_POOL_LOCK:
        if _DISPLAY_POOL is None:
            _DISPLAY_POOL = ThreadPoolExecutor(
                max_workers=DISPLAY_BANDS, thread_name_prefix="exr-display")
        return _DISPLAY_POOL


def _banded(fill, shape, dtype=np.uint8):
    """Build an array by row bands. `fill(out, a, b)` writes rows [a:b).

    Falls back to one call for anything small enough that threading it would
    be a loss, and for anything with fewer rows than bands.
    """
    out = np.empty(shape, dtype)
    rows = shape[0]
    px = rows * (shape[1] if len(shape) > 1 else 1)
    if DISPLAY_BANDS <= 1 or rows < DISPLAY_BANDS or px < BAND_FLOOR_PX:
        fill(out, 0, rows)
        return out
    edges = [rows * i // DISPLAY_BANDS for i in range(DISPLAY_BANDS + 1)]
    # list() so exceptions surface here rather than being swallowed by the pool
    list(_display_pool().map(
        lambda i: fill(out, edges[i], edges[i + 1]), range(DISPLAY_BANDS)))
    return out

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

# Around the picture - chosen on the node's Settings tab. Black by default, as
# a viewer's is: a grey surround lifts how dark the shadows read against it.
# Enumeration knobs are saved by their TEXT, so the names are permanent.
BACKGROUND_NAMES = ("Black", "Dark grey", "Grey")
# Grey is the surround the player always had (28); Dark grey sits half way
# between it and black.
BACKGROUND_RGB = ((0, 0, 0), (14, 14, 14), (28, 28, 28))
BACKGROUND = QtGui.QColor(*BACKGROUND_RGB[0])


def background_color(index):
    """QColor for a Background choice (an index into BACKGROUND_NAMES)."""
    try:
        index = int(index)
    except (TypeError, ValueError):
        index = 0
    if not 0 <= index < len(BACKGROUND_RGB):
        index = 0
    return QtGui.QColor(*BACKGROUND_RGB[index])
NEUTRAL_LUT_F = build_lut_f()    # the same, unclipped, for the band checks


def build_cc_lut(gain, gamma, black=0.0):
    """256 -> 256 uint8: white/black point and gamma over finished bytes, or None.

    This is the CC path for the QC effects - it tints the result of the check
    but does not touch the data the check was computed from.
    """
    if (abs(float(gain) - 1.0) < 1e-6 and abs(float(gamma) - 1.0) < 1e-6
            and abs(float(black)) < 1e-9):
        return None
    v = (np.linspace(0.0, 1.0, 256, dtype=np.float32) - float(black)) \
        * float(gain)
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
    # Shift-drag sized a tool: ("draw" | "text", image pixels). `sizeDragging`
    # while the mouse is still moving, `sizeDragged` once, on release.
    sizeDragged = QtCore.Signal(str, float)
    sizeDragging = QtCore.Signal(str, float)
    textWanted = QtCore.Signal(float, float)  # image point a note was asked for
    # the canvas check's seam crossing was dragged: (shift_x %, shift_y %)
    canvasShifted = QtCore.Signal(float, float)

    def __init__(self, parent=None):
        super(ImageView, self).__init__(parent)
        self.setMinimumSize(160, 120)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setAutoFillBackground(False)
        self.setMouseTracking(True)           # probe even without a held button
        self.probe_frozen = False             # the P key freezes the readout

        self._frame = None           # (h,w,4) float16 scene-linear
        self._blank = False          # no data on this frame: draw it empty
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
        self.gain = 1.0              # 1 / (white point - black point)
        self.black = 0.0             # CC's black point, scene-linear
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
        self.log_curves = None       # names for the log view - set_log_curves
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
        self.annot_tool = None       # None | "draw" | "erase" | "text"
        self.show_annotations = False  # drawn only in Annotation mode
        self.annot_color = 0
        self.annot_pen = annotate.LINE_W
        self.annot_text = annotate.TEXT_H
        self.annot_frame = 0
        self._stroke = None          # the stroke being drawn, in image pixels
        self._erasing = False        # the rubber is down (see _erase_at)
        # Shift-drag with the pen: (press x on screen, press point, width it
        # started at). None when not resizing.
        # Shift-drag sizing: (press x on screen, press point, size it started
        # at, tool). None when not sizing.
        self._size_drag = None
        # the panel sets the real ones, from the tool panel's own sliders
        self.annot_ranges = {"draw": (0.5, 40.0), "text": (6.0, 200.0)}
        self._rubber_at = None       # where to draw the rubber's circle
        self._erased = False         # it took something out during this drag
        self._note_drag = None       # the note being dragged, while it is held
        self._canvas_drag = False    # the canvas check's centre is held
        self._windows = None         # (data window, display window) - set_windows
        self.show_bbox = False       # True: the picture outside the format shows
        self.background = QtGui.QColor(BACKGROUND)   # around the picture
        self.line_format = False     # outline the format, solid
        self.line_bbox = True        # outline the bounding box, dashed
        self._canvas_hover = False   # the pointer is over it
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
        # ANAMORPHIC. The pixels are stored square but were shot through a
        # squeezing lens, so they are only the right SHAPE when the picture is
        # drawn this much wider than it is stored. It scales the display and
        # nothing else: the array, the probe coordinates, the scopes and the
        # notes all stay in stored pixels, which is the only frame of reference
        # that survives switching the squeeze off again.
        self._par = 1.0
        # STABILISATION, in image pixels, added to the pan for drawing only.
        # Going through the pan is what makes it correct everywhere at once:
        # the paint transform, the crop that gets rendered and the pixel under
        # the cursor all read the pan already, so none of them can be left
        # behind describing an unstabilised picture.
        self._stab = [0.0, 0.0]
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
    def set_blank(self):
        """No picture on this frame: the input has no data here. Drawn as the
        bare background - an empty frame, not a held one, and not the
        '(no frame)' of a window with nothing attached."""
        self._blank = True
        self._frame = None
        self._prev = self._other = self._matte = None
        self._dirty = True
        self.update()

    def set_frame(self, arr, prev=None, other=None, matte=None):
        """`prev` = the previous frame (temporal check),
        `other` = the same frame from the other input (difference),
        `matte` = the DiMatte input frame (mattes in the RGBA channels)."""
        self._blank = False
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

    def set_color(self, gain=None, gamma=None, channels=None, saturation=None,
                  black=None):
        """`gain` and `black` are CC's white and black point as a Grade uses
        them: (in - black) * gain, gain = 1 / (white - black)."""
        rebuild = False
        if saturation is not None and abs(saturation - self.saturation) > 1e-6:
            self.saturation = float(saturation)
            self._sat_matrix = build_saturation_matrix(self.saturation)
            self._dirty = True
        if gain is not None and abs(gain - self.gain) > 1e-9:
            self.gain = float(gain); rebuild = True
        if gamma is not None and abs(gamma - self.gamma) > 1e-9:
            self.gamma = float(gamma); rebuild = True
        if black is not None and abs(black - self.black) > 1e-9:
            self.black = float(black); rebuild = True
        if channels is not None and int(channels) != self.channels:
            self.channels = int(channels)
            self._dirty = True
        if rebuild:
            self._rebuild_luts()
            self._dirty = True
        self.update()

    def _rebuild_luts(self):
        self._lut = nukelut.display_lut(self.nuke_display, self.nuke_input,
                                        self.gain, self.gamma, self.black)
        # gamma is inside the grade now (nukelut.grade), on the linear value,
        # for OCIO as well - nothing is laid over the finished bytes any more
        self._gamma_lut = None
        self._cc_lut = build_cc_lut(self.gain, self.gamma, self.black)

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
        # ONE step serves both axes, so it is set by whichever of them is drawn
        # LARGER. A desqueezed 2x plate at fit is 0.21 down the screen but only
        # 0.42 across: decimating 5:1 for the height would throw away half the
        # columns the width has room for, and the picture would go soft
        # sideways. With no squeeze max(1, par) is 1 and this is what it was.
        z = self._effective_zoom() * max(1.0, self._par)
        if z >= 1.0:
            return 1
        return max(1, min(8, int(round(1.0 / z))))

    def _visible_box(self, z, margin=None, bounds=None):
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
        px, py = self._pan_xy()
        fcx, fcy = self._format_centre()
        cx = fcx + px
        cy = fcy + py
        if margin is None:
            if self.playing:
                margin = 0.0
            else:
                margin = EFFECT_MARGIN if self.effect != fx.NONE else self.margin
        half_w = vw / (2.0 * self._zoom_x(z)) * (1.0 + margin)
        half_h = vh / (2.0 * z) * (1.0 + margin)
        # `bounds` (x0, y0, x1, y1) instead of the frame - the canvas check
        # asks for its format, which a small data window does not fill
        lx0, ly0, lx1, ly1 = bounds if bounds is not None else (0, 0, w, h)
        x0 = int(max(lx0, cx - half_w))
        y0 = int(max(ly0, cy - half_h))
        x1 = int(min(lx1, cx + half_w + 1))
        y1 = int(min(ly1, cy + half_h + 1))
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
            # THE DISPLAY STEP IS WHAT THE ZOOM ASKED FOR, NEVER COARSER.
            #
            # This used to raise it to fit a pixel budget, and the comment at
            # GRAIN_SS_BUDGET claimed that at 100 % zoom the crop "sits under
            # this and is computed exactly". It does not, on any panel bigger
            # than about 2048x1152: at 1:1 on a 2560x1440 view the grain was
            # computed at step 2, and at step 3 while anything moved. Drawn
            # magnified, that is a soft picture of every second pixel - and a
            # grain check that subsamples is not showing grain at all, which
            # is the one thing it exists to do.
            #
            # Zoomed OUT the budget still applies, through _supersample_step:
            # there the display step is already above 1, so the check can be
            # computed at a divisor of it and averaged down. That path is
            # correct - the averaging is what keeps the grain from crawling.
            # It just must not be allowed to coarsen past the zoom.
            #
            # WHILE SOMETHING IS MOVING it still may, and that is the point of
            # _fast: a scrub or a playing frame gets the cheap coarse render,
            # and 130 ms after everything goes quiet _refine_now re-renders at
            # the zoom's own step (see _begin_fast). What was wrong before was
            # not the coarse render - it was that the REFINED one was coarse
            # too, so on a panel over about 2048x1152 the grain was never once
            # shown at full resolution, however long you waited.
            # qc_full_play wins outright, exactly as it does in _ss_budget:
            # switching it on while a coarse render is already up has to take
            # effect on the next frame, not when the refinement happens to
            # fire.
            if self.qc_full_play or not self._fast:
                return s
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
        if self.effect == fx.LOG:
            return self._render_log(src)
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

    def set_log_curves(self, curves):
        """The names the log view's 'curve' index points at (see
        effects.log_view) - Nuke's log curves or the OCIO config's log spaces."""
        curves = list(curves) if curves else None
        if curves != self.log_curves:
            self.log_curves = curves
            if self.effect == fx.LOG:
                self.invalidate()

    def _render_log(self, src):
        """The log view: Nuke's curve by table, or OCIO's log space exactly."""
        curves = self.log_curves or fx.LOG_CURVES
        encoder = None
        if self.ocio_active():
            name = fx.log_curve_name(self.effect_params, curves)
            encoder = self.ocio.log_encoder(name)
        out = fx.log_view(src, self.effect_params, curves, encoder,
                          self.qc_threads)
        if out is not None and out.ndim == 3 and out.shape[2] == 1:
            out = out[:, :, 0]
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
            # the log view encodes VALUES too - it has to start from linear,
            # or a log plate would be shown as log of log
            if table is None or self.effect in (fx.VALUEMAP, fx.LOG) \
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
            # _isolate_channel gives ONE channel, (h,w,1) - it was made single
            # for the QC checks' sake. OCIO is told it is getting RGB, so the
            # channel has to be written into all three here; handing it the
            # one channel made OCIO refuse the buffer as a third too short.
            one = self._isolate_channel(arr)
            src = (np.repeat(one, 3, axis=2) if one.shape[2] == 1
                   else one[:, :, :3])
        try:
            rgb = self.ocio.apply(src, self.gain, self.black, self.gamma)
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
                "black": self.black,
                "sat_matrix": self._sat_matrix,
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
            #
            # ON THE CANVAS ONLY. With a bounding box that is not the format
            # the check wraps the FORMAT: overscan is cut away rather than
            # rolled into the middle, where it would be mistaken for the edge
            # of the shot, and a smaller data window is padded out black.
            src, offx, offy = self._canvas_source()
            fh, fw = src.shape[0], src.shape[1]
            bx0, by0 = max(0, x0 - offx), max(0, y0 - offy)
            bx1, by1 = min(fw, x1 - offx), min(fh, y1 - offy)
            if bx1 <= bx0 or by1 <= by0:
                return
            arr = fx.canvas_crop(src, bx0, by0, bx1, by1, step,
                                 self.effect_params)
            box = (bx0 + offx, by0 + offy, bx1 + offx, by1 + offy)
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
            if self.effect == fx.DIFF:
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
            lut = self._lut
            if self.channels == CH_RGB:
                def fill(out, a, b, _b=bits, _l=lut):
                    out[a:b] = _l[_b[a:b, :, :3]]
                rgb = _banded(fill, (rows, cols, 3))
                rgb = self._apply_saturation(rgb)    # RGB only, not on greys
            elif self.channels in (CH_R, CH_G, CH_B):
                idx = {CH_R: 0, CH_G: 1, CH_B: 2}[self.channels]

                def fill(out, a, b, _b=bits, _l=lut, _i=idx):
                    out[a:b] = _l[_b[a:b, :, _i]]
                rgb = _banded(fill, (rows, cols))
            elif self.channels == CH_A:
                # alpha raw (0-1 -> 0-255), no exposure and no display transform
                a = np.clip(arr[:, :, 3].astype(np.float32), 0.0, 1.0)
                rgb = (a * 255.0 + 0.5).astype(np.uint8)
            else:                                    # luminance
                # the whole chain in one pass per band - three gathers, the
                # weighted sum and the display lookup - so the intermediate
                # luminance is never built for the full frame at once
                def fill(out, a, b, _b=bits, _l=lut):
                    out[a:b] = _l[_luma_bits(_b[a:b])]
                rgb = _banded(fill, (rows, cols))

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
                "black": self.black,
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
        black = float(look.get("black", 0.0))

        # Its own tables, built from the look - the window is not disturbed,
        # and an export started while someone is dragging a slider still comes
        # out as the note was made.
        if (abs(gain - self.gain) > 1e-9 or abs(gamma - self.gamma) > 1e-9
                or abs(black - self.black) > 1e-9):
            lut = nukelut.display_lut(self.nuke_display, self.nuke_input,
                                      gain, gamma, black)
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
            # the channels of the LOOK being exported, not of whatever the
            # window happens to be showing now - _render_ocio reads them off
            # the view, so they are lent to it for the one call
            was, self.channels = self.channels, channels
            try:
                rgb = self._render_ocio(arr)
            finally:
                self.channels = was
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

    # ------------------------------------------------------------ format
    def set_windows(self, data_window, display_window):
        """The EXR's data window (what the pixels cover) and display window
        (the FORMAT), both (x0, y0, x1, y1) inclusive, y down as in the file.

        None for either = no format known: the frame is its own format, as
        before. Kept raw and checked against the frame at use, so a stale pair
        can never misplace a frame of a different size.
        """
        windows = None
        if data_window and display_window:
            windows = (tuple(int(v) for v in data_window),
                       tuple(int(v) for v in display_window))
        if windows == self._windows:
            return
        self._windows = windows
        self.update()

    def set_background(self, color):
        color = QtGui.QColor(color)
        if color != self.background:
            self.background = color
            self.update()

    def set_show_bbox(self, on, line_format=None, line_bbox=None):
        """Picture outside the format shown or cut, and the two outlines."""
        state = (bool(on),
                 self.line_format if line_format is None else bool(line_format),
                 self.line_bbox if line_bbox is None else bool(line_bbox))
        if state != (self.show_bbox, self.line_format, self.line_bbox):
            self.show_bbox, self.line_format, self.line_bbox = state
            self.update()

    def format_box(self):
        """(x, y, w, h) of the format in FRAME pixels, or None.

        None when there is no format, or when it is the frame itself - then
        there is nothing outside it to mark.
        """
        w, h = self.image_size
        if not self._windows or not w or not h:
            return None
        (x0, y0, x1, y1), (dx0, dy0, dx1, dy1) = self._windows
        if (x1 - x0 + 1, y1 - y0 + 1) != (w, h):
            return None                 # the pair belongs to another frame
        fw, fh = dx1 - dx0 + 1, dy1 - dy0 + 1
        if fw <= 0 or fh <= 0:
            return None
        box = (dx0 - x0, dy0 - y0, fw, fh)
        if box == (0, 0, w, h):
            return None
        return box

    def bbox_differs(self):
        """Is the data window anything other than the format - bigger on any
        side, smaller, or shifted? format_box is None exactly when they match."""
        return self.format_box() is not None

    def _format_centre(self):
        """The frame point the view centres on: the middle of the FORMAT.

        Not of the data window - an overscan or a per-frame bounding box would
        otherwise move the shot about on screen as the bbox changed.
        """
        box = self.format_box()
        if box is None:
            w, h = self.image_size
            return w / 2.0, h / 2.0
        return box[0] + box[2] / 2.0, box[1] + box[3] / 2.0

    def _fit_zoom(self):
        box = self.format_box()
        w, h = (box[2], box[3]) if box else self.image_size
        if not w or not h:
            return 1.0
        # against the DESQUEEZED width - fitting an anamorphic plate by its
        # stored width would leave it hanging out of the window
        return min(self.width() / float(w * self._par),
                   self.height() / float(h))

    def _effective_zoom(self):
        return self._fit_zoom() if self._zoom <= 0.0 else self._zoom

    def set_pixel_aspect(self, par):
        """How much wider than stored the picture is drawn. 1 = no squeeze."""
        par = max(0.1, min(10.0, float(par or 1.0)))
        if abs(par - self._par) < 1e-6:
            return
        self._par = par
        self._moved()               # the fit, the visible box and the pan all move

    @property
    def pixel_aspect(self):
        return self._par

    def set_stabilise(self, dx, dy):
        """Where this frame's picture sits, relative to the reference frame."""
        dx, dy = float(dx or 0.0), float(dy or 0.0)
        if abs(dx - self._stab[0]) < 1e-6 and abs(dy - self._stab[1]) < 1e-6:
            return
        self._stab = [dx, dy]
        self._begin_fast()      # a new crop is needed, and it moves every frame
        self.update()

    def _pan_xy(self):
        """The pan the PICTURE is drawn at: what was dragged, plus the track.

        The user's own pan is kept separate (see viewport) so that panning
        still means the same thing while stabilised, and so the two windows in
        Sync share where you looked rather than where the track was.
        """
        return (self._pan[0] + self._stab[0], self._pan[1] + self._stab[1])

    def _zoom_x(self, z=None):
        """The horizontal scale: the zoom, widened by the squeeze."""
        return (self._effective_zoom() if z is None else z) * self._par

    def zoom_percent(self):
        return self._effective_zoom() * 100.0

    def fit(self):
        self._zoom = 0.0
        self._pan = [0.0, 0.0]
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
        self._pan[0] += dx / self._par * (1.0 / old - 1.0 / new)
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
                shift = bool(event.modifiers() & QtCore.Qt.ShiftModifier)
                if shift and self.annot_tool in ("draw", "text"):
                    # SHIFT-DRAG SIZES THE TOOL, the way the brush is sized in
                    # Nuke's own paint tools: nothing is drawn or written, the
                    # drag sideways changes the size, and a preview shows it -
                    # a circle for the pen, letters for the text.
                    pos = event_pos(event)
                    start = (self.annot_pen if self.annot_tool == "draw"
                             else self.annot_text)
                    self._size_drag = (pos.x(), (pos.x(), pos.y()),
                                       float(start), self.annot_tool)
                    self.setCursor(QtCore.Qt.SizeHorCursor)
                    self.update()
                elif self.annot_tool == "text":
                    self._press_note(float(at[0]), float(at[1]))
                elif self.annot_tool == "erase":
                    self._erasing = True
                    self._erase_at(float(at[0]), float(at[1]))
                else:
                    self._stroke = [(float(at[0]), float(at[1]))]
                return
        if (event.button() == QtCore.Qt.LeftButton
                and self._canvas_hit(event_pos(event))):
            # the centre of the canvas check is a handle: it moves the seams,
            # anywhere else the left button still pans
            self._canvas_drag = True
            self.setCursor(QtCore.Qt.SizeAllCursor)
            self.update()
            return
        if event.button() == QtCore.Qt.LeftButton:
            self._drag = event_pos(event)
            self.setCursor(QtCore.Qt.ClosedHandCursor)
        elif event.button() == QtCore.Qt.MiddleButton and self.annot_tool:
            self._drag = event_pos(event)
            self.setCursor(QtCore.Qt.ClosedHandCursor)

    # The rubber's radius in SCREEN pixels - the circle drawn round the cursor.
    # Divided by the zoom before it reaches the notes, so it takes exactly the
    # ink the circle is drawn over whatever you are zoomed to.
    ERASE_RADIUS = 12.0

    def _size_drag_to(self, screen_x):
        """The tool size for a Shift-drag that has reached `screen_x`.

        Sizes are kept in IMAGE pixels, like every mark, so the drag is divided
        by the zoom: what you drag out on screen is what you get on screen.

        The pen grows by TWICE the drag, because the preview is a circle and
        its edge is what follows the cursor. Text grows by the drag itself -
        the preview letters grow upwards from where the drag started.
        """
        x0, _at, start, tool = self._size_drag
        z = max(0.02, self._effective_zoom())
        lo, hi = self.annot_ranges.get(tool, (0.5, 200.0))
        gain = 2.0 if tool == "draw" else 1.0
        size = start + gain * (float(screen_x) - x0) / z
        size = max(float(lo), min(float(hi), size))
        attr = "annot_pen" if tool == "draw" else "annot_text"
        if abs(size - getattr(self, attr)) > 1e-6:
            setattr(self, attr, size)
            self.sizeDragging.emit(tool, size)   # the panel's number follows
            self.update()
        return size

    def _erase_at(self, x, y):
        """Takes out the pen marks under the rubber, if there are any."""
        if self.annotations is None:
            return
        z = max(0.02, self._effective_zoom())
        gone = self.annotations.erase_at(self.annot_frame, x, y,
                                         self.ERASE_RADIUS / z,
                                         self.current_look())
        if gone:
            self._erased = True
            self.update()

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
        if self.annot_tool == "erase":
            # the circle follows the cursor whether or not it is pressed, so
            # you can see what it will take before it takes it
            self._rubber_at = (p.x(), p.y())
            self.update()
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
        if self._size_drag is not None:
            self._size_drag_to(p.x())
            return
        if self._erasing:
            at = self._widget_to_image(p)
            if at is not None:
                self._erase_at(float(at[0]), float(at[1]))
            return
        if self._stroke is not None:
            at = self._widget_to_image(p)
            if at is not None:
                self._stroke.append((float(at[0]), float(at[1])))
                self.update()
            return
        if self._canvas_drag:
            self._canvas_drag_to(p)
            return
        if self._drag is None:
            hover = self._canvas_hit(p)
            if hover != self._canvas_hover:
                self._canvas_hover = hover
                if hover:
                    self.setCursor(QtCore.Qt.OpenHandCursor)
                else:
                    self.unsetCursor()
                self.update()
            self._emit_probe(p)
            return
        z = self._effective_zoom()
        self._pan[0] -= (p.x() - self._drag.x()) / self._zoom_x(z)
        self._pan[1] -= (p.y() - self._drag.y()) / z
        self._drag = p
        self._moved()

    def leaveEvent(self, _event):
        if self._rubber_at is not None:
            self._rubber_at = None
            self.update()
        if not self.probe_frozen:
            self.probeChanged.emit(None)

    def mouseReleaseEvent(self, _event):
        if self._canvas_drag:
            self._canvas_drag = False
            self._canvas_hover = False
            self.unsetCursor()
            self.update()
            return
        if self._note_drag is not None:
            drag, self._note_drag = self._note_drag, None
            self.unsetCursor()
            if drag["moved"]:
                self.annotated.emit()       # it was dragged: put it down
            else:
                gx, gy = drag["grab"]       # it was only clicked: open it
                self.textWanted.emit(gx, gy)
            return
        if self._size_drag is not None:
            tool = self._size_drag[3]
            self._size_drag = None
            self.setCursor(QtCore.Qt.CrossCursor)
            self.sizeDragged.emit(tool, float(
                self.annot_pen if tool == "draw" else self.annot_text))
            self.update()
            return
        if self._erasing:
            self._erasing = False
            if self._erased:
                self._erased = False
                self.annotated.emit()    # once per drag, not once per mark
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
        zx = self._zoom_x(z)
        px, py = self._pan_xy()
        fcx, fcy = self._format_centre()
        ox = self.width() / 2.0 - (px + fcx) * zx
        oy = self.height() / 2.0 - (py + fcy) * z
        ix = int((pos.x() - ox) / zx)
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
                shown = self.ocio.apply(px[:, :, :3], self.gain,
                                        self.black, self.gamma).reshape(3)
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
            painter.fillRect(self.rect(), self.background)
        else:
            # the background is not drawn, so the other input shows through
            painter.setOpacity(self._opacity)
        if self._frame is None:
            if getattr(self, "_blank", False):
                return                  # an empty frame: background only
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
        fmt_box = self.format_box()
        if self.effect == fx.CANVAS and fmt_box is not None:
            # only the canvas is computed (see _render), so only the canvas is
            # asked for - otherwise the overscan would never count as covered
            # and every repaint would render again
            fx0, fy0, fw, fh = fmt_box
            box = self._visible_box(z, bounds=(fx0, fy0, fx0 + fw, fy0 + fh))
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
        # ABOVE 100 % ZOOM, NEVER SMOOTH.
        #
        # Zooming in is how you look at individual pixels - at grain, at a
        # paint edge, at one stuck sample. Bilinear there does not soften a
        # rendering artefact, it softens THE DATA, and a check that shows you
        # a blurred version of the pixels you asked to see is worse than no
        # check. Hard square pixels are what every image inspector shows and
        # what this one has to show too.
        #
        # The old rule was `rstep * z > 1`, meant for the coarse fast render
        # (rstep > 1 at z = 1), where bilinear makes half-resolution grain read
        # as slightly soft grain rather than blocks, so the full-resolution
        # refinement lands without a visible jump. That case is kept - it is
        # only ever at or below 100 %. Plain magnification was caught by the
        # same test by accident, because z > 1 satisfies it on its own.
        magnified = z > 1.001
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform,
                              not magnified and rstep * z > 1.001)
        # the top left corner of the WHOLE image on screen
        zx = self._zoom_x(z)             # widened if the plate is anamorphic
        px, py = self._pan_xy()          # shifted if the track is holding it
        fcx, fcy = self._format_centre()
        ox = self.width() / 2.0 - (px + fcx) * zx
        oy = self.height() / 2.0 - (py + fcy) * z
        target = QtCore.QRectF(ox + rx0 * zx, oy + ry0 * z,
                               cols * rstep * zx, rows * rstep * z)
        fmt = self.format_box()
        fmt_rect = None
        if fmt is not None:
            fmt_rect = QtCore.QRectF(ox + fmt[0] * zx, oy + fmt[1] * z,
                                     fmt[2] * zx, fmt[3] * z)
            # inside the format but outside the data window is BLACK, as in
            # Nuke - not the grey of the window around the frame
            painter.fillRect(fmt_rect, QtGui.QColor(0, 0, 0))
            cut = not self.show_bbox or self.effect == fx.CANVAS
            if cut:
                painter.save()
                painter.setClipRect(fmt_rect)
        painter.drawImage(target, self._qimage)
        if fmt_rect is not None:
            if cut:
                painter.restore()
            if self.line_format or self.line_bbox:
                self._draw_format_lines(painter, fmt_rect, QtCore.QRectF(
                    ox, oy, w * zx, h * z))

        # The notes go on LAST and use the same ox/oy/z the picture was drawn
        # with, so they sit on the pixels they were drawn on at any zoom.
        if self.annotations is not None and self.show_annotations:
            self.annotations.draw(painter, self.annot_frame, ox, oy, z,
                                  self.current_look(), w, h, zoom_x=zx)
            if self._size_drag is not None:
                _x0, (cx, cy), _start, tool = self._size_drag
                ink = QtGui.QColor(*annotate.color_rgb(self.annot_color))
                painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
                if tool == "draw":
                    # the pen's width, where the drag started, at the size it
                    # will actually lay down on screen
                    r = max(1.0, self.annot_pen * z / 2.0)
                    painter.setBrush(ink)
                    painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 170), 1.5))
                    painter.drawEllipse(QtCore.QPointF(cx, cy), r, r)
                    painter.setBrush(QtCore.Qt.NoBrush)
                else:
                    # letters at the height a note will be written at, bold as
                    # notes are, with the same contrasting edge they get
                    font = painter.font()
                    font.setBold(True)
                    font.setPixelSize(max(1, int(round(self.annot_text * z))))
                    path = QtGui.QPainterPath()
                    path.addText(QtCore.QPointF(cx, cy), font, "Aa")
                    rgb = annotate.color_rgb(self.annot_color)
                    edge = (QtGui.QColor(255, 255, 255, 200) if sum(rgb) < 200
                            else QtGui.QColor(0, 0, 0, 190))
                    painter.strokePath(path, QtGui.QPen(edge, 2.0))
                    painter.fillPath(path, ink)
            if self.annot_tool == "erase" and self._rubber_at is not None:
                # the rubber's reach, in the same screen pixels _erase_at uses
                painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
                painter.setBrush(QtCore.Qt.NoBrush)
                cx, cy = self._rubber_at
                r = self.ERASE_RADIUS
                for colour, width in ((QtGui.QColor(0, 0, 0, 160), 3.0),
                                      (QtGui.QColor(255, 255, 255, 230), 1.2)):
                    pen = QtGui.QPen(colour)
                    pen.setWidthF(width)
                    painter.setPen(pen)
                    painter.drawEllipse(QtCore.QPointF(cx, cy), r, r)
            if self._stroke and len(self._stroke) > 1:
                pen = QtGui.QPen(QtGui.QColor(
                    *annotate.color_rgb(self.annot_color)))
                pen.setWidthF(max(1.0, self.annot_pen * z))
                pen.setCapStyle(QtCore.Qt.RoundCap)
                pen.setJoinStyle(QtCore.Qt.RoundJoin)
                painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
                painter.setPen(pen)
                painter.drawPolyline(QtGui.QPolygonF(
                    [QtCore.QPointF(ox + x * zx, oy + y * z)
                     for x, y in self._stroke]))

        if self._canvas_handle_live():
            self._draw_canvas_handle(painter)

    def _draw_format_lines(self, painter, fmt_rect, bbox_rect):
        """The format as a thin solid line, the bounding box DASHED - the way
        the Nuke Viewer marks what lies outside the frame."""
        painter.save()
        painter.setOpacity(1.0)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
        painter.setBrush(QtCore.Qt.NoBrush)
        # Each line is drawn twice: dark underneath, light on top. A single
        # light line vanished over a bright overscan, a dark one over a black
        # one; the pair reads on anything.
        dark = QtGui.QColor(0, 0, 0, 200)
        light = QtGui.QColor(230, 230, 230, 240)
        # drawn just OUTSIDE the edges, so neither line covers picture
        fmt_rect = fmt_rect.adjusted(-1.0, -1.0, 0.0, 0.0)
        bbox_rect = bbox_rect.adjusted(-1.0, -1.0, 0.0, 0.0)
        if self.line_format:
            # the canvas line is a hint, not a frame: its white at 25 %
            # (the dark edge under it scaled down with it)
            soft_dark = QtGui.QColor(0, 0, 0, 64)
            soft_light = QtGui.QColor(255, 255, 255, 64)
            for colour in (soft_dark, soft_light):
                pen = QtGui.QPen(colour)
                pen.setWidthF(1.0)
                pen.setCosmetic(True)
                painter.setPen(pen)
                painter.drawRect(fmt_rect if colour is soft_light
                                 else fmt_rect.adjusted(-1.0, -1.0, 1.0, 1.0))
        if not self.line_bbox:
            painter.restore()
            return
        under = QtGui.QPen(dark)
        under.setWidthF(1.0)
        under.setCosmetic(True)
        painter.setPen(under)
        painter.drawRect(bbox_rect)
        dashed = QtGui.QPen(light)
        dashed.setWidthF(1.0)
        dashed.setCosmetic(True)
        dashed.setStyle(QtCore.Qt.CustomDashLine)
        dashed.setDashPattern([5.0, 4.0])
        painter.setPen(dashed)
        painter.drawRect(bbox_rect)
        painter.restore()

    # ------------------------------------------------------- canvas handle
    # How close (screen pixels) the pointer has to be to take hold of the
    # point where the canvas check's seams cross.
    CANVAS_GRAB = 16.0

    def _canvas_handle_live(self):
        """The canvas check is on and nothing else owns the left button."""
        return (self.effect == fx.CANVAS and self._frame is not None
                and not self.annot_tool and all(self.image_size))

    def canvas_seam(self):
        """(x, y) in image pixels where the ORIGINAL edges meet on screen.

        Mirrors effects.canvas_source_index: a displayed pixel x shows source
        (x + dx) % w, so source column 0 - the plate's left edge - lands on
        (w - dx) % w. With the default half shift that is the middle.
        """
        offx, offy, w, h = self._canvas_rect()
        dx = int(w * fx.param(self.effect_params, "shift_x", 50.0) / 100.0)
        dy = int(h * fx.param(self.effect_params, "shift_y", 50.0) / 100.0)
        return (w - dx) % w + offx, (h - dy) % h + offy

    def _canvas_rect(self):
        """(x, y, w, h) of what the canvas check wraps, in frame pixels: the
        format when the frame has one that differs, else the frame itself."""
        box = self.format_box()
        if box is not None:
            return box
        w, h = self.image_size
        return 0, 0, w, h

    def _canvas_source(self):
        """(array the size of the canvas, its x, its y) - see _render.

        The frame cut to its format (overscan) or padded out to it (a smaller
        data window), kept for as long as the frame and its format stay the
        same, so a pan or a slider costs no copy.
        """
        arr = self._frame
        box = self.format_box()
        if box is None:
            return arr, 0, 0
        key = (id(arr), box)
        cached = getattr(self, "_canvas_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1], box[0], box[1]
        fx0, fy0, fw, fh = box
        h, w = arr.shape[0], arr.shape[1]
        if fx0 >= 0 and fy0 >= 0 and fx0 + fw <= w and fy0 + fh <= h:
            src = arr[fy0:fy0 + fh, fx0:fx0 + fw]
        else:
            src = np.zeros((fh, fw) + arr.shape[2:], dtype=arr.dtype)
            if arr.shape[2] >= 4:
                src[:, :, 3] = 1.0
            sx0, sy0 = max(0, fx0), max(0, fy0)
            sx1, sy1 = min(w, fx0 + fw), min(h, fy0 + fh)
            if sx1 > sx0 and sy1 > sy0:
                src[sy0 - fy0:sy1 - fy0, sx0 - fx0:sx1 - fx0] = \
                    arr[sy0:sy1, sx0:sx1]
        self._canvas_cache = (key, src)
        return src, fx0, fy0

    def _image_origin(self):
        """(ox, oy, zx, z): the image's top left on screen and the scales."""
        w, h = self.image_size
        z = self._effective_zoom()
        zx = self._zoom_x(z)
        px, py = self._pan_xy()
        fcx, fcy = self._format_centre()
        return (self.width() / 2.0 - (px + fcx) * zx,
                self.height() / 2.0 - (py + fcy) * z, zx, z)

    def _canvas_seam_screen(self):
        ox, oy, zx, z = self._image_origin()
        sx, sy = self.canvas_seam()
        return ox + sx * zx, oy + sy * z

    def _canvas_hit(self, pos):
        if not self._canvas_handle_live():
            return False
        cx, cy = self._canvas_seam_screen()
        return math.hypot(pos.x() - cx, pos.y() - cy) <= self.CANVAS_GRAB

    def _canvas_drag_to(self, pos):
        """Puts the seam crossing under the pointer -> new shift_x / shift_y."""
        offx, offy, w, h = self._canvas_rect()
        ox, oy, zx, z = self._image_origin()
        ix = max(0.0, min(float(w), (pos.x() - ox) / zx - offx))
        iy = max(0.0, min(float(h), (pos.y() - oy) / z - offy))
        params = dict(self.effect_params)
        params["shift_x"] = round((w - ix) / float(w) * 100.0, 2) % 100.0
        params["shift_y"] = round((h - iy) / float(h) * 100.0, 2) % 100.0
        if params != self.effect_params:
            self.set_effect_params(params)
            self.canvasShifted.emit(params["shift_x"], params["shift_y"])

    def _draw_canvas_handle(self, painter):
        cx, cy = self._canvas_seam_screen()
        if not (-40 < cx < self.width() + 40 and -40 < cy < self.height() + 40):
            return
        painter.setOpacity(1.0)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setBrush(QtCore.Qt.NoBrush)
        hot = self._canvas_drag or self._canvas_hover
        arm = 14.0 if hot else 11.0          # a plain cross, no ring
        for colour, width in ((QtGui.QColor(0, 0, 0, 170), 3.2),
                              (QtGui.QColor(255, 200, 60, 240) if hot
                               else QtGui.QColor(255, 255, 255, 220), 1.4)):
            pen = QtGui.QPen(colour)
            pen.setWidthF(width)
            painter.setPen(pen)
            painter.drawLine(QtCore.QPointF(cx - arm, cy),
                             QtCore.QPointF(cx + arm, cy))
            painter.drawLine(QtCore.QPointF(cx, cy - arm),
                             QtCore.QPointF(cx, cy + arm))
