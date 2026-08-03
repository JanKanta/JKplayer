"""
The EXRplayer panel - the UI layer (viewer + timeline + cache bar).

INPUTS AND WINDOWS ARE TWO DIFFERENT THINGS:
  * a node input (A, B) = a sequence of files
  * a panel window (1, 2) = a place on screen that PICKS which input it shows,
    and has its own layer and its own loader

Thanks to that both windows can show THE SAME input and differ only by layer
(rgba against depth of the same plate). If the loader belonged to the input,
this would not work - an input would only have one layer at a time.

The cache is shared and the keys carry the layer too (FrameLoader.key_for), so
two windows with the same input and layer share the same data for free.

Colour (CC) and the check mode (QC) are per window: comparing two images only
makes sense when both are measured with the same ruler.

In Double both windows are exactly the same size (stretch 1:1), so the divider
always sits in the middle regardless of the panel's aspect ratio. Zoom and pan
are carried between the windows, so the pixels line up.

Playback logic:
  * A timer ticks at the target FPS. A tick ONLY reads from the cache and
    redraws - it never decodes anything and never touches Nuke. That is why
    playback cannot get stuck.
  * realtime=ON:   the step follows ELAPSED TIME (uncached frames are skipped)
                   -> holds the pace like RV.
    realtime=OFF:  one frame per tick -> plays every frame.
  * cached_only:   the playhead does not leave the cached region.
  * loop / ping-pong / stop at the end of the range.
The look-ahead is rebuilt in the playback direction on every move.
"""

import time

import nuke
from .qtcompat import QtCore, QtGui, QtWidgets, QShortcut, event_pos

from . import effects as fx
from . import exrcore
from . import node as exrnode
from . import nukelut
from . import ocio
from . import reader
from .cache import FrameCache
from .imageview import ImageView
from .loader import FrameLoader
from . import overlay as overlay_mod
from .overlay import ControlStack, ScopeStack, SlotBar

_Stack_GAP = ControlStack.GAP        # the gap between the controls of both inputs

# The top bar - the look of the Nuke Viewer: a dark strip, low flat controls
# with no raised edges, thin separators. Applies ONLY inside #cvTopBar.
TOP_BAR_STYLE = """
QWidget#cvTopBar { background: #333333; }
QWidget#cvTopBar QComboBox, QWidget#cvTopBar QPushButton {
    background: #3f3f3f; color: #d6d6d6;
    border: 1px solid #262626; border-radius: 2px;
    padding: 1px 6px; min-height: 18px;
}
QWidget#cvTopBar QComboBox:hover, QWidget#cvTopBar QPushButton:hover {
    background: #4a4a4a; border-color: #6a6a6a;
}
QWidget#cvTopBar QPushButton:pressed { background: #2c2c2c; }
QWidget#cvTopBar QComboBox::drop-down { width: 14px; border: none; }
QWidget#cvTopBar QComboBox QAbstractItemView {
    background: #3f3f3f; color: #d6d6d6;
    selection-background-color: #5a5a5a;
}
QWidget#cvTopBar QFrame[frameShape="5"] {   /* vertical separator */
    color: #262626; background: #262626; max-width: 1px;
}
"""
from .sequence import from_read_node
from .timeline import Timeline

PANEL_ID = "com.honza.EXRplayerPanel"

# How often at most the scopes are recomputed while panning/zooming. They read
# the VISIBLE crop (see ImageView.visible_linear), so they have to follow the
# canvas - but a drag fires mouse moves at 60-120 Hz and all three scopes cost
# about 10 ms together, which would make panning crawl. 100 ms gives a steady
# ~10 updates a second and the drag stays smooth.
SCOPE_REFRESH_MS = 100


def _nbsp(text):
    """Qt collapses spaces in rich text - the number padding would fall apart."""
    return text.replace(" ", "&nbsp;")


def panel_toggle(settings, slot_index, slot_label, key, on, slot_count=2):
    """Toggling a panel: (new settings, [(knob, value)] to write).

    Deliberately a PURE function without Qt - the Qt layer cannot be tested
    inside Nuke (the terminal only has QCoreApplication, a standalone
    QApplication freezes), so at least this much is verifiable. It also
    guarantees the knob name matches what exrnode.settings() reads.
    """
    s = dict(settings)
    values = list(s.get(key, ()))
    while len(values) <= slot_index:
        values.append(False)
    values[slot_index] = bool(on)
    s[key] = tuple(values[:max(slot_count, slot_index + 1)])
    return s, [("cv_%s_%s" % (key, slot_label), bool(on))]


def overlay_anchors(box, stage_w, wipe_top=None, edge=overlay_mod.EDGE):
    """Where one window's controls belong: ((bar x, y), (scope x, y)).

    `box` is (x, y, right) of the window. The scope column hangs off the RIGHT
    edge - _Stack.RIGHT then subtracts its own width, so it grows leftwards and
    its right edge stays put.

    In Wipe the windows overlap, so the blocks stand under each other instead:
    `wipe_top` is the y this one starts at and the scopes hang off the right
    edge of the whole stage.

    A pure function without Qt on purpose, like panel_toggle - the Qt layer
    cannot be tested inside Nuke, and this is exactly the arithmetic that
    decides whether the scopes sit at the edge or somewhere in the middle.
    """
    if wipe_top is not None:
        return (edge, wipe_top), (stage_w - edge, wipe_top)
    x, y, right = box
    return (x + edge, y + edge), (right - edge, y + edge)


def _halfplane(w, h, px, py, nx, ny):
    """The part of the rect (0,0,w,h) where (point - P) . n <= 0. A vertex list.

    Clips the rect with a single line (Sutherland-Hodgman with one edge) -
    exactly what the wipe mask needs: half the image on one side of the line,
    whatever angle it is at.
    """
    poly = [(0.0, 0.0), (float(w), 0.0), (float(w), float(h)), (0.0, float(h))]
    out = []
    for i in range(len(poly)):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % len(poly)]
        da = (ax - px) * nx + (ay - py) * ny
        db = (bx - px) * nx + (by - py) * ny
        if da <= 0.0:
            out.append((ax, ay))
        if (da <= 0.0) != (db <= 0.0):
            t = da / (da - db)
            out.append((ax + (bx - ax) * t, ay + (by - ay) * t))
    return out


class _WipeLine(QtWidgets.QWidget):
    """Draws the wipe line and its circle. It does not take the mouse - _Stage
    handles that through an event filter, so the image can still be panned and
    probed underneath the line.

    The central circle is just a handle for moving it - it is transparent, so
    nothing is less visible through it than through the rest of the image. The
    blend intensity is held by a SECOND handle that runs along the wipe axis a
    little away from the centre: the further from the centre, the higher the
    intensity. They are two different places, so moving the line cannot change
    the intensity.
    """

    HANDLE = 11
    OP_HANDLE = 7                 # radius of the intensity handle
    OP_MIN = 34.0                 # where its travel starts on the axis (px from centre)
    OP_MAX = 118.0                # ... and where it ends

    def __init__(self, stage):
        super(_WipeLine, self).__init__(stage)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self._stage = stage

    def paintEvent(self, _event):
        w, h = self.width(), self.height()
        cx, cy, dx, dy = self._stage.wipe_geometry(w, h)
        far = float(w + h)
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        for color, width in ((QtGui.QColor(0, 0, 0, 160), 3),
                             (QtGui.QColor(240, 200, 40, 220), 1)):
            p.setPen(QtGui.QPen(color, width))
            p.drawLine(int(cx - dx * far), int(cy - dy * far),
                       int(cx + dx * far), int(cy + dy * far))

        # the central circle: OUTLINE ONLY, transparent - it is a move handle
        r = self.HANDLE
        box = QtCore.QRectF(cx - r, cy - r, 2 * r, 2 * r)
        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 160), 3))
        p.drawEllipse(box)
        p.setPen(QtGui.QPen(QtGui.QColor(240, 200, 40, 230), 1))
        p.drawEllipse(box)

        # the travel of the intensity handle along the wipe axis
        ox, oy, odx, ody = self._stage.opacity_axis(w, h)
        ax, ay = ox + odx * self.OP_MIN, oy + ody * self.OP_MIN
        bx, by = ox + odx * self.OP_MAX, oy + ody * self.OP_MAX
        p.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 130), 4))
        p.drawLine(int(ax), int(ay), int(bx), int(by))
        p.setPen(QtGui.QPen(QtGui.QColor(240, 200, 40, 90), 2))
        p.drawLine(int(ax), int(ay), int(bx), int(by))

        hx, hy = self._stage.opacity_handle(w, h)
        hr = self.OP_HANDLE
        hbox = QtCore.QRectF(hx - hr, hy - hr, 2 * hr, 2 * hr)
        p.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 170), 3))
        p.setBrush(QtGui.QColor(240, 200, 40, 220))
        p.drawEllipse(hbox)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 235, 150, 230), 1))
        p.drawEllipse(hbox)

        level = max(0.0, min(1.0, self._stage.wipe_opacity))
        font = p.font()
        font.setPointSize(max(6, font.pointSize() - 1))
        p.setFont(font)
        text = "%d %%" % round(level * 100)
        rect = QtCore.QRectF(hx + hr + 3, hy - 8, 40, 16)
        p.setPen(QtGui.QColor(0, 0, 0, 190))
        p.drawText(rect.translated(1, 1), QtCore.Qt.AlignVCenter, text)
        p.setPen(QtGui.QColor(240, 200, 40, 235))
        p.drawText(rect, QtCore.Qt.AlignVCenter, text)
        p.end()


class _Slot(object):
    """One panel window.

    It has its own selected input, layer, loader, image - and also ITS OWN
    panels (CC, QC, histogram, vectorscope). In Double two different images are
    being compared and each deserves its own scope and its own check mode.
    """

    def __init__(self, index, cache, workers, on_ready):
        self.index = index
        self.label = exrnode.SLOT_LABELS[index]
        self.source = min(index, len(exrnode.INPUT_LABELS) - 1)  # 0=A, 1=B
        self.sequence = None
        self.source_info = "-"
        self.layers = []                 # layers in the selected input's file
        self.view = ImageView()
        self.loader = FrameLoader(cache, workers=workers, on_ready=on_ready)
        self.controls = None             # ControlStack (CC, QC) - on the left
        self.scopes = None               # ScopeStack (H, V) - on the right
        self.bar = None                  # SlotBar, created by _build_ui
        self.fx_params = {}              # slider settings, remembered per mode
        self.legend = ""                 # explanation of the active QC mode
        self.fitted = False              # we have fitted the image once already

    def layer(self):
        return self.loader.layer

    def source_label(self):
        return exrnode.INPUT_LABELS[self.source]


class _Stage(QtWidgets.QWidget):
    """The area holding the image windows.

    It lays them out ITSELF, without a layout: in Wipe the windows have to
    overlap, which no layout can do. The wipe line handling lives here too - it
    is caught through an event filter on the windows, so away from the line the
    mouse keeps its usual function (panning the image, probing pixels).
    """

    GRAB = 12                        # how close to the centre the move still grabs
    LINE_GRAB = 7                    # ... and to the line for rotation
    wipeChanged = QtCore.Signal()
    wipeOpacityChanged = QtCore.Signal(float)
    relaid = QtCore.Signal()         # windows rearranged - panels should rise

    def __init__(self, parent=None):
        super(_Stage, self).__init__(parent)
        self.views = []
        self.mode = exrnode.VIEW_SINGLE
        self.split = exrnode.SPLIT_SIDE
        self.wipe = [0.5, 0.5, 0.0]  # share of width, share of height, angle in degrees
        self.wipe_opacity = 1.0      # blend intensity (drawn at the circle)
        self._drag = None            # "move" | "rotate"
        self.line = _WipeLine(self)

    def set_views(self, views):
        self.views = list(views)
        for view in self.views:
            view.installEventFilter(self)
        self.line.raise_()

    def set_mode(self, mode, split):
        self.mode, self.split = mode, split
        self.relayout()

    # ------------------------------------------------------------- layout
    def resizeEvent(self, event):
        super(_Stage, self).resizeEvent(event)
        self.relayout()

    def relayout(self):
        if not self.views:
            return
        w, h = self.width(), self.height()
        gap = 2                                  # a thin divider
        if self.mode == exrnode.VIEW_DOUBLE:
            if self.split == exrnode.SPLIT_STACK:
                half = max(1, (h - gap) // 2)
                boxes = [(0, 0, w, half), (0, half + gap, w, h - half - gap)]
            else:
                half = max(1, (w - gap) // 2)
                boxes = [(0, 0, half, h), (half + gap, 0, w - half - gap, h)]
        else:
            # Single and Wipe: both windows over the whole area. In Single the
            # second one is hidden, in Wipe it is clipped by a mask.
            boxes = [(0, 0, w, h), (0, 0, w, h)]
        for view, box in zip(self.views, boxes):
            view.setGeometry(*box)
        self.line.setGeometry(0, 0, w, h)
        self.line.setVisible(self.mode == exrnode.VIEW_WIPE)
        self.apply_wipe()
        # Z order: the windows, the wipe line above them, the control panels
        # above that. Had the line been raised last, it would be drawn OVER the
        # panels and its circle would leave a half-moon in them.
        self.line.raise_()
        self.relaid.emit()

    # --------------------------------------------------------------- wipe
    def wipe_geometry(self, w=None, h=None):
        """(centre x, centre y, direction x, direction y) in stage pixels."""
        w = self.width() if w is None else w
        h = self.height() if h is None else h
        import math
        a = math.radians(self.wipe[2])
        return (self.wipe[0] * w, self.wipe[1] * h, math.cos(a), math.sin(a))

    def apply_wipe(self):
        """Clips the second window to one side of the line.

        It masks the WINDOW, not the drawing - the control panels are no longer
        its children (they live on the stage), so the mask does not cut them off.
        """
        if len(self.views) < 2:
            return
        view = self.views[1]
        if self.mode != exrnode.VIEW_WIPE:
            view.clearMask()
            return
        w, h = max(1, view.width()), max(1, view.height())
        cx, cy, dx, dy = self.wipe_geometry(w, h)
        pts = _halfplane(w, h, cx, cy, -dy, dx)   # the normal perpendicular to the line
        poly = QtGui.QPolygon([QtCore.QPoint(int(round(x)), int(round(y)))
                               for x, y in pts])
        view.setMask(QtGui.QRegion(poly) if len(poly) >= 3
                     else QtGui.QRegion())

    def eventFilter(self, obj, event):
        if self.mode != exrnode.VIEW_WIPE or obj not in self.views:
            return False
        kind = event.type()
        if kind == QtCore.QEvent.MouseButtonPress:
            grab = self._grab_at(event_pos(event))
            if grab is None:
                return False                      # away from the line - let the window have it
            self._drag = grab
            return True
        if kind == QtCore.QEvent.MouseMove and self._drag:
            self._drag_to(event_pos(event))
            return True
        if kind == QtCore.QEvent.MouseButtonRelease and self._drag:
            self._drag = None
            return True
        return False

    def opacity_axis(self, w=None, h=None):
        """(centre x, centre y, direction x, direction y) of the intensity travel.

        The travel runs along the wipe axis towards whichever side it fits - at
        the edge of the image the handle would otherwise run out and could not
        be grabbed.
        """
        w = self.width() if w is None else w
        h = self.height() if h is None else h
        cx, cy, dx, dy = self.wipe_geometry(w, h)
        end = _WipeLine.OP_MAX
        fits = 0 <= cx + dx * end <= w and 0 <= cy + dy * end <= h
        back = 0 <= cx - dx * end <= w and 0 <= cy - dy * end <= h
        if not fits and back:
            dx, dy = -dx, -dy
        return cx, cy, dx, dy

    def opacity_handle(self, w=None, h=None):
        """Where on the axis the intensity handle sits - further out is more."""
        cx, cy, dx, dy = self.opacity_axis(w, h)
        lo, hi = _WipeLine.OP_MIN, _WipeLine.OP_MAX
        dist = lo + (hi - lo) * max(0.0, min(1.0, self.wipe_opacity))
        return cx + dx * dist, cy + dy * dist

    def _grab_at(self, pos):
        """What dragging from this point would do.

        The order matters: the intensity handle lies ON the axis, so it has to
        be tested before rotation, otherwise the line would be grabbed instead.
        """
        w, h = max(1, self.width()), max(1, self.height())
        cx, cy, dx, dy = self.wipe_geometry(w, h)
        mx, my = pos.x(), pos.y()
        hx, hy = self.opacity_handle(w, h)
        if (mx - hx) ** 2 + (my - hy) ** 2 <= (_WipeLine.OP_HANDLE + 5) ** 2:
            return "opacity"
        if (mx - cx) ** 2 + (my - cy) ** 2 <= self.GRAB ** 2:
            return "move"
        # distance from the line = the projection onto the normal
        if abs((mx - cx) * -dy + (my - cy) * dx) <= self.LINE_GRAB:
            return "rotate"
        return None

    def _drag_to(self, pos):
        import math
        w, h = max(1, self.width()), max(1, self.height())
        cx, cy, dx, dy = self.wipe_geometry(w, h)
        if self._drag == "opacity":
            # project the mouse onto the handle axis -> distance from the
            # centre -> intensity
            ax, ay, adx, ady = self.opacity_axis(w, h)
            dist = (pos.x() - ax) * adx + (pos.y() - ay) * ady
            lo, hi = _WipeLine.OP_MIN, _WipeLine.OP_MAX
            self.wipe_opacity = max(0.0, min(1.0, (dist - lo) / (hi - lo)))
            self.line.update()
            self.wipeOpacityChanged.emit(self.wipe_opacity)
            return
        if self._drag == "move":
            self.wipe[0] = min(1.0, max(0.0, pos.x() / float(w)))
            self.wipe[1] = min(1.0, max(0.0, pos.y() / float(h)))
        else:
            self.wipe[2] = math.degrees(math.atan2(pos.y() - cy, pos.x() - cx))
        self.apply_wipe()
        self.line.update()
        self.wipeChanged.emit()


class PlayerPanel(QtWidgets.QWidget):

    _frame_ready = QtCore.Signal(int)          # from a worker -> the GUI thread

    def __init__(self, parent=None):
        super(PlayerPanel, self).__init__(parent)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        self._node_name = None
        self.frame = 1
        self.cache_runs = []
        self.mark_in = 1
        self.mark_out = 1
        self._cache_anchor = 1        # where the rolling cache window is planned from
        self._hint = "looking for an EXRplayer node..."
        self._last_error = None
        self._input_note = ""            # a problem with the inputs (type, frame range)
        self._knob_note = ""             # writing to the node failed
        self._follow_note = ""           # the node watcher crashed
        self._toggle_note = ""           # what the last toggle did
        self._toggle_note_t = 0.0        # ... and when (it disappears shortly)
        # the QC mode and its sliders are held per window (see _Slot)
        self._ocio = None                # ocio.DisplayTransform, when enabled
        self._ocio_note = ""             # an OCIO error for the status line
        self._temporal_note = ""         # the temporal check result for this frame
        self._settings = {}
        self._direction = 1
        self._playing = False
        self._play_t0 = 0.0
        self._play_f0 = 0
        self._shown = 0                        # frames shown (for the FPS)
        self._fps_t0 = time.monotonic()
        self._fps_shown = 0.0

        # One cache for both inputs: the key is the file path (+ layer), so they
        # do not overwrite each other and the RAM budget stays one number the
        # user sets in one place.
        self.cache = FrameCache(4096)
        # 4 threads is the measured optimum (more is held back by memory
        # bandwidth); it can be changed with the cv_workers knob and takes
        # effect after reopening
        self._slots = [_Slot(i, self.cache, 4, self._frame_ready.emit)
                       for i in range(len(exrnode.SLOT_LABELS))]
        # node inputs: A, B and DiMatte (mattes) - see exrnode.ALL_INPUTS
        self._sequences = [None] * len(exrnode.ALL_INPUTS)
        # DiMatte is not displayed, only the mattes are taken from it - hence
        # its own loader but no window
        self._matte_loader = FrameLoader(self.cache, workers=2,
                                         on_ready=self._frame_ready.emit)
        self._tl_range = None            # the range already set on the timeline
        self._active = 0                 # which window the scopes and probe read
        self._placing = False            # currently placing the controls (see below)
        self._view_mode = exrnode.VIEW_SINGLE
        self._split = exrnode.SPLIT_SIDE
        self._frame_ready.connect(self._on_frame_ready)

        self._build_ui()

        self._play_timer = QtCore.QTimer(self)
        # A PRECISE timer. Windows ticks every 15.625 ms and an ordinary
        # (coarse) Qt timer is rounded onto that grid - 24 and 25 fps then both
        # fall onto an interval of 46.9 ms, i.e. 21.3 fps. Exactly what the
        # panel showed:
        #     target 24 -> 21.3    target 25 -> 21.3    target 40 -> 32.0
        # PreciseTimer asks for a higher resolution and the interval lines up.
        self._play_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self._play_timer.timeout.connect(self._tick)
        # the other timers need not be precise - a coarser grid saves battery
        self._ui_timer = QtCore.QTimer(self)    # refreshing the bar and stats
        self._ui_timer.setInterval(200)
        self._ui_timer.timeout.connect(self._refresh_status)
        self._ui_timer.start()
        self._follow_timer = QtCore.QTimer(self)
        self._follow_timer.setInterval(400)
        self._follow_timer.timeout.connect(self._follow_tick)
        self._follow_timer.start()
        # Recompute of the scopes after a pan/zoom. Single-shot and only
        # restarted when it is NOT running: the first move schedules a
        # recompute, further moves inside that window are swallowed. So a drag
        # updates at a steady rate instead of once at the end.
        self._scope_timer = QtCore.QTimer(self)
        self._scope_timer.setSingleShot(True)
        self._scope_timer.setInterval(SCOPE_REFRESH_MS)
        self._scope_timer.timeout.connect(self._refresh_scopes)
        self._install_wheel_filter()

    # ------------------------------------------------------ access to windows
    # Most of the panel works with "whatever I am looking at right now" - hence
    # these three shortcuts. Settings that apply to both windows (colour, QC)
    # are sent out through _each_view().
    @property
    def active(self):
        return self._slots[self._active]

    @property
    def view(self):
        return self._slots[self._active].view

    @property
    def loader(self):
        return self._slots[self._active].loader

    @property
    def sequence(self):
        return self._slots[self._active].sequence

    def _each_view(self):
        return [s.view for s in self._slots]

    def _both_slots(self):
        """Are both windows visible? (Double side by side and Wipe over each other)"""
        return self._view_mode in (exrnode.VIEW_DOUBLE, exrnode.VIEW_WIPE)

    # it used to be called _double(); the name stays as a shorthand for the same
    _double = _both_slots

    def _live_slots(self):
        """Windows that are VISIBLE and have something to show.

        Only for those is anything pre-fetched and cached - in Single,
        decoding the second window would be RAM and time thrown away.

        The exception is the difference: it needs BOTH inputs even when only
        one window is visible. Without that it would have nothing to subtract.
        """
        shown = self._live_slots_all()
        if any(fx.needs_other(s.view.effect) for s in shown):
            shown = list(self._slots)
        return [s for s in shown if s.sequence is not None]

    def _timeline_sequence(self):
        """The sequence the timeline range follows (the first connected one)."""
        for seq in self._sequences:
            if seq is not None:
                return seq
        return None

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        # The image should go edge to edge - only the bars around it keep an
        # inset, so their buttons have room to breathe.
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)

        # The top bar in the style of the Nuke Viewer: a dark strip with flat,
        # low controls. The style only matches this strip (objectName), so it
        # touches neither the in-image panels nor the timeline.
        bar_host = QtWidgets.QWidget(self)
        bar_host.setObjectName("cvTopBar")
        bar_host.setStyleSheet(TOP_BAR_STYLE)
        bar = QtWidgets.QHBoxLayout(bar_host)
        bar.setContentsMargins(5, 3, 5, 3)
        bar.setSpacing(3)

        def button(text, slot, tip=""):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(slot)
            b.setToolTip(tip)
            bar.addWidget(b)
            return b

        # How many windows. Which input and layer is picked inside each window
        # (see SlotBar), so in Double it is clear which controls are which.
        self._mode_combo = QtWidgets.QComboBox()
        self._mode_combo.addItems(exrnode.VIEW_MODES)
        self._mode_combo.setFixedWidth(78)
        self._mode_combo.setToolTip(
            "Single = one window.\n"
            "Double = two side by side or stacked (the split is on the node).\n"
            "Wipe   = both over each other, revealed by a line:\n"
            "         drag the circle to move it, the line to rotate it.\n"
            "Each window picks its own input and layer - top left inside it.")
        self._mode_combo.currentIndexChanged.connect(self._on_view_mode_ui)
        bar.addWidget(self._mode_combo)

        bar.addWidget(self._vline())

        self._chan = QtWidgets.QComboBox()
        self._chan.addItems(["RGB", "R", "G", "B", "A", "Luminance"])
        self._chan.setToolTip(
            "Channels - keys R G B A, Y = luminance.\n"
            "A second press of the same key returns to RGB (as in the Nuke Viewer).")
        self._chan.currentIndexChanged.connect(self._on_color_ui)
        bar.addWidget(self._chan)

        # Display always goes through the colour path: first the display
        # (Viewer Process), then the input transform.
        self._ocio_view = QtWidgets.QComboBox()
        self._ocio_view.setToolTip(
            "OCIO display - how linear is drawn onto the monitor.\n"
            "The display device from the config is in brackets.")
        self._ocio_view.currentIndexChanged.connect(self._on_ocio_view)
        bar.addWidget(self._ocio_view)

        self._ocio_in = QtWidgets.QComboBox()
        self._ocio_in.setToolTip(
            "Input transform - what the data in the FILE is in.\n"
            "EXR is usually scene-linear, but a log recording can turn up\n"
            "(AlexaV3LogC, ARRILogC4, SLog3, Log3G10, Cineon...).\n"
            "It is converted from there into linear, and only then to the monitor.")
        self._ocio_in.currentIndexChanged.connect(self._on_ocio_input)
        bar.addWidget(self._ocio_in)

        # the QC mode selector and CC (gain/gamma/saturation) are in the
        # panels right in the image
        button("Fit", self._fit_all,
               "Fit into the window - key F (or double click in the image)")

        bar.addStretch(1)
        root.addWidget(bar_host)

        # The stage lays the windows out itself (see _Stage): in Double into
        # exact halves, so the divider sits in the middle even after an aspect
        # change, and in Wipe over each other. The control panels are children
        # of the STAGE, not of a window - in Wipe a window is clipped by a mask
        # and that would cut them off with it.
        self._stage = _Stage(self)
        self._slot_bars = []
        for slot in self._slots:
            slot.view.setParent(self._stage)
            slot.view.probeChanged.connect(
                lambda info, s=slot: self._show_probe(info, s))
            slot.view.viewportChanged.connect(
                lambda src=slot.view: self._sync_viewport(src))
            # the scopes measure the visible crop, so panning and zooming has
            # to recompute them (throttled, see SCOPE_REFRESH_MS)
            slot.view.viewportChanged.connect(self._viewport_moved)
            # a click in a window makes it the active one (scopes, probe)
            slot.view.picked.connect(lambda s=slot: self._set_active_slot(s.index))

            # the window controls: the input toggle and the layer next to it,
            # top left
            sb = SlotBar(exrnode.INPUT_LABELS, self._stage)
            sb.sourceChanged.connect(
                lambda src, s=slot: self._on_source_ui(s, src))
            sb.layerChanged.connect(
                lambda layer, s=slot: self._on_layer_ui(s, layer))
            sb.panelToggled.connect(
                lambda key, on, s=slot: self._on_panel_toggle(s, key, on))
            sb.picked.connect(lambda s=slot: self._set_active_slot(s.index))
            sb.set_source(slot.source)
            slot.bar = sb
            self._slot_bars.append(sb)

            # The controls (CC, QC) on the left below the bar, the measuring
            # panels (histogram, vectorscope) opposite, top right. Each window
            # has its own set: in Double they are two different images and
            # measuring them with one scope would make no sense.
            ctrl = ControlStack(self._stage, below=sb)
            ctrl.cc.changed.connect(lambda vals, s=slot: self._on_cc(s, vals))
            ctrl.fx.changed.connect(
                lambda params, s=slot: self._on_effect_params(s, params))
            # the index is read from the combo, not from the signal argument -
            # see the note at the toggles in overlay.SlotBar
            ctrl.fx.combo.currentIndexChanged.connect(
                lambda *_a, s=slot: self._on_effect_ui(s,
                                                       s.controls.fx.combo.currentIndex()))
            ctrl.matte.toggled.connect(self._on_matte_ui)
            ctrl.matte.changed.connect(self._on_matte_values)
            slot.controls = ctrl

            sc = ScopeStack(self._stage)
            for scope in (sc.hist, sc.vscope):
                # expanding or enlarging a panel = compute that scope right away
                scope.resized.connect(lambda s=slot: self._refresh_scopes(s))
            slot.scopes = sc
            # in Wipe the controls of input B stand below those of input A, so
            # when A changes height (collapse, on/off), B has to move
            ctrl.refitted.connect(self._place_overlays)
            sb.moved.connect(self._place_overlays)

        self._stage.set_views([s.view for s in self._slots])
        # after every rearrangement raise the panels above the wipe line again
        # Resizing the panel moves the windows, so the controls have to be
        # placed again - not just raised. Without this the anchors kept the
        # values from the old size and the scope column, which hangs off the
        # RIGHT edge, stayed behind in the middle of a widened panel.
        self._stage.relaid.connect(self._place_overlays)
        # the blend intensity is held by the wipe circle right in the image
        self._stage.wipeOpacityChanged.connect(self._on_wipe_opacity)
        root.addWidget(self._stage, 1)

        # the default state (Single) has to apply straight away - _apply_settings
        # would not set it when it matches the node's default value
        self._apply_view_mode(exrnode.VIEW_SINGLE)

        # timeline (cache bar, frame numbers, mark IN/OUT and playhead in one)
        self.timeline = Timeline(self)
        self.timeline.frameChanged.connect(self.goto)
        self.timeline.rangeChanged.connect(self._on_range_changed)
        root.addWidget(self.timeline)

        tl = QtWidgets.QHBoxLayout()
        tl.setContentsMargins(4, 0, 4, 0)
        tl.setSpacing(4)

        self._start_btn = QtWidgets.QPushButton("|<")
        self._start_btn.setFixedWidth(30)
        self._start_btn.setToolTip("To the start of the range")
        self._start_btn.clicked.connect(lambda: self.goto(self.mark_in))
        tl.addWidget(self._start_btn)

        cache_btn = QtWidgets.QPushButton("Cache Range")
        cache_btn.setToolTip("Cache the IN..OUT range in the background")
        cache_btn.clicked.connect(self._cache_range)
        tl.addWidget(cache_btn)

        clear_btn = QtWidgets.QPushButton("Clear")
        clear_btn.setFixedWidth(52)
        clear_btn.setToolTip("Empty the RAM cache")
        clear_btn.clicked.connect(self._clear_cache)
        tl.addWidget(clear_btn)

        tl.addWidget(self._vline())

        self._back_btn = QtWidgets.QPushButton("<")
        self._back_btn.setFixedWidth(30)
        self._back_btn.setToolTip("Play backwards (J)")
        self._back_btn.clicked.connect(lambda: self._play(-1))
        tl.addWidget(self._back_btn)

        self._play_btn = QtWidgets.QPushButton("Play")
        self._play_btn.setCheckable(True)
        self._play_btn.setFixedWidth(56)
        self._play_btn.setToolTip("Play / stop (K)")
        self._play_btn.toggled.connect(self._on_play_toggled)
        tl.addWidget(self._play_btn)

        self._end_btn = QtWidgets.QPushButton(">|")
        self._end_btn.setFixedWidth(30)
        self._end_btn.setToolTip("To the end of the range")
        self._end_btn.clicked.connect(lambda: self.goto(self.mark_out))
        tl.addWidget(self._end_btn)

        self._mode = QtWidgets.QComboBox()
        self._mode.addItems(["Loop", "Ping-pong", "Once"])
        self._mode.setFixedWidth(92)
        self._mode.setToolTip("What to do at the end of the range")
        self._mode.currentIndexChanged.connect(self._on_mode_ui)
        tl.addWidget(self._mode)

        tl.addWidget(self._vline())
        tl.addWidget(QtWidgets.QLabel("In"))
        self._in_spin = QtWidgets.QSpinBox()
        self._in_spin.setFixedWidth(70)
        self._in_spin.setToolTip("Mark IN (the I key sets it to the current frame)")
        self._in_spin.valueChanged.connect(self._on_inout_spin)
        tl.addWidget(self._in_spin)

        tl.addWidget(QtWidgets.QLabel("Out"))
        self._out_spin = QtWidgets.QSpinBox()
        self._out_spin.setFixedWidth(70)
        self._out_spin.setToolTip("Mark OUT (the O key sets it to the current frame)")
        self._out_spin.valueChanged.connect(self._on_inout_spin)
        tl.addWidget(self._out_spin)

        reset = QtWidgets.QPushButton("Reset")
        reset.setFixedWidth(52)
        reset.setToolTip("Clear IN/OUT (or the middle mouse button in the timeline)")
        reset.clicked.connect(self._reset_in_out)
        tl.addWidget(reset)

        tl.addStretch(1)

        # The readout of the pixel under the cursor. The font is not set - it is
        # taken from the panel so it matches the FPS next to it. A fixed width,
        # so the numbers do not push the FPS around as the mouse moves.
        self._probe_lbl = QtWidgets.QLabel("")
        self._probe_lbl.setFixedWidth(238)     # + room for the A/B input tag
        self._probe_lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self._probe_lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self._probe_lbl.setToolTip(
            "Scene-linear RGBA of the pixel under the mouse.\n"
            "The P key freezes the readout so you can move the mouse away.")
        tl.addWidget(self._probe_lbl)
        tl.addWidget(self._vline())

        # the frame number is right in the timeline (a bubble at the playhead),
        # a separate spin box is no longer needed
        self._fps_lbl = QtWidgets.QLabel("-- fps")
        self._fps_lbl.setFixedWidth(66)
        self._fps_lbl.setAlignment(QtCore.Qt.AlignCenter)
        self._fps_lbl.setToolTip("Real playback FPS")
        tl.addWidget(self._fps_lbl)
        root.addLayout(tl)

        # the status line at the very bottom left
        self._status = QtWidgets.QLabel("-")
        self._status.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self._status.setToolTip(
            "source | cache | fill rate | queue | zoom\n\n"
            "KEYS:  J back  K stop/play  L forward  arrows = step a frame\n"
            "       R G B A channels, Y luminance (a second press returns RGB)\n"
            "       C CC, Q QC, H histogram, V vectorscope, W waveform\n"
            "       F fit into the window\n"
            "       1-7 QC modes, I/O mark in/out, P freeze the readout\n"
            "       X switch window (in Double)")
        self._status.setContentsMargins(4, 0, 4, 2)
        root.addWidget(self._status)

        shortcuts = [
            ("J", lambda: self._play(-1)),
            ("K", self._toggle_play),
            ("L", lambda: self._play(1)),
            ("Left", lambda: self.step(-1)),
            ("Right", lambda: self.step(1)),
            # channels as in the Nuke Viewer: a second press returns RGB
            ("R", lambda: self._toggle_channel(1)),
            ("G", lambda: self._toggle_channel(2)),
            ("B", lambda: self._toggle_channel(3)),
            ("A", lambda: self._toggle_channel(4)),
            ("Y", lambda: self._toggle_channel(5)),      # luminance
            ("I", self._set_mark_in),                    # mark IN here
            ("O", self._set_mark_out),                   # mark OUT here
            ("P", self._toggle_probe_freeze),            # freeze the pixel readout
            ("X", self._cycle_slot),                     # switch the active window
            ("F", self._fit_all),                        # fit into the window
        ]
        # Panels of the active window - each toggle has its own letter (PANEL_KEYS).
        for key, _label, _tip in overlay_mod.PANEL_BUTTONS:
            shortcuts.append((self.PANEL_KEYS[key],
                              lambda k=key: self._toggle_panel(k)))
        for i in range(len(fx.ORDER)):                   # 1=difference, 2=grain...
            shortcuts.append((str(i + 1),
                              lambda idx=i: self._set_effect(idx)))
        for key, fn in shortcuts:
            sc = QShortcut(QtGui.QKeySequence(key), self)
            sc.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)

    # ---- insets from Nuke ------------------------------------------------
    def showEvent(self, event):
        """Takes off the grey frame around us.

        Nuke wraps a registered Python panel in several containers and each has
        its own inset - together they make a wide grey frame that just takes
        room around the image. We walk a few levels up and zero their insets.
        Deliberately only a few: higher up is Nuke's own panel layout and we do
        not want to reach into that.
        """
        super(PlayerPanel, self).showEvent(event)
        widget, level = self.parentWidget(), 0
        while widget is not None and level < 4:
            try:
                lay = widget.layout()
                if lay is not None:
                    lay.setContentsMargins(0, 0, 0, 0)
                    lay.setSpacing(0)
            except Exception:
                pass
            widget, level = widget.parentWidget(), level + 1

    # ---- the wheel over the timeline -------------------------------------
    # Nuke wraps panels in a scrollable container (PythonPanel(scrollable=True))
    # which takes the wheel for itself and never lets it reach the timeline. So
    # we catch it at application level and, when the cursor is over the
    # timeline, hand it there.
    def _install_wheel_filter(self):
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    def eventFilter(self, obj, event):
        try:
            if event.type() == QtCore.QEvent.Wheel and self.isVisible():
                tl = getattr(self, "timeline", None)
                if tl is not None and tl.isVisible() and tl.underMouse():
                    tl.wheelEvent(event)
                    return True
        except Exception:
            pass
        return super(PlayerPanel, self).eventFilter(obj, event)

    # ------------------------------------------------- double view and inputs
    def _fit_all(self):
        for view in self._each_view():
            view.fit()

    def _viewport_moved(self):
        """Pan or zoom - the scopes describe a different crop now.

        Deliberately through a timer, not straight away: at the moment the
        signal arrives the window has only SCHEDULED a repaint, so the visible
        crop is still the old one. By the time the timer fires the repaint has
        happened and the scopes measure what is really on screen.
        """
        if not self._scope_timer.isActive():
            self._scope_timer.start()

    def _sync_viewport(self, source):
        """Zoom/pan from one window into the other - so the pixels line up.

        It is only carried in double view; in a single one it would just
        redraw the hidden window for nothing.
        """
        if not self._double():
            return
        state = source.viewport()
        for view in self._each_view():
            if view is not source:
                view.set_viewport(state)

    def _apply_view_mode(self, mode=None, split=None):
        """Single / Double / Wipe and how the window is split in Double."""
        if mode is not None:
            self._view_mode = max(0, min(len(exrnode.VIEW_MODES) - 1, int(mode)))
        if split is not None:
            self._split = max(0, min(len(exrnode.SPLIT_MODES) - 1, int(split)))
        self._stage.set_mode(self._view_mode, self._split)
        self._apply_slot_visibility()
        self._stage.wipe_opacity = self._settings.get("wipe_opacity", 1.0)
        self._apply_wipe_opacity(self._stage.wipe_opacity)
        self._apply_matte(self._settings)
        if self._mode_combo.currentIndex() != self._view_mode:
            self._mode_combo.blockSignals(True)
            self._mode_combo.setCurrentIndex(self._view_mode)
            self._mode_combo.blockSignals(False)
        # Single/Double decides whether the scopes are available -> reapply the panels
        for slot in self._slots:
            self._apply_panel_flags(self._settings, slot)
        self._sync_panel_buttons(self._settings)
        # a newly revealed window has nothing yet - fetch and start the cache
        self._show_current()
        self._request_around()
        self._schedule_cache()

    def _apply_slot_visibility(self):
        """Which windows are visible, which is active and where its controls go."""
        both = self._both_slots()
        if not both:
            self._active = 0              # in Single window 1 is always active
        visible = self._live_slots_all()
        for slot, sb in zip(self._slots, self._slot_bars):
            shown = slot in visible
            slot.view.setVisible(shown)
            sb.setVisible(shown)
            slot.controls.setVisible(shown)
            slot.scopes.setVisible(shown)
            sb.set_active(both and slot.index == self._active)
        self._place_overlays()
        if both:
            # when switching to two windows, line the views up on each other
            self._sync_viewport(self.view)

    def _place_overlays(self):
        """Where each window's controls belong.

        In Single and Double they sit in the corner of their window. In Wipe
        the windows overlap, so the controls would overlap too - therefore they
        stand one below the other: input A first, input B below it.
        """
        if self._placing:                 # set_anchor reports the move back here
            return
        self._placing = True
        try:
            self._place_overlays_inner()
        finally:
            self._placing = False

    def _place_overlays_inner(self):
        stage_w = self._stage.width()
        wipe = self._view_mode == exrnode.VIEW_WIPE
        for slot, sb in zip(self._slots, self._slot_bars):
            top = None
            if wipe:
                # the blocks below each other, as far apart as the first one is
                # from the top edge
                top = (overlay_mod.EDGE if slot.index == 0
                       else self._slots[0].controls.bottom() + overlay_mod.EDGE)
            box = slot.view.geometry()
            bar_at, scope_at = overlay_anchors(
                (box.x(), box.y(), box.right()), stage_w, top)
            sb.set_anchor(*bar_at)
            slot.scopes.set_anchor(*scope_at)
        self._raise_overlays()

    def _raise_overlays(self):
        """The controls belong above the wipe line - otherwise it draws over them."""
        for slot, sb in zip(self._slots, self._slot_bars):
            sb.raise_()
            slot.controls.raise_()
            slot.scopes.raise_()

    def _set_active_slot(self, index):
        """Switches which window the scopes and the pixel readout read from.

        It only makes sense in Double - in Single only window 1 is visible, so
        that is the active one too.
        """
        index = max(0, min(len(self._slots) - 1, int(index)))
        if not self._double():
            index = 0
        if index == self._active:
            return
        self._active = index
        self._apply_slot_visibility()
        self._show_current()
        self._request_around()
        self._schedule_cache()
        self._refresh_scopes()

    def _cycle_slot(self):
        """The X key: switch between the windows."""
        self._set_active_slot((self._active + 1) % len(self._slots))

    # ----------------------------------------------------- panel toggles
    # One key per panel: enabled means computed and shown, disabled neither.
    # The node (cv_<key>_<window>) and the in-image toggles use the same keys.
    PANEL_FLAG_KEYS = tuple(key for key, _l, _t in overlay_mod.PANEL_BUTTONS)

    # A key per toggle, matching the letter on it. Fit sits on F, so that H is
    # left for the histogram.
    PANEL_KEYS = {"cc": "C", "qc": "Q", "hist": "H", "vscope": "V",
                  "wave": "W"}

    def _flag(self, s, name, slot, default=False):
        """The value of a per-window setting. On the node it is a tuple, one item per window."""
        value = s.get(name, default)
        if isinstance(value, (tuple, list)):
            return value[slot.index] if slot.index < len(value) else default
        return value                      # an older node had one shared value

    def _set_slot_value(self, s, name, slot, value):
        """Sets a tuple item for one window and returns the new settings dict."""
        values = list(s.get(name, ()))
        while len(values) <= slot.index:
            values.append(False)
        values[slot.index] = value
        s[name] = tuple(values)
        return s

    def _toggle_panel(self, key):
        """The C/Q/H/V/W keys: toggle a panel of the ACTIVE window.

        In Single that is window 1, in Double the one you touched last. The
        scopes are unavailable in Double, so there the key stays silent - just
        like the greyed-out toggle.
        """
        if key in overlay_mod.SCOPE_KEYS and not self._scopes_allowed():
            return
        slot = self.active
        self._on_panel_toggle(slot, key,
                              not self._flag(self._settings, key, slot))

    def _on_panel_toggle(self, slot, key, on):
        """A CC/QC/H/V toggle in the image - it applies to ITS OWN window.

        An enabled panel is both computed and shown, a disabled one neither -
        it is one thing, so it is one checkbox on the node (cv_<key>_<window>).
        """
        s, writes = panel_toggle(self._settings, slot.index, slot.label,
                                 key, on, len(self._slots))
        self._settings = s
        for name, value in writes:
            self._write_knob(name, value)
        panel = None
        for stack in (slot.controls, slot.scopes):
            panel = panel or (stack.panels.get(key) if stack else None)
        if on and panel is not None:
            # I am switching it on because I want to see it - a collapsed panel
            # would only show a strip with the title
            panel.expand()
        self._apply_panel_flags(s, slot)
        self._sync_panel_buttons(s)
        self._set_active_slot(slot.index)
        if key == "qc":
            # the QC mode on the node only makes sense with the QC panel on
            node = self._get_node()
            if node is not None:
                exrnode.apply_view_visibility(node)

        # A short message about what actually happened. The Qt layer cannot be
        # tested inside Nuke (the terminal only has QCoreApplication, a
        # standalone QApplication freezes), so this is the only way to tell
        # from the outside which part of the chain a problem is in.
        self._toggle_note = "%s window %s: %s%s" % (
            key.upper(), slot.label, "on" if on else "off",
            "" if panel is None else
            (", panel " + ("visible" if panel.isVisible() else "HIDDEN")))
        self._toggle_note_t = time.monotonic()

    def _scopes_allowed(self):
        """The scopes are Single only.

        In Double each window is half the size and the histogram with the
        vectorscope would leave almost nothing of it. The setting on the node
        is NOT overwritten - it just does not apply for the time being, so
        after going back to Single the scope is exactly as it was.
        """
        return not self._double()

    def _apply_panel_flags(self, s, slot):
        """Applies the panel state of one window (wherever the settings came from)."""
        if slot.controls is None:
            return
        scopes_ok = self._scopes_allowed()
        slot.controls.set_visibility(
            self._flag(s, "cc", slot),
            self._flag(s, "qc", slot),
            matte=self._view_mode == exrnode.VIEW_DIMATTE)
        hist = self._flag(s, "hist", slot) and scopes_ok
        vscope = self._flag(s, "vscope", slot) and scopes_ok
        wave = self._flag(s, "wave", slot) and scopes_ok
        slot.scopes.set_visibility(hist, vscope, wave)
        slot.scopes.set_scope_active(hist, vscope, wave)
        self._on_cc(slot, slot.controls.cc.values())                  # cc
        self._apply_effect(slot, slot.controls.fx.combo.currentIndex())  # qc
        self._refresh_scopes(slot)

    def _sync_panel_buttons(self, s):
        """A toggle lights up when the panel is on in that window."""
        scopes_ok = self._scopes_allowed()
        for slot in self._slots:
            if slot.bar is None:
                continue
            slot.bar.set_scopes_available(scopes_ok)
            for key in self.PANEL_FLAG_KEYS:
                on = self._flag(s, key, slot)
                if key in overlay_mod.SCOPE_KEYS and not scopes_ok:
                    on = False
                slot.bar.set_panel(key, on)

    def _apply_wipe_opacity(self, value):
        """The blend of input B into A in Wipe.

        The window handles it itself while drawing (ImageView.set_opacity).
        QGraphicsOpacityEffect did not work: the effect draws the image aside
        and composites it by the widget's bounding rect, which the wipe mask
        changes - so the image jumped left and travelled while dragging the line.
        """
        value = max(0.0, min(1.0, float(value)))
        if self._view_mode != exrnode.VIEW_WIPE:
            value = 1.0
        self._slots[1].view.set_opacity(value)

    def _on_wipe_opacity(self, value):
        """The wipe circle in the image -> the image + the node (the node is truth)."""
        self._apply_wipe_opacity(value)
        self._write_knob("cv_wipe_opacity", float(value))
        self._settings["wipe_opacity"] = float(value)

    # ------------------------------------------------------------- DiMatte
    def _apply_matte(self, s):
        """Applies the mattes to the windows and looks after decoding DiMatte."""
        channels = s.get("matte", (False, False, False, True))
        shape = (s.get("matte_light", 1.0), s.get("matte_gain", 1.0),
                 s.get("matte_gamma", 1.0))
        on = self._view_mode == exrnode.VIEW_DIMATTE
        for slot in self._slots:
            slot.view.set_matte(channels if on else (), *shape)
            if slot.controls is not None:
                for ch, want in zip(overlay_mod.MATTE_CHANNELS, channels):
                    slot.controls.matte.set_channel(ch, want)
                for key, value in zip(("light", "gain", "gamma"), shape):
                    slot.controls.matte.set_value(key, value)
        self._request_around()
        self._schedule_cache()
        self._show_current()

    def _on_matte_ui(self, channel, on):
        """An RGBA toggle in the image -> the image + the node."""
        channels = list(self._settings.get("matte",
                                           (False, False, False, True)))
        try:
            idx = list(overlay_mod.MATTE_CHANNELS).index(channel)
        except ValueError:
            return
        while len(channels) <= idx:
            channels.append(False)
        channels[idx] = bool(on)
        s = dict(self._settings)
        s["matte"] = tuple(channels)
        self._settings = s
        self._write_knob("cv_matte_%s" % channel, bool(on))
        self._apply_matte(s)

    def _on_matte_values(self, values):
        """The matte sliders (lightness, gain, gamma) -> the image + the node."""
        s = dict(self._settings)
        for key in ("light", "gain", "gamma"):
            value = float(values.get(key, 1.0))
            s["matte_%s" % key] = value
            self._write_knob("cv_matte_%s" % key, value)
        self._settings = s
        self._apply_matte(s)

    def _on_view_mode_ui(self, index):
        self._apply_view_mode(mode=index)
        self._write_knob("cv_view_mode", self._view_mode)
        self._settings["view_mode"] = self._view_mode
        # in Single there is no point showing the settings of window 2 on the node
        node = self._get_node()
        if node is not None:
            exrnode.apply_view_visibility(node)

    def _on_source_ui(self, slot, source):
        """The toggle: the window should show a different node input."""
        self._set_source(slot, source)
        self._write_knob("cv_source_%s" % slot.label, slot.source)
        s = dict(self._settings)
        sources = list(s.get("sources", (0, 1)))
        sources[slot.index] = slot.source
        s["sources"] = tuple(sources)
        self._settings = s

    def _set_source(self, slot, source):
        """Switches a window to a different node input (with its sequence and layers)."""
        source = max(0, min(len(self._sequences) - 1, int(source)))
        if source == slot.source and slot.sequence is self._sequences[source]:
            return
        slot.source = source
        self._slot_bars[slot.index].set_source(source)
        self._bind_slot(slot)

    def _write_knob(self, name, value):
        """Writes to the node. Returns True on success.

        The error MUST NOT be swallowed: 400 ms later the watcher reads the old
        value off the node and puts the control back - from the outside it
        looks as if the button did not work. When the knob is missing (a node
        from an earlier version), we first try to add it; only if that does not
        help either is it reported.
        """
        node = self._get_node()
        if node is None:
            self._knob_note = "no EXRplayer node to store %s in" % name
            return False
        try:
            node[name].setValue(value)
            self._knob_note = ""
            return True
        except Exception:
            pass
        try:
            exrnode.ensure_knobs(node)        # a knob missing from an earlier version
            node[name].setValue(value)
            self._knob_note = ""
            return True
        except Exception as exc:
            self._knob_note = ("knob %s cannot be stored on node %s (%s)"
                               % (name, node.name(), exc))
            nuke.tprint("EXRplayer: " + self._knob_note)
            return False

    def _vline(self):
        """A vertical separator for the bar."""
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.VLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        return line

    # -------------------------------------------------------------- sources
    def _follow_tick(self):
        """A wrapper: one error must neither silence the panel forever nor
        flood the console.

        CAREFUL: the error has to be VISIBLE. It used to end up only in _hint,
        which is shown only when no sequence is attached - so when the watcher
        crashed on every tick over an attached plate, from the outside "nothing
        was happening".
        """
        try:
            self._follow_tick_inner()
            self._follow_note = ""
        except Exception as exc:
            msg = "%s: %s" % (type(exc).__name__, exc)
            if msg != self._last_error:          # report only NEW errors
                self._last_error = msg
                nuke.tprint("EXRplayer: error in follow: %s" % msg)
            self._follow_note = "NODE WATCHER ERROR: %s" % msg
            self._hint = "error: %s" % msg

    def _follow_tick_inner(self):
        """Follows the selected EXRplayer node and polices its input."""
        try:
            sel = [n for n in nuke.selectedNodes() if exrnode.is_player_node(n)]
        except Exception as exc:
            self._hint = "error reading the selection: %s" % exc
            return
        if sel:
            self._node_name = sel[0].fullName()
        node = self._get_node()
        if node is None:
            self._hint = ("there is no EXRplayer node "
                          "(EXRplayer > Create EXRplayer Node)")
            for i, seq in enumerate(self._sequences):
                if seq is not None:
                    self._set_input_sequence(i, None)
            return
        problem = exrnode.enforce_input(node)
        if problem != self._input_note:
            self._input_note = problem or ""
            if problem:
                nuke.tprint("EXRplayer: " + problem)
        # a safeguard: a Viewer attaches DOWNSTREAM (Viewer.input = our node),
        # so it is found differently than the inputs - the global guard watches
        # it too
        for name in exrnode.enforce_no_viewer():
            nuke.tprint("EXRplayer: Viewer '%s' disconnected (display is "
                        "handled by the EXRplayer panel)." % name)
        self._apply_settings(node)
        count = exrnode.input_count(node)     # an old NoOp node has only one
        for i in range(len(self._sequences)):
            src = node.input(i) if i < count else None
            seq = from_read_node(src)
            if seq != self._sequences[i]:
                self._set_input_sequence(i, seq)
        self._hint = self._input_hint(node, count)

    def _input_hint(self, node, count):
        """What the user is missing.

        An empty input is a legitimate state - the window simply stays black
        and this just says why. No automatic jumping elsewhere: when I switch a
        window to B, I want to see B even if it is empty for now.
        """
        empty = [exrnode.INPUT_LABELS[s.source] for s in self._live_slots_all()
                 if s.sequence is None]
        if not empty:
            return None
        if count < 2 and "B" in empty:
            return ("node '%s' has only one input (it is from an earlier "
                    "version) - create a new one for A/B: EXRplayer > Create "
                    "EXRplayer Node" % node.name())
        return ("input %s is empty - attach a Read with .exr to it"
                % " and ".join(sorted(set(empty))))

    def _live_slots_all(self):
        """Windows that are VISIBLE (even when they currently have no sequence).

        In Single it is ALWAYS window 1 - "single" means one view, not a switch
        between two. Whoever wants to see both switches to Double.
        """
        return self._slots if self._double() else [self._slots[0]]

    def _get_node(self):
        if self._node_name:
            n = nuke.toNode(self._node_name)
            if exrnode.is_player_node(n):
                return self._upgrade(n)
        found = exrnode.find_all()
        if found:
            self._node_name = found[0].fullName()
            return self._upgrade(found[0])
        return None

    def _upgrade(self, node):
        """Adds whatever the node is missing against today's version (once per node).

        MIND THE ORDER: _follow_tick_inner sets _node_name from the SELECTION
        before calling _get_node(). When the upgrade hung only on the find_all()
        branch, a node picked with the mouse was never upgraded - and a missing
        knob only shows up through its consequences: writing to it throws, 400
        ms later the watcher reads the default value and puts the choice back
        (that is how the input switch kept returning to A).
        """
        self._upgraded = getattr(self, "_upgraded", set())
        name = node.fullName()
        if name not in self._upgraded:
            self._upgraded.add(name)
            exrnode.prune_knobs(node)         # the obsolete ones out first
            exrnode.ensure_knobs(node)
            exrnode.ensure_order(node)        # ... and then fix the order
            exrnode.ensure_inputs(node)
            exrnode.apply_view_visibility(node)
        return node

    def _apply_settings(self, node):
        s = exrnode.settings(node)
        if s == self._settings:
            return
        old = self._settings
        self._settings = s
        if s["cache_mb"] != old.get("cache_mb"):
            self.cache.set_budget_mb(s["cache_mb"])
        # gain/gamma/saturation are handled by the in-image CC panel, the node
        # no longer holds them
        for view in self._each_view():
            view.set_color(channels=s["channels"])
        # the window sources before the mode: switching to Double reveals the
        # second window and it should already know which input it shows
        for i, slot in enumerate(self._slots):
            want = s.get("sources", ())
            want = want[i] if i < len(want) else slot.source
            if want != slot.source:
                self._set_source(slot, want)
        if (s.get("view_mode", exrnode.VIEW_SINGLE) != old.get("view_mode")
                or s.get("split", exrnode.SPLIT_SIDE) != old.get("split")):
            self._apply_view_mode(s.get("view_mode", exrnode.VIEW_SINGLE),
                                  s.get("split", exrnode.SPLIT_SIDE))
        if any(s.get(k) != old.get(k)
               for k in ("color_mgmt", "ocio_config", "ocio_display",
                         "ocio_view", "ocio_input", "nuke_display",
                         "nuke_input")):
            self._apply_color(s)
        for widget, key in ((self._chan, "channels"), (self._mode, "loop")):
            if widget.currentIndex() != s[key]:
                widget.blockSignals(True)
                widget.setCurrentIndex(s[key])
                widget.blockSignals(False)

        # ---- panels, each window its own ----
        for slot in self._slots:
            if slot.controls is None:
                continue
            # the QC mode: the combo has its signals blocked while syncing, so
            # a change on the node has to be applied to the image by hand
            want = self._flag(s, "effect", slot, 0)
            combo = slot.controls.fx.combo
            if combo.currentIndex() != want:
                combo.blockSignals(True)
                combo.setCurrentIndex(want)
                combo.blockSignals(False)
            if want != self._flag(old, "effect", slot, -1):
                self._apply_effect(slot, want)
            if any(self._flag(s, k, slot) != self._flag(old, k, slot, None)
                   for k in self.PANEL_FLAG_KEYS):
                self._apply_panel_flags(s, slot)
            if s.get("scope_opacity") != old.get("scope_opacity"):
                slot.scopes.set_opacity(s.get("scope_opacity", 0.75))
        if s.get("wipe_opacity") != old.get("wipe_opacity"):
            value = s.get("wipe_opacity", 1.0)
            self._stage.wipe_opacity = value
            self._stage.line.update()
            self._apply_wipe_opacity(value)
        if any(s.get(k) != old.get(k) for k in
               ("matte", "matte_light", "matte_gain", "matte_gamma")):
            self._apply_matte(s)
        if any(s.get(k) != old.get(k) for k in self.PANEL_FLAG_KEYS):
            self._sync_panel_buttons(s)

        if s.get("layers") != old.get("layers"):
            for i, slot in enumerate(self._slots):
                want = s.get("layers", ())
                want = want[i] if i < len(want) else None
                if want in (slot.layers or []) and want != slot.layer():
                    self._apply_layer(slot, want)
                    self._slot_bars[i].set_layers(slot.layers, want)
        # the exposure no longer has a widget in the panel - it comes off the node
        if self._playing:
            self._play_timer.setInterval(max(1, int(1000.0 / s["fps"])))

    def _matte_sequence(self):
        """The sequence of the DiMatte input, or None."""
        idx = exrnode.MATTE_INPUT
        return self._sequences[idx] if idx < len(self._sequences) else None

    def _matte_live(self):
        """Should DiMatte be decoded at all?"""
        return (self._view_mode == exrnode.VIEW_DIMATTE
                and self._matte_sequence() is not None
                and any(self._settings.get("matte", ())))

    def _set_input_sequence(self, index, seq):
        """The source of one node INPUT changed - rebind the windows showing it."""
        self._sequences[index] = seq
        self._sync_timeline_range()
        if index == exrnode.MATTE_INPUT:
            self._matte_loader.set_sequence(seq)
            self._request_around()
            self._schedule_cache()
            self._show_current()
            return
        for slot in self._slots:
            if slot.source == index:
                self._bind_slot(slot)

    def _sync_timeline_range(self):
        """The timeline range follows the first attached input.

        Both inputs have to have the same range (exrnode.enforce_input polices
        that), so it does not matter that the first one is taken - and when
        only one is attached, that is the one that applies.
        """
        seq = self._timeline_sequence()
        if seq is None:
            self._tl_range = None          # after disconnecting, let it be set again
            return
        # CAREFUL: we remember the range HERE, we do not read it from Timeline.
        # That one holds it in _first/_last and reading a non-existent .first
        # raised an exception that _follow_tick swallowed - from the outside it
        # looked as if an attached input never loaded at all.
        rng = (seq.first, seq.last)
        if rng == self._tl_range:
            return
        self._tl_range = rng
        for w in (self._in_spin, self._out_spin):
            w.blockSignals(True)
            w.setRange(seq.first, seq.last)
            w.blockSignals(False)
        self.timeline.set_range(seq.first, seq.last)
        self.timeline.set_in_out(seq.first, seq.last)
        self.mark_in, self.mark_out = seq.first, seq.last
        self.frame = seq.clamp(self.frame)
        self._sync_frame_widgets()

    def _bind_slot(self, slot):
        """Binds a window to the sequence of the input it picked."""
        seq = self._sequences[slot.source]
        slot.sequence = seq
        slot.loader.set_sequence(seq)
        slot.layers = []
        if seq is None:
            slot.view.set_frame(None)
            slot.source_info = "-"
            slot.fitted = False           # after a new connection, fit again
            self._slot_bars[slot.index].set_layers([exrcore.ROOT_LAYER])
            return

        # The image is fitted into the window ONLY THE FIRST TIME. When
        # switching the input with the toggle (or the layer), we keep the zoom
        # and pan - otherwise every switch would jump back to the whole frame
        # and comparing detail would be impossible.
        if not slot.fitted:
            slot.view.fit()
            slot.fitted = True
        self._fill_layers(slot)

        # Probe the first file RIGHT AWAY - so any problem (unsupported
        # compression, a missing file) shows up immediately instead of as
        # silent emptiness
        info = reader.probe(seq.path_for(self.frame))
        if not info.get("supported"):
            self._input_note = "input %s: we cannot read this EXR: %s (%s)" % (
                slot.source_label(), info.get("reason", "?"), seq.label())
            nuke.tprint("EXRplayer: " + self._input_note)
        else:
            slot.source_info = "%dx%d  %s" % (info["width"], info["height"],
                                              info.get("compression", "?"))
            nuke.tprint("EXRplayer: %s = %s  %s  channels %s  [reader: %s]"
                        % (slot.source_label(), seq.label(), slot.source_info,
                           ",".join(info.get("channels", [])),
                           info.get("backend", "?")))
        self._show_current()                   # in case it is already cached
        self._request_around()
        if self._settings.get("auto_cache", True):
            self._schedule_cache()

    # ------------------------------------------------------------- playback
    def _request_around(self):
        """Look-ahead around the playhead, but only within mark IN..OUT.

        Only for the inputs that are visible - in a single view, pre-fetching
        the other one would waste RAM and disk bandwidth for nothing.
        """
        s = self._settings
        loaders = [slot.loader for slot in self._live_slots()]
        if self._matte_live():
            loaders.append(self._matte_loader)
        for loader in loaders:
            loader.set_playhead(self.frame, self._direction,
                                ahead=s.get("lookahead", 24),
                                behind=s.get("behind", 4),
                                lo=self.mark_in, hi=self.mark_out)

    def _show_slot(self, slot):
        """Puts the frame FROM THE CACHE into a window. Returns whether it worked."""
        seq = slot.sequence
        if seq is None:
            return False
        arr = self.cache.get(slot.loader.key_for(self.frame))
        if arr is None:
            return False
        # we keep the previous frame for the temporal check (peek = do not
        # touch the LRU). The offset is adjustable - sometimes a duplicate is a
        # few frames back.
        prev = None
        back = self._temporal_offset(slot)
        if self.frame - back >= seq.first:
            prev = self.cache.peek(slot.loader.key_for(self.frame - back))
        # The difference needs the same frame from the OTHER INPUT. A window
        # showing a different input than this one takes priority - had we
        # simply taken "the other one", then after switching both windows to
        # the same input there would be nothing to subtract and the mask would
        # disappear.
        other = None
        if fx.needs_other(slot.view.effect):
            mates = [m for m in self._slots
                     if m is not slot and m.sequence is not None]
            mates.sort(key=lambda m: m.source == slot.source)
            if mates:
                other = self.cache.peek(mates[0].loader.key_for(self.frame))
        matte = (self.cache.peek(self._matte_loader.key_for(self.frame))
                 if self._matte_live() else None)
        slot.view.set_frame(arr, prev, other, matte)
        if slot.index == self._active:
            self._update_temporal_note(arr, prev)
        return True

    def _show_current(self):
        """Shows the current frame FROM THE CACHE. Never decodes on the GUI thread.

        Every visible window is handled separately: when one does not have the
        frame yet, the other is not held up by it and keeps its previous image.
        Only the active window counts towards the FPS, so the number always
        means the same thing.
        """
        shown = False
        for slot in self._live_slots():
            if self._show_slot(slot) and slot.index == self._active:
                shown = True
        if shown:
            self._refresh_scopes()
            self._shown += 1
        return shown

    def _refresh_scopes(self, slot=None):
        """Histogram + vectorscope from the frame currently displayed.

        Collapsed scopes are not computed at all (see ScopeStack.wants_scopes),
        so whoever does not use them pays nothing for them. Without an argument
        all VISIBLE windows are recomputed - a hidden one would be pointless.
        """
        slots = [slot] if slot is not None else self._live_slots_all()
        for s in slots:
            sc = s.scopes
            if sc is None or not sc.wants_scopes():   # can arrive while building the UI
                continue
            sc.update_scopes(s.view.scope_source())

    def _temporal_offset(self, slot=None):
        """How many frames back the comparison goes (only for the temporal check, else 1)."""
        slot = slot or self.active
        params = slot.fx_params.get(fx.TEMPORAL)
        if not params:
            return 1
        return max(1, int(round(fx.param(params, "offset", 1.0))))

    def _update_temporal_note(self, arr, prev):
        """Flags a duplicate / almost identical frame (only while the temporal check runs)."""
        if self.active.view.effect != fx.TEMPORAL:
            self._temporal_note = ""
            return
        ref = self.frame - self._temporal_offset()
        if prev is None:
            self._temporal_note = "frame %d is not in the cache" % ref
            return
        diff, identical = fx.frame_difference(arr, prev)
        if identical:
            # a bitwise match = an unambiguous duplicate, the only thing we alarm on
            self._temporal_note = ("!! DUPLICATE: frame %d is identical to %d"
                                   % (self.frame, ref))
        elif diff is not None:
            self._temporal_note = "difference against frame %d: %.4f" % (ref, diff)
        else:
            self._temporal_note = ""

    def goto(self, frame):
        if self.sequence is None:
            return
        self.frame = self.sequence.clamp(frame)
        self._sync_frame_widgets()
        self._show_current()
        self._request_around()
        self._maybe_reschedule_cache()   # a jump elsewhere -> move the cache window there
        self.timeline.update()

    def step(self, delta):
        self._direction = 1 if delta >= 0 else -1
        self.goto(self.frame + delta)

    def _sync_frame_widgets(self):
        self.timeline.set_frame(self.frame)

    def _on_slider(self, value):
        if value != self.frame:
            self.goto(value)

    # ---- mark IN / OUT ----
    def _on_range_changed(self, mark_in, mark_out):
        changed = (mark_in, mark_out) != (self.mark_in, self.mark_out)
        self.mark_in, self.mark_out = mark_in, mark_out
        for widget, value in ((self._in_spin, mark_in), (self._out_spin, mark_out)):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        if not changed or self.sequence is None:
            return
        # the cache follows the IN..OUT range: move the queue onto the new range
        for slot in self._slots:
            slot.loader.cancel_background()
        self._request_around()
        if self._settings.get("auto_cache", True):
            self._schedule_cache()

    def _on_inout_spin(self, _v):
        self.timeline.set_in_out(self._in_spin.value(), self._out_spin.value())

    def _set_mark_in(self):
        self.timeline.set_in_out(self.frame, max(self.frame, self.mark_out))

    def _set_mark_out(self):
        self.timeline.set_in_out(min(self.frame, self.mark_in), self.frame)

    def _reset_in_out(self):
        if self.sequence is not None:
            self.timeline.set_in_out(self.sequence.first, self.sequence.last)

    def _on_mode_ui(self, index):
        """Panel -> node (the node is the source of truth). 0 loop, 1 ping-pong, 2 once."""
        node = self._get_node()
        if node is not None:
            try:
                node["cv_loop"].setValue(index)
            except Exception:
                pass
        self._settings["loop"] = index

    def _play(self, direction):
        self._direction = direction
        if not self._play_btn.isChecked():
            self._play_btn.setChecked(True)
        else:
            self._play_t0 = time.monotonic()
            self._play_f0 = self.frame

    def _toggle_play(self):
        self._play_btn.setChecked(not self._play_btn.isChecked())

    def _on_play_toggled(self, on):
        self._playing = on
        self._play_btn.setText("Stop" if on else "Play")
        if on:
            fps = self._settings.get("fps", 24.0)
            self._play_t0 = time.monotonic()
            self._play_f0 = self.frame
            self._fps_t0 = time.monotonic()
            self._shown = 0
            self._schedule_cache()      # the cache window follows the playback direction
            self._play_timer.start(max(1, int(1000.0 / fps)))
        else:
            self._play_timer.stop()
            self._fps_lbl.setText("-- fps")

    def _tick(self):
        seq = self.sequence
        if seq is None:
            return
        s = self._settings
        if s.get("realtime", True):
            # the step follows elapsed time -> holds the pace, skips uncached
            elapsed = time.monotonic() - self._play_t0
            target = self._play_f0 + int(elapsed * s.get("fps", 24.0)) * self._direction
        else:
            target = self.frame + self._direction

        if s.get("cached_only", False):
            target = self._clamp_to_cached(target)

        self.frame = self._wrap(target)
        self._sync_frame_widgets()
        self._show_current()          # with no frame, the previous image stays
        self._request_around()
        self._maybe_reschedule_cache()   # the cache window travels with the playhead
        self.timeline.update()

        # CAREFUL: _shown is incremented by _show_current, and ONLY when the
        # frame really was in the cache. If it was not, the tick runs empty and
        # does not count towards the FPS - so the number says how many frames a
        # second were ACTUALLY shown.
        now = time.monotonic()
        dt = now - self._fps_t0
        if dt >= 0.5:
            self._fps_shown = self._shown / dt
            self._fps_lbl.setText("%.1f fps" % self._fps_shown)
            self._shown = 0
            self._fps_t0 = now

    def _wrap(self, frame):
        """Handles the end of the range (respects mark IN/OUT)."""
        lo, hi = self.mark_in, self.mark_out
        if lo <= frame <= hi:
            return int(frame)
        mode = self._settings.get("loop", 0)          # 0 loop, 1 ping-pong, 2 once
        clamped = max(lo, min(hi, int(frame)))
        if mode == 2:                                  # once -> stop at the end
            self._play_btn.setChecked(False)
            return clamped
        if mode == 1:                                  # ping-pong -> reverse
            self._direction *= -1
            self._play_t0 = time.monotonic()
            self._play_f0 = clamped
            return clamped
        span = hi - lo + 1                             # loop -> start again
        self._play_t0 = time.monotonic()
        self._play_f0 = lo + (int(frame) - lo) % span
        return self._play_f0

    def _frame_cached(self, frame):
        """Is the frame cached for ALL visible inputs?

        In Double there is no point jumping to a frame only one window has -
        it would be compared against whatever was left in the other one.
        """
        live = self._live_slots()
        if not live:
            return False
        return all(self.cache.contains(slot.loader.key_for(frame))
                   for slot in live)

    def _clamp_to_cached(self, target):
        """Do not leave the cached region ('cached only' mode)."""
        seq = self.sequence
        f = seq.clamp(target)
        if self._frame_cached(f):
            return f
        step = -1 if self._direction >= 0 else 1
        probe = f
        for _ in range(seq.frame_count):
            probe += step
            if probe < seq.first or probe > seq.last:
                break
            if self._frame_cached(probe):
                return probe
        return self.frame

    # ---------------------------------------------------------------- other
    def _on_frame_ready(self, frame):
        """A worker delivered a frame (already on the GUI thread thanks to the signal)."""
        if self.sequence is None:
            return
        if frame == self.frame and not self._playing:
            self._show_current()

    def _set_channel(self, index):
        """Switches the channel (the combo emits -> _on_color_ui -> node + display)."""
        if self._chan.currentIndex() != index:
            self._chan.setCurrentIndex(index)
        else:
            self._on_color_ui()

    def _toggle_channel(self, index):
        """The R/G/B/A/Y keys: switch the channel, a second press returns RGB (as in Nuke)."""
        self._set_channel(0 if self._chan.currentIndex() == index else index)

    def _apply_effect(self, slot, index):
        """The effect -> one window's image + its overlay. Does not touch the node.

        When QC is off (the QC toggle), the ordinary image is shown - but the
        panel keeps showing the selected mode and its sliders, so it can be set
        up and then simply switched on.
        """
        index = max(0, min(len(fx.ORDER) - 1, int(index)))
        effect = fx.ORDER[index]
        params = slot.fx_params.setdefault(effect, fx.defaults(effect))
        active = self._flag(self._settings, "qc", slot)
        slot.view.set_effect(effect if active else fx.NONE, params)
        slot.controls.fx.set_effect(effect, params)
        slot.legend = fx.legend(effect) if active else ""
        if fx.needs_other(effect):
            # from now on the second input has to be decoded, even when hidden
            self._request_around()
            self._schedule_cache()
            self._show_current()

    def _on_effect_ui(self, slot, index):
        """The QC mode choice -> display + a write to the node (the node is truth)."""
        index = max(0, min(len(fx.ORDER) - 1, int(index)))
        self._apply_effect(slot, index)
        self._set_active_slot(slot.index)
        self._write_knob("cv_effect_%s" % slot.label, index)
        s = dict(self._settings)
        values = list(s.get("effect", (0,) * len(self._slots)))
        while len(values) < len(self._slots):
            values.append(0)
        values[slot.index] = index
        s["effect"] = tuple(values)
        self._settings = s

    # ---------------------------------------------------------- colour path
    def _apply_color(self, s):
        """A junction driven by 'Color management' on the node: Nuke or OCIO."""
        if (s.get("color_mgmt", exrnode.MGMT_NUKE) == exrnode.MGMT_NUKE
                or not ocio.available()):
            self._apply_nuke_color(s)
        else:
            self._apply_ocio(s)

    def _apply_nuke_color(self, s):
        """The built-in transforms - one table, no OCIO."""
        self._ocio = None
        for view in self._each_view():
            view.set_ocio(None)
        self._ocio_note = ("OCIO is unavailable, using the built-in transforms"
                           if not ocio.available() else "")
        display = s.get("nuke_display") or nukelut.DEFAULT_DISPLAY
        space = s.get("nuke_input") or nukelut.DEFAULT_INPUT
        if display not in nukelut.DISPLAY_NAMES:
            display = nukelut.DEFAULT_DISPLAY
        if space not in nukelut.INPUT_NAMES:
            space = nukelut.DEFAULT_INPUT
        self._sync_color_combos(nukelut.DISPLAY_NAMES, display,
                                nukelut.INPUT_NAMES, space)
        for view in self._each_view():
            view.set_nuke_color(display, space)
        self._refresh_scopes()

    def _apply_ocio(self, s):
        """Builds the OCIO transform from the settings on the node.

        Baking the cube takes ~46 ms, so it is done ONLY on a real change of
        config / display / view - not on every tick.
        """
        configs = ocio.find_configs()
        if not configs:
            self._ocio_note = "OCIO: no config found"
            return
        idx = max(0, min(len(configs) - 1, int(s.get("ocio_config", 0))))
        path = configs[idx][1]

        if self._ocio is None or self._ocio.config_path != path:
            try:
                self._ocio = ocio.DisplayTransform(path)
            except ocio.OcioError as exc:
                self._ocio, self._ocio_note = None, "OCIO: %s" % exc
                return

        pairs = self._ocio.display_views()
        display = s.get("ocio_display")
        view = s.get("ocio_view")
        if (display, view) not in [(d, v) for _l, d, v in pairs]:
            display = self._ocio.default_display()
            view = self._ocio.default_view(display)
        space = s.get("ocio_input") or self._ocio.default_input
        if space not in self._ocio.input_spaces():
            space = self._ocio.default_input

        if (display, view, space) != (self._ocio.display, self._ocio.view,
                                      self._ocio.input_space):
            if not self._bake_ocio(display, view, space):
                return
            self._write_ocio_knobs(display, view, space)
        for v in self._each_view():
            v.set_ocio(self._ocio)

    def _bake_ocio(self, display, view, space):
        """Bakes the input -> linear -> monitor path and redraws RIGHT AWAY.

        The redraw has to be forced by hand - the transform is still the same
        object, so the change would otherwise only show up on the next frame.
        """
        try:
            self._ocio.bake(display, view, space)
            self._ocio_note = ""
        except ocio.OcioError as exc:
            self._ocio_note = "OCIO: %s" % exc
            return False
        self._sync_ocio_combos(display, view, space)
        for v in self._each_view():
            v.set_ocio(self._ocio)
            v.invalidate()
        self._refresh_scopes()          # the scopes describe the displayed values
        return True

    def _sync_ocio_combos(self, display, view, space):
        # Viewer Process: the label is "view (device)", the value (display, view)
        self._sync_color_combos(self._ocio.display_views(), (display, view),
                                self._ocio.input_spaces(), space)

    def _sync_color_combos(self, view_items, view_sel, in_names, in_sel):
        """Fills both combos. The items are either names or (label, d, v)."""
        pairs = [(v, v) if isinstance(v, str) else (v[0], (v[1], v[2]))
                 for v in view_items]
        self._ocio_view.blockSignals(True)
        have = [self._ocio_view.itemData(i)
                for i in range(self._ocio_view.count())]
        if have != [d for _l, d in pairs]:
            self._ocio_view.clear()
            for label, value in pairs:
                self._ocio_view.addItem(label, value)
        for i in range(self._ocio_view.count()):
            if self._ocio_view.itemData(i) == view_sel:
                self._ocio_view.setCurrentIndex(i)
                break
        self._ocio_view.blockSignals(False)

        names = list(in_names)
        self._ocio_in.blockSignals(True)
        if [self._ocio_in.itemText(i)
                for i in range(self._ocio_in.count())] != names:
            self._ocio_in.clear()
            self._ocio_in.addItems(names)
        if in_sel in names:
            self._ocio_in.setCurrentIndex(names.index(in_sel))
        self._ocio_in.blockSignals(False)

    def _write_ocio_knobs(self, display, view, space):
        node = self._get_node()
        if node is None:
            return
        try:
            node["cv_ocio_display"].setValue(display)
            node["cv_ocio_view"].setValue(view)
            node["cv_ocio_input"].setValue(space)
        except Exception as exc:
            # MUST NOT be swallowed: when the write fails (e.g. a missing knob
            # on an old node), 400 ms later the watcher puts the choice back
            # and it looks as if nothing is happening. At least let it be
            # visible why.
            self._ocio_note = "OCIO: the settings cannot be stored on the node (%s)" % exc

    def _on_ocio_view(self, *_a):
        if self._ocio is None:
            return self._on_nuke_color()
        pair = self._ocio_view.currentData()
        if pair:
            self._apply_ocio_choice(pair[0], pair[1], self._ocio.input_space)

    def _on_ocio_input(self, *_a):
        if self._ocio is None:
            return self._on_nuke_color()
        self._apply_ocio_choice(self._ocio.display, self._ocio.view,
                                self._ocio_in.currentText())

    def _on_nuke_color(self):
        """A choice in the built-in mode - also right away, not on the next watcher tick."""
        display = self._ocio_view.currentData() or nukelut.DEFAULT_DISPLAY
        space = self._ocio_in.currentText() or nukelut.DEFAULT_INPUT
        for view in self._each_view():
            view.set_nuke_color(display, space)
        self._refresh_scopes()
        node = self._get_node()
        if node is not None:
            try:
                node["cv_nuke_display"].setValue(display)
                node["cv_nuke_input"].setValue(space)
            except Exception as exc:
                self._ocio_note = "the settings cannot be stored on the node (%s)" % exc
        self._settings["nuke_display"] = display
        self._settings["nuke_input"] = space

    def _apply_ocio_choice(self, display, view, space):
        """A choice from the panel: bake RIGHT AWAY, not on the next watcher tick.

        The write to the node would otherwise come back only after 400 ms, so
        the image would react to a switch with a delay.
        """
        if (display, view, space) != (self._ocio.display, self._ocio.view,
                                      self._ocio.input_space):
            if not self._bake_ocio(display, view, space):
                return
        self._write_ocio_knobs(display, view, space)
        # so the node watcher does not take our own changes for a new choice
        self._settings["ocio_display"] = display
        self._settings["ocio_view"] = view
        self._settings["ocio_input"] = space

    # each channel in its own colour - the same one its histogram curve has
    PROBE_COLORS = ("#ff5050", "#50e050", "#6090ff", "#c0c0c0")

    def _show_probe(self, info, slot=None):
        """Scene-linear RGBA of the pixel under the cursor, coloured per channel.

        The values come from the window the mouse is over - so in Double the
        input of that window is written in front of them.
        """
        if info is None:
            self._probe_lbl.setText("")
            return
        lin, raw = info["linear"], info["raw"]
        vals = [float(lin[0]), float(lin[1]), float(lin[2]),
                float(raw[3]) if raw.shape[0] > 3 else 1.0]

        parts = []
        for value, color in zip(vals, self.PROBE_COLORS):
            # negatives and values above 1.0 are a finding, not a detail - they
            # are shown in bold, but the channel colour stays so it is still
            # clear which one it is
            cell = _nbsp("%7.4f" % value)
            if value < 0.0 or value > 1.0:
                cell = "<b>%s</b>" % cell
            parts.append('<span style="color:%s">%s</span>' % (color, cell))

        snow = "&#10052;" if self.view.probe_frozen else ""
        tag = ("<b>%s</b>&nbsp;" % slot.source_label()
               if slot is not None and self._double() else "")
        self._probe_lbl.setText(snow + tag + "&nbsp;".join(parts))

    def _fill_layers(self, slot):
        """Finds the layers in the file (including multipart EXR parts) for a window."""
        seq = slot.sequence
        slot.layers = ([exrcore.ROOT_LAYER] if seq is None
                       else reader.layers_of(seq.path_for(seq.first)))
        # The layer stored on the node can only be applied NOW, once we know
        # which layers the file actually has. _apply_settings runs before the
        # sequence is known, so the stored choice would otherwise never take
        # effect.
        want = self._settings.get("layers", ())
        want = want[slot.index] if slot.index < len(want) else None
        if want in slot.layers:
            self._apply_layer(slot, want)
        elif slot.layer() not in slot.layers:
            self._apply_layer(slot, slot.layers[0])
        self._slot_bars[slot.index].set_layers(
            slot.layers or [exrcore.ROOT_LAYER], slot.layer())

    def _on_layer_ui(self, slot, layer):
        """A layer choice -> new data for this window."""
        self._apply_layer(slot, layer)
        self._set_active_slot(slot.index)
        self._write_knob("cv_layer_%s" % slot.label, layer)
        s = dict(self._settings)
        layers = list(s.get("layers", ()))
        while len(layers) <= slot.index:
            layers.append(exrcore.ROOT_LAYER)
        layers[slot.index] = layer
        s["layers"] = tuple(layers)
        self._settings = s

    def _apply_layer(self, slot, layer):
        """Switches the layer of one window.

        The cache is NOT cleared - the keys carry the layer too
        (FrameLoader.key_for), so layer data does not mix and going back to the
        original layer is instant. It used to be cleared entirely, which with
        two windows threw the other one away too.
        """
        if not slot.loader.set_layer(layer):
            return
        slot.view.set_frame(None)
        self._show_current()
        self._request_around()
        if self._settings.get("auto_cache", True):
            self._schedule_cache()

    def _toggle_probe_freeze(self):
        """P: freezes the readout, so the mouse can be moved away (e.g. onto a slider panel)."""
        frozen = not self.view.probe_frozen
        for view in self._each_view():
            view.probe_frozen = frozen
        if not frozen:
            self._probe_lbl.setText("")

    def _on_cc(self, slot, values):
        """The in-image CC panel: gain / gamma / saturation, for its window.

        When CC is off (the CC toggle), neutral values are sent - the image
        goes through with no colour correction and saturation no longer costs
        20 ms a frame. The sliders in the panel remember their values, so
        switching it on brings them back.
        """
        if not self._flag(self._settings, "cc", slot):
            values = {}
        slot.view.set_color(gain=values.get("gain", 1.0),
                            gamma=values.get("gamma", 1.0),
                            saturation=values.get("sat", 1.0))
        self._refresh_scopes(slot)      # the scopes describe what is visible

    def _on_effect_params(self, slot, params):
        """A slider in the overlay moved -> redraw with the new settings."""
        effect = slot.view.effect
        slot.fx_params[effect] = dict(params)
        slot.view.set_effect_params(params)
        if effect == fx.TEMPORAL:
            self._show_current()          # a changed "offset" wants a different previous frame

    def _set_effect(self, index):
        """Keys 1-7: the QC mode of the active window (switched off by the QC toggle)."""
        combo = self.active.controls.fx.combo
        if combo.currentIndex() != index:
            combo.setCurrentIndex(index)

    def _on_color_ui(self, *_a):
        """A colour change from the panel -> propagated to the node (the node is truth)."""
        node = self._get_node()
        if node is not None:
            try:
                node["cv_channels"].setValue(self._chan.currentIndex())
            except Exception:
                pass
        for view in self._each_view():
            view.set_color(channels=self._chan.currentIndex())
        self._refresh_scopes()

    def _cache_capacity(self):
        """How many frames fit into the cache budget (0 = we do not know yet).

        In double view it is a PAIR of frames - half as many fit into the same
        RAM, and that is how it is written in the status line as well.
        """
        total = 0
        for slot in self._live_slots():
            arr = slot.view.current_frame_array()
            if arr is not None and arr.nbytes:
                total += int(arr.nbytes)
        if not total:
            return 0
        return self.cache.frame_capacity(total)

    def _schedule_cache(self):
        """Plans the rolling cache window from the playhead in the playback direction.

        On a long shot that does not fit in memory as a whole, only the window
        ahead of the playhead is cached; frames behind fall out through the
        LRU. Thanks to that the cache does not thrash and playback does not
        stutter.
        """
        live = self._live_slots()
        if not live:
            return
        cap = self._cache_capacity()
        limit = None
        if cap:
            limit = max(8, int(cap * 0.9))     # headroom, so we do not evict ourselves
        self._cache_anchor = self.frame
        loaders = [slot.loader for slot in live]
        if self._matte_live():
            loaders.append(self._matte_loader)
        for loader in loaders:
            loader.cache_range(self.mark_in, self.mark_out,
                               anchor=self.frame,
                               direction=self._direction, limit=limit)

    def _cache_range(self):
        """The Cache Range button: caches the IN..OUT range from the playhead."""
        self._schedule_cache()

    def _maybe_reschedule_cache(self):
        """When the playhead has run away from the anchor, move the cache window forward."""
        if self.sequence is None or not self._settings.get("auto_cache", True):
            return
        cap = self._cache_capacity()
        if not cap:
            return
        # re-anchor once we have travelled a third of the window (not every frame)
        if abs(self.frame - self._cache_anchor) >= max(8, cap // 3):
            self._schedule_cache()

    def _clear_cache(self):
        self.cache.clear()
        self.cache_runs = []
        self.timeline.set_cache_runs([])
        self._request_around()

    def _cache_runs(self):
        """Continuous runs for the cache bar.

        In Double the INTERSECTION is shown - a frame is green only when both
        windows have it. Playback would not go further anyway (see
        _frame_cached), so one-sided green would lie.
        """
        live = self._live_slots()
        if not live:
            return []
        sets = [set(self.cache.cached_frames(s.sequence, s.loader.key_fn()))
                for s in live]
        frames = sorted(set.intersection(*sets))
        runs = []
        for f in frames:
            if runs and f == runs[-1][1] + 1:
                runs[-1][1] = f
            else:
                runs.append([f, f])
        return runs

    def _source_label(self):
        """A description of the source: in Double both windows."""
        parts = []
        for slot in self._live_slots():
            name = slot.sequence.label()
            if slot.layer() != exrcore.ROOT_LAYER:
                name += " [%s]" % slot.layer()
            parts.append("%s %s  [%s]" % (slot.source_label(), name,
                                          slot.source_info) if self._double()
                         else "%s  [%s]" % (name, slot.source_info))
        return "   ||   ".join(parts)

    def _refresh_status(self):
        if not self._playing:
            # catches the cases where a scope appears without a new frame
            # (switching tabs, ticking a checkbox on the node). During playback
            # it is pointless - there the scopes are recomputed with every
            # displayed frame.
            self._refresh_scopes()
        # CAREFUL: the timeline keeps ITS OWN copy of the runs, they have to be
        # handed to it (forget that and the cache does not show in the timeline
        # at all)
        self.cache_runs = self._cache_runs()
        self.timeline.set_cache_runs(self.cache_runs)
        for slot in self._slots:
            if slot.view.last_error:
                self._status.setText("DISPLAY ERROR (window %s): %s"
                                     % (slot.label, slot.view.last_error))
                return
        if self.sequence is None:
            self._status.setText(self._hint or "no input")
            return
        cs = self.cache.stats()
        # the fill rate and the queue are the sum of both inputs - together
        # they share one disk and one cache
        fill = sum(s.loader.fill_fps for s in self._slots)
        pending = sum(s.loader.pending for s in self._slots)
        failures = sum(s.loader.stats()["failures"] for s in self._slots)
        if failures and cs["frames"] == 0:
            errs = [s.loader.last_error for s in self._slots
                    if s.loader.last_error]
            self._status.setText("CANNOT LOAD: %s" % (errs[0] if errs else "?"))
            return
        # how many PAIRS of frames fit into the budget (on 6K this is crucial)
        capacity = ""
        cap = self._cache_capacity()
        if cap:
            need = self.mark_out - self.mark_in + 1
            capacity = " | fits %d f%s" % (
                cap, "" if cap >= need else " of %d NEEDS MORE RAM" % need)
        txt = ("%s | RAM %.0f/%.0f MB (%d f)%s | fill %.0f fps "
               "| queue %d | zoom %.0f%%"
               % (self._source_label(),
                  cs["mb_used"], cs["mb_budget"], cs["frames"], capacity,
                  fill, pending, self.view.zoom_percent()))
        if failures:
            txt += "  | errors %d" % failures
        if self.view.ocio_active():
            txt += "  | OCIO %s" % self._ocio.label()
        # the node the panel is following - when there are several in the graph
        # it is immediately clear which one is being written to
        if self._node_name:
            txt += "  | node %s" % self._node_name
        # ordered by severity: what failed, then what is missing, then labels.
        # The confirmation of the last toggle disappears on its own, so it does
        # not take room permanently.
        if self._toggle_note and time.monotonic() - self._toggle_note_t > 4.0:
            self._toggle_note = ""
        note = (self._follow_note or self._knob_note or self._input_note
                or self._toggle_note or self._ocio_note or self._temporal_note
                or self.active.legend)      # the QC legend of the active window
        if note:
            txt = note + "     ||     " + txt
        self._status.setText(txt)
        # errors, a duplicate and a disconnected input -> make them obvious
        if (self._follow_note or self._knob_note or self._input_note
                or self._temporal_note.startswith("!!")):
            self._status.setStyleSheet("color:#ff6060; font-weight:bold;")
        else:
            self._status.setStyleSheet("")

    # CAREFUL: no closeEvent stopping the loaders! When docking, Nuke closes and
    # reopens the panel - stopped threads would never start again and the panel
    # would stay empty forever. The threads are daemons, they end with Nuke.
