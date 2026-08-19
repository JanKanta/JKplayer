"""
Timeline in the style of the Nuke Viewer.

Everything in one strip (no separate slider plus a bar below it):
  * the cache bar right in the timeline (yellow strip along the bottom edge)
  * frame numbers and ticks
  * the mark IN / OUT range (outside it the timeline is dimmed), draggable
    triangles
  * a playhead with the frame number
Click / drag = scrub. Dragging a triangle moves IN/OUT.
"""

from .qtcompat import QtCore, QtGui, QtWidgets, event_pos

# heights of the individual bands (from the bottom) - kept low so the timeline
# does not eat space
TIMELINE_H = 30      # total height of the strip
CACHE_H = 4          # cache strip, one input
CACHE_H2 = 7         # ...and with two, so each line is still 3 px
MARK_H = 6           # band with the IN/OUT triangles
ALT_H = 11           # second number row: input B in ITS OWN numbering
LABEL_H = 12         # bubble with the frame number
HANDLE_GRAB = 7      # how close to a marker the mouse grabs it


class Timeline(QtWidgets.QWidget):

    frameChanged = QtCore.Signal(int)
    rangeChanged = QtCore.Signal(int, int)

    def __init__(self, parent=None):
        super(Timeline, self).__init__(parent)
        self.setFixedHeight(TIMELINE_H)
        self.setMouseTracking(True)
        # WheelFocus + NoMousePropagation: the wheel should end here and must
        # not fall through into the scrollable container Nuke wraps panels in
        self.setFocusPolicy(QtCore.Qt.WheelFocus)
        self.setAttribute(QtCore.Qt.WA_NoMousePropagation, True)
        self.setToolTip(
            "Timeline:\n"
            "  drag / click    = scrub\n"
            "  wheel           = zoom in / out\n"
            "  Ctrl+wheel      = step one frame\n"
            "  middle button   = pan a zoomed timeline\n"
            "  double click    = zoom out to the whole range\n"
            "  triangles       = mark IN / OUT (drag them)")

        self._first = 1                # the whole sequence range
        self._last = 100
        self._vfirst = 1               # what is VISIBLE right now (timeline zoom)
        self._vlast = 100
        self._in = 1
        self._out = 100
        self._frame = 1
        self._cache_runs = []          # [(from, to), ...] - input A
        self._cache_runs_b = []        # the same for input B, its own line
        self._annot_runs = []          # frames carrying a drawn note
        # Input B's own numbering, for the second row under the cache lanes.
        # (shift, first, last) on the TIMELINE; shift is what has to come off a
        # timeline frame to get back to B's own. None = one input, no row.
        self._alt = None
        self._drag = None              # None | "frame" | "in" | "out" | "pan"
        self._pan_x = 0.0
        self._pan_view = None
        self.MIN_SPAN = 4              # most zoomed in = 4 frames across

        # colours (dark Nuke look)
        self.c_bg = QtGui.QColor(38, 38, 38)
        self.c_bg_out = QtGui.QColor(26, 26, 26)      # outside IN/OUT
        self.c_tick = QtGui.QColor(90, 90, 90)
        self.c_text = QtGui.QColor(170, 170, 170)
        # Cache lines. Saturated on purpose - they are thin, and a washed-out
        # 2 px line at the bottom of a dark strip is not a readout. Input A
        # keeps the yellow it always had; B gets orange, far enough apart to
        # tell at a glance which one is behind.
        self.c_cache = QtGui.QColor(255, 205, 20)     # A = in RAM
        self.c_cache_b = QtGui.QColor(255, 120, 20)   # B = in RAM
        self.c_cache_bg = QtGui.QColor(240, 200, 40, 46)   # soft tint
        # blue = there is a drawn note on this frame. Semi-transparent so the
        # cache underneath still shows through - the two say different things
        # and neither should hide the other.
        self.c_annot = QtGui.QColor(70, 150, 255, 150)
        self.c_play = QtGui.QColor(235, 235, 235)
        # the second row is tinted like B's cache line, so there is no doubt
        # which input those numbers belong to
        self.c_alt_text = QtGui.QColor(255, 150, 70)
        self.c_alt_bg = QtGui.QColor(22, 22, 22)
        self.c_mark = QtGui.QColor(120, 170, 255)

    # ------------------------------------------------------------------ API
    def set_range(self, first, last):
        first, last = int(first), int(last)
        if last < first:
            last = first
        keep_in = self._in if first <= self._in <= last else first
        keep_out = self._out if first <= self._out <= last else last
        self._first, self._last = first, last
        self._in, self._out = keep_in, keep_out
        self._vfirst, self._vlast = first, last      # zoom back to the whole range
        self.update()

    def view_all(self):
        """Zoom the timeline out to the whole range (double click)."""
        self._vfirst, self._vlast = self._first, self._last
        self.update()

    def zoom_view(self, factor, at_frame=None):
        """Zooms the timeline in/out around the given frame (mouse wheel)."""
        span = self._vlast - self._vfirst + 1
        new_span = int(round(span * factor))
        new_span = max(self.MIN_SPAN, min(self._last - self._first + 1, new_span))
        if new_span == span:
            return
        anchor = self._frame if at_frame is None else at_frame
        # the anchor keeps its relative position in the window -> zoom "under
        # the cursor"
        frac = (anchor - self._vfirst) / float(max(1, span - 1)) if span > 1 else 0.5
        vf = int(round(anchor - frac * (new_span - 1)))
        vf = max(self._first, min(self._last - new_span + 1, vf))
        self._vfirst, self._vlast = vf, vf + new_span - 1
        self.update()

    def pan_view(self, delta_frames):
        span = self._vlast - self._vfirst + 1
        vf = int(round(self._vfirst + delta_frames))
        vf = max(self._first, min(self._last - span + 1, vf))
        if vf != self._vfirst:
            self._vfirst, self._vlast = vf, vf + span - 1
            self.update()

    def ensure_visible(self, frame):
        """When the playhead leaves the zoomed part, move the window after it."""
        if self._vfirst <= frame <= self._vlast:
            return
        span = self._vlast - self._vfirst + 1
        vf = max(self._first, min(self._last - span + 1, int(frame) - span // 2))
        self._vfirst, self._vlast = vf, vf + span - 1

    def set_frame(self, frame):
        frame = self._clamp(frame)
        if frame != self._frame:
            self._frame = frame
            self.ensure_visible(frame)     # when zoomed in, follow the playhead
            self.update()

    def set_cache_runs(self, runs, second=None):
        """Cached frames. `second` is the OTHER input, drawn as its own line.

        Two lines rather than one: with a single bar showing where both inputs
        agree, an input still filling up looked identical to one that was not
        loading at all, and there was no way to tell which of the two was
        holding playback back.
        """
        runs = runs or []
        second = second or []
        if runs == self._cache_runs and second == self._cache_runs_b:
            return
        self._cache_runs = runs
        self._cache_runs_b = second
        self.update()

    def set_alt_numbering(self, shift=None, first=None, last=None):
        """A second row of frame numbers: input B in ITS OWN numbering.

        The timeline counts in one set of numbers, but the two inputs are
        almost never delivered on the same one - a plate at 1001 against a
        render at 1. Reading the shift off the node and doing the arithmetic in
        your head is exactly the sort of thing that goes wrong at 8pm, so B's
        own numbers are written under the cache lanes and can simply be read.

        `shift` is what to take off a timeline frame to get B's own number.
        None (or no B at all) hides the row and the strip goes back to its
        original height.
        """
        alt = None if shift is None else (int(shift), int(first), int(last))
        if alt == self._alt:
            return
        self._alt = alt
        self.setFixedHeight(TIMELINE_H + (ALT_H if alt else 0))
        self.update()

    def set_annotated(self, runs):
        """Which frames carry a drawn note - see annotate.Annotations.runs()."""
        runs = runs or []
        if runs == self._annot_runs:
            return
        self._annot_runs = runs
        self.update()

    def set_in_out(self, mark_in, mark_out):
        self._in = self._clamp(mark_in)
        self._out = self._clamp(mark_out)
        if self._out < self._in:
            self._in, self._out = self._out, self._in
        self.update()
        self.rangeChanged.emit(self._in, self._out)

    @property
    def frame(self):
        return self._frame

    @property
    def mark_in(self):
        return self._in

    @property
    def mark_out(self):
        return self._out

    # ------------------------------------------------------------- geometry
    def _clamp(self, f):
        return max(self._first, min(self._last, int(f)))

    def _count(self):
        """How many frames are VISIBLE right now (the timeline may be zoomed)."""
        return max(1, self._vlast - self._vfirst + 1)

    def _x(self, frame):
        """Left edge of that frame's cell (within the displayed range)."""
        return (frame - self._vfirst) * self.width() / float(self._count())

    def _frame_at(self, x):
        f = self._vfirst + int(x * self._count() / max(1.0, float(self.width())))
        return self._clamp(f)

    def _frame_at_exact(self, x):
        """Without clamping to the range - for anchoring the zoom."""
        return self._vfirst + x * self._count() / max(1.0, float(self.width()))

    def _cell_w(self):
        return self.width() / float(self._count())

    # -------------------------------------------------------------- drawing
    def paintEvent(self, _event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, False)
        w, h = self.width(), self.height()
        cw = self._cell_w()
        alt_h = ALT_H if self._alt else 0
        track_h = h - MARK_H - alt_h            # above B's numbers and the markers

        # background: the whole strip dark, the IN..OUT area lighter
        p.fillRect(0, 0, w, track_h, self.c_bg_out)
        x_in = self._x(self._in)
        x_out = self._x(self._out) + cw
        p.fillRect(QtCore.QRectF(x_in, 0, x_out - x_in, track_h), self.c_bg)

        # CACHE - two layers, so it reads well even in a narrow strip:
        #   1) a soft tint over the whole cached area
        #   2) a solid strip along the bottom edge - one LINE PER INPUT, so an
        #      input that is still filling cannot be mistaken for one that is
        #      not loading at all. With a single input there is one line, the
        #      full height it always was.
        p.setPen(QtCore.Qt.NoPen)
        two = bool(self._cache_runs_b)
        if two:
            # Taller when it has to carry two: splitting the single-input
            # height would leave 1.5 px a line, which is not a readout.
            line_h = (CACHE_H2 - 1) / 2.0
            lanes = [(self._cache_runs, track_h - CACHE_H2, line_h,
                      self.c_cache),
                     (self._cache_runs_b, track_h - line_h, line_h,
                      self.c_cache_b)]
        else:
            lanes = [(self._cache_runs, track_h - CACHE_H, CACHE_H,
                      self.c_cache)]
        for runs, y, line_h, color in lanes:
            for a, b in runs:
                if b < self._vfirst or a > self._vlast:
                    continue                          # outside the zoomed part
                xa = self._x(max(a, self._vfirst))
                xb = self._x(min(b, self._vlast)) + cw
                width_px = max(1.0, xb - xa)
                p.fillRect(QtCore.QRectF(xa, 0, width_px, track_h),
                           self.c_cache_bg)
                p.fillRect(QtCore.QRectF(xa, y, width_px, line_h), color)

        # ANNOTATED frames, over the cache and under the ticks: a solid block
        # down the whole track, because "there is a note here" is what you scan
        # the timeline for, and a thin line at the edge would be lost among the
        # cache stripes.
        for a, b in self._annot_runs:
            if b < self._vfirst or a > self._vlast:
                continue
            xa = self._x(max(a, self._vfirst))
            xb = self._x(min(b, self._vlast)) + cw
            p.fillRect(QtCore.QRectF(xa, 0, max(1.0, xb - xa), track_h),
                       self.c_annot)

        # ticks and frame numbers (spacing follows the width)
        font = p.font()
        font.setPointSize(7)
        p.setFont(font)
        step = self._label_step()
        f = self._vfirst - (self._vfirst % step) if step else self._vfirst
        while f <= self._vlast:
            if f >= self._vfirst:
                x = self._x(f)
                p.setPen(self.c_tick)
                p.drawLine(int(x), 0, int(x), 4)
                p.setPen(self.c_text)
                p.drawText(QtCore.QRectF(x + 2, 0, 60, LABEL_H),
                           QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop, str(f))
            f += step

        # INPUT B'S OWN NUMBERS, directly under the cache lanes.
        # Deliberately at the SAME tick positions as the row above, because the
        # whole point is reading one against the other: 1001 over 0001 says the
        # shift at a glance, where two rows on different steps would say nothing.
        if self._alt:
            shift, afirst, alast = self._alt
            p.fillRect(0, track_h, w, alt_h, self.c_alt_bg)
            p.setPen(self.c_alt_text)
            f = self._vfirst - (self._vfirst % step) if step else self._vfirst
            while f <= self._vlast:
                # only where B actually has frames - past its end it is holding
                # a still, and printing numbers there would invent footage
                if f >= self._vfirst and afirst <= f <= alast:
                    p.drawText(QtCore.QRectF(self._x(f) + 2, track_h, 60, alt_h),
                               QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                               str(f - shift))
                f += step

        # the IN/OUT marker band
        p.fillRect(0, track_h + alt_h, w, MARK_H, QtGui.QColor(20, 20, 20))
        self._draw_marker(p, self._x(self._in), track_h + alt_h, True)
        self._draw_marker(p, self._x(self._out) + cw, track_h + alt_h, False)

        # playhead
        px = self._x(self._frame)
        p.setPen(QtCore.Qt.NoPen)
        p.fillRect(QtCore.QRectF(px, 0, max(1.0, cw), track_h),
                   QtGui.QColor(255, 255, 255, 60))
        p.setPen(QtGui.QPen(self.c_play, 1))
        p.drawLine(int(px), 0, int(px), track_h)

        # bubble with the frame number (the only place the frame is shown -
        # the spin box is gone)
        label = str(self._frame)
        fm = p.fontMetrics()
        tw = fm.horizontalAdvance(label) + 10
        bx = min(max(px - tw / 2.0, 0), max(0, w - tw))
        by = max(0.0, (track_h - CACHE_H - LABEL_H) / 2.0)
        rect = QtCore.QRectF(bx, by, tw, LABEL_H)
        p.setPen(QtCore.Qt.NoPen)
        p.fillRect(rect, QtGui.QColor(15, 15, 15, 235))
        p.setPen(self.c_play)
        p.drawText(rect, QtCore.Qt.AlignCenter, label)

    def _draw_marker(self, p, x, y, is_in):
        """IN triangle (pointing right) / OUT triangle (pointing left)."""
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(self.c_mark)
        s = MARK_H
        if is_in:
            pts = [QtCore.QPointF(x, y), QtCore.QPointF(x + s, y),
                   QtCore.QPointF(x, y + s)]
        else:
            pts = [QtCore.QPointF(x, y), QtCore.QPointF(x - s, y),
                   QtCore.QPointF(x, y + s)]
        p.drawPolygon(QtGui.QPolygonF(pts))
        p.setBrush(QtCore.Qt.NoBrush)

    def _label_step(self):
        """A sensible spacing for the numbers, so they do not overlap."""
        px_per_frame = self.width() / float(self._count())
        for step in (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000):
            if step * px_per_frame >= 44:
                return step
        return max(1, self._count() // 8)

    # ----------------------------------------------------------- interaction
    def _hit_marker(self, x):
        cw = self._cell_w()
        if abs(x - self._x(self._in)) <= HANDLE_GRAB:
            return "in"
        if abs(x - (self._x(self._out) + cw)) <= HANDLE_GRAB:
            return "out"
        return None

    def mousePressEvent(self, event):
        x = event_pos(event).x()
        if event.button() == QtCore.Qt.LeftButton:
            hit = (self._hit_marker(x)
                   if event_pos(event).y() >= self.height() - MARK_H - 6 else None)
            self._drag = hit or "frame"
            self._apply_drag(x)
        elif event.button() == QtCore.Qt.MiddleButton:
            self._drag = "pan"                  # pan a zoomed timeline
            self._pan_x = x
            self._pan_view = self._vfirst
            self.setCursor(QtCore.Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        x = event_pos(event).x()
        if self._drag == "pan":
            per_px = self._count() / max(1.0, float(self.width()))
            self.pan_view((self._pan_x - x) * per_px
                          + (self._pan_view - self._vfirst))
            return
        if self._drag:
            self._apply_drag(x)
            return
        near = self._hit_marker(x)
        self.setCursor(QtCore.Qt.SizeHorCursor if near else QtCore.Qt.ArrowCursor)

    def mouseReleaseEvent(self, _event):
        if self._drag == "pan":
            self.unsetCursor()
        self._drag = None

    def mouseDoubleClickEvent(self, _event):
        self.view_all()                          # double click = zoom out to all

    def _apply_drag(self, x):
        f = self._frame_at(x)
        if self._drag == "in":
            self._in = min(f, self._out)
            self.update()
            self.rangeChanged.emit(self._in, self._out)
        elif self._drag == "out":
            self._out = max(f, self._in)
            self.update()
            self.rangeChanged.emit(self._in, self._out)
        else:
            if f != self._frame:
                self._frame = f
                self.update()
                self.frameChanged.emit(f)

    def wheelEvent(self, event):
        """Wheel = zoom the timeline around the cursor (as in Nuke).

        Ctrl+wheel = step one frame (in case anyone finds it handy).
        """
        delta = event.angleDelta().y()
        if delta == 0:
            return
        if event.modifiers() & QtCore.Qt.ControlModifier:
            f = self._clamp(self._frame + (1 if delta > 0 else -1))
            if f != self._frame:
                self._frame = f
                self.update()
                self.frameChanged.emit(f)
        else:
            at = self._frame_at_exact(event_pos(event).x())
            self.zoom_view(1.0 / 1.25 if delta > 0 else 1.25, at_frame=at)
        event.accept()
