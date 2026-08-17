"""
Semi-transparent panels sitting RIGHT IN THE IMAGE (top right, stacked):
  CC          - gain / gamma / saturation, collapsed by default
  QC          - the check mode selector plus the sliders of the active one
  Histogram   - axis 0 to 55, the clipping boundary at 1.0
  Vectorscope - pixel colour

The input and layer selection is NOT here - it is in the bar at the top of the
window, because it concerns WHAT is displayed, not HOW it is displayed.

The scopes are computed ONLY while their panel is expanded (see is_open) - a
collapsed panel costs nothing.

The sliders deliberately take no focus and ignore the wheel - the panel has
shortcuts (J/K/L, arrows) and the wheel zooms the image, so the overlay would
otherwise steal them. The menus (_Combo) are the exception: over them both the
wheel and the arrows change the choice.
"""

import math

import numpy as np
from .qtcompat import QtCore, QtGui, QtWidgets, event_pos

from . import annotate
from . import effects as fx
from . import scopes

PANEL_W = 238                      # both panels the same width, so they line up
CC_LABEL_W = 62                    # CC has short labels (Gain, Gamma, ...)
FX_LABEL_W = 104                   # QC has longer parameter names
EDGE = 10                          # inset from the edge of the image

STYLE = """
QFrame#cvOverlay {
    background-color: rgba(18, 18, 18, 190);
    border: 1px solid rgba(255, 255, 255, 40);
    border-radius: 5px;
}
QLabel { color: #d8d8d8; background: transparent; }
QLabel#cvTitle { color: #ffffff; font-weight: bold; }
QSlider::groove:horizontal {
    height: 3px; background: rgba(255,255,255,55); border-radius: 1px;
}
QSlider::handle:horizontal {
    width: 9px; margin: -4px 0; border-radius: 4px; background: #c8c8c8;
}
QToolButton { color: #d8d8d8; background: transparent; border: none; }
/* The value readout is typed into, so it has to look reachable without
   turning every row into a box: a bare number until the mouse is on it. */
QLineEdit {
    color: #d8d8d8; background: transparent;
    border: 1px solid transparent; border-radius: 2px; padding: 0px 1px;
    selection-background-color: rgba(255,255,255,60);
}
QLineEdit:hover { border: 1px solid rgba(255,255,255,45); }
QLineEdit:focus {
    background: rgba(255,255,255,22); border: 1px solid rgba(255,255,255,80);
    color: #ffffff;
}
QComboBox {
    color: #e0e0e0; background: rgba(255,255,255,22);
    border: 1px solid rgba(255,255,255,45); border-radius: 3px;
    padding: 1px 4px;
}
"""

# CC: (key, label, min, max, default, decimals, curve)
# A curve > 1 packs the short end of the range towards the left of the slider.
# Gain and gamma can therefore go quite high (16 and 8) and still be adjustable
# in hundredths around 1.00 - with a straight slider the whole useful 0.5-2
# stretch would take a few pixels. With a curve of 3 the value 1.00 sits at
# roughly two fifths of the length.
CC_PARAMS = [
    ("gain", "Gain", 0.0, 16.0, 1.0, 2, 3.0),
    ("gamma", "Gamma", 0.1, 8.0, 1.0, 2, 3.0),
    ("sat", "Saturation", 0.0, 4.0, 1.0, 2, 1.0),
]


# Slider steps. Longer ranges (gain up to 16) need a finer step, otherwise one
# nudge around 1.00 would jump by a few hundredths.
SLIDER_STEPS = 400


def cc_defaults():
    return {p[0]: p[4] for p in CC_PARAMS}


class _Slider(QtWidgets.QSlider):
    """A slider that steals neither the keyboard nor the wheel.

    A MIDDLE click puts this one control back to its default - which is why
    there is no reset button on the panels: resetting the one thing you just
    moved is what is actually wanted, and it needs no room in the image.
    """

    resetRequested = QtCore.Signal()

    def __init__(self, parent=None):
        super(_Slider, self).__init__(QtCore.Qt.Horizontal, parent)
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self.setRange(0, SLIDER_STEPS)   # the steps map onto min..max
        self.setFixedHeight(12)

    def wheelEvent(self, event):
        event.ignore()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MiddleButton:
            self.resetRequested.emit()
            event.accept()
            return
        super(_Slider, self).mousePressEvent(event)


class _NumberEdit(QtWidgets.QLineEdit):
    """The value readout, typed into directly.

    Left/right (and up/down) step the value by 1, or by 0.1 with shift held -
    the arrows are taken over from the text cursor on purpose: the field holds
    one short number, so nudging the value is far more useful than moving
    between its digits. A middle click resets, like the slider.
    """

    stepped = QtCore.Signal(float)          # how far, signed
    resetRequested = QtCore.Signal()

    def __init__(self, parent=None):
        super(_NumberEdit, self).__init__(parent)
        self.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.setFocusPolicy(QtCore.Qt.ClickFocus)   # never steals the keyboard
        self.setFixedWidth(40)
        # Hand the keyboard straight back on enter. Otherwise the field keeps
        # focus and space, J/K/L and the rest would be typed into it instead of
        # driving playback.
        self.returnPressed.connect(self.clearFocus)

    def keyPressEvent(self, event):
        key = event.key()
        down = key in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Down)
        up = key in (QtCore.Qt.Key_Right, QtCore.Qt.Key_Up)
        if down or up:
            step = 0.1 if event.modifiers() & QtCore.Qt.ShiftModifier else 1.0
            self.stepped.emit(-step if down else step)
            event.accept()
            return
        if key in (QtCore.Qt.Key_Escape,):
            self.clearFocus()               # give up, keep the value as it was
            event.accept()
            return
        super(_NumberEdit, self).keyPressEvent(event)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MiddleButton:
            self.resetRequested.emit()
            event.accept()
            return
        super(_NumberEdit, self).mousePressEvent(event)

    def focusInEvent(self, event):
        super(_NumberEdit, self).focusInEvent(event)
        self.selectAll()                    # type over it, no need to clear it


class _Combo(QtWidgets.QComboBox):
    """A menu that can be changed as soon as the mouse is over it.

    The wheel changes the choice (Qt does that itself) and so do the up/down
    arrows - for which the menu takes focus on enter and releases it on leave.
    Without that the arrows would belong to the panel and one would have to
    click first.

    Focus is NOT released while the popup is open - otherwise the list would
    close the moment the mouse moved down onto an item.
    """

    def __init__(self, parent=None):
        super(_Combo, self).__init__(parent)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def enterEvent(self, event):
        super(_Combo, self).enterEvent(event)
        self.setFocus(QtCore.Qt.MouseFocusReason)

    def leaveEvent(self, event):
        super(_Combo, self).leaveEvent(event)
        try:
            popup_open = self.view().isVisible()
        except Exception:
            popup_open = False
        if not popup_open:
            self.clearFocus()


class _BandSlider(QtWidgets.QWidget):
    """One slider with several handles - the value map band boundaries.

    The track carries exactly the colours the bands have in the image, so while
    dragging you see how far each band reaches: between the second and third
    handle the green is exactly the green in the image.

    The scale is not linear. The four grey steps live below the first boundary
    (around 1.0), whereas the other boundaries usually sit at 20 and 55 - on a
    linear axis the first handle would be squashed into a few pixels and could
    not be grabbed. Below 1.0 therefore gets SUB_PART of the width, the rest is
    logarithmic.
    """

    changed = QtCore.Signal(dict)

    HEIGHT = 34
    BAR_H = 12
    GRAB = 7                     # how far from a handle it still grabs
    SUB_PART = 0.35              # share of the width for values below 1.0
    NEG_W = 9                    # red block for negative values, before zero

    def __init__(self, specs, parent=None):
        super(_BandSlider, self).__init__(parent)
        self.setFixedHeight(self.HEIGHT)
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self.setMouseTracking(True)
        # specs: [(key, label, min, max, value, decimals)]
        self._keys = [s[0] for s in specs]
        self._lo = [float(s[2]) for s in specs]
        self._hi = [float(s[3]) for s in specs]
        self._dec = [int(s[5]) for s in specs]
        self._vals = [float(s[4]) for s in specs]
        self._vmax = max(self._hi)
        self._drag = None

    # ---- value <-> position conversion ---------------------------------
    def _to_t(self, value):
        value = max(0.0, float(value))
        if value <= 1.0:
            return self.SUB_PART * value
        return self.SUB_PART + (1.0 - self.SUB_PART) * (
            math.log(value) / math.log(self._vmax))

    def _to_value(self, t):
        t = min(1.0, max(0.0, t))
        if t <= self.SUB_PART:
            return t / self.SUB_PART
        return math.exp((t - self.SUB_PART) / (1.0 - self.SUB_PART)
                        * math.log(self._vmax))

    def _span(self):
        """Width of the value axis - without the red negative block on the left."""
        return max(1, self.width() - self.NEG_W - 1)

    def _x(self, value):
        return self.NEG_W + int(round(self._to_t(value) * self._span()))

    def _t_at(self, x):
        return (x - self.NEG_W) / float(self._span())

    # ---- state ---------------------------------------------------------
    def keys(self):
        return list(self._keys)

    def values(self):
        return {k: v for k, v in zip(self._keys, self._vals)}


    def set_value(self, key, value):
        """A value from outside (node, reset). No signal is sent."""
        if key not in self._keys:
            return
        i = self._keys.index(key)
        self._vals[i] = self._clamp(i, float(value))
        self.update()

    def _clamp(self, i, value):
        """A handle leans on its neighbours and on its own range."""
        value = max(self._lo[i], min(self._hi[i], value))
        if i > 0:
            value = max(value, self._vals[i - 1] * 1.02)
        if i < len(self._vals) - 1:
            value = min(value, self._vals[i + 1] / 1.02)
        return value

    # ---- mouse ---------------------------------------------------------
    def wheelEvent(self, event):
        event.ignore()

    def mousePressEvent(self, event):
        x = event_pos(event).x()
        near = [(abs(self._x(v) - x), i) for i, v in enumerate(self._vals)]
        dist, i = min(near)
        if dist <= self.GRAB + 3:
            self._drag = i
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag is None:
            return
        t = self._t_at(event_pos(event).x())
        self._vals[self._drag] = self._clamp(self._drag, self._to_value(t))
        self.update()
        self.changed.emit(self.values())

    def mouseReleaseEvent(self, _event):
        self._drag = None

    # ---- drawing -------------------------------------------------------
    def paintEvent(self, _event):
        p = QtGui.QPainter(self)
        w = max(1, self.width())
        top = 2
        xs = [self._x(v) for v in self._vals]

        # A small red block at the far left = negative values. Zero is right
        # after it and only from there does the value axis run, so it is
        # visible where zero sits.
        p.fillRect(0, top, self.NEG_W, self.BAR_H,
                   QtGui.QColor(*fx.VM_NEGATIVE))

        # four grey steps below the first boundary - as in the image
        grays = fx.VM_GRAYS
        for step in range(4):
            a = self._x(self._vals[0] * step / 4.0)
            b = self._x(self._vals[0] * (step + 1) / 4.0)
            p.fillRect(a, top, max(1, b - a), self.BAR_H,
                       QtGui.QColor(grays[step], grays[step], grays[step]))
        for start, end, color in ((xs[0], xs[1], fx.VM_OVER_1),
                                  (xs[1], xs[2], fx.VM_OVER_20),
                                  (xs[2], w, fx.VM_OVER_55)):
            p.fillRect(start, top, max(1, end - start), self.BAR_H,
                       QtGui.QColor(*color))
        p.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 120), 1))
        p.drawRect(0, top, w - 1, self.BAR_H)

        font = p.font()
        font.setPointSize(max(6, font.pointSize() - 2))
        p.setFont(font)

        # the zero mark at the boundary of the red block and the axis
        zero_x = self.NEG_W
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 200), 1))
        p.drawLine(zero_x, top, zero_x, top + self.BAR_H)
        if xs[0] - zero_x > 16:            # keep the labels off each other
            p.setPen(QtGui.QColor(200, 200, 200))
            p.drawText(QtCore.QRect(zero_x - 12, top + self.BAR_H + 2, 24, 12),
                       QtCore.Qt.AlignCenter, "0")

        for i, (x, value) in enumerate(zip(xs, self._vals)):
            p.setPen(QtGui.QPen(QtGui.QColor(20, 20, 20), 3))
            p.drawLine(x, top - 2, x, top + self.BAR_H + 2)
            p.setPen(QtGui.QPen(QtGui.QColor(240, 240, 240), 1))
            p.drawLine(x, top - 2, x, top + self.BAR_H + 2)
            text = ("%%.%df" % self._dec[i]) % value
            rect = QtCore.QRect(x - 24, top + self.BAR_H + 2, 48, 12)
            p.setPen(QtGui.QColor(200, 200, 200))
            p.drawText(rect, QtCore.Qt.AlignCenter, text)
        p.end()


class _Panel(QtWidgets.QFrame):
    """A semi-transparent frame with a title, a collapse button and a body."""

    resized = QtCore.Signal()          # when the height changes (stack relayout)

    GRIP = 6                     # how wide the strip at the edge that drags is
    CORNER = 14                  # ... and the square in the corner, both axes
    # Wider than it was (0.7-3.0). A histogram blown right up is a legitimate
    # way to work - you want to see the shape of the toe, not a thumbnail - and
    # the panel is inside the image, so the person dragging it can see exactly
    # how much room it is taking.
    MIN_SCALE, MAX_SCALE = 0.5, 6.0

    # Whether the BOTTOM edge drags the height on its own. Off by default: on a
    # panel of rows of sliders there is no spare height to hand out, so a grip
    # there would be a cursor that changes and then does nothing. The scopes
    # turn it on, because a histogram genuinely wants to be short and wide one
    # minute and tall the next.
    RESIZE_V = False

    def __init__(self, title, width, label_w, collapsed=False, parent=None):
        super(_Panel, self).__init__(parent)
        self.setObjectName("cvOverlay")
        self.setStyleSheet(STYLE)
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self.setFixedWidth(width)
        self.setMouseTracking(True)          # for the cursor over the edge

        self.grip_side = QtCore.Qt.RightEdge  # which edge the panel drags by
        # ("w"|"h", position on press, size on press)
        self._drag = None
        self._alpha = None                    # opacity of the backdrop
        self._base_w = width
        self._scale = 1.0
        self._vscale = 1.0                    # height, dragged on its own
        self.active = True           # should anything be computed for it (is_open)
        self._label_w = label_w
        self._collapsed = bool(collapsed)
        self._values = {}
        self._sliders = {}
        self._labels = {}
        self._ranges = {}            # key -> (min, max, decimals, curve)
        self._choices = {}           # key -> menu (parameters without a scale)
        self._defaults = {}          # key -> value a middle click goes back to
        self._bands = None           # the multi-handle slider (value map)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(8, 5, 8, 7)
        outer.setSpacing(3)

        head = QtWidgets.QHBoxLayout()
        head.setSpacing(4)
        self.head = head               # subclasses can add controls here
        self._title = QtWidgets.QLabel(title, self)
        self._title.setObjectName("cvTitle")
        head.addWidget(self._title)
        head.addStretch(1)
        self._toggle = QtWidgets.QToolButton(self)
        self._toggle.setFixedSize(16, 16)
        self._toggle.setFocusPolicy(QtCore.Qt.NoFocus)
        self._toggle.setToolTip("Collapse / expand")
        self._toggle.clicked.connect(self._on_toggle)
        head.addWidget(self._toggle)
        outer.addLayout(head)

        self._body = QtWidgets.QWidget(self)
        self.form = QtWidgets.QVBoxLayout(self._body)
        self.form.setContentsMargins(0, 1, 0, 0)
        self.form.setSpacing(2)
        outer.addWidget(self._body)

        self._apply_collapsed()

    # ------------------------------------------------------------------ API
    def values(self):
        return dict(self._values)

    def is_open(self):
        """Should anything be computed for it?

        For the scopes it is the checkbox on the node (`active`) that decides,
        NOT visibility - a panel can be hidden and the scope still computed, or
        the other way round. CC and QC do not use `active`, the panel handles
        those elsewhere.
        """
        if not self.active:
            return False
        return not self._collapsed and self.isVisible()

    # ------------------------------------------------- resizing with the mouse
    # The size is not stored anywhere - it is a thing of the moment, not a
    # script setting. It is dragged by the edge that faces into the image: the
    # right one for a panel on the left, the left one for a panel on the right.
    # Anywhere else in the panel the mouse behaves as before.
    def _on_grip(self, x):
        if self.grip_side == QtCore.Qt.LeftEdge:
            return x <= self.GRIP
        return x >= self.width() - self.GRIP

    def _on_grip_v(self, y):
        """Over the bottom edge - the height grip."""
        return self.RESIZE_V and y >= self.height() - self.GRIP

    def _on_corner(self, x, y):
        """Over the bottom corner ON THE SIDE THAT FACES THE IMAGE.

        Both axes at once, which is what a hand reaches for when it wants the
        panel simply bigger. It is the same corner the panel is dragged from
        sideways - bottom left for a panel on the right, bottom right for one
        on the left - so the grips stay on the edge you can actually get at
        without crossing the panel.
        """
        if not self.RESIZE_V or y < self.height() - self.CORNER:
            return False
        if self.grip_side == QtCore.Qt.LeftEdge:
            return x <= self.CORNER
        return x >= self.width() - self.CORNER

    def mouseMoveEvent(self, event):
        pos = event_pos(event)
        x, y = int(pos.x()), int(pos.y())
        if self._drag is not None:
            axis, start, size = self._drag
            if axis in ("h", "wh"):
                sy = start[1] if axis == "wh" else start
                sh = size[1] if axis == "wh" else size
                self.set_height_scale((sh + (y - sy)) / float(self.base_height()))
            if axis in ("w", "wh"):
                sx = start[0] if axis == "wh" else start
                sw = size[0] if axis == "wh" else size
                delta = x - sx
                if self.grip_side == QtCore.Qt.LeftEdge:
                    delta = -delta           # dragging left -> the panel grows
                self.set_scale((sw + delta) / float(self._base_w))
            return
        # The CORNER is asked first, then the bottom, then the side - from the
        # most specific to the least, so the corner is not swallowed by the
        # edge strips that overlap it.
        if self._on_corner(x, y):
            self.setCursor(QtCore.Qt.SizeBDiagCursor
                           if self.grip_side == QtCore.Qt.LeftEdge
                           else QtCore.Qt.SizeFDiagCursor)
        elif self._on_grip_v(y):
            self.setCursor(QtCore.Qt.SizeVerCursor)
        elif self._on_grip(x):
            self.setCursor(QtCore.Qt.SizeHorCursor)
        else:
            self.setCursor(QtCore.Qt.ArrowCursor)

    def mousePressEvent(self, event):
        pos = event_pos(event)
        x, y = int(pos.x()), int(pos.y())
        if event.button() != QtCore.Qt.LeftButton:
            return
        if self._on_corner(x, y):
            self._drag = ("wh", (x, y), (self.width(), self.content_height()))
            event.accept()
        elif self._on_grip_v(y):
            self._drag = ("h", y, self.content_height())
            event.accept()
        elif self._on_grip(x):
            self._drag = ("w", x, self.width())
            event.accept()

    def mouseReleaseEvent(self, _event):
        self._drag = None

    def leaveEvent(self, _event):
        self.unsetCursor()

    def set_scale(self, scale):
        """Panel enlargement (by dragging its edge)."""
        scale = max(self.MIN_SCALE, min(self.MAX_SCALE, float(scale)))
        if abs(scale - self._scale) < 1e-3:
            return
        self._scale = scale
        self.setFixedWidth(self.scaled_width())
        self._scaled()
        self.updateGeometry()
        self.adjustSize()
        self.resized.emit()

    def set_base_width(self, width):
        """The base width of the panel (before dragging the edge).

        The panels below a window's bar follow it, so the whole column lines
        up - the bar is as wide as its content (layer names come in different
        lengths), so a fixed width would drift apart from it.
        """
        width = max(120, int(width))
        if width == self._base_w:
            return
        self._base_w = width
        self.setFixedWidth(self.scaled_width())
        self._scaled()
        self.updateGeometry()
        self.adjustSize()
        self.resized.emit()

    def scaled_width(self):
        return int(round(self._base_w * self._scale))

    # ---- height, on its own ----
    def base_height(self):
        """The unscaled height of the content. Subclasses that resize say so."""
        return 100

    def content_height(self):
        return int(round(self.base_height() * self._vscale))

    def set_height_scale(self, scale):
        """Enlargement of the CONTENT height (by dragging the bottom edge)."""
        if not self.RESIZE_V:
            return
        scale = max(self.MIN_SCALE, min(self.MAX_SCALE, float(scale)))
        if abs(scale - self._vscale) < 1e-3:
            return
        self._vscale = scale
        self._scaled()
        self.updateGeometry()
        self.adjustSize()
        self.resized.emit()

    def inner_width(self):
        """The width inside the frame - what the content has available."""
        m = self.layout().contentsMargins()
        return max(20, self.scaled_width() - m.left() - m.right())

    def _scaled(self):
        """Overridden by subclasses that scale their content as well as the width."""

    def set_title(self, text):
        self._title.setText(text)

    def set_opacity(self, value):
        """Backdrop opacity (0 = outlines only, 1 = opaque)."""
        alpha = int(round(max(0.0, min(1.0, float(value))) * 255))
        if alpha == self._alpha:
            return
        self._alpha = alpha
        self.setStyleSheet(
            STYLE + "QFrame#cvOverlay { background-color: rgba(18,18,18,%d); }"
            % alpha)

    # ----------------------------------------------------------- collapsing
    def expand(self):
        """Expands the panel. Called when switched on by a toggle - otherwise
        only a strip with the title would appear and it would look as if
        nothing had happened."""
        if self._collapsed:
            self._collapsed = False
            self._apply_collapsed()

    def _on_toggle(self):
        self._collapsed = not self._collapsed
        self._apply_collapsed()

    def _apply_collapsed(self):
        self._body.setVisible(not self._collapsed)
        self._toggle.setText("+" if self._collapsed else "-")
        self.updateGeometry()          # the parent layout recomputes the panel
        self.adjustSize()
        self.resized.emit()

    # -------------------------------------------------------------- sliders
    def _clear_form(self):
        self._sliders.clear()
        self._labels.clear()
        self._ranges.clear()
        self._choices.clear()
        self._defaults.clear()
        self._bands = None
        self._drop_layout(self.form)

    def _drop_layout(self, layout):
        """Empties a layout and really THROWS the widgets away.

        CAREFUL with setParent(None): in Qt a widget without a parent is a
        top-level WINDOW. When the sliders of the previous QC mode were
        detached like that and Qt managed to draw them before Python cleaned
        them up, a floating window with the whole settings popped up over Nuke.
        So: hide() first, only then detach, and leave the cleanup to
        deleteLater().
        """
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.hide()
                w.setParent(None)
                w.deleteLater()
            elif item.layout() is not None:
                self._drop_layout(item.layout())
                item.layout().deleteLater()

    def _add_slider(self, key, label, lo, hi, value, decimals, curve=1.0,
                    default=None):
        # Widgets are created WITH A PARENT straight away. When they are
        # created without one and Qt manages to draw them before the layout
        # takes over, each of them is a separate window - see the note in
        # _drop_layout.
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(4)

        name = QtWidgets.QLabel(label, self)
        name.setFixedWidth(self._label_w)
        font = name.font()
        font.setPointSize(max(6, font.pointSize() - 1))
        name.setFont(font)
        row.addWidget(name)

        slider = _Slider(self)
        slider.setValue(self._to_slider(value, lo, hi, curve))
        slider.valueChanged.connect(lambda v, k=key: self._on_slider(k, v))
        slider.resetRequested.connect(lambda k=key: self._reset_one(k))
        row.addWidget(slider, 1)

        readout = _NumberEdit(self)
        readout.setText(self._format(value, decimals))
        readout.setFont(font)
        readout.editingFinished.connect(lambda k=key: self._on_typed(k))
        readout.stepped.connect(lambda d, k=key: self._on_stepped(k, d))
        readout.resetRequested.connect(lambda k=key: self._reset_one(k))
        row.addWidget(readout)

        self.form.addLayout(row)
        self._values[key] = value
        self._sliders[key] = slider
        self._labels[key] = (readout, decimals)
        self._ranges[key] = (lo, hi, decimals, curve)
        # what a middle click goes back to - the DEFAULT, which is not the same
        # as the value the panel happens to open with
        self._defaults[key] = value if default is None else default

    def set_choice(self, key, index):
        """A menu choice from outside. No signal is sent."""
        combo = self._choices.get(key)
        if combo is None:
            return
        index = max(0, min(combo.count() - 1, int(round(index))))
        self._values[key] = float(index)
        if combo.currentIndex() != index:
            combo.blockSignals(True)
            combo.setCurrentIndex(index)
            combo.blockSignals(False)

    def _add_choice(self, key, label, choices, value):
        """A menu instead of a slider - for parameters whose values are named
        (e.g. how the difference should be displayed), not a number on a scale."""
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(4)
        name = QtWidgets.QLabel(label, self)
        name.setFixedWidth(self._label_w)
        font = name.font()
        font.setPointSize(max(6, font.pointSize() - 1))
        name.setFont(font)
        row.addWidget(name)

        combo = _Combo(self)
        combo.addItems(list(choices))
        combo.setCurrentIndex(max(0, min(len(choices) - 1, int(round(value)))))
        combo.currentIndexChanged.connect(
            lambda *_a, k=key, c=combo: self._on_choice(k, c.currentIndex()))
        row.addWidget(combo, 1)

        self.form.addLayout(row)
        self._values[key] = float(combo.currentIndex())
        self._choices[key] = combo

    def _on_choice(self, key, index):
        self._values[key] = float(index)
        self._emit()

    def _add_bands(self, specs):
        """One multi-handle slider instead of a row of separate ones."""
        widget = _BandSlider(specs, self)
        widget.changed.connect(self._on_bands)
        self.form.addWidget(widget)
        self._bands = widget
        self._values.update(widget.values())

    def _on_bands(self, values):
        self._values.update(values)
        self._emit()

    def set_value(self, key, value):
        """A value from outside (from the node). No signal is sent."""
        if self._bands is not None and key in self._bands.keys():
            self._bands.set_value(key, value)
            self._values[key] = float(value)
            return
        rng = self._ranges.get(key)
        slider = self._sliders.get(key)
        if rng is None or slider is None:
            return
        lo, hi, decimals, curve = rng
        if abs(float(value) - float(self._values.get(key, 0.0))) < 10 ** -decimals:
            return
        self._values[key] = float(value)
        slider.blockSignals(True)
        slider.setValue(self._to_slider(value, lo, hi, curve))
        slider.blockSignals(False)
        readout, _dec = self._labels[key]
        readout.setText(self._format(value, decimals))

    STEPS = SLIDER_STEPS

    @classmethod
    def _to_slider(cls, value, lo, hi, curve=1.0):
        if hi <= lo:
            return 0
        t = (float(value) - lo) / (hi - lo)
        if curve != 1.0:
            t = max(0.0, t) ** (1.0 / curve)
        return int(round(t * cls.STEPS))

    @classmethod
    def _from_slider(cls, pos, lo, hi, curve=1.0):
        t = pos / float(cls.STEPS)
        if curve != 1.0:
            t = t ** curve
        return lo + (hi - lo) * t

    @staticmethod
    def _format(value, decimals):
        return ("%%.%df" % decimals) % value

    def _on_slider(self, key, pos):
        lo, hi, _dec, curve = self._ranges[key]
        value = self._from_slider(pos, lo, hi, curve)
        self._values[key] = value
        readout, decimals = self._labels[key]
        readout.setText(self._format(value, decimals))
        self._emit()

    def _apply_typed(self, key, value):
        """A value that came from the number field - clamped, then everything
        (slider, text, listeners) brought into line with it."""
        lo, hi, decimals, _curve = self._ranges[key]
        value = max(lo, min(hi, float(value)))
        self.set_value(key, value)          # moves the slider and the text
        self._emit()

    def _on_typed(self, key):
        """Enter pressed or the field left. Anything unreadable is ignored and
        the field goes back to showing the value that is actually set."""
        readout, decimals = self._labels[key]
        text = readout.text().strip().replace(",", ".")
        try:
            self._apply_typed(key, float(text))
        except ValueError:
            readout.setText(self._format(self._values.get(key, 0.0), decimals))

    def _on_stepped(self, key, delta):
        self._apply_typed(key, self._values.get(key, 0.0) + delta)

    def _reset_one(self, key):
        """Middle click - this one control back to its default."""
        if key in self._defaults:
            self.set_value(key, self._defaults[key])
            self._emit()

    def _emit(self):
        """Overridden by subclasses - each panel sends its own signal."""


class CCPanel(_Panel):
    """Gain / gamma / saturation.

    It does not collapse: on/off is handled by the CC toggle in the window bar.
    A collapsed panel would show only a strip with the title once switched on
    and it would look as if nothing had happened.
    """

    changed = QtCore.Signal(dict)

    def __init__(self, parent=None):
        super(CCPanel, self).__init__("CC", PANEL_W, CC_LABEL_W,
                                      parent=parent)
        for key, label, lo, hi, default, decimals, curve in CC_PARAMS:
            self._add_slider(key, label, lo, hi, default, decimals, curve)

        self._apply_collapsed()

    def _emit(self):
        self.changed.emit(self.values())


# An enabled PANEL = the muted yellow from the timeline (QColor(240, 200, 40)),
# but only as a translucent fill - a solid colour would glow in the image and
# pull the eye away from the plate. The letters stay grey so the button does
# not shout.
# The INPUT toggle has no colour at all: it is not an on/off switch, it only
# shows which input the window displays. The active window is told apart just
# by a lighter frame.
SLOT_STYLE = """
QToolButton#cvSlot, QToolButton#cvToggle {
    color: #d8d8d8; background: #2c2c2c;
    border: 1px solid #555; border-radius: 3px;
    font-weight: bold;
}
QToolButton#cvSlot:hover, QToolButton#cvToggle:hover { border-color: #999; }
QToolButton#cvSlot[cvActive="true"] {
    color: #ffffff; border-color: #999;    /* the active window just lighter, no
                                              colour - the input toggle is not
                                              a switch */
}
QToolButton#cvSlot::menu-indicator { image: none; }
QToolButton#cvToggle { color: #8a8a8a; background: #262626; }
QToolButton#cvToggle:checked {
    color: #d0d0d0; background: rgba(240, 200, 40, 60);
    border-color: rgba(240, 200, 40, 150);
}
QToolButton#cvToggle:disabled {
    color: #4a4a4a; background: #212121; border-color: #3a3a3a;
}
"""

# The scopes take a lot of room in the image, so in Double (where each window
# is only half the size) they are not available - the H and V toggles grey out.
SCOPE_KEYS = ("hist", "vscope", "wave")


# Panel toggles in the image: (key, label on the button, tooltip).
# The key is used for the node knobs too - cv_<key>_<window>.
PANEL_BUTTONS = (
    ("cc", "CC", "Colour: gain, gamma, saturation."),
    ("qc", "QC", "Check mode (grain, high-pass, saturation, value map...)."),
    ("hist", "H", "Histogram: axis 0 to 55, the line at 1.0 marks clipping."),
    ("vscope", "V", "Vectorscope: pixel colour as it is on screen."),
    ("wave", "W", "Waveform: values along the columns, line at 1.0, top 55."),
)


class OverlayPanel(_Panel):
    """Everything Overlay mode needs, in ONE panel.

    A and B are two rows of the same panel rather than a bar apiece: in Overlay
    the windows are on top of each other, so a bar sitting in "its" window
    would sit on the other one just as much. There is no input picker on the
    rows either - in this mode window A shows input A and window B shows input
    B, that is what the mode is - so only the layer is left to choose, which is
    the one thing that genuinely differs (rgba against depth of the same plate).

    Below them the dissolve, then the comparison and whatever that comparison
    is set up with. The panel toggles are gone: CC is off in Overlay (see
    PlayerPanel._apply_panel_flags) and the check is right here.
    """

    LABEL_W = 74

    sourceLayerChanged = QtCore.Signal(int, str)   # window index, layer
    mixChanged = QtCore.Signal(float)
    modeChanged = QtCore.Signal(str)               # one of fx.OVERLAY_MODES
    paramsChanged = QtCore.Signal(dict)

    MIX_KEY = "mix"

    def __init__(self, labels=("A", "B"), parent=None):
        super(OverlayPanel, self).__init__("Difference", PANEL_W, self.LABEL_W,
                                           parent=parent)
        self._mode = fx.NONE
        self.layers = []

        # --- the two inputs, one row each -------------------------------
        for index, label in enumerate(labels):
            row = QtWidgets.QHBoxLayout()
            row.setSpacing(4)
            # The same boxed tile the window bars use, so A and B read as the
            # inputs they are. Not clickable here: in this mode window A IS
            # input A, so there is nothing to choose.
            name = QtWidgets.QLabel(label, self)
            name.setObjectName("cvSlot")
            name.setStyleSheet(SLOT_STYLE)
            name.setFixedSize(24, 20)
            name.setAlignment(QtCore.Qt.AlignCenter)
            row.addWidget(name)
            combo = _Combo(self)
            combo.addItem("rgba")
            combo.setToolTip("The EXR layer shown for input %s." % label)
            combo.currentIndexChanged.connect(
                lambda *_a, i=index, c=combo:
                self.sourceLayerChanged.emit(i, c.currentText()))
            row.addWidget(combo, 1)
            self.form.addLayout(row)
            self.layers.append(combo)

        # --- the dissolve ----------------------------------------------
        # Built by hand rather than through _add_slider: _clear_form wipes the
        # slider bookkeeping every time the comparison changes, and the mix has
        # to survive that.
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(4)
        name = QtWidgets.QLabel("Mix", self)
        name.setFixedWidth(self.LABEL_W)
        small = name.font()
        small.setPointSize(max(6, small.pointSize() - 1))
        name.setFont(small)
        row.addWidget(name)
        self.mix = _Slider(self)
        self.mix.setValue(SLIDER_STEPS)
        self.mix.valueChanged.connect(self._on_mix)
        self.mix.resetRequested.connect(lambda: self._set_mix(1.0, True))
        row.addWidget(self.mix, 1)
        self.mix_num = _NumberEdit(self)
        self.mix_num.setFont(small)
        self.mix_num.setText("1.00")
        self.mix_num.editingFinished.connect(self._on_mix_typed)
        self.mix_num.stepped.connect(
            lambda d: self._set_mix(self.mix_value() + d, True))
        self.mix_num.resetRequested.connect(lambda: self._set_mix(1.0, True))
        row.addWidget(self.mix_num)
        self.form.addLayout(row)
        self.setToolTip("How much of input B is mixed over A.\n"
                        "0 = only A, 1 = only B. With a comparison on, this is\n"
                        "the opacity of its result over A.")

        # --- the comparison ---------------------------------------------
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(4)
        name = QtWidgets.QLabel("Compare", self)
        name.setFixedWidth(self.LABEL_W)
        name.setFont(small)
        row.addWidget(name)
        self.qc = _Combo(self)
        self.qc.addItems([fx.OVERLAY_LABELS[m] for m in fx.OVERLAY_MODES])
        self.qc.currentIndexChanged.connect(self._on_mode)
        row.addWidget(self.qc, 1)
        self.form.addLayout(row)

        # --- the settings of whichever comparison is chosen -------------
        self._params_host = QtWidgets.QWidget(self)
        self.form.addWidget(self._params_host)
        host = QtWidgets.QVBoxLayout(self._params_host)
        host.setContentsMargins(0, 2, 0, 0)
        host.setSpacing(2)
        self.form = host              # from here on the sliders go in the host
        # Every comparison gets the SAME room, measured once from the tallest
        # of them. Difference is two menus and two sliders, the high-pass one
        # is four sliders, and menus are taller - so without this the panel
        # changed height every time the comparison was switched and everything
        # under the cursor moved.
        self._params_h = 0
        for mode in fx.OVERLAY_MODES:
            self._mode = mode
            self._build_params(measure=True)
            host.activate()               # or sizeHint is the one from before
            self._params_h = max(self._params_h,
                                 self._params_host.sizeHint().height())
        self._mode = fx.NONE
        self._build_params()

    # ---- inputs --------------------------------------------------------
    def set_layers(self, index, layers, current=None):
        """Fills one row's layer menu. No signal is sent."""
        if not 0 <= index < len(self.layers):
            return
        combo = self.layers[index]
        layers = list(layers) or ["rgba"]
        have = [combo.itemText(i) for i in range(combo.count())]
        combo.blockSignals(True)
        if have != layers:
            combo.clear()
            combo.addItems(layers)
        if current in layers:
            combo.setCurrentIndex(layers.index(current))
        combo.blockSignals(False)
        combo.setEnabled(len(layers) > 1)

    # ---- dissolve ------------------------------------------------------
    def mix_value(self):
        return self.mix.value() / float(SLIDER_STEPS)

    def set_mix(self, value):
        """From the node. No signal is sent."""
        self._set_mix(value, False)

    def _set_mix(self, value, emit):
        value = max(0.0, min(1.0, float(value)))
        self.mix.blockSignals(True)
        self.mix.setValue(int(round(value * SLIDER_STEPS)))
        self.mix.blockSignals(False)
        self.mix_num.setText("%.2f" % value)
        if emit:
            self.mixChanged.emit(value)

    def _on_mix(self, pos):
        self.mix_num.setText("%.2f" % (pos / float(SLIDER_STEPS)))
        self.mixChanged.emit(pos / float(SLIDER_STEPS))

    def _on_mix_typed(self):
        try:
            self._set_mix(float(self.mix_num.text().strip().replace(",", ".")),
                          True)
        except ValueError:
            self.mix_num.setText("%.2f" % self.mix_value())

    # ---- comparison ----------------------------------------------------
    def mode(self):
        return self._mode

    def set_mode(self, mode, values=None):
        """From the node. No signal is sent."""
        if mode not in fx.OVERLAY_MODES:
            mode = fx.NONE
        index = fx.OVERLAY_MODES.index(mode)
        if self.qc.currentIndex() != index:
            self.qc.blockSignals(True)
            self.qc.setCurrentIndex(index)
            self.qc.blockSignals(False)
        self._mode = mode
        self._build_params(values)

    def _on_mode(self, index):
        self._mode = fx.OVERLAY_MODES[
            max(0, min(len(fx.OVERLAY_MODES) - 1, index))]
        self._build_params()
        self.modeChanged.emit(self._mode)

    def _build_params(self, values=None, measure=False):
        """The sliders of the chosen comparison. Off has none, so the block
        disappears entirely rather than leaving a gap."""
        self._clear_form()
        self._values = dict(fx.defaults(self._mode))
        if values:
            self._values.update({k: v for k, v in values.items()
                                 if k in self._values})
        specs = fx.PARAMS.get(self._mode, [])
        for spec in specs:
            key, label, lo, hi, default, decimals = spec[:6]
            value = self._values.get(key, default)
            if len(spec) > 6:
                self._add_choice(key, label, spec[6], value)
            else:
                self._add_slider(key, label, lo, hi, value, decimals,
                                 default=default)
        self.qc.setToolTip(fx.DESCRIPTION.get(self._mode, "")
                           or "Compares input A against input B.")
        if measure:
            self._params_host.setVisible(True)
            return
        # A FIXED height, not a minimum: with a minimum the block only ever
        # grew, so turning the comparison off left the panel as tall as the
        # comparison had made it. Every comparison gets the same reserved
        # height (measured in __init__) so switching between them moves
        # nothing; Off gets none at all and the panel really does shrink.
        specs_h = getattr(self, "_params_h", 0) if specs else 0
        self._params_host.setVisible(bool(specs))
        self._params_host.setFixedHeight(specs_h)
        self.updateGeometry()
        self.adjustSize()
        self.resize(self.sizeHint())       # adjustSize alone would not shrink
        self.resized.emit()

    def values(self):
        return dict(self._values)

    def _emit(self):
        self.paramsChanged.emit(self.values())

    # ---- placing -------------------------------------------------------
    def set_anchor(self, x, y):
        anchor = (int(x), int(y))
        if anchor == getattr(self, "_anchor", None):
            return
        self._anchor = anchor
        self.move(*anchor)
        self.raise_()

    def bottom(self):
        return self.y() + self.height()


class ExportScopes(object):
    """The scopes, drawn into an exported picture instead of onto the screen.

    Its own set of canvases, built once and never shown: the ones in the window
    are measuring the frame you are looking at, while an export walks over many
    frames, and borrowing them would both fight over the same widgets and put
    the wrong numbers in the file.

    Kept here rather than in annotate.py because these are the very canvases
    the panels use - the same drawing code, so a scope in a JPEG looks like the
    one that was on screen.
    """

    WIDTH_PART = 0.24        # share of the picture width the column takes
    MIN_W, MAX_W = 220, 560
    GAP = 10

    def __init__(self):
        self._hist = HistogramCanvas()
        self._vscope = VectorscopeCanvas()
        self._wave = WaveformCanvas()
        for c in (self._hist, self._vscope, self._wave):
            c.setAttribute(QtCore.Qt.WA_DontShowOnScreen, True)
            c.opacity = 1.0

    def draw(self, painter, ctx, width, height):
        """Histogram, vectorscope and waveform down the RIGHT edge, inside the
        format - the picture keeps its size, the scopes sit on top of it."""
        w = int(max(self.MIN_W, min(self.MAX_W, width * self.WIDTH_PART)))
        if w * 2 > width:                       # a tiny plate has no room
            return
        channel = scopes.channel_key(ctx.get("channels", 0))
        qc = bool(ctx.get("qc"))
        display, linear = ctx.get("display"), ctx.get("linear")

        gain, sat = ctx.get("gain", 1.0), ctx.get("sat_matrix")
        gamma, lz = ctx.get("gamma", 1.0), ctx.get("linearize")
        # Exactly the split ScopeStack.update_scopes uses: with a check on
        # there is no scene-linear equivalent of what is on screen, so the
        # histogram and the waveform measure the finished image; the
        # vectorscope always does.
        if qc:
            self._hist.set_data(scopes.histogram_display(display, channel))
            self._wave.set_data(scopes.waveform(display, channel),
                                scopes.WF_AXIS_DISPLAY)
        else:
            self._hist.set_data(scopes.histogram(linear, channel, gain, sat,
                                                 linearize=lz, gamma=gamma))
            self._wave.set_data(
                scopes.waveform_linear(linear, channel, gain, sat,
                                       linearize=lz, gamma=gamma),
                scopes.WF_AXIS_LINEAR)
        self._vscope.set_data(scopes.vectorscope(display, channel))

        # All three SQUARE, at the column's width. On screen the histogram and
        # the waveform are wide and short because they share a narrow panel
        # with everything else; in an export there is room, and a column of
        # three equal tiles reads as one block instead of two slivers beside a
        # square. It also gives the waveform far more vertical room, which is
        # the axis its levels are actually read on.
        # Three squares have to FIT, so the column is also capped by the height.
        # Without this a 2K plate silently lost the waveform off the bottom -
        # and a scope that is quietly missing is worse than a smaller one.
        w = int(min(w, (height - self.GAP * 4) / 3.0))
        if w < 80:
            return                              # no room worth drawing in
        x = width - w - self.GAP
        y = self.GAP
        for canvas in (self._hist, self._vscope, self._wave):
            canvas.resize(w, w)
            painter.save()
            painter.translate(x, y)
            canvas.render(painter, QtCore.QPoint(0, 0),
                          QtGui.QRegion(),
                          QtWidgets.QWidget.RenderFlags(
                              QtWidgets.QWidget.DrawChildren))
            painter.restore()
            y += w + self.GAP


class _Swatches(QtWidgets.QWidget):
    """The annotation colours as a row of buttons.

    One widget for both places that offer them - the tool settings, where it
    sets what the NEXT mark will be, and the note box, where it sets that one
    note. Painted rather than styled by the theme: the button IS the colour.
    """

    colorChanged = QtCore.Signal(int)         # index into annotate.COLORS

    def __init__(self, parent=None, size=16):
        super(_Swatches, self).__init__(parent)
        self._color = 0
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)
        self._buttons = []
        for i, rgb in enumerate(annotate.COLORS):
            b = QtWidgets.QToolButton(self)
            b.setCheckable(True)
            b.setFixedSize(size, size)
            b.setFocusPolicy(QtCore.Qt.NoFocus)
            b.setToolTip(annotate.COLOR_NAMES[i])
            b.setStyleSheet(
                "QToolButton { background: rgb(%d,%d,%d); border: 1px solid "
                "#20242c; border-radius: 2px; }"
                "QToolButton:checked { border: 2px solid #f0f0f0; }" % rgb)
            b.clicked.connect(lambda *_a, k=i: self._pick(k))
            lay.addWidget(b)
            self._buttons.append(b)
        self.set_color(0)

    def color(self):
        return self._color

    def set_color(self, index):
        """From outside. No signal is sent."""
        self._color = int(index) % len(annotate.COLORS)
        for i, b in enumerate(self._buttons):
            b.blockSignals(True)
            b.setChecked(i == self._color)
            b.blockSignals(False)

    def _pick(self, index):
        self.set_color(index)
        self.colorChanged.emit(self._color)


class NoteDialog(QtWidgets.QDialog):
    """Where an annotation note is written.

    A proper box with several lines rather than the one-line prompt this used
    to be: a review note is a sentence about what is wrong, not a word, and it
    was being typed into a field narrower than the sentence.

    Resizable by the grip in the bottom right corner, and the size is REMEMBERED
    for the rest of the session - having to stretch the same box open for every
    note on a shot is exactly the kind of thing that stops people writing them.
    """

    _last_size = None            # shared by every note in this session

    def __init__(self, frame, parent=None, text="", per_line=0, edit=False,
                 color=0):
        super(NoteDialog, self).__init__(parent)
        self.setWindowTitle("%s note - frame %d"
                            % ("Edit" if edit else "Annotation", int(frame)))
        self.setSizeGripEnabled(True)          # the corner grip
        self.setMinimumSize(320, 160)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        self._per_line = per_line or annotate.LINE_MAX

        # QTextEdit and not QPlainTextEdit: only this one can be told to break
        # its lines at a COLUMN, which is what makes the box show the same 50
        # characters a line that the picture will get. Rich text is turned off,
        # so a paste from a browser does not arrive with its formatting.
        self.edit = QtWidgets.QTextEdit(self)
        self.edit.setAcceptRichText(False)
        self.edit.setPlainText((text or "")[:annotate.MAX_CHARS])
        self.edit.setPlaceholderText("What is wrong with this frame?")
        self.edit.setTabChangesFocus(True)     # tab leaves the box, as expected
        wrap = QtWidgets.QTextEdit.LineWrapMode
        self.edit.setLineWrapMode(getattr(wrap, "FixedColumnWidth", wrap))
        self.edit.setLineWrapColumnOrWidth(self._per_line)
        self.edit.textChanged.connect(self._on_changed)
        lay.addWidget(self.edit, 1)

        # Where the line breaks fall and how much room is left: typing to a
        # width you cannot see is the thing that had people lining notes up by
        # hand. In edit mode the box also says how to get rid of the note,
        # which is otherwise unguessable.
        self._hint = QtWidgets.QLabel("", self)
        font = self._hint.font()
        font.setPointSize(max(6, font.pointSize() - 1))
        self._hint.setFont(font)
        self._hint.setWordWrap(True)
        self._edit_mode = edit
        lay.addWidget(self._hint)

        # The colour lives HERE and not only on the tool panel, because that
        # one sets what the next mark will be - it cannot reach a note already
        # written. This is the only way to recolour one.
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(QtWidgets.QLabel("Colour", self))
        self._swatches = _Swatches(self, 18)
        self._swatches.set_color(color)
        row.addWidget(self._swatches)
        row.addStretch(1)
        lay.addLayout(row)
        self._on_changed()

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

        self.resize(NoteDialog._last_size or QtCore.QSize(460, 240))
        self.edit.setFocus()

    def _on_changed(self):
        """Holds the note to MAX_CHARS and says how much is left.

        Truncating as it is typed rather than on OK: losing the tail of a note
        at the moment it is accepted, with the box already shut, is the worst
        possible time to find out about the limit.
        """
        text = self.edit.toPlainText()
        if len(text) > annotate.MAX_CHARS:
            cursor = self.edit.textCursor()
            at = cursor.position()
            self.edit.blockSignals(True)
            self.edit.setPlainText(text[:annotate.MAX_CHARS])
            cursor.setPosition(min(at, annotate.MAX_CHARS))
            self.edit.setTextCursor(cursor)
            self.edit.blockSignals(False)
            text = text[:annotate.MAX_CHARS]
        note = "Wrapped at %d characters.  %d / %d used." % (
            self._per_line, len(text), annotate.MAX_CHARS)
        if self._edit_mode:
            note += "  Empty the box to delete it."
        self._hint.setText(note)

    def text(self):
        return self.edit.toPlainText().strip()

    def color(self):
        return self._swatches.color()

    def done(self, result):
        NoteDialog._last_size = self.size()
        super(NoteDialog, self).done(result)

    @classmethod
    def ask(cls, frame, parent=None, text="", per_line=0, edit=False, color=0):
        """(text, colour, accepted)."""
        dlg = cls(frame, parent, text, per_line, edit, color)
        ok = dlg.exec_() if hasattr(dlg, "exec_") else dlg.exec()
        return dlg.text(), dlg.color(), bool(ok)


class AnnotBar(QtWidgets.QFrame):
    """Pencil, text and export - the tools of Annotation mode.

    A strip of its own under the window controls rather than more buttons on
    them: the notes belong to the SHOT, not to a window, and in this mode there
    is only one window anyway.
    """

    toolChanged = QtCore.Signal(str)      # "" | "draw" | "text"
    exportWanted = QtCore.Signal()
    undoWanted = QtCore.Signal()
    clearWanted = QtCore.Signal()

    def __init__(self, parent=None):
        super(AnnotBar, self).__init__(parent)
        self.setObjectName("cvOverlay")
        self.setStyleSheet(STYLE + SLOT_STYLE)
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self._anchor = (0, 0)
        self._tool = ""

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(4)
        lay.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)

        self._tools = {}
        for key, glyph, tip in (
                ("draw", "✎", "Pencil - drag on the image to draw.\n"
                                   "The frame is then marked blue in the "
                                   "timeline.\nThe MIDDLE button still pans."),
                ("text", "T", "Text - click where the note belongs and type "
                              "it.\nClick a note you already wrote to change "
                              "it.\nThe frame is marked blue too.")):
            b = QtWidgets.QToolButton(self)
            b.setObjectName("cvToggle")
            b.setText(glyph)
            b.setCheckable(True)
            b.setFixedSize(24, 20)
            b.setFocusPolicy(QtCore.Qt.NoFocus)
            b.setToolTip(tip)
            b.clicked.connect(lambda *_a, k=key: self._pick(k))
            lay.addWidget(b)
            self._tools[key] = b

        for text, tip, sig in (
                ("Export", "Writes every annotated frame as a JPEG into the\n"
                           "folder set on the node. Frames without a note are\n"
                           "not written.", self.exportWanted),
                ("Undo", "Takes back the last note on this frame.",
                 self.undoWanted),
                ("Clear", "Removes every note on this frame.",
                 self.clearWanted)):
            b = QtWidgets.QToolButton(self)
            b.setObjectName("cvToggle")
            b.setText(text)
            b.setFixedHeight(20)
            b.setFocusPolicy(QtCore.Qt.NoFocus)
            b.setToolTip(tip)
            b.clicked.connect(lambda *_a, s=sig: s.emit())
            lay.addWidget(b)

    def tool(self):
        return self._tool

    def set_tool(self, tool):
        """From outside. No signal is sent."""
        tool = tool if tool in self._tools else ""
        self._tool = tool
        for key, b in self._tools.items():
            b.blockSignals(True)
            b.setChecked(key == tool)
            b.blockSignals(False)

    def _pick(self, key):
        # clicking the armed tool again puts the pencil down
        self.set_tool("" if self._tool == key else key)
        self.toolChanged.emit(self._tool)

    def set_anchor(self, x, y):
        anchor = (int(x), int(y))
        if anchor == self._anchor:
            return
        self._anchor = anchor
        self.move(*anchor)
        self.raise_()

    def bottom(self):
        return self.y() + self.height()


class AnnotOptions(QtWidgets.QFrame):
    """What the armed tool draws with - colour and size.

    Its own little panel UNDER the tool strip rather than a second row inside
    it: the strip would change height every time a tool was picked up or put
    down, and everything anchored below it moved with it. Two panels each keep
    their own size, so nothing shifts.
    """

    colorChanged = QtCore.Signal(int)         # index into annotate.COLORS
    sizeChanged = QtCore.Signal(str, float)   # tool, image pixels

    # What the size slider spans for each tool, and where a middle click puts
    # it back to. A pen and a caption are not the same order of size, so one
    # shared range would leave the pen squeezed into the first tenth of it.
    RANGES = {"draw": (0.5, 40.0, annotate.LINE_W),
              "text": (6.0, 200.0, annotate.TEXT_H)}

    def __init__(self, parent=None):
        super(AnnotOptions, self).__init__(parent)
        self.setObjectName("cvOverlay")
        self.setStyleSheet(STYLE + SLOT_STYLE)
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self._anchor = (0, 0)
        self._tool = ""
        self._sizes = {"draw": annotate.LINE_W, "text": annotate.TEXT_H}

        opt = QtWidgets.QHBoxLayout(self)
        opt.setContentsMargins(6, 4, 6, 4)
        opt.setSpacing(3)
        opt.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)

        self._swatches = _Swatches(self)
        self._swatches.colorChanged.connect(self.colorChanged)
        opt.addWidget(self._swatches)

        opt.addSpacing(6)
        self._size_label = QtWidgets.QLabel("Width", self)
        opt.addWidget(self._size_label)

        self._size = _Slider(self)
        self._size.setFixedWidth(90)
        self._size.valueChanged.connect(self._on_size_slider)
        self._size.resetRequested.connect(self._reset_size)
        opt.addWidget(self._size)

        self._size_edit = _NumberEdit(self)
        self._size_edit.editingFinished.connect(self._on_size_typed)
        self._size_edit.stepped.connect(self._step_size)
        self._size_edit.resetRequested.connect(self._reset_size)
        opt.addWidget(self._size_edit)

    def set_tool(self, tool):
        """Which tool the panel is showing the settings of. No signal is sent."""
        self._tool = tool if tool in self.RANGES else ""
        if self._tool:
            self._size_label.setText(
                "Width" if self._tool == "draw" else "Size")
            self._show_size(self._sizes[self._tool])

    # ---- colour ----
    def color(self):
        return self._swatches.color()

    def set_color(self, index):
        """From outside. No signal is sent."""
        self._swatches.set_color(index)

    # ---- size ----
    def set_size(self, tool, value):
        """From outside. No signal is sent."""
        if tool not in self.RANGES:
            return
        lo, hi, _d = self.RANGES[tool]
        self._sizes[tool] = max(lo, min(hi, float(value)))
        if tool == self._tool:
            self._show_size(self._sizes[tool])

    def _show_size(self, value):
        """Both readouts, without either one answering back."""
        lo, hi, _d = self.RANGES[self._tool]
        for w in (self._size, self._size_edit):
            w.blockSignals(True)
        self._size.setValue(
            int(round((value - lo) / (hi - lo) * SLIDER_STEPS)))
        # a pen is half-pixel work, a caption is not - one decimal either way
        self._size_edit.setText(("%.1f" % value).rstrip("0").rstrip("."))
        for w in (self._size, self._size_edit):
            w.blockSignals(False)

    def _commit_size(self, value):
        if not self._tool:
            return
        lo, hi, _d = self.RANGES[self._tool]
        value = max(lo, min(hi, float(value)))
        self._sizes[self._tool] = value
        self._show_size(value)
        self.sizeChanged.emit(self._tool, value)

    def _on_size_slider(self, steps):
        if not self._tool:
            return
        lo, hi, _d = self.RANGES[self._tool]
        self._commit_size(lo + (hi - lo) * steps / float(SLIDER_STEPS))

    def _on_size_typed(self):
        try:
            self._commit_size(float(self._size_edit.text().replace(",", ".")))
        except ValueError:
            self._show_size(self._sizes.get(self._tool, 1.0))   # put it back

    def _step_size(self, delta):
        if self._tool:
            self._commit_size(self._sizes[self._tool] + delta)

    def _reset_size(self):
        if self._tool:
            self._commit_size(self.RANGES[self._tool][2])

    def set_anchor(self, x, y):
        anchor = (int(x), int(y))
        if anchor == self._anchor:
            return
        self._anchor = anchor
        self.move(*anchor)
        self.raise_()

    def bottom(self):
        return self.y() + self.height()


class SlotBar(QtWidgets.QFrame):
    """The controls of one window, sitting RIGHT INSIDE IT (top left).

    The toggle shows WHICH node input that window displays and, when clicked,
    offers A or B. The layer sits next to it. Each window has its own - so both
    can show the same input and differ only by layer (rgba against depth of the
    same plate).

    Why in the image and not in the top bar: in Double each set of controls
    belongs to its own half, and with a single shared bar at the top there is
    no telling which pair is which. This way it is right by the window you are
    looking at.
    """

    sourceChanged = QtCore.Signal(int)         # 0 = A, 1 = B
    layerChanged = QtCore.Signal(str)
    panelToggled = QtCore.Signal(str, bool)    # panel key, on/off

    picked = QtCore.Signal()                   # the user touched the window
    moved = QtCore.Signal()                    # position or height changed

    def __init__(self, labels=("A", "B"), parent=None):
        super(SlotBar, self).__init__(parent)
        self.setObjectName("cvOverlay")
        self.setStyleSheet(STYLE + SLOT_STYLE)
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self._labels = list(labels)

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(4)
        lay.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)

        self.button = QtWidgets.QToolButton(self)
        self.button.setObjectName("cvSlot")
        self.button.setFixedSize(24, 20)
        self.button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        self.button.setFocusPolicy(QtCore.Qt.NoFocus)
        menu = QtWidgets.QMenu(self.button)
        for i, name in enumerate(self._labels):
            act = menu.addAction("Input %s" % name)
            act.triggered.connect(lambda _c=False, idx=i: self._pick(idx))
        self.button.setMenu(menu)
        lay.addWidget(self.button)

        self.layer = _Combo(self)
        self.layer.addItem("rgba")
        self.layer.setMinimumWidth(96)
        self.layer.setToolTip(
            "The EXR layer (rgba, depth, normal...) for this window.\n"
            "In a multipart file the AOVs from the other parts are here too.")
        self.layer.currentIndexChanged.connect(
            lambda *_a: self.layerChanged.emit(self.layer.currentText()))
        lay.addWidget(self.layer)

        # Panel toggles. The panels are shared across the whole image, so in
        # Double both bars show the same thing - you switch from the one you
        # happen to be at and the other follows.
        self._toggles, self._tips = {}, {}
        for key, label, tip in PANEL_BUTTONS:
            b = QtWidgets.QToolButton(self)
            b.setObjectName("cvToggle")
            b.setText(label)
            b.setCheckable(True)
            b.setFixedSize(24, 20)
            b.setFocusPolicy(QtCore.Qt.NoFocus)
            self._tips[key] = (
                "%s\n\nA click switches both at once - the computation and the\n"
                "panel. On the node (Viewer tab) the two can be split: e.g.\n"
                "compute a scope but not show the panel." % tip)
            b.setToolTip(self._tips[key])
            # CAREFUL: the state is read from the BUTTON, not from the signal
            # argument. In Qt, clicked has an argument with a default value
            # (checked=false) and PySide6 then calls the lambda WITHOUT it -
            # "missing 1 required positional argument". The button toggled but
            # it never got anywhere.
            b.clicked.connect(
                lambda *_a, k=key, btn=b: self.panelToggled.emit(k, btn.isChecked()))
            lay.addWidget(b)
            self._toggles[key] = b

        self._source = 0
        self._anchor = (EDGE, EDGE)
        self.set_source(0)
        self._reposition()

    # ---- anchoring -----------------------------------------------------
    # The position is decided by the panel (see _Stage): in Single and Double
    # the bar sits in the corner of its window, in Wipe the windows overlap and
    # the bar of input B stands BELOW the controls of input A.
    def set_anchor(self, x, y):
        if (x, y) == self._anchor:
            return
        self._anchor = (int(x), int(y))
        self._reposition()

    def bottom(self):
        """The bottom edge - from the REAL position, not from the anchor.

        The stack below computes its own place from it; taking just `_anchor`
        would give the value from before the last move and the blocks would
        overlap.
        """
        return self.y() + self.height()

    def resizeEvent(self, event):
        super(SlotBar, self).resizeEvent(event)
        self._reposition()                     # a layer change changes the width
        self.moved.emit()

    def _reposition(self):
        self.move(*self._anchor)
        self.raise_()
        # A MOVE has to be announced the same as a height change - the stack
        # below computes where it belongs from our bottom edge
        self.moved.emit()

    def _pick(self, index):
        if index != self._source:
            self.set_source(index)
            self.sourceChanged.emit(index)
        self.picked.emit()

    # ---- state from outside (the node is the source of truth, the panel
    #      only synchronises) ----------------------------------------------
    def set_source(self, index):
        self._source = max(0, min(len(self._labels) - 1, int(index)))
        self.button.setText(self._labels[self._source])
        self.button.setToolTip(
            "The window shows input %s. Click to pick another one.\n"
            "Both windows may show the same input - then it is handy to give\n"
            "each of them a different layer." % self._labels[self._source])

    def source(self):
        return self._source

    def set_active(self, active):
        """Highlights the window the scopes and the pixel readout read from."""
        if self.button.property("cvActive") == bool(active):
            return
        self.button.setProperty("cvActive", bool(active))
        # a property change only reaches the look after the style is recomputed
        self.button.style().unpolish(self.button)
        self.button.style().polish(self.button)

    def set_layers(self, layers, current=None):
        """Fills the layer menu. No signal is sent."""
        layers = list(layers) or ["rgba"]
        have = [self.layer.itemText(i) for i in range(self.layer.count())]
        self.layer.blockSignals(True)
        if have != layers:
            self.layer.clear()
            self.layer.addItems(layers)
        if current in layers:
            self.layer.setCurrentIndex(layers.index(current))
        self.layer.blockSignals(False)
        self.layer.setEnabled(len(layers) > 1)

    def current_layer(self):
        return self.layer.currentText()

    def set_panel(self, key, on):
        """The toggle state from the node. No signal is sent."""
        b = self._toggles.get(key)
        if b is not None and b.isChecked() != bool(on):
            b.blockSignals(True)
            b.setChecked(bool(on))
            b.blockSignals(False)

    def set_scopes_available(self, available):
        """In Double the scopes are unavailable - the H and V toggles grey out."""
        for key in SCOPE_KEYS:
            b = self._toggles.get(key)
            if b is None or b.isEnabled() == bool(available):
                continue
            b.setEnabled(bool(available))
            b.setToolTip(self._tips[key] if available else
                         "The scopes are Single only - in Double they would\n"
                         "leave almost nothing of two half-size windows.")


MATTE_CHANNELS = ("r", "g", "b", "a")
MATTE_LABELS = ("R", "G", "B", "A")


class MattePanel(_Panel):
    """DiMatte: which matte channels are drawn over the image.

    The toggles look the same as the panel switches - it is the same gesture,
    only instead of a panel it switches on a matte.
    """

    toggled = QtCore.Signal(str, bool)      # channel, on/off
    changed = QtCore.Signal(dict)           # matte lightness, gain and gamma
    layerChanged = QtCore.Signal(str)       # which layer carries the mattes

    # sliders below the toggles: (key, label, min, max), the default is always 1.00
    SHAPE = (("light", "Lightness", 0.0, 1.0),
             ("gain", "Gain", 0.0, 8.0),
             ("gamma", "Gamma", 0.1, 4.0))

    def __init__(self, parent=None):
        super(MattePanel, self).__init__("DiMatte", PANEL_W, CC_LABEL_W,
                                         parent=parent)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(4)
        self._toggles = {}
        for ch, label in zip(MATTE_CHANNELS, MATTE_LABELS):
            b = QtWidgets.QToolButton(self)
            b.setObjectName("cvToggle")
            b.setStyleSheet(SLOT_STYLE)
            b.setText(label)
            b.setCheckable(True)
            b.setFixedSize(24, 20)
            b.setFocusPolicy(QtCore.Qt.NoFocus)
            b.setToolTip("Overlay the image with the matte from channel %s of "
                         "the DiMatte input." % label)
            # the state from the button, not from the signal argument - see SlotBar
            b.clicked.connect(
                lambda *_a, k=ch, btn=b: self.toggled.emit(k, btn.isChecked()))
            row.addWidget(b)
            self._toggles[ch] = b

        row.addStretch(1)
        self.form.addLayout(row)

        # WHICH layer of the file carries the mattes. On the node as well, but
        # picking it there means leaving the picture to go and find the node -
        # and hunting for the right cryptomatte or ID pass is exactly the thing
        # you do while looking at the image.
        lrow = QtWidgets.QHBoxLayout()
        lrow.setSpacing(4)
        name = QtWidgets.QLabel("Layer", self)
        name.setFixedWidth(CC_LABEL_W)
        font = name.font()
        font.setPointSize(max(6, font.pointSize() - 1))
        name.setFont(font)
        lrow.addWidget(name)
        self.layer = _Combo(self)
        self.layer.setToolTip("The layer the mattes are read from.\n"
                              "The list comes from the file that is\n"
                              "actually attached.")
        self.layer.currentIndexChanged.connect(
            lambda *_a: self.layerChanged.emit(self.layer.currentText()))
        lrow.addWidget(self.layer, 1)
        self.form.addLayout(lrow)

        for key, label, lo, hi in self.SHAPE:
            self._add_slider(key, label, lo, hi, 1.0, 2)
        self._apply_collapsed()

    def set_layers(self, layers, current=None):
        """Fills the layer menu. No signal is sent."""
        layers = list(layers) or ["rgba"]
        have = [self.layer.itemText(i) for i in range(self.layer.count())]
        self.layer.blockSignals(True)
        if have != layers:
            self.layer.clear()
            self.layer.addItems(layers)
        if current in layers:
            self.layer.setCurrentIndex(layers.index(current))
        self.layer.blockSignals(False)

    def _emit(self):
        self.changed.emit(self.values())

    def set_channel(self, channel, on):
        """The state from the node. No signal is sent."""
        b = self._toggles.get(channel)
        if b is not None and b.isChecked() != bool(on):
            b.blockSignals(True)
            b.setChecked(bool(on))
            b.blockSignals(False)


class EffectPanel(_Panel):
    """The QC mode selector plus the sliders of whichever mode is active."""

    changed = QtCore.Signal(dict)      # a parameter of the active mode changed

    def __init__(self, parent=None):
        super(EffectPanel, self).__init__("QC", PANEL_W, FX_LABEL_W,
                                          parent=parent)
        self._effect = fx.NONE

        self.combo = _Combo(self)
        self.combo.addItems([fx.LABELS[e] for e in fx.ORDER])
        # the mode description lives in the tooltip, so it takes no room in the image
        self.form.addWidget(self.combo)

        self._params_host = QtWidgets.QWidget(self)   # a parent right away, see _drop_layout
        self.form.addWidget(self._params_host)
        host_layout = QtWidgets.QVBoxLayout(self._params_host)
        host_layout.setContentsMargins(0, 2, 0, 0)
        host_layout.setSpacing(2)
        # the sliders go into the host, not straight into form (the combo is there)
        self.form = host_layout
        self.set_effect(fx.ORDER[0])

    def set_effect(self, effect, values=None):
        """Rebuilds the sliders for the mode. `values` = previously set values."""
        self._effect = effect
        self._clear_form()
        self._values = dict(fx.defaults(effect))
        if values:
            self._values.update({k: v for k, v in values.items()
                                 if k in self._values})
        self.combo.setToolTip(
            fx.DESCRIPTION.get(effect, "")
            or "Check display (keys 1-%d).\n"
               "Switched off by the QC toggle, not by an item in the list."
               % len(fx.ORDER))
        # parameters tagged fx.BANDS go into ONE multi-handle slider
        specs = fx.PARAMS.get(effect, [])
        bands = [s for s in specs if len(s) > 6 and s[6] == fx.BANDS]
        if bands:
            self._add_bands([(s[0], s[1], s[2], s[3],
                              self._values.get(s[0], s[4]), s[5])
                             for s in bands])
        for spec in specs:
            if len(spec) > 6 and spec[6] == fx.BANDS:
                continue
            key, label, lo, hi, default, decimals = spec[:6]
            value = self._values.get(key, default)
            if len(spec) > 6:                 # named values -> a menu
                self._add_choice(key, label, spec[6], value)
            else:
                self._add_slider(key, label, lo, hi, value, decimals,
                                 default=default)
        self._params_host.setVisible(bool(fx.PARAMS.get(effect)))
        self.updateGeometry()
        self.adjustSize()
        self.resized.emit()

    def _emit(self):
        self.changed.emit(self.values())


# The axes that count in DISPLAY levels; everything else is scene-linear.
# Listed rather than guessed, because the histogram and the waveform each have
# their own pair and the probe marker has to know which world it is in.
DISPLAY_AXES = (scopes.AXIS_DISPLAY, scopes.WF_AXIS_DISPLAY)


class _Canvas(QtWidgets.QWidget):
    """The common base of the scope canvases.

    Each of them draws its own backdrop, so the panel frame's transparency
    cannot reach them on its own - and the vectorscope paints an opaque colour
    target on top of that. The opacity is therefore driven through
    QPainter.setOpacity().
    """

    PROBE = QtGui.QColor(255, 255, 255, 225)
    PROBE_DARK = QtGui.QColor(0, 0, 0, 170)

    def __init__(self, parent=None):
        super(_Canvas, self).__init__(parent)
        self.setFocusPolicy(QtCore.Qt.NoFocus)
        self.opacity = 1.0
        # The pixel under the cursor, as probe_at built it, or None. Only ever
        # DRAWN with - the scope itself is not recomputed - so following the
        # mouse costs a repaint and nothing else.
        self._probe = None

    def set_probe(self, probe):
        """The pixel to mark, or None to take the marker off."""
        if probe is self._probe:
            return
        self._probe = probe
        self.update()

    def _probe_pos(self):
        """Where the probed pixel sits on the value axis (0..1), or None.

        WHICH value that is depends on the axis in use: a scene-linear axis
        goes through the same encoding the curves did, a display axis is plain
        0..255. Reading the wrong one would put the marker somewhere the trace
        never was, which is worse than having no marker at all.
        """
        if not self._probe:
            return None
        axis = getattr(self, "_axis", None)
        try:
            if axis in DISPLAY_AXES:
                shown = self._probe.get("shown")
                if shown is None:
                    return None
                lum = (0.2126 * float(shown[0]) + 0.7152 * float(shown[1])
                       + 0.0722 * float(shown[2]))
                return max(0.0, min(1.0, lum / 255.0))
            return float(scopes.value_to_pos(self._probe.get("lum", 0.0)))
        except Exception:
            return None

    def set_opacity(self, value):
        value = max(0.0, min(1.0, float(value)))
        if abs(value - self.opacity) < 1e-3:
            return
        self.opacity = value
        self.update()


class HistogramCanvas(_Canvas):
    """Axis 0 to 55 with a break at 1.0 - right of the line is clipping."""

    BASE_H = 104

    def __init__(self, parent=None):
        super(HistogramCanvas, self).__init__(parent)
        self.setFixedHeight(self.BASE_H)
        self._curves = None          # (n, HIST_BINS) 0..1
        self._colors = []            # a colour per curve
        self._clipped = 0.0          # the share of clipped pixels
        self._axis = scopes.AXIS_LINEAR     # (clipping line position, labels)

    def set_data(self, result):
        """`result` is the return value of scopes.histogram*, or None."""
        if result is None:
            self._curves, self._colors, self._clipped = None, [], 0.0
        else:
            (self._curves, self._colors, self._clipped, self._axis) = result
        self.update()

    def paintEvent(self, _event):
        p = QtGui.QPainter(self)
        p.setOpacity(self.opacity)
        w, h = self.width(), self.height()
        plot_h = h - 10                            # room for the labels below
        p.fillRect(0, 0, w, plot_h, QtGui.QColor(0, 0, 0, 120))

        n = scopes.HIST_BINS
        clip_pos, labels = self._axis
        clip_x = clip_pos * w

        # A faint line at every mark, drawn BEFORE the curves so it never sits
        # on top of the data. Without it the numbers along the bottom say where
        # a value is only if it happens to be exactly at one of them.
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 28), 1))
        for pos, _text in labels:
            if 0.0 < pos < 1.0:
                x = int(pos * w)
                p.drawLine(x, 0, x, plot_h)
        # the clipping area is tinted, so it reads at a glance
        if clip_x < w - 1:
            p.fillRect(int(clip_x), 0, int(w - clip_x), plot_h,
                       QtGui.QColor(255, 60, 0, 34))

        if self._curves is not None:
            p.setRenderHint(QtGui.QPainter.Antialiasing, True)
            for ch in range(self._curves.shape[0]):
                row = self._curves[ch]
                path = QtGui.QPainterPath()
                path.moveTo(0, plot_h)
                for i in range(n):
                    x = i / float(n - 1) * w
                    path.lineTo(x, plot_h - row[i] * (plot_h - 2))
                path.lineTo(w, plot_h)
                path.closeSubpath()
                r, g, b = self._colors[ch]
                p.fillPath(path, QtGui.QColor(r, g, b, 90))
                p.setPen(QtGui.QPen(QtGui.QColor(r, g, b, 200), 1))
                p.drawPath(path)
            p.setRenderHint(QtGui.QPainter.Antialiasing, False)

        # the line at the clipping boundary
        if clip_x < w - 1:
            p.setPen(QtGui.QPen(QtGui.QColor(255, 210, 0, 220), 1))
            p.drawLine(int(clip_x), 0, int(clip_x), plot_h)

        # THE PIXEL UNDER THE CURSOR. Drawn last so it is never buried under a
        # curve: it answers "where in this distribution is the thing I am
        # pointing at", which is only worth anything if it can be seen.
        # A dark line under a light one, because it has to read over both a
        # bright curve and the empty black to the right of it.
        pos = self._probe_pos()
        if pos is not None:
            x = int(pos * w)
            p.setPen(QtGui.QPen(self.PROBE_DARK, 3))
            p.drawLine(x, 0, x, plot_h)
            p.setPen(QtGui.QPen(self.PROBE, 1))
            p.drawLine(x, 0, x, plot_h)

        font = p.font()
        font.setPointSize(max(6, font.pointSize() - 2))
        p.setFont(font)
        p.setPen(QtGui.QColor(150, 150, 150))
        for pos, text in labels:
            x = int(pos * w)
            if pos >= 1.0:
                x -= 6 * len(text)
            elif pos > 0.0:
                x -= 3 * len(text)
            p.drawText(max(0, x), h - 1, text)
        if self._clipped > 0.0005:
            p.setPen(QtGui.QColor(255, 150, 60))
            p.drawText(0, 0, w, 11, QtCore.Qt.AlignRight,
                       "clipped %.1f %%" % (self._clipped * 100.0))
        p.end()


class VectorscopeCanvas(_Canvas):
    """Pixel colour: Cb horizontally, Cr vertically, the centre is neutral.

    It is fed the FINISHED IMAGE, so the radius means saturation relative to a
    full-scale signal and an over-exposed area lands in the centre (on screen
    it is white). See the header of scopes.py for why a scene-linear axis up
    to 55 does not work here.
    """

    SQUARE = True                # height = width, so the circle fills the panel

    def __init__(self, parent=None):
        super(VectorscopeCanvas, self).__init__(parent)
        self._plate = scopes.hue_plate()          # the backdrop is computed once
        self._rgb = None                          # KEEPS the buffer alive for the QImage
        self._image = None

    def set_data(self, grid):
        if grid is None:
            self._image = None
            self._rgb = None
        else:
            rgb = (self._plate.astype(np.float32) * grid[:, :, None])
            self._rgb = np.ascontiguousarray(rgb.astype(np.uint8))
            size = self._rgb.shape[0]
            self._image = QtGui.QImage(self._rgb.data, size, size, size * 3,
                                       QtGui.QImage.Format_RGB888)
        self.update()

    def paintEvent(self, _event):
        p = QtGui.QPainter(self)
        p.setOpacity(self.opacity)     # the target is an opaque image, otherwise
        w, h = self.width(), self.height()   # the opacity knob would do nothing
        side = min(w, h)
        x0, y0 = (w - side) // 2, (h - side) // 2
        box = QtCore.QRect(x0, y0, side, side)

        # NO backdrop fill of our own: the panel already has its backdrop and a
        # second layer over it made the scope square a darker patch that stood
        # out from the rest of the panel. There is one backdrop, shared.
        if self._image is not None:
            p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
            p.drawImage(box, self._image)

        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 55), 1))
        # a pixel less, so the outline is not clipped by the edge when the
        # panel is filled
        p.drawEllipse(QtCore.QRect(x0, y0, side - 1, side - 1))
        cx, cy = x0 + side / 2.0, y0 + side / 2.0
        p.drawLine(int(cx), y0, int(cx), y0 + side)
        p.drawLine(x0, int(cy), x0 + side, int(cy))

        font = p.font()
        font.setPointSize(max(6, font.pointSize() - 2))
        p.setFont(font)

        # The graticule of a broadcast vectorscope: two boxes per primary -
        # the 75 % colour bars, which material is normally graded against, and
        # 100 % further out as the absolute limit - plus the hexagon joining
        # the 75 % points. Each in ITS OWN colour, so it is instantly clear
        # which colour the trace pulls towards.
        def at(pos):
            tx, ty = pos
            return QtCore.QPointF(cx + tx * side / 2.0, cy - ty * side / 2.0)

        # VS_TARGETS is in circular order, so this joins them the short way
        p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 45), 1))
        p.drawPolygon(QtGui.QPolygonF([at(scopes.TARGET_POS[n])
                                       for n in scopes.VS_TARGETS]))

        for name in scopes.VS_TARGETS:
            color = QtGui.QColor(*scopes.TARGET_COLORS[name])
            for pos, half_w in ((scopes.TARGET_POS[name], 2),
                                (scopes.TARGET_POS_100[name], 3)):
                q = at(pos)
                box = QtCore.QRect(int(q.x()) - half_w, int(q.y()) - half_w,
                                   half_w * 2, half_w * 2)
                p.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 150), 3))
                p.drawRect(box)               # dark outline, so it reads on
                p.setPen(QtGui.QPen(color, 1))   # top of the trace as well
                p.drawRect(box)
            # the label belongs to the 75 % box - that is the one you read
            q = at(scopes.TARGET_POS[name])
            p.setPen(QtGui.QColor(*scopes.TARGET_COLORS[name]))
            p.drawText(int(q.x()) + 5, int(q.y()) + 3, name)

        # THE PIXEL UNDER THE CURSOR, as a ring rather than a filled dot: on a
        # dense trace a solid dot disappears into it, while a ring keeps its
        # own outline and still lets the colour underneath show through, which
        # is the thing being pointed at. Its position comes from
        # scopes.vectorscope_point, i.e. the same maths the trace was built
        # with, so it lands exactly where that pixel's contribution is.
        if self._probe:
            xy = scopes.vectorscope_point(self._probe.get("shown"))
            if xy is not None:
                q = at(xy)
                p.setBrush(QtCore.Qt.NoBrush)
                p.setPen(QtGui.QPen(self.PROBE_DARK, 3))
                p.drawEllipse(q, 4.5, 4.5)
                p.setPen(QtGui.QPen(self.PROBE, 1.4))
                p.drawEllipse(q, 4.5, 4.5)
        p.end()


class WaveformCanvas(_Canvas):
    """Values along the image columns: columns horizontally, the value
    vertically (0 at the bottom). The channels are drawn over each other, so
    where they overlap white comes out.

    The axis is decided by whoever supplies the data: with scene-linear the top
    is 55 and the clipping line sits at HIST_SPLIT, in QC mode it is 0-255 off
    the screen.
    """

    BASE_H = 110

    def __init__(self, parent=None):
        super(WaveformCanvas, self).__init__(parent)
        self.setFixedHeight(self.BASE_H)
        self._rgb = None                          # KEEPS the buffer alive
        self._image = None
        self._axis = scopes.WF_AXIS_DISPLAY

    def set_data(self, grid, axis=None):
        self._axis = axis or scopes.WF_AXIS_DISPLAY
        if grid is None:
            self._image, self._rgb = None, None
        else:
            self._rgb = np.ascontiguousarray(
                np.clip(grid * 255.0, 0, 255).astype(np.uint8))
            h, w = self._rgb.shape[0], self._rgb.shape[1]
            self._image = QtGui.QImage(self._rgb.data, w, h, w * 3,
                                       QtGui.QImage.Format_RGB888)
        self.update()

    def paintEvent(self, _event):
        p = QtGui.QPainter(self)
        p.setOpacity(self.opacity)
        w, h = self.width(), self.height()
        if self._image is not None:
            p.drawImage(QtCore.QRect(0, 0, w, h), self._image)

        # horizontal lines - orientation in levels. Positions on the axis are
        # measured from the bottom (0 = value 0), hence the flip when drawing.
        font = p.font()
        font.setPointSize(max(6, font.pointSize() - 2))
        p.setFont(font)
        clip, marks = self._axis
        for pos, text in marks:
            y = int(round((1.0 - pos) * (h - 1)))
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 45), 1))
            p.drawLine(0, y, w, y)
            if text:
                p.setPen(QtGui.QColor(170, 170, 170))
                p.drawText(2, max(9, min(h - 2, y + (9 if pos > 0.5 else -2))),
                           text)
        # the clipping line stands out more - that is the boundary worth watching
        if clip is not None:
            y = int(round((1.0 - clip) * (h - 1)))
            p.setPen(QtGui.QPen(QtGui.QColor(255, 90, 90, 150), 1))
            p.drawLine(0, y, w, y)

        # THE PIXEL UNDER THE CURSOR: a cross, because here BOTH axes mean
        # something - which column it is in and how bright it is. The column
        # needs the image width, which only the panel knows, so it comes in
        # with the probe.
        pos, col = self._probe_pos(), self._probe_column()
        if pos is not None and col is not None:
            x, y = int(col * (w - 1)), int(round((1.0 - pos) * (h - 1)))
            for pen in (QtGui.QPen(self.PROBE_DARK, 3),
                        QtGui.QPen(self.PROBE, 1)):
                p.setPen(pen)
                p.drawLine(x, 0, x, h)
                p.drawLine(max(0, x - 5), y, min(w, x + 5), y)
        p.end()

    def _probe_column(self):
        """Where along the width the probed pixel is (0..1), or None."""
        if not self._probe:
            return None
        width = self._probe.get("image_w") or 0
        if width <= 1:
            return None
        return max(0.0, min(1.0, float(self._probe.get("x", 0)) / (width - 1.0)))


class ScopePanel(_Panel):
    """The shared wrapper for the histogram and the vectorscope."""

    def __init__(self, title, canvas, parent=None):
        # expanded: it is switched on by a checkbox on the node, so once on the
        # graph should be visible right away (it can still be collapsed with
        # the button in the header)
        super(ScopePanel, self).__init__(title, PANEL_W, CC_LABEL_W,
                                         parent=parent)
        self.canvas = canvas
        # a circle cannot be stretched, so only the others get the height grip
        self.RESIZE_V = not getattr(canvas, "SQUARE", False)
        self.form.addWidget(self.canvas)
        # No sliders here. The vectorscope used to have a multiplier that
        # stretched the trace from the centre, but with a fixed graticule there
        # is nothing left to compare a stretched trace against - the whole
        # point of the 75 % and 100 % boxes is that they are absolute.
        self._scaled()
        self._apply_collapsed()

    def set_probe(self, probe):
        self.canvas.set_probe(probe)

    def set_opacity(self, value):
        """Besides the frame also the canvas - it draws its own backdrop."""
        super(ScopePanel, self).set_opacity(value)
        self.canvas.set_opacity(value)

    def base_height(self):
        return getattr(self.canvas, "BASE_H", 100)

    def _scaled(self):
        """The canvas fills the panel right to the edges.

        The vectorscope is SQUARE, so its height follows the inner width and it
        has no height grip at all: the radius is what saturation is read from,
        and a stretched circle would put the 75 % boxes at two different
        distances depending on the angle - a graticule that no longer means
        anything.

        The histogram and the waveform have no such tie, so their height is a
        separate drag. Width and height used to move together, which meant the
        only way to a taller histogram was a wider one - pushing the rest of
        the column out of the picture to read the top of a curve.
        """
        inner = self.inner_width()
        if getattr(self.canvas, "SQUARE", False):
            self.canvas.setFixedHeight(inner)
        else:
            self.canvas.setFixedHeight(self.content_height())


class _Stack(QtWidgets.QWidget):
    """A column of panels in the image, held at its edge.

    CAREFUL: no WA_TransparentForMouseEvents - in Qt that disables the mouse
    for all the children too, so the sliders could not be clicked. The widget
    is therefore exactly as large as the panels themselves and is never in the
    way of the image anywhere else.
    """

    GAP = 6                          # the gap below the window bar
    RIGHT = False                    # which edge the stack stands at

    refitted = QtCore.Signal()       # the height changed (collapse, on/off)

    def __init__(self, parent=None, below=None):
        super(_Stack, self).__init__(parent)
        self._below = below          # the window bar the stack lines up under
        self._anchor = (EDGE, EDGE)
        if below is not None:
            # the bar changes width with the layer names - the panels below
            # follow it
            below.moved.connect(self._match_width)
            below.moved.connect(self._reposition)
        self.setFocusPolicy(QtCore.Qt.NoFocus)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        # the widget sizes itself to its content - had it stayed smaller, the
        # sliders would be clipped and could not be clicked
        lay.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self._lay = lay
        self._panels = []
        if parent is not None:
            parent.installEventFilter(self)      # to hold on to its edge

    def _add_panels(self, panels):
        """CAREFUL: the panels are created ONLY NOW, with us as the parent.

        In Qt a widget without a parent is a top-level WINDOW. When one was
        created earlier and anything displayed it in the meantime, a floating
        window with the panel popped up over Nuke. This way no panel is ever
        parentless, not even for a moment.
        """
        align = QtCore.Qt.AlignRight if self.RIGHT else QtCore.Qt.AlignLeft
        for p in panels:
            # towards the edge the stack stands at, so the panels line up even
            # when they have different widths (each is dragged separately)
            p.grip_side = (QtCore.Qt.LeftEdge if self.RIGHT
                           else QtCore.Qt.RightEdge)
            self._lay.addWidget(p, 0, align)
            p.resized.connect(self._refit)
            self._panels.append(p)
        self._match_width()
        self._refit()

    def _match_width(self):
        """The panels are as wide as the window bar above them."""
        if self._below is None:
            return
        width = self._below.width()
        for p in self._panels:
            p.set_base_width(width)

    def set_anchor(self, x, y):
        """Where the stack belongs. For a right-hand column x is the right edge."""
        if (x, y) == self._anchor:
            return
        self._anchor = (int(x), int(y))
        self._reposition()

    def bottom(self):
        """The bottom edge from the REAL position - see SlotBar.bottom()."""
        return self.y() + self.height()

    def eventFilter(self, obj, event):
        if (event.type() == QtCore.QEvent.Resize
                and obj is self.parentWidget()):
            self._reposition()
        return False

    def resizeEvent(self, event):
        super(_Stack, self).resizeEvent(event)
        self._reposition()                       # collapsing changes width and height

    def _reposition(self):
        if self.parentWidget() is None:
            return
        x, y = self._anchor
        if self._below is not None:
            # It stands below the bar of its window - in BOTH axes. When only y
            # was taken, the stack stayed at the default x and in Double the
            # controls of the right window opened under the left half.
            x, y = self._below.x(), self._below.bottom() + self.GAP
        if self.RIGHT:
            x = max(0, x - self.width())
        self.move(x, y)

    def _refit(self):
        """The size follows whatever is expanded."""
        self.adjustSize()
        self._reposition()
        self.raise_()
        self.refitted.emit()

    def _apply(self, pairs):
        """CAREFUL: no "only set it when it differs from isVisible()".
        isVisible() is False even when the PARENT is hidden - and that is an
        ordinary state (window 2 in Single, or the whole panel before it is
        first shown). Hiding was then skipped and the panel popped up as soon
        as the parent appeared; the button could no longer close it."""
        for panel, want in pairs:
            panel.setVisible(bool(want))
        self._refit()


class ControlStack(_Stack):
    """On the left below the window bar: CC and QC.

    The whole control set of a window is thus together in one column - first
    what is displayed (the bar), then how (colour and check mode).
    """

    RIGHT = False

    def __init__(self, parent=None, below=None):
        super(ControlStack, self).__init__(parent, below)
        self.cc = CCPanel(self)
        self.fx = EffectPanel(self)
        self.matte = MattePanel(self)
        self.panels = {"cc": self.cc, "qc": self.fx}
        # DiMatte is LAST in the column. When CC or QC is switched on, it
        # appears above it and DiMatte moves down - instead of the panels
        # wedging themselves into the middle and it jumping somewhere else
        # every other minute.
        self._add_panels((self.cc, self.fx, self.matte))
        self.matte.setVisible(False)

    def set_visibility(self, cc=True, qc=True, matte=False):
        self._apply(((self.cc, cc), (self.fx, qc), (self.matte, matte)))


class ScopeStack(_Stack):
    """Top right: the histogram and the vectorscope.

    The measuring panels stand opposite the control ones - you look at them
    once things are set, and they are not in the way where the clicking happens.
    """

    RIGHT = True

    def __init__(self, parent=None):
        super(ScopeStack, self).__init__(parent)
        self.hist = ScopePanel("Histogram", HistogramCanvas(), self)
        self.vscope = ScopePanel("Vectorscope", VectorscopeCanvas(), self)
        self.wave = ScopePanel("Waveform", WaveformCanvas(), self)
        self.panels = {"hist": self.hist, "vscope": self.vscope,
                       "wave": self.wave}
        self._add_panels((self.hist, self.vscope, self.wave))

    def set_probe(self, probe):
        """The pixel under the cursor, for the marker on each scope."""
        for panel in (self.hist, self.vscope, self.wave):
            panel.set_probe(probe)

    def set_visibility(self, hist=False, vscope=False, wave=False):
        self._apply(((self.hist, hist), (self.vscope, vscope),
                     (self.wave, wave)))

    def set_opacity(self, value):
        """The backdrop opacity of both scopes (a knob on the node)."""
        for panel in (self.hist, self.vscope):
            panel.set_opacity(value)

    def set_scope_active(self, hist, vscope, wave=False):
        """Should the scope be computed at all?

        On switching off the graph is cleared - a frozen graph would lie about
        what is currently on screen.
        """
        for panel, want in ((self.hist, hist), (self.vscope, vscope),
                            (self.wave, wave)):
            want = bool(want)
            if panel.active and not want:
                panel.canvas.set_data(None)
            panel.active = want

    # The size is not set from here - each scope is dragged by its own left
    # edge (see _Panel.mouseMoveEvent).

    # ------------------------------------------------------------- scopes
    def wants_scopes(self):
        """Do not compute the scopes needlessly - a collapsed panel does not need them."""
        return (self.hist.is_open() or self.vscope.is_open()
                or self.wave.is_open())

    def update_scopes(self, ctx):
        """Recomputes the open scopes. `ctx` is ImageView.scope_source().

        The histogram and the waveform follow one switch: in ordinary display
        they measure scene-linear data (and everything above 1 is visible in
        them), in QC mode the finished image - a QC visualisation has no
        scene-linear equivalent. The vectorscope always takes the finished
        image.
        """
        channel = scopes.channel_key(ctx.get("channels", 0))
        qc = bool(ctx.get("qc"))
        display, linear = ctx.get("display"), ctx.get("linear")
        gain, sat = ctx.get("gain", 1.0), ctx.get("sat_matrix")
        gamma, lz = ctx.get("gamma", 1.0), ctx.get("linearize")

        if self.vscope.is_open():
            # ALWAYS off the finished image - in both modes. It measures colour,
            # not level, and that only means anything in a bounded domain (see
            # the header of scopes.py). CC and the display transform are
            # already in that image, so it follows them for free.
            # The channel selection ISOLATES here: with a single channel the
            # trace is a line pointing at that primary and its length is how
            # much of the channel is in the shot (see scopes.vectorscope).
            self.vscope.canvas.set_data(scopes.vectorscope(display, channel))
        if self.wave.is_open():
            if qc:
                self.wave.canvas.set_data(scopes.waveform(display, channel),
                                          scopes.WF_AXIS_DISPLAY)
            else:
                self.wave.canvas.set_data(
                    scopes.waveform_linear(linear, channel, gain, sat,
                                           linearize=lz, gamma=gamma),
                    scopes.WF_AXIS_LINEAR)
        if not self.hist.is_open():
            return
        if qc:
            self.hist.canvas.set_data(
                scopes.histogram_display(display, channel))
            return
        self.hist.canvas.set_data(
            scopes.histogram(linear, channel, gain, sat, linearize=lz,
                             gamma=gamma))
