"""Look through the pictures an export is about to write, and crop them.

Opened by Export, before anything is written. Each picture can be zoomed into
to be looked at, and given a crop; what comes out of the export is the crop,
scaled back up to the size of the frame. Nothing is written if it is cancelled.

CROPS ARE KEPT AS FRACTIONS OF THE PICTURE (x, y, w, h in 0..1), not pixels:
the preview is drawn at whatever size fits the window, and the export renders
the frame again at full resolution - a fraction means the same thing in both.
"""

from .qtcompat import QtCore, QtGui, QtWidgets, event_pos

# How close to an edge or a corner the pointer has to be to take hold of it,
# in screen pixels - generous, because a crop edge is a one-pixel line.
HANDLE = 9
# The smallest crop, as a fraction of the picture. Below this a drag is taken
# as a click, which clears the crop instead of making an invisible one.
MIN_CROP = 0.01
THUMB = 168             # thumbnail width in the strip


# ---------------------------------------------------------------------------
# The arithmetic - no widgets, so it can be tested on its own
# ---------------------------------------------------------------------------
def normalize(x0, y0, x1, y1):
    """A crop from two corners in any order, held inside the picture."""
    xa, xb = sorted((min(1.0, max(0.0, x0)), min(1.0, max(0.0, x1))))
    ya, yb = sorted((min(1.0, max(0.0, y0)), min(1.0, max(0.0, y1))))
    return (xa, ya, xb - xa, yb - ya)


def is_crop(rect):
    """True for a crop worth keeping - not None, not a click, not everything."""
    if rect is None:
        return False
    x, y, w, h = rect
    if w < MIN_CROP or h < MIN_CROP:
        return False
    return not (x <= 1e-6 and y <= 1e-6 and w >= 1 - 1e-6 and h >= 1 - 1e-6)


def move(rect, dx, dy):
    """The crop moved by (dx, dy), stopped at the edges without changing size."""
    x, y, w, h = rect
    x = min(1.0 - w, max(0.0, x + dx))
    y = min(1.0 - h, max(0.0, y + dy))
    return (x, y, w, h)


def resize(rect, edge, dx, dy):
    """The crop with one edge or corner dragged by (dx, dy).

    `edge` is a compass point - n, s, e, w or a corner like "nw". The opposite
    side stays where it is, the dragged one stops at the picture's edge and
    cannot be pulled past its partner: a crop that turns inside out would be a
    different crop from the one being made.
    """
    x0, y0, w, h = rect
    x1, y1 = x0 + w, y0 + h
    if "w" in edge:
        x0 = min(x1 - MIN_CROP, max(0.0, x0 + dx))
    if "e" in edge:
        x1 = max(x0 + MIN_CROP, min(1.0, x1 + dx))
    if "n" in edge:
        y0 = min(y1 - MIN_CROP, max(0.0, y0 + dy))
    if "s" in edge:
        y1 = max(y0 + MIN_CROP, min(1.0, y1 + dy))
    return (x0, y0, x1 - x0, y1 - y0)


def hit(box, pos, reach=HANDLE):
    """What a press at `pos` takes hold of on a crop drawn at `box`.

    -> a corner ("nw", "ne", "se", "sw"), an edge ("n", "e", "s", "w"),
    "move" inside it, or None outside. Corners win over edges: they sit
    where two edges meet, and the corner is what a hand aims for there.
    """
    if box is None:
        return None
    x, y = pos
    left, top, right, bottom = box
    near_l, near_r = abs(x - left) <= reach, abs(x - right) <= reach
    near_t, near_b = abs(y - top) <= reach, abs(y - bottom) <= reach
    inside_x = left - reach <= x <= right + reach
    inside_y = top - reach <= y <= bottom + reach
    if not (inside_x and inside_y):
        return None
    vert = "n" if near_t else ("s" if near_b else "")
    horz = "w" if near_l else ("e" if near_r else "")
    if vert and horz:
        return vert + horz
    if vert and left < x < right:
        return vert
    if horz and top < y < bottom:
        return horz
    if left < x < right and top < y < bottom:
        return "move"
    return None


def crop_and_fit(image, rect):
    """The crop of `image`, scaled back up to the frame's size.

    Up to FIT, keeping its shape: a free crop is rarely the frame's shape, and
    stretching it to fill the frame would put a distortion into the review
    that is not in the shot. The result is as big as it can be inside the
    original width and height.
    """
    if image is None or image.isNull() or not is_crop(rect):
        return image
    W, H = image.width(), image.height()
    x, y, w, h = rect
    px = int(round(x * W))
    py = int(round(y * H))
    pw = max(1, min(W - px, int(round(w * W))))
    ph = max(1, min(H - py, int(round(h * H))))
    part = image.copy(px, py, pw, ph)
    return part.scaled(W, H, QtCore.Qt.KeepAspectRatio,
                       QtCore.Qt.SmoothTransformation)


# ---------------------------------------------------------------------------
# The canvas: one picture, zoomable, with the crop drawn on it
# ---------------------------------------------------------------------------
_CURSORS = {
    "move": QtCore.Qt.SizeAllCursor,
    "n": QtCore.Qt.SizeVerCursor, "s": QtCore.Qt.SizeVerCursor,
    "e": QtCore.Qt.SizeHorCursor, "w": QtCore.Qt.SizeHorCursor,
    "nw": QtCore.Qt.SizeFDiagCursor, "se": QtCore.Qt.SizeFDiagCursor,
    "ne": QtCore.Qt.SizeBDiagCursor, "sw": QtCore.Qt.SizeBDiagCursor,
}


class CropCanvas(QtWidgets.QWidget):
    """Draws one picture and the crop on it, and lets both be handled.

    Left button: drag on the picture to draw a crop, drag its edges or corners
    to reshape it, drag inside it to move it; a click without a drag clears it.
    Wheel: zoom at the pointer. Middle button: pan. Double-click: fit.
    """

    cropChanged = QtCore.Signal(object)       # (x, y, w, h) fractions, or None

    def __init__(self, parent=None):
        super(CropCanvas, self).__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMinimumSize(480, 300)
        self._image = None
        self._note = ""
        self._crop = None
        self._zoom = 1.0          # on top of fit; 1 = the picture fits
        self._pan = QtCore.QPointF(0.0, 0.0)   # screen px, from centred
        self._grab = None         # (what, press point in fractions, crop then)
        self._panning = None      # press point on screen, pan then

    # ---- content -------------------------------------------------------
    def set_image(self, image, note=""):
        self._image = image if image is not None and not image.isNull() \
            else None
        self._note = note
        self._zoom, self._pan = 1.0, QtCore.QPointF(0.0, 0.0)
        self.update()

    def set_crop(self, rect):
        self._crop = rect if is_crop(rect) else None
        self.update()

    def crop(self):
        return self._crop

    # ---- where the picture is on screen --------------------------------
    def _frame(self):
        """(left, top, width, height) of the whole picture on screen."""
        if self._image is None:
            return None
        iw, ih = self._image.width(), self._image.height()
        pad = 16.0
        fit = min((self.width() - 2 * pad) / float(iw),
                  (self.height() - 2 * pad) / float(ih))
        s = max(1e-3, fit * self._zoom)
        w, h = iw * s, ih * s
        left = (self.width() - w) / 2.0 + self._pan.x()
        top = (self.height() - h) / 2.0 + self._pan.y()
        return left, top, w, h

    def _to_frac(self, pos):
        f = self._frame()
        if f is None:
            return None
        left, top, w, h = f
        return ((pos.x() - left) / w, (pos.y() - top) / h)

    def _crop_box(self):
        """The crop on screen as (left, top, right, bottom), or None."""
        f = self._frame()
        if f is None or self._crop is None:
            return None
        left, top, w, h = f
        x, y, cw, ch = self._crop
        return (left + x * w, top + y * h, left + (x + cw) * w,
                top + (y + ch) * h)

    # ---- painting ------------------------------------------------------
    def paintEvent(self, _event):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(24, 24, 26))
        f = self._frame()
        if f is None:
            p.setPen(QtGui.QColor(150, 150, 150))
            p.drawText(self.rect(), QtCore.Qt.AlignCenter,
                       self._note or "(nothing to show)")
            p.end()
            return
        left, top, w, h = f
        target = QtCore.QRectF(left, top, w, h)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, self._zoom <= 1.0)
        p.drawImage(target, self._image)

        box = self._crop_box()
        if box is not None:
            l, t, r, b = box
            inner = QtCore.QRectF(l, t, r - l, b - t)
            # everything that will be cut away, dimmed
            outside = QtGui.QPainterPath()
            outside.addRect(target)
            keep = QtGui.QPainterPath()
            keep.addRect(inner)
            p.fillPath(outside.subtracted(keep), QtGui.QColor(0, 0, 0, 150))
            p.setRenderHint(QtGui.QPainter.Antialiasing, True)
            # thirds, faint - where people put the thing they want seen
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 60), 1.0))
            for i in (1, 2):
                x = l + (r - l) * i / 3.0
                y = t + (b - t) * i / 3.0
                p.drawLine(QtCore.QPointF(x, t), QtCore.QPointF(x, b))
                p.drawLine(QtCore.QPointF(l, y), QtCore.QPointF(r, y))
            p.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 160), 3.0))
            p.drawRect(inner)
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1.2))
            p.drawRect(inner)
            # corner handles, as thicker L-shapes the way photo apps draw them
            arm = min(18.0, (r - l) / 3.0, (b - t) / 3.0)
            p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 3.0))
            for cx, cy, sx, sy in ((l, t, 1, 1), (r, t, -1, 1),
                                   (r, b, -1, -1), (l, b, 1, -1)):
                p.drawLine(QtCore.QPointF(cx, cy),
                           QtCore.QPointF(cx + sx * arm, cy))
                p.drawLine(QtCore.QPointF(cx, cy),
                           QtCore.QPointF(cx, cy + sy * arm))
        p.end()

    # ---- mouse ---------------------------------------------------------
    def mousePressEvent(self, event):
        pos = event_pos(event)
        if event.button() == QtCore.Qt.MiddleButton:
            self._panning = (QtCore.QPointF(pos), QtCore.QPointF(self._pan))
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            return
        if event.button() != QtCore.Qt.LeftButton:
            return
        frac = self._to_frac(pos)
        if frac is None:
            return
        what = hit(self._crop_box(), (pos.x(), pos.y()))
        if what is None:
            # A FRESH CROP from this corner - so the old one goes now, not
            # when the mouse first moves. Otherwise a plain click outside,
            # with no move in it, left the old crop standing on release.
            what = "new"
            self._crop = None
            self.update()
        self._grab = (what, frac, self._crop)

    def mouseMoveEvent(self, event):
        pos = event_pos(event)
        if self._panning is not None:
            start, pan0 = self._panning
            self._pan = QtCore.QPointF(pan0.x() + pos.x() - start.x(),
                                       pan0.y() + pos.y() - start.y())
            self.update()
            return
        if self._grab is None:
            what = hit(self._crop_box(), (pos.x(), pos.y()))
            self.setCursor(_CURSORS.get(what, QtCore.Qt.CrossCursor))
            return
        frac = self._to_frac(pos)
        what, start, before = self._grab
        dx, dy = frac[0] - start[0], frac[1] - start[1]
        if what == "new":
            self._crop = normalize(start[0], start[1], frac[0], frac[1])
        elif what == "move" and before is not None:
            self._crop = move(before, dx, dy)
        elif before is not None:
            self._crop = resize(before, what, dx, dy)
        self.update()

    def mouseReleaseEvent(self, event):
        if self._panning is not None and event.button() == QtCore.Qt.MiddleButton:
            self._panning = None
            self.setCursor(QtCore.Qt.CrossCursor)
            return
        if self._grab is None:
            return
        what = self._grab[0]
        self._grab = None
        if what == "new" and not is_crop(self._crop):
            self._crop = None                 # a click, not a crop: clear it
        elif not is_crop(self._crop):
            self._crop = None
        self.update()
        self.cropChanged.emit(self._crop)

    def mouseDoubleClickEvent(self, _event):
        self._zoom, self._pan = 1.0, QtCore.QPointF(0.0, 0.0)
        self.update()

    def wheelEvent(self, event):
        """Zoom at the pointer - the spot under it stays under it."""
        if self._image is None:
            return
        delta = event.angleDelta().y()
        if not delta:
            return
        pos = event_pos(event)
        before = self._to_frac(pos)
        self._zoom = max(1.0, min(40.0,
                                  self._zoom * (1.2 if delta > 0 else 1 / 1.2)))
        if self._zoom <= 1.0:
            self._pan = QtCore.QPointF(0.0, 0.0)
        else:
            left, top, w, h = self._frame()
            # move the pan so the same fraction lands back under the pointer
            self._pan = QtCore.QPointF(
                self._pan.x() + pos.x() - (left + before[0] * w),
                self._pan.y() + pos.y() - (top + before[1] * h))
        self.update()
        event.accept()


# ---------------------------------------------------------------------------
# The dialog
# ---------------------------------------------------------------------------
class CropDialog(QtWidgets.QDialog):
    """Every picture of the export in a strip, the chosen one big, and a crop.

    `jobs` is a list of (label, render) - render() gives the full-resolution
    picture without its frame stamp, or None when the frame is not cached.
    Pictures are rendered when they are looked at, and the strip fills in
    behind that one at a time, so the dialog opens straight away even for a
    long export.
    """

    _last_size = None

    def __init__(self, jobs, parent=None):
        super(CropDialog, self).__init__(parent)
        self.setWindowTitle("Export - crop the pictures")
        self._jobs = list(jobs)
        self._crops = [None] * len(self._jobs)
        self._current = -1
        self._thumbs_done = 0

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        self.canvas = CropCanvas(self)
        self.canvas.cropChanged.connect(self._on_crop)
        lay.addWidget(self.canvas, 1)

        hint = QtWidgets.QLabel(
            "Drag on the picture to crop  ·  drag the edges to reshape  "
            "·  wheel to zoom  ·  middle button to pan  ·  "
            "double-click to fit  ·  a click clears the crop", self)
        hint.setStyleSheet("color: #8a8a8a;")
        lay.addWidget(hint)

        self.strip = QtWidgets.QListWidget(self)
        self.strip.setViewMode(QtWidgets.QListView.IconMode)
        self.strip.setFlow(QtWidgets.QListView.LeftToRight)
        self.strip.setWrapping(False)
        self.strip.setMovement(QtWidgets.QListView.Static)
        self.strip.setIconSize(QtCore.QSize(THUMB, int(THUMB * 9 / 16)))
        self.strip.setFixedHeight(int(THUMB * 9 / 16) + 48)
        self.strip.setSpacing(4)
        for label, _render in self._jobs:
            item = QtWidgets.QListWidgetItem(label)
            item.setTextAlignment(QtCore.Qt.AlignHCenter)
            self.strip.addItem(item)
        self.strip.currentRowChanged.connect(self._show)
        lay.addWidget(self.strip)

        row = QtWidgets.QHBoxLayout()
        self._prev = QtWidgets.QPushButton("◀", self)
        self._next = QtWidgets.QPushButton("▶", self)
        for b in (self._prev, self._next):
            b.setFixedWidth(34)
        self._prev.clicked.connect(lambda: self._step(-1))
        self._next.clicked.connect(lambda: self._step(1))
        self._count = QtWidgets.QLabel("", self)
        self._count.setMinimumWidth(60)
        reset = QtWidgets.QPushButton("Reset crop", self)
        reset.clicked.connect(self._reset)
        same = QtWidgets.QPushButton("Same crop for all", self)
        same.setToolTip("Puts this picture's crop on every picture of the "
                        "export.")
        same.clicked.connect(self._same_for_all)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=self)
        buttons.button(QtWidgets.QDialogButtonBox.Ok).setText("Export")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        for w in (self._prev, self._count, self._next):
            row.addWidget(w)
        row.addSpacing(12)
        row.addWidget(reset)
        row.addWidget(same)
        row.addStretch(1)
        row.addWidget(buttons)
        lay.addLayout(row)

        self.resize(CropDialog._last_size
                    if isinstance(CropDialog._last_size, QtCore.QSize)
                    else QtCore.QSize(1180, 820))

        self._thumb_timer = QtCore.QTimer(self)
        self._thumb_timer.setInterval(0)
        self._thumb_timer.timeout.connect(self._next_thumb)
        if self._jobs:
            self.strip.setCurrentRow(0)
            self._thumb_timer.start()

    # ---- navigation ----------------------------------------------------
    def keyPressEvent(self, event):
        key = event.key()
        if key in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Up):
            self._step(-1)
        elif key in (QtCore.Qt.Key_Right, QtCore.Qt.Key_Down):
            self._step(1)
        elif key == QtCore.Qt.Key_R:
            self._reset()
        else:
            super(CropDialog, self).keyPressEvent(event)

    def _step(self, d):
        if self._jobs:
            self.strip.setCurrentRow(
                max(0, min(len(self._jobs) - 1, self._current + d)))

    def _show(self, index):
        if not (0 <= index < len(self._jobs)):
            return
        self._current = index
        label, render = self._jobs[index]
        image = None
        try:
            image = render()
        except Exception:
            image = None
        self.canvas.set_image(
            image, "frame not in the cache - it will be skipped")
        self.canvas.set_crop(self._crops[index])
        self._count.setText("%d / %d" % (index + 1, len(self._jobs)))
        self._prev.setEnabled(index > 0)
        self._next.setEnabled(index < len(self._jobs) - 1)
        if image is not None:
            self._set_thumb(index, image)

    # ---- crops ---------------------------------------------------------
    def _on_crop(self, rect):
        if self._current >= 0:
            self._crops[self._current] = rect if is_crop(rect) else None
            self._mark(self._current)

    def _reset(self):
        self.canvas.set_crop(None)
        self._on_crop(None)

    def _same_for_all(self):
        rect = self.canvas.crop()
        for i in range(len(self._crops)):
            self._crops[i] = rect
            self._mark(i)

    def _mark(self, index):
        """The strip says which pictures have a crop, so none is forgotten."""
        item = self.strip.item(index)
        label = self._jobs[index][0]
        item.setText(("✂  " + label) if self._crops[index] else label)

    def crops(self):
        return list(self._crops)

    # ---- the strip, filled in the background ---------------------------
    def _set_thumb(self, index, image):
        item = self.strip.item(index)
        if item is None or item.data(QtCore.Qt.UserRole):
            return
        small = image.scaled(THUMB, int(THUMB * 9 / 16),
                             QtCore.Qt.KeepAspectRatio,
                             QtCore.Qt.SmoothTransformation)
        item.setIcon(QtGui.QIcon(QtGui.QPixmap.fromImage(small)))
        item.setData(QtCore.Qt.UserRole, True)

    def _next_thumb(self):
        while self._thumbs_done < len(self._jobs):
            i = self._thumbs_done
            self._thumbs_done += 1
            item = self.strip.item(i)
            if item.data(QtCore.Qt.UserRole):
                continue
            try:
                image = self._jobs[i][1]()
            except Exception:
                image = None
            if image is not None:
                self._set_thumb(i, image)
            return                          # one per tick, the UI stays live
        self._thumb_timer.stop()

    def done(self, result):
        CropDialog._last_size = self.size()
        self._thumb_timer.stop()
        super(CropDialog, self).done(result)

    @classmethod
    def ask(cls, jobs, parent=None):
        """The crop for every job (None = the whole picture), or None if the
        export was cancelled."""
        if not jobs:
            return []
        dlg = cls(jobs, parent)
        ok = dlg.exec_() if hasattr(dlg, "exec_") else dlg.exec()
        return dlg.crops() if ok else None
