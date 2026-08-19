"""Notes drawn straight onto the picture, per frame.

The point of this is a review pass: scribble on the frames that are wrong, then
hand someone the JPEGs. So only the frames actually drawn on exist here, and
only those come out of the export.

EVERYTHING IS STORED IN IMAGE PIXELS, never in screen ones. A circle drawn
around a wire at 200 % zoom has to stay on that wire at 50 % and in the
exported full-resolution JPEG - which it only does if what is remembered is
where it sits IN THE PLATE. The one transform (see `draw`) then serves both the
viewer and the export: the viewer passes the pan and zoom it is drawing at, the
export passes none.
"""

import os
import re

from .qtcompat import QtCore, QtGui

# Deliberately loud - a note has to be unmistakable against a plate, and these
# read on both a dark and a bright one. Red first: it is what people reach for.
COLORS = ((255, 60, 60), (255, 210, 40), (60, 220, 90), (80, 170, 255),
          (255, 255, 255), (0, 0, 0))
COLOR_NAMES = ("Red", "Yellow", "Green", "Blue", "White", "Black")

LINE_W = 3.0            # in IMAGE pixels, so the stroke keeps its weight
TEXT_H = 34.0           # cap height of a note, also in image pixels

def look_key(look):
    """What makes two views the same for annotation purposes.

    The check and the channels: those decide WHAT is on screen. The exposure or
    a check's own sliders do not - a note must not vanish because a slider
    moved.
    """
    if not look:
        return (None, None)
    return (look.get("effect"), look.get("channels"))


class Annotations(object):
    """Strokes and text notes, keyed by frame number."""

    def __init__(self):
        # Every item carries the LOOK it was made in - the check, its settings,
        # the channels, the exposure. Per item and not per frame: one frame is
        # reviewed in several checks, and a note drawn in the grain check has
        # to stay with the grain check even if the frame was already written on
        # in another one. The export then writes one picture per check.
        self._strokes = {}      # frame -> [(colour, width, [(x, y), ...], look)]
        self._texts = {}        # frame -> [(x, y, colour, size, text, look)]
        # The longest line, off the node. Kept HERE rather than passed down
        # through draw() and text_at() by every caller: it is one setting for
        # the whole set of notes, and the two must never be given different
        # ones or a note would be caught somewhere other than it is drawn.
        self.line_max = LINE_MAX
        # The family draw() last painted with. Hit testing has no painter to
        # ask, and measuring with a different font than the one on screen
        # catches a note somewhere other than it is seen.
        self.font_family = ""

    # ---- contents ------------------------------------------------------
    def frames(self):
        """Every frame that has something on it, in order."""
        return sorted(set(self._strokes) | set(self._texts))

    def has(self, frame):
        return bool(self._strokes.get(frame) or self._texts.get(frame))

    def runs(self):
        """Annotated frames as (from, to) runs - what the timeline draws."""
        out = []
        for f in self.frames():
            if out and f == out[-1][1] + 1:
                out[-1][1] = f
            else:
                out.append([f, f])
        return out

    # ---- editing -------------------------------------------------------
    def add_stroke(self, frame, points, color=0, width=LINE_W, look=None):
        """One pen stroke. Fewer than two points is a stray click, not a mark."""
        points = [(float(x), float(y)) for x, y in points]
        if len(points) < 2:
            return False
        frame = int(frame)
        self._strokes.setdefault(frame, []).append(
            (int(color), float(width), points, dict(look) if look else None))
        return True

    def add_text(self, frame, x, y, text, color=0, size=TEXT_H, look=None):
        # capped here and not only in the box: this is the one door every note
        # comes through, whatever writes it
        text = (text or "").strip()[:MAX_CHARS]
        if not text:
            return False
        frame = int(frame)
        self._texts.setdefault(frame, []).append(
            (float(x), float(y), int(color), float(size), text,
             dict(look) if look else None))
        return True

    def text_at(self, frame, x, y, look=None, image_width=0, image_height=0):
        """Index of the note written at (x, y), or None.

        Searched from the LAST one back, so where two overlap the one drawn on
        top - the one being pointed at - is the one found.
        """
        items = self._texts.get(int(frame))
        if not items:
            return None
        want = None if look is None else look_key(look)
        for i in range(len(items) - 1, -1, -1):
            tx, ty, _color, size, text, ilook = items[i]
            if want is not None and ilook is not None \
                    and look_key(ilook) != want:
                continue
            metrics = _metrics(size, self.font_family)
            # exactly what draw() will do with it, or the note would be caught
            # somewhere other than where it is seen
            lines, tx, ty = layout_note(text, metrics, tx, ty,
                                        image_width, image_height,
                                        EDGE_PAD, self.line_max)
            lines = lines or [""]
            wide = max([_advance(metrics, ln) for ln in lines] or [0])
            pad = max(4.0, size * 0.2)      # a note is small; be easy to hit
            # drawText puts the BASELINE of the first line on ty, so the block
            # reaches an ascent above it
            top = ty - metrics.ascent() - pad
            bottom = (ty + (len(lines) - 1) * metrics.lineSpacing()
                      + metrics.descent() + pad)
            if tx - pad <= x <= tx + wide + pad and top <= y <= bottom:
                return i
        return None

    def text_of(self, frame, index):
        """What a note says, for putting back into the box to be edited."""
        items = self._texts.get(int(frame)) or ()
        if 0 <= index < len(items):
            return items[index][4]
        return ""

    def move_text(self, frame, index, x, y):
        """Puts a note somewhere else on the plate. True when it really moved."""
        items = self._texts.get(int(frame))
        if not items or not (0 <= index < len(items)):
            return False
        old = items[index]
        if old[0] == float(x) and old[1] == float(y):
            return False
        items[index] = (float(x), float(y)) + old[2:]
        return True

    def text_size(self, frame, index):
        """How big a note is written, so the box can say what fits on a line."""
        items = self._texts.get(int(frame)) or ()
        if 0 <= index < len(items):
            return items[index][3]
        return TEXT_H

    def text_pos(self, frame, index):
        """(x, y) of a note, to drag it from where it already is."""
        items = self._texts.get(int(frame)) or ()
        if 0 <= index < len(items):
            return items[index][0], items[index][1]
        return 0.0, 0.0

    def text_color(self, frame, index):
        """Which colour a note is written in, to show it in the box."""
        items = self._texts.get(int(frame)) or ()
        if 0 <= index < len(items):
            return items[index][2]
        return 0

    def replace_text(self, frame, index, text, color=None):
        """Rewrites a note in place. An empty text deletes it.

        Position, size and the view it belongs to are kept, and so is the
        colour unless a new one is given: this is a correction of a note, not
        a new one.
        """
        items = self._texts.get(int(frame))
        if not items or not (0 <= index < len(items)):
            return False
        text = (text or "").strip()[:MAX_CHARS]
        if not text:
            items.pop(index)
            if not items:
                self._texts.pop(int(frame), None)
            return True
        old = items[index]
        color = old[2] if color is None else int(color)
        if old[4] == text and old[2] == color:
            return False
        items[index] = (old[0], old[1], color, old[3], text, old[5])
        return True

    def notes(self, frame, look=None):
        """The written notes on a frame, in the order they were put there.

        `look` narrows it to one view, matching what draw() would show and what
        the export writes into one file.
        """
        want = None if look is None else look_key(look)
        return [t[4] for t in self._texts.get(int(frame), ())
                if want is None or t[5] is None or look_key(t[5]) == want]

    def strokes_count(self, frame, look=None):
        """How many drawn marks - a frame can be flagged without a word on it."""
        want = None if look is None else look_key(look)
        return len([s for s in self._strokes.get(int(frame), ())
                    if want is None or s[3] is None or look_key(s[3]) == want])

    def looks(self, frame):
        """The distinct views this frame was written on in, in first-use order.

        One entry per view, so the export knows how many pictures this frame
        turns into and which to render for each.
        """
        found = {}
        for item in self._strokes.get(int(frame), ()):
            found.setdefault(look_key(item[3]), item[3])
        for item in self._texts.get(int(frame), ()):
            found.setdefault(look_key(item[5]), item[5])
        return list(found.values())

    def _last_index(self, items, look_at, look):
        """The last item in `items` belonging to `look` (None = any)."""
        want = None if look is None else look_key(look)
        for i in range(len(items) - 1, -1, -1):
            if want is None or items[i][look_at] is None \
                    or look_key(items[i][look_at]) == want:
                return i
        return None

    def undo(self, frame, look=None):
        """Takes back the last thing put on this frame IN THIS VIEW.

        Restricted to the view on purpose: undo must not reach into notes that
        are not on screen, or it silently removes work you only miss at export.
        """
        frame = int(frame)
        for store, look_at in ((self._strokes, 3), (self._texts, 5)):
            items = store.get(frame)
            if not items:
                continue
            i = self._last_index(items, look_at, look)
            if i is None:
                continue
            items.pop(i)
            if not items:
                del store[frame]
            return True
        return False

    def clear(self, frame, look=None):
        """Everything on this frame, or everything belonging to one view."""
        frame = int(frame)
        if look is None:
            had = self.has(frame)
            self._strokes.pop(frame, None)
            self._texts.pop(frame, None)
            return had
        want = look_key(look)
        removed = False
        for store, look_at in ((self._strokes, 3), (self._texts, 5)):
            items = store.get(frame)
            if not items:
                continue
            keep = [it for it in items
                    if it[look_at] is not None and look_key(it[look_at]) != want]
            if len(keep) != len(items):
                removed = True
                if keep:
                    store[frame] = keep
                else:
                    del store[frame]
        return removed

    # ---- drawing -------------------------------------------------------
    def draw(self, painter, frame, ox=0.0, oy=0.0, zoom=1.0, look=None,
             width=0, height=0):
        """Draws the notes of one frame.

        `ox`/`oy`/`zoom` place the image on whatever is being painted - the
        widget for the viewer, or nothing at all for the export, which is why
        they default to the identity. `width`/`height` are the image size in
        pixels, which notes are wrapped and held inside; 0 leaves them alone.

        `look` is what is on screen now. Notes made in a DIFFERENT view are
        left out: a circle around a grain problem drawn over a plain plate
        points at nothing. The export passes none, because it has already
        rebuilt the very view the notes were made in.
        """
        frame = int(frame)
        strokes = self._strokes.get(frame)
        texts = self._texts.get(frame)
        if look is not None:
            want = look_key(look)
            strokes = [s for s in (strokes or ())
                       if s[3] is None or look_key(s[3]) == want]
            texts = [t for t in (texts or ())
                     if t[5] is None or look_key(t[5]) == want]
        if not strokes and not texts:
            return
        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        for color, pen_w, points, _look in (strokes or ()):
            pen = QtGui.QPen(QtGui.QColor(*COLORS[color % len(COLORS)]))
            # The line keeps its weight IN THE PLATE, so a note drawn while
            # zoomed in does not become a hairline when you zoom out.
            pen.setWidthF(max(1.0, pen_w * zoom))
            pen.setCapStyle(QtCore.Qt.RoundCap)
            pen.setJoinStyle(QtCore.Qt.RoundJoin)
            painter.setPen(pen)
            path = QtGui.QPolygonF([QtCore.QPointF(ox + x * zoom, oy + y * zoom)
                                    for x, y in points])
            painter.drawPolyline(path)

        if texts:
            font = painter.font()
            font.setBold(True)
            self.font_family = font.family()   # so hit testing measures alike
            for x, y, color, size, text, _look in texts:
                # THE LAYOUT IS DONE IN IMAGE PIXELS and only then scaled.
                # Doing it in screen pixels meant the font size was rounded to
                # a whole number first, so the line breaks and the spacing
                # shifted about as the view was zoomed instead of the note
                # simply growing with the picture.
                font.setPixelSize(max(1, int(round(size))))
                base = QtGui.QFontMetrics(font)
                lines, tx, ty = layout_note(text, base, x, y, width, height,
                                            EDGE_PAD, self.line_max)
                step = base.lineSpacing()
                # per note, so changing the size later does not rewrite the
                # ones already placed
                font.setPixelSize(max(1, int(round(size * zoom))))
                painter.setFont(font)
                rgb = COLORS[color % len(COLORS)]
                # A contrasting outline: white text on a white sky - or black
                # text on a dark frame - is invisible otherwise, and a note
                # that cannot be read is not a note.
                edge = (QtGui.QColor(255, 255, 255, 200) if sum(rgb) < 200
                        else QtGui.QColor(0, 0, 0, 190))
                # LINE BY LINE: drawText at a point lays out one line and
                # renders a newline as a box.
                for row, line in enumerate(lines):
                    if not line:
                        continue
                    at = QtCore.QPointF(ox + tx * zoom,
                                        oy + (ty + row * step) * zoom)
                    painter.setPen(edge)
                    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        painter.drawText(at + QtCore.QPointF(dx, dy), line)
                    painter.setPen(QtGui.QColor(*rgb))
                    painter.drawText(at, line)
        painter.restore()


def draw_frame_number(painter, frame, width, height, label=None):
    """The frame number, top LEFT of the exported picture.

    Without it a folder of JPEGs is a pile of pictures with notes on them and
    no way back to the shot. Sized off the picture, so it is legible on a 4K
    plate and does not swamp a small one, and set on a dark plate so it can be
    read over a bright sky.
    """
    size = max(14, int(round(height * 0.028)))
    pad = max(6, size // 2)
    text = "%d" % int(frame)
    if label:
        text += "   %s" % label

    font = painter.font()
    font.setPixelSize(size)
    font.setBold(True)
    painter.save()
    painter.setFont(font)
    metrics = QtGui.QFontMetrics(font)
    box = metrics.boundingRect(text)
    painter.fillRect(QtCore.QRect(0, 0,
                                  box.width() + pad * 2, size + pad * 2),
                     QtGui.QColor(0, 0, 0, 150))
    painter.setPen(QtGui.QColor(255, 255, 255))
    painter.drawText(QtCore.QPoint(pad, pad + size - metrics.descent() // 2),
                     text)
    painter.restore()


EDGE_PAD = 30            # image pixels kept clear of every edge
# The line length FLOATS with how far the note is from the NEARER side edge:
# out in the middle it gets long lines, pushed towards either edge it gets
# short ones. Below LINE_MIN a note is a column of single words; LINE_MAX is
# only the default - the node sets it.
LINE_MIN = 30
LINE_MAX = 130
# How far in from a side edge a note starts shortening its lines, as a share
# of the image width. Only a strip along the edge, so a note anywhere in the
# open keeps full-length lines instead of shrinking on the way there.
EDGE_BAND = 0.08
MAX_CHARS = 1000         # longest note - a review note, not an essay


def _advance(metrics, text):
    """Width of a string. horizontalAdvance is Qt 5.11+, width() before it."""
    if hasattr(metrics, "horizontalAdvance"):
        return metrics.horizontalAdvance(text)
    return metrics.width(text)


def _metrics(size, family=""):
    """Metrics for a note of that size, for code with no painter to hand.

    Used to work out WHERE a note is - hit testing and the hint in the box.
    `family` is the one drawing last used, so the two agree; without it a
    default font is measured and a note can be caught beside itself.
    """
    font = QtGui.QFont()
    if family:
        font.setFamily(family)
    font.setPixelSize(max(1, int(round(size))))
    font.setBold(True)
    return QtGui.QFontMetrics(font)


def _fits(text, metrics, avail, chars):
    """Whether a line may still grow: the character count and the room for it.

    Two limits rather than one. The count is what floats with where the note
    sits; the measured width is the backstop that keeps even a line of wide
    letters between the insets. Whichever bites first wins.
    """
    if len(text) > chars:
        return False
    return avail <= 0 or _advance(metrics, text) <= avail


def _break_word(word, metrics, avail, chars):
    """Splits a word too long for a line of its own, character by character.

    Without this a pasted file path - one 'word' with no spaces in it - would
    still run off the picture however narrow the column is.
    """
    out = []
    part = ""
    for ch in word:
        if part and not _fits(part + ch, metrics, avail, chars):
            out.append(part)
            part = ch
        else:
            part += ch
    if part:
        out.append(part)
    return out


def wrap_lines(text, metrics, avail, chars=LINE_MAX):
    """Breaks a note into lines of at most `chars` that also fit `avail`.

    Done when the note is DRAWN, not when it is written: the metrics are then
    the ones actually painting the text, so what is measured is what appears.
    Wrapping at entry meant guessing the font, and a wrong guess is invisible
    until a line already hangs off the plate. It is also why the lines re-flow
    while a note is dragged - the room to the edge changes as it moves.

    The viewer and the export cannot disagree either - both come through here.
    """
    lines = []
    for para in (text or "").split("\n"):
        if not para.strip():
            lines.append("")              # a blank line separates paragraphs
            continue
        line = ""
        for word in para.split():
            probe = word if not line else line + " " + word
            if _fits(probe, metrics, avail, chars):
                line = probe
                continue
            if line:
                lines.append(line)
            if _fits(word, metrics, avail, chars):
                line = word
                continue
            pieces = _break_word(word, metrics, avail, chars)
            lines.extend(pieces[:-1])
            line = pieces[-1] if pieces else ""
        if line:
            lines.append(line)
    return lines


def line_chars(metrics, x, image_width, pad=EDGE_PAD, line_max=LINE_MAX):
    """How many characters a line holds for a note sitting at `x`.

    Two things shorten a line, and the tighter one wins:

    * the room actually left to the right-hand inset - a block has to FIT, so
      a note out towards the right gets fewer characters however you feel
      about it;
    * being inside the EDGE_BAND of either side, which is what gives the left
      edge the same behaviour as the right.

    The band matters because the left cannot be done by room: a full line is
    wider than half the frame, so measuring from the left inset would start
    shortening lines from the middle of the picture outwards - the text would
    shrink long before it was anywhere near the side.
    """
    if not image_width:
        return int(line_max)
    unit = _advance(metrics, "n") or 1.0
    width = float(image_width)
    room = (width - pad - float(x)) / unit
    band = min(float(x) - pad, width - pad - float(x))
    reach = max(1.0, width * EDGE_BAND)
    near = LINE_MIN + (line_max - LINE_MIN) * min(1.0, band / reach)
    return int(max(LINE_MIN, min(float(line_max), room, near)))


def layout_note(text, metrics, x, y, image_width, image_height, pad=EDGE_PAD,
                line_max=LINE_MAX):
    """(lines, x, y) - a note broken up and held inside the picture.

    Laid out first and MOVED second. The line length floats with where the note
    sits (see line_chars), but never below LINE_MIN - so a note dropped at the
    very edge is not squeezed into a column of single letters; the whole block
    slides back inside instead.

    The same on every side: pushed left or up when it would cross the inset,
    never right or down, and never past the inset on the other side. A block
    taller than the frame keeps its TOP - the first line is the one to read.

    Every unit here is the caller's - image pixels for hit testing, screen
    pixels for drawing - as long as they are all the same one.
    """
    chars = line_chars(metrics, x, image_width, pad, line_max)
    # the widest a block may ever be, wherever it ends up
    limit = (float(image_width) - 2.0 * pad) if image_width else 0.0
    lines = wrap_lines(text, metrics, limit, chars)
    nx, ny = float(x), float(y)
    if image_width:
        widest = max([_advance(metrics, ln) for ln in lines] or [0])
        nx = max(float(pad), min(nx, float(image_width) - pad - widest))
    if image_height:
        # y is the BASELINE of the first line, so the block starts an ascent
        # above it and ends a descent below the last one
        up = metrics.ascent()
        down = (len(lines) - 1) * metrics.lineSpacing() + metrics.descent()
        ny = min(ny, float(image_height) - pad - down)
        ny = max(ny, float(pad) + up)
    return lines, nx, ny


def fits_per_line(size, x, image_width, line_max=LINE_MAX):
    """How many characters a line holds where the note is - for the box.

    The same figure the drawing will use, so the width typed to is the width
    that lands on the plate.
    """
    return line_chars(_metrics(size), x, image_width, EDGE_PAD, line_max)


REPORT_NAME = "annotations.csv"


def write_report(path, rows):
    """The list of exported frames, next to the pictures.

    A folder of JPEGs cannot be read without opening every one of them; this
    is the same review as a table - frame, which check it was seen in, the file
    it went to, and what was written on it.

    SEMICOLONS and a BOM, not plain comma-separated UTF-8: notes get written in
    Czech and opened in Excel, where a comma file lands in a single column and
    the diacritics come out as mojibake. Python's csv writer does the quoting,
    so a note with a semicolon or several lines in it survives the trip.
    """
    import csv
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        out = csv.writer(fh, delimiter=";", quoting=csv.QUOTE_MINIMAL)
        out.writerow(["frame", "check", "file", "marks", "note"])
        for frame, check, filename, marks, note in rows:
            out.writerow([frame, check or "-", filename, marks, note])
    return path


def export_name(pattern, frame, label=None):
    """The file name for one exported frame.

    `####` in the pattern is replaced by the zero-padded frame, as everywhere
    else in this trade. Without it the number is appended, so a pattern that
    forgot it still produces one file per frame instead of one file overwritten
    by every frame.
    """
    name = pattern or "annotation_####.jpg"
    if not os.path.splitext(name)[1]:
        name += ".jpg"
    match = re.search(r"#+", name)
    if match:
        pad = len(match.group(0))
        name = name[:match.start()] + str(int(frame)).zfill(pad) + name[match.end():]
    else:
        stem, ext = os.path.splitext(name)
        name = "%s_%04d%s" % (stem, int(frame), ext)
    if label:
        stem, ext = os.path.splitext(name)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")
        if safe:
            name = "%s_%s%s" % (stem, safe, ext)
    return name
