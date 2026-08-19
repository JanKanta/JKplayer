"""
The Metadata tab on the node: pick the header fields and their order.

A real Qt widget rather than a row of Nuke knobs. Nuke has no knob that picks
and reorders, and the alternative - one checkbox per key - cannot work at all
here, because which keys exist depends on the file that happens to be attached.
PyCustom_Knob already hosts the whole player panel, so a widget on the node is
a road this plugin has been down before.

TWO panels, left and right. The left is the catalogue: what that input's file
carries, with the values. The right is the META list as that input's panel will
draw it, top to bottom.

ONE INPUT AT A TIME, picked from the menu at the top, and the menu governs BOTH
panels. In Sync the two windows are two different files and each draws its own
list, so Comp and Plate are two separate orders. Filtering only the right-hand
side would leave the catalogue offering Comp attributes while the list being
edited was Plate's - an arrangement whose only use is putting a line where it
does not belong.
"""

from . import meta
from . import node as exrnode
from .qtcompat import QtCore, QtGui, QtWidgets

def _say(text):
    """Into the script editor. Nuke is imported late - see MetaKnob._node."""
    try:
        import nuke
        nuke.tprint("JKplayer: %s" % text)
    except Exception:
        pass


COL_KEY, COL_VALUE = 0, 1
ROW_H = 18
PICKED_FG = "#666"          # already in the META list, greyed in the catalogue

# A FLOOR, not a size - the tables expand past it with the panel. Kept at what
# the tab used to be fixed at, so that if a Nuke build does not hand the knob
# any spare height the tab is no smaller than it was before.
MIN_LIST_H = 200


class MetaTable(QtWidgets.QWidget):
    """Left: what that input's headers hold. Right: what its panel draws.

    Emits `changed` with the whole [(tag, key)] - every input, in menu order -
    whenever anything moves.
    """

    changed = QtCore.Signal(object)

    def __init__(self, parent=None):
        super(MetaTable, self).__init__(parent)
        self._rows = []            # [(tag, key, value)] as harvested, all inputs
        self._listed = []          # [(tag, key, value)] the catalogue is showing
        self._by_tag = {}          # tag -> [(tag, key)], the order for that input
        self._quiet = False        # while filling, so nothing is emitted
        self._refresh_hook = None  # set by the knob, see MetaKnob

        # GROW WITH THE NODE PANEL. Without this the widget is handed its size
        # hint and keeps it, so dragging the panel taller left the two tables
        # the same size with grey below them.
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        outer.addLayout(self._top())

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(6)
        body.addLayout(self._left_side(), 3)
        body.addLayout(self._middle(), 0)
        body.addLayout(self._right_side(), 2)
        outer.addLayout(body, 1)          # the tables take the spare height

        foot = QtWidgets.QHBoxLayout()
        foot.setSpacing(6)
        self._note = QtWidgets.QLabel("", self)
        self._note.setStyleSheet("color: #999;")
        foot.addWidget(self._note, 1)
        for text, slot, tip in (
                ("Refresh", self._refresh,
                 "Read the headers of the frame the panel is on again."),
                ("Defaults", self._defaults,
                 "Take the usual review fields, for the input on show."),
                ("Clear", self._none,
                 "Empty the META list of the input on show.")):
            b = QtWidgets.QPushButton(text, self)
            b.setFixedHeight(20)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            foot.addWidget(b)
        outer.addLayout(foot)

    # ---- building -------------------------------------------------------
    def _top(self):
        """The input picker, above both panels because it governs both."""
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(self._heading("Input", ""))
        self.input_menu = QtWidgets.QComboBox(self)
        self.input_menu.setToolTip(
            "Which input is being set up. Each window draws the list of the\n"
            "input it is showing, so Comp and Plate are two separate orders -\n"
            "and a line can only come from the file it was read out of.")
        for tag, label in zip(exrnode.INPUT_TAGS, exrnode.INPUT_LABELS):
            # the name, not the letter - C and P are how the choice is STORED,
            # which is no reason to make anyone read it that way
            self.input_menu.addItem(str(label), tag)
        self.input_menu.currentIndexChanged.connect(lambda *_a: self._show_tag())
        row.addWidget(self.input_menu)

        # NEXT TO THE INPUT PICKER, and it survives switching between them: a
        # header has forty-odd attributes and the one being looked for is
        # usually the same one on both files.
        self.search = QtWidgets.QLineEdit(self)
        self.search.setPlaceholderText("search")
        self.search.setClearButtonEnabled(True)
        self.search.setFixedWidth(160)
        self.search.setToolTip(
            "Narrows the Available list as you type, matching from the start\n"
            "of the key. It stays put when the input is switched.")
        self.search.textChanged.connect(lambda *_a: self._show_tag())
        row.addWidget(self.search)
        row.addStretch(1)
        return row

    def _matches(self, key):
        """Does this key answer what is being typed?

        FROM THE START, not anywhere inside: typing "co" should offer
        compression, not every key with an "o" in the middle. A key like
        nuke/version is really two words, so each of its parts counts as a
        start - otherwise the half that is actually memorable cannot be typed.
        """
        text = str(self.search.text() or "").strip().lower()
        if not text:
            return True
        key = str(key).lower()
        if key.startswith(text):
            return True
        parts = key.replace("/", " ").replace("_", " ").replace(".", " ").split()
        return any(part.startswith(text) for part in parts)

    def _left_side(self):
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(2)
        col.addWidget(self._heading(
            "Available", "Everything this input's file carries on this frame."))
        self.table = QtWidgets.QTableWidget(0, 2, self)
        self.table.setHorizontalHeaderLabels(["Key", "Value"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setMinimumHeight(MIN_LIST_H)
        self.table.setMinimumWidth(260)
        self.table.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                 QtWidgets.QSizePolicy.Expanding)
        try:
            hh = self.table.horizontalHeader()
            hh.setStretchLastSection(True)
            hh.resizeSection(COL_KEY, 170)
        except Exception:
            pass
        # a double click is the fast way; the arrow is the discoverable one
        self.table.itemDoubleClicked.connect(lambda *_a: self._take())
        col.addWidget(self.table, 1)
        return col

    def _middle(self):
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(4)
        col.addStretch(1)
        for text, slot, tip in (
                (u"→", self._take, "Add the selected rows to META."),
                (u"←", self._drop, "Remove the selected rows from META."),
                (u"↑", lambda: self._move(-1),
                 "Move what is selected in META one place up."),
                (u"↓", lambda: self._move(1),
                 "Move what is selected in META one place down.")):
            b = QtWidgets.QPushButton(text, self)
            b.setFixedSize(26, 22)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            col.addWidget(b)
            if text == u"←":
                col.addSpacing(10)    # across is one job, up and down another
        col.addStretch(1)
        return col

    def _right_side(self):
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(2)
        col.addWidget(self._heading(
            "META", "What this input's panel draws, in this order. "
                    "Drag a row, or use the arrows."))
        self.chosen = QtWidgets.QListWidget(self)
        self.chosen.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection)
        # Drag a row to move it. This IS the ordering control, the arrows are
        # the same thing for anyone who does not think to try dragging.
        self.chosen.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.chosen.setMinimumHeight(MIN_LIST_H)
        self.chosen.setMinimumWidth(150)
        self.chosen.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                  QtWidgets.QSizePolicy.Expanding)
        self.chosen.itemDoubleClicked.connect(lambda *_a: self._drop())
        self.chosen.model().rowsMoved.connect(self._emit)
        col.addWidget(self.chosen, 1)
        return col

    def _heading(self, text, tip):
        lab = QtWidgets.QLabel(text, self)
        lab.setStyleSheet("color: #bbb; font-weight: bold;")
        if tip:
            lab.setToolTip(tip)
        return lab

    # ---- which input is on show -----------------------------------------
    def _tags(self):
        return list(exrnode.INPUT_TAGS)

    def _tag(self):
        """The input both panels are currently showing."""
        data = self.input_menu.currentData()
        return data if data in self._tags() else self._tags()[0]

    # ---- filling --------------------------------------------------------
    def set_data(self, rows, order):
        """`rows` is what the headers hold, `order` the chosen (tag, key) list."""
        self._rows = list(rows or [])
        have = {(t, k) for t, k, _v in self._rows}

        self._by_tag = dict((t, []) for t in self._tags())
        for pair in (order or []):
            if pair in have and pair[0] in self._by_tag:
                self._by_tag[pair[0]].append(pair)
        self._show_tag()

    def _show_tag(self):
        """Put the current input into BOTH panels."""
        tag = self._tag()
        self._listed = [r for r in self._rows
                        if r[0] == tag and self._matches(r[1])]

        self._quiet = True
        try:
            self.table.setRowCount(len(self._listed))
            for i, (_t, key, value) in enumerate(self._listed):
                self._fill_row(i, key, value)
            self.chosen.clear()
            for pair in self._by_tag.get(tag, ()):
                item = QtWidgets.QListWidgetItem(str(pair[1]))
                item.setData(QtCore.Qt.UserRole, pair)
                self.chosen.addItem(item)
        finally:
            self._quiet = False
        self._grey_taken()
        self._retell()

    def _fill_row(self, i, key, value):
        item = QtWidgets.QTableWidgetItem(str(key))
        item.setData(QtCore.Qt.UserRole, str(key))
        self.table.setItem(i, COL_KEY, item)
        val = QtWidgets.QTableWidgetItem(str(value))
        val.setToolTip(str(value))
        self.table.setItem(i, COL_VALUE, val)
        self.table.setRowHeight(i, ROW_H)

    def _retell(self):
        if not self._rows:
            self._note.setText("nothing attached, or the headers are empty - "
                               "open the player panel")
            return
        counts = ", ".join("%s %d" % (exrnode.input_name(t),
                                      len(self._by_tag.get(t, ())))
                           for t in self._tags())
        tag = self._tag()
        whole = len([r for r in self._rows if r[0] == tag])
        # say BOTH numbers while a search is on, or an empty-looking list reads
        # as an empty file rather than as a search that matched nothing
        found = ("%d of %d fields" % (len(self._listed), whole)
                 if len(self._listed) != whole
                 else "%d fields on this frame" % whole)
        self._note.setText("%s: %s   |   in META: %s"
                           % (exrnode.input_name(tag), found, counts))

    def _grey_taken(self):
        """Dim in the catalogue whatever is already in this input's list."""
        taken = {k for _t, k in self._by_tag.get(self._tag(), ())}
        grey = QtGui.QColor(PICKED_FG)
        for i in range(self.table.rowCount()):
            item = self.table.item(i, COL_KEY)
            if item is None:
                continue
            on = item.data(QtCore.Qt.UserRole) in taken
            for c in (COL_KEY, COL_VALUE):
                cell = self.table.item(i, c)
                if cell is not None:
                    cell.setForeground(grey if on else QtGui.QBrush())

    # ---- reading back ---------------------------------------------------
    def _live(self):
        """[(tag, key)] as the list widget currently stands."""
        out = []
        for i in range(self.chosen.count()):
            pair = self.chosen.item(i).data(QtCore.Qt.UserRole)
            if pair:
                out.append(tuple(pair))
        return out

    def order(self):
        """Every input's list, one after another, in menu order.

        Flat because that is what the node stores; which input a line belongs
        to is in the line itself, so splitting it up again on the way back is
        just a matter of reading the tag.
        """
        out = []
        for tag in self._tags():
            out.extend(self._by_tag.get(tag, ()))
        return out

    def _emit(self, *_a):
        if self._quiet:
            return
        self._by_tag[self._tag()] = self._live()   # the widget is the truth
        self._grey_taken()
        self._retell()
        self.changed.emit(self.order())

    # ---- moving across --------------------------------------------------
    def _take(self):
        """Selected catalogue rows -> the end of this input's META list."""
        tag = self._tag()
        mine = self._by_tag.setdefault(tag, [])
        added = False
        for i in sorted(idx.row() for idx in
                        self.table.selectionModel().selectedRows()):
            item = self.table.item(i, COL_KEY)
            if item is None:
                continue
            key = item.data(QtCore.Qt.UserRole)
            # the same key twice would draw twice; a second click is a no-op
            if not key or (tag, key) in mine:
                continue
            mine.append((tag, key))
            added = True
        if added:
            self._show_tag()
            self._emit()

    def _drop(self):
        rows = sorted((self.chosen.row(it) for it in self.chosen.selectedItems()),
                      reverse=True)
        for r in rows:
            self.chosen.takeItem(r)
        if rows:
            self._emit()
            self._grey_taken()

    def _move(self, delta):
        """Shift what is selected in META one place up (-1) or down (+1).

        The block moves or it does not. A selection that has reached the edge
        could still be moved row by row until it piled up there, which quietly
        changes the order INSIDE the selection - so hitting the edge stops the
        whole thing instead.
        """
        rows = sorted(self.chosen.row(it) for it in self.chosen.selectedItems())
        if not rows:
            return
        if delta < 0:
            if rows[0] <= 0:
                return
            walk = rows                  # topmost first, into the gap above
        else:
            if rows[-1] >= self.chosen.count() - 1:
                return
            walk = list(reversed(rows))  # bottom first, or they overwrite

        self._quiet = True               # one signal for the move, not one each
        try:
            for r in walk:
                item = self.chosen.takeItem(r)
                self.chosen.insertItem(r + delta, item)
                item.setSelected(True)   # taking it out cleared the selection
        finally:
            self._quiet = False
        self._emit()

    # ---- the row of buttons ---------------------------------------------
    def _refresh(self):
        if callable(self._refresh_hook):
            self._refresh_hook()

    def _none(self):
        """Empty the list on show. The other input is left alone."""
        self._by_tag[self._tag()] = []
        self._show_tag()
        self._emit()

    def _defaults(self):
        """The usual review fields, for the input on show."""
        tag = self._tag()
        self._by_tag[tag] = [p for p in meta.default_order(self._rows)
                             if p[0] == tag]
        self._show_tag()
        self._emit()


class MetaKnob(object):
    """What PyCustom_Knob instantiates. Nuke calls makeUI() to get the widget.

    The chosen order is written straight back to the hidden string knob, so it
    is saved with the script and there is no separate state to keep in step.
    """

    def __init__(self, node_name, knob_name):
        self.node_name = node_name
        self.knob_name = knob_name
        self.widget = None

    def makeUI(self):
        self.widget = MetaTable()
        self.widget.changed.connect(self._store)
        self.widget._refresh_hook = self.refresh
        self.refresh()
        return self.widget

    def updateValue(self):
        """Nuke calls this when the knob is asked to re-read itself."""
        self.refresh()

    def _node(self):
        import nuke
        return nuke.toNode(self.node_name)

    def refresh(self):
        if self.widget is None:
            return
        node = self._node()
        text = ""
        if node is not None:
            try:
                text = node[self.knob_name].value()
            except Exception as exc:
                # An unreadable knob looks exactly like an unconfigured one, so
                # without this the saved choice would quietly turn back into
                # the defaults with nothing to say why.
                _say("cannot read %s on %s (%s)"
                     % (self.knob_name, self.node_name, exc))
                text = ""
        rows = meta.harvest(self.node_name)
        order = meta.parse_order(text)
        if order is None:
            # Untouched: show what the panels are drawing anyway, so the tab
            # opens on the truth rather than on an empty list.
            order = meta.default_order(rows)
        self.widget.set_data(rows, order)

    def _store(self, order):
        node = self._node()
        if node is None:
            return
        try:
            node[self.knob_name].setValue(meta.format_order(order))
        except Exception as exc:
            # Losing the choice without a word is the one thing worse than
            # losing it - the tab would look right and the panel would not.
            _say("cannot store the metadata choice on %s (%s)"
                 % (self.node_name, exc))
