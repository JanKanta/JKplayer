"""
One place where Qt is imported, so the rest of the package does not care
whether it is running on Qt5 or Qt6.

WHY: Nuke changed the binding in the middle of the supported range -
Nuke 14 ships PySide2 (Qt5), Nuke 15 and up ship PySide6 (Qt6). The modules
are otherwise compatible enough, but three things differ and all three are
used here:

  * mouse and wheel events lost pos() and gained position(), and it returns
    a QPointF instead of a QPoint
  * QShortcut moved from QtWidgets to QtGui
  * enterEvent gets a QEnterEvent in Qt6 and a plain QEvent in Qt5 (the
    signature is the same, so only the annotation would differ - nothing to
    solve here, just do not touch the argument)

Import Qt from here, never straight from PySide.
"""

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    QT6 = True
except ImportError:                       # Nuke 14 and older
    from PySide2 import QtCore, QtGui, QtWidgets
    QT6 = False

# Qt6 moved it out of QtWidgets
QShortcut = QtGui.QShortcut if QT6 else QtWidgets.QShortcut


def event_pos(event):
    """The position of a mouse/wheel event as a QPointF, on Qt5 and Qt6 alike.

    Qt6 has position() and returns a QPointF; Qt5 only has pos() with whole
    pixels. Everything here works in floats (the wipe line, the panel edges),
    so the Qt5 value is converted rather than the other way round.
    """
    if QT6:
        return event.position()
    return QtCore.QPointF(event.pos())
