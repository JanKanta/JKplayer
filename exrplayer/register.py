"""Registration into Nuke - menu + panel. Called from ~/.nuke/menu.py."""

import nuke

# IMPORTANT: nukescripts.registerWidgetAsPanel does not store the class, it
# builds the TEXT
#   "...WidgetKnob(exrplayer.panel.PlayerPanel)"
# and EVALUATES it later in Nuke's global namespace. For that to work,
# (a) the attribute `exrplayer.panel` has to exist (i.e. the submodule must
# already be imported) and (b) the name `exrplayer` has to be visible in
# __main__. Without it the panel works as a floating window but cannot be
# docked in Nuke.
from . import panel as _panel_module          # makes sure exrplayer.panel exists

_floating = None                 # the fallback window, when there is no Viewer
_docked = None                   # the pane beside the Viewer, once opened


def _expose_in_main():
    """Expose the package in __main__, where the panel expression is evaluated."""
    try:
        import __main__
        import exrplayer
        setattr(__main__, "exrplayer", exrplayer)
    except Exception as exc:
        nuke.tprint("JKplayer: cannot expose in __main__: %s" % exc)


PANEL_ID = "com.honza.EXRplayerPanel"
PANEL_NAME = "JKplayer"

# The widget is named as an EXPRESSION, evaluated later in __main__ - which is
# what _expose_in_main is for.
_WIDGET_EXPR = ("__import__('nukescripts').panels.WidgetKnob("
                "exrplayer.panel.PlayerPanel)")


def _new_pane_panel():
    """A Nuke pane wrapping our widget.

    The same thing nukescripts.registerWidgetAsPanel builds, made here so it
    can be docked into a CHOSEN pane: the ready-made one only ever lands in
    nuke.thisPane(), which is set while a Pane menu entry is being run and
    nowhere else. Built here it also leaves the Pane menu alone - registering
    again just to get an instance would re-add the entry.
    """
    import nukescripts
    panel = nukescripts.PythonPanel(PANEL_NAME, PANEL_ID)
    panel.addKnob(nuke.PyCustom_Knob(PANEL_NAME, "", _WIDGET_EXPR))
    return panel


def _viewer_pane():
    """The pane the Viewer lives in, or None when there is no Viewer."""
    for name in ("Viewer.1", "Viewer1"):
        try:
            pane = nuke.getPaneFor(name)
        except Exception:
            pane = None
        if pane is not None:
            return pane
    return None


def open_panel(force=False):
    """Opens the panel DOCKED beside the Viewer, floating only as a fallback.

    `force` opens another one even if we already opened one - that is the menu
    entry. Everything else reuses, because a second panel would fight the first
    over the same node and fill a second cache in RAM.
    """
    global _floating, _docked
    if not force and _docked is not None:
        return _docked
    if not force and _floating is not None:
        try:
            if _floating.isVisible():
                _floating.raise_()
                _floating.activateWindow()
                return _floating
        except RuntimeError:
            _floating = None        # Qt threw the window away, build a new one

    pane = _viewer_pane()
    if pane is not None:
        try:
            _docked = _new_pane_panel()
            _docked.addToPane(pane)
            return _docked
        except Exception as exc:
            _docked = None
            nuke.tprint("JKplayer: could not dock beside the Viewer (%s), "
                        "opening a window instead" % exc)

    _floating = _panel_module.PlayerPanel()
    _floating.setWindowTitle(PANEL_NAME)
    _floating.resize(1280, 820)
    _floating.show()
    _floating.raise_()
    return _floating


def create_node():
    """Creates the node AND shows the panel - the node on its own displays
    nothing, so making one and seeing no picture is a puzzle, not a workflow."""
    from . import node as nodemod
    node = nodemod.create()
    try:
        open_panel()
    except Exception as exc:
        # a panel that will not open must not lose you the node
        nuke.tprint("JKplayer: node created, panel did not open: %s" % exc)
    return node


_guard_busy = False
_guard_installed = False
_knob_watch_installed = False


def _viewer_guard():
    """Disconnects any Viewer that has an JKplayer node on its input.

    Registered with nodeClass='Viewer', so it only runs for Viewers (not for
    every node in the graph) and `nuke.thisNode()` is that very Viewer.
    """
    global _guard_busy
    if _guard_busy:
        return                      # setInput inside the callback would loop
    _guard_busy = True
    try:
        from . import node as nodemod
        viewer = nuke.thisNode()
        names = []
        if viewer is not None and viewer.Class() == "Viewer":
            for i in range(viewer.inputs()):
                if nodemod.is_player_node(viewer.input(i)):
                    viewer.setInput(i, None)
                    names.append(viewer.name())
        else:
            names = nodemod.enforce_no_viewer()
        if names:
            nuke.tprint("JKplayer: a Viewer cannot be attached - disconnected "
                        "(%s). Display is handled by the JKplayer panel."
                        % ", ".join(set(names)))
    except Exception:
        pass
    finally:
        _guard_busy = False


def _install_viewer_guard():
    global _guard_installed
    if _guard_installed:
        return
    try:
        nuke.addUpdateUI(_viewer_guard, nodeClass="Viewer")
        _guard_installed = True
    except Exception as exc:
        nuke.tprint("JKplayer: cannot install the Viewer guard: %s" % exc)


def _on_knob_changed():
    """Color management switched -> show only the knobs of that mode.

    Runs on every Group (and legacy NoOp) node, so it first checks whether
    this really is one of ours.
    """
    try:
        name = nuke.thisKnob().name()
        from . import node as nodemod
        # what is visible on the node depends on the view mode and on which
        # panels are on - hence QC as well (it hides the QC mode selector)
        watched = ("cv_view_mode",) + tuple("cv_qc_%s" % s
                                            for s in nodemod.SLOT_LABELS)
        if name != "cv_color_mgmt" and name not in watched:
            return
        n = nuke.thisNode()
        if not nodemod.is_player_node(n):
            return
        if name == "cv_color_mgmt":
            nodemod.apply_color_visibility(n)
        else:
            nodemod.apply_view_visibility(n)
    except Exception:
        pass


def _install_knob_watch():
    global _knob_watch_installed
    if _knob_watch_installed:
        return
    from . import node as nodemod
    ok = False
    for cls in nodemod.NODE_CLASSES:       # Group = today, NoOp = legacy nodes
        try:
            nuke.addKnobChanged(_on_knob_changed, nodeClass=cls)
            ok = True
        except Exception as exc:
            nuke.tprint("JKplayer: cannot install the knob watch for %s: "
                        "%s" % (cls, exc))
    _knob_watch_installed = ok


_registered = False


def register():
    """Everything Nuke needs to know about us. Safe to call more than once.

    Nuke can end up running our menu.py twice - the same folder reachable
    through two plugin paths is enough - and then every menu entry would be
    there twice. The callbacks already guard themselves; this guards the rest.
    """
    global _registered
    if _registered:
        return
    _registered = True

    _expose_in_main()
    _install_viewer_guard()
    _install_knob_watch()

    try:
        import nukescripts
        # Puts JKplayer in the Pane menu and makes the layout able to restore
        # it. Opening one ourselves goes through _new_pane_panel instead - see
        # there for why.
        nukescripts.registerWidgetAsPanel(
            "exrplayer.panel.PlayerPanel", PANEL_NAME, PANEL_ID)
    except Exception as exc:
        nuke.tprint("JKplayer: panel registration failed: %s" % exc)

    try:
        m = nuke.menu("Nuke").addMenu("JKplayer")
        m.addCommand("Create JKplayer Node",
                     "import exrplayer.register as r; r.create_node()")
        m.addCommand("Open JKplayer Panel",
                     "import exrplayer.register as r; r.open_panel(force=True)")
    except Exception as exc:
        nuke.tprint("JKplayer: menu failed: %s" % exc)

    try:
        nuke.menu("Nodes").addCommand(
            "JKplayer/JKplayer",
            "import exrplayer.register as r; r.create_node()")
    except Exception:
        pass
