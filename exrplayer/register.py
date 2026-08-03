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

_floating = None


def _expose_in_main():
    """Expose the package in __main__, where the panel expression is evaluated."""
    try:
        import __main__
        import exrplayer
        setattr(__main__, "exrplayer", exrplayer)
    except Exception as exc:
        nuke.tprint("EXRplayer: cannot expose in __main__: %s" % exc)


def open_panel():
    """Opens the EXRplayer panel as a floating window."""
    global _floating
    _floating = _panel_module.PlayerPanel()
    _floating.setWindowTitle("EXRplayer")
    _floating.resize(1280, 820)
    _floating.show()
    _floating.raise_()
    return _floating


def create_node():
    from . import node as nodemod
    return nodemod.create()


_guard_busy = False
_guard_installed = False
_knob_watch_installed = False


def _viewer_guard():
    """Disconnects any Viewer that has an EXRplayer node on its input.

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
            nuke.tprint("EXRplayer: a Viewer cannot be attached - disconnected "
                        "(%s). Display is handled by the EXRplayer panel."
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
        nuke.tprint("EXRplayer: cannot install the Viewer guard: %s" % exc)


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
            nuke.tprint("EXRplayer: cannot install the knob watch for %s: "
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
        nukescripts.registerWidgetAsPanel(
            "exrplayer.panel.PlayerPanel", "EXRplayer",
            "com.honza.EXRplayerPanel", create=True)
    except Exception as exc:
        nuke.tprint("EXRplayer: panel registration failed: %s" % exc)

    try:
        m = nuke.menu("Nuke").addMenu("EXRplayer")
        m.addCommand("Create EXRplayer Node",
                     "import exrplayer.register as r; r.create_node()")
        m.addCommand("Open EXRplayer Panel",
                     "import exrplayer.register as r; r.open_panel()")
    except Exception as exc:
        nuke.tprint("EXRplayer: menu failed: %s" % exc)

    try:
        nuke.menu("Nodes").addCommand(
            "EXRplayer/EXRplayer",
            "import exrplayer.register as r; r.create_node()")
    except Exception:
        pass
