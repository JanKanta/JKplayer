"""Registration into Nuke - menu + panel. Called from ~/.nuke/menu.py."""

import nuke

# IMPORTANT: nukescripts.registerWidgetAsPanel does not store the class, it
# builds the TEXT
#   "...WidgetKnob(jkplayer.panel.PlayerPanel)"
# and EVALUATES it later in Nuke's global namespace. For that to work,
# (a) the attribute `jkplayer.panel` has to exist (i.e. the submodule must
# already be imported) and (b) the name `jkplayer` has to be visible in
# __main__. Without it the panel works as a floating window but cannot be
# docked in Nuke.
from . import panel as _panel_module          # makes sure jkplayer.panel exists

# ONE PANEL PER NODE, kept by the node's own id (node.uid_of). Two panels on
# one node would write each other's settings over and fill two caches with the
# same frames, which is what following the SELECTION used to cause the moment
# there was more than one node in the script.
#
# AND A REGISTRATION OF ITS OWN. Nuke keys panels by the id given to
# registerWidgetAsPanel: that id is what the Pane menu entry restores and what
# a saved layout writes down. One shared id for every panel meant Nuke could
# not tell two of them apart, which is the real reason a second node made
# things go strange - our own binding was only half of it.
#
# So each node registers a panel named after the node (JKplayer, JKplayer1,
# JKplayer2) under an id built from the node's own uid, and the widget
# expression carries that uid - see panel.bound.
_panels = {}                     # uid -> the panel object we opened for it
_pane_entries = {}               # uid -> the name its Pane entry goes by


def _expose_in_main():
    """Expose the package in __main__, where the panel expression is evaluated."""
    try:
        import __main__
        import jkplayer
        setattr(__main__, "jkplayer", jkplayer)
    except Exception as exc:
        nuke.tprint("JKplayer: cannot expose in __main__: %s" % exc)


PANEL_ID = "com.honza.JKplayerPanel"
PANEL_NAME = "JKplayer"


def _new_pane_panel(name, uid):
    """A Nuke pane wrapping our widget.

    The same thing nukescripts.registerWidgetAsPanel builds, made here so it
    can be docked into a CHOSEN pane: the ready-made one only ever lands in
    nuke.thisPane(), which is set while a Pane menu entry is being run and
    nowhere else. Built here it also leaves the Pane menu alone - registering
    again just to get an instance would re-add the entry.
    """
    import nukescripts
    title = name or PANEL_NAME
    ident = panel_id_for(uid)
    # the widget is named as an EXPRESSION, evaluated later in __main__ -
    # which is what _expose_in_main is for - so the uid travels as text
    expr = "jkplayer.panel.bound(%r)" % uid
    panel = nukescripts.PythonPanel(title, ident)
    panel.addKnob(nuke.PyCustom_Knob(
        title, "",
        "__import__('nukescripts').panels.WidgetKnob(%s)" % expr))
    return panel


# Where a player can be docked when none is open yet, in the order tried.
# The Viewer first because that is where a picture is expected; the Node
# Graph and the Properties because one of them is on screen in any layout
# anybody actually works in.
DOCK_BESIDE = ("Viewer.1", "Viewer1", "DAG.1", "Properties.1")


def _dock_pane(uid=""):
    """The pane a player panel goes into, or None if Nuke has none to offer.

    NEXT TO ANOTHER PLAYER FIRST. Several nodes means several players, and
    they belong together as tabs of one pane rather than scattered one into
    each pane the layout happens to have.
    """
    ids = [panel_id_for(other) for other in list(_pane_entries)
           if other != uid]
    ids.extend(DOCK_BESIDE)
    for ident in ids:
        try:
            pane = nuke.getPaneFor(ident)
        except Exception:
            pane = None
        if pane is not None:
            return pane
    return None


def panel_id_for(uid):
    """The id Nuke knows this node's panel by."""
    return "%s.%s" % (PANEL_ID, uid)


def register_for_node(node):
    """Give the node a panel of its own in the Pane menu. Returns its uid.

    Idempotent: registering the same node twice would add the Pane entry
    twice, so what has been done is remembered in _pane_entries.
    """
    import nukescripts
    from . import node as nodemod
    uid = nodemod.ensure_unique_uid(node)
    if not uid:
        return ""
    try:
        name = node.name()
    except Exception:
        name = PANEL_NAME
    was = _pane_entries.get(uid)
    if was == name:
        return uid
    if was is not None and was != name:
        # THE NODE WAS RENAMED. Register again under the new name, but take
        # the old entry out first - otherwise the Pane menu grows one item per
        # rename and only the newest of them means anything.
        try:
            nuke.menu("Pane").removeItem(was)
        except Exception:
            pass
    _expose_in_main()
    # The widget is named as an expression evaluated later in __main__, so the
    # uid has to survive as TEXT - hence the repr rather than a reference.
    expr = "jkplayer.panel.bound(%r)" % uid
    try:
        nukescripts.registerWidgetAsPanel(expr, name, panel_id_for(uid))
        _pane_entries[uid] = name
    except Exception as exc:
        nuke.tprint("JKplayer: cannot register a panel for %s (%s)" % (name, exc))
        return ""
    return uid


def forget_node(uid):
    """Take the panel away when its node is gone - entry, registration, window."""
    if not uid:
        return
    name = _pane_entries.pop(uid, None)
    try:
        import nukescripts
        nukescripts.panels.unregisterPanel(panel_id_for(uid), None)
    except Exception:
        pass                                  # never registered, or gone already
    if name:
        try:
            nuke.menu("Pane").removeItem(name)
        except Exception:
            pass
    panel = _panels.pop(uid, None)
    if _alive(panel):
        try:
            panel.close()
        except Exception:
            pass


def _on_script_load():
    """Every node in the freshly opened script gets its Pane entry."""
    try:
        made = register_existing()
        if made:
            nuke.tprint("JKplayer: %d node panel(s) registered" % made)
    except Exception as exc:
        nuke.tprint("JKplayer: cannot register the panels (%s)" % exc)


def register_existing():
    """A panel entry for every node already in the script.

    Called after a script is loaded: the nodes come with their ids already in
    them, and without this their Pane entries would only appear for whatever
    was created by hand afterwards.
    """
    from . import node as nodemod
    made = 0
    for node in nodemod.find_all():
        if register_for_node(node):
            made += 1
    return made


def _alive(panel):
    """Is that panel still a live Qt object? Qt deletes them behind our back."""
    if panel is None:
        return False
    try:
        panel.isVisible()                     # touches the C++ side
        return True
    except Exception:
        return False


def _target_node():
    """Which node a newly opened panel should belong to.

    The selected one, or - when nothing is selected - the first in the script.
    A panel with no node has nothing to show, so it is better to guess than to
    open an empty one; which node it took is written in its status line.
    """
    from . import node as nodemod
    try:
        sel = [n for n in nuke.selectedNodes() if nodemod.is_player_node(n)]
    except Exception:
        sel = []
    if sel:
        return sel[0]
    found = nodemod.find_all()
    return found[0] if found else None


def _live_widget(uid):
    """The PlayerPanel on screen for this node, or None.

    Looked for among the widgets Qt actually has, rather than trusted from
    _panels. What _panels holds for a DOCKED panel is Nuke's PythonPanel
    wrapper, which is not a Qt widget and cannot be asked whether it is still
    there - asking it always failed, so every open looked like the first and
    another panel was stacked on top of the one already open.
    """
    if not uid:
        return None
    try:
        from .qtcompat import QtWidgets
        app = QtWidgets.QApplication.instance()
        widgets = app.allWidgets() if app is not None else ()
    except Exception:
        return None
    for w in widgets:
        try:
            if isinstance(w, _panel_module.PlayerPanel) \
                    and getattr(w, "_node_uid", None) == uid:
                w.isVisible()                 # still alive on the C++ side?
                return w
        except Exception:
            continue
    return None


def _bring_forward(widget):
    """Makes a panel the one you are looking at, docked or floating.

    Docked, it is a page in one of Nuke's tabbed panes, and raising the widget
    does nothing for a page behind another tab - so every tabbed pane it sits
    in is switched to the page it is on, from the inside out.
    """
    from .qtcompat import QtWidgets
    try:
        chain = []
        w = widget
        while w is not None:
            chain.append(w)
            w = w.parentWidget()
        for holder in chain:
            if isinstance(holder, QtWidgets.QTabWidget):
                for page in chain:
                    i = holder.indexOf(page)
                    if i >= 0:
                        holder.setCurrentIndex(i)
                        break
            elif isinstance(holder, QtWidgets.QStackedWidget):
                for page in chain:
                    if holder.indexOf(page) >= 0:
                        holder.setCurrentWidget(page)
                        break
        top = widget.window()
        top.raise_()
        top.activateWindow()
    except Exception:
        pass


def open_panel(force=False, node=None):
    """Opens a panel for one node, docked beside the Viewer where there is one.

    `node` names the one it is for; without it the selection decides. A node
    that already has a panel gets that one raised instead of a second - see
    _panels.

    `force` opens another panel even so. It exists for the case where the
    first one was closed in a way we could not see, and for a second view of
    the same node when somebody really wants it.
    """
    target = node if node is not None else _target_node()
    if target is None:
        # A PANEL BELONGS TO A NODE, always. One with no node behind it has
        # nothing to show and nowhere to keep its settings, and it was the
        # anonymous "any node" panel that kept turning up as the odd one out.
        nuke.message("There is no JKplayer node to open.\n\n"
                     "Create one first: JKplayer > Create JKplayer Node.")
        return None
    uid = ""
    try:
        uid = register_for_node(target)
    except Exception as exc:
        nuke.tprint("JKplayer: cannot identify the node (%s)" % exc)
    if not uid:
        return None

    if not force:
        live = _live_widget(uid)
        if live is not None:
            _bring_forward(live)
            return _panels.get(uid) or live
        _panels.pop(uid, None)

    # PENDING_UID is the belt to the braces: a docked panel gets its uid from
    # the expression it was registered with, but a floating one is built here
    # and the two paths should not disagree about which node it is.
    _panel_module.PENDING_UID = uid
    try:
        opened = _build_panel(target, uid)
    finally:
        _panel_module.PENDING_UID = None
    if opened is not None:
        _panels[uid] = opened
    return opened


def _build_panel(target, uid):
    """Docks the node's player into a pane. NEVER a floating window.

    There used to be a floating fallback for when no Viewer could be found.
    It is gone on purpose: a window that is not part of the layout does not
    come back with it, sits over the work, and is the one thing people kept
    asking to be rid of. If there is nowhere at all to dock, nothing opens and
    the Script Editor says why - that is a layout to fix, not a reason to
    float.
    """
    try:
        name = target.name()
    except Exception:
        name = PANEL_NAME

    pane = _dock_pane(uid)
    if pane is None:
        nuke.tprint("JKplayer: no pane to dock the %s panel into - open a "
                    "Viewer or the Node Graph and try again." % name)
        return None
    try:
        docked = _new_pane_panel(name, uid)
        docked.addToPane(pane)
        return docked
    except Exception as exc:
        nuke.tprint("JKplayer: could not dock the %s panel (%s)" % (name, exc))
        return None


def create_node():
    """Creates the node AND shows the panel - the node on its own displays
    nothing, so making one and seeing no picture is a puzzle, not a workflow."""
    from . import node as nodemod
    node = nodemod.create()
    try:
        open_panel(node=node)                 # the panel for THIS node
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


_viewer_keys_blocked = False


def _selected_for_viewer():
    """The node Nuke's number keys would put into a Viewer.

    The same choice nukescripts.connect_selected_to_viewer makes for itself -
    the first selected node that is not a Viewer - so what is blocked is
    exactly what would have been connected, and nothing else.
    """
    try:
        for node in nuke.selectedNodes(recursive=True):
            if node.Class() != "Viewer":
                return node
    except Exception:
        pass
    return None


def _install_viewer_key_block():
    """Number keys on a JKplayer node do nothing - no Viewer, made or wired.

    The Viewer guard already DISCONNECTS a Viewer from our node, but by then
    pressing 1 has created one: a Viewer node appears in the graph for no
    reason, and in a script with no Viewer it is a whole new one. Better that
    the key does nothing at all on this node.

    The keys run the text "nukescripts.connect_selected_to_viewer(N)", looked
    up when pressed, so wrapping that one function catches every one of them,
    number row and keypad alike. For any other node it is Nuke's own function,
    untouched.
    """
    global _viewer_keys_blocked
    if _viewer_keys_blocked:
        return
    try:
        import nukescripts
        original = nukescripts.connect_selected_to_viewer
    except Exception as exc:
        nuke.tprint("JKplayer: cannot guard the Viewer keys (%s)" % exc)
        return
    if getattr(original, "_jkplayer_guard", False):
        _viewer_keys_blocked = True
        return

    def connect_selected_to_viewer(inputIndex):
        from . import node as nodemod
        chosen = _selected_for_viewer()
        if chosen is not None and nodemod.is_player_node(chosen):
            nuke.tprint("JKplayer: %s is shown in its own panel - no Viewer "
                        "is made for it." % chosen.name())
            return None
        return original(inputIndex)

    connect_selected_to_viewer._jkplayer_guard = True
    connect_selected_to_viewer.__doc__ = original.__doc__
    nukescripts.connect_selected_to_viewer = connect_selected_to_viewer
    try:
        import nukescripts.misc as misc
        misc.connect_selected_to_viewer = connect_selected_to_viewer
    except Exception:
        pass
    _viewer_keys_blocked = True


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
        if name == "showPanel":
            _on_node_opened(nuke.thisNode())
            return
        if name == "name":
            _on_node_renamed(nuke.thisNode())
            return
        # what is visible on the node depends on the view mode and on which
        # panels are on - hence QC as well (it hides the QC mode selector)
        watched = ("cv_view_mode",) + tuple("cv_qc_%s" % s
                                            for s in nodemod.SLOT_LABELS)
        colour = ("cv_color_mgmt", "cv_ocio_config")
        if name not in colour and name not in watched:
            return
        n = nuke.thisNode()
        if not nodemod.is_player_node(n):
            return
        if name in colour:     # picking 'custom' shows the file field
            nodemod.apply_color_visibility(n)
        else:
            nodemod.apply_view_visibility(n)
    except Exception:
        pass


def _on_node_opened(node):
    """The node's Properties were opened - bring its player up with them.

    `showPanel` is what Nuke sends when a node is double-clicked open, so this
    is the same gesture that opens any node, and the player for THAT node is
    the one that comes up - raised if it is already open, made if it is not.

    Deferred to the next turn of the event loop: this runs inside Nuke's knob
    callback, which is no place to build a docked panel, and doing it there
    would open the player before the Properties it came with had finished
    opening.
    """
    from . import node as nodemod
    if not nodemod.is_player_node(node):
        return
    try:
        from .qtcompat import QtCore
        QtCore.QTimer.singleShot(0, lambda n=node: _open_for(n))
    except Exception:
        _open_for(node)


def _on_node_renamed(node):
    """The node got a new name - so do its Pane entry and its tab.

    JKplayer2's panel has to say JKplayer2. A tab still reading the old name
    after a rename is how two players get confused for each other, which is
    exactly what naming them after their nodes is meant to prevent.
    """
    from . import node as nodemod
    if not nodemod.is_player_node(node):
        return
    try:
        uid = register_for_node(node)          # swaps the Pane entry
        new = node.name()
    except Exception:
        return
    if uid:
        _retitle(_live_widget(uid), new)


def _retitle(widget, name):
    """Puts a new name on the tab a docked player sits in."""
    if widget is None:
        return
    from .qtcompat import QtWidgets
    try:
        chain = []
        w = widget
        while w is not None:
            chain.append(w)
            w = w.parentWidget()
        for holder in chain:
            if isinstance(holder, QtWidgets.QTabWidget):
                for page in chain:
                    i = holder.indexOf(page)
                    if i >= 0:
                        holder.setTabText(i, name)
                        break
    except Exception:
        pass


def _open_for(node):
    try:
        open_panel(node=node)
    except Exception as exc:
        nuke.tprint("JKplayer: cannot open the panel for %s (%s)"
                    % (getattr(node, "name", lambda: "?")(), exc))


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
    _install_viewer_key_block()
    _install_knob_watch()

    # NO GENERIC PANEL. There used to be a "JKplayer (any node)" entry among
    # the Custom panels, for opening a player with no node behind it. Every
    # player now belongs to a node and is listed under that node's name -
    # JKplayer, JKplayer1, JKplayer2 - so the anonymous one was only ever the
    # odd one out.

    try:
        m = nuke.menu("Nuke").addMenu("JKplayer")
        m.addCommand("Create JKplayer Node",
                     "import jkplayer.register as r; r.create_node()")
        # NOT force. It used to be, back when there was one panel for the whole
        # script and the entry had to be able to open a second after the first
        # was closed. Now the entry means "show me the panel for the node I
        # have selected", and a node that already has one gets that one raised.
        m.addCommand("Open JKplayer Panel",
                     "import jkplayer.register as r; r.open_panel()")
    except Exception as exc:
        nuke.tprint("JKplayer: menu failed: %s" % exc)

    try:
        nuke.menu("Nodes").addCommand(
            "JKplayer/JKplayer",
            "import jkplayer.register as r; r.create_node()")
    except Exception:
        pass

    # A SCRIPT THAT IS OPENED brings its nodes with it, and each of them wants
    # its own Pane entry. Without this only nodes made by hand afterwards would
    # have one, and opening yesterday's script would offer nothing to open.
    try:
        nuke.addOnScriptLoad(_on_script_load)
    except Exception as exc:
        nuke.tprint("JKplayer: cannot watch for script loads (%s)" % exc)
