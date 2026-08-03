"""
The EXRplayer node - an anchor in the Node Graph plus ALL THE SETTINGS.

Deliberately in Python, not C++: the node no longer touches pixels (we read the
files ourselves), so it is just a settings holder. Python = no compilation, no
DLL locked while Nuke runs, instant changes.

Rules (the v2 brief):
  * two inputs A and B, and ONLY Read nodes with .exr (anything else is
    disconnected)
  * both inputs must have THE SAME frame range (otherwise an error and B is
    disconnected)
  * a Viewer node cannot be attached
  * no nuke.execute - the node renders nothing

WHY A GROUP AND NOT NoOp: NoOp has a single input and the input count cannot be
set from Python. A Group has as many as there are Input nodes inside - so two
Inputs named A and B give exactly what we need, and the innards stay hidden
(nuke.allNodes does not return them). Nodes from earlier versions are NoOp and
are recognised by their tag, so they keep working - just with a single input.
"""

import nuke

from . import effects
from . import exrcore
from . import nukelut
from . import ocio
from .paths import total_ram_mb

NODE_TAG = "cv_is_exrplayer"          # hidden knob, this is how we recognise the node
# The tag used before the rename to EXRplayer. Nodes saved by an older version
# still carry it, so they are recognised as well and keep all their settings.
LEGACY_TAGS = ("cv_is_compareviewer",)
DEFAULT_NAME = "EXRplayer"

# Group = today's node (multiple inputs), NoOp = nodes from earlier versions.
# Both are recognised by their tag, so an old script opens and keeps working.
NODE_CLASSES = ("Group", "NoOp")

INPUT_LABELS = ("A", "B")            # image inputs, the windows pick between them

# The third input carries mattes in its RGBA channels. It is not an image, so a
# window does not pick it as a source - it is only used in DiMatte mode to draw
# a coloured overlay over what is visible.
MATTE_LABEL = "DiMatte"
MATTE_INPUT = len(INPUT_LABELS)      # input index
ALL_INPUTS = INPUT_LABELS + (MATTE_LABEL,)

# Matte channels and their colours - the same family as the difference, so it
# feels like one tool in the image.
MATTE_CHANNELS = ("r", "g", "b", "a")
MATTE_LABELS = ("R", "G", "B", "A")

# Windows in the panel. Each one picks WHICH input it shows - so both can show
# the same input and differ only by layer (rgba against depth of the same plate).
SLOT_LABELS = ("1", "2")

VIEW_SINGLE, VIEW_DOUBLE, VIEW_WIPE, VIEW_DIMATTE = range(4)
VIEW_MODES = ["Single", "Double", "Wipe", "DiMatte"]

SPLIT_SIDE, SPLIT_STACK = range(2)
SPLIT_MODES = ["Side by side", "Stacked"]

MGMT_NUKE, MGMT_OCIO = range(2)      # the order of the items in cv_color_mgmt

# Panels in the image. Every window has its own, the knob is cv_<key>_<window>.
# ONE checkbox per panel: an enabled panel is both computed and shown, a
# disabled one neither. There used to be two checkboxes (compute / show) and
# there was no telling which one applied - the button in the image toggled both
# anyway.
# Everything is OFF by default: the panel opens on a clean image in Single and
# you switch on what you need. CC on at startup would push a colour correction
# into the image and QC on would push some check mode straight away.
PANELS = (
    ("cc", "CC", False,
     "Gain / gamma / saturation.\n"
     "OFF = the image goes through with no colour correction (and saturation\n"
     "no longer costs 20 ms a frame)."),
    ("qc", "QC", False,
     "Check mode (grain, saturation, canvas, value map, temporal).\n"
     "OFF = the ordinary image is shown, even when a mode is selected."),
    ("hist", "Histogram", False,
     "Axis 0 to 55, the line at 1.0 marks the clipping boundary."),
    ("vscope", "Vectorscope", False,
     "Pixel colour as it is on screen (angle = hue, radius = saturation)."),
    ("wave", "Waveform", False,
     "Values along the image columns (0 at the bottom, 55 at the top, line at 1.0)."),
)

# Panels available only in Single (greyed out in the image, hidden on the
# node). A test checks that this matches overlay.SCOPE_KEYS - overlay
# deliberately does not import node, so it stays free of any nuke dependency.
SCOPE_KEYS = ("hist", "vscope", "wave")

EFFECT_TIP = (
    "Check display (keys 1-%d in the panel). Switched off by the\n"
    "QC checkbox above, not by an item in the list:\n"
    "  Difference - difference of inputs A and B (overlay or plain)\n"
    "  Grain      - shows the grain, suppresses edges by normalisation\n"
    "  High-pass  - keeps only detail in the given size band;\n"
    "               better than Grain for paint fixes and texture changes\n"
    "  Saturation - levels the brightness, only colour stays\n"
    "  Canvas     - swaps the quadrants, edges meet in the middle\n"
    "  Value map  - false colours from the scene-linear values\n"
    "  Temporal   - difference against the previous frame; reports\n"
    "               duplicate and almost identical frames"
    % len(effects.ORDER))

# Defaults that cannot be derived from the project:
#   FPS 24 instead of "follow the project" - QC is done on what gets delivered,
#   and that is 24, even when the script currently open says something else.
#   Nuke instead of OCIO - the built-in transforms are 3.5x faster and are
#   plenty for checking; OCIO gets switched on when exact display matters.
DEFAULT_FPS = 24.0
DEFAULT_SCOPE_OPACITY = 0.75         # 190/255, how the panel backdrop looked so far


def default_cache_mb():
    """Default cache size = a quarter of the machine's RAM (min 2 GB, max 64 GB).

    A quarter because Nuke and the system need something too. On a 96 GB
    machine that comes out at 24 GB = ~165 frames of 6K or ~1500 of 1080p.
    """
    total = total_ram_mb()
    if total <= 0:
        return 4096
    return int(max(2048, min(65536, total // 4)))


# ---------------------------------------------------------------------------
# Settings definition: (knob_name, label, type, default, min, max, tooltip)
# ---------------------------------------------------------------------------
def _add_knobs(node):
    """Adds the knobs the node does not have yet.

    It is IDEMPOTENT on purpose, so it can be run on an existing node and fill
    in knobs from a newer version - see ensure_knobs(). Without that a node
    saved by an older version would silently lose settings: writing to a
    missing knob throws, the watcher then reads nothing and puts the choice
    back to its default.
    """
    order = []                    # the order the knobs belong in on the node

    def add(k):
        order.append(k.name())
        if k.name() not in node.knobs():
            node.addKnob(k)

    def add_hidden(k):
        """A knob that holds state but is not shown on the node.

        The INVISIBLE flag alone is not enough - knob.visible() still reports
        True, and on an old node where the knob already exists the flag would
        not apply at all. So setVisible(False) is called on the one actually
        on the node as well.
        """
        k.setFlag(nuke.INVISIBLE)
        add(k)
        try:
            node[k.name()].setVisible(False)
        except Exception:
            pass

    # CAREFUL: the first knob added MUST be a Tab_Knob, otherwise Nuke creates
    # its own "User" tab and puts our knobs in it.
    add(nuke.Tab_Knob("cv_viewer_tab", "Viewer"))

    tag = nuke.Boolean_Knob(NODE_TAG, "")
    tag.setValue(True)
    tag.setFlag(nuke.INVISIBLE)
    add(tag)

    # ---- double view ----
    # The view mode comes first: it decides what else on the node makes sense
    # at all. Here and at the top of the panel - both sides follow each other.
    k = nuke.Enumeration_Knob("cv_view_mode", "View", VIEW_MODES)
    k.setTooltip("Single = one window (always window 1), Double = both.\n"
                 "The same control is at the top left of the panel.")
    add(k)

    add(nuke.Text_Knob("cv_matte_head", "DiMatte - mattes over the image"))
    for ch, label in zip(MATTE_CHANNELS, MATTE_LABELS):
        k = nuke.Boolean_Knob("cv_matte_%s" % ch, label)
        k.setValue(False)                 # start on a clean image
        k.setFlag(nuke.STARTLINE if ch == MATTE_CHANNELS[0] else 0)
        k.setTooltip("Overlay the image with the matte from channel %s of the\n"
                     "%s input. The RGBA toggles in the image do the same."
                     % (label, MATTE_LABEL))
        add(k)

    for name, label, default, lo, hi, tip in (
            ("cv_matte_light", "Matte lightness", 1.0, 0.0, 1.0,
             "Lightness of the colour the matte is drawn in.\n"
             "The matte is mixed by its own value, so the transitions\n"
             "stay - a blurred edge is blurred in the overlay too."),
            ("cv_matte_gain", "Matte gain", 1.0, 0.0, 8.0,
             "Matte gain. Pulls a weak matte up without touching\n"
             "its shape."),
            ("cv_matte_gamma", "Matte gamma", 1.0, 0.1, 4.0,
             "The shape of the matte transition. Below 1 the edge is\n"
             "harder, above 1 softer. The matte boundaries stay, only\n"
             "the ramp changes.")):
        k = nuke.Double_Knob(name, label)
        k.setValue(default)
        k.setRange(lo, hi)
        k.setTooltip(tip)
        add(k)

    k = nuke.Double_Knob("cv_wipe_opacity", "Wipe opacity")
    k.setValue(1.0)
    k.setRange(0.0, 1.0)
    k.setTooltip("Opacity of input B in Wipe mode.\n"
                 "1 = a hard reveal at the line, lower blends B into A.\n"
                 "The same slider is in the image below the B controls.")
    add(k)

    k = nuke.Enumeration_Knob("cv_split", "Image split", SPLIT_MODES)
    k.setTooltip("How the window is split in Double.\n"
                 "It is always split exactly in half, so the divider stays in\n"
                 "the middle even when the panel is resized to another aspect.\n"
                 "Zoom and pan are shared - the pixels line up.")
    add(k)

    # The input and layer of each window are picked with a toggle right in the
    # image. On the node they are therefore HIDDEN - only so the choice
    # survives saving the script. Showing them would be pointless: they could
    # not be changed and would only confuse.
    for i, label in enumerate(SLOT_LABELS):
        k = nuke.Enumeration_Knob("cv_source_%s" % label, "Window %s" % label,
                                  list(INPUT_LABELS))
        k.setValue(min(i, len(INPUT_LABELS) - 1))     # window 1 = A, window 2 = B
        add_hidden(k)

    # EVERY WINDOW has its own panels - in Double two different images are
    # being compared and each deserves its own scope and its own check mode.
    for slot in SLOT_LABELS:
        add(nuke.Text_Knob("cv_panels_head_%s" % slot,
                           "Panels of window %s" % slot))

        # One checkbox per panel - the same ones the CC/QC/H/V/W toggles switch
        # right in the image. An enabled panel is both computed and shown.
        for key, label, default, tip in PANELS:
            k = nuke.Boolean_Knob("cv_%s_%s" % (key, slot), label)
            k.setValue(default)
            k.setFlag(nuke.STARTLINE)
            k.setTooltip("%s\n\nThe %s toggle in the image does the same."
                         % (tip, label))
            add(k)

        # Opacity belongs with the scopes, so it sits right below them. It is
        # shared and the scopes only exist in Single, so window 1 is enough.
        if slot == SLOT_LABELS[0]:
            k = nuke.Double_Knob("cv_scope_opacity", "Scope opacity")
            k.setValue(DEFAULT_SCOPE_OPACITY)
            k.setRange(0.0, 1.0)
            k.setTooltip("Opacity of the histogram and vectorscope backdrop.\n"
                         "1 = opaque, 0 = just outlines over the image.")
            add(k)

        k = nuke.Enumeration_Knob("cv_effect_%s" % slot, "QC mode",
                                  [effects.LABELS[e] for e in effects.ORDER])
        k.setTooltip(EFFECT_TIP)
        add(k)

    # Panel size is dragged with the mouse by their edge right in the image, so
    # there is no slider here. Nor is there a pixel readout - that lives in the
    # bottom bar of the panel next to the FPS.

    # ---- Playback ----
    add(nuke.Tab_Knob("cv_play_tab", "Playback"))

    k = nuke.Double_Knob("cv_fps", "FPS")
    k.setValue(DEFAULT_FPS)
    k.setRange(0, 120)
    k.setTooltip("Target playback FPS. 0 = use the project FPS.")
    add(k)

    k = nuke.Enumeration_Knob("cv_loop", "At the end",
                              ["loop", "ping-pong", "stop"])
    k.setTooltip("What to do after reaching the end of the range.")
    add(k)

    k = nuke.Boolean_Knob("cv_realtime", "Realtime (drop frames)")
    k.setValue(True)
    k.setFlag(nuke.STARTLINE)
    k.setTooltip("ON:  holds the target FPS, skips uncached frames.\n"
                 "OFF: plays every frame, even at a lower FPS.")
    add(k)

    k = nuke.Boolean_Knob("cv_play_cached_only", "Play cached only")
    k.setValue(False)
    k.setFlag(nuke.STARTLINE)
    k.setTooltip("Playback stays inside the cached region (like RV).")
    add(k)

    # ---- Cache ----
    add(nuke.Tab_Knob("cv_cache_tab", "Cache"))

    ram = total_ram_mb()
    k = nuke.Int_Knob("cv_cache_mb", "Cache memory (MB)")
    k.setValue(default_cache_mb())
    k.setRange(256, 65536)
    k.setTooltip("How much RAM the EXRplayer panel may use for frames.\n"
                 "Default = a quarter of the machine's RAM%s.\n\n"
                 "Memory per frame (RGBA half):\n"
                 "  1080p ~16 MB   |   4K ~66 MB   |   6K ~148 MB"
                 % (" (%d MB)" % ram if ram else ""))
    add(k)

    k = nuke.Int_Knob("cv_lookahead", "Look-ahead (frames)")
    k.setValue(24)
    k.setRange(0, 200)
    k.setTooltip("How many frames ahead of the playhead are pre-fetched in the\n"
                 "playback direction.")
    add(k)

    k = nuke.Int_Knob("cv_behind", "Look-behind (frames)")
    k.setValue(4)
    k.setRange(0, 50)
    k.setTooltip("How many frames behind the playhead to keep (scrubbing back "
                 "and forth).")
    add(k)

    k = nuke.Int_Knob("cv_workers", "Decoding threads")
    k.setValue(4)
    k.setRange(1, 32)
    k.setTooltip("Number of threads decoding EXR.\n"
                 "More is NOT more: measured 4 threads = 115 fps, 8 threads =\n"
                 "100 fps, 12 threads = 92 fps (memory bandwidth is the limit,\n"
                 "not the core count).\n"
                 "The change takes effect after reopening the panel.")
    add(k)

    k = nuke.Boolean_Knob("cv_auto_cache", "Cache the whole range automatically")
    k.setValue(True)
    k.setFlag(nuke.STARTLINE)
    k.setTooltip("Once a sequence is attached, start filling the whole range "
                 "in the background.")
    add(k)

    # ---- Color management ----
    add(nuke.Tab_Knob("cv_color_tab", "Color management"))

    # The same split as Project Settings in Nuke: either the built-in
    # transforms (fast, one table) or OCIO (exact, per the pipeline). Only the
    # relevant part is shown - see apply_color_visibility().
    k = nuke.Enumeration_Knob("cv_color_mgmt", "Color management",
                              ["Nuke", "OCIO"])
    k.setValue(MGMT_NUKE)
    k.setTooltip("Nuke - built-in transforms, 3.5x faster display.\n"
                 "OCIO  - per the studio config.\n"
                 "The display and the input space are picked at the top of the "
                 "panel.")
    add(k)

    configs = ocio.find_configs() if ocio.available() else []
    k = nuke.Enumeration_Knob("cv_ocio_config", "Config",
                              [c[0] for c in configs] or ["(none found)"])
    if configs:
        k.setValue(ocio.default_config_index(configs))    # nuke-default
    k.setTooltip("A config from the OCIO variable or from the Nuke installation.")
    add(k)

    # The display and the view are kept as TEXT, not as an enumeration - their
    # list depends on the config and would change under our hands; a name
    # survives a config change. Empty = use the config default.
    for name, label, default in (
            ("cv_ocio_input", "Input space", ""),
            ("cv_ocio_display", "Display", ""),
            ("cv_ocio_view", "View", ""),
            ("cv_nuke_display", "Display", nukelut.DEFAULT_DISPLAY),
            ("cv_nuke_input", "Input space", nukelut.DEFAULT_INPUT)):
        k = nuke.String_Knob(name, label, default)
        k.setEnabled(False)                   # set from the panel
        add(k)

    add(nuke.Text_Knob("cv_chan_div", ""))

    # The layer is kept as TEXT - the list depends on the file, not on the
    # node. Each window has its own: comparing rgba against depth is a
    # legitimate use.
    for label in SLOT_LABELS:
        k = nuke.String_Knob("cv_layer_%s" % label, "Layer of window %s" % label,
                             exrcore.ROOT_LAYER)
        add_hidden(k)                         # picked in the image, see above

    k = nuke.Enumeration_Knob("cv_channels", "Channels",
                              ["RGB", "R", "G", "B", "A", "Luminance"])
    k.setTooltip("Which channels to show. Switches instantly.")
    add(k)

    # The QC mode is per window - see cv_effect_<window> in the Viewer tab. The
    # settings of the individual modes (grain contrast, canvas shift, ...) live
    # right in the image in a semi-transparent panel, each mode with its own
    # sliders.

    # ---- info ----
    add(nuke.Text_Knob("cv_info_div", ""))
    info = nuke.Text_Knob("cv_info", "",
                          "Display is handled by the panel:  "
                          "EXRplayer > Open EXRplayer Panel")
    add(info)
    apply_color_visibility(node)
    apply_view_visibility(node)
    return order


# knobs belonging to the individual colour management modes
OCIO_KNOBS = ("cv_ocio_config", "cv_ocio_input", "cv_ocio_display",
              "cv_ocio_view")
NUKE_KNOBS = ("cv_nuke_display", "cv_nuke_input")


def slot_knobs(slot):
    """The knobs belonging to one window (apart from the hidden ones)."""
    return (["cv_panels_head_%s" % slot, "cv_effect_%s" % slot]
            + ["cv_%s_%s" % (key, slot) for key, _l, _d, _t in PANELS])


def apply_view_visibility(node):
    """Shows only what makes sense in the current state.

    A setting that cannot change anything is worse than no setting at all - a
    person toggles it and wonders why nothing happened. Hence:
      * the image split only in Double (in Single there is nothing to split)
      * window 2 only in Double
      * the histogram and the vectorscope only in Single (in Double they are
        not available, see overlay.SCOPE_KEYS)
      * the QC mode only when the QC panel is on
    """
    try:
        mode = int(node["cv_view_mode"].getValue())
    except Exception:
        return
    both = mode in (VIEW_DOUBLE, VIEW_WIPE)   # both windows have something to do
    # DiMatte is like Single: one window, only with mattes over it

    def show(name, want):
        if name in node.knobs():
            node[name].setVisible(bool(want))

    show("cv_split", mode == VIEW_DOUBLE)     # a wipe is not split, it overlaps
    show("cv_wipe_opacity", mode == VIEW_WIPE)
    show("cv_scope_opacity", not both)        # belongs to the scopes, Single only
    show("cv_matte_head", mode == VIEW_DIMATTE)
    for name in ("cv_matte_light", "cv_matte_gain", "cv_matte_gamma"):
        show(name, mode == VIEW_DIMATTE)
    for ch in MATTE_CHANNELS:
        show("cv_matte_%s" % ch, mode == VIEW_DIMATTE)
    for i, slot in enumerate(SLOT_LABELS):
        used = both or i == 0
        show("cv_panels_head_%s" % slot, used)
        for key, _l, _d, _t in PANELS:
            show("cv_%s_%s" % (key, slot),
                 used and not (both and key in SCOPE_KEYS))
        try:
            qc_on = bool(node["cv_qc_%s" % slot].value())
        except Exception:
            qc_on = True
        show("cv_effect_%s" % slot, used and qc_on)


def apply_color_visibility(node):
    """Shows only the knobs of the selected mode."""
    try:
        ocio_mode = int(node["cv_color_mgmt"].getValue()) == MGMT_OCIO
    except Exception:
        return
    for name in OCIO_KNOBS:
        if name in node.knobs():
            node[name].setVisible(ocio_mode)
    for name in NUKE_KNOBS:
        if name in node.knobs():
            node[name].setVisible(not ocio_mode)


# ---------------------------------------------------------------------------
def create():
    """Creates an EXRplayer node and returns it.

    `nuke.createNode` (unlike `nuke.nodes.Group()`) behaves like an ordinary
    Nuke node: it attaches to the selected node and is placed right below it.

    The inputs are made by two Input nodes inside the group. They are named A
    and B, so Nuke labels the arrows in the Node Graph that way too. Output is
    connected to A only so the group is not "empty" - nothing is ever rendered
    through it.
    """
    node = nuke.createNode("Group", inpanel=False)
    try:
        node.setName(DEFAULT_NAME)
    except Exception:
        pass                                  # the name exists -> Nuke adds a number
    _build_inputs(node)
    _add_knobs(node)
    node["tile_color"].setValue(0x7A3FBFFF)   # purple, so it stands out
    node["label"].setValue("[value cv_channels]")
    return node


def _inner_inputs(node):
    """{input number: Input node} inside the group."""
    out = {}
    try:
        inner = node.nodes()
    except Exception:
        return out                            # not a Group (an old NoOp)
    for k in inner:
        try:
            if k.Class() == "Input":
                out[int(k["number"].value())] = k
        except Exception:
            continue
    return out


def _build_inputs(node):
    """Adds the missing Input nodes inside the group. Returns their names.

    This is done on an existing node too (see ensure_inputs): a Group with no
    Input nodes has zero inputs and there is nowhere to attach a Read to it in
    the Node Graph.
    """
    if node.Class() != "Group":
        return []
    added = []
    node.begin()
    try:
        have = _inner_inputs(node)
        for i, label in enumerate(ALL_INPUTS):
            if i in have:
                continue
            k = nuke.createNode("Input", inpanel=False)
            k["number"].setValue(i)
            try:
                k.setName(label)
            except Exception:
                pass                          # name taken, the number is enough
            have[i] = k
            added.append(label)
        # Output only so the group is not empty - nothing renders through it
        if not [k for k in node.nodes() if k.Class() == "Output"]:
            out = nuke.createNode("Output", inpanel=False)
            out.setInput(0, have.get(0))
    finally:
        node.end()
    return added


def ensure_inputs(node):
    """Tops up the input count on an existing node."""
    if not is_player_node(node) or node.Class() != "Group":
        return []
    try:
        added = _build_inputs(node)
        if added:
            nuke.tprint("EXRplayer: added inputs on %s: %s"
                        % (node.name(), ", ".join(added)))
        return added
    except Exception as exc:
        nuke.tprint("EXRplayer: cannot add the inputs (%s)" % exc)
        return []


# Knobs left on the node by earlier versions that no longer do anything.
# They have to be REMOVED, not just ignored: the node would still show
# checkboxes that toggle nothing and there would be no telling which ones apply.
OBSOLETE_KNOBS = tuple(
    # panels used to be shared, today every window has its own (_<window>)
    ["cv_panels_head", "cv_panels_hint", "cv_effect",
     "cv_hist_scale", "cv_vscope_scale",
     "cv_cc_active", "cv_show_cc", "cv_qc_active", "cv_show_qc",
     "cv_hist_active", "cv_show_hist", "cv_vscope_active", "cv_show_vscope",
     # the layer used to be per input, today it is per window (cv_layer_1 / _2)
     "cv_layer", "cv_layer_b",
     # the active input is no longer kept on the node, it is just panel state
     "cv_active_input", "cv_matte_threshold"]
    # the per-window versions with a split "compute / show" and size sliders;
    # today a panel has one checkbox and the size is dragged with the mouse
    + ["cv_%s_active_%s" % (key, slot)
       for slot in SLOT_LABELS for key, _l, _d, _t in PANELS]
    + ["cv_show_%s_%s" % (key, slot)
       for slot in SLOT_LABELS for key, _l, _d, _t in PANELS]
    + ["cv_%s_scale_%s" % (name, slot)
       for slot in SLOT_LABELS for name in ("hist", "vscope")]
    + ["cv_panels_hint_%s" % slot for slot in SLOT_LABELS])


def prune_knobs(node):
    """Removes knobs that no longer mean anything. Returns their names."""
    removed = []
    for name in OBSOLETE_KNOBS:
        if name not in node.knobs():
            continue
        try:
            node.removeKnob(node[name])
            removed.append(name)
        except Exception:
            continue                          # will not go? at least do not crash
    if removed:
        nuke.tprint("EXRplayer: removed obsolete knobs on %s: %s"
                    % (node.name(), ", ".join(removed)))
    return removed


def ensure_order(node):
    """Puts the knobs back in today's order. Returns True when it touched anything.

    Nuke adds a new knob at the END, so a node from an earlier version is
    ordered by how the knobs were added historically - "View" then ends up
    somewhere at the bottom. It is fixed by removing our knobs and creating
    them again in the right order; the values are copied out first and put back
    afterwards.
    """
    try:
        want = [n for n in _add_knobs(node) if n in node.knobs()]
        have = [n for n in node.knobs() if n.startswith("cv_")]
        if have == want:
            return False

        values = {}
        for name in have:
            try:
                values[name] = node[name].value()
            except Exception:
                pass                      # Text_Knob and friends have no value
        for name in have:
            try:
                node.removeKnob(node[name])
            except Exception:
                pass
        _add_knobs(node)
        for name, value in values.items():
            try:
                node[name].setValue(value)
            except Exception:
                pass
        apply_color_visibility(node)
        apply_view_visibility(node)
        nuke.tprint("EXRplayer: knob order fixed on %s" % node.name())
        return True
    except Exception as exc:
        nuke.tprint("EXRplayer: cannot fix the knob order (%s)" % exc)
        return False


def ensure_knobs(node):
    """Adds knobs from a newer version onto an already existing node.

    Otherwise a node saved by an older version silently loses settings: the
    panel writes into a missing knob (that throws and is swallowed), the
    watcher then reads nothing and puts the choice back to its default - so,
    for instance, the OCIO input space could not be switched away from linear.
    """
    try:
        before = set(node.knobs())
        _add_knobs(node)
        added = set(node.knobs()) - before
        if added:
            nuke.tprint("EXRplayer: added knobs on %s: %s"
                        % (node.name(), ", ".join(sorted(added))))
        return added
    except Exception as exc:
        nuke.tprint("EXRplayer: cannot add the knobs (%s)" % exc)
        return set()


def is_player_node(node):
    """Is this one of ours? Nodes from before the rename carry a legacy tag."""
    try:
        if node is None:
            return False
        knobs = node.knobs()
        return NODE_TAG in knobs or any(tag in knobs for tag in LEGACY_TAGS)
    except Exception:
        return False


def find_all():
    out = []
    for cls in NODE_CLASSES:
        out.extend(n for n in nuke.allNodes(cls) if is_player_node(n))
    return out


def input_count(node):
    """How many inputs the node has. Nodes from earlier versions (NoOp) have one."""
    try:
        return max(1, min(len(ALL_INPUTS), node.maxInputs()))
    except Exception:
        return 1


def settings(node):
    """Reads the settings off the node into a dict (with sensible fallbacks)."""
    def val(name, default):
        try:
            return node[name].value()
        except Exception:
            return default

    def enum(name, default=0):
        """CAREFUL: Enumeration_Knob.value() returns the item TEXT, not the
        index! The index comes from getValue()."""
        try:
            return int(node[name].getValue())
        except Exception:
            return int(default)

    fps = float(val("cv_fps", DEFAULT_FPS))
    if fps <= 0:                              # 0 = follow the project
        try:
            fps = float(nuke.root().fps()) or 24.0
        except Exception:
            fps = 24.0
    out = {
        "cache_mb": int(val("cv_cache_mb", 4096)),
        "lookahead": int(val("cv_lookahead", 24)),
        "behind": int(val("cv_behind", 4)),
        "workers": int(val("cv_workers", 8)),
        "auto_cache": bool(val("cv_auto_cache", True)),
        "fps": fps,
        "loop": enum("cv_loop", 0),              # 0 loop, 1 ping-pong, 2 stop
        "realtime": bool(val("cv_realtime", True)),
        "cached_only": bool(val("cv_play_cached_only", False)),
        # the layer and source of each window (see SLOT_LABELS)
        "layers": tuple(str(val("cv_layer_%s" % s, exrcore.ROOT_LAYER))
                        or exrcore.ROOT_LAYER for s in SLOT_LABELS),
        "sources": tuple(enum("cv_source_%s" % s, min(i, 1))
                         for i, s in enumerate(SLOT_LABELS)),
        "view_mode": enum("cv_view_mode", VIEW_SINGLE),
        "split": enum("cv_split", SPLIT_SIDE),
        "channels": enum("cv_channels", 0),      # 0 RGB,1 R,2 G,3 B,4 A,5 Luma
        "color_mgmt": enum("cv_color_mgmt", MGMT_NUKE),
        "ocio_config": enum("cv_ocio_config", ocio.default_config_index()),
        "ocio_input": str(val("cv_ocio_input", "")),
        "ocio_display": str(val("cv_ocio_display", "")),
        "ocio_view": str(val("cv_ocio_view", "")),
        "nuke_display": str(val("cv_nuke_display", nukelut.DEFAULT_DISPLAY)),
        "nuke_input": str(val("cv_nuke_input", nukelut.DEFAULT_INPUT)),
        "scope_opacity": float(val("cv_scope_opacity", DEFAULT_SCOPE_OPACITY)),
        "wipe_opacity": float(val("cv_wipe_opacity", 1.0)),
        "matte": tuple(bool(val("cv_matte_%s" % ch, False))
                       for ch in MATTE_CHANNELS),
        "matte_light": float(val("cv_matte_light", 1.0)),
        "matte_gain": float(val("cv_matte_gain", 1.0)),
        "matte_gamma": float(val("cv_matte_gamma", 1.0)),
        # panels per window: the value is a tuple, one item per window
        "effect": tuple(enum("cv_effect_%s" % s, 0) for s in SLOT_LABELS),
    }
    for key, _label, default, _tip in PANELS:
        out[key] = tuple(bool(val("cv_%s_%s" % (key, s), default))
                         for s in SLOT_LABELS)
    return out


# ---------------------------------------------------------------------------
# Input policing: only a Read with .exr, no Viewer
# ---------------------------------------------------------------------------
def enforce_no_viewer():
    """Disconnects every Viewer that has an EXRplayer node on its input.

    MIND THE DIRECTION: when a user "attaches a Viewer to the node", it is the
    VIEWER that has our node on its input (Viewer.input(0) = EXRplayer). Our
    node knows nothing about it in its own inputs - which is why it has to be
    searched for this way, downstream. Returns the names of the disconnected
    Viewers.
    """
    disconnected = []
    try:
        viewers = nuke.allNodes("Viewer")
    except Exception:
        return disconnected
    for viewer in viewers:
        try:
            for i in range(viewer.inputs()):
                if is_player_node(viewer.input(i)):
                    viewer.setInput(i, None)
                    disconnected.append(viewer.name())
        except Exception:
            continue
    return disconnected


def _frame_range(src):
    try:
        return int(src.firstFrame()), int(src.lastFrame())
    except Exception:
        return None


def _enforce_one(node, index):
    """Checks one input. Returns the text of the problem, or None."""
    label = ALL_INPUTS[index] if index < len(ALL_INPUTS) else str(index)
    try:
        src = node.input(index)
    except Exception:
        return None
    if src is None:
        return None

    cls = src.Class()
    if cls == "Viewer":
        node.setInput(index, None)
        return "A Viewer node cannot be attached."
    if cls != "Read":
        node.setInput(index, None)
        return ("Input %s: only a Read node can be attached (this is %s). "
                "EXRplayer reads EXR files directly and does not work with "
                "other nodes." % (label, cls))
    try:
        path = src["file"].value()
    except Exception:
        path = ""
    if not path.lower().endswith(".exr"):
        node.setInput(index, None)
        return "Input %s: the Read must point at an .exr sequence." % label
    return None


def enforce_input(node):
    """Disconnects a disallowed input. Returns the problem text, or None if fine.

    Besides the input type it also checks that both have THE SAME frame range.
    Without that the playhead would point at different frames on the two sides
    and the comparison would lie - so B is disconnected and an error reported.
    """
    if not is_player_node(node):
        return None
    for i in range(input_count(node)):
        problem = _enforce_one(node, i)
        if problem:
            return problem

    if input_count(node) < 2:
        return None
    try:
        a, b = node.input(0), node.input(1)
    except Exception:
        return None
    if a is None or b is None:
        return None
    ra, rb = _frame_range(a), _frame_range(b)
    if ra is None or rb is None or ra == rb:
        return None
    node.setInput(1, None)
    return ("Input B disconnected: both inputs must have the same frame range "
            "(A %d-%d, B %d-%d)." % (ra[0], ra[1], rb[0], rb[1]))
