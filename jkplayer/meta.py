"""
Which header fields the META panel shows, and in what order.

No Qt and no Nuke, so it can be tested from the console. The two halves are:

  * the CHOICE - a list of (input tag, key) pairs, stored on the node as text
    so it is saved with the script and readable in the .nk
  * the HARVEST - whatever the headers of the current frame happen to hold,
    which the panel refreshes as the playhead moves

They are kept apart on purpose. A key that is chosen but missing from this
frame simply does not draw; it is not dropped from the choice, because the
next shot may well have it and re-ticking things every time you change plate
would make the feature not worth using.
"""

import threading

# What to show before anyone has chosen anything. An empty panel on first use
# says nothing about whether the feature works, and these are the fields a
# review actually starts from.
DEFAULTS = ("timeCode", "framesPerSecond", "cameraModel", "reelName",
            "nuke/version", "compression")

SEP = ":"                       # tag:key on one line

# "I want nothing shown" and "I have never touched this" are different answers
# and an empty knob cannot say both. Without this the None button did nothing:
# it wrote an empty string, which was read back as "not configured" and put the
# defaults straight back on screen.
NONE = "-none-"


def format_order(pairs):
    """[(tag, key)] -> the text stored on the node."""
    pairs = list(pairs)
    if not pairs:
        return NONE
    return "\n".join("%s%s%s" % (tag, SEP, key) for tag, key in pairs)


def parse_order(text):
    """The text on the node -> [(tag, key)], or None when never chosen."""
    text = (text or "").strip()
    if not text:
        return None
    if text == NONE:
        return []
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or SEP not in line:
            continue
        tag, key = line.split(SEP, 1)
        tag, key = tag.strip(), key.strip()
        if tag and key and (tag, key) not in out:
            out.append((tag, key))
    return out


def default_order(rows):
    """A first choice made from what this frame actually has.

    In DEFAULTS order rather than file order: those are listed by how much a
    review cares about them, and the file's own grouping is a different thing
    (it is what the writing tool thought, which is why the harvest keeps it).
    """
    have = {}
    for tag, key, _value in rows:
        have.setdefault(key, []).append(tag)
    out = []
    for key in DEFAULTS:
        for tag in have.get(key, ()):
            out.append((tag, key))
    return out


def select(rows, order):
    """[(tag, key, value)] to draw: the chosen ones, in the chosen order.

    `order` of None means nobody has chosen yet, and the defaults stand in so
    the panel is useful before it is configured. An EMPTY list is a choice and
    is obeyed - see NONE.
    """
    if order is None:
        order = default_order(rows)
    found = {(tag, key): value for tag, key, value in rows}
    out = []
    for tag, key in order:
        if (tag, key) in found:
            out.append((tag, key, found[(tag, key)]))
    return out


# ---------------------------------------------------------------- harvest
# The node's own panel needs the list of available fields, but only the player
# panel knows which files are attached and which frame is up. So the panel
# publishes what it harvested and the knob widget reads it from here.
#
# Keyed by node name: two player nodes in one script are two different sets of
# inputs, and a shared list would offer the wrong keys on whichever was not
# looked at last.

_lock = threading.Lock()
_available = {}


def set_available(node_name, rows):
    """The panel says what the headers of the current frame hold."""
    with _lock:
        _available[str(node_name)] = list(rows or [])


def available(node_name):
    """[(tag, key, value)] last harvested for that node, or []."""
    with _lock:
        return list(_available.get(str(node_name), ()))


def forget(node_name):
    with _lock:
        _available.pop(str(node_name), None)


# The node's Refresh button has to work even when the META panel is off - that
# is the normal way round, you configure the list and THEN turn the panel on.
# So the panel also leaves a way to be asked for a fresh read.

_harvesters = {}


def set_harvester(node_name, fn):
    """The panel offers a zero-argument callable returning [(tag, key, value)]."""
    with _lock:
        _harvesters[str(node_name)] = fn


def harvest(node_name):
    """Ask the panel to read the headers now. Falls back to the last publish.

    A dead panel is not an error worth raising here: the button says Refresh,
    and 'what I had last time' is the honest answer when there is nothing to
    read from.
    """
    with _lock:
        fn = _harvesters.get(str(node_name))
    if fn is None:
        return available(node_name)
    try:
        rows = list(fn() or [])
    except Exception:
        return available(node_name)
    set_available(node_name, rows)
    return rows
