"""
Registers JKplayer into Nuke.

Nuke runs the menu.py of every directory on the plugin path, so this file is
picked up automatically once the folder is added - see the README. The folder
is on sys.path by then (nuke.pluginAddPath puts it there), so the package
imports without any path juggling.

The one thing that needs handling is numpy, which Nuke does not ship. Four
places are tried, in this order:

  1. whatever is already importable - a numpy the studio put there wins, and
     is never shadowed by ours
  2. pylibs/<platform>-cp<version> next to this file
  3. ~/.nuke/pylibs/<platform>-cp<version>
  4. ~/.nuke/pylibs, flat, for installs made by hand before the tags existed

Every one of those is asked for by exact platform and Python version, because a
compiled wheel is locked to both and a mismatch takes Nuke down instead of
raising something readable.

When none of them has it, the first GUI launch offers to fetch it. That reaches
the network, so it asks first, and it never happens in terminal mode - a render
node must not stop to talk to PyPI.
"""

import os
import sys

import nuke

USER_PYLIBS_FLAT = os.path.join(os.path.expanduser("~"), ".nuke", "pylibs")


def _have(name):
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def _try_path(path):
    """Puts a folder on sys.path and reports whether numpy showed up."""
    if path and os.path.isdir(path) and path not in sys.path:
        sys.path.append(path)
    return _have("numpy")


def _candidates():
    """The tagged folders to look in, best first."""
    from jkplayer.installer import package_root, user_pylibs
    from jkplayer.paths import platform_tag
    return [os.path.join(package_root(), "pylibs", platform_tag()),
            user_pylibs(),
            USER_PYLIBS_FLAT]


def platform_tag_or_unknown():
    """The tag this Nuke needs, for the message that says it is missing."""
    try:
        from jkplayer.paths import platform_tag
        return platform_tag()
    except Exception:
        return "unknown platform"


def _find_numpy():
    if _have("numpy"):
        return "already on sys.path"
    for path in _candidates():
        if _try_path(path):
            return path
    return None


def _report(where):
    """Says WHICH numpy is being used, and which version.

    Because "whatever is already importable wins" (see above), the player may
    quietly be running on a numpy the studio shipped years ago. Without this
    line a support mail starts with archaeology; with it, the artist can paste
    one line and the question is answered.
    """
    parts = []
    for name in ("numpy", "scipy"):
        try:
            mod = __import__(name)
            parts.append("%s %s" % (name, getattr(mod, "__version__", "?")))
        except ImportError:
            parts.append("%s missing" % name)
    nuke.tprint("JKplayer: %s  (from %s)" % (", ".join(parts), where))


def _offer_to_fetch():
    """Asks whether to fetch numpy, and does it. Returns True when numpy is in.

    Only ever called in GUI mode, with a human in front of the machine.
    """
    from jkplayer import installer

    target = installer.target_dir()
    if installer.nuke_python() is None:
        nuke.tprint("JKplayer: cannot find Nuke's python, so it cannot fetch "
                    "numpy for you.")
        return False

    if not nuke.ask(
            "JKplayer needs numpy, which Nuke does not ship.\n\n"
            "Download it now? About %d MB, usually under a minute.\n"
            "It goes into:\n%s\n\n"
            "Nothing outside that folder is touched."
            % (installer.DOWNLOAD_MB, target)):
        # Nothing more here - _load prints the how-to once, for every way of
        # ending up without numpy.
        nuke.tprint("JKplayer: declined. Use JKplayer > Install dependencies "
                    "to do it later.")
        return False

    nuke.tprint("JKplayer: fetching numpy and scipy into %s" % target)
    ok, result = installer.install(target, log=nuke.tprint)
    if not ok:
        nuke.tprint("JKplayer: install failed - %s\n"
                    "  A studio firewall usually blocks PyPI. Install it by "
                    "hand with\n      %s"
                    % (result, installer.pip_command_text(target)))
        nuke.message("JKplayer could not download numpy.\n\n%s\n\n"
                     "See the script editor for the full log." % result)
        return False

    installer.strip(target, log=nuke.tprint)
    if not _try_path(target):
        nuke.tprint("JKplayer: numpy installed into %s but still will not "
                    "import - the wheel may not match this Nuke." % target)
        return False
    nuke.tprint("JKplayer: numpy is in. Ready.")
    return True


def _install_menu():
    """A way back in after saying no, so the offer is not one-shot."""
    try:
        m = nuke.menu("Nuke").addMenu("JKplayer")
        m.addCommand("Install dependencies",
                     "import jkplayer.setup_deps as s; s.run()")
    except Exception:
        pass


def _load():
    from jkplayer import installer
    where = _find_numpy()

    if where is None and nuke.GUI:
        # A managed install never asks. Its libraries were put there by
        # whoever rolled it out, so a missing one is a deployment fault to be
        # reported, not something an artist should fix over the firewall.
        reason = installer.managed()
        if reason:
            nuke.tprint(
                "JKplayer: not loaded - no numpy for this Nuke (%s).\n"
                "  This install is managed (%s), so nothing was downloaded.\n"
                "  Expected it in: %s"
                % (platform_tag_or_unknown(),
                   reason, "\n                  ".join(_candidates())))
            return
        if _offer_to_fetch():
            where = "downloaded"

    if where is None:
        nuke.tprint(
            "JKplayer: not loaded - no numpy for this Nuke.\n"
            "  Nuke does not ship numpy and none is installed for this\n"
            "  interpreter. Install it with\n"
            "      %s\n"
            "  (scipy is optional - the grain and high-pass checks are sharper\n"
            "  with it, and fall back to a coarser blur without it)"
            % installer.pip_command_text(installer.target_dir()))
        if nuke.GUI:
            _install_menu()
        return

    _report(where)
    import jkplayer.register
    jkplayer.register.register()


try:
    _load()
except Exception as _exc:
    # Never raise from here: an exception in menu.py stops Nuke from loading
    # the rest of its plugins. Say what happened and let Nuke carry on.
    nuke.tprint("JKplayer: loading failed: %s: %s"
                % (type(_exc).__name__, _exc))
