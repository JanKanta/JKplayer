"""
Registers EXRplayer into Nuke.

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
    from exrplayer.installer import package_root, user_pylibs
    from exrplayer.paths import platform_tag
    return [os.path.join(package_root(), "pylibs", platform_tag()),
            user_pylibs(),
            USER_PYLIBS_FLAT]


def _find_numpy():
    if _have("numpy"):
        return "already on sys.path"
    for path in _candidates():
        if _try_path(path):
            return path
    return None


def _offer_to_fetch():
    """Asks whether to fetch numpy, and does it. Returns True when numpy is in.

    Only ever called in GUI mode, with a human in front of the machine.
    """
    from exrplayer import installer

    target = installer.target_dir()
    if installer.nuke_python() is None:
        nuke.tprint("EXRplayer: cannot find Nuke's python, so it cannot fetch "
                    "numpy for you.")
        return False

    if not nuke.ask(
            "EXRplayer needs numpy, which Nuke does not ship.\n\n"
            "Download it now? About %d MB, usually under a minute.\n"
            "It goes into:\n%s\n\n"
            "Nothing outside that folder is touched."
            % (installer.DOWNLOAD_MB, target)):
        # Nothing more here - _load prints the how-to once, for every way of
        # ending up without numpy.
        nuke.tprint("EXRplayer: declined. Use EXRplayer > Install dependencies "
                    "to do it later.")
        return False

    nuke.tprint("EXRplayer: fetching numpy and scipy into %s" % target)
    ok, result = installer.install(target, log=nuke.tprint)
    if not ok:
        nuke.tprint("EXRplayer: install failed - %s\n"
                    "  A studio firewall usually blocks PyPI. Install it by "
                    "hand with\n      %s"
                    % (result, " ".join(installer.pip_command(target))))
        nuke.message("EXRplayer could not download numpy.\n\n%s\n\n"
                     "See the script editor for the full log." % result)
        return False

    installer.strip(target, log=nuke.tprint)
    if not _try_path(target):
        nuke.tprint("EXRplayer: numpy installed into %s but still will not "
                    "import - the wheel may not match this Nuke." % target)
        return False
    nuke.tprint("EXRplayer: numpy is in. Ready.")
    return True


def _install_menu():
    """A way back in after saying no, so the offer is not one-shot."""
    try:
        m = nuke.menu("Nuke").addMenu("EXRplayer")
        m.addCommand("Install dependencies",
                     "import exrplayer.setup_deps as s; s.run()")
    except Exception:
        pass


def _load():
    where = _find_numpy()

    if where is None and nuke.GUI:
        if _offer_to_fetch():
            where = "downloaded"

    if where is None:
        from exrplayer import installer
        nuke.tprint(
            "EXRplayer: not loaded - no numpy for this Nuke.\n"
            "  Nuke does not ship numpy and none is installed for this\n"
            "  interpreter. Install it with\n"
            "      %s\n"
            "  (scipy is optional - the grain and high-pass checks are sharper\n"
            "  with it, and fall back to a coarser blur without it)"
            % " ".join(installer.pip_command(installer.target_dir())))
        if nuke.GUI:
            _install_menu()
        return

    import exrplayer.register
    exrplayer.register.register()


try:
    _load()
except Exception as _exc:
    # Never raise from here: an exception in menu.py stops Nuke from loading
    # the rest of its plugins. Say what happened and let Nuke carry on.
    nuke.tprint("EXRplayer: loading failed: %s: %s"
                % (type(_exc).__name__, _exc))
