"""
The "Install dependencies" menu entry.

Exists so that saying no to the offer at startup is not final: this fetches the
same thing on demand and then finishes loading the player, without a restart.

Kept apart from menu.py because that file only runs once, at startup, while
this is called from a menu long afterwards.
"""

import sys

import nuke

from . import installer


def _loaded():
    try:
        import numpy                     # noqa: F401
        return True
    except ImportError:
        return False


def run():
    """Fetches numpy and scipy, then registers the player. Safe to call twice."""
    if _loaded():
        nuke.message("numpy is already available - nothing to install.")
        _register()
        return

    # Same rule as at startup: a managed install does not fetch. Without this
    # the menu entry would be a way straight round the marker.
    reason = installer.managed()
    if reason:
        nuke.message("This JKplayer install is managed (%s), so it does not\n"
                     "download anything.\n\n"
                     "numpy is missing for this Nuke - that is a deployment\n"
                     "problem. Please pass it on to whoever looks after the\n"
                     "install." % reason)
        return

    target = installer.target_dir()
    if installer.nuke_python() is None:
        nuke.message("JKplayer cannot find Nuke's own python, so it cannot\n"
                     "install anything. Install numpy by hand into\n%s" % target)
        return

    if not nuke.ask("Download numpy and scipy into\n%s ?\n\n"
                    "About %d MB." % (target, installer.DOWNLOAD_MB)):
        return

    ok, result = installer.install(target, log=nuke.tprint)
    if not ok:
        nuke.tprint("JKplayer: install failed - %s" % result)
        nuke.message("Install failed.\n\n%s\n\nSee the script editor for the "
                     "full log." % result)
        return

    installer.strip(target, log=nuke.tprint)
    if target not in sys.path:
        sys.path.append(target)
    if not _loaded():
        nuke.message("numpy installed into\n%s\nbut it still will not import.\n"
                     "The wheel may not match this Nuke." % target)
        return

    _register()
    nuke.message("JKplayer is ready.\n\nnumpy went into\n%s" % target)


def _register():
    from . import register
    register.register()
