"""
Fetching numpy and scipy for the Nuke that is missing them.

They are not shipped with JKplayer any more. A compiled wheel is locked to
both the platform and the CPython version, so shipping one build would only
ever help the people who happen to run that exact Nuke on that exact OS, while
everyone else carries a hundred-odd megabytes of libraries they cannot load.
Fetching instead means every Nuke gets the build that is right for it - pip
resolves that from the interpreter doing the asking, so there is nothing to
choose and nothing to get wrong.

The cost is that this reaches the network, which is why nothing here happens on
its own: menu.py asks first, and only in GUI mode. A render node must never
stop to talk to PyPI.

Nothing in here imports numpy - this is the module that runs when numpy is not
there yet.
"""

import os
import subprocess
import sys

from .paths import nuke_dirs, platform_tag

# scipy is not required, but the grain and high-pass checks lose their sliders
# without it, so it is fetched together with numpy rather than left as homework.
#
# RANGES, not bare names and not pins. Bare names mean whoever installs next
# year gets a different set than the one this was tested against, and a bug
# report cannot be reproduced. A pin cannot work either: this is fetched by
# Nuke's own interpreter, which is cp39 on Nuke 14 and cp311 on 16/17, and no
# single version has wheels for all of them. A range lets pip pick the newest
# build that fits the Nuke doing the asking, which is the whole point of
# fetching rather than shipping.
#
# The upper bounds are the next MAJOR: numpy 2.0 moved the C ABI under
# everyone's feet and 3.0 may do it again, and scipy renames and removes
# functions across majors.
PACKAGES = ("numpy>=1.24,<3", "scipy>=1.10,<2")

DOWNLOAD_MB = 50          # measured: numpy 12.6 + scipy 36.6; grows slowly

# A managed install - a studio rollout - has its libraries put there on
# purpose, and must never ask an artist to download anything: on two hundred
# workstations that is two hundred dialogs, most of which then hit the
# firewall. Either marker turns the offer into a line in the console.
NO_FETCH_ENV = "EXRPLAYER_NO_FETCH"
MANAGED_MARKER = "MANAGED"


def package_root():
    """The EXRPlayer folder - the one that holds both exrplayer/ and pylibs/."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def user_pylibs():
    """~/.nuke/pylibs/<tag>, the fallback when the package folder is read-only.

    Tagged, not flat: one ~/.nuke is shared by every Nuke on the machine, and a
    flat folder would put a cp311 numpy where a cp39 Nuke would find it and
    die on it.
    """
    return os.path.join(os.path.expanduser("~"), ".nuke", "pylibs", platform_tag())


def managed():
    """Whether this install is looked after by somebody else. (reason, ) or None.

    Returns the reason as text so the console can say WHICH marker did it -
    "it just did not ask" is the kind of thing that costs an afternoon.
    """
    value = os.environ.get(NO_FETCH_ENV, "").strip()
    if value and value.lower() not in ("0", "false", "no"):
        return "%s=%s" % (NO_FETCH_ENV, value)
    marker = os.path.join(package_root(), "pylibs", MANAGED_MARKER)
    if os.path.isfile(marker):
        return marker
    return None


def _writable(path):
    """Whether a folder can be created or written to, without leaving a mess."""
    probe = path
    while probe and not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            return False
        probe = parent
    return os.access(probe, os.W_OK)


def target_dir():
    """Where a fetched build should land.

    Next to the package when that is writable, so the whole tool stays one
    folder you can copy. A shared install on a server usually is not writable,
    and then it goes into the user's own .nuke instead.
    """
    bundle = os.path.join(package_root(), "pylibs", platform_tag())
    return bundle if _writable(bundle) else user_pylibs()


def nuke_python():
    """Nuke's own interpreter, the one that can run pip. None when not found.

    Not sys.executable: inside Nuke that is the Nuke binary, and running it
    with -t takes an interpreter licence for the length of the install. The
    python that sits next to it takes none, and being the same build it
    resolves the same wheels.
    """
    names = ["python.exe"] if sys.platform == "win32" else ["python3", "python"]
    for base in nuke_dirs():
        for name in names:
            path = os.path.join(base, name)
            if os.path.isfile(path):
                return path
    return None


def pip_command(target, executable=None, packages=PACKAGES):
    """The exact command that would be run, as an argv list for subprocess."""
    return [executable or nuke_python() or "<nuke>/python",
            "-m", "pip", "install",
            "--target", target,
            # Never build from source: without a compiler that ends in a long
            # hang and a wall of errors, and a wheel exists for every Nuke we
            # support anyway. Failing straight away says more.
            "--only-binary=:all:",
            "--no-input",
            "--disable-pip-version-check"] + list(packages)


def pip_command_text(target, executable=None, packages=PACKAGES):
    """The same command as ONE LINE somebody can paste into a shell.

    Not " ".join(argv): a requirement carries '>' and '<', which every shell
    there is reads as redirection - `pip install numpy>=1.24,<3` writes an
    empty file called '=1.24,' and installs nothing. Anything with a space or
    a shell character gets quoted, so what is printed is what runs.
    """
    out = []
    for arg in pip_command(target, executable, packages):
        if any(c in arg for c in ' <>|&^"'):
            out.append('"%s"' % arg.replace('"', '\\"'))
        else:
            out.append(arg)
    return " ".join(out)


def install(target=None, packages=PACKAGES, log=None):
    """Fetches the packages into `target`. Returns (ok, message).

    `log` is called with each line pip produces, so a caller can show progress.
    Never raises - the caller is a menu.py, and an exception there stops Nuke
    from loading the rest of its plugins.
    """
    say = log or (lambda line: None)
    exe = nuke_python()
    if exe is None:
        return False, ("cannot find Nuke's python next to %s"
                       % (nuke_dirs()[0] if nuke_dirs() else "Nuke"))

    target = target or target_dir()
    try:
        if not os.path.isdir(target):
            os.makedirs(target)
    except OSError as exc:
        return False, "cannot create %s: %s" % (target, exc)

    cmd = pip_command(target, exe, packages)
    say(pip_command_text(target, exe, packages))

    kwargs = {}
    if sys.platform == "win32":
        # Otherwise every pip call flashes a console window over the UI.
        kwargs["creationflags"] = 0x08000000       # CREATE_NO_WINDOW
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                universal_newlines=True, **kwargs)
    except OSError as exc:
        return False, "cannot run pip: %s" % exc

    tail = []
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            say(line)
            tail.append(line)
            del tail[:-12]                # only the end is worth reporting
    proc.wait()

    if proc.returncode != 0:
        return False, ("pip failed (exit %d):\n  %s"
                       % (proc.returncode, "\n  ".join(tail)))
    return True, target


def strip(target, log=None):
    """Removes what pip installs but a player never reads: tests and bytecode.

    numpy and scipy ship their own test suites, which are a good third of the
    size and are never imported by us. Deleting them is safe - nothing outside
    the suites imports them - and it is the difference between a 130 MB folder
    and an 80 MB one.
    """
    say = log or (lambda line: None)
    freed = 0
    for root, dirs, files in os.walk(target, topdown=True):
        for name in list(dirs):
            if name in ("tests", "test", "__pycache__"):
                path = os.path.join(root, name)
                freed += _rmtree(path)
                dirs.remove(name)
        for name in files:
            if name.endswith((".pyc", ".pyi", ".pyx", ".pxd")):
                path = os.path.join(root, name)
                try:
                    freed += os.path.getsize(path)
                    os.remove(path)
                except OSError:
                    pass
    say("trimmed %d MB of tests and bytecode" % (freed / (1024 * 1024)))
    return freed


def _rmtree(path):
    freed = 0
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            full = os.path.join(root, name)
            try:
                freed += os.path.getsize(full)
                os.remove(full)
            except OSError:
                pass
        for name in dirs:
            try:
                os.rmdir(os.path.join(root, name))
            except OSError:
                pass
    try:
        os.rmdir(path)
    except OSError:
        pass
    return freed
