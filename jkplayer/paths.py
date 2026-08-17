"""
Finding the Nuke installation, the bundled libraries and the machine's memory,
on every platform.

One place for it, because unrelated modules need the same answers: exrcore
looks for the OpenEXRCore library, ocio for the shipped OCIO configs and
menu.py for numpy. They would otherwise each carry their own copy of the same
guesswork.

Nothing in here imports numpy - menu.py has to be able to ask WHERE numpy is
before numpy exists.
"""

import os
import platform
import sys


def nuke_dirs():
    """Where the running Nuke lives, best guess first.

    `sys.executable` is the strongest hint and needs no configuration: inside
    Nuke it is the Nuke binary and in the bundled interpreter it is
    <nuke>/python.exe, so its directory is the installation either way. On
    macOS that is <...>/Nuke.app/Contents/MacOS, which is where the libraries
    and the plugins sit too.

    NUKE_LOCATION is checked as well - a studio may set it - and sys.prefix
    last, for the case where the package is used outside Nuke entirely.
    """
    out = []
    for path in (os.path.dirname(os.path.abspath(sys.executable)),
                 os.environ.get("NUKE_LOCATION"),
                 getattr(sys, "prefix", None)):
        if path and os.path.isdir(path) and path not in out:
            out.append(path)
    return out


def nuke_subdir(*parts):
    """The first existing <nuke>/parts... directory, or None."""
    for base in nuke_dirs():
        path = os.path.join(base, *parts)
        if os.path.isdir(path):
            return path
    return None


def platform_tag():
    """The folder name the bundled libraries for THIS interpreter live under.

    numpy and scipy are compiled, so a build is locked to both the platform and
    the CPython version - a cp311 Windows numpy will not load in Nuke 14
    (cp39) or on Linux, and an ABI mismatch takes Nuke down rather than raising
    something readable. Hence one folder per combination and a strict match:
    when the folder for this interpreter is not there, the bundle is simply not
    used and we fall back to whatever is installed on the system.

    Looks like "win_amd64-cp311", "linux_x86_64-cp310", "macos_arm64-cp39".
    """
    machine = platform.machine().lower()
    if sys.platform == "win32":
        name = "win"
        arch = "amd64" if machine in ("amd64", "x86_64") else machine
    elif sys.platform == "darwin":
        name = "macos"
        arch = machine or "x86_64"
    else:
        name = "linux"
        arch = machine or "x86_64"
    return "%s_%s-cp%d%d" % (name, arch, sys.version_info[0], sys.version_info[1])


def total_ram_mb():
    """How much RAM the machine has (MB). 0 when it cannot be determined.

    Used to size the frame cache - see node.default_cache_mb.
    """
    if sys.platform == "win32":
        try:
            import ctypes

            class MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = MemStatus()
            st.dwLength = ctypes.sizeof(MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return int(st.ullTotalPhys / (1024 * 1024))
        except Exception:
            pass
        return 0

    # Linux and macOS both answer this one; on macOS SC_PHYS_PAGES has been
    # there since 10.4, so there is no need for a sysctl call.
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return int(pages * page_size / (1024 * 1024))
    except (ValueError, OSError, AttributeError):
        pass
    return 0
