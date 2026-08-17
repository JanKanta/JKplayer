Fetched dependencies land here, one folder per Nuke:

    pylibs/win_amd64-cp311/      Windows, Python 3.11 (Nuke 16, 17)
    pylibs/linux_x86_64-cp310/   Linux, Python 3.10 (Nuke 15)
    pylibs/macos_arm64-cp311/    macOS Apple Silicon, Python 3.11

The folder name is what jkplayer.paths.platform_tag() returns under the Nuke
that is running, and it is matched exactly. numpy and scipy are compiled, so a
build is locked to both the platform and the CPython version - loading the
wrong one takes Nuke down rather than raising something readable.

Nothing is shipped in here. On the first GUI launch JKplayer offers to fetch
what this Nuke needs; the same is available later under
JKplayer > Install dependencies. To do it by hand, run the Nuke you want to
support:

    <nuke>/python -m pip install --target pylibs/<tag> numpy scipy

This folder is safe to delete - it is rebuilt on demand.
