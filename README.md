# JKplayer

An EXR review player for Nuke. It reads the EXR files itself instead of going
through `nuke.execute`, so playback comes off a RAM cache rather than the comp
graph — a 1080p ZIP plate decodes at about 55 fps on 8 threads, against 22 fps
for Nuke's own viewer on the same material.

It is a QC tool, not a grading tool: everything in it exists to answer "is this
plate good to send".

## What it does

**Playback** — RAM cache with a byte budget (a quarter of the machine's memory
by default), look-ahead in the direction you are playing, a rolling cache
window so a long shot does not thrash, and a timeline that shows what is
cached. Realtime mode holds the target FPS and skips uncached frames; turn it
off and every frame is shown.

**Two inputs, six view modes** — Single, Double (side by side or stacked),
Difference (A/B dissolve with a plain or high-pass compare), Wipe (draggable,
rotatable line with a blend handle), DiMatte (mattes drawn over the image as
colours) and Annotation. Both windows share zoom and pan, so the pixels line
up, and each picks its own input and its own EXR layer — so you can put rgba
against depth of the same plate.

The inputs are **Comp** and **Plate** (tagged `C` and `P` on the buttons inside
the image). `-` swaps the window between them, which is the quickest A/B there
is: same frame, same zoom, same place on the eye.

Each input has its own **Start at** and **Offset** on the Playback tab, so a
1-100 render lines up under a 1001-1100 plate without a TimeOffset node. They
do not have to be the same length — the timeline follows Comp and the shorter
side holds its end frame, with its own frame numbers written under the cache
bars so the two can be lined up by eye.

**Annotation** — pencil and text over the picture, with the colour and size
right there in the image. A note remembers WHICH view it was made in, so a
circle drawn around a grain problem does not reappear over a plain plate. Notes
wrap to fit the frame, can be dragged, clicked to re-open and recoloured.
Export writes one JPEG per frame and view into the folder set on the node,
optionally with the scopes of that frame, a frame stamp, and a CSV listing
every note.

**DiMatte** takes its mattes either from a third input of its own or from a
**layer of the Comp EXR** — a comp that already carries them needs nothing
wired up, and then the node does not even grow the third input.

**QC modes** — Difference (A against B, as an overlay or a plain difference),
Grain check, High-pass, Temporal (finds duplicate frames), Saturation check,
Value map and Canvas check. Each has its own sliders, remembered per mode.

**Scopes** — histogram and waveform on a scene-linear axis from 0 to 55 with
the clipping line at 1.0, so you can see how far over an exposure goes, not
just that it clips. That axis is gamma-encoded below 1 and logarithmic above
it, so it carries marks at 0.18, 1, 2, 4, 8, 16, 32 and 55 — otherwise there is
no telling 8 from 30. Vectorscope with the standard 75 % / 100 % graticule. All
three follow the visible crop, so zooming in measures what you are looking at,
and all three mark the pixel under the cursor: a line on the histogram, a ring
on the vectorscope, a cross on the waveform.

The histogram and the waveform are resized by their edges (side for width,
bottom for height) or by the bottom corner for both. The vectorscope only
scales square — a stretched circle would put the 75 % boxes at two different
radii and the graticule would stop meaning anything.

**fMIN / fMAX** next to the FPS: the lowest and highest scene-linear value in
the whole frame, every pixel counted rather than sampled, through the input
transform. One stray negative is the thing worth catching. Held frames only.

**Colour** — Nuke's built-in transforms (one table, fast) or full OCIO with the
studio config. Input transforms for log material (LogC, S-Log3, Cineon,
Log3G10 and the rest).

## Requirements

* **Nuke 14 or newer**, on Windows, Linux or macOS. Both Qt bindings are
  supported — Nuke 14 ships PySide2, Nuke 15+ PySide6.
* **numpy** (required) and **scipy** (optional) — Nuke ships neither, and this
  download does not carry them either. **On the first launch JKplayer offers
  to fetch them**, which takes about 50 MB and twenty seconds. Say yes and
  there is nothing else to do; say no and it is still there later under
  *JKplayer > Install dependencies*.

They are not shipped because they are compiled: a wheel is locked to both the
platform and the CPython version, so one bundled build would only help the
people on that exact Nuke and that exact OS, while everyone else carried 120 MB
they could not load. Fetching gets each Nuke the build that is right for it —
pip resolves that from the interpreter asking, so there is nothing to pick and
nothing to get wrong.

They land in `pylibs/<platform>-cp<version>` next to the package, or in
`~/.nuke/pylibs/<tag>` when the install folder is read-only (a shared install on
a server usually is).

`menu.py` looks in four places and stops at the first numpy it finds:

1. whatever is **already importable** — a numpy the studio put there wins and
   is never shadowed by ours
2. `pylibs/<tag>` next to the package
3. `~/.nuke/pylibs/<tag>`
4. `~/.nuke/pylibs`, flat, for installs made by hand

All of them are matched by exact platform and Python version, because an ABI
mismatch takes Nuke down rather than raising something readable.

Fetching **only ever happens in GUI mode, and only after you agree** — a render
node must not stop to talk to PyPI. In terminal mode a missing numpy just
prints the pip command, with the paths already filled in, and JKplayer does
not load. Nothing else in Nuke is affected either way.

To do it by hand, or to prepare a folder for someone else, run the Nuke you
want to support:

```
<nuke>/python -m pip install --target pylibs/<tag> --only-binary=:all: numpy scipy
```

The tag is what `jkplayer.paths.platform_tag()` returns under that Nuke.

Without scipy the grain and high-pass checks fall back to a fixed 3x3 blur:
they still show something, but their sliders stop doing anything and edges are
no longer suppressed, which is most of the point of the grain check.

PyOpenColorIO and the OpenEXR libraries come with Nuke and are found next to
the running Nuke automatically, on every platform.

## Install

1. Copy this whole `JKplayer` folder into `~/.nuke/`
   (Windows: `%USERPROFILE%\.nuke\`).

2. Add one line to `~/.nuke/init.py`, creating the file if it is not there:

   ```python
   nuke.pluginAddPath("./JKplayer")
   ```

   The folder may be called anything and may live anywhere - the code works out
   where it is from its own path. Only this line has to match.

3. Restart Nuke. If numpy is not there yet it offers to fetch it — see
   Requirements. Then there will be a **JKplayer** menu in the menu bar and
   a **JKplayer** entry in the Nodes toolbar.

`pluginAddPath` puts the folder on both the plugin path and `sys.path`, so
Nuke runs `menu.py` from here and the package imports on its own.

## Rolling it out to other people

The offer to download numpy is right for one person on one machine and wrong
for a facility: two hundred workstations means two hundred dialogs, most of
which then hit the firewall, and each artist ends up with whatever pip gave
them that day. For a managed install, prepare the libraries once and switch the
offer off:

1. On one machine per (OS, Nuke version), fetch into the folder that ships:

   ```
   <nuke>/python -m pip install --target pylibs/<tag> --only-binary=:all: "numpy>=1.24,<3" "scipy>=1.10,<2"
   ```

   Nuke 16 and 17 share `cp311`, so one folder covers both; Nuke 14 is `cp39`
   and Nuke 15 `cp310`.

2. Put the folder on the share, **read-only** for artists. A read-only install
   also means an accidental fetch redirects into the user's own `~/.nuke`
   instead of corrupting the shared copy.

3. Set `JKPLAYER_NO_FETCH=1`, or drop a file called `MANAGED` into `pylibs/`.
   Either one turns the download offer into a line in the console naming the
   folder it expected — a deployment fault to report, not something an artist
   should fix over the firewall. The *Install dependencies* menu entry refuses
   too, so it is not a way round the marker.

On startup the console says which numpy and scipy were used and where they came
from. With the "already importable wins" rule above, that line is the
difference between answering a support mail and doing archaeology.

## Use

1. **JKplayer > Create JKplayer Node** and connect a Read with an `.exr`
   sequence to input **Comp** (optionally a second one to **Plate**, and a
   matte to **DiMatte** if the mattes come as their own files).
2. **JKplayer > Open JKplayer Panel**, or dock the panel from the pane menu.
3. The node holds all the settings; the panel follows whichever JKplayer node
   is selected.

Only Read nodes with `.exr` can be attached (a Dot on the way is fine) —
anything else is disconnected and the status line says which node it was and
what is allowed instead. The player reads the files off disk itself, so a node
that changes the picture cannot be honoured: it would show the untouched plate
and pass it off as the result. Shifting in time is done on the node, not with a
TimeOffset; fitting one input onto the other is done by the player when it
compares them, not with a Reformat.

The two inputs may be different lengths and different resolutions. A comparison
across resolutions fits one onto the other and **says so in the status line** —
a difference over a resampled input is not the same measurement as one over two
plates that already match.

A Viewer cannot be attached either. The node deliberately renders nothing —
display is the panel's job.

### Keys

```
J K L        play backwards / stop / play forwards
arrows       step a frame
R G B A Y    channels (a second press returns to RGB)
C Q          CC and QC panels
H V W        histogram, vectorscope, waveform
F            fit into the window
1-7          QC mode
I O          mark in / out
P            freeze the pixel readout
X            switch window (in Double)
-            swap the window between Comp and Plate
;            pencil / text tools live in the image, in Annotation mode
```

(`-` is bound on both the number row and the number pad — Qt tells the two
minus keys apart, and the pad one is the one over plus.)

## Tests

They are not in here on purpose - installing a player should not drag along
5 MB of test plates. They live in the development tree, alongside the EXR set
they measure against, and point at this folder through `JKPLAYER_DIR`. There
is one copy of the code, so there is nothing to keep in sync.

What they cover, since the Qt layer cannot be tested inside Nuke at all (the
terminal only has a `QCoreApplication` and a standalone `QApplication`
freezes): both EXR readers against a reference set in every compression, the
colour transforms against OCIO, the scopes, the QC modes, every knob on the
node, and the geometry that Qt only draws - the wipe mask, where the in-image
panels are anchored, the slider curves. Plus a static check that catches calls
to methods that do not exist and wrong argument counts, which is otherwise the
kind of thing that only surfaces in Nuke as "the panel does not work".

## What has actually been run

Being straight about it, because "Nuke 14+, all platforms" is a claim and not a
measurement:

| | tested |
|---|---|
| Windows, Nuke 17 (PySide6, cp311) | yes — everything below, plus daily use |
| Windows, Nuke 16 | same interpreter and binding, expected to behave the same |
| Nuke 14 / 15 (PySide2 and PySide6 on cp39/cp310) | **no** — the Qt5 branch is written from the known differences but has not been run |
| Linux, macOS | **no** — the library lookup has patterns for both, never exercised |

What "tested" covers: both EXR readers against a reference set in every
compression, the colour transforms against OCIO, the scopes, the QC modes,
every knob on the node, the geometry Qt only draws, and a static pass over the
whole package. The Qt layer itself cannot be tested inside Nuke at all — see
Tests.

Reports from other platforms are welcome; that table is how it gets shorter.

## Layout

```
jkplayer/
  panel.py      the panel - playback, layout, the whole UI
  imageview.py  one image window: display, pan/zoom, pixel probe
  overlay.py    the panels drawn inside the image (CC, QC, scopes)
  timeline.py   timeline with the cache bar and mark in/out
  node.py       the node and every setting on it
  register.py   registration into Nuke's menus
  loader.py     background decoding, two queues
  cache.py      RAM cache with a byte budget
  sequence.py   frame number -> file path
  exrcore.py    EXR through Nuke's own library (ctypes)
  exrread.py    pure Python EXR reader, the fallback
  reader.py     picks between the two
  scopes.py     histogram, vectorscope, waveform
  effects.py    the QC modes
  annotate.py   notes on frames: storage, drawing, export, CSV
  resample.py   fits one input onto the other, for a comparison
  ocio.py       OCIO display transform through a baked 3D LUT
  nukelut.py    the built-in colour transforms
  paths.py      finding Nuke and the machine's memory
  installer.py  fetching numpy and scipy for this Nuke
  setup_deps.py the "Install dependencies" menu entry
  qtcompat.py   PySide2 / PySide6
```

## Licence

MIT — see [LICENSE](LICENSE). Use it, change it, ship it.
