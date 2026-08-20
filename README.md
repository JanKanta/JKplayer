# JKplayer

A review player for Nuke. It reads the files itself instead of going through
`nuke.execute`, so playback comes off a RAM cache rather than the comp graph —
a 1080p ZIP plate decodes at **285 fps on 8 threads** and a 4K one at **68**,
off Nuke's own OpenEXR library. See [Measured](#measured) for the rest, and
for what has deliberately not been claimed.

It is a QC tool, not a grading tool: everything in it exists to answer "is this
plate good to send".

## Formats

**EXR** — through Nuke's own OpenEXR library when it is there (every
compression, and 5 to 10x faster — the wider the frame the bigger the gap),
with a pure Python reader as the fallback for NONE, ZIP and ZIPS. Both were
checked against each other and give bit-identical results.

**DPX** — 8, 10, 12 and 16 bit, both endiannesses, RGB / RGBA / ABGR /
luminance. What comes out is the code value normalised to 0–1, not linear
light, because that is exactly what the Cineon input transform already expects
— so a 10-bit log DPX needs the input space set and nothing else. Run-length
encoding, flipped orientation, depth maps and anything else not understood are
**refused by name**, on the grounds that a wrong picture is worse than no
picture.

**Movies** — `.mov`, `.mp4`, `.mxf`, `.m4v` through ffmpeg. ProRes, DNxHD/HR,
MJPEG and the other all-intra codecs seek for the cost of one frame; long-GOP
works too and is frame-accurate, but a jump costs up to a whole group of
pictures. H.264 is tested; H.265 goes down the same path and has not been.
Which kind a file is comes back from `probe()`, so the panel can say so —
though it does not show it yet. Movies need **ffmpeg installed** — see
Requirements.

A movie is one file for every frame, which the rest of the player cannot say —
it addresses frames by path. So internally a movie frame is written
`clip.mov|1042`. It is a small lie in a string, and it buys not touching the
cache key, the loader queue or the sequence cache at all.

Mind the **chroma**: a 4:2:2 codec carries half the colour resolution and 4:2:0
a quarter. It is reported in the metadata, because a difference check between an
EXR comp and a 4:2:0 delivery shows artefacts that are in the codec and not in
the comp. It is not yet said out loud where it would matter most, which is in
the Difference mode itself.

## What it does

**Playback** — RAM cache with a byte budget (a quarter of the machine's memory
by default), look-ahead in the direction you are playing, a rolling cache
window so a long shot does not thrash, and a timeline that shows what is
cached. Realtime mode holds the target FPS and skips uncached frames; turn it
off and every frame is shown.

**Two inputs, six view modes** — Base, Sync (side by side or stacked),
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

**Metadata** — the headers of both inputs, read fresh every frame (a header is
a kilobyte at the front of the file, measured at 0.05 ms, so following the
playhead is free). The **META** button puts them bottom left of the image, one
panel per window showing that window's own input — in Sync the two windows are
two different files, and a single shared panel could only ever be right about
one of them.

Which fields, and in what order, is set on the node's Metadata tab: the file's
whole catalogue on the left with a search, the drawn list on the right, one
input at a time. DPX gives the most here — timecode, slate, keycode, frame
position, the scanner and its serial number.

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

### ffmpeg, for movies only

EXR and DPX are read by this package alone. Movies are not: ProRes and H.264
are not things to decode in numpy, so `.mov`, `.mp4`, `.mxf` and `.m4v` go
through **ffmpeg**, which Nuke does not ship. Everything else works without it.

**Both `ffmpeg` and `ffprobe` are needed**, and a machine with only the first
counts as having no movie support at all. ffprobe is the only one that reports
the frame rate as the exact ratio it is: ffmpeg's own output rounds 24000/1001
to "23.98", and since a seek is `frame / fps` that error grows with the frame
number — about one whole frame by six thousand in. A player that quietly hands
back the neighbouring frame is worse than one that will not open the file.

They are looked for in three places, first hit wins:

1. `$JKPLAYER_FFMPEG` — a folder, for a studio that keeps its own copy
2. `jkplayer/bin/` — a copy dropped in beside the code
3. `PATH`

When neither is found the panel says which one is missing and where it looked,
and nothing else in the player is affected.

They are not bundled here, and that is a decision rather than an oversight.
The obvious pip route, `imageio-ffmpeg`, ships **ffmpeg without ffprobe** (88 MB
of it), which by the paragraph above is not enough. Shipping both ourselves
means picking a build: we only ever decode, and the encoders — x264, x265 —
are what make a build GPL, so an LGPL build covers this and carries no such
obligation.

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

1. **JKplayer > Create JKplayer Node** and connect a Read to input **Comp**
   (optionally a second one to **Plate**, and a matte to **DiMatte** if the
   mattes come as their own files). An EXR or DPX sequence, or a movie.
2. **JKplayer > Open JKplayer Panel**, or dock the panel from the pane menu.
3. The node holds all the settings; the panel follows whichever JKplayer node
   is selected.

Only Read nodes pointing at a format this player decodes can be attached (a Dot
on the way is fine) — anything else is disconnected and the status line says
which node it was and what is allowed instead. The player reads the files off
disk itself, so a node that changes the picture cannot be honoured: it would
show the untouched plate and pass it off as the result. Shifting in time is done on the node, not with a
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
X            switch window (in Sync)
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
DPX reader against files from a second implementation, movie frame identity on
clips that carry their own frame number, the colour transforms against OCIO,
the scopes, the QC modes, every knob on the node, and the geometry that Qt only
draws - the wipe mask, where the in-image panels are anchored, the slider
curves. Plus a static check that catches calls to methods that do not exist and
wrong argument counts, which is otherwise the kind of thing that only surfaces
in Nuke as "the panel does not work".

The parts that talk to Qt but hold real logic - the Metadata tab, the banded
display path - are tested against a stub that validates every Qt name against
the real PySide6, so an attribute that does not exist fails here rather than in
Nuke. What the stub cannot do is store items or lay anything out, so which row
a click lands on is still something only Nuke can answer.

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

### The newer formats

DPX and movies are younger than the rest and their evidence is thinner, so:

| | how far it has been taken |
|---|---|
| DPX | against files written by ffmpeg — a second implementation — at 8, 10 and 16 bit, both endiannesses, plus hand-built headers for the film and television fields ffmpeg omits. **Not yet against a DPX from a real pipeline.** |
| ProRes, DNxHR, MJPEG | frame identity proved on clips whose frame number is painted into them as black and white blocks: 96 frames each, every one fetched by its own cold seek, forwards, backwards and at random |
| H.264 / long-GOP | the same, with a 250-frame group of pictures |
| Both | concurrent access from several threads, the open-file cap, and no leaked processes |

The 10-bit DPX packing convention was settled by decoding a reference file both
ways and seeing which gave back the values that went in. The spec sentence can
be read either way, and a guess would have been wrong half the time with
nothing to show for it.

Movie frame identity is checked with **blocks, not pixel values**. The first
attempt wrote the frame number into a pixel and reported DNxHR as broken; it
was not — DNxHR is tagged limited range, so the value came back scaled and the
instrument was blaming the frame for what the colour path did.

## Measured

Everything below was taken on one machine — **Ryzen 9 9950X, 16 cores / 32
threads, 94 GB** — and every number says which backend produced it, because
this file used to quote a figure without saying, and it turned out to be the
fallback rather than the path that runs.

### Decode

ZIP16 half RGB, the same 64 frames through both readers, **warm in RAM** so
this is decode and not the disk:

| | | 1 thread | 4 | 8 | 16 | 24 |
|---|---|---|---|---|---|---|
| **HD** 1920x1080, 7.1 MB/frame | Nuke's OpenEXR DLL | 56.0 | 163.7 | **285.5** | 350.8 | 401.3 |
| | pure Python fallback | 14.9 | 40.0 | 57.6 | 58.9 | 54.3 |
| **4K** 4096x2160, 27.9 MB/frame | Nuke's OpenEXR DLL | 13.3 | 41.9 | **68.3** | 89.6 | 103.5 |
| | pure Python fallback | 3.5 | 6.0 | 6.9 | 7.1 | 7.0 |

fps. The DLL is **5x** the fallback at HD on 8 threads and **10x** at 4K, and
the gap widens with threads because the fallback stops scaling past 8. Older
versions of this file put that ratio at 2.3x and the headline rate at 55 fps;
both came from the fallback, which is not what runs when Nuke is there.

Other formats, same conditions:

| | |
|---|---|
| 4K DPX 10-bit | 25 fps at 16 threads |
| HD ProRes 422 HQ | 72 fps |
| 4K ProRes 422 HQ | 15–17 fps |

4K movies are held back by the pipe out of ffmpeg, not by the decoder: 53 MB a
frame in 16-bit RGB against about 1100 MB/s. HD and 2K have room to spare.
Stepping one frame on in a 4K movie costs 56 ms; a jump backwards restarts
ffmpeg and costs about half a second, because a pipe cannot be run in reverse.

### Display

Turning half floats into screen bytes, spread over eight row bands. Output is
bit-identical either way — the change is which cores do it:

| | one thread | eight bands | |
|---|---|---|---|
| zoom 1:1, full 4K | 26 fps | **136 fps** | 5.1x |
| fit a 2K window | 100 | 409 | 4.1x |
| fit a 1.4K window | 224 | 829 | 3.7x |

The first row is the one that mattered: full-resolution playback used to sit
under 24 fps, which is not a review.

### Disk

The ceiling nothing in the code can lift:

| | sequential read | 4K ZIP EXR at that rate |
|---|---|---|
| SATA SSD (Samsung 870 EVO) | 566 MB/s, flat from two threads up | 20 fps |
| NVMe (Crucial T705) | 8814 MB/s at 16 threads | 314 fps |

On the SATA drive the full cache fill measures 19.9 fps against 20.2 for the
raw bytes with no decoding at all — the disk is 98.5 % of it and there is
nothing to tune. Moving a working set to the NVMe is worth more than any code
change in this file.

### What has NOT been measured

How any of this compares to **Nuke's own viewer**. This file used to claim
22 fps for it; that line is gone rather than repeated, because no Foundry
licence was reachable to check it and an unverifiable comparison flatters
whoever wrote it.

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
  sequence.py   frame number -> what the reader is handed
  exrcore.py    EXR through Nuke's own library (ctypes)
  exrread.py    pure Python EXR reader, the fallback
  dpxread.py    DPX reader
  movread.py    movies, through ffmpeg
  reader.py     picks between them - the whole format boundary is four calls
  meta.py       which header fields are shown, and in what order
  metaknob.py   the Metadata tab on the node
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
