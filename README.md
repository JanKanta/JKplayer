# JKplayer 0.5

A review player for Nuke. It reads the files itself instead of going through
`nuke.execute`, so playback comes off a RAM cache rather than the comp graph —
a 1080p ZIP plate decodes at 285 fps on 8 threads and 401 at 24, measured warm
in RAM so it is decode and not disk. Whether that beats Nuke's own viewer is
**not claimed here**: no Foundry licence has been reachable to measure it, and
an earlier version of this file put a number on it anyway.

It is a QC tool, not a grading tool: everything in it exists to answer "is this
plate good to send".

## New in 0.5

* **One node, one panel.** Clicking or opening a JKplayer node opens *its*
  panel, always docked (never a floating window) and named after the node —
  `JKplayer1` opens the panel `JKplayer1`. Renaming the node renames the panel;
  deleting the node closes its panel and frees its RAM cache.
* **Annotations, grown up.** Export button first on the bar, a real **rubber**
  that cuts strokes where it passes, a colour row with a palette of more
  colours, **Shift-drag** to size the pen or the text, and a note dialog with
  its own text size and a *visible in the exported picture* switch.
* **Crop before export.** Export opens a crop dialog over every annotated
  frame: zoom in, crop, same crop for all, and the crop is scaled back up to the
  frame size in the JPEGs and the PDF.
* **A proper PDF report** — composition name on top, `frame N · check` per page,
  every note numbered in its own colour, with the text size set on the node.
* **Stabilise** on the Input tab: paste a 2D tracker's x/y and the picture holds
  still on the reference frame.
* **Bounding box vs format.** An EXR whose data window is not its display
  window is placed by its format; the picture outside can be shown or cut, the
  format and the bbox outlined on demand, and the window's control strip turns
  red with *BBox is different then Canvas*.
* **Canvas check** has a cross where the edges meet — drag it to move the join.
* **Log view with camera logs.** Nuke's 16 log curves on the built-in
  transforms; under OCIO the log spaces of the config in use (ACES 1.3 and 2.0
  each have their own list), converted exactly. A log plate is linearised first,
  so it is never log of log.
* **CC like a Grade** — WhitePoint, BlackPoint and Gamma with the ranges and
  straight sliders of Nuke's Grade, followed by the scopes, the readout and
  the export.
* **Ctrl+J** in the Node Graph creates a JKplayer node, and a selected Read goes
  straight into Comp.
* **Comp/Plate swap on `;`** — the key left of 1 — taken from Nuke, which binds
  it itself, whenever the panel has the keyboard.
* **Custom OCIO config** on the node, copied over from Project Settings when the
  script uses one.
* **Timeline zooms out** past the shot on the wheel; a middle click fits it back.
* **Metadata follows playback** frame by frame, in step with the picture.
* **Input timing fixed:** Offset nudges an input against a timeline that stays
  put, instead of dragging the timeline along with the Comp.
* **Vectorscope** measures the plate, not the monitor LUT — like the histogram
  and the waveform.
* **Settings tab** on the node; the status line is hidden unless switched on
  there, and the RAM in use sits next to *Clear cache*.
* The Read's Frame tab (start at, offset, expression) no longer breaks the
  player — only the Read's path and file numbering are taken from it.

## Formats

**EXR** — through Nuke's own OpenEXR library when it is there (every
compression, and **5x faster at HD, 10x at 4K** on 8 threads), with a pure
Python reader as the fallback for NONE, ZIP and ZIPS. Both were checked against
each other and give bit-identical results. The gap widens with thread count,
because the fallback stops scaling past 8 while the library keeps going.

The **data window and display window** are read from the header of every frame
(a header only, not the pixels), so a render with an animated bounding box is
placed correctly on each frame — see *Bounding box* below.

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
Movies need **ffmpeg installed** — see Requirements.

A movie is one file for every frame, which the rest of the player cannot say —
it addresses frames by path. So internally a movie frame is written
`clip.mov|1042`. It is a small lie in a string, and it buys not touching the
cache key, the loader queue or the sequence cache at all.

Mind the **chroma**: a 4:2:2 codec carries half the colour resolution and 4:2:0
a quarter. It is reported in the metadata, because a difference check between an
EXR comp and a 4:2:0 delivery shows artefacts that are in the codec and not in
the comp.

## What it does

**Playback** — RAM cache with a byte budget (a quarter of the machine's memory
by default), look-ahead in the direction you are playing, a rolling cache
window so a long shot does not thrash, and a timeline that shows what is
cached. Realtime mode holds the target FPS and skips uncached frames; turn it
off and every frame is shown. The RAM this player's cache holds, against its
budget, is shown next to **Clear cache** (the tooltip adds every open player
together and the machine's total).

**Two inputs, six view modes** — Base, Sync (side by side or stacked),
Difference (A/B dissolve with a plain or high-pass compare), Wipe (draggable,
rotatable line with a blend handle), DiMatte (mattes drawn over the image as
colours) and Annotation. Both windows share zoom and pan, so the pixels line
up, and each picks its own input and its own EXR layer — so you can put rgba
against depth of the same plate.

The inputs are **Comp** and **Plate** (tagged `C` and `P` on the buttons inside
the image). `;` swaps the window between them, which is the quickest A/B there
is: same frame, same zoom, same place on the eye.

### Input tab

Each input has its own block on the node's **Input** tab.

* **Start at** places the input's first frame on a timeline frame — it
  renumbers. Start the Comp at 1001 and the timeline reads 1001 onwards.
* **Offset** nudges the input against the timeline, which stays where it is.
  Comp Offset +2 makes the Comp start two frames later while the Plate stays
  put; the timeline widens just enough that no nudged frame is cut off.

They put a 1-100 render under a 1001-1100 plate without a TimeOffset node. The
two inputs do not have to be the same length — the timeline follows Comp and
the shorter side holds its end frame. When the two were *delivered* on
different numbering, the plate's own numbers are written under the cache bars
so they can be lined up by eye.

From the Read only **the path and how the files are numbered** (its Original
Range) are taken. Its Frame tab — start at, offset, expression — and its Frame
Range do not move the player: that is what Start at and Offset here are for.

**Anamorphic** is there too, per input: `from file` trusts the pixel aspect in
the header, and the fixed ratios are for when it lies — a scan off 2x negative
is written square by plenty of scanners. It stretches the *drawing* only. The
pixels, the probe, the scopes and the notes all stay in stored coordinates.

**Colorspace** is per input as well — see Colour below.

**Stabilise** sits at the bottom of the tab, under a line. Paste the x/y
animation of a 2D tracker into **Track**, pick the **Reference frame**, switch
**Stabilise** on, and the picture is shifted so the tracked feature holds still
where it was on the reference frame. Only for looking at the result — nothing
is rendered, and the notes stay on the pixels they were drawn on.

### Bounding box

An EXR can carry a data window (the pixels) that is not its display window (the
format): overscan is bigger, a cropped render smaller. The player places every
frame by its **format** — fit and centre are the format's, so a changing bbox
does not move the shot about — and the node's **Settings** tab decides the
rest:

| Setting | Default | What it does |
|---|---|---|
| Show outside format | on | the picture outside the format is shown; off cuts it to the format |
| Format line | off | the format outlined with a faint solid line (25 % white) |
| Bbox line | off | the bounding box outlined dashed, as the Nuke Viewer does |
| Warn when bbox differs | on | the window's control strip turns **red** and says *BBox is different then Canvas* |

Nothing is drawn and nothing turns red when the bbox *is* the format. Inside the
format but outside a smaller data window is black, as in Nuke. The warning is
worked out per window and per frame, so in Sync only the input that differs
goes red.

### Annotation

Switch the view to **Annotation** and a tool bar appears in the image:

```
Export | ✎ pencil  rubber  T text | Undo  Clear
```

* **Pencil** draws; **Shift-drag** sizes it (a circle shows the width, and the
  Width number on the tool panel follows as you drag).
* **Rubber** takes out only the ink that passes under its circle — cross a line
  and it is cut in two. It never touches text.
* **Text** — click where the note belongs; click an existing note to edit it,
  drag it to move it; **Shift-drag** sizes the text. The note dialog has its own
  **Size** and **Visible in exported picture** — a note can live in the CSV and
  the PDF only, without covering the image.
* **Colour** — five quick colours and a multi-colour swatch that opens a palette
  of more, grouped by hue.
* **Undo** and **Clear** act on the current frame.

Notes are only drawn in Annotation mode. A note remembers WHICH view it was made
in, so a circle drawn around a grain problem does not reappear over a plain
plate. Frames with notes are marked in the timeline.

**Export** first opens the **crop dialog**: every annotated frame in a strip on
the left, the picture on the right. Drag to draw a crop, drag its edges or
corners to reshape it, drag inside to move it, click outside to clear it; wheel
zooms at the pointer, middle button pans, double click fits. **Same crop for
all** copies it to every frame; **Reset crop** (or `R`) removes it; arrow keys
step through the frames. The crop is scaled back up to the frame size.

It then writes, into a subfolder named after the clip:

* `annotation_####.jpg` per annotated frame and view, optionally with a frame
  stamp
* a **CSV** of every note
* a **PDF** report (switchable on the node) — the composition name on top,
  `frame N · check` per page, the cropped picture, and every note numbered
  `1.` `2.` `3.` in its own colour. Its text size is a knob on the node.

### DiMatte, QC modes, scopes

**DiMatte** takes its mattes either from a third input of its own or from a
**layer of the Comp EXR** — a comp that already carries them needs nothing
wired up, and then the node does not even grow the third input.

**QC modes** — Log view, Grain check, High-pass, Temporal (finds duplicate
frames), Saturation check, Value map and Canvas check; plus Difference and
High-pass difference in the Difference view. Each has its own sliders,
remembered per mode. A middle click on a slider resets it.

**Log view** shows the shot in a camera log, whatever the monitor is set to.
The curves on offer follow the colour management: Nuke's log curves (Cineon,
AlexaV3LogC, ARRILogC4, SLog3, Log3G10...) on the built-in transforms, and
under OCIO the log spaces of the config in use — ACES 1.3 and 2.0 each bring
their own list (ACEScct, ARRI LogC3/LogC4, S-Log3, V-Log...), converted exactly
through OCIO, gamut included. A log plate is taken to linear first, so it is
never shown as log of log.

**CC** — **WhitePoint** and **BlackPoint** as in a Grade, (in − black) /
(white − black) in scene-linear, plus **Gamma** and **Saturation**. The ranges
and straight sliders are Nuke's Grade: white like its gain (0–4), black like its
lift (−1 to 1), gamma 0.2–5; the number fields take anything in between exactly. The histogram, waveform and vectorscope, the
clipping line, the pixel readout and the exported notes all follow it.

**Canvas check** swaps the image so the original edges meet in the middle. A
cross marks the point where they meet: **drag it** to move the join anywhere,
and the Shift X / Shift Y sliders follow. Anywhere else the left button still
pans.

**Scopes** — histogram and waveform on a scene-linear axis from 0 to 55 with
the clipping line at 1.0, so you can see how far over an exposure goes, not
just that it clips. That axis is gamma-encoded below 1 and logarithmic above
it, with marks at 0.18, 1, 2, 4, 8, 16, 32 and 55. Vectorscope with the
standard 75 % / 100 % graticule.

All three measure **the plate through the input transform and CC — not the
monitor LUT**: switching the viewer from sRGB to rec709 moves none of them (the
vectorscope uses a fixed sRGB encoding so its targets still mean something). In
QC mode they measure the finished image instead. All three follow the visible
crop and mark the pixel under the cursor: a line on the histogram, a ring on the
vectorscope, a cross on the waveform.

The histogram and the waveform are resized by their edges or the bottom corner.
The vectorscope only scales square — a stretched circle would put the 75 %
boxes at two different radii.

**fMIN / fMAX** next to the FPS: the lowest and highest scene-linear value in
the whole frame, every pixel counted, through the input transform. Held frames
only.

**Metadata** — the headers of both inputs. The **META** button (or `M`) puts
them bottom left of the image, one panel per window. While playing they are
re-read **with every frame that is actually shown**, so a timecode counts up in
step with the picture rather than jumping every few frames; a frame that is not
cached yet shows neither a new picture nor new numbers.

Which fields, and in what order, is set on the node's Metadata tab. DPX gives
the most here — timecode, slate, keycode, frame position, the scanner and its
serial number.

### Colour

Nuke's built-in transforms (one table, fast) or full OCIO.

The built-in list is **Nuke's list, all 27 of it**: linear, sRGB, sRGBf,
rec709, Cineon, the four gammas, Panalog, REDLog, ViperLog, AlexaV3LogC,
PLogLin, SLog/1/2/3, CLog, Log3G10, Log3G12, HybridLogGamma, Protune, BT1886,
st2084, Blackmagic Film Generation 5 and ARRILogC4. The curves are transcribed
from Foundry's own generator — `make.py` in the `nuke-default` OCIO config —
and every one was checked back against that config through PyOpenColorIO over
the whole code range, in both directions. Where Foundry's code disagrees with
the spec sheet it is copied **as shipped**.

**OCIO configs** — the ones Nuke ships, or **custom**: pick *custom* in the
config menu and point **Custom config** at your own `config.ocio`. When the
script's Project Settings use a custom config, it is copied onto the node
following the same rules as everything else here. A path that does not load
falls back to `nuke-default` and says so.

The **input space belongs to the input, not to the player**. A log plate under
a linear comp of it is the ordinary delivery, so Comp and Plate each carry
their own. Picking a single channel (R, G, B, Luminance) goes through OCIO too.

Both sides are filled in for you, and neither takes the choice away:

* **Project Settings lead the display side, live.** Switch the script from
  Nuke to OCIO, change the config or the monitor LUT, and the player follows.
* **The Read leads the input side.** Nuke's own Input Transform on each Read
  becomes that input's colorspace.
* **What you set in the player holds** until the thing that fed it says
  something different. Both work on the *transition*, never on the standing
  value, so a node loaded out of a saved script is never trampled.

### Settings tab

| Setting | Default | |
|---|---|---|
| Show outside format | on | see *Bounding box* |
| Format line | off | |
| Bbox line | off | |
| Warn when bbox differs | on | |
| Show status line | off | the line at the very bottom: render and decode speed, resolution, cache fill, queue, zoom, OCIO, node name |

With the status line off it still **appears by itself when there is an error**
to report — a display error, a file that cannot be loaded, a disconnected
input — and goes again when it clears.

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
people on that exact Nuke and that exact OS. Fetching gets each Nuke the build
that is right for it.

They land in `pylibs/<platform>-cp<version>` next to the package, or in
`~/.nuke/pylibs/<tag>` when the install folder is read-only.

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
prints the pip command, and JKplayer does not load.

To do it by hand, run the Nuke you want to support:

```
<nuke>/python -m pip install --target pylibs/<tag> --only-binary=:all: numpy scipy
```

The tag is what `jkplayer.paths.platform_tag()` returns under that Nuke.

Without scipy the grain and high-pass checks fall back to a fixed 3x3 blur:
they still show something, but their sliders stop doing anything.

PyOpenColorIO and the OpenEXR libraries come with Nuke and are found next to
the running Nuke automatically.

### ffmpeg, for movies only

EXR and DPX are read by this package alone. `.mov`, `.mp4`, `.mxf` and `.m4v`
go through **ffmpeg**, which Nuke does not ship. Everything else works without
it.

**Both `ffmpeg` and `ffprobe` are needed.** ffprobe is the only one that
reports the frame rate as the exact ratio it is: ffmpeg rounds 24000/1001 to
"23.98", and since a seek is `frame / fps` that error grows to a whole frame by
six thousand in.

They are looked for in three places, first hit wins:

1. `$JKPLAYER_FFMPEG` — a folder, for a studio that keeps its own copy
2. `jkplayer/bin/` — a copy dropped in beside the code
3. `PATH`

When neither is found the panel says which one is missing and where it looked.

They are not bundled: `imageio-ffmpeg` ships ffmpeg without ffprobe, and
shipping both means picking a build. We only ever decode, so an LGPL build
covers it.

## Install

1. Copy this whole `JKplayer` folder into `~/.nuke/`
   (Windows: `%USERPROFILE%\.nuke\`).

2. Add one line to `~/.nuke/init.py`, creating the file if it is not there:

   ```python
   nuke.pluginAddPath("./JKplayer")
   ```

   The folder may be called anything and may live anywhere — only this line
   has to match.

3. Restart Nuke. If numpy is not there yet it offers to fetch it. Then there
   will be a **JKplayer** menu in the menu bar and a **JKplayer** entry in the
   Nodes toolbar.

After updating the code, **restart Nuke** — a running Nuke keeps the old
modules.

## Rolling it out to other people

For a managed install, prepare the libraries once and switch the download offer
off:

1. On one machine per (OS, Nuke version), fetch into the folder that ships:

   ```
   <nuke>/python -m pip install --target pylibs/<tag> --only-binary=:all: "numpy>=1.24,<3" "scipy>=1.10,<2"
   ```

   Nuke 16 and 17 share `cp311`; Nuke 14 is `cp39` and Nuke 15 `cp310`.

2. Put the folder on the share, **read-only** for artists.

3. Set `JKPLAYER_NO_FETCH=1`, or drop a file called `MANAGED` into `pylibs/`.
   Either turns the download offer into a console line naming the folder it
   expected. The *Install dependencies* menu entry refuses too.

On startup the console says which numpy and scipy were used and where they came
from.

## Use

1. **JKplayer > Create JKplayer Node** — or **Ctrl+J** in the Node Graph — and
   connect a Read to input **Comp** (optionally a second one to **Plate**, and a
   matte to **DiMatte** if the mattes come as their own files). An EXR or DPX
   sequence, or a movie. With a Read (or a Dot in front of one) **selected**,
   the new node comes already wired to it on Comp.
2. **Click the node** (or open its properties) and its panel opens, docked next
   to the Viewer. *JKplayer > Open JKplayer Panel* does the same for the
   selected node.
3. The node holds all the settings. Each JKplayer node has exactly one panel
   with its name; several players can be open side by side.

Only Read nodes pointing at a format this player decodes can be attached (a Dot
on the way is fine) — anything else is disconnected and a message says which
node it was. The player reads the files off disk itself, so a node that changes
the picture cannot be honoured. Shifting in time is done on the node, not with
a TimeOffset; fitting one input onto the other is done by the player when it
compares them, not with a Reformat.

A comparison across resolutions fits one onto the other and **says so** — a
difference over a resampled input is not the same measurement as one over two
plates that already match.

A Viewer cannot be attached either, and the **number keys do not create one** on
a JKplayer node. The node deliberately renders nothing — display is the
panel's job.

### The bar under the timeline

```
Handles [  ] | In [    ] Out [    ] Reset      Frame [    ] Play  Loop
```

**Handles** is how many frames were delivered either side of the cut; setting
it pulls IN and OUT in by that many. Moving either mark by hand puts the field
back to 0. **Reset** clears IN/OUT. **Frame** shows what is on screen and can be
typed into.

Narrow the panel and whole groups disappear rather than overlapping, from the
least useful end. **Play and the frame it is on never go.**

## Keys and mouse

### Node Graph

```
Ctrl + J           create a JKplayer node (the selected Read goes into Comp)
```

Plain `J` is left alone: in the Node Graph it is Nuke's *Jump to Bookmarked
Node*.

### Keyboard (in the panel)

```
J / K / L          play backwards / stop-play / play forwards
Left / Right       step one frame
R  G  B  A         show that channel (a second press returns to RGB)
Y                  luminance (a second press returns to RGB)
C                  CC panel (WhitePoint, BlackPoint, gamma, saturation)
Q                  QC panel
H                  histogram
V                  vectorscope
W                  waveform
M                  metadata (bottom left of the image)
1 - 7              QC mode: Log view, Grain, High-pass, Temporal,
                   Saturation, Value map, Canvas check
F                  fit into the window (fits the FORMAT)
I / O              mark IN / OUT at the current frame
P                  freeze the pixel readout
X                  switch the active window (in Sync)
;                  swap the window between Comp and Plate - the key left
                   of 1 (';' on a Czech layout, '`' on an English one);
                   taken from Nuke while the panel has the keyboard
```

### Image

```
wheel              zoom at the pointer
left drag          pan
double click       fit
Canvas check:
  drag the cross   move the join (Shift X / Shift Y follow)
Annotation:
  left drag        draw / rub out, with the pencil or the rubber
  click            place a note, with the text tool
  click a note     edit it
  drag a note      move it
  Shift-drag       size the pen or the text
  middle drag      pan while a tool is armed
```

### Timeline

```
click / drag       scrub
wheel              zoom in / out (out past the shot, darker outside it)
Ctrl + wheel       step one frame
middle drag        pan a zoomed timeline
middle click       back to the whole range
double click       back to the whole range
triangles          drag to move IN / OUT
```

### Sliders and number fields

```
middle click       reset to the default
Up / Right         +1   (Shift: +0.1)
Down / Left        -1   (Shift: -0.1)
Esc                leave the field, keep the value
Enter              apply and hand the keyboard back to playback
```

### Crop dialog (Export)

```
drag on picture    draw a crop
drag edge/corner   reshape it
drag inside        move it
click outside      clear it
wheel              zoom at the pointer
middle drag        pan
double click       fit
Left/Up, Right/Down   previous / next frame
R                  reset the crop
```

## Tests

They are not in here on purpose — installing a player should not drag along
test plates. They live in the development tree and point at this folder.

What they cover: both EXR readers against a reference set in every compression,
the DPX reader against files from a second implementation, movie frame identity
on clips that carry their own frame number, the colour transforms against OCIO,
the scopes, the QC modes, every knob on the node, and the parts Qt only draws,
rendered off-screen and checked pixel by pixel — the wipe mask, the annotation
bar and palette, the PDF pages, the crop dialog, the timeline, the canvas cross,
the bounding box lines and the red strip. The log view against OCIO's own
processors for every camera space, the CC grade through every path (built-in,
OCIO, scopes, export), and the swap key against a window that binds the same
keys the way Nuke does.

## What has actually been run

| | tested |
|---|---|
| Windows, Nuke 17 (PySide6, cp311) | yes — everything below, plus daily use |
| Windows, Nuke 16 | same interpreter and binding, expected to behave the same |
| Nuke 14 / 15 (PySide2 and PySide6 on cp39/cp310) | **no** — the Qt5 branch is written from the known differences but has not been run |
| Linux, macOS | **no** — the library lookup has patterns for both, never exercised |

**The colour transforms** are the strongest part, because there is something
exact to check against: all 27 are compared to the `nuke-default` OCIO config
over the whole code range. Decoding matches to 5e-5 at worst and around 1e-6
typically, and every one round-trips. The display direction matches on 26 of
27; the odd one out is HybridLogGamma near black, where Foundry's curve has no
inverse.

### The newer formats

| | how far it has been taken |
|---|---|
| DPX | against files written by ffmpeg at 8, 10 and 16 bit, both endiannesses, plus hand-built headers. **Not yet against a DPX from a real pipeline.** |
| ProRes, DNxHR, MJPEG | frame identity proved on clips with the frame number painted in as blocks: 96 frames each, every one by its own cold seek |
| H.264 / long-GOP | the same, with a 250-frame group of pictures |
| Both | concurrent access from several threads, the open-file cap, and no leaked processes |

Measured on this machine (Ryzen 9 9950X), decoding only:

| | 8 threads | 24 threads |
|---|---|---|
| HD EXR ZIP, Nuke's library | 285 fps | 401 fps |
| HD EXR ZIP, Python fallback | 58 fps | stops scaling past 8 |
| 4K EXR ZIP, Nuke's library | 68 fps | 103 fps |
| 4K EXR ZIP, Python fallback | 7 fps | stops scaling past 8 |
| 4K DPX 10-bit | | 25 fps at 16 threads |
| HD ProRes 422 HQ | | 72 fps |
| 4K ProRes 422 HQ | | 15–17 fps |

Material: ZIP16 half RGB, 7 MB a frame at HD and 28 at 4K, held warm in RAM so
these measure decoding rather than the disk. On SATA drives the disk decides
instead — a full cache fill there runs at 19.9 fps against 20.2 for reading the
same bytes with no decode at all.

4K movies are held back by the pipe out of ffmpeg, not by the decoder. Stepping
one frame on in a 4K movie costs 56 ms; a jump backwards restarts ffmpeg and
costs about half a second.

## Layout

```
jkplayer/
  panel.py      the panel - playback, layout, the whole UI
  imageview.py  one image window: display, pan/zoom, pixel probe, bbox
  overlay.py    the panels drawn inside the image (CC, QC, scopes, notes bar)
  timeline.py   timeline with the cache bar and mark in/out
  node.py       the node and every setting on it
  register.py   registration into Nuke's menus, one panel per node
  loader.py     background decoding, two queues
  cache.py      RAM cache with a byte budget
  sequence.py   frame number -> what the reader is handed
  exrcore.py    EXR through Nuke's own library (ctypes)
  exrread.py    pure Python EXR reader, the fallback
  dpxread.py    DPX reader
  movread.py    movies, through ffmpeg
  reader.py     picks between them - the whole format boundary
  meta.py       which header fields are shown, and in what order
  metaknob.py   the Metadata tab on the node
  scopes.py     histogram, vectorscope, waveform
  effects.py    the QC modes
  annotate.py   notes on frames: storage, drawing, export, CSV, PDF
  cropper.py    the crop dialog before export
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
