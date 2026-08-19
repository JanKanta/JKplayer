"""
OCIO display transform.

WHY THROUGH A 3D LUT AND NOT DIRECTLY:
The exact ACES path through the OCIO CPU processor runs at 12 Mpx/s - 170 ms
on 1080p, i.e. 6 fps. Unusable. A plain 3D LUT, however, is applied by OCIO
tetrahedrally at 229 Mpx/s (9 ms on 1080p), so we bake the transform into a
cube ONCE and then only interpolate. That is exactly what every player does,
just on the GPU.

THE PATH OF ONE FRAME:
    half bits --(table lookup)---> float32 position 0..1  (shaper + gain)
              --(OCIO 3D LUT)----> float32 0..1           (display transform)
              --(conversion)-----> uint8                  (image)
The shaper is a log2 table over all 65536 half float values, so the conversion
from half, the exposure and the shaper together cost a single array lookup.

ACCURACY (measured against exact OCIO, 64^3 cube):
    realistic footage   mean 0.15 of an 8bit level, max 9, above 1 level 0.10 %
    saturated colours   mean 0.22, max 37
The error is concentrated in the sharp bends of gamut mapping at extremely
saturated colours; on ordinary material it is below the observable threshold.
"""

import os

import numpy as np

from .paths import nuke_subdir

try:
    import PyOpenColorIO as _OCIO
except Exception:                       # Nuke has it, but do not crash
    _OCIO = None

# Shaper range in stops and the cube size.
#
# The cube axis is NOT a pure logarithm: the first node is EXACTLY ZERO and
# only from the second one does log2 start at 2^LOG_MIN. Without that the
# blacks would lift - the shaper would pin zero to 2^-12 and something like
# rec1886 (pure gamma 2.4 with no toe) turns that into 8/255, so black is not
# black (measured: 0.0 -> 7 instead of 0).
#
# SECOND CONDITION: the value 1.0 has to land EXACTLY ON A NODE. Most display
# transforms bend into a clip there and when the node does not line up, the
# interpolation undershoots - white is not white (measured: 1.0 -> 249
# instead of 255).
#     node index for 1.0 = 1 + (CUBE_SIZE-2) * (-LOG_MIN)/(LOG_MAX-LOG_MIN)
#                        = 1 + 94 * 12/24 = 48   (a whole number, it fits)
LOG_MIN, LOG_MAX = -12.0, 12.0

# 96^3. APPLY time barely depends on the cube size (measured 1.1 ms for 64^3
# vs 1.2 ms for 96^3 on 512x512), only the bake time grows - and that happens
# only when the choice changes. A denser cube is therefore worth it:
#     64^3   bake  4-35 ms,  mean error 0.25-1.50 levels
#     96^3   bake  9-114 ms, mean error 0.21-1.02
#    128^3   bake 21-267 ms, mean error 0.03-0.62
# 96^3 is the compromise: the worst case (ACES SDR Video) bakes in 114 ms,
# ordinary nuke-default views in under 30 ms.
CUBE_SIZE = 96


_HALF = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)
_HALF_SAFE = np.nan_to_num(_HALF, nan=0.0, posinf=2.0 ** LOG_MAX, neginf=0.0)

# The configs Nuke ships, found next to the running Nuke (see paths.nuke_dirs)
# so it works on every platform and every Nuke version. None when there are
# none - a stripped install, or running outside Nuke; the OCIO environment
# variable is then the only source.
NUKE_CONFIG_DIR = nuke_subdir("plugins", "OCIOConfigs", "configs")

DEFAULT_CONFIG = "nuke-default"     # what we run on by default

# TWO INDEPENDENT THINGS, exactly as in Nuke:
#
#   INPUT SPACE      - what the data in the FILE is in. EXR is usually
#                      scene-linear, but log can turn up too (AlexaV3LogC,
#                      SLog3, Cineon...). The conversion goes from there INTO
#                      linear.
#   DISPLAY + VIEW   - how it is drawn on the monitor (Viewer Process in Nuke).
#                      The conversion goes from linear ONTO the screen.
#
# So the whole path is: file -> [input space] -> linear -> [view] -> monitor


def available():
    return _OCIO is not None


def version():
    return _OCIO.GetVersion() if _OCIO is not None else None


# ---------------------------------------------------------------------------
# Finding configs
# ---------------------------------------------------------------------------
def find_configs():
    """[(label, path)] - the OCIO variable, Nuke's configs, the built-in ones."""
    found = []
    env = os.environ.get("OCIO")
    if env and os.path.isfile(env):
        found.append(("$OCIO: %s" % os.path.basename(env), env))
    if NUKE_CONFIG_DIR and os.path.isdir(NUKE_CONFIG_DIR):
        for name in sorted(os.listdir(NUKE_CONFIG_DIR)):
            path = os.path.join(NUKE_CONFIG_DIR, name)
            if name.endswith(".ocio"):
                found.append((name[:-5], path))
            elif os.path.isdir(path):
                inner = os.path.join(path, "config.ocio")
                if os.path.isfile(inner):
                    found.append((name, inner))
    return found


def default_config_index(configs=None):
    """Index of the nuke-default config (or 0 when it is not there)."""
    configs = configs if configs is not None else find_configs()
    for i, (label, _path) in enumerate(configs):
        if label == DEFAULT_CONFIG:
            return i
    return 0


class OcioError(Exception):
    pass


# distinguishes "not computed yet" from "computed and came out None"
_UNSET = object()


# ---------------------------------------------------------------------------
class DisplayTransform(object):
    """One baked combination of config + display + view.

    Holds the 3D LUT and the shaper. An exposure change only rebuilds the
    shaper (1 ms), a display/view change requires re-baking the cube (~46 ms).
    """

    def __init__(self, config_path):
        if _OCIO is None:
            raise OcioError("PyOpenColorIO is not available")
        self.config_path = config_path
        try:
            self._config = _OCIO.Config.CreateFromFile(config_path)
        except Exception as exc:
            raise OcioError("cannot load the config: %s" % exc)
        # CAREFUL: OCIO caches processors by the fingerprint of the transform
        # and for Lut3DTransform the fingerprints of different cubes collide -
        # after a view change it then returns the OLD processor and the image
        # does not change (measured: switching to Un-tone-mapped kept
        # returning ACES). So we turn the cache off; we only bake when the
        # choice changes, so it costs us nothing.
        try:
            self._config.setProcessorCacheFlags(_OCIO.PROCESSOR_CACHE_OFF)
        except Exception:
            pass
        self._cpu = None
        self._to_lin = None            # input space -> linear (for the scopes)
        self._lin_table = _UNSET       # the same as a table, for the QC checks
        self._fold_input = True        # the shaper linearises, not the cube
        self._shaper = None
        self._gain = None
        self.display = None
        self.view = None
        self.input_space = self._pick_input_space()
        self.default_input = self.input_space

    # -------------------------------------------------------------- listings
    def _pick_input_space(self):
        """Default input: scene-linear, because that is how EXR is usually written."""
        for role in (_OCIO.ROLE_SCENE_LINEAR, _OCIO.ROLE_REFERENCE):
            try:
                cs = self._config.getColorSpace(role)
                if cs is not None:
                    return cs.getName()
            except Exception:
                pass
        return _OCIO.ROLE_SCENE_LINEAR

    def displays(self):
        try:
            return list(self._config.getDisplays())
        except Exception:
            return []

    def views(self, display):
        try:
            return list(self._config.getViews(display))
        except Exception:
            return []

    def default_display(self):
        try:
            return self._config.getDefaultDisplay()
        except Exception:
            return ""

    def default_view(self, display):
        try:
            return self._config.getDefaultView(display)
        except Exception:
            return ""

    def colorspaces(self):
        """All colour spaces - the list offered for the INPUT transform."""
        try:
            return [c.getName() for c in self._config.getColorSpaces()]
        except Exception:
            return []

    def input_spaces(self):
        """Spaces for the input, with scene-linear and data first (most common)."""
        spaces = self.colorspaces()
        common = [s for s in self._role_spaces() if s in spaces]
        return common + [s for s in spaces if s not in common]

    def _role_spaces(self):
        """Spaces from the scene_linear and data roles - so it works even for
        configs where they are named differently than linear/raw (in ACES they
        are ACEScg and Raw)."""
        out = []
        for role in (_OCIO.ROLE_SCENE_LINEAR, _OCIO.ROLE_DATA):
            try:
                cs = self._config.getColorSpace(role)
                if cs is not None and cs.getName() not in out:
                    out.append(cs.getName())
            except Exception:
                pass
        return out

    def display_views(self):
        """[(label, display, view)] - every device/view combination.

        The label has the device in brackets, just like Viewer Process in Nuke:
        'sRGB (default)', 'ACES 1.0 - SDR Video (Rec.1886 Rec.709 - Display)'.
        """
        out = []
        for d in self.displays():
            for v in self.views(d):
                out.append(("%s (%s)" % (v, d), d, v))
        return out

    # ---------------------------------------------------------------- baking
    def bake(self, display, view, input_space=None):
        """Bakes the whole input -> linear -> monitor path into a 3D LUT.

        Called only when the choice changes (baking costs tens of ms).
        """
        if input_space and input_space != self.input_space:
            self.input_space = input_space
            self._to_lin = None                # different input = different conversion
            self._lin_table = _UNSET

        # The input conversion is better BAKED INTO THE SHAPER than into the cube.
        #
        # The cube is log-uniform because it expects scene-linear data. Bake a
        # log input into it and it gets values 0-1 on its input, so most of the
        # nodes are wasted on numbers around zero - exactly where nothing
        # happens. Measured on Cineon: up to 11 8bit levels of error just below
        # white. When the shaper does the linearisation, the cube still sees
        # scene-linear and the resolution lines up. On top of that, exposure
        # then behaves correctly, since it must be applied AFTER linearisation.
        self._fold_input = self.is_linear_input() or self.linear_table() is not None
        exact = self._exact_processor(display, view)
        axis = self.grid_values(CUBE_SIZE)
        grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
        lattice = np.ascontiguousarray(grid.reshape(-1, 1, 3).copy())
        exact.apply(_OCIO.PackedImageDesc(lattice, 1, CUBE_SIZE ** 3, 3))

        lut = _OCIO.Lut3DTransform(gridSize=CUBE_SIZE)
        lut.setData(np.ascontiguousarray(np.clip(lattice, 0.0, 1.0).ravel()))
        lut.setInterpolation(_OCIO.INTERP_TETRAHEDRAL)
        self._cpu = self._config.getProcessor(lut).getOptimizedCPUProcessor(
            _OCIO.BIT_DEPTH_F32, _OCIO.BIT_DEPTH_F32, _OCIO.OPTIMIZATION_DEFAULT)
        self.display, self.view = display, view
        self._shaper, self._gain = None, None      # the shaper is built in apply

    def _exact_processor(self, display, view):
        # when the shaper linearises, the cube only knows linear -> monitor
        space = self.default_input if self._fold_input else self.input_space
        try:
            proc = self._config.getProcessor(space, display, view,
                                             _OCIO.TRANSFORM_DIR_FORWARD)
            return proc.getOptimizedCPUProcessor(
                _OCIO.BIT_DEPTH_F32, _OCIO.BIT_DEPTH_F32,
                _OCIO.OPTIMIZATION_DEFAULT)
        except Exception as exc:
            raise OcioError("%s / %s (input %s): %s"
                            % (display, view, self.input_space, exc))

    @staticmethod
    def grid_values(n):
        """Scene-linear values of the cube nodes.

        Node 0 is exactly zero, the rest are log-uniform from 2^LOG_MIN up -
        see the comment at LOG_MIN; without that zero the blacks lift.
        """
        out = np.empty(n, dtype=np.float32)
        out[0] = 0.0
        t = np.linspace(0.0, 1.0, n - 1, dtype=np.float32)
        out[1:] = 2.0 ** (LOG_MIN + t * (LOG_MAX - LOG_MIN))
        return out

    def _shaper_table(self, gain):
        """65536 -> position in the cube. Input conversion and exposure live here.

        The order matters: linearise the input first, only then multiply by
        exposure - it applies to scene-linear values, not to a log code.

        The first cell of the cube covers 0 to 2^LOG_MIN linearly (so that zero
        lands on zero), from 2^LOG_MIN up it is log2.
        """
        v = _HALF_SAFE
        if self._fold_input and not self.is_linear_input():
            table = self.linear_table()
            if table is not None:
                v = table.astype(np.float32)
        v = v * float(gain)
        lo = 2.0 ** LOG_MIN
        first = 1.0 / (CUBE_SIZE - 1)          # width of the first cell in positions
        log_pos = first + (1.0 - first) * (
            (np.log2(np.maximum(v, lo)) - LOG_MIN) / (LOG_MAX - LOG_MIN))
        pos = np.where(v <= 0.0, 0.0,
                       np.where(v < lo, v / lo * first, log_pos))
        return np.clip(pos, 0.0, 1.0).astype(np.float32)

    # ----------------------------------------------------------------- usage
    def ready(self):
        return self._cpu is not None

    def apply(self, half_rgb, gain=1.0):
        """(h,w,3) half scene-linear -> (h,w,3) uint8 ready for display."""
        if self._cpu is None:
            raise OcioError("nothing is baked, call bake()")
        if self._shaper is None or self._gain != gain:
            self._shaper = self._shaper_table(gain)
            self._gain = gain
        # A table lookup can read from a NON-CONTIGUOUS slice and its output is
        # always contiguous, so the extra copy is pointless - and it costs 27 %
        # of the whole display (measured on a 6K crop: 9.1 -> 6.6 ms).
        try:
            bits = half_rgb.view(np.uint16)
        except (TypeError, ValueError):
            bits = np.ascontiguousarray(half_rgb).view(np.uint16)
        buf = self._shaper[bits]
        h, w = buf.shape[0], buf.shape[1]
        self._cpu.apply(_OCIO.PackedImageDesc(buf, w, h, 3))
        np.clip(buf, 0.0, 1.0, out=buf)
        buf *= 255.0
        return buf.astype(np.uint8)

    def is_linear_input(self):
        """Is the input space already scene-linear? Then there is nothing to convert."""
        return self.input_space == self.default_input

    def to_linear(self, rgb_f32):
        """Converts the input space to scene-linear, in place.

        For the scopes: the histogram axis is scene-linear, so when the file
        holds a log recording it has to be linearised first - otherwise it
        would show encoded values and the clipping boundary at 1.0 would mean
        nothing.

        Done DIRECTLY through OCIO, without the cube: the scopes work from
        subsampled data (~150k pixels), so even the slow path costs a fraction
        of a millisecond.
        """
        if self.is_linear_input() or rgb_f32 is None or rgb_f32.size == 0:
            return rgb_f32
        if self._to_lin is None:
            try:
                proc = self._config.getProcessor(self.input_space,
                                                 self.default_input)
                self._to_lin = proc.getOptimizedCPUProcessor(
                    _OCIO.BIT_DEPTH_F32, _OCIO.BIT_DEPTH_F32,
                    _OCIO.OPTIMIZATION_DEFAULT)
            except Exception:
                return rgb_f32
        h, w = rgb_f32.shape[0], rgb_f32.shape[1]
        self._to_lin.apply(_OCIO.PackedImageDesc(rgb_f32, w, h, 3))
        return rgb_f32

    def linear_table(self):
        """65536 -> float16: raw value from the file -> scene-linear.

        Returns None when the input is already linear (nothing to do) or when
        the conversion mixes channels together - a per-channel table would lie
        there and the exact path through to_linear() has to be used.

        Why a table: the QC checks need linearised data (grain computed from
        log values measures nonsense), but it must not cost time every frame.
        This way it is a single array lookup, exactly like the shaper.
        """
        if self.is_linear_input():
            return None
        if self._lin_table is _UNSET:
            self._lin_table = self._build_linear_table()
        return self._lin_table

    def _build_linear_table(self):
        ramp = np.ascontiguousarray(
            np.repeat(_HALF_SAFE.reshape(-1, 1, 1), 3, axis=2).astype(np.float32))
        self.to_linear(ramp)
        # clip before converting to half, otherwise large values overflow to inf
        vals = np.nan_to_num(ramp[:, 0, 0], nan=0.0, posinf=2.0 ** LOG_MAX,
                             neginf=0.0)
        table = np.clip(vals, -65504.0, 65504.0).astype(np.float16)
        if not self._separable(table):
            return None                   # the table would lie, go the exact way
        return table

    def _separable(self, table):
        """Can the conversion be done per channel? Log recordings yes, gamut matrices no."""
        probe = np.ascontiguousarray(np.array(
            [[[0.2, 0.5, 0.9], [0.05, 0.8, 0.3], [1.0, 0.1, 0.6],
              [0.02, 0.02, 0.02]]], dtype=np.float32))
        exact = self.to_linear(probe.copy())
        approx = table[probe.astype(np.float16).view(np.uint16)].astype(np.float32)
        scale = max(1e-3, float(np.abs(exact).max()))
        return float(np.abs(exact - approx).max()) / scale < 0.01

    def label(self):
        txt = "%s (%s)" % (self.view or "-", self.display or "-")
        if self.input_space != self.default_input:
            txt = "input %s -> %s" % (self.input_space, txt)
        return txt
