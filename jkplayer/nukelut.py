"""
Built-in colour transforms - the "Nuke" mode, i.e. without OCIO.

WHY ALONGSIDE OCIO: it is substantially faster. OCIO goes through a 3D cube
(28 ms on 1080p), this is a single table over the bits of a half float (9 ms).
On 6K playback that is the difference between comfort and a crawl.

WHERE THE CURVES COME FROM: Foundry's own generator, make.py in
plugins/OCIOConfigs/configs/nuke-default, which is the script that wrote the
.spi1d files the nuke-default config uses. Every one is then checked back
against that config over the whole code range, in both directions. The list is
Nuke's list, name for name.

The one place the check cannot reach is the display direction of
HybridLogGamma near black: Foundry's curve is x*x/3 below encoded 0.5, so it is
symmetric about zero and the table it produces is not monotonic - encoded
-0.125 gives MORE light than encoded 0. OCIO can only invert that by guessing,
which is where the two answers part. Ours is the exact inverse of Foundry's own
formula, and it round-trips; OCIO's is an artefact of inverting a table that
has no inverse there.

The order is the same as in Nuke: each transform works forwards (scene-linear
-> encoded, used for display) and backwards (encoded -> scene-linear, used as
the input transform).
"""

import numpy as np

_HALF = np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32)
_HALF_SAFE = np.nan_to_num(_HALF, nan=0.0, posinf=65504.0, neginf=-65504.0)


# ---------------------------------------------------------------------------
# Transform functions. Input and output are float32 arrays.
# ---------------------------------------------------------------------------
def _pow10(x):
    """10^x with a clamped exponent.

    Log transforms are computed over ALL half float values, including numbers
    far beyond the valid code range (0-1). Without the clamp 10^x overflows
    float32 there and numpy reports an overflow; 10^30 is way past the half
    range anyway.
    """
    return np.power(10.0, np.clip(x, -30.0, 30.0))


def _srgb_fwd(v):
    v = np.clip(v, 0.0, None)
    return np.where(v <= 0.0031308, v * 12.92,
                    1.055 * np.power(v, 1.0 / 2.4) - 0.055)


def _srgb_inv(v):
    return np.where(v <= 0.04045, v / 12.92,
                    np.power(np.clip((v + 0.055) / 1.055, 0.0, None), 2.4))


def _rec709_fwd(v):
    v = np.clip(v, 0.0, None)
    return np.where(v < 0.018, v * 4.5, 1.099 * np.power(v, 0.45) - 0.099)


def _rec709_inv(v):
    return np.where(v < 0.081, v / 4.5,
                    np.power(np.clip((v + 0.099) / 1.099, 0.0, None), 1.0 / 0.45))


def _gamma_fwd(g):
    return lambda v: np.power(np.clip(v, 0.0, None), 1.0 / g)


def _gamma_inv(g):
    return lambda v: np.power(np.clip(v, 0.0, None), g)


# Cineon: 10bit log, black 95, white 685, gamma 0.6 (as in Nuke)
_CN_BLACK, _CN_WHITE, _CN_GAMMA = 95.0, 685.0, 0.6
_CN_OFF = 10.0 ** ((_CN_BLACK - _CN_WHITE) * 0.002 / _CN_GAMMA)


def _cineon_inv(v):
    """encoded 0-1 -> scene-linear"""
    c10 = v * 1023.0
    return (_pow10((c10 - _CN_WHITE) * 0.002 / _CN_GAMMA) - _CN_OFF) / (1.0 - _CN_OFF)


def _cineon_fwd(v):
    lin = np.clip(v, 0.0, None) * (1.0 - _CN_OFF) + _CN_OFF
    c10 = np.log10(np.maximum(lin, 1e-10)) * _CN_GAMMA / 0.002 + _CN_WHITE
    return c10 / 1023.0


# ARRI LogC v3, EI 800
_LC = dict(cut=0.010591, a=5.555556, b=0.052272, c=0.247190,
           d=0.385537, e=5.367655, f=0.092809)


def _logc_fwd(v):
    p = _LC
    return np.where(v > p["cut"],
                    p["c"] * np.log10(np.maximum(p["a"] * v + p["b"], 1e-10))
                    + p["d"],
                    p["e"] * v + p["f"])


def _logc_inv(v):
    p = _LC
    return np.where(v > p["e"] * p["cut"] + p["f"],
                    (_pow10((v - p["d"]) / p["c"]) - p["b"]) / p["a"],
                    (v - p["f"]) / p["e"])


# Sony S-Log3
def _slog3_fwd(v):
    return np.where(v >= 0.01125000,
                    (420.0 + np.log10(np.maximum((v + 0.01) / (0.18 + 0.01),
                                                 1e-10)) * 261.5) / 1023.0,
                    (v * (171.2102946929 - 95.0) / 0.01125000 + 95.0) / 1023.0)


def _slog3_inv(v):
    return np.where(v >= 171.2102946929 / 1023.0,
                    (_pow10((v * 1023.0 - 420.0) / 261.5) * (0.18 + 0.01) - 0.01),
                    (v * 1023.0 - 95.0) * 0.01125000 / (171.2102946929 - 95.0))


# RED Log3G10
_G10_A, _G10_B, _G10_C, _G10_G = 0.224282, 155.975327, 0.01, 15.1927
_G10_OFF = 0.01


def _log3g10_fwd(v):
    x = v + _G10_OFF
    return np.where(x < 0.0, x * _G10_G,
                    _G10_A * np.log10(np.maximum(x * _G10_B + 1.0, 1e-10)))


def _log3g10_inv(v):
    return np.where(v < 0.0, v / _G10_G,
                    (_pow10(v / _G10_A) - 1.0) / _G10_B) - _G10_OFF


# ---------------------------------------------------------------------------
# THE REST OF NUKE'S LIST.
#
# Every curve below is transcribed from Foundry's own generator - make.py in
# plugins/OCIOConfigs/configs/nuke-default - which is the script that wrote the
# .spi1d files the nuke-default config uses. So these are not a spec sheet
# remembered, they are Nuke's definition - and each was then checked back
# against that config through PyOpenColorIO over the whole code range, in both
# directions. Anything changed here should be checked the same way.
#
# Where Foundry's code looks wrong it is still copied AS SHIPPED. Two places
# are worth naming, because they will look like typos on the next read:
#
#   * SLog1 computes its own curve and then writes fromSLog() into slog1.spi1d.
#     So in Nuke, SLog1 IS SLog. Ours matches what Nuke does, not what the
#     comment above it intended.
#   * CLog reads 10**(x - 0.0730597)/0.529136, and Python divides AFTER the
#     power. Written the way a Canon spec would suggest it would be
#     10**((x - 0.0730597)/0.529136) - a different curve, and not the one in
#     the file Nuke ships.
#
# Getting "closer to correct" than Nuke would mean our picture and Nuke's
# disagree, which is the one thing a review player must not do.
# ---------------------------------------------------------------------------


def _log_offset(code_black, code_white, span):
    """The Cineon family: 10**((1023x - white)/span), black subtracted off."""
    black = 10.0 ** ((code_black - code_white) / span)

    def fwd(v):
        inner = np.maximum(v * (1.0 - black) + black, 1e-10)
        return (np.log10(inner) * span + code_white) / 1023.0

    def inv(v):
        return (_pow10((1023.0 * v - code_white) / span) - black) / (1.0 - black)
    return fwd, inv


# Sony/Panavision Genesis, RED Log - the same shape as Cineon, other constants
_panalog_fwd, _panalog_inv = _log_offset(64.0, 681.0, 444.0)
_redlog_fwd, _redlog_inv = _log_offset(0.0, 1023.0, 511.0)


# Viper - no black offset at all
def _viperlog_inv(v):
    return _pow10((1023.0 * v - 1023.0) / 500.0)


def _viperlog_fwd(v):
    return (np.log10(np.maximum(v, 1e-10)) * 500.0 + 1023.0) / 1023.0


# Josh Pines pivoted log/lin, 445 -> 0.18
_PLL_REF, _PLL_LOG, _PLL_NG, _PLL_DPCV = 0.18, 445.0, 0.6, 0.002
_PLL_D_OVER_N = _PLL_DPCV / _PLL_NG


def _ploglin_inv(v):
    return _pow10((v * 1023.0 - _PLL_LOG) * _PLL_D_OVER_N) * _PLL_REF


def _ploglin_fwd(v):
    ratio = np.maximum(v / _PLL_REF, 1e-10)
    return (np.log10(ratio) / _PLL_D_OVER_N + _PLL_LOG) / 1023.0


# Sony S-Log. SLog1 is the same curve in Nuke - see the note at the top.
def _slog_inv(v):
    return _pow10((v - 0.616596 - 0.03) / 0.432699) - 0.037584


def _slog_fwd(v):
    return (np.log10(np.maximum(v + 0.037584, 1e-10)) * 0.432699
            + 0.616596 + 0.03)


# Sony S-Log2
_SLOG2_CUT = 0.030001222851889303


def _slog2_inv(v):
    i = (v - 0.06256) / 0.8563
    high = (219.0 * (_pow10((i - 0.616596 - 0.03) / 0.432699) - 0.037584)
            / 155.0) * 0.9
    low = (i - _SLOG2_CUT) * 0.28258064516129 * 0.9
    return np.where(i >= _SLOG2_CUT, high, low)


def _slog2_fwd(v):
    ratio = np.maximum(v / 0.9 * 155.0 / 219.0 + 0.037584, 1e-10)
    high = np.log10(ratio) * 0.432699 + 0.616596 + 0.03
    low = v / 0.9 / 0.28258064516129 + _SLOG2_CUT
    i = np.where(v >= 0.0, high, low)
    return i * 0.8563 + 0.06256


# Canon CLog - the precedence is Foundry's, see the note at the top
def _clog_inv(v):
    return ((_pow10(v - 0.0730597) / 0.529136) - 1.0) / 10.1596


def _clog_fwd(v):
    inner = np.maximum((v * 10.1596 + 1.0) * 0.529136, 1e-10)
    return np.log10(inner) + 0.0730597


# RED Log3G12
def _log3g12_inv(v):
    s = np.where(v < 0.0, -1.0, 1.0)
    return s * (_pow10(np.abs(v) / 0.184904) - 1.0) / 347.189667


def _log3g12_fwd(v):
    s = np.where(v < 0.0, -1.0, 1.0)
    return s * np.log10(np.maximum(np.abs(v) * 347.189667 + 1.0,
                                   1e-10)) * 0.184904


# GoPro Protune
def _protune_inv(v):
    return (np.power(113.0, v) - 1.0) / 112.0


def _protune_fwd(v):
    return np.log(np.maximum(v * 112.0 + 1.0, 1e-10)) / np.log(113.0)


# ITU-R BT.2100 Hybrid Log-Gamma
def _hlg_inv(v):
    return np.where(v < 0.5, v * v / 3.0,
                    np.exp((v - 1.00429347) / 0.17883277) + 0.02372241)


def _hlg_fwd(v):
    low = np.sqrt(np.maximum(v, 0.0) * 3.0)
    high = np.log(np.maximum(v - 0.02372241, 1e-10)) * 0.17883277 + 1.00429347
    return np.where(v < 1.0 / 12.0, low, high)


# SMPTE ST 2084 (PQ)
def _st2084_inv(v):
    s = np.where(v < 0.0, -1.0, 1.0)
    m = np.maximum(np.power(np.abs(v), 1.0 / 78.84375), 0.8359375)
    top = np.maximum(m - 0.8359375, 0.0)
    bottom = 18.8515625 - m * 18.6875
    return s * 10000.0 * np.power(top / bottom, 6.277394636)


def _st2084_fwd(v):
    s = np.where(v < 0.0, -1.0, 1.0)
    L = np.power(np.maximum(np.abs(v) / 10000.0, 0.0), 1.0 / 6.277394636)
    m = (0.8359375 + 18.8515625 * L) / (1.0 + 18.6875 * L)
    return s * np.power(m, 78.84375)


# ARRI LogC4
_C4_A = (2.0 ** 18.0 - 16.0) / 117.45
_C4_B = (1023.0 - 95.0) / 1023.0
_C4_C = 95.0 / 1023.0
_C4_T = (2.0 ** (14.0 * (-_C4_C / _C4_B) + 6.0) - 64.0) / _C4_A


def _logc4_inv(v):
    pos = (np.power(2.0, 14.0 * (v - _C4_C) / _C4_B + 6.0) - 64.0) / _C4_A
    neg = -((np.power(2.0, 14.0 * (-v - _C4_C) / _C4_B + 6.0) - 64.0) / _C4_A)
    return np.where(v >= 0.0, pos, neg + 2.0 * _C4_T)


def _logc4_fwd(v):
    def code(x):
        inner = np.maximum(x * _C4_A + 64.0, 1e-10)
        return (np.log2(inner) - 6.0) * _C4_B / 14.0 + _C4_C
    return np.where(v >= _C4_T, code(v), -code(2.0 * _C4_T - v))


# Blackmagic Film Generation 5
_BM_A = 0.08692876065491224
_BM_B = 0.005494072432257808
_BM_C = 0.5300133392291939
_BM_D = 8.283605932402494
_BM_E = 0.09246575342465753
_BM_LIN_CUT = 0.005
_BM_LOG_CUT = _BM_D * _BM_LIN_CUT + _BM_E


def _bmfilm5_inv(v):
    return np.where(v < _BM_LOG_CUT, (v - _BM_E) / _BM_D,
                    np.exp((v - _BM_C) / _BM_A) - _BM_B)


def _bmfilm5_fwd(v):
    return np.where(v < _BM_LIN_CUT, _BM_D * v + _BM_E,
                    _BM_A * np.log(np.maximum(v + _BM_B, 1e-10)) + _BM_C)


# ---------------------------------------------------------------------------
# The list of transforms. Ordered as in Nuke.
# ---------------------------------------------------------------------------
# name -> (forward linear->encoded, backward encoded->linear)
TRANSFORMS = [
    ("linear", (None, None)),                 # None = identity
    ("sRGB", (_srgb_fwd, _srgb_inv)),
    ("sRGBf", (_srgb_fwd, _srgb_inv)),        # sRGB, float range kept
    ("rec709", (_rec709_fwd, _rec709_inv)),
    ("Cineon", (_cineon_fwd, _cineon_inv)),
    ("Gamma1.8", (_gamma_fwd(1.8), _gamma_inv(1.8))),
    ("Gamma2.2", (_gamma_fwd(2.2), _gamma_inv(2.2))),
    ("Gamma2.4", (_gamma_fwd(2.4), _gamma_inv(2.4))),
    ("Gamma2.6", (_gamma_fwd(2.6), _gamma_inv(2.6))),
    ("Panalog", (_panalog_fwd, _panalog_inv)),
    ("REDLog", (_redlog_fwd, _redlog_inv)),
    ("ViperLog", (_viperlog_fwd, _viperlog_inv)),
    ("AlexaV3LogC", (_logc_fwd, _logc_inv)),
    ("PLogLin", (_ploglin_fwd, _ploglin_inv)),
    ("SLog", (_slog_fwd, _slog_inv)),
    ("SLog1", (_slog_fwd, _slog_inv)),        # Nuke ships SLog here - see above
    ("SLog2", (_slog2_fwd, _slog2_inv)),
    ("SLog3", (_slog3_fwd, _slog3_inv)),
    ("CLog", (_clog_fwd, _clog_inv)),
    ("Log3G10", (_log3g10_fwd, _log3g10_inv)),
    ("Log3G12", (_log3g12_fwd, _log3g12_inv)),
    ("HybridLogGamma", (_hlg_fwd, _hlg_inv)),
    ("Protune", (_protune_fwd, _protune_inv)),
    ("BT1886", (_gamma_fwd(2.4), _gamma_inv(2.4))),
    ("st2084", (_st2084_fwd, _st2084_inv)),
    ("Blackmagic Film Generation 5", (_bmfilm5_fwd, _bmfilm5_inv)),
    ("ARRILogC4", (_logc4_fwd, _logc4_inv)),
    # OUR OLD NAME FOR BT1886, kept so scripts saved before Nuke's list was
    # matched still resolve. It is last so it never wins a lookup by value,
    # and it is left out of INPUT_NAMES so it cannot be picked afresh.
    ("rec1886", (_gamma_fwd(2.4), _gamma_inv(2.4))),
]
_BY_NAME = dict(TRANSFORMS)
_LEGACY = ("rec1886",)

# One curve, two spellings. Nuke's COLORSPACE for gamma 2.4 is BT1886 and
# "rec1886" is only the name of a VIEW - but rec1886 is what this player called
# it before the list was matched to Nuke's, so it is what sits in scripts saved
# until now. Kept as a match so those still resolve, and so a project or a Read
# that says either word lands on the same transform.
ALIASES = {"rec1886": "BT1886"}

# What to offer as a display (viewer process) - log spaces make no sense
# there, they are not meant to be sent to a monitor.
DISPLAY_NAMES = ["sRGB", "rec709", "BT1886", "Gamma2.2", "Gamma2.4", "linear"]
INPUT_NAMES = [name for name, _fn in TRANSFORMS if name not in _LEGACY]

# The camera and film LOG encodings in Nuke's list, in Nuke's order - what the
# log view offers when colour is on the built-in transforms. Not the gammas,
# not sRGB/rec709 and not the HDR display curves (HybridLogGamma, st2084):
# those are not how a camera records, and a "log view" in one is not a log view.
LOG_NAMES = ["Cineon", "Panalog", "REDLog", "ViperLog", "AlexaV3LogC",
             "PLogLin", "SLog", "SLog1", "SLog2", "SLog3", "CLog", "Log3G10",
             "Log3G12", "Protune", "Blackmagic Film Generation 5", "ARRILogC4"]

DEFAULT_DISPLAY = "sRGB"
DEFAULT_INPUT = "linear"


def names():
    return [name for name, _fn in TRANSFORMS]


def has(name):
    return name in _BY_NAME


def match_name(value, names):
    """`value` matched against `names` ignoring case, or None.

    Nuke's LUT names and ours are the same vocabulary - sRGB, rec709, Cineon,
    AlexaV3LogC - because both come from the same set of transforms, so taking
    a colorspace off a Read or off Project Settings is a spelling check rather
    than a translation. It returns OUR spelling, so what lands on the knob is
    a name the rest of the player can look up.

    Anything not in the list is dropped rather than passed through: a knob
    reading "ACEScct" on a player that has no such transform would be a worse
    answer than the default it replaced.
    """
    if not isinstance(value, str):
        return None
    value = value.strip()
    for name in names:
        if name.lower() == value.lower():
            return name
    canonical = ALIASES.get(value.lower())
    if canonical:
        for name in names:
            if name == canonical:
                return name
    return None


def encode(name, values):
    """scene-linear -> encoded (for display)."""
    fn = _BY_NAME.get(name, (None, None))[0]
    return values if fn is None else fn(values)


def decode(name, values):
    """encoded -> scene-linear (input transform)."""
    fn = _BY_NAME.get(name, (None, None))[1]
    return values if fn is None else fn(values)


# ---------------------------------------------------------------------------
# Tables over the bits of a half float - what actually gets used at runtime
# ---------------------------------------------------------------------------
def display_lut(display=DEFAULT_DISPLAY, input_space=DEFAULT_INPUT,
                gain=1.0, gamma=1.0, black=0.0):
    """65536 -> uint8. The whole file -> monitor path in one table.

    Everything is baked in: the input transform, the black and white point
    (as `black` and `gain` = 1 / (white - black), a Grade's
    (in - black) / (white - black)), the display and the gamma from CC. At
    runtime it is then a single lookup.
    """
    v = decode(input_space, _HALF_SAFE.copy()).astype(np.float32)
    v = np.clip((v - float(black)) * float(gain), 0.0, None)
    v = np.asarray(encode(display, v), dtype=np.float32)
    if abs(float(gamma) - 1.0) > 1e-6:
        v = np.power(np.clip(v, 0.0, None), 1.0 / max(float(gamma), 1e-3))
    return (np.clip(v, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def linear_table(input_space):
    """65536 -> float16: raw value from the file -> scene-linear.

    None when the input is already linear. Same purpose as ocio.linear_table().
    """
    if input_space == DEFAULT_INPUT or not has(input_space):
        return None
    v = np.asarray(decode(input_space, _HALF_SAFE.copy()), dtype=np.float32)
    v = np.nan_to_num(v, nan=0.0, posinf=65504.0, neginf=-65504.0)
    return np.clip(v, -65504.0, 65504.0).astype(np.float16)
