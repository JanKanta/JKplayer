"""
Built-in colour transforms - the "Nuke" mode, i.e. without OCIO.

WHY ALONGSIDE OCIO: it is substantially faster. OCIO goes through a 3D cube
(28 ms on 1080p), this is a single table over the bits of a half float (9 ms).
On 6K playback that is the difference between comfort and a crawl.

Every transform is implemented from the published formulas and the tests
compare them against the nuke-default OCIO config - see tests/test_nukelut.py.
Anything that does not match has no business being here.

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
# The list of transforms. Ordered as in Nuke.
# ---------------------------------------------------------------------------
# name -> (forward linear->encoded, backward encoded->linear)
TRANSFORMS = [
    ("linear", (None, None)),                 # None = identity
    ("sRGB", (_srgb_fwd, _srgb_inv)),
    ("rec709", (_rec709_fwd, _rec709_inv)),
    ("rec1886", (_gamma_fwd(2.4), _gamma_inv(2.4))),
    ("Gamma2.2", (_gamma_fwd(2.2), _gamma_inv(2.2))),
    ("Cineon", (_cineon_fwd, _cineon_inv)),
    ("AlexaV3LogC", (_logc_fwd, _logc_inv)),
    ("SLog3", (_slog3_fwd, _slog3_inv)),
    ("Log3G10", (_log3g10_fwd, _log3g10_inv)),
]
_BY_NAME = dict(TRANSFORMS)

# What to offer as a display (viewer process) - log spaces make no sense
# there, they are not meant to be sent to a monitor.
DISPLAY_NAMES = ["sRGB", "rec709", "rec1886", "Gamma2.2", "linear"]
INPUT_NAMES = [name for name, _fn in TRANSFORMS]

DEFAULT_DISPLAY = "sRGB"
DEFAULT_INPUT = "linear"


def names():
    return [name for name, _fn in TRANSFORMS]


def has(name):
    return name in _BY_NAME


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
                gain=1.0, gamma=1.0):
    """65536 -> uint8. The whole file -> monitor path in one table.

    Everything is baked in: the input transform, exposure, the display and the
    gamma from CC. At runtime it is then a single lookup.
    """
    v = decode(input_space, _HALF_SAFE.copy()).astype(np.float32)
    v = np.clip(v * float(gain), 0.0, None)
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
