"""Time/frequency-domain features of a resultant-acceleration segment (paper Sec. III-D)."""
from __future__ import annotations
import numpy as np
from . import config as C

BASE_NAMES = ["max", "mean", "rms", "shape_factor", "crest_factor", "clearance_factor",
              "impulse_factor", "mean_freq", "peak_amp", "peak_freq", "band_power"]


def segment_features(x, fs, nfft=256, band=C.FEATURE_BAND_HZ):
    x = np.asarray(x, float)
    if len(x) < 5 or not np.all(np.isfinite(x)):
        return np.full(len(BASE_NAMES), np.nan)
    eps = 1e-9
    ax = np.abs(x)
    rms = np.sqrt(np.mean(x ** 2)) + eps
    mabs = ax.mean() + eps
    time_f = [x.max(), x.mean(), rms, rms / mabs, ax.max() / rms,
              ax.max() / (np.mean(np.sqrt(ax)) ** 2 + eps), ax.max() / mabs]
    xd = (x - x.mean()) * np.hanning(len(x))
    spec = np.abs(np.fft.rfft(xd, nfft))
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    p = spec ** 2
    freq_f = [(f * p).sum() / (p.sum() + eps), spec.max(), f[spec.argmax()],
              p[(f >= band[0]) & (f <= band[1])].sum()]
    return np.array(time_f + freq_f)


def feature_names():
    names = {}
    for kind in ("loading", "stance"):
        names[kind] = [f"{ear}_{kind}_{n}" for ear in ("L", "R") for n in BASE_NAMES]
    return names


def step_features(ag_l, ag_r, i0, i1, fs):
    """Features for the stance segment [i0, i1) and its first-25 % loading sub-segment,
    for BOTH ears (concatenated: [left ear | right ear])."""
    if i1 - i0 < 8:
        n = 2 * len(BASE_NAMES)
        return np.full(n, np.nan), np.full(n, np.nan)
    il = i0 + max(5, int(round(C.LOADING_FRACTION * (i1 - i0))))
    load = np.r_[segment_features(ag_l[i0:il], fs), segment_features(ag_r[i0:il], fs)]
    stance = np.r_[segment_features(ag_l[i0:i1], fs), segment_features(ag_r[i0:i1], fs)]
    return load, stance
