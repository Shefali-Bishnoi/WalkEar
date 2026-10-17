"""Heel-strike / toe-off detection and left-right labelling (paper Sec. III-A.4, III-B)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.signal import find_peaks
from . import config as C
from .preprocessing import lowpass

LEFT, RIGHT = 0, 1


@dataclass
class GaitEvents:
    hs_t: np.ndarray        # heel-strike times (s)
    hs_side: np.ndarray     # 0 = left, 1 = right
    to_after: np.ndarray    # for each HS i: TO time of the *opposite* foot that follows it (nan if none)


def resultant(acc):
    return np.linalg.norm(acc, axis=1)


def detect_heel_strikes(a_g, fs):
    """Peaks of the jerk of the resultant acceleration."""
    a_f = lowpass(a_g, C.HS_LOWPASS_HZ, fs)
    jerk = np.gradient(a_f) * fs
    peaks, _ = find_peaks(jerk, distance=int(C.HS_MIN_DISTANCE_S * fs))
    if len(peaks) == 0:
        return np.array([], int), jerk
    h = jerk[peaks]
    keep = h >= C.HS_REL_THRESH * np.percentile(h, 90)
    peaks = peaks[keep]
    # Impacts ring (damped oscillation), so the tallest jerk peak is not always the first lobe.
    # Take the earliest local maximum within 0.1 s before it that is itself a "strong" impact
    # peak (>= HS_FIRST_LOBE_REL x the 90th percentile of detected peaks).
    strong = C.HS_FIRST_LOBE_REL * np.percentile(h, 90)
    all_peaks = find_peaks(jerk)[0]
    refined = []
    for p in peaks:
        near = all_peaks[(all_peaks >= p - int(0.1 * fs)) & (all_peaks <= p)]
        near = near[jerk[near] >= strong]
        refined.append(near[0] if len(near) else p)
    return np.array(refined, int), jerk


def classify_sides(hs_idx, acc_l_ml, acc_r_ml, fs):
    """Left/right label from sign of the lowpassed (a_ml_left - a_ml_right):
    positive near an HS => body leans left => LEFT step (paper Fig. 7)."""
    d = lowpass(acc_l_ml - acc_r_ml, C.SIDE_LOWPASS_HZ, fs)
    w = int(0.05 * fs)
    sides = np.array([LEFT if d[max(0, i - w):i + w + 1].mean() > 0 else RIGHT for i in hs_idx], int)
    return sides


def enforce_alternation(hs_idx, sides, strength):
    """Left and right HS must alternate; if two consecutive events carry the same
    label keep the stronger one."""
    keep = np.ones(len(hs_idx), bool)
    last = None
    for i in range(len(hs_idx)):
        if last is not None and sides[i] == sides[last]:
            if strength[i] > strength[last]:
                keep[last] = False; last = i
            else:
                keep[i] = False
        else:
            last = i
    return keep


def detect_toe_offs(a_vt, hs_t, fs, t0):
    """Paper: jerk of a_vt -> 2 Hz Butterworth -> trough that is followed by a zero crossing,
    searched between consecutive HS. Returns, per HS i, the TO time of the OPPOSITE foot."""
    jerk = np.gradient(a_vt) * fs
    sm = lowpass(jerk, C.TO_LOWPASS_HZ, fs, order=2)
    troughs, _ = find_peaks(-sm)
    tt = t0 + np.arange(len(sm)) / fs
    out = np.full(len(hs_t), np.nan)
    for i in range(len(hs_t) - 1):
        lo, hi = hs_t[i] + 0.04, hs_t[i + 1]
        cand = troughs[(tt[troughs] >= lo) & (tt[troughs] < hi)]
        if len(cand) == 0:
            continue
        k = cand[np.argmin(sm[cand])]
        # require a zero crossing (negative -> positive) shortly after the trough
        seg = sm[k:k + int(0.4 * fs)]
        if np.any(seg > 0):
            out[i] = tt[k]
    return out


def detect_gait_events(acc_l, acc_r, fs, t0):
    """acc_l, acc_r: calibrated accelerations (G) in Coord_g = [AP, ML, VT]."""
    a_g = 0.5 * (resultant(acc_l) + resultant(acc_r))      # fuse both ears
    hs_idx, jerk = detect_heel_strikes(a_g, fs)
    if len(hs_idx) < 4:
        return GaitEvents(np.array([]), np.array([], int), np.array([]))
    sides = classify_sides(hs_idx, acc_l[:, 1], acc_r[:, 1], fs)
    keep = enforce_alternation(hs_idx, sides, jerk[hs_idx])
    hs_idx, sides = hs_idx[keep], sides[keep]
    hs_t = t0 + hs_idx / fs
    a_vt = 0.5 * (acc_l[:, 2] + acc_r[:, 2])
    to_after = detect_toe_offs(a_vt, hs_t, fs, t0)
    return GaitEvents(hs_t, sides, to_after)
