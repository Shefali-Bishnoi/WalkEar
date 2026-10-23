"""
Gait parameter definitions.
 * estimate_* : from the earable IMU pipeline
 * ground_truth_steps : from force plates / treadmill (+ mocap for vertical displacement)
Both sides use IDENTICAL definitions so errors are comparable:
  step_time  = HS_i -> next HS (opposite foot)
  stride     = HS_i -> next HS (same foot)
  stance     = HS_i -> TO of same foot
  swing      = stride - stance
  cadence    = 60 / step_time   (steps/min)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from . import config as C
from .preprocessing import highpass, lowpass

_trapz = getattr(np, "trapezoid", None) or np.trapz


# =========================================================== ground truth
def detect_stances(t, f, thr=C.FORCE_THRESHOLD_N, min_dur=0.25, max_dur=1.6):
    on = f > thr
    d = np.diff(on.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1
    if on[0]:
        ends = ends[1:]
    starts = starts[:len(ends)]
    out = []
    for s, e in zip(starts, ends):
        dur = t[e] - t[s]
        if min_dur <= dur <= max_dur:
            out.append((s, e))
    return out


def stance_scalars(tt, ff):
    """WA force, PO force (N), loading rate (N/s, 20-80 % of WA peak slope), impulse (N s)."""
    n = len(ff)
    half = n // 2
    wa, po = ff[:half].max(), ff[half:].max()
    ipk = int(np.argmax(ff[:half]))
    rise = ff[:ipk + 1]
    i20 = int(np.argmax(rise >= 0.2 * wa))
    i80 = int(np.argmax(rise >= 0.8 * wa))
    lr = (ff[i80] - ff[i20]) / (tt[i80] - tt[i20]) if i80 > i20 else np.nan
    return wa, po, lr, float(_trapz(ff, tt))


def ground_truth_steps(trial, n_curve=C.CURVE_POINTS_GT):
    """One row per ground-truth stance (both feet)."""
    if not trial.has_force:
        return pd.DataFrame(), np.empty((0, n_curve))
    t = trial.force_t
    mocap_z = trial.mocap_z_mm
    if mocap_z is not None and len(mocap_z) > 20:
        fm = 1.0 / np.median(np.diff(trial.mocap_t))
        mocap_z = lowpass(mocap_z, C.MOCAP_LOWPASS_HZ, fm)      # standard mocap smoothing
    rows, curves = [], []
    stances = {0: detect_stances(t, trial.force_left), 1: detect_stances(t, trial.force_right)}
    forces = {0: trial.force_left, 1: trial.force_right}
    hs_all = sorted([(t[s], side, s, e) for side, lst in stances.items() for s, e in lst])
    hs_times = np.array([h[0] for h in hs_all])
    sides = np.array([h[1] for h in hs_all])
    for k, (ths, side, s, e) in enumerate(hs_all):
        row = dict(side=side, hs_t=ths, to_t=t[e])
        # next opposite HS (step) and next same-foot HS (stride)
        nxt = [j for j in range(k + 1, min(k + 4, len(hs_all)))]
        opp = next((j for j in nxt if sides[j] != side and hs_times[j] - ths < 1.2), None)
        same = next((j for j in nxt if sides[j] == side and hs_times[j] - ths < 2.2), None)
        # a stride is only valid if the opposite step in between was observed
        row["step_time"] = hs_times[opp] - ths if opp is not None else np.nan
        row["stride"] = hs_times[same] - ths if (same is not None and opp is not None and opp < same) else np.nan
        row["stance"] = t[e] - ths
        row["swing"] = row["stride"] - row["stance"]
        row["cadence"] = 60.0 / row["step_time"] if opp is not None else np.nan
        tt, ff = t[s:e], forces[side][s:e]
        wa, po, lr, imp = stance_scalars(tt, ff)
        row.update(wa=wa, po=po, lr=lr, impulse=imp)
        # vertical displacement of the CoM during the step HS_i -> HS_(i+1)
        vd = np.nan
        if trial.mocap_t is not None and opp is not None:
            m = (trial.mocap_t >= ths) & (trial.mocap_t < hs_times[opp])
            if m.sum() > 3:
                z = mocap_z[m]
                vd = z.max() - z.min()
        row["vd_mm"] = vd
        x = (tt - tt[0]) / (tt[-1] - tt[0])
        curves.append(np.interp(np.linspace(0, 1, n_curve), x, ff) / trial.weight_n)
        rows.append(row)
    return pd.DataFrame(rows), np.array(curves).reshape(-1, n_curve)


# ============================================================ estimates
def estimate_vertical_displacement(a_vt_g, fs):
    """Low-pass -> double integration with a high-pass after every integration (paper Sec. III-C).
    Returns displacement in mm.
    * the low-pass (VD_LOWPASS_HZ) removes heel-strike shocks/noise that would otherwise leak into
      the displacement (double integration weights low frequencies by 1/f^4);
    * all filters use long padding (VD_PAD_S) so the slow transients of a 0.3 Hz high-pass do not
      corrupt the first / last steps of a recording."""
    pad = min(int(C.VD_PAD_S * fs), len(a_vt_g) - 2)
    acc = (a_vt_g - np.mean(a_vt_g)) * 9.81
    acc = lowpass(acc, C.VD_LOWPASS_HZ, fs, padlen=pad)
    acc = highpass(acc, C.VD_HIGHPASS_HZ, fs, padlen=pad)
    vel = cumulative_trapezoid(acc, dx=1 / fs, initial=0)
    vel = highpass(vel, C.VD_HIGHPASS_HZ, fs, padlen=pad)
    pos = cumulative_trapezoid(vel, dx=1 / fs, initial=0)
    pos = highpass(pos, C.VD_HIGHPASS_HZ, fs, padlen=pad)
    return pos * 1000.0


def smooth_flagged(df, cols, group=("participant", "trial", "side")):
    """Head-motion handling: flagged steps are discarded and replaced by the mean of the
    nearest unflagged previous/next step of the same foot (paper Sec. III-A.3)."""
    df = df.copy()
    for _, g in df.groupby(list(group), sort=False):
        idx = g.index.to_numpy()
        flag = g["head_motion"].to_numpy()
        for c in cols:
            v = g[c].to_numpy(float).copy()
            good = (~flag) & np.isfinite(v)
            if good.sum() < 2:
                continue
            gi = np.where(good)[0]
            for k in np.where(flag)[0]:
                prev = gi[gi < k]; nxt = gi[gi > k]
                vals = ([v[prev[-1]]] if len(prev) else []) + ([v[nxt[0]]] if len(nxt) else [])
                if vals:
                    v[k] = np.mean(vals)
            df.loc[idx, c] = v
    return df
