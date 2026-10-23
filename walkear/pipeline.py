"""
End-to-end IMU pipeline for ONE trial:  raw Trial -> per-step table + kinetic features.
Identical code path is used for training data, test data and brand-new recordings.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config as C
from .preprocessing import (resample_both_ears, calibrate_orientation, detect_head_motion, lowpass)
from .gait_events import detect_gait_events, resultant
from .parameters import estimate_vertical_displacement
from .features import step_features, feature_names


def run_imu_pipeline(trial, fs=C.FS):
    t, al, gl, ar, gr = resample_both_ears(trial, fs)
    al_c, gl_c, ncal_l = calibrate_orientation(al, gl, fs)
    ar_c, gr_c, ncal_r = calibrate_orientation(ar, gr, fs)
    head = detect_head_motion(gl_c, fs) | detect_head_motion(gr_c, fs)
    ev = detect_gait_events(al_c, ar_c, fs, t[0])

    n_feat = 2 * 11
    cols = ["participant", "trial", "step_i", "side", "hs_t", "to_t", "step_time", "stride",
            "stance", "swing", "cadence", "vd_mm", "head_motion", "edge_step", "seg_valid"]
    if len(ev.hs_t) < 4:
        return pd.DataFrame(columns=cols), np.empty((0, n_feat)), np.empty((0, n_feat))

    # spatial signal (fused ears) and kinetic input (resultant acceleration, light lowpass)
    vd_sig = estimate_vertical_displacement(0.5 * (al_c[:, 2] + ar_c[:, 2]), fs)
    ag_l = lowpass(resultant(al_c), 30.0, fs)
    ag_r = lowpass(resultant(ar_c), 30.0, fs)

    hs, side, to_after = ev.hs_t, ev.hs_side, ev.to_after
    t_first, t_last = hs[0], hs[-1]
    lo, hi = C.STEP_TIME_RANGE_S
    rows, XL, XS = [], [], []
    for i in range(len(hs) - 2):
        ok = (side[i + 1] != side[i]) and (side[i + 2] == side[i])
        st = hs[i + 1] - hs[i]
        ok &= lo <= st <= hi and lo <= hs[i + 2] - hs[i + 1] <= hi
        to_same = to_after[i + 1]                       # TO of foot i occurs after HS_(i+1)
        ok &= np.isfinite(to_same) and (hs[i] < to_same < hs[i + 2])
        row = dict(participant=trial.participant, trial=trial.name, step_i=i, side=int(side[i]),
                   hs_t=hs[i], to_t=to_same if ok else np.nan, step_time=np.nan, stride=np.nan,
                   stance=np.nan, swing=np.nan, cadence=np.nan, vd_mm=np.nan,
                   head_motion=False, seg_valid=bool(ok),
                   # start-up / stop transients: the 0.3 Hz high-pass chain has not settled
                   edge_step=bool(hs[i] - t_first < C.VD_EDGE_S or t_last - hs[i + 2] < C.VD_EDGE_S))
        if ok:
            stride = hs[i + 2] - hs[i]
            stance = to_same - hs[i]
            row.update(step_time=st, stride=stride, stance=stance, swing=stride - stance,
                       cadence=60.0 / st)
            a, b = int(round((hs[i] - t[0]) * fs)), int(round((hs[i + 1] - t[0]) * fs))
            row["vd_mm"] = vd_sig[a:b].max() - vd_sig[a:b].min()
            a2 = int(round((hs[i + 2] - t[0]) * fs))
            row["head_motion"] = bool(head[a:a2].any())
            i0, i1 = a, int(round((to_same - t[0]) * fs))
            fl, fs_ = step_features(ag_l, ag_r, i0, i1, fs)
        else:
            fl = fs_ = np.full(n_feat, np.nan)
        rows.append(row); XL.append(fl); XS.append(fs_)
    return pd.DataFrame(rows), np.array(XL), np.array(XS)
