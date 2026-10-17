"""Build / load the per-step dataset (IMU estimates + features + matched ground truth)."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from . import config as C
from .io import iter_trials
from .pipeline import run_imu_pipeline
from .parameters import ground_truth_steps
from .features import feature_names

GT_COLS = ["gt_hs_t", "gt_step_time", "gt_stride", "gt_stance", "gt_swing", "gt_cadence",
           "gt_vd_mm", "gt_wa", "gt_po", "gt_lr", "gt_impulse"]


def match_ground_truth(steps, gt, tol=C.GT_MATCH_TOL_S):
    """Match each estimated step to the nearest GT stance of the same foot (1-to-1)."""
    n = len(steps)
    out = np.full((n, len(GT_COLS)), np.nan)
    gt_idx = np.full(n, -1)
    if n == 0 or len(gt) == 0:
        return out, gt_idx
    used = set()
    for s in (0, 1):
        ii = np.where(steps["side"].to_numpy() == s)[0]
        gj = np.where(gt["side"].to_numpy() == s)[0]
        if len(gj) == 0:
            continue
        gts = gt["hs_t"].to_numpy()[gj]
        for i in ii:
            d = np.abs(gts - steps["hs_t"].iloc[i])
            k = int(np.argmin(d))
            if d[k] <= tol and gj[k] not in used:
                used.add(gj[k]); gt_idx[i] = gj[k]
    for i, j in enumerate(gt_idx):
        if j >= 0:
            r = gt.iloc[j]
            out[i] = [r.hs_t, r.step_time, r.stride, r.stance, r.swing, r.cadence,
                      r.vd_mm, r.wa, r.po, r.lr, r.impulse]
    return out, gt_idx


def build_dataset(raw_dir=C.DATA_RAW, out_dir=C.DATA_PROCESSED, verbose=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    all_steps, XL, XS, CV = [], [], [], []
    for trial in iter_trials(raw_dir):
        steps, xl, xs = run_imu_pipeline(trial)
        gt, curves = ground_truth_steps(trial)
        gtv, gt_idx = match_ground_truth(steps, gt)
        for k, c in enumerate(GT_COLS):
            steps[c] = gtv[:, k]
        cv = np.full((len(steps), C.CURVE_POINTS_GT), np.nan)
        m = gt_idx >= 0
        if m.any():
            cv[m] = curves[gt_idx[m]]
        steps["gt_matched"] = m
        steps["weight_n"] = trial.weight_n
        steps["speed_kph"] = trial.meta.get("speed_kph", np.nan)
        steps["surface"] = trial.meta.get("surface", "unknown")
        all_steps.append(steps); XL.append(xl); XS.append(xs); CV.append(cv)
        if verbose:
            print(f"{trial.participant}/{trial.name:<16} IMU steps={len(steps):4d} "
                  f"seg_valid={int(steps.seg_valid.sum()):4d} GT-matched={int(m.sum()):4d} "
                  f"head-motion={int(steps.head_motion.sum()):3d}")
    df = pd.concat(all_steps, ignore_index=True)
    df.to_csv(out_dir / "steps.csv", index=False)
    np.save(out_dir / "X_loading.npy", np.vstack(XL))
    np.save(out_dir / "X_stance.npy", np.vstack(XS))
    np.save(out_dir / "Y_curve100.npy", np.vstack(CV))
    (out_dir / "feature_names.json").write_text(json.dumps(feature_names(), indent=1))
    return df


def load_dataset(out_dir=C.DATA_PROCESSED):
    df = pd.read_csv(out_dir / "steps.csv", dtype={"participant": str})
    return (df, np.load(out_dir / "X_loading.npy"), np.load(out_dir / "X_stance.npy"),
            np.load(out_dir / "Y_curve100.npy"))
