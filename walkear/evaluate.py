"""Cross-validation driver + result tables (paper Tables II, III, V)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config as C
from .metrics import error_stats, asymmetry_table, curve_metrics
from .models import WalkEarKinetics, resample_curves
from .parameters import smooth_flagged

KIN = ["wa", "po", "lr", "impulse"]
TEMPORAL = ["cadence", "stride", "stance", "swing"]


def normalised_targets(df):
    w = df.weight_n.to_numpy()
    return {k: df[f"gt_{k}"].to_numpy() / w for k in KIN}


def train_mask(df):
    """Rows usable for TRAINING: ground truth available and not corrupted by head motion."""
    return (df.gt_matched & df.seg_valid & ~df.head_motion).to_numpy()


def cross_validate(df, XL, XS, CV100, splits, curve_points=C.CURVE_POINTS, gp_max_train=C.GP_MAX_TRAIN,
                   verbose=True):
    N = len(df)
    y = normalised_targets(df)
    feats = {"loading": XL, "stance": XS}
    pred = {k: np.full(N, np.nan) for k in KIN}
    pred_curve = np.full((N, curve_points), np.nan)
    ok_train = train_mask(df)
    for name, tr, te in splits:
        tr = tr[ok_train[tr]]
        m = WalkEarKinetics(curve_points, gp_max_train).fit(
            {k: v[tr] for k, v in feats.items()}, {k: v[tr] for k, v in y.items()}, CV100[tr])
        p = m.predict({k: v[te] for k, v in feats.items()})
        for k in KIN:
            pred[k][te] = p[k]
        pred_curve[te] = p["curve"]
        if verbose:
            print(f"  fold {name:<14} train={len(tr):5d} test={len(te):5d}", flush=True)
    out = df.copy()
    for k in KIN:                                   # personalisation: BW -> physical units
        out[f"{k}_est"] = pred[k] * out.weight_n
    return out, pred_curve


def make_tables(out, pred_curve, CV100, curve_points=C.CURVE_POINTS):
    out = out.rename(columns={"wa_est": "wa", "po_est": "po", "lr_est": "lr", "impulse_est": "impulse",
                              "vd_mm": "vd"}).copy()
    out = out.rename(columns={"gt_vd_mm": "gt_vd"})
    est_cols = TEMPORAL + ["vd"] + KIN
    out = smooth_flagged(out, est_cols)             # head-motion handling
    m = out[out.gt_matched]
    rows = []
    for c in est_cols:
        s = error_stats(m[c], m["gt_" + c]); s["parameter"] = c; rows.append(s)
    mi = m[~m.edge_step]                              # vertical displacement away from recording edges
    s = error_stats(mi["vd"], mi["gt_vd"]); s["parameter"] = "vd (excl. edge steps)"; rows.append(s)
    t2 = pd.DataFrame(rows).set_index("parameter")[["n", "MAE", "SDE", "MAPE", "ME"]]

    pairs = {c: (c, "gt_" + c) for c in ["stride", "stance", "swing"] + KIN}
    t3 = asymmetry_table(m, pairs).set_index("parameter")
    for name, members in (("Timing (stride/stance/swing)", ["stride", "stance", "swing"]),
                          ("Overall kinetic", KIN)):
        sub = t3.loc[members]
        t3.loc[name] = sub.mean(numeric_only=True)

    gt_curve = resample_curves(CV100, curve_points)
    mm = out.gt_matched.to_numpy()
    cm = curve_metrics(pred_curve[mm], gt_curve[mm])
    return t2, t3, cm, out
