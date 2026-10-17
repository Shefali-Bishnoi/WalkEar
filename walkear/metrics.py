from __future__ import annotations
import numpy as np
import pandas as pd


def error_stats(est, gt):
    est, gt = np.asarray(est, float), np.asarray(gt, float)
    ok = np.isfinite(est) & np.isfinite(gt)
    e = est[ok] - gt[ok]
    if len(e) == 0:
        return dict(n=0, MAE=np.nan, SDE=np.nan, MAPE=np.nan, ME=np.nan)
    return dict(n=int(ok.sum()), MAE=np.abs(e).mean(), SDE=e.std(), ME=e.mean(),
                MAPE=100 * np.mean(np.abs(e) / np.abs(gt[ok])))


def symmetry_index(left, right):
    """SI = (L - R) / mean(L, R) * 100  (paper Sec. II-C)."""
    ref = 0.5 * (left + right)
    return (left - right) / ref * 100.0


def curve_metrics(pred, gt):
    """Per-step NRMSE (% of the GT curve range) and Pearson correlation, averaged."""
    ok = np.all(np.isfinite(pred), axis=1) & np.all(np.isfinite(gt), axis=1)
    p, g = pred[ok], gt[ok]
    rmse = np.sqrt(((p - g) ** 2).mean(axis=1))
    nrmse = 100 * rmse / (g.max(axis=1) - g.min(axis=1))
    corr = np.array([np.corrcoef(a, b)[0, 1] for a, b in zip(p, g)])
    return dict(n=int(ok.sum()), NRMSE_pct=float(nrmse.mean()), corr=float(np.nanmean(corr)))


def asymmetry_table(df, pairs, group=("participant", "trial")):
    """pairs: {name: (est_col, gt_col)}. Per trial, SI of the left/right means (est vs GT)."""
    rows = []
    for key, g in df.groupby(list(group)):
        for name, (ec, gc) in pairs.items():
            L, R = g[g.side == 0], g[g.side == 1]
            if len(L) < 5 or len(R) < 5:
                continue
            gl, gr = L[gc].mean(), R[gc].mean()
            el, er = L[ec].mean(), R[ec].mean()
            if not np.all(np.isfinite([gl, gr, el, er])):
                continue
            rows.append(dict(parameter=name, key=key, si_gt=symmetry_index(gl, gr), si_est=symmetry_index(el, er)))
    r = pd.DataFrame(rows)
    out = []
    for name, g in r.groupby("parameter", sort=False):
        out.append(dict(parameter=name, abs_SI_GT=g.si_gt.abs().mean(), abs_SI_est=g.si_est.abs().mean(),
                        MAE=(g.si_est - g.si_gt).abs().mean(),
                        direction_agree_pct=100 * (np.sign(g.si_est) == np.sign(g.si_gt)).mean(), n_trials=len(g)))
    return pd.DataFrame(out)
