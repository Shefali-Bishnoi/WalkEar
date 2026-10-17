"""
Kinetic models (paper Sec. III-D)
  * one exponential-kernel Gaussian-process regressor per scalar (WA force, PO force, LR, impulse)
  * Histogram-based Gradient Boosting for the vGRF stance curve (20 or 100 points)
Targets are normalised by body weight (BW) so a model trained on all subjects transfers to a
new user; predictions are re-scaled with the user's weight ("personalisation").
Feature scaling is part of each sklearn Pipeline => fitted on training data only.
"""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from . import config as C

# scalar -> which feature set it uses (paper: WA/LR use HS->25 % stance, PO/Impulse the whole stance)
SCALAR_FEATURES = {"wa": "loading", "lr": "loading", "po": "stance", "impulse": "stance"}


def resample_curves(curves, n_out):
    """(n, m) -> (n, n_out) by linear interpolation along the normalised stance axis."""
    m = curves.shape[1]
    if m == n_out:
        return curves
    x_in, x_out = np.linspace(0, 1, m), np.linspace(0, 1, n_out)
    return np.array([np.interp(x_out, x_in, c) for c in curves])


def _gp(n_features):
    k = (ConstantKernel(1.0, (1e-2, 1e2))
         * Matern(length_scale=np.sqrt(n_features), length_scale_bounds=(1e-1, 1e3), nu=0.5)
         + WhiteKernel(0.1, (1e-5, 1.0)))
    return make_pipeline(StandardScaler(),
                         GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=0,
                                                  random_state=C.SEED))


def _hgb():
    return make_pipeline(StandardScaler(),
                         MultiOutputRegressor(HistGradientBoostingRegressor(
                             max_iter=150, learning_rate=0.1, random_state=C.SEED)))


class WalkEarKinetics:
    def __init__(self, curve_points=C.CURVE_POINTS, gp_max_train=C.GP_MAX_TRAIN, seed=C.SEED,
                 fit_curve=True):
        self.curve_points, self.gp_max_train, self.seed, self.fit_curve = curve_points, gp_max_train, seed, fit_curve
        self.gps, self.curve_model = {}, None

    def fit(self, feats, y_norm, curves_norm=None):
        """feats: {'loading': X, 'stance': X}; y_norm: {'wa','po','lr','impulse'} arrays in BW units."""
        rng = np.random.default_rng(self.seed)
        for name, fset in SCALAR_FEATURES.items():
            X, y = feats[fset], y_norm[name]
            ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
            idx = np.where(ok)[0]
            if len(idx) > self.gp_max_train:
                idx = rng.choice(idx, self.gp_max_train, replace=False)
            self.gps[name] = _gp(X.shape[1]).fit(X[idx], y[idx])
        if self.fit_curve and curves_norm is not None:
            X = np.hstack([feats["loading"], feats["stance"]])
            Y = resample_curves(curves_norm, self.curve_points)
            ok = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(Y), axis=1)
            self.curve_model = _hgb().fit(X[ok], Y[ok])
        return self

    def predict(self, feats):
        out = {}
        for name, fset in SCALAR_FEATURES.items():
            X = feats[fset]
            ok = np.all(np.isfinite(X), axis=1)
            p = np.full(len(X), np.nan)
            if ok.any():
                p[ok] = self.gps[name].predict(X[ok])
            out[name] = p
        if self.curve_model is not None:
            X = np.hstack([feats["loading"], feats["stance"]])
            ok = np.all(np.isfinite(X), axis=1)
            c = np.full((len(X), self.curve_points), np.nan)
            if ok.any():
                c[ok] = self.curve_model.predict(X[ok])
            out["curve"] = c
        return out
