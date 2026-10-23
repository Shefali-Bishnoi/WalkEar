"""
Pre-processing (paper Sec. III-A). All functions are *stateless per trial*:
nothing is fitted across trials/participants, so there is no train/test leakage
and the same code runs unchanged on new raw data.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation
from . import config as C


# ---------------------------------------------------------------- filtering
def lowpass(x, fc, fs, order=4, padlen=None):
    """Zero-phase Butterworth low-pass (second-order sections: stable for very low cutoffs)."""
    sos = butter(order, fc / (fs / 2), btype="low", output="sos")
    return sosfiltfilt(sos, x, axis=0, padlen=padlen)


def highpass(x, fc, fs, order=2, padlen=None):
    """Zero-phase Butterworth high-pass. For very low cutoffs pass a large `padlen`
    (samples): the default padding is far shorter than the filter's transient."""
    sos = butter(order, fc / (fs / 2), btype="high", output="sos")
    return sosfiltfilt(sos, x, axis=0, padlen=padlen)


# --------------------------------------------------------------- resampling
def resample_uniform(t, x, fs, t_start=None, t_end=None):
    """Linear interpolation of a non-uniformly sampled signal onto a uniform grid."""
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    order = np.argsort(t, kind="stable")
    t, x = t[order], x[order]
    keep = np.r_[True, np.diff(t) > 0]
    t, x = t[keep], x[keep]
    t_start = t[0] if t_start is None else t_start
    t_end = t[-1] if t_end is None else t_end
    tu = np.arange(t_start, t_end, 1.0 / fs)
    xu = np.column_stack([np.interp(tu, t, x[:, k]) for k in range(x.shape[1])])
    return tu, xu


def resample_both_ears(trial, fs=C.FS):
    """Put left and right IMU streams on ONE common uniform time grid."""
    t0 = max(trial.t_left.min(), trial.t_right.min())
    t1 = min(trial.t_left.max(), trial.t_right.max())
    t, al = resample_uniform(trial.t_left, trial.acc_left, fs, t0, t1)
    _, gl = resample_uniform(trial.t_left, trial.gyro_left, fs, t0, t1)
    _, ar = resample_uniform(trial.t_right, trial.acc_right, fs, t0, t1)
    _, gr = resample_uniform(trial.t_right, trial.gyro_right, fs, t0, t1)
    n = min(len(t), len(al), len(ar))
    return t[:n], al[:n], gl[:n], ar[:n], gr[:n]


# ---------------------------------------------------- orientation calibration
def find_stationary_windows(acc, gyro, fs, win_s=C.STATIONARY_WIN_S):
    """Return list of (start, end) sample indices where the head is still."""
    w = int(win_s * fs)
    step = max(1, w // 2)
    mag = np.linalg.norm(acc, axis=1)
    gmag = np.linalg.norm(gyro, axis=1)
    flags = []
    for s in range(0, len(acc) - w, step):
        seg = slice(s, s + w)
        ok = (mag[seg].std() < C.STATIONARY_ACC_STD
              and abs(mag[seg].mean() - 1.0) < 0.1
              and gmag[seg].max() < C.STATIONARY_GYRO_MAX)
        flags.append((s, s + w, ok))
    wins, cur = [], None
    for s, e, ok in flags:
        if ok and cur is None:
            cur = [s, e]
        elif ok:
            cur[1] = e
        elif cur is not None:
            wins.append(tuple(cur)); cur = None
    if cur is not None:
        wins.append(tuple(cur))
    return wins


def tilt_rotation(g_vec):
    """Rotation taking the measured gravity vector (device frame) onto +VT = [0,0,1].
    The rotation angle is exactly arccos(mean(a_i)/G) of paper Eq. (1)."""
    g = np.asarray(g_vec, float)
    g = g / np.linalg.norm(g)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(g, z)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return Rotation.identity()
    angle = np.arccos(np.clip(np.dot(g, z), -1, 1))
    return Rotation.from_rotvec(axis / s * angle)


def calibrate_orientation(acc, gyro, fs):
    """Opportunistic orientation calibration. Uses every stationary window found
    (start of walking and any pauses); each calibration applies until the next one.
    Falls back to the trial-mean gravity direction if no pause exists."""
    wins = find_stationary_windows(acc, gyro, fs)
    n = len(acc)
    if not wins:
        rots, starts = [tilt_rotation(acc.mean(axis=0))], [0]
    else:
        rots = [tilt_rotation(acc[s:e].mean(axis=0)) for s, e in wins]
        starts = [0] + [e for _, e in wins[1:]]
    acc_o, gyro_o = np.empty_like(acc), np.empty_like(gyro)
    bounds = starts + [n]
    for k, r in enumerate(rots):
        sl = slice(bounds[k], bounds[k + 1])
        acc_o[sl] = r.apply(acc[sl])
        gyro_o[sl] = r.apply(gyro[sl])
    return acc_o, gyro_o, len(wins)


# ------------------------------------------------------- head-motion removal
def detect_head_motion(gyro, fs, thresh=C.HEAD_GYRO_THRESH):
    """Boolean mask of samples affected by head motion (|gyro| > 3 rad/s after
    high-pass filtering to remove yaw drift)."""
    g = highpass(gyro, C.HEAD_HP_CUTOFF, fs)
    mask = np.any(np.abs(g) > thresh, axis=1)
    k = int(C.HEAD_DILATE_S * fs)
    if k > 0 and mask.any():
        mask = np.convolve(mask.astype(float), np.ones(2 * k + 1), mode="same") > 0
    return mask
