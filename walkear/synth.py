"""
Synthetic WalkEar-like dataset generator.

Physics-inspired (NOT a copy of real data):
  * a step timeline (HS/TO per foot) with per-participant asymmetry and speed-dependent cadence
  * a double-hump vGRF per stance whose amplitude/width vary step-to-step
  * head acceleration  a_vt = 1 G + kappa * (F_total/BW - mean)  + heel-strike transients
    (amplitude tied to the loading rate) -> kinetic parameters are *learnable* from the IMU
  * lateral sway so that (a_ml_left - a_ml_right) leans left at left HS
  * random earbud tilt (device frame != anatomical frame), gyro bias drift, head turns
  * non-uniform IMU timestamps, separate clocks per ear, +-2 g / +-250 deg/s clipping
It writes exactly the on-disk format described in README.md.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation

FS_SIM = 1000.0
PRELUDE_S = 4.0          # standing still at the start (used for orientation calibration)
KAPPA = 0.10             # head-to-CoM coupling of vertical acceleration
SENSOR_DLPF_HZ = 44.0    # MPU6050 on-chip digital low-pass (anti-aliasing) before ~100 Hz sampling
ACC_NOISE_G = 0.006      # white accelerometer noise (G); MPU6050 datasheet ~0.004 G over 75 Hz
GT_F_LO_HZ = 0.3         # lower band edge of the ground-truth CoM displacement
MOCAP_NOISE_MM = 0.10    # marker noise on the ground-truth CoM height (mm)
IMPACT_G = 0.22          # heel-strike transient amplitude at the ear (G)
HEAD_LEAD_S = 0.07       # phase of the head's vertical acceleration relative to the force sum
                         # (tuned so the paper's 2 Hz jerk-trough TO heuristic is unbiased)


def _lp(x, fc, fs=FS_SIM, order=4):
    b, a = butter(order, fc / (fs / 2)); return filtfilt(b, a, x)


def _hp(x, fc, fs=FS_SIM, order=2):
    b, a = butter(order, fc / (fs / 2), btype="high"); return filtfilt(b, a, x)


def _double_integrate_spectral(acc, fs, f_lo=0.3):
    """Exact double integration in the frequency domain for components above f_lo.
    Avoids the quadratic drift that time-domain integration of a nearly-zero-mean signal creates."""
    n = len(acc)
    A = np.fft.rfft(acc)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    Z = np.zeros_like(A)
    m = f >= f_lo
    Z[m] = -A[m] / (2 * np.pi * f[m]) ** 2
    return np.fft.irfft(Z, n)


def _impact_template(fs=FS_SIM, dur=0.3, f=12.0, tau=0.04, rise=0.004):
    """Heel-strike shock at the ear: damped ringing, normalised to max |a| = 1.
    Built as the 2nd derivative of a smooth bump x(k) that starts and ends at rest, so BOTH its
    net velocity and net displacement are ~0 (a real shock does not permanently move the head,
    and a plain damped sine would inject spurious displacement into the vertical-displacement
    estimate that the ground truth does not contain)."""
    k = np.arange(0, 1.0, 1.0 / fs)                      # compute on 1 s, keep the first `dur`
    x = np.exp(-k / tau) * np.sin(2 * np.pi * f * k) * (1 - np.exp(-k / rise)) ** 3
    # exact DISCRETE second difference: two cumulative sums give x back, so the template's net
    # velocity and net displacement are zero by construction (up to the 0.3 s truncation)
    a = np.diff(x, n=2, prepend=[0.0, 0.0]) * fs ** 2
    a = a[:int(dur * fs)]
    return a / np.abs(a).max()


IMPACT_TEMPLATE = _impact_template()


def _edge(x, w_in=0.04, w_out=0.10):
    """Smooth on/off ramps of a stance (real forces fade over ~50 ms, they do not step)."""
    r_in = np.clip(x / w_in, 0, 1); r_out = np.clip((1 - x) / w_out, 0, 1)
    return (3 * r_in ** 2 - 2 * r_in ** 3) * (3 * r_out ** 2 - 2 * r_out ** 3)


def make_participants(n, rng):
    rows = []
    for k in range(n):
        female = rng.random() < 0.31
        rows.append(dict(
            participant_id=f"P{k + 1:02d}", sex="F" if female else "M",
            age=int(rng.integers(21, 40)),
            weight_kg=round(float(np.clip(rng.normal(62 if female else 77, 8), 50, 100)), 1),
            height_m=round(float(np.clip(rng.normal(1.68 if female else 1.82, 0.06), 1.5, 2.0)), 2),
            foot_length_cm=round(float(rng.normal(25 if female else 28, 1.0)), 1),
            # hidden generator parameters (not written to participants.csv)
            _cad_off=rng.normal(0, 6), _asym_step=rng.normal(0, 0.012),
            _asym_stance=rng.normal(0, 0.015), _asym_force=rng.normal(0, 0.025),
            _force_scale=rng.normal(1.0, 0.03), _sway=rng.uniform(0.05, 0.10)))
    return rows


def _step_timeline(rng, subj, speed_fn, t_end):
    hs, step_t, speed = [PRELUDE_S + 0.3], [], []
    j = 0
    while hs[-1] < t_end + 3.0:
        v = speed_fn(j, hs[-1])
        ramp = min(1.0, 0.55 + 0.45 * j / 5.0)            # gentle start-up
        cad = (94 + 4.3 * v + subj["_cad_off"]) * (0.85 + 0.15 * ramp)
        base = 60.0 / cad
        sgn = 1 if j % 2 == 0 else -1
        st = base * (1 + subj["_asym_step"] * sgn) * (1 + rng.normal(0, 0.02))
        step_t.append(st); speed.append(v)
        hs.append(hs[-1] + st); j += 1
    return np.array(hs[:-1]), np.array(step_t), np.array(speed)


def simulate_trial(rng, subj, speed_kph, duration_s, surface):
    t_end = PRELUDE_S + duration_s
    if surface == "treadmill":
        speed_fn = lambda j, t: speed_kph
    else:  # free walking: slowly varying speed (std ~0.8 kph)
        state = {"v": speed_kph}
        def speed_fn(j, t):
            state["v"] = 0.9 * state["v"] + 0.1 * speed_kph + rng.normal(0, 0.45)
            return float(np.clip(state["v"], 2.5, 6.0))
    hs, step_t, speed = _step_timeline(rng, subj, speed_fn, t_end)
    n_steps = len(hs)
    foot = np.arange(n_steps) % 2                                   # 0 = left, 1 = right
    stride = np.r_[step_t[:-1] + step_t[1:], step_t[-1] * 2]
    sgn = np.where(foot == 0, 1, -1)
    frac = 0.63 - 0.01 * (speed - 4)
    stance = stride * frac * (1 + subj["_asym_stance"] * sgn) + rng.normal(0, 0.012, n_steps)
    stance = np.maximum(stance, np.r_[step_t[:-1], step_t[-1]] + 0.06)
    to = hs + stance

    # per-step intensity (AR(1)) -> correlated force & impact amplitude
    e = np.zeros(n_steps)
    for k in range(1, n_steps):
        e[k] = 0.7 * e[k - 1] + rng.normal(0, 0.045)
    s = 1 + e
    A1 = (1.08 + 0.05 * (speed - 4)) * s * (1 + subj["_asym_force"] * sgn) * subj["_force_scale"] + rng.normal(0, 0.01, n_steps)
    A2 = (1.05 + 0.04 * (speed - 4)) * s * (1 - subj["_asym_force"] * sgn) * subj["_force_scale"] + rng.normal(0, 0.01, n_steps)
    B = 0.80 * np.sqrt(s)
    w1 = 0.11 * (1 + rng.normal(0, 0.06, n_steps))

    n = int((t_end + 2) * FS_SIM)
    t = np.arange(n) / FS_SIM
    F = {0: np.zeros(n), 1: np.zeros(n)}
    for j in range(n_steps):
        i0, i1 = int(hs[j] * FS_SIM), int(to[j] * FS_SIM)
        if i1 >= n:
            break
        x = (t[i0:i1] - hs[j]) / stance[j]
        g = (A1[j] * np.exp(-((x - 0.20) / w1[j]) ** 2) + A2[j] * np.exp(-((x - 0.78) / 0.12) ** 2)
             + B[j] * np.exp(-((x - 0.5) / 0.35) ** 2) * 0.9)
        g *= _edge(x)
        F[foot[j]][i0:i1] += g
    Ftot = F[0] + F[1]                                             # in BW

    # -------- head acceleration in the anatomical frame [AP, ML(outward), VT] (G)
    walk = (t > hs[0] - 0.2) & (t < t_end)
    Fc = np.zeros(n)
    Ftot_lead = np.interp(t + HEAD_LEAD_S, t, Ftot)
    # Newton: over a stride the mean vertical force equals body weight, so the CoM has no net
    # vertical acceleration below the stride frequency -> remove content < 0.6 Hz.
    Fc[walk] = _hp(_lp(Ftot_lead - Ftot_lead[walk].mean(), 15.0), 0.6)[walk]
    gate = np.clip((t - (hs[0] - 0.3)) / 0.3, 0, 1)
    a_vt_base = 1.0 + KAPPA * Fc * gate

    imp_v = np.zeros(n); imp_ap = np.zeros(n)
    for j in range(n_steps):
        i0 = int(hs[j] * FS_SIM)
        if i0 >= n - 200:
            break
        tr = IMPACT_TEMPLATE[:n - i0]
        amp = IMPACT_G * (A1[j] / 1.1) * (0.11 / w1[j])
        imp_v[i0:i0 + len(tr)] += amp * tr
        imp_ap[i0:i0 + len(tr)] += -0.5 * amp * tr

    ph_t = np.r_[hs, hs[-1] + 1.0]
    ph_v = np.r_[np.pi * np.arange(n_steps), np.pi * n_steps]
    phi = np.interp(t, ph_t, ph_v)
    a_lat = subj["_sway"] * np.cos(phi) * gate
    a_ap_base = 0.05 * np.cos(2 * phi + 1.0) * gate

    # -------- head motion events (turns/nods)
    gyro_anat = np.zeros((n, 3)); acc_dist = np.zeros((n, 3))
    gyro_anat[:, 0] = 0.35 * np.sin(2 * phi) * gate
    gyro_anat[:, 1] = 0.30 * np.sin(phi) * gate
    gyro_anat[:, 2] = 0.25 * np.cos(phi) * gate
    n_head = rng.poisson(duration_s / 18.0)
    for _ in range(n_head):
        t0 = rng.uniform(hs[0] + 2, t_end - 2)
        dur = rng.uniform(0.5, 0.9)
        m = (t >= t0) & (t < t0 + dur)
        u = (t[m] - t0) / dur
        ax = rng.integers(0, 3)
        gyro_anat[m, ax] += rng.choice([-1, 1]) * rng.uniform(3.6, 4.6) * np.sin(np.pi * u)
        acc_dist[m, 1] += rng.normal(0.25, 0.05) * np.sin(np.pi * u)

    bias = rng.normal(0, 0.03, 3)
    drift = rng.normal(0, 0.04, 3) / t_end
    gyro_anat += bias + np.outer(t, drift)

    # -------- per ear: anatomical -> device frame, noise, clipping, jittered clock
    streams = {}
    for ear, sign in (("left", 1.0), ("right", -1.0)):
        gain = 1 + rng.normal(0, 0.08)
        a = np.column_stack([a_ap_base + gain * imp_ap,
                             sign * a_lat + 0.25 * gain * imp_v * sign * 0.3 + acc_dist[:, 1],
                             a_vt_base + gain * imp_v])
        a += rng.normal(0, ACC_NOISE_G, a.shape)
        g = gyro_anat + rng.normal(0, 0.03, gyro_anat.shape)
        R = Rotation.from_euler("zyx", np.deg2rad([rng.normal(0, 6), rng.normal(0, 18), rng.normal(0, 18)]))
        a_dev, g_dev = R.inv().apply(a), R.inv().apply(g)
        a_dev = np.column_stack([_lp(a_dev[:, k], SENSOR_DLPF_HZ) for k in range(3)])
        g_dev = np.column_stack([_lp(g_dev[:, k], SENSOR_DLPF_HZ) for k in range(3)])
        a_dev = np.clip(a_dev, -2, 2); g_dev = np.clip(g_dev, -np.deg2rad(250), np.deg2rad(250))
        dt = 0.01 * (1 + rng.normal(0, 0.08, int(t_end * 105)))
        ts = np.cumsum(np.clip(dt, 0.004, 0.03)) + rng.uniform(0, 0.02)
        ts = ts[ts < t_end]
        A = np.column_stack([np.interp(ts, t, a_dev[:, k]) for k in range(3)])
        G = np.column_stack([np.interp(ts, t, g_dev[:, k]) for k in range(3)])
        streams[ear] = (ts, A, G)

    # -------- ground truth: force plates (free walk -> only 4 steps per ~14-step lap)
    BW_N = subj["weight_kg"] * 9.81
    visible = np.ones(n_steps, bool)
    if surface != "treadmill":
        visible = np.isin(np.arange(n_steps) % 14, [5, 6, 7, 8])
    Fvis = {0: np.zeros(n), 1: np.zeros(n)}
    for j in range(n_steps):
        if not visible[j]:
            continue
        i0, i1 = int(hs[j] * FS_SIM), int(to[j] * FS_SIM)
        if i1 >= n:
            break
        x = (t[i0:i1] - hs[j]) / stance[j]
        g = (A1[j] * np.exp(-((x - 0.20) / w1[j]) ** 2) + A2[j] * np.exp(-((x - 0.78) / 0.12) ** 2)
             + B[j] * np.exp(-((x - 0.5) / 0.35) ** 2) * 0.9)
        g *= _edge(x)
        Fvis[foot[j]][i0:i1] += g
    force = pd.DataFrame({"time_s": t, "fz_left_N": Fvis[0] * BW_N + rng.normal(0, 1.5, n) * (Fvis[0] > 0),
                          "fz_right_N": Fvis[1] * BW_N + rng.normal(0, 1.5, n) * (Fvis[1] > 0)})
    mocap = None
    if surface == "treadmill":
        acc = KAPPA * Fc * gate * 9.81
        z = _double_integrate_spectral(acc, FS_SIM, f_lo=GT_F_LO_HZ) * 1000.0     # mm, no integration drift
        tm = np.arange(0, t_end, 0.01)
        zm = np.interp(tm, t, z) + rng.normal(0, MOCAP_NOISE_MM, len(tm))      # marker noise only
        mocap = pd.DataFrame({"time_s": tm, "com_z_mm": zm})
    return streams, force, mocap


def generate_dataset(out_dir, n_participants=10, seed=0, treadmill_s=60, free_s=90, verbose=True):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    subjects = make_participants(n_participants, rng)
    pd.DataFrame([{k: v for k, v in s.items() if not k.startswith("_")} for s in subjects]).to_csv(
        out_dir / "participants.csv", index=False)
    plan = [("treadmill_3kph", 3.0, treadmill_s, "treadmill"),
            ("treadmill_5kph", 5.0, treadmill_s, "treadmill"),
            ("freewalk", 4.3, free_s, "forceplate")]
    for subj in subjects:
        for name, v, dur, surf in plan:
            streams, force, mocap = simulate_trial(rng, subj, v, dur, surf)
            d = out_dir / subj["participant_id"] / name
            d.mkdir(parents=True, exist_ok=True)
            for ear in ("left", "right"):
                ts, A, G = streams[ear]
                pd.DataFrame(np.column_stack([ts, A, G]),
                             columns=["time_s", "ax", "ay", "az", "gx", "gy", "gz"]).to_csv(
                    d / f"imu_{ear}.csv", index=False, float_format="%.5f")
            # keep force files small: only write the time span where anything happens
            force.to_csv(d / "force.csv", index=False, float_format="%.3f")
            if mocap is not None:
                mocap.to_csv(d / "mocap.csv", index=False, float_format="%.4f")
            (d / "meta.json").write_text(json.dumps(
                {"speed_kph": v, "surface": surf, "duration_s": dur}))
            if verbose:
                print("wrote", d)
