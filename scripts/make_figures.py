"""Generate the explanatory figures used by ProjectExplanation.md (from the shipped data)."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
from walkear import config as C
from walkear.io import load_trial
from walkear.preprocessing import *
from walkear.gait_events import *
from walkear.parameters import ground_truth_steps, estimate_vertical_displacement, stance_scalars
from walkear.features import segment_features

import textwrap
_orig_set_title = plt.Axes.set_title
def _wrapped_set_title(self, label, *a, **k):          # wrap long titles so they never get cut off
    width = int(self.figure.get_figwidth() * 8.5)
    return _orig_set_title(self, textwrap.fill(label, width), *a, **k)
plt.Axes.set_title = _wrapped_set_title
OUT = C.ROOT / "docs" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3, "font.size": 9})
fs = C.FS
tr = load_trial(C.DATA_RAW / "P01" / "treadmill_3kph", "P01", 82.1)
t, al, gl, ar, gr = resample_both_ears(tr, fs)
al_c, gl_c, _ = calibrate_orientation(al, gl, fs); ar_c, gr_c, _ = calibrate_orientation(ar, gr, fs)
gt, curves = ground_truth_steps(tr)
a_g = 0.5 * (resultant(al_c) + resultant(ar_c))
hs_idx, jerk = detect_heel_strikes(a_g, fs)
hs_est = t[0] + hs_idx / fs
def save(name): plt.tight_layout(); plt.savefig(OUT / name); plt.close()

# ---- fig01 gait cycle from force plates
w0, w1 = 14.9, 18.1
m = (tr.force_t >= w0) & (tr.force_t <= w1)
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.plot(tr.force_t[m], tr.force_left[m], "C0", label="left foot force (N)")
ax.plot(tr.force_t[m], tr.force_right[m], "C1", label="right foot force (N)")
g = gt[(gt.hs_t > w0) & (gt.hs_t < w1)]
for _, r in g.iterrows():
    ax.axvline(r.hs_t, color="g", ls="--", lw=.8); ax.axvline(r.to_t, color="r", ls=":", lw=.8)
ax.plot([], [], "g--", label="heel strike (HS)"); ax.plot([], [], "r:", label="toe off (TO)")
L = gt[(gt.side == 0) & (gt.hs_t > 15.4) & (gt.hs_t < 15.5)].iloc[0]
ax.annotate("", xy=(L.hs_t, 1350), xytext=(L.to_t, 1350), arrowprops=dict(arrowstyle="<->", color="C0"))
ax.text((L.hs_t + L.to_t) / 2, 1380, "STANCE of left foot\n(foot on ground)", ha="center", color="C0")
ax.annotate("", xy=(L.to_t, 1350), xytext=(L.hs_t + L.stride, 1350), arrowprops=dict(arrowstyle="<->", color="gray"))
ax.text((L.to_t + L.hs_t + L.stride) / 2, 1380, "SWING of left foot\n(foot in the air)", ha="center", color="gray")
ax.annotate("", xy=(L.hs_t, 1150), xytext=(L.hs_t + L.stride, 1150), arrowprops=dict(arrowstyle="<->", color="k"))
ax.text(L.hs_t + L.stride / 2, 1170, "STRIDE (same foot HS to HS)", ha="center")
ax.set_ylim(-50, 1550); ax.set_xlabel("time (s)"); ax.set_ylabel("vertical force (N)"); ax.legend(loc="center right", fontsize=7)
ax.set_title("Fig 1. One walking cycle measured with force plates (participant P01, 3 km/h)"); save("fig01_gait_cycle.png")

# ---- fig02 annotated vGRF curve
r = gt[(gt.side == 0) & (gt.hs_t > 15.4) & (gt.hs_t < 15.5)].iloc[0]
mm = (tr.force_t >= r.hs_t) & (tr.force_t < r.to_t); tt, ff = tr.force_t[mm], tr.force_left[mm]
wa, po, lr, imp = stance_scalars(tt, ff); half = len(ff) // 2; ipk = int(np.argmax(ff[:half])); rise = ff[:ipk + 1]
i20 = int(np.argmax(rise >= .2 * wa)); i80 = int(np.argmax(rise >= .8 * wa)); x = (tt - tt.iloc[0]) if hasattr(tt, "iloc") else tt - tt[0]
x = np.asarray(x)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, ff, "C0", lw=2); ax.fill_between(x, ff, alpha=.12, color="C0"); ax.text(.36, 250, f"IMPULSE = area under curve\n= {imp:.0f} N*s", ha="center")
ax.plot([x[i20], x[i80]], [ff[i20], ff[i80]], "r-o", lw=2); ax.text(x[i80] + .02, 60, f"LOADING RATE = slope\n{lr:.0f} N/s", color="r")
ax.plot(x[ipk], wa, "ks"); ax.annotate(f"WA force (1st peak)\n{wa:.0f} N", (x[ipk], wa), (0.03, 1330), arrowprops=dict(arrowstyle="->"))
ip2 = half + int(np.argmax(ff[half:])); ax.plot(x[ip2], po, "ks"); ax.annotate(f"PO force (2nd peak)\n{po:.0f} N", (x[ip2], po), (0.40, 1340), arrowprops=dict(arrowstyle="->"))
iv = ipk + int(np.argmin(ff[ipk:ip2])); ax.annotate(f"valley (mid-stance)\n{ff[iv]:.0f} N", (x[iv], ff[iv]), (0.26, 1020), arrowprops=dict(arrowstyle="->"))
ax.axhline(82.1 * 9.81, color="gray", ls="--"); ax.text(.02, 82.1 * 9.81 + 20, "body weight = 805 N", color="gray")
ax.set_ylim(0, 1500); ax.set_xlabel("time since heel strike (s)"); ax.set_ylabel("vertical ground reaction force (N)")
ax.set_title("Fig 2. The vertical ground reaction force (vGRF) curve of ONE stance and its 4 scalar parameters"); save("fig02_vgrf_curve.png")

# ---- fig03 calibration
fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
sl = t < 12
for k, n in enumerate(["ax (AP)", "ay (ML)", "az (VT)"]):
    axs[0].plot(t[sl], al[sl, k], label=n); axs[1].plot(t[sl], al_c[sl, k], label=n)
axs[0].set_title("Fig 3a. LEFT-ear accelerometer as measured, in the earbud's own (tilted) axes"); axs[1].set_title("Fig 3b. After orientation calibration: gravity is now on the vertical axis")
for a in axs: a.set_ylabel("acceleration (G)"); a.legend(ncol=3, fontsize=7); a.axvspan(0, 4, color="y", alpha=.15)
axs[0].text(.2, .2, "standing still\n(used to find gravity direction)", transform=axs[0].transAxes, fontsize=8); axs[1].set_xlabel("time (s)"); save("fig03_calibration.png")

# ---- fig04 HS detection
w0, w1 = 15.0, 18.0; s = (t >= w0) & (t <= w1)
fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
axs[0].plot(t[s], a_g[s], "k"); axs[1].plot(t[s], jerk[s], "k")
for a in axs:
    for h in hs_est[(hs_est > w0) & (hs_est < w1)]: a.axvline(h, color="g", ls="--", lw=1)
    for h in gt.hs_t[(gt.hs_t > w0) & (gt.hs_t < w1)]: a.axvline(h, color="r", ls=":", lw=1)
axs[1].plot(hs_est[(hs_est > w0) & (hs_est < w1)], jerk[hs_idx][(hs_est > w0) & (hs_est < w1)], "go", label="detected jerk peaks = heel strikes")
axs[1].plot([], [], "r:", label="true heel strike (force plate)"); axs[1].legend(fontsize=7)
axs[0].set_ylabel("resultant acceleration (G)"); axs[1].set_ylabel("jerk (G/s)"); axs[1].set_xlabel("time (s)")
axs[0].set_title("Fig 4. Heel strikes show up as sharp changes of acceleration; we find the peaks of its rate of change (jerk)"); save("fig04_heel_strike.png")

# ---- fig05 side signal
d = lowpass(al_c[:, 1] - ar_c[:, 1], C.SIDE_LOWPASS_HZ, fs); sides = classify_sides(hs_idx, al_c[:, 1], ar_c[:, 1], fs)
w0, w1 = 15.0, 20.0; s = (t >= w0) & (t <= w1)
fig, ax = plt.subplots(figsize=(10, 3.2)); ax.plot(t[s], d[s], "k"); ax.axhline(0, color="gray")
for h, sd in zip(hs_est, sides):
    if w0 < h < w1: ax.plot(h, d[int((h - t[0]) * fs)], "o", color="C0" if sd == 0 else "C1", ms=8)
ax.plot([], [], "o", color="C0", label="labelled LEFT (signal > 0)"); ax.plot([], [], "o", color="C1", label="labelled RIGHT (signal < 0)")
ax.legend(fontsize=7); ax.set_xlabel("time (s)"); ax.set_ylabel("a_ML(left ear) - a_ML(right ear)  (G)")
ax.set_title("Fig 5. Telling left from right: the head sways, and the sign of the left-minus-right sideways acceleration flips each step"); save("fig05_left_right.png")

# ---- fig06 toe off
a_vt = 0.5 * (al_c[:, 2] + ar_c[:, 2]); jv = np.gradient(a_vt) * fs; sm = lowpass(jv, C.TO_LOWPASS_HZ, fs, order=2)
ev = detect_gait_events(al_c, ar_c, fs, t[0]); w0, w1 = 15.0, 18.2; s = (t >= w0) & (t <= w1)
fig, ax = plt.subplots(figsize=(10, 3.6)); ax.plot(t[s], sm[s], "k", label="vertical jerk, low-passed at 2 Hz")
for h in ev.hs_t[(ev.hs_t > w0) & (ev.hs_t < w1)]: ax.axvline(h, color="g", ls="--", lw=.8)
for x_ in ev.to_after[(ev.to_after > w0) & (ev.to_after < w1)]: ax.axvline(x_, color="m", lw=1.5)
for x_ in gt.to_t[(gt.to_t > w0) & (gt.to_t < w1)]: ax.axvline(x_, color="r", ls=":", lw=1.2)
ax.plot([], [], "g--", label="heel strike"); ax.plot([], [], "m-", label="estimated toe off (trough)"); ax.plot([], [], "r:", label="true toe off (force plate)")
ax.legend(fontsize=7, ncol=2); ax.set_xlabel("time (s)"); ax.set_ylabel("G/s"); ax.set_title("Fig 6. Toe off is subtle: we look for the trough of a heavily smoothed vertical jerk after each heel strike"); save("fig06_toe_off.png")

# ---- fig07 vertical displacement
est = estimate_vertical_displacement(a_vt, fs); mz = lowpass(np.interp(t, tr.mocap_t, tr.mocap_z_mm), 6.0, fs)
w0, w1 = 20.0, 27.0; s = (t >= w0) & (t <= w1)
fig, ax = plt.subplots(figsize=(10, 3.6)); ax.plot(t[s], mz[s], "r", lw=2, label="motion capture (ground truth)"); ax.plot(t[s], est[s], "k", label="WalkEar estimate from earbuds")
for h in hs_est[(hs_est > w0) & (hs_est < w1)]: ax.axvline(h, color="g", ls="--", lw=.7)
ax.set_xlabel("time (s)"); ax.set_ylabel("vertical position of body (mm)"); ax.legend(fontsize=7)
ax.set_title("Fig 7. Vertical displacement: the body bounces up and down ~7 mm per step; per-step value = max - min between two heel strikes (dashed)"); save("fig07_vertical_displacement.png")

# ---- fig08 head motion
tr2 = load_trial(C.DATA_RAW / "P06" / "treadmill_5kph", "P06", 51.7)
t2, al2, gl2, ar2, gr2 = resample_both_ears(tr2, fs); gl2c = calibrate_orientation(al2, gl2, fs)[1]; g2 = highpass(gl2c, C.HEAD_HP_CUTOFF, fs); mk = detect_head_motion(gl2c, fs)
w0, w1 = 28, 38; s = (t2 >= w0) & (t2 <= w1)
fig, ax = plt.subplots(figsize=(10, 3.4))
for k, n in enumerate(["gyro x", "gyro y", "gyro z"]): ax.plot(t2[s], g2[s, k], label=n, lw=1)
ax.axhline(3, color="r", ls="--"); ax.axhline(-3, color="r", ls="--"); ax.fill_between(t2[s], -5, 5, where=mk[s], color="r", alpha=.15, label="flagged as head motion")
ax.set_ylim(-5, 5); ax.legend(fontsize=7, ncol=4); ax.set_xlabel("time (s)"); ax.set_ylabel("angular velocity (rad/s)")
ax.set_title("Fig 8. Walking only rotates the head gently; a real head turn spikes the gyroscope above 3 rad/s"); save("fig08_head_motion.png")

# ---- fig09 filters demo
rng = np.random.default_rng(1); tt = np.arange(0, 4, 1 / fs)
sig = 0.05 * np.sin(2 * np.pi * 1.85 * tt); drift = 0.06 * tt / 4 - .03; noise = rng.normal(0, .02, len(tt)); shock = np.zeros(len(tt))
for h in (.5, 1.04, 1.58, 2.12, 2.66, 3.2): k = int(h * fs); shock[k:k + 8] += 0.2 * np.exp(-np.arange(8) / 2.) * np.cos(np.arange(8) * 1.6)
x_ = sig + drift + noise + shock
fig, axs = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
axs[0].plot(tt, x_, "k"); axs[0].set_title("Fig 9a. A measured signal = slow 1.85 Hz bounce + slow drift + noise + sharp impacts")
axs[1].plot(tt, lowpass(x_, 5, fs), "C0"); axs[1].plot(tt, sig + drift, "gray", ls="--", lw=1); axs[1].set_title("9b. LOW-pass 5 Hz: keeps slow parts (bounce + drift), removes noise and impacts")
axs[2].plot(tt, highpass(x_, .5, fs), "C1"); axs[2].set_title("9c. HIGH-pass 0.5 Hz: removes the slow drift, keeps bounce + noise + impacts")
axs[3].plot(tt, highpass(lowpass(x_, 5, fs), .5, fs), "C2"); axs[3].plot(tt, sig, "gray", ls="--", lw=1, label="true bounce"); axs[3].legend(fontsize=7); axs[3].set_title("9d. Low-pass THEN high-pass (band-pass 0.5-5 Hz): recovers the bounce")
axs[3].set_xlabel("time (s)"); save("fig09_filters.png")

# ---- fig10 toy GP
xs = np.array([1., 2., 3., 5.]); ys = np.array([1.0, 2.1, 2.9, 5.2]); ell = 2.; noise = .01; ym = ys.mean()
kf = lambda a, b: np.exp(-np.abs(a[:, None] - b[None, :]) / ell); Ki = np.linalg.inv(kf(xs, xs) + noise * np.eye(4)); xg = np.linspace(0, 12, 200); ks = kf(xg, xs)
mu = ym + ks @ Ki @ (ys - ym); sd = np.sqrt(np.clip(1 - np.einsum("ij,jk,ik->i", ks, Ki, ks), 0, None)); b = np.polyfit(xs, ys, 1)
fig, ax = plt.subplots(figsize=(8, 4)); ax.fill_between(xg, mu - 2 * sd, mu + 2 * sd, alpha=.2, label="GP: +-2 std (uncertainty)")
ax.plot(xg, mu, "C0", lw=2, label="GP prediction (exponential kernel, length 2)"); ax.plot(xg, np.polyval(b, xg), "C3--", label="straight-line regression"); ax.plot(xs, ys, "ko", ms=8, label="training data")
ax.axhline(ym, color="gray", ls=":"); ax.text(9, ym + .1, "mean of y", color="gray"); ax.legend(fontsize=7); ax.set_xlabel("input x"); ax.set_ylabel("output y")
ax.set_title("Fig 10. Toy Gaussian process: follows the data nearby, falls back to the average far away and says how unsure it is"); save("fig10_toy_gp.png")

# ---- fig11 toy boosting
X = np.arange(1, 9.); Y = np.array([1.0, 1.2, 1.1, 3.0, 3.2, 2.9, 5.1, 5.0]); pred = np.full(8, Y.mean()); fig, axs = plt.subplots(1, 4, figsize=(12, 2.9), sharey=True); xx = np.linspace(.5, 8.5, 200); fn = lambda p: np.interp(xx, X, p, left=p[0], right=p[-1])
def step_plot(a, p, title): a.plot(X, Y, "ko"); a.step(X, p, where="mid", color="C0"); a.set_title(title, fontsize=8)
step_plot(axs[0], pred, "start: predict the mean")
for r in range(3):
    res = Y - pred; best = None
    for th in (1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5):
        Lr, Rr = res[X < th], res[X >= th]; sse = ((Lr - Lr.mean()) ** 2).sum() + ((Rr - Rr.mean()) ** 2).sum()
        if best is None or sse < best[0]: best = (sse, th, Lr.mean(), Rr.mean())
    pred = pred + .5 * np.where(X < best[1], best[2], best[3]); step_plot(axs[r + 1], pred, f"after tree {r + 1} (split at x<{best[1]})")
fig.suptitle("Fig 11. Toy gradient boosting: each small tree fixes what is still wrong (learning rate 0.5)", fontsize=9); save("fig11_toy_boosting.png")

# ---- fig13 force model
xx = np.linspace(0, 1, 400); A1, A2, B, w1 = 1.1, 1.05, .8, .11
c1 = A1 * np.exp(-((xx - .2) / w1) ** 2); c2 = A2 * np.exp(-((xx - .78) / .12) ** 2); c3 = B * .9 * np.exp(-((xx - .5) / .35) ** 2)
r_in = np.clip(xx / .04, 0, 1); r_out = np.clip((1 - xx) / .10, 0, 1); edge = (3 * r_in ** 2 - 2 * r_in ** 3) * (3 * r_out ** 2 - 2 * r_out ** 3)
fig, ax = plt.subplots(figsize=(8, 4)); ax.plot(xx, c1, "C0--", label="bump 1 (heel loading) at 20% of stance"); ax.plot(xx, c2, "C1--", label="bump 2 (push-off) at 78%"); ax.plot(xx, c3, "C2--", label="broad base (body weight support)")
ax.plot(xx, edge, "gray", ls=":", label="edge ramp (force fades in/out)"); ax.plot(xx, (c1 + c2 + c3) * edge, "k", lw=2.5, label="SUM x edge = synthetic vGRF (in body weights)")
ax.legend(fontsize=7); ax.set_xlabel("fraction of stance (0 = heel strike, 1 = toe off)"); ax.set_ylabel("force / body weight"); ax.set_title("Fig 13. How the synthetic vGRF curve is built: three smooth bumps added together"); save("fig13_synthetic_force.png")

# ---- fig14 impact template
import walkear.synth as S
T = S.IMPACT_TEMPLATE; k = np.arange(len(T)) / S.FS_SIM; v = np.cumsum(T) / S.FS_SIM; xdisp = np.cumsum(v) / S.FS_SIM
fig, axs = plt.subplots(3, 1, figsize=(8, 5), sharex=True)
axs[0].plot(k * 1000, T, "k"); axs[0].set_ylabel("acceleration\n(relative)"); axs[1].plot(k * 1000, v, "C0"); axs[1].set_ylabel("velocity\n(integral)"); axs[2].plot(k * 1000, xdisp, "C3"); axs[2].set_ylabel("displacement\n(2nd integral)")
axs[2].set_xlabel("ms after heel strike"); axs[0].set_title("Fig 14. The synthetic heel-strike 'ringing': it starts and ends at rest, so it leaves the head where it was")
for a in axs: a.axhline(0, color="gray", lw=.5)
save("fig14_impact_template.png")

# ---- fig16 scatter est vs GT
p = pd.read_csv(C.RESULTS_DIR / "predictions_loso.csv", dtype={"participant": str}); p = p[p.gt_matched]
items = [("stance", "gt_stance", "stance time (s)"), ("swing", "gt_swing", "swing time (s)"), ("cadence", "gt_cadence", "cadence (steps/min)"), ("vd", "gt_vd", "vertical displacement (mm)"),
         ("wa", "gt_wa", "WA force (N)"), ("po", "gt_po", "PO force (N)"), ("lr", "gt_lr", "loading rate (N/s)"), ("impulse", "gt_impulse", "impulse (N*s)")]
fig, axs = plt.subplots(2, 4, figsize=(13, 6))
for a, (e, g, n) in zip(axs.ravel(), items):
    q = p[[e, g]].dropna(); a.scatter(q[g], q[e], s=3, alpha=.3); lo, hi = q.min().min(), q.max().max(); a.plot([lo, hi], [lo, hi], "r-", lw=1)
    a.set_title(f"{n}\ncorr={np.corrcoef(q[g], q[e])[0, 1]:.2f}", fontsize=8); a.set_xlabel("ground truth", fontsize=7); a.set_ylabel("WalkEar estimate", fontsize=7)
fig.suptitle("Fig 16. Estimate vs ground truth for every matched step (leave-one-subject-out). Points on the red line = perfect", fontsize=10); save("fig16_scatter.png")

# ---- fig12 segments for features
r = gt[(gt.side == 0) & (gt.hs_t > 15.4) & (gt.hs_t < 15.5)].iloc[0]; agl = lowpass(resultant(al_c), 30., fs); s = (t >= r.hs_t - .3) & (t <= r.to_t + .3)
fig, ax = plt.subplots(figsize=(9, 3.4)); ax.plot(t[s], agl[s], "k", lw=1.5, label="resultant acceleration, left ear")
ax.axvspan(r.hs_t, r.hs_t + .25 * r.stance, color="C0", alpha=.25, label="LOADING segment (first 25% of stance): features for WA force + loading rate")
ax.axvspan(r.hs_t + .25 * r.stance, r.to_t, color="C1", alpha=.15, label="rest of STANCE: with the loading part, features for PO force + impulse + curve")
ax.legend(fontsize=7, loc="upper right"); ax.set_xlabel("time (s)"); ax.set_ylabel("G"); ax.set_title("Fig 12. The pieces of signal that features are computed from, for one left-foot stance"); save("fig12_segments.png")

# ---- fig17 timing jitter
dt = np.diff(tr.t_left) * 1000; fig, ax = plt.subplots(figsize=(7, 3)); ax.hist(dt, bins=40); ax.set_xlabel("time between consecutive IMU samples (ms)"); ax.set_ylabel("count")
ax.set_title("Fig 17. Raw IMU timestamps are not evenly spaced (nominal 10 ms), hence resampling"); save("fig17_timestamp_jitter.png")
import shutil; shutil.copy(C.RESULTS_DIR / "vgrf_examples_loso.png", OUT / "fig15_vgrf_predictions.png")
print("figures:", sorted(p.name for p in OUT.iterdir()))
