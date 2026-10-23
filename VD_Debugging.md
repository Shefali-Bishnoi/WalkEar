# Vertical displacement (VD): debugging report

**Problem.** On the synthetic data VD had 16.9 % MAPE (paper: 2.72 %), far worse than every other output.
**Result after the fixes below:** 5.0 % over all steps, **4.2 % excluding the first/last 3 s of a recording**
(1.96 k steps, LOSO pipeline, `results/table2_errors_loso.csv`).

## Method: separate the pipeline into layers and measure each one
Per-step VD = max-min of the double-integrated head acceleration over one step. I compared it with ground truth
while switching error sources on one at a time (script logic in the transcript; knobs are module constants in `walkear/synth.py`).

| Layer added (cumulative) | VD MAPE, steady-state steps |
|---|---|
| Filter chain alone, exact acceleration input | 3.3 % |
| + resampling + orientation calibration, ideal sensors | 2.65 % |
| + 6 mg accelerometer noise | 3.9 % |
| + heel-strike shocks, **no sensor anti-alias filter** | 7.9 % |
| + heel-strike shocks, with sensor anti-alias filter (44 Hz) | 3.4 % |
| + realistic mocap marker noise (0.1 mm), unfiltered GT | ~6 % |

## What was wrong (6 causes; 4 in the data generator / ground truth, 2 in the code you will reuse)
| # | Cause | Where | Fix |
|---|---|---|---|
| 1 | Mocap GT built by time-domain double integration -> slow integration drift in the "truth" | generator | spectral (FFT) double integration, band >= 0.3 Hz |
| 2 | Random +-5 % scale on the mocap that no sensor could know | generator | removed |
| 3 | Per-step force intensity wandered slowly, i.e. a net vertical force that does not average to body weight (violates Newton; double integration amplifies it enormously) | generator | no force content below 0.6 Hz |
| 4 | Heel-strike transient (plain damped sine) had non-zero net velocity *and* displacement, injecting displacement the GT does not contain | generator | transient built as a second difference of a smooth bump (zero first and second moments) |
| 5 | **No anti-alias filter before ~100 Hz sampling**: the broadband shock aliased into the displacement band. Real MPU6050 has an on-chip low-pass | generator | 44 Hz sensor DLPF before sampling |
| 6 | Mocap max-min inflated by marker noise (a positive bias of ~0.3 mm in GT) | **ground-truth extraction** | low-pass mocap at 6 Hz (`MOCAP_LOWPASS_HZ`) before max-min |
| 7 | Shocks/noise leak through double integration (power weighted by 1/f^4) | **estimator** | 5 Hz low-pass before integrating (`VD_LOWPASS_HZ`) |
| 8 | Filter transients of a 0.3 Hz high-pass (3 passes) corrupt the first/last steps; default `filtfilt` padding is only a few samples | **estimator** | 4 s padding (`VD_PAD_S`) + `edge_step` flag for the first/last 3 s (`VD_EDGE_S`); report VD with and without them |
| - | Filters switched to second-order sections (numerically safe for very low cutoffs; output identical at these settings) | **estimator** | `sosfiltfilt` |

## Hypotheses I tested and rejected (so you do not repeat them)
* *Heel-strike timing jitter corrupts the step window.* False: VD with the **true** HS windows gave the same error.
* *The b,a-form Butterworth is unstable at 0.3 Hz.* False: SOS and b,a outputs were identical.
* *A higher high-pass cutoff would remove drift.* Worse (0.5 Hz: 8 %, 1 Hz: 34 %): the VD signal needs content around the stride frequency.
* *The pre-integration low-pass alone is the fix.* It helps (~1 point) but only after the anti-alias and ground-truth problems are fixed.
* An early test showed 997 % error: **harness bug** (FFT second derivative of a non-periodic signal creates edge spikes). The same lesson applies to real data: edge steps are unreliable.

## What remains, honestly
* 4.2 % vs the paper's 2.72 %: the rest is sensor noise + shocks + start/stop transients; the estimator is slightly low
  (ME -0.18 mm) because of the 5 Hz low-pass and 0.3 Hz high-pass.
* The generator's CoM motion is a scaled copy of the force sum (`KAPPA`), so VD on real data depends on how the
  head really moves relative to the CoM. Treat these numbers as a check of the *code*, not of the paper.
* HS jitter (std ~11 ms) is a *by-product*: the new shock shape needed `HS_LOWPASS_HZ=45`, `HS_FIRST_LOBE_REL=0.7`.
  **Re-tune these on real data.**

## Files changed
`walkear/synth.py`, `walkear/preprocessing.py` (SOS filters, `padlen`), `walkear/parameters.py` (VD estimator, mocap filtering),
`walkear/pipeline.py` (`edge_step` column), `walkear/evaluate.py` (`vd (excl. edge steps)` row), `walkear/config.py`
(new: `VD_LOWPASS_HZ`, `VD_PAD_S`, `VD_EDGE_S`, `MOCAP_LOWPASS_HZ`; changed: `HS_LOWPASS_HZ` 25->45, `HS_FIRST_LOBE_REL` 0.6->0.7).
