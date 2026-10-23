# WalkEar – re-implementation (earable gait monitoring)

> **New to gait analysis / signal processing / these ML models?** Read `ProjectExplanation.md` first (a from-zero, ~2,900-line guide with figures in `docs/figures/`). `VD_Debugging.md` explains how the vertical-displacement error was reduced.

Implements the full pipeline of *"WalkEar: Holistic Gait Monitoring using Earables"*:
orientation calibration → head-motion removal → HS/TO + left/right detection → temporal & spatial
parameters → kinetic parameters (GP) → vGRF curve (gradient boosting) → asymmetry → LOSO evaluation.

The original dataset/code are not public, so `data/raw/` contains **synthetic data in the exact format
the real data must follow**. Replace it, run two commands, everything downstream works.

## Quick start
```bash
pip install -r requirements.txt
python scripts/generate_synthetic_data.py      # (optional) regenerate data/raw
python scripts/build_dataset.py                # raw trials -> data/processed (IMU pipeline + GT matching)
python scripts/train_eval.py --mode loso       # paper-style tables (II, III, V) + plot, ~3-4 min on 1 CPU
python scripts/train_final.py                  # participant hold-out report + models/walkear.joblib
python scripts/predict.py data/raw/P03/treadmill_5kph --weight_kg 70   # inference on a new recording
```
Other protocols: `--mode kfold` (5-fold over mixed users, as related work reports) and
`--mode individual` (per-user models); `--curve_points 100` for the offline vGRF model.

## Data format (what the real data must look like)
```
data/raw/
  participants.csv              participant_id, weight_kg  (+ sex, age, height_m, foot_length_cm: optional)
  P01/
    treadmill_3kph/             one folder per trial; any trial name works
      imu_left.csv              time_s, ax, ay, az, gx, gy, gz     (device frame, own clock, may be non-uniform)
      imu_right.csv             same for the right earbud
      force.csv                 time_s, fz_left_N, fz_right_N      (1 kHz; optional – ground truth)
      mocap.csv                 time_s, com_z_mm                   (optional – ground truth for vertical displacement)
      meta.json                 {"speed_kph": 3.0, "surface": "treadmill"}   (optional)
```
* Accelerometer in **G**, gyroscope in **rad/s**. If your raw files are in LSB / deg/s, set `ACC_TO_G`,
  `GYRO_TO_RADS` in `walkear/config.py` (MPU6050 values are given as a comment there).
* Column names / file names are configurable in `config.py`; if the clocks of IMU and force plates are offset use
  `FORCE_TIME_OFFSET_S`. Anything more exotic (different file type, single merged IMU file) → edit only `walkear/io.py`.
* Ground-truth files are only needed to *train/evaluate*. Trials without `force.csv` are still processed
  (see `scripts/predict.py`). Steps that are not on a force plate (free walking) are simply unmatched and ignored for training.
* Real data should include ~2–4 s of standing still at the start of each trial: this is used for the opportunistic
  orientation calibration (if there is none, the trial-mean gravity direction is used).

## What each module does (paper section)
| module | content |
|---|---|
| `io.py` | the only file that knows the on-disk layout |
| `preprocessing.py` | uniform 150 Hz resampling, gravity-based tilt calibration (Eq. 1), head-motion mask (>3 rad/s) |
| `gait_events.py` | HS from jerk peaks, TO from 2 Hz-filtered a_vt-jerk troughs, L/R from a_ml(L)−a_ml(R) |
| `parameters.py` | ground-truth extraction from force plates, vertical displacement (double integration + high-pass), head-motion replacement |
| `features.py` | 11 time/frequency features per segment and ear (loading window and stance) |
| `models.py` | exponential-kernel GP per scalar (WA, PO, LR, impulse), HistGradientBoosting for the vGRF curve (20/100 pts), BW-normalised targets |
| `metrics.py`, `evaluate.py` | MAE/SDE/MAPE/ME, symmetry index, NRMSE/correlation, CV driver |
| `splits.py` | LOSO, mixed K-fold, per-user K-fold, participant hold-out |
| `synth.py` | synthetic generator |

## Pre-processing and splitting – design decisions (so real data "just works")
* **No cross-trial statistics anywhere in preprocessing.** Resampling, calibration, filtering and event detection work
  per trial; the only fitted objects (feature scaler, GP, boosting) live inside `WalkEarKinetics` and are fitted on the
  training fold only → no leakage, and a new recording needs no refitting of preprocessing.
* **Splits are by participant** (LOSO = paper protocol). Never split steps randomly for headline numbers – steps of one
  person are highly correlated. K-fold/individual modes exist only to compare with related work.
* Training rows = matched ground truth, valid segmentation and **no head motion**. Test rows keep head-motion steps;
  their outputs are replaced by the mean of the neighbouring clean steps of the same foot (as in the paper).
* Kinetic targets are divided by body weight; predictions are multiplied by the user's weight (personalisation).
* GP is O(n³): at most `GP_MAX_TRAIN` (=800) random training steps per scalar model. Raise it if you have the time.
* All thresholds are in `config.py`; **expect to re-tune** `HS_LOWPASS_HZ`, `HS_REL_THRESH`, `HS_FIRST_LOBE_REL`, `STATIONARY_*` on real data
  (see “Things to check on real data”).

## Honest caveats
1. **Numbers on the synthetic data validate the code, not the paper.** The generator is physics-inspired (double-hump
   vGRF, head acceleration coupled to total force, impact transients tied to loading rate, random earbud tilt, gyro
   drift, head turns, jittered timestamps), but its force curves come from a low-dimensional parametric family, so
   vGRF NRMSE (~1.9 %, corr 0.999) is far better than anything real data will give. Peak forces are ~1.4 BW (paper: ~1.15 BW).
   The head's vertical-acceleration phase is tuned (`HEAD_LEAD_S`) so that the paper's TO heuristic is unbiased on it.
2. The paper does not specify some details; my choices: LR = slope between 20 % and 80 % of the weight-acceptance
   peak; band power = 3–20 Hz; both ears' features are concatenated; HS = first strong jerk lobe (impacts ring);
   L/R label = sign of the 5 Hz-filtered ML difference at the HS (+ alternation enforcement) instead of the paper's
   peak-sequence detector; tilt-only orientation calibration (yaw is not observable from gravity).
3. Filters are zero-phase (offline). For a real-time phone implementation use causal filters (expect some delay/bias).
4. The paper's real-time/battery results (Table VIII) are not reproduced – no phone app here.

## Things to check on real data
Run `build_dataset.py` and read the per-trial printout: `seg_valid` should be ≈ the number of steps and `GT-matched`
close to the number of steps on force plates. If not: (i) check sync (`FORCE_TIME_OFFSET_S`), (ii) plot jerk vs. force
HS and adjust `HS_REL_THRESH`/`HS_FIRST_LOBE_REL`, (iii) check that the ML axis sign convention holds
(outward-positive per ear after calibration; if left/right are swapped, flip `classify_sides`), (iv) check unit conversion.

## Vertical displacement
VD error was 16.9 % on the first version and is now 5.0 % (4.2 % excluding the first/last 3 s, which are flagged `edge_step`; paper: 2.72 %). The causes (several in the synthetic generator, two in the estimator: missing pre-integration low-pass and edge padding) are documented in `VD_Debugging.md`.

## Reference results on the shipped synthetic data (LOSO, 10 participants, 20-pt curve)
See `results/table2_errors_loso.csv`, `table3_asymmetry_loso.csv`, `vgrf_loso_20.json`, `vgrf_examples_loso.png`.
