"""
Central configuration for the WalkEar re-implementation.

Everything that depends on *your* raw data (file names, column names, units,
thresholds) lives here, so that swapping the synthetic data for the real data
should only require editing this file (and, at most, `walkear/io.py`).
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# ----------------------------------------------------------------------------
# Raw data contract (see README.md, section "Data format")
# ----------------------------------------------------------------------------
PARTICIPANTS_FILE = "participants.csv"   # participant_id, weight_kg, ...
IMU_LEFT_FILE = "imu_left.csv"
IMU_RIGHT_FILE = "imu_right.csv"
FORCE_FILE = "force.csv"                 # optional (ground truth)
MOCAP_FILE = "mocap.csv"                 # optional (ground truth for VD)
META_FILE = "meta.json"                  # optional (speed, surface...)

IMU_COLS = dict(time="time_s",
                acc=["ax", "ay", "az"],
                gyro=["gx", "gy", "gz"])
FORCE_COLS = dict(time="time_s", left="fz_left_N", right="fz_right_N")
MOCAP_COLS = dict(time="time_s", z="com_z_mm")

# Unit conversion applied by the loader. Pipeline works in G and rad/s.
# MPU6050 raw LSB example: ACC_TO_G = 1/16384 (+-2 g), GYRO_TO_RADS = (1/131)*pi/180 (+-250 deg/s)
ACC_TO_G = 1.0
GYRO_TO_RADS = 1.0

# Time offset (s) added to force/mocap timestamps to bring them to the IMU clock
FORCE_TIME_OFFSET_S = 0.0
MOCAP_TIME_OFFSET_S = 0.0

# ----------------------------------------------------------------------------
# Signal processing (paper, Sec. III)
# ----------------------------------------------------------------------------
FS = 150.0                       # uniform resampling rate (Hz)
HEAD_GYRO_THRESH = 3.0           # rad/s, head-motion threshold
HEAD_HP_CUTOFF = 0.5             # Hz, high-pass applied to gyro before thresholding
HEAD_DILATE_S = 0.25             # s, grow flagged regions
STATIONARY_WIN_S = 1.0
STATIONARY_ACC_STD = 0.03        # G
STATIONARY_GYRO_MAX = 0.4        # rad/s

HS_LOWPASS_HZ = 25.0             # lowpass before computing the jerk for HS detection
HS_MIN_DISTANCE_S = 0.30
HS_REL_THRESH = 0.35             # peak must be >= this * (90th pct of candidate peaks)
HS_FIRST_LOBE_REL = 0.6          # impact ringing: earliest jerk peak >= this * P90 within 0.1 s before the tallest
TO_LOWPASS_HZ = 2.0              # paper: Butterworth 2 Hz on jerk of a_vt
SIDE_LOWPASS_HZ = 5.0            # paper: 5 Hz lowpass on (a_ml_left - a_ml_right)
VD_HIGHPASS_HZ = 0.3             # high-pass used between integration stages

STEP_TIME_RANGE_S = (0.30, 1.00)  # sanity limits for a step (HS -> opposite HS)

# ----------------------------------------------------------------------------
# Ground truth extraction
# ----------------------------------------------------------------------------
FORCE_THRESHOLD_N = 20.0         # stance = vGRF above this
GT_MATCH_TOL_S = 0.20            # est HS <-> GT HS matching tolerance
CURVE_POINTS_GT = 100            # stance curves stored at 100 pts (20-pt version is derived)

# ----------------------------------------------------------------------------
# Kinetic models
# ----------------------------------------------------------------------------
LOADING_FRACTION = 0.25          # WA force + loading rate use HS -> 25 % of stance
FEATURE_BAND_HZ = (3.0, 20.0)
GP_MAX_TRAIN = 800               # GP is O(n^3): train on a random subset
CURVE_POINTS = 20                # 20 (real-time) or 100 (offline)
SEED = 0
