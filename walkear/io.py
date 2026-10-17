"""Loading raw trials from disk. This is the ONLY place that knows the file layout."""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
from . import config as C


@dataclass
class Trial:
    participant: str
    name: str
    meta: dict
    weight_kg: float
    # IMU streams (each earbud has its own, possibly non-uniform, clock)
    t_left: np.ndarray
    acc_left: np.ndarray      # (n,3) in G, device frame
    gyro_left: np.ndarray     # (n,3) in rad/s
    t_right: np.ndarray
    acc_right: np.ndarray
    gyro_right: np.ndarray
    # Ground truth (optional)
    force_t: np.ndarray | None = None
    force_left: np.ndarray | None = None
    force_right: np.ndarray | None = None
    mocap_t: np.ndarray | None = None
    mocap_z_mm: np.ndarray | None = None

    @property
    def weight_n(self) -> float:
        return self.weight_kg * 9.81

    @property
    def has_force(self) -> bool:
        return self.force_t is not None


def load_participants(raw_dir: Path = C.DATA_RAW) -> pd.DataFrame:
    return pd.read_csv(Path(raw_dir) / C.PARTICIPANTS_FILE, dtype={"participant_id": str})


def list_trials(raw_dir: Path = C.DATA_RAW):
    """Yield (participant_id, trial_dir) for every folder that contains both IMU files."""
    raw_dir = Path(raw_dir)
    for pdir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        for tdir in sorted(t for t in pdir.iterdir() if t.is_dir()):
            if (tdir / C.IMU_LEFT_FILE).exists() and (tdir / C.IMU_RIGHT_FILE).exists():
                yield pdir.name, tdir


def _read_imu(path: Path):
    df = pd.read_csv(path)
    t = df[C.IMU_COLS["time"]].to_numpy(float)
    acc = df[C.IMU_COLS["acc"]].to_numpy(float) * C.ACC_TO_G
    gyro = df[C.IMU_COLS["gyro"]].to_numpy(float) * C.GYRO_TO_RADS
    return t, acc, gyro


def load_trial(trial_dir: Path, participant: str, weight_kg: float) -> Trial:
    trial_dir = Path(trial_dir)
    meta = {}
    if (trial_dir / C.META_FILE).exists():
        meta = json.loads((trial_dir / C.META_FILE).read_text())
    tl, al, gl = _read_imu(trial_dir / C.IMU_LEFT_FILE)
    tr, ar, gr = _read_imu(trial_dir / C.IMU_RIGHT_FILE)
    tr_obj = Trial(participant, trial_dir.name, meta, weight_kg, tl, al, gl, tr, ar, gr)
    if (trial_dir / C.FORCE_FILE).exists():
        f = pd.read_csv(trial_dir / C.FORCE_FILE)
        tr_obj.force_t = f[C.FORCE_COLS["time"]].to_numpy(float) + C.FORCE_TIME_OFFSET_S
        tr_obj.force_left = f[C.FORCE_COLS["left"]].to_numpy(float)
        tr_obj.force_right = f[C.FORCE_COLS["right"]].to_numpy(float)
    if (trial_dir / C.MOCAP_FILE).exists():
        m = pd.read_csv(trial_dir / C.MOCAP_FILE)
        tr_obj.mocap_t = m[C.MOCAP_COLS["time"]].to_numpy(float) + C.MOCAP_TIME_OFFSET_S
        tr_obj.mocap_z_mm = m[C.MOCAP_COLS["z"]].to_numpy(float)
    return tr_obj


def iter_trials(raw_dir: Path = C.DATA_RAW):
    parts = load_participants(raw_dir).set_index("participant_id")
    for pid, tdir in list_trials(raw_dir):
        yield load_trial(tdir, pid, float(parts.loc[pid, "weight_kg"]))
