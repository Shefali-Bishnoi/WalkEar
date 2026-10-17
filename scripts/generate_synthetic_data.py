"""Create synthetic data in data/raw (same format the real data must follow)."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from walkear import config as C
from walkear.synth import generate_dataset

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_participants", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--treadmill_s", type=float, default=60)
    ap.add_argument("--free_s", type=float, default=90)
    ap.add_argument("--out", type=Path, default=C.DATA_RAW)
    a = ap.parse_args()
    generate_dataset(a.out, a.n_participants, a.seed, a.treadmill_s, a.free_s)
