"""raw trials -> per-step dataset (IMU pipeline + matched ground truth). Run after (re)placing data/raw."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from walkear import config as C
from walkear.dataset import build_dataset

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=C.DATA_RAW)
    ap.add_argument("--out", type=Path, default=C.DATA_PROCESSED)
    a = ap.parse_args()
    df = build_dataset(a.raw, a.out)
    print(f"\n{len(df)} steps, {int(df.gt_matched.sum())} with matched ground truth -> {a.out}")
