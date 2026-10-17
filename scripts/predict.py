"""Run the full WalkEar pipeline on ONE new raw trial folder (no ground truth needed)."""
import argparse, sys
from pathlib import Path
import numpy as np, joblib
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from walkear import config as C
from walkear.io import load_trial
from walkear.pipeline import run_imu_pipeline
from walkear.parameters import smooth_flagged

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("trial_dir", type=Path, help="folder with imu_left.csv / imu_right.csv")
    ap.add_argument("--weight_kg", type=float, required=True)
    ap.add_argument("--model", type=Path, default=C.MODELS_DIR / "walkear.joblib")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    trial = load_trial(a.trial_dir, a.trial_dir.parent.name, a.weight_kg)
    steps, XL, XS = run_imu_pipeline(trial)
    model = joblib.load(a.model)
    p = model.predict({"loading": XL, "stance": XS})
    for k, col in (("wa", "wa_N"), ("po", "po_N"), ("lr", "lr_N_per_s"), ("impulse", "impulse_Ns")):
        steps[col] = p[k] * trial.weight_n
    steps = steps.rename(columns={"vd_mm": "vertical_disp_mm"})
    steps = smooth_flagged(steps, ["step_time", "stride", "stance", "swing", "cadence", "vertical_disp_mm",
                                   "wa_N", "po_N", "lr_N_per_s", "impulse_Ns"])
    out = a.out or (C.RESULTS_DIR / f"predict_{trial.participant}_{trial.name}.csv")
    C.RESULTS_DIR.mkdir(exist_ok=True)
    steps.to_csv(out, index=False)
    np.save(Path(out).with_suffix(".curves.npy"), p["curve"])
    print(steps.drop(columns=["participant"]).describe().T[["count", "mean", "std"]].round(3))
    print("wrote", out)
