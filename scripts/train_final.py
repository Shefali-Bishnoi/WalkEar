"""
Fixed participant-level hold-out report + final model for deployment.
  python scripts/train_final.py                 -> hold out ~25 % of participants, report, then
                                                  retrain on ALL participants and save models/walkear.joblib
"""
import argparse, sys
from pathlib import Path
import numpy as np
import joblib
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from walkear import config as C
from walkear.dataset import load_dataset
from walkear.splits import holdout_subjects
from walkear.evaluate import cross_validate, make_tables, normalised_targets, train_mask
from walkear.models import WalkEarKinetics

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_frac", type=float, default=0.25)
    ap.add_argument("--curve_points", type=int, default=C.CURVE_POINTS, choices=[20, 100])
    ap.add_argument("--gp_max_train", type=int, default=C.GP_MAX_TRAIN)
    a = ap.parse_args()
    df, XL, XS, CV = load_dataset()
    train_p, test_p = holdout_subjects(df.participant.unique(), a.test_frac, C.SEED)
    print("train participants:", train_p, "\ntest participants :", test_p)
    tr = np.where(df.participant.isin(train_p))[0]; te = np.where(df.participant.isin(test_p))[0]
    out, pc = cross_validate(df, XL, XS, CV, [("holdout", tr, te)], a.curve_points, a.gp_max_train, verbose=False)
    # restrict the report to the held-out participants
    keep = np.zeros(len(df), bool); keep[te] = True
    out = out[keep].reset_index(drop=True); pc_t = pc[keep]; CV_t = CV[keep]
    t2, t3, cm, _ = make_tables(out, pc_t, CV_t, a.curve_points)
    print("\nHELD-OUT participants - errors\n", t2.round(4).to_string())
    print("\nasymmetry\n", t3.round(3).to_string()); print("\nvGRF", cm)

    # final model on everything
    ok = np.where(train_mask(df))[0]
    y = normalised_targets(df)
    model = WalkEarKinetics(a.curve_points, a.gp_max_train).fit(
        {"loading": XL[ok], "stance": XS[ok]}, {k: v[ok] for k, v in y.items()}, CV[ok])
    C.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, C.MODELS_DIR / "walkear.joblib")
    print("\nsaved", C.MODELS_DIR / "walkear.joblib")
