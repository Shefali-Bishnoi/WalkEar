"""Cross-validated evaluation reproducing the paper's tables (II: errors, III: asymmetry, V: vGRF)."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from walkear import config as C
from walkear.dataset import load_dataset
from walkear.splits import loso_splits, kfold_mixed_splits, individual_splits
from walkear.evaluate import cross_validate, make_tables
from walkear.models import resample_curves

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["loso", "kfold", "individual"], default="loso")
    ap.add_argument("--curve_points", type=int, default=C.CURVE_POINTS, choices=[20, 100])
    ap.add_argument("--gp_max_train", type=int, default=C.GP_MAX_TRAIN)
    a = ap.parse_args()

    df, XL, XS, CV = load_dataset()
    print(f"{len(df)} steps | {df.participant.nunique()} participants | {int(df.gt_matched.sum())} with GT")
    splits = {"loso": lambda: loso_splits(df.participant.to_numpy()),
              "kfold": lambda: kfold_mixed_splits(len(df)),
              "individual": lambda: individual_splits(df.participant.to_numpy())}[a.mode]()
    t0 = time.time()
    out, pc = cross_validate(df, XL, XS, CV, splits, a.curve_points, a.gp_max_train)
    t2, t3, cm, out = make_tables(out, pc, CV, a.curve_points)
    print(f"\n=== Table II-style errors ({a.mode}) ===\n", t2.round(4).to_string())
    print("\n=== Table III-style asymmetry (symmetry index, %) ===\n", t3.round(3).to_string())
    print("\n=== vGRF curve ===\n", cm, f"\n(time {time.time() - t0:.0f}s)")

    C.RESULTS_DIR.mkdir(exist_ok=True)
    t2.to_csv(C.RESULTS_DIR / f"table2_errors_{a.mode}.csv")
    t3.to_csv(C.RESULTS_DIR / f"table3_asymmetry_{a.mode}.csv")
    (C.RESULTS_DIR / f"vgrf_{a.mode}_{a.curve_points}.json").write_text(json.dumps(cm, indent=1))
    out.drop(columns=[]).to_csv(C.RESULTS_DIR / f"predictions_{a.mode}.csv", index=False)

    # example reconstructions
    gt = resample_curves(CV, a.curve_points); mm = np.where(out.gt_matched.to_numpy() & np.isfinite(pc).all(1))[0]
    pick = np.random.default_rng(0).choice(mm, 6, replace=False)
    fig, ax = plt.subplots(2, 3, figsize=(11, 5), sharex=True, sharey=True)
    x = np.linspace(0, 100, a.curve_points)
    for a_, i in zip(ax.ravel(), pick):
        a_.plot(x, gt[i], "k-", label="ground truth"); a_.plot(x, pc[i], "r--", label="WalkEar")
        a_.set_title(f"{out.participant.iloc[i]} {out.trial.iloc[i]}", fontsize=8)
    ax[0, 0].legend(fontsize=7); fig.supxlabel("stance (%)"); fig.supylabel("vGRF (BW)")
    fig.tight_layout(); fig.savefig(C.RESULTS_DIR / f"vgrf_examples_{a.mode}.png", dpi=120)
