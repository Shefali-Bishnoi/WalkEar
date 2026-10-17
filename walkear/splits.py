"""
Train/test split strategies. ALL splits are by PARTICIPANT (or by explicit design for the
within-subject / mixed variants) because steps of one person are strongly correlated; a random
step-level split would leak identity and over-estimate performance.
"""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import KFold


def loso_splits(groups):
    """Leave-one-subject-out (the paper's main protocol)."""
    groups = np.asarray(groups)
    for g in np.unique(groups):
        yield f"LOSO:{g}", np.where(groups != g)[0], np.where(groups == g)[0]


def kfold_mixed_splits(n, k=5, seed=0):
    """5-fold CV over a mixture of all users' steps (what earlier related work reports)."""
    for i, (tr, te) in enumerate(KFold(k, shuffle=True, random_state=seed).split(np.arange(n))):
        yield f"KFold:{i}", tr, te


def individual_splits(groups, k=5, seed=0):
    """Per-user models: k-fold inside each participant."""
    groups = np.asarray(groups)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        for i, (tr, te) in enumerate(KFold(k, shuffle=True, random_state=seed).split(idx)):
            yield f"Indiv:{g}:{i}", idx[tr], idx[te]


def holdout_subjects(participants, test_frac=0.25, seed=0):
    """Fixed participant-level hold-out for a final train/test report."""
    rng = np.random.default_rng(seed)
    p = np.array(sorted(str(x) for x in participants))
    n_test = max(1, int(round(test_frac * len(p))))
    test = set(str(x) for x in rng.choice(p, n_test, replace=False))
    return [str(x) for x in p if x not in test], sorted(test)
