"""
Exports the pickled LogRegModel objects (per fold) into plain numpy
arrays saved as .npz files. Pickle is not safely portable across
scikit-learn versions (Kaggle's environment often differs from local),
so we extract only the raw learned parameters here, which have no
version dependency at all.

Run this locally, once, after train.py has produced the fold pickles.
Upload the resulting res/models_npz/ folder to the Kaggle dataset
alongside (or instead of) the .pkl files.

Usage:
    python experiments/00_manual_features/export_logreg_npz.py
"""

import pickle
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).resolve().parent
MODELS_DIR = EXP_DIR / "res" / "models"
OUT_DIR = EXP_DIR / "res" / "models_npz"

N_FOLDS = 5


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for fold in range(N_FOLDS):
        pkl_path = MODELS_DIR / f"logreg_fold{fold}.pkl"
        with open(pkl_path, "rb") as f:
            wrapper = pickle.load(f)

        lr = wrapper.model
        scaler = wrapper.scaler

        np.savez(
            OUT_DIR / f"logreg_fold{fold}.npz",
            coef=lr.coef_,
            intercept=lr.intercept_,
            classes=lr.classes_,
            scaler_mean=scaler.mean_,
            scaler_scale=scaler.scale_,
        )
        print(f"[export] fold {fold}: coef shape {lr.coef_.shape}, classes {lr.classes_}")

    print(f"[done] exported {N_FOLDS} folds to {OUT_DIR}")


if __name__ == "__main__":
    main()