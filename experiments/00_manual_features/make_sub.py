"""
Runs inference on test.csv using the 5 fold-models trained by train.py
(LogReg + LightGBM), averages predictions across folds (bagging) and
blends the two model types using the tuned ensemble weight, then
writes submissions/manual_features.csv in the format the competition
expects.

Reuses the same text-parsing and feature-engineering logic as the
preprocessing scripts, so train/test features are built identically.

Usage:
    python experiments/00_manual_features/inference.py
"""

import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT / "preprocessing"))
from common import parse_text_columns  # type: ignore # noqa: E402
from classical import ( # type: ignore # noqa: E402
    add_length_features,
    add_formatting_features,
    add_refusal_features,
)

TEST_PATH = ROOT / "data/raw/test.csv"
SUB_DIR = ROOT / "submissions"
SUB_PATH = SUB_DIR / "manual_features.csv"

N_FOLDS = 5
CLASS_NAMES = ["a", "b", "tie"]  # matches LabelEncoder alphabetical order from train.py


def load_config() -> dict:
    with open(EXP_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def build_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    df = parse_text_columns(df)
    df = add_length_features(df)
    df = add_formatting_features(df)
    df = add_refusal_features(df)
    return df[feature_cols]


def predict_fold(fold: int, X: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray]:
    models_dir = EXP_DIR / "res" / "models"

    with open(models_dir / f"logreg_fold{fold}.pkl", "rb") as f:
        logreg = pickle.load(f)
    logreg_pred = logreg.predict_proba(X)

    booster = lgb.Booster(model_file=str(models_dir / f"lgbm_fold{fold}.txt"))
    lgbm_pred = booster.predict(X, num_iteration=booster.best_iteration)

    return logreg_pred, lgbm_pred


def main() -> None:
    config = load_config()
    feature_cols = config["data"]["feature_cols"]
    w = config["ensemble"]["lgbm_weight"]

    df = pd.read_csv(TEST_PATH)
    print(f"[load] {len(df)} rows from {TEST_PATH}")

    X = build_features(df.copy(), feature_cols).values

    logreg_preds, lgbm_preds = [], []
    for fold in range(N_FOLDS):
        logreg_pred, lgbm_pred = predict_fold(fold, X, config)
        logreg_preds.append(logreg_pred)
        lgbm_preds.append(lgbm_pred)
        print(f"[fold {fold}] predicted")

    # bagging across folds: simple average
    logreg_avg = np.mean(logreg_preds, axis=0)
    lgbm_avg = np.mean(lgbm_preds, axis=0)

    # blend the two model types with the tuned weight
    final_pred = w * lgbm_avg + (1 - w) * logreg_avg

    submission = pd.DataFrame({
        "id": df["id"],
        "winner_model_a": final_pred[:, CLASS_NAMES.index("a")],
        "winner_model_b": final_pred[:, CLASS_NAMES.index("b")],
        "winner_tie": final_pred[:, CLASS_NAMES.index("tie")],
    })

    SUB_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUB_PATH, index=False)
    print(f"[save] submission -> {SUB_PATH}")
    print(submission.head())


if __name__ == "__main__":
    main()
