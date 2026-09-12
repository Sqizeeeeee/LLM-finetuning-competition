"""
Trains LogReg and LightGBM on hand-crafted features, evaluates both
with GroupKFold CV (folds precomputed in preprocessing), and builds a
simple weighted-average ensemble of the two.

Saves: OOF predictions, trained model weights, training-curve plots,
and a metrics summary, all under res/.

Usage:
    python experiments/00_manual_features/train.py
"""

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

from boosting import BoostingModel
from logreg import LogRegModel

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent


def load_config() -> dict:
    with open(EXP_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def run_cv(df: pd.DataFrame, config: dict):
    feature_cols = config["data"]["feature_cols"]
    target_col = config["data"]["target_col"]
    fold_col = config["data"]["fold_col"]

    le = LabelEncoder()
    y_all = le.fit_transform(df[target_col])
    class_names = le.classes_
    n_classes = len(class_names)

    n_rows = len(df)
    oof_logreg = np.zeros((n_rows, n_classes))
    oof_lgbm = np.zeros((n_rows, n_classes))

    fold_scores = {"logreg": [], "lgbm": [], "ensemble": []}
    lgbm_eval_history = []

    folds = sorted(df[fold_col].unique())

    for fold in folds:
        val_mask = df[fold_col] == fold
        train_mask = ~val_mask

        X_train = df.loc[train_mask, feature_cols].values
        X_val = df.loc[val_mask, feature_cols].values
        y_train = y_all[train_mask.values]
        y_val = y_all[val_mask.values]

        print(f"\n=== Fold {fold} === train={len(X_train)} val={len(X_val)}")

        # --- LogReg ---
        logreg = LogRegModel(config["logreg"])
        logreg.fit(X_train, y_train)
        val_pred_logreg = logreg.predict_proba(X_val)
        oof_logreg[val_mask.values] = val_pred_logreg
        score_logreg = log_loss(y_val, val_pred_logreg, labels=list(range(n_classes)))
        fold_scores["logreg"].append(score_logreg)
        print(f"[logreg] fold {fold} log loss: {score_logreg:.5f}")

        # --- LightGBM ---
        boosting = BoostingModel(config["lightgbm"])
        boosting.fit(X_train, y_train, X_val, y_val)
        val_pred_lgbm = boosting.predict_proba(X_val)
        oof_lgbm[val_mask.values] = val_pred_lgbm
        score_lgbm = log_loss(y_val, val_pred_lgbm, labels=list(range(n_classes)))
        fold_scores["lgbm"].append(score_lgbm)
        print(f"[lgbm]   fold {fold} log loss: {score_lgbm:.5f} "
              f"(best_iteration={boosting.model.best_iteration})")

        lgbm_eval_history.append(boosting.model.best_score)

        # --- Ensemble (this fold) ---
        w = config["ensemble"]["lgbm_weight"]
        val_pred_ens = w * val_pred_lgbm + (1 - w) * val_pred_logreg
        score_ens = log_loss(y_val, val_pred_ens, labels=list(range(n_classes)))
        fold_scores["ensemble"].append(score_ens)
        print(f"[ens]    fold {fold} log loss: {score_ens:.5f}")

        # save model weights for this fold
        models_dir = EXP_DIR / "res" / "models"
        with open(models_dir / f"logreg_fold{fold}.pkl", "wb") as f:
            pickle.dump(logreg, f)
        boosting.model.save_model(str(models_dir / f"lgbm_fold{fold}.txt"))

    w = config["ensemble"]["lgbm_weight"]
    oof_ensemble = w * oof_lgbm + (1 - w) * oof_logreg

    overall_scores = {
        "logreg": log_loss(y_all, oof_logreg, labels=list(range(n_classes))),
        "lgbm": log_loss(y_all, oof_lgbm, labels=list(range(n_classes))),
        "ensemble": log_loss(y_all, oof_ensemble, labels=list(range(n_classes))),
    }

    return {
        "class_names": class_names,
        "fold_scores": fold_scores,
        "overall_scores": overall_scores,
        "oof_logreg": oof_logreg,
        "oof_lgbm": oof_lgbm,
        "oof_ensemble": oof_ensemble,
        "y_all": y_all,
    }


def save_oof(df: pd.DataFrame, results: dict, config: dict):
    oof_dir = EXP_DIR / "res" / "oof"
    class_names = results["class_names"]

    out = df[["id", config["data"]["fold_col"]]].copy()
    for model_name in ["logreg", "lgbm", "ensemble"]:
        preds = results[f"oof_{model_name}"]
        for i, cls in enumerate(class_names):
            out[f"{model_name}_pred_{cls}"] = preds[:, i]

    out.to_parquet(oof_dir / "oof_predictions.parquet", index=False)
    print(f"\n[save] OOF predictions -> {oof_dir / 'oof_predictions.parquet'}")


def plot_fold_scores(results: dict, config: dict):
    plots_dir = EXP_DIR / "res" / "plots"
    fold_scores = results["fold_scores"]

    fig, ax = plt.subplots(figsize=(7, 5))
    folds = range(len(fold_scores["logreg"]))
    for model_name, marker in [("logreg", "o"), ("lgbm", "s"), ("ensemble", "^")]:
        ax.plot(folds, fold_scores[model_name], marker=marker, label=model_name)
    ax.set_xlabel("fold")
    ax.set_ylabel("log loss")
    ax.set_title("CV log loss by fold")
    ax.legend()
    ax.set_xticks(list(folds))
    plt.tight_layout()
    fig.savefig(plots_dir / "cv_scores_by_fold.png", dpi=150)
    plt.close(fig)
    print(f"[save] plot -> {plots_dir / 'cv_scores_by_fold.png'}")


def save_metrics(results: dict):
    metrics_path = EXP_DIR / "res" / "metrics.json"
    payload = {
        "fold_scores": results["fold_scores"],
        "overall_oof_scores": results["overall_scores"],
    }
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[save] metrics -> {metrics_path}")


def main():
    config = load_config()
    df = pd.read_parquet(ROOT / config["data"]["path"])
    print(f"[load] {len(df)} rows from {config['data']['path']}")

    results = run_cv(df, config)

    print("\n=== Overall OOF log loss ===")
    for name, score in results["overall_scores"].items():
        print(f"{name:10s}: {score:.5f}")

    save_oof(df, results, config)
    plot_fold_scores(results, config)
    save_metrics(results)


if __name__ == "__main__":
    main()
