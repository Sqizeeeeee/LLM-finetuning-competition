"""Evaluate frozen encoder features against the saved exp0 LightGBM OOF."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import log_loss

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[1]
EXP0 = ROOT / "experiments" / "00_manual_features"
CLASSES = ["a", "b", "tie"]
LABELS = list(range(len(CLASSES)))


def score(y, preds):
    tie = y == CLASSES.index("tie")
    return {
        "log_loss": log_loss(y, preds, labels=LABELS),
        "tie_log_loss": log_loss(y[tie], preds[tie], labels=LABELS),
    }


def compare(df, y, baseline, manual, similarity, difference, config):
    sys.path.insert(0, str(EXP0))
    from boosting import BoostingModel

    folds = df[config["data"]["fold_col"]].to_numpy()
    results = {"exp0_lightgbm": baseline}
    features = np.column_stack((manual, similarity))
    for name, matrix in [
        ("similarity", features),
        ("similarity_and_difference", np.column_stack((features, difference))),
    ]:
        preds = np.empty_like(baseline)
        for fold in np.unique(folds):
            val = folds == fold
            model = BoostingModel(config["lightgbm"])
            model.fit(matrix[~val], y[~val], matrix[val], y[val])
            preds[val] = model.predict_proba(matrix[val])
            print(f"{name} fold {fold}: {score(y[val], preds[val])}", flush=True)
        results[name] = preds
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=EXP_DIR / "res" / "features.npz")
    parser.add_argument("--output-dir", type=Path, default=EXP_DIR / "res")
    args = parser.parse_args()

    config = yaml.safe_load((EXP0 / "config.yaml").read_text())
    data_config = config["data"]
    feature_cols = data_config["feature_cols"]
    target_col, fold_col = data_config["target_col"], data_config["fold_col"]
    df = pd.read_parquet(
        ROOT / data_config["path"], columns=["id", fold_col, target_col, *feature_cols]
    )
    oof = pd.read_parquet(EXP0 / "res" / "oof" / "oof_predictions.parquet")
    oof = df[["id", fold_col]].merge(
        oof, on=["id", fold_col], how="left", validate="one_to_one", indicator=True,
    )
    if not oof["_merge"].eq("both").all():
        raise ValueError("Saved exp0 predictions do not match the training IDs and folds.")
    baseline = oof[[f"lgbm_pred_{cls}" for cls in CLASSES]].to_numpy()
    y = df[target_col].map({cls: i for i, cls in enumerate(CLASSES)}).to_numpy()
    print(f"exp0 LightGBM: {score(y, baseline)}", flush=True)

    with np.load(args.cache) as cache:
        ids = pd.Index(cache["ids"])
        order = ids.get_indexer(df["id"])
        if len(ids) != len(df) or (order < 0).any():
            raise ValueError("Embedding cache must contain every exp0 training ID.")
        similarity = cache["similarity"][order]
        difference = cache["difference"][order]
        encoder = {"model": str(cache["model"]), "max_length": int(cache["max_length"])}
    if not np.isfinite(similarity).all() or not np.isfinite(difference).all():
        raise ValueError("Embedding cache contains non-finite features.")

    results = compare(
        df, y, baseline, df[feature_cols].to_numpy(), similarity, difference, config,
    )
    metrics = {}
    predictions = df[["id", fold_col]].copy()
    baseline_score = score(y, baseline)["log_loss"]
    folds = df[fold_col].to_numpy()
    for name, preds in results.items():
        overall = score(y, preds)
        fold_scores = {}
        for fold in np.unique(folds):
            val = folds == fold
            fold_scores[int(fold)] = score(y[val], preds[val])
        metrics[name] = {
            **overall,
            "delta_vs_exp0": overall["log_loss"] - baseline_score,
            "folds": fold_scores,
        }
        for i, cls in enumerate(CLASSES):
            predictions[f"{name}_pred_{cls}"] = preds[:, i]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(args.output_dir / "oof_predictions.parquet", index=False)
    (args.output_dir / "metrics.json").write_text(
        json.dumps({"encoder": encoder, "models": metrics}, indent=2) + "\n"
    )
    for name, model_metrics in metrics.items():
        print(f"{name}: {model_metrics['log_loss']:.6f} "
              f"(delta {model_metrics['delta_vs_exp0']:+.6f})")


if __name__ == "__main__":
    main()
