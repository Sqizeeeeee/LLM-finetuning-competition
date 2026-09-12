# Experiment 00: manual features baseline

## Setup
- Features: length (abs + relative), markdown formatting (bullets, numbered, headers, bold, code), refusal detection
- Models: LogisticRegression (scaled features), LightGBM (multiclass, early stopping)
- Ensemble: weighted average of predicted probabilities (lgbm_weight in config.yaml)
- CV: 5 precomputed GroupKFold folds (grouped by prompt), from data/processed/common

## Results

| model                     | OOF log loss |
|---------------------------|--------------|
| logreg                    | ~1.062       |
| lgbm                      | 1.04585      |
| ensemble (w=0.5)          | ~1.0507      |
| ensemble (w=0.95, tuned)  | 1.04576      |

Kaggle Public Score (submitted notebook, full private test.csv): **1.04598**

Local OOF and Kaggle public score match almost exactly (1.04576 vs 1.04598) — no
overfitting to local folds, pipeline ported cleanly.

## Observations
- LightGBM stably outperforms LogReg on every fold (no leakage/anomalies across folds).
- Naive 50/50 ensemble is worse than pure LightGBM - LogReg is too weak and drags it down.
- Tuned ensemble weight (w=0.95) barely beats pure LightGBM - practically negligible value add.
- LogReg adds ~no real value on top of LightGBM for these features, but is cheap to keep.

## Porting to Kaggle (code competition) - gotchas for next time
- Pickled sklearn objects are NOT safely portable across environments: local sklearn 1.9.1
  vs Kaggle's 1.6.1 broke `LogisticRegression.predict_proba` (removed `multi_class` attr).
  Fix: export raw params (coef_, intercept_, scaler mean_/scale_) to .npz and reimplement
  predict_proba manually via softmax — version-independent.
- LightGBM's native `.txt` model format has no such issue, loads fine across versions.
- Re-uploading a dataset folder via drag & drop can double-nest the path
  (e.g. `models/models/...`) — always verify actual paths with `os.walk` after upload,
  don't assume the first path that worked interactively still holds after Save & Run All.
- Watch for stale variable redefinitions across cells: an early debug cell that
  redefines a path variable can silently override a later "fixed" cell during a
  full top-to-bottom Run All, even though it worked fine when run interactively
  out of order.
- Kaggle code competitions replace the tiny local test.csv (here: 3 rows) with the
  full private test set (~25k rows) only during the actual submission run, not during
  a regular "Save & Run All" commit against the visible sample - runtime and correctness
  should be validated with that scale gap in mind.
