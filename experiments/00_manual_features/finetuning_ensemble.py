import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
import lightgbm as lgb
from pathlib import Path

oof = pd.read_parquet("res/oof/oof_predictions.parquet")
df = pd.read_parquet("../../data/processed/classical/train_features.parquet")

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_true = le.fit_transform(df["winner"])
classes = le.classes_

logreg_proba = oof[[f"logreg_pred_{c}" for c in classes]].values
lgbm_proba = oof[[f"lgbm_pred_{c}" for c in classes]].values

best_w, best_score = None, np.inf
for w in np.arange(0, 1.01, 0.05):
    blend = w * lgbm_proba + (1 - w) * logreg_proba
    score = log_loss(y_true, blend, labels=list(range(len(classes))))
    if score < best_score:
        best_w, best_score = w, score

print(f"best lgbm_weight={best_w:.2f}, log loss={best_score:.5f}")
print(f"pure lgbm log loss={log_loss(y_true, lgbm_proba, labels=list(range(len(classes)))):.5f}")

models_dir = Path("res/models")
for fold in range(5):
    booster = lgb.Booster(model_file=str(models_dir / f"lgbm_fold{fold}.txt"))
    print(f"fold {fold}: {booster.num_trees()} trees")
