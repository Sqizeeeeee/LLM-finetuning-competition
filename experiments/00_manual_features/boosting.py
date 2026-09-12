"""
LightGBM wrapper for the manual-features baseline.

Uses early stopping against a validation fold, and exposes a
fit/predict_proba interface matching logreg.py.
"""

import lightgbm as lgb


class BoostingModel:
    def __init__(self, config: dict):
        self.config = config
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        params = {
            "objective": self.config["objective"],
            "num_class": self.config["num_class"],
            "learning_rate": self.config["learning_rate"],
            "num_leaves": self.config["num_leaves"],
            "random_state": self.config["random_state"],
            "verbose": self.config["verbose"],
        }

        train_set = lgb.Dataset(X_train, label=y_train)
        valid_sets = [train_set]
        valid_names = ["train"]

        if X_val is not None:
            val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
            valid_sets.append(val_set)
            valid_names.append("valid")

        callbacks = [lgb.log_evaluation(period=0)]
        if X_val is not None:
            callbacks.append(lgb.early_stopping(self.config["early_stopping_rounds"], verbose=False))

        self.model = lgb.train(
            params,
            train_set,
            num_boost_round=self.config["n_estimators"],
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        self.evals_result = self.model.model_to_string()
        return self

    def predict_proba(self, X):
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def feature_importance(self, feature_names):
        importances = self.model.feature_importance(importance_type="gain")
        return dict(zip(feature_names, importances))
