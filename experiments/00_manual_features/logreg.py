"""
Logistic Regression wrapper for the manual-features baseline.

Scales features (important for LogReg, unlike tree models) and exposes
a simple fit/predict_proba interface matching boosting.py, so train.py
can treat both models uniformly.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class LogRegModel:
    def __init__(self, config: dict):
        self.config = config
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=config["C"],
            max_iter=config["max_iter"],
            random_state=config["random_state"],
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)
        return self

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    @property
    def classes_(self):
        return self.model.classes_
