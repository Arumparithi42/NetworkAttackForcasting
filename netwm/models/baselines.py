"""Baselines trained on exactly the same samples/features as the world model.

B0 persistence       : attack within K  <=> the current window is an attack window
B1 logreg_now        : logistic regression on S_t only
B2 logreg_lagged     : logistic regression on the flattened history S_{t-L+1..t}
B3 xgboost_lagged    : gradient-boosted trees on the flattened history (optional dependency)
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class Persistence:
    name = "B0_persistence"

    def fit(self, X, y, y_now=None):
        return self

    def predict_proba(self, X, y_now=None):
        return (np.asarray(y_now) > 0).astype(np.float64)


class LogRegNow:
    name = "B1_logreg_now"

    def __init__(self, seed=0):
        self.m = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0,
                                    random_state=seed)

    def fit(self, X, y, y_now=None):
        self.m.fit(X[:, -1, :], y)
        return self

    def predict_proba(self, X, y_now=None):
        return self.m.predict_proba(X[:, -1, :])[:, 1]


class LogRegLagged:
    name = "B2_logreg_lagged"

    def __init__(self, seed=0):
        self.m = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.3,
                                    random_state=seed)

    def fit(self, X, y, y_now=None):
        self.m.fit(X.reshape(len(X), -1), y)
        return self

    def predict_proba(self, X, y_now=None):
        return self.m.predict_proba(X.reshape(len(X), -1))[:, 1]


class XGBLagged:
    name = "B3_xgboost_lagged"

    def __init__(self, seed=0):
        import xgboost as xgb
        self.m = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                   random_state=seed, n_jobs=4, eval_metric="aucpr")

    def fit(self, X, y, y_now=None):
        pos = max(y.sum(), 1)
        self.m.set_params(scale_pos_weight=float((len(y) - pos) / pos))
        self.m.fit(X.reshape(len(X), -1), y)
        return self

    def predict_proba(self, X, y_now=None):
        return self.m.predict_proba(X.reshape(len(X), -1))[:, 1]

    def contributions(self, X):
        """TreeSHAP contributions (built into xgboost; no extra dependency)."""
        import xgboost as xgb
        return self.m.get_booster().predict(xgb.DMatrix(X.reshape(len(X), -1)), pred_contribs=True)


def all_baselines(seed=0):
    models = [Persistence(), LogRegNow(seed), LogRegLagged(seed)]
    try:
        models.append(XGBLagged(seed))
    except ImportError:
        pass
    return models
