"""
XGBoost training pipeline with cost-sensitive learning and hyperparameter tuning.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix,
)
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


def compute_scale_pos_weight(y: np.ndarray, fn_weight: float = 10.0, fp_weight: float = 1.0) -> float:
    """
    Scale positive weight for XGBoost.
    fn_weight/fp_weight controls how aggressively the model penalises missed failures
    DURING TRAINING — independent of the cost-threshold calibration done post-training.
    Higher fn_weight → model is more sensitive to failures → higher recall, lower precision.
    """
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    if n_pos == 0:
        return 1.0
    return round((n_neg / n_pos) * (fn_weight / (fn_weight + fp_weight)), 2)


DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "use_label_encoder": False,
    "eval_metric": "aucpr",
    "random_state": 42,
    "n_jobs": -1,
}


class ModelTrainer:
    def __init__(self, params: dict = None, fn_train_weight: float = 10.0):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.fn_train_weight = fn_train_weight
        self.model: xgb.XGBClassifier | None = None
        self.feature_names: list = []
        self.threshold: float = 0.5
        self.cv_results: dict = {}

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: list):
        self.feature_names = feature_names
        spw = compute_scale_pos_weight(y, fn_weight=self.fn_train_weight)

        self.model = xgb.XGBClassifier(
            **self.params,
            scale_pos_weight=spw,
        )
        self.model.fit(X, y, verbose=False)
        return self

    def cross_validate(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
        spw = compute_scale_pos_weight(y, fn_weight=self.fn_train_weight)
        cv_model = xgb.XGBClassifier(**self.params, scale_pos_weight=spw)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        scoring = ["precision", "recall", "f1", "roc_auc", "average_precision"]
        results = cross_validate(cv_model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        self.cv_results = {k: v for k, v in results.items() if k.startswith("test_")}
        return self.cv_results

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        t = threshold if threshold is not None else self.threshold
        return (self.predict_proba(X) >= t).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray, threshold: float = None) -> dict:
        t = threshold if threshold is not None else self.threshold
        proba = self.predict_proba(X)
        preds = (proba >= t).astype(int)
        cm = confusion_matrix(y, preds)
        return {
            "precision": precision_score(y, preds, zero_division=0),
            "recall": recall_score(y, preds, zero_division=0),
            "f1": f1_score(y, preds, zero_division=0),
            "roc_auc": roc_auc_score(y, proba),
            "avg_precision": average_precision_score(y, proba),
            "confusion_matrix": cm.tolist(),
            "threshold": t,
        }

    def feature_importances(self) -> pd.Series:
        imp = self.model.feature_importances_
        return pd.Series(imp, index=self.feature_names).sort_values(ascending=False)

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "ModelTrainer":
        return joblib.load(path)
