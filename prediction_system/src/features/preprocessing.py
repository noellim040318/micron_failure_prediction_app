"""
Data preprocessing pipeline: missing value imputation, encoding,
scaling, and SMOTE oversampling for class imbalance.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
import joblib
from pathlib import Path


class PreprocessingPipeline:
    def __init__(self, smote_strategy: str = "smote", random_state: int = 42):
        self.smote_strategy = smote_strategy
        self.random_state = random_state
        self.num_imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.cat_encoders = {}
        self.feature_names_out = None
        self.fitted = False

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        df = df.copy()
        cat_cols = ["tool_chamber_id", "operator_shift", "wafer_edge_flag"]
        for col in cat_cols:
            if col not in df.columns:
                continue
            # Normalise to consistent string: int-like floats become "1" not "1.0"
            def _to_str(s):
                try:
                    return str(int(float(s)))
                except (ValueError, TypeError):
                    return str(s)
            col_str = df[col].apply(_to_str)
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(col_str)
                self.cat_encoders[col] = le
            else:
                le = self.cat_encoders.get(col)
                if le is not None:
                    # Handle unseen labels by mapping to most frequent seen class
                    known = set(le.classes_)
                    col_str = col_str.apply(lambda v: v if v in known else le.classes_[0])
                    df[col] = le.transform(col_str)
        return df

    def fit_transform(self, df: pd.DataFrame, feature_cols: list, labels: pd.Series):
        df = self._encode_categoricals(df, fit=True)
        X = df[feature_cols].copy()

        # Impute numerics
        X_imputed = self.num_imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)

        self.feature_names_out = feature_cols
        self.fitted = True

        # Resample
        X_res, y_res = self._resample(X_scaled, labels.values)
        return X_res, y_res

    def transform(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        df = self._encode_categoricals(df, fit=False)
        X = df[feature_cols].copy()
        X_imputed = self.num_imputer.transform(X)
        return self.scaler.transform(X_imputed)

    def _resample(self, X, y):
        n_minority = (y == 1).sum()
        n_majority = (y == 0).sum()
        if n_minority < 6:
            return X, y

        target_ratio = min(0.3, n_minority / n_majority + 0.2)

        if self.smote_strategy == "smote":
            sampler = SMOTE(
                sampling_strategy=target_ratio,
                k_neighbors=min(5, n_minority - 1),
                random_state=self.random_state,
            )
        elif self.smote_strategy == "adasyn":
            sampler = ADASYN(
                sampling_strategy=target_ratio,
                n_neighbors=min(5, n_minority - 1),
                random_state=self.random_state,
            )
        else:
            sampler = SMOTETomek(random_state=self.random_state)

        X_res, y_res = sampler.fit_resample(X, y)
        return X_res, y_res

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "PreprocessingPipeline":
        return joblib.load(path)
