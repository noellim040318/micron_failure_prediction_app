"""
Concept drift detection for ongoing model monitoring.
Uses Population Stability Index (PSI) and Kolmogorov-Smirnov tests
to flag when incoming data distribution diverges from training data.
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


PSI_STABLE = 0.1
PSI_MONITOR = 0.25   # warning zone
PSI_DRIFT = 0.25     # drift confirmed

KS_ALPHA = 0.05


@dataclass
class DriftReport:
    timestamp: str
    n_features_tested: int
    n_drifted: int
    psi_scores: dict = field(default_factory=dict)
    ks_pvalues: dict = field(default_factory=dict)
    drifted_features: list = field(default_factory=list)
    overall_psi: float = 0.0
    drift_detected: bool = False
    severity: str = "stable"  # stable / warning / drift

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "n_features_tested": self.n_features_tested,
            "n_drifted": self.n_drifted,
            "overall_psi": round(self.overall_psi, 4),
            "drift_detected": self.drift_detected,
            "severity": self.severity,
            "drifted_features": self.drifted_features,
        }


def _psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index between two distributions."""
    eps = 1e-6
    bins = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    if len(bins) < 3:
        return 0.0
    exp_pct, _ = np.histogram(expected, bins=bins)
    act_pct, _ = np.histogram(actual, bins=bins)
    exp_pct = exp_pct / (exp_pct.sum() + eps)
    act_pct = act_pct / (act_pct.sum() + eps)
    psi = np.sum((act_pct - exp_pct) * np.log((act_pct + eps) / (exp_pct + eps)))
    return float(np.clip(psi, 0, 10))


class DriftDetector:
    def __init__(self, reference_df: pd.DataFrame, feature_cols: list):
        self.feature_cols = [c for c in feature_cols if c in reference_df.columns]
        self.reference = reference_df[self.feature_cols].dropna()
        self.history: list[DriftReport] = []

    def check(self, current_df: pd.DataFrame) -> DriftReport:
        import datetime
        current = current_df[self.feature_cols].dropna()
        psi_scores = {}
        ks_pvals = {}
        drifted = []

        for col in self.feature_cols:
            ref_vals = self.reference[col].dropna().values
            cur_vals = current[col].dropna().values
            if len(ref_vals) < 10 or len(cur_vals) < 5:
                continue
            psi_val = _psi(ref_vals, cur_vals)
            _, ks_p = stats.ks_2samp(ref_vals, cur_vals)
            psi_scores[col] = round(psi_val, 4)
            ks_pvals[col] = round(float(ks_p), 4)
            if psi_val > PSI_MONITOR or ks_p < KS_ALPHA:
                drifted.append(col)

        overall_psi = float(np.mean(list(psi_scores.values()))) if psi_scores else 0.0
        drift_detected = overall_psi > PSI_DRIFT or len(drifted) > len(self.feature_cols) * 0.3
        severity = (
            "drift" if overall_psi > PSI_DRIFT
            else "warning" if overall_psi > PSI_STABLE
            else "stable"
        )

        report = DriftReport(
            timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
            n_features_tested=len(psi_scores),
            n_drifted=len(drifted),
            psi_scores=psi_scores,
            ks_pvalues=ks_pvals,
            drifted_features=drifted,
            overall_psi=overall_psi,
            drift_detected=drift_detected,
            severity=severity,
        )
        self.history.append(report)
        return report

    def history_df(self) -> pd.DataFrame:
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in self.history])
