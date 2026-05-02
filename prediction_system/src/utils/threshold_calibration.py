"""
Cost-aligned decision threshold calibration.
Derives the optimal classification threshold from the economics of missed
defects (false negatives) versus unnecessary re-tests (false positives).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve


# Default cost assumptions (USD equivalents, illustrative)
DEFAULT_COST_MISS = 850     # cost of one field return (repair + logistics + NPS)
DEFAULT_COST_FP = 45        # cost of one unnecessary re-test / hold


def compute_cost_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_miss: float = DEFAULT_COST_MISS,
    cost_fp: float = DEFAULT_COST_FP,
) -> dict:
    """
    Sweep classification thresholds and find the one that minimises
    total expected cost = FN*cost_miss + FP*cost_fp.
    Returns dict with optimal threshold, cost curve, and analysis.
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    n = len(y_true)
    n_pos = y_true.sum()
    n_neg = n - n_pos

    records = []
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        fp = ((preds == 1) & (y_true == 0)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()
        tp = ((preds == 1) & (y_true == 1)).sum()
        tn = ((preds == 0) & (y_true == 0)).sum()
        total_cost = fn * cost_miss + fp * cost_fp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        records.append({
            "threshold": t,
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "tn": int(tn),
            "total_cost": float(total_cost),
            "precision": precision,
            "recall": recall,
            "cost_fn": float(fn * cost_miss),
            "cost_fp": float(fp * cost_fp),
        })

    df = pd.DataFrame(records)
    best_idx = df["total_cost"].idxmin()
    best_row = df.iloc[best_idx]
    default_row = df.iloc[(df["threshold"] - 0.5).abs().idxmin()]

    return {
        "optimal_threshold": float(best_row["threshold"]),
        "optimal_cost": float(best_row["total_cost"]),
        "default_cost": float(default_row["total_cost"]),
        "cost_reduction_pct": float(
            (default_row["total_cost"] - best_row["total_cost"]) / max(default_row["total_cost"], 1) * 100
        ),
        "cost_miss": cost_miss,
        "cost_fp": cost_fp,
        "curve": df,
        "best_row": best_row.to_dict(),
    }


def theoretical_optimal_threshold(cost_miss: float, cost_fp: float, prevalence: float) -> float:
    """
    Analytical threshold from Bayes decision theory:
      t* = cost_fp / (cost_fp + cost_miss) adjusted for prevalence
    """
    base = cost_fp / (cost_fp + cost_miss)
    # Apply prevalence correction (rare events shift threshold lower)
    adjusted = base * (1 - prevalence) / max(prevalence, 1e-6)
    adjusted = adjusted / (1 + adjusted)
    return float(np.clip(adjusted, 0.01, 0.99))
