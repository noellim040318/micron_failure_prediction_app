"""
SHAP computation engine using TreeExplainer for XGBoost.
Returns SHAP values and generates standard visualisations.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import io
import warnings
warnings.filterwarnings("ignore")


class SHAPEngine:
    def __init__(self, model, feature_names: list):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)
        self._background_shap_values = None

    def explain(self, X: np.ndarray) -> np.ndarray:
        """Return SHAP values for the given input array."""
        sv = self.explainer.shap_values(X)
        # XGBoost binary classification returns a single array
        if isinstance(sv, list):
            sv = sv[1]
        return sv

    def explain_single(self, x: np.ndarray) -> dict:
        """Return per-feature SHAP attribution for one unit."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        sv = self.explain(x)[0]
        base = float(self.explainer.expected_value)
        if isinstance(base, (list, np.ndarray)):
            base = base[-1]
        return {
            "base_value": base,
            "shap_values": dict(zip(self.feature_names, sv.tolist())),
            "total_contribution": float(sv.sum()),
        }

    def waterfall_figure(self, x: np.ndarray, max_display: int = 15) -> bytes:
        """Render a SHAP waterfall plot for one unit. Returns PNG bytes."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        sv = self.explainer(x)
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.plots.waterfall(sv[0], max_display=max_display, show=False)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        plt.close("all")
        buf.seek(0)
        return buf.read()

    def beeswarm_figure(self, X: np.ndarray, max_display: int = 20) -> bytes:
        """Render a SHAP beeswarm summary plot. Returns PNG bytes."""
        sv = self.explain(X)
        fig, ax = plt.subplots(figsize=(11, 8))
        shap.summary_plot(sv, X, feature_names=self.feature_names,
                          max_display=max_display, show=False)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        plt.close("all")
        buf.seek(0)
        return buf.read()

    def bar_figure(self, X: np.ndarray, max_display: int = 20) -> bytes:
        """Render a SHAP mean-absolute bar chart. Returns PNG bytes."""
        sv = self.explain(X)
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(sv, X, feature_names=self.feature_names,
                          plot_type="bar", max_display=max_display, show=False)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        plt.close("all")
        buf.seek(0)
        return buf.read()

    def top_features(self, x: np.ndarray, n: int = 10) -> pd.DataFrame:
        """Return top-N features by |SHAP| magnitude for one unit."""
        exp = self.explain_single(x)
        sv = exp["shap_values"]
        df = pd.DataFrame({
            "feature": list(sv.keys()),
            "shap_value": list(sv.values()),
        })
        df["abs_shap"] = df["shap_value"].abs()
        df = df.sort_values("abs_shap", ascending=False).head(n).reset_index(drop=True)
        df["direction"] = df["shap_value"].apply(lambda v: "↑ increases risk" if v > 0 else "↓ decreases risk")
        return df
