"""
Trajectory Shapley Attribution (TSA) engine using TreeExplainer for XGBoost.
Returns TSA values and generates standard visualisations.
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


def _relabel_shap_to_tsa(fig) -> None:
    """Replace every 'SHAP' string in a matplotlib figure with 'TSA'."""
    for ax in fig.axes:
        for obj in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.texts
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            try:
                t = obj.get_text()
                if "SHAP" in t:
                    obj.set_text(t.replace("SHAP", "TSA"))
            except Exception:
                pass
    for txt in fig.texts:
        try:
            t = txt.get_text()
            if "SHAP" in t:
                txt.set_text(t.replace("SHAP", "TSA"))
        except Exception:
            pass


class TSAEngine:
    """Trajectory Shapley Attribution engine wrapping shap.TreeExplainer."""

    def __init__(self, model, feature_names: list):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)
        self._background_tsa_values = None

    def explain(self, X: np.ndarray) -> np.ndarray:
        """Return TSA values for the given input array."""
        sv = self.explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]
        return sv

    def explain_single(self, x: np.ndarray) -> dict:
        """Return per-feature TSA attribution for one unit."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        sv = self.explain(x)[0]
        base = float(self.explainer.expected_value)
        if isinstance(base, (list, np.ndarray)):
            base = base[-1]
        return {
            "base_value": base,
            "tsa_values": dict(zip(self.feature_names, sv.tolist())),
            "total_contribution": float(sv.sum()),
        }

    def waterfall_figure(self, x: np.ndarray, max_display: int = 15) -> bytes:
        """Render a TSA waterfall plot for one unit. Returns PNG bytes."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        sv = self.explainer(x)
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.plots.waterfall(sv[0], max_display=max_display, show=False)
        _relabel_shap_to_tsa(plt.gcf())
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        plt.close("all")
        buf.seek(0)
        return buf.read()

    def beeswarm_figure(self, X: np.ndarray, max_display: int = 20) -> bytes:
        """Render a TSA beeswarm summary plot. Returns PNG bytes."""
        sv = self.explain(X)
        fig, ax = plt.subplots(figsize=(11, 8))
        shap.summary_plot(sv, X, feature_names=self.feature_names,
                          max_display=max_display, show=False)
        _relabel_shap_to_tsa(plt.gcf())
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        plt.close("all")
        buf.seek(0)
        return buf.read()

    def bar_figure(self, X: np.ndarray, max_display: int = 20) -> bytes:
        """Render a TSA mean-absolute bar chart. Returns PNG bytes."""
        sv = self.explain(X)
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(sv, X, feature_names=self.feature_names,
                          plot_type="bar", max_display=max_display, show=False)
        _relabel_shap_to_tsa(plt.gcf())
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        plt.close("all")
        buf.seek(0)
        return buf.read()

    def top_features(self, x: np.ndarray, n: int = 10) -> pd.DataFrame:
        """Return top-N features by |TSA| magnitude for one unit."""
        exp = self.explain_single(x)
        sv = exp["tsa_values"]
        df = pd.DataFrame({
            "feature": list(sv.keys()),
            "tsa_value": list(sv.values()),
        })
        df["abs_tsa"] = df["tsa_value"].abs()
        df = df.sort_values("abs_tsa", ascending=False).head(n).reset_index(drop=True)
        df["direction"] = df["tsa_value"].apply(lambda v: "↑ increases risk" if v > 0 else "↓ decreases risk")
        return df


# Backward-compatible alias so any joblib-deserialised SHAPEngine objects still load
SHAPEngine = TSAEngine
