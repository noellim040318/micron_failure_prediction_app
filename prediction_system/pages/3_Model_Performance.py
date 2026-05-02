"""
Model performance dashboard: ROC/PR curves, confusion matrix,
feature importance, Trajectory Shapley Attribution summary plots, and threshold calibration analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")
st.title("Model Performance & Explainability")


@st.cache_resource(show_spinner="Loading model...")
def get_artifacts():
    from src.model.pipeline import load_artifacts, train_pipeline
    arts = load_artifacts()
    if arts is None:
        arts = train_pipeline(n_samples=3000)
        arts = load_artifacts()
    return arts


arts = get_artifacts()
_OLD_COLS = {"bond_force_gf", "wire_pull_strength_gf", "ball_shear_gf"}
_stale = arts is not None and (
    any(c in arts.get("feature_cols", []) for c in _OLD_COLS)
    or "tsa_engine" not in arts
)
if _stale:
    st.cache_resource.clear()
    st.rerun()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_roc, tab_pr, tab_cm, tab_thresh, tab_imp, tab_shap = st.tabs([
    "ROC Curve", "PR Curve", "Confusion Matrix",
    "Threshold Calibration", "Feature Importance", "Trajectory Shapley Attribution"
])

y_val = arts["y_val"].values
val_proba = arts["val_proba"]
threshold = arts["optimal_threshold"]
trainer = arts["trainer"]
feature_cols = arts["feature_cols"]

# ── ROC Curve ─────────────────────────────────────────────────────────────────
with tab_roc:
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, roc_thresh = roc_curve(y_val, val_proba)
    roc_auc = auc(fpr, tpr)

    # Find point closest to optimal threshold
    dists = np.abs(roc_thresh - threshold)
    idx_t = dists.argmin()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                             name=f"XGBoost (AUC = {roc_auc:.3f})",
                             line=dict(color="#3498db", width=2.5)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             name="Random", line=dict(color="grey", dash="dash")))
    fig.add_trace(go.Scatter(
        x=[fpr[idx_t]], y=[tpr[idx_t]], mode="markers",
        name=f"Threshold={threshold:.3f}",
        marker=dict(color="red", size=12, symbol="star"),
    ))
    fig.update_layout(
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        height=480, legend=dict(x=0.6, y=0.1),
        margin=dict(l=10, r=10, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.metric("Validation AUC-ROC", f"{roc_auc:.4f}")

# ── PR Curve ──────────────────────────────────────────────────────────────────
with tab_pr:
    from sklearn.metrics import precision_recall_curve, average_precision_score
    prec, rec, pr_thresh = precision_recall_curve(y_val, val_proba)
    ap = average_precision_score(y_val, val_proba)

    dists = np.abs(pr_thresh - threshold)
    idx_t = dists.argmin()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                             name=f"XGBoost (AUCPR = {ap:.3f})",
                             line=dict(color="#e67e22", width=2.5)))
    fig.add_hline(y=y_val.mean(), line_dash="dash", line_color="grey",
                  annotation_text="Baseline (prevalence)", annotation_position="right")
    fig.add_trace(go.Scatter(
        x=[rec[idx_t]], y=[prec[idx_t]], mode="markers",
        name=f"Threshold={threshold:.3f}",
        marker=dict(color="red", size=12, symbol="star"),
    ))
    fig.update_layout(
        xaxis_title="Recall", yaxis_title="Precision",
        height=480, legend=dict(x=0.5, y=0.9),
        margin=dict(l=10, r=10, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.metric("Avg. Precision (AUCPR)", f"{ap:.4f}")

# ── Confusion Matrix ──────────────────────────────────────────────────────────
with tab_cm:
    preds = (val_proba >= threshold).astype(int)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_val, preds)
    labels = ["Pass (0)", "Fail (1)"]

    fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Blues",
        x=[f"Predicted {l}" for l in labels],
        y=[f"Actual {l}" for l in labels],
    )
    fig.update_layout(height=400, margin=dict(l=50, r=10, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    tn, fp, fn, tp = cm.ravel()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("True Positives (caught defects)", tp)
    c2.metric("False Negatives (missed defects)", fn,
              delta=f"−{fn} escapes", delta_color="inverse")
    c3.metric("False Positives (false alarms)", fp,
              delta=f"+{fp} re-tests", delta_color="inverse")
    c4.metric("True Negatives (correct passes)", tn)

# ── Threshold Calibration ─────────────────────────────────────────────────────
with tab_thresh:
    st.subheader("Cost-Aligned Threshold Calibration")
    threshold_result = arts["threshold_result"]
    curve_df = threshold_result["curve"]
    cost_miss = threshold_result["cost_miss"]
    cost_fp = threshold_result["cost_fp"]

    st.markdown(f"""
    **Cost assumptions:**
    - Missed defect (FN) cost: **${cost_miss:,.0f}** per unit (field return + logistics + warranty)
    - False alarm (FP) cost: **${cost_fp:,.0f}** per unit (unnecessary re-test / hold)

    **Result:** Optimal threshold = **{threshold:.3f}** (vs default 0.50),
    reducing expected screening cost by **{threshold_result['cost_reduction_pct']:.1f}%**
    """)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve_df["threshold"], y=curve_df["total_cost"],
                             mode="lines", name="Total Cost",
                             line=dict(color="#8e44ad", width=2.5)))
    fig.add_trace(go.Scatter(x=curve_df["threshold"], y=curve_df["cost_fn"],
                             mode="lines", name=f"FN Cost (×${cost_miss})",
                             line=dict(color="#e74c3c", width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=curve_df["threshold"], y=curve_df["cost_fp"],
                             mode="lines", name=f"FP Cost (×${cost_fp})",
                             line=dict(color="#3498db", width=1.5, dash="dot")))
    fig.add_vline(x=threshold, line_color="black", line_dash="dash",
                  annotation_text=f"Optimal={threshold:.3f}", annotation_position="top right")
    fig.add_vline(x=0.5, line_color="grey", line_dash="dot",
                  annotation_text="Default=0.50", annotation_position="top left")
    fig.update_layout(
        xaxis_title="Classification Threshold",
        yaxis_title="Expected Cost ($)",
        height=420, margin=dict(l=10, r=10, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    cols = st.columns(3)
    cols[0].metric("Optimal threshold", f"{threshold:.3f}")
    cols[1].metric("Cost at optimal", f"${threshold_result['optimal_cost']:,.0f}")
    cols[2].metric("Cost at default (0.50)", f"${threshold_result['default_cost']:,.0f}",
                   delta=f"−${threshold_result['default_cost']-threshold_result['optimal_cost']:,.0f}")

# ── Feature Importance ────────────────────────────────────────────────────────
with tab_imp:
    st.subheader("XGBoost Feature Importance (Gain)")
    imp = trainer.feature_importances().head(25)
    fig = px.bar(
        x=imp.values, y=imp.index, orientation="h",
        color=imp.values, color_continuous_scale="Oranges",
        labels={"x": "Importance (Gain)", "y": "Feature"},
    )
    fig.update_layout(height=600, yaxis=dict(autorange="reversed"),
                      coloraxis_showscale=False, margin=dict(l=10, r=10, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

# ── Trajectory Shapley Attribution Summary ────────────────────────────────────
with tab_shap:
    st.subheader("Trajectory Shapley Attribution Global Summary (Training Sample)")
    st.caption("Beeswarm plot — each dot is one unit. Red = high feature value, blue = low.")

    @st.cache_data(show_spinner="Computing Trajectory Shapley Attribution values...")
    def compute_shap_plot(seed: int = 42):
        from src.data.dataset import generate_synthetic_dataset
        from src.features.lineage_engineering import engineer_lineage_features
        tsa_engine = arts["tsa_engine"]
        prep = arts["prep"]
        df_s = generate_synthetic_dataset(n_samples=400, seed=seed)
        df_s = engineer_lineage_features(df_s)
        X_s = prep.transform(df_s, feature_cols)
        bar_bytes = tsa_engine.bar_figure(X_s, max_display=20)
        bee_bytes = tsa_engine.beeswarm_figure(X_s, max_display=20)
        return bar_bytes, bee_bytes

    bar_b, bee_b = compute_shap_plot()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Mean |TSA| (feature impact)**")
        st.image(bar_b, use_container_width=True)
    with c2:
        st.markdown("**Beeswarm (value × direction)**")
        st.image(bee_b, use_container_width=True)
