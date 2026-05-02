"""
Micron IC Packaging Failure Prediction System
Main Streamlit entry point — Dashboard Overview
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys, os

# Make src importable from the app root
sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="Micron Failure Prediction",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared state helpers ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Training model on first launch...")
def get_artifacts():
    from src.model.pipeline import load_artifacts, train_pipeline
    arts = load_artifacts()
    if arts is None:
        arts = train_pipeline(n_samples=3000)
        arts = load_artifacts()
    return arts


def risk_gauge(score: float) -> go.Figure:
    color = "#e74c3c" if score >= 0.6 else "#f39c12" if score >= 0.3 else "#27ae60"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        number={"suffix": "%", "font": {"size": 36}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 30], "color": "#d5f5e3"},
                {"range": [30, 60], "color": "#fdebd0"},
                {"range": [60, 100], "color": "#fadbd8"},
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "value": 50},
        },
        title={"text": "Failure Probability"},
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
    return fig


# ── Page content ─────────────────────────────────────────────────────────────

def main():
    st.title("🔬 Micron IC Packaging Failure Prediction System")
    st.caption("Explainable XGBoost · SMOTE · SHAP · Lineage Features · Concept Drift Monitoring")

    arts = get_artifacts()

    # ── KPI strip ────────────────────────────────────────────────────────────
    m = arts["eval_metrics"]
    cv = arts["cv_results"]
    fail_rate = arts["failure_rate"]
    threshold_result = arts["threshold_result"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Validation AUC-ROC", f"{m['roc_auc']:.3f}")
    c2.metric("Avg. Precision (AUCPR)", f"{m['avg_precision']:.3f}")
    c3.metric("Recall (Defect Detection)", f"{m['recall']:.3f}",
              help="Proportion of true failures caught by the model")
    c4.metric("Optimal Threshold", f"{m['threshold']:.3f}",
              delta=f"vs 0.50 default",
              delta_color="off")
    c5.metric("Cost Reduction vs Default", f"{threshold_result['cost_reduction_pct']:.1f}%",
              help="Expected cost saving from cost-calibrated threshold vs 0.5 default")

    st.divider()

    # ── Two-column layout ────────────────────────────────────────────────────
    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Failure Mode Distribution in Training Data")
        from src.data.dataset import load_dataset, FAILURE_MODES
        from src.features.lineage_engineering import engineer_lineage_features
        df = load_dataset(n_samples=3000, seed=42)
        mode_counts = df["failure_mode_name"].value_counts().reset_index()
        mode_counts.columns = ["Failure Mode", "Count"]
        fig_pie = px.pie(
            mode_counts[mode_counts["Failure Mode"] != "Pass"],
            names="Failure Mode", values="Count",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Defective Unit Breakdown by Failure Mode",
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(height=380, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    with right:
        st.subheader("Bathtub Curve — Where This System Intervenes")
        t = np.linspace(0, 10, 300)
        early = 2.5 * np.exp(-1.2 * t)
        useful = np.full_like(t, 0.3)
        wearout = 0.3 + 0.15 * np.exp(0.5 * (t - 8))
        hazard = early + useful + wearout - 0.25
        hazard = np.clip(hazard, 0.05, None)

        fig_bath = go.Figure()
        fig_bath.add_trace(go.Scatter(x=t, y=hazard, mode="lines",
                                      line=dict(color="#2c3e50", width=3),
                                      name="Hazard Rate"))
        # Shade early-life zone
        mask = t <= 2.5
        fig_bath.add_trace(go.Scatter(
            x=np.concatenate([t[mask], t[mask][::-1]]),
            y=np.concatenate([hazard[mask], np.full(mask.sum(), 0.05)]),
            fill="toself", fillcolor="rgba(231,76,60,0.18)",
            line=dict(color="rgba(0,0,0,0)"), name="Early-Life Zone",
        ))
        fig_bath.add_annotation(
            x=1.2, y=1.8, text="← This system<br>targets here",
            showarrow=True, arrowhead=2, ax=40, ay=-30,
            font=dict(color="#c0392b", size=12),
        )
        fig_bath.update_layout(
            height=380, xaxis_title="Time in Service",
            yaxis_title="Failure Rate (Hazard)",
            showlegend=False, margin=dict(l=10, r=10, t=30, b=30),
        )
        st.plotly_chart(fig_bath, use_container_width=True)

    st.divider()

    # ── Cross-validation summary ──────────────────────────────────────────────
    st.subheader("5-Fold Cross-Validation Results")
    cv_data = {
        "Metric": ["Precision", "Recall", "F1", "ROC-AUC", "Avg. Precision"],
        "Mean": [
            cv["test_precision"].mean(),
            cv["test_recall"].mean(),
            cv["test_f1"].mean(),
            cv["test_roc_auc"].mean(),
            cv["test_average_precision"].mean(),
        ],
        "Std": [
            cv["test_precision"].std(),
            cv["test_recall"].std(),
            cv["test_f1"].std(),
            cv["test_roc_auc"].std(),
            cv["test_average_precision"].std(),
        ],
    }
    cv_df = pd.DataFrame(cv_data)
    cv_df["Mean ± Std"] = cv_df.apply(
        lambda r: f"{r['Mean']:.3f} ± {r['Std']:.3f}", axis=1
    )

    fig_cv = go.Figure()
    for i, row in cv_df.iterrows():
        fig_cv.add_trace(go.Bar(
            x=[row["Metric"]], y=[row["Mean"]],
            error_y=dict(type="data", array=[row["Std"]], visible=True),
            name=row["Metric"],
            marker_color=px.colors.qualitative.Safe[i],
        ))
    fig_cv.update_layout(
        height=320, showlegend=False,
        yaxis=dict(range=[0, 1.1], title="Score"),
        margin=dict(l=10, r=10, t=20, b=20),
    )
    st.plotly_chart(fig_cv, use_container_width=True)

    st.divider()
    st.subheader("System Architecture")
    st.info(
        """
        **Four integrated components — each addressing a documented enterprise AI adoption barrier:**

        1. **Data Integration Layer** — ingests process data, engineers 17 lineage features capturing lot context,
           tool state, and production sequence. *(Addresses: data silos)*

        2. **Predictive Modelling Core** — XGBoost with cost-sensitive learning (FN penalty = 10×FP)
           and SMOTE oversampling for class imbalance. *(Addresses: class imbalance, 95% AI failure rate)*

        3. **Explainability Layer** — SHAP TreeExplainer with an engineer-readable translation layer
           that maps feature attributions to physics-of-failure explanations. *(Addresses: trust barrier)*

        4. **Decision Support Interface** — cost-calibrated threshold, phased deployment workflow,
           and concept drift monitoring for continuous reliability. *(Addresses: industry inertia, model decay)*
        """
    )

    st.caption("Navigate using the sidebar → to predict units, run batch analysis, or monitor drift.")


main()
