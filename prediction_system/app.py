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
    st.title("Zero-Defect HBM Packaging — Early-Life Failure Prediction")
    st.caption(
        "Edge-Deployed XGBoost · SMOTE · Trajectory Shapley Attribution · Lineage Features · Concept Drift Monitoring  |  "
        "Addressing interfacial delamination & microbump/TSV cracking in HBM3e / HBM4 stacking"
    )

    arts = get_artifacts()

    # Self-healing: if the in-memory cache holds stale (wire-bond era) artifacts,
    # clear it and rerun so train_pipeline() executes with the current HBM schema.
    _OLD_COLS = {"bond_force_gf", "wire_pull_strength_gf", "ball_shear_gf"}
    _stale = arts is not None and (
        any(c in arts.get("feature_cols", []) for c in _OLD_COLS)
        or "tsa_engine" not in arts
    )
    if _stale and not st.session_state.get("_schema_cache_cleared"):
        st.session_state["_schema_cache_cleared"] = True
        st.cache_resource.clear()
        st.rerun()

    # ── KPI strip ────────────────────────────────────────────────────────────
    m = arts["eval_metrics"]
    cv = arts["cv_results"]
    fail_rate = arts["failure_rate"]
    threshold_result = arts["threshold_result"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Validation AUC-ROC", f"{m['roc_auc']:.3f}")
    c2.metric("Avg. Precision (AUCPR)", f"{m['avg_precision']:.3f}")
    recall_val = m['recall']
    recall_target = 0.90
    c3.metric(
        "Recall (Defect Detection)",
        f"{recall_val:.3f}",
        delta=f"{'✓' if recall_val >= recall_target else '↑'} Target ≥ {recall_target:.0%}",
        delta_color="normal" if recall_val >= recall_target else "inverse",
        help="KPI target: recall ≥ 90% on defective HBM units (Section 4.1.4)"
    )
    c4.metric("Optimal Threshold", f"{m['threshold']:.3f}",
              delta="vs 0.50 default", delta_color="off")
    c5.metric("Cost Reduction vs Default", f"{threshold_result['cost_reduction_pct']:.1f}%",
              help="Cost saving from cost-calibrated threshold vs 0.5 default (FN=$30k GPU, FP=re-inspection)")

    st.divider()

    # ── Two-column layout ────────────────────────────────────────────────────
    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("HBM Failure Mode Distribution in Training Data")
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
    st.subheader("System Architecture — Four Integrated Modules")
    st.info(
        """
        **Proactive Edge-AI Framework for Early-Life HBM Failure Prevention**
        *(Bridges the 6-week factory test → 5-year warranty gap)*

        1. **Data Aggregation & SMOTE Engine** — ingests raw TCB bonding telemetry, underfill cure parameters,
           and reflow profiles; engineers 17 lineage features (lot context, tool state, production sequence);
           applies SMOTE to overcome the <1% defect rate in fab data. *(Addresses: class imbalance gap)*

        2. **Edge AI Inference Node** — XGBoost classifier deployed locally on the equipment controller,
           eliminating cloud latency. Cost-sensitive training (FN penalty >> FP) to aggressively minimise
           missed defects. Target: inference latency < 100 ms per unit. *(Addresses: inference-latency gap)*

        3. **Trajectory Shapley Attribution Module** — TSA TreeExplainer with sequential process ordering and an
           engineer-readable translation layer mapping attributions to physics-of-failure explanations
           (delamination, microbump cracking, TSV fracture). *(Addresses: static-explainability gap)*

        4. **Adaptive Concept Drift Filter** — continuous PSI + KS-test monitoring; triggers model
           recalibration when equipment ageing, recipe revisions, or material lot variation shifts the
           process distribution. *(Addresses: concept-drift gap)*
        """
    )

    st.divider()
    st.subheader("KPI Targets (Section 4.1.4)")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Recall Target", "≥ 90%", help="Proportion of true HBM failures caught before shipping")
    k2.metric("FN Rate Target", "< 2%", help="False-negative rate — missed defects reaching the field")
    k3.metric("Inference Latency", "< 100 ms", help="Edge inference latency per unit at the bonding station")
    k4.metric("Field Returns Reduction", "≥ 40%", help="Reduction in early-life HBM field returns within 12 months")
    k5.metric("Engineer Adoption", "≥ 80%", help="Flagged cases actively reviewed by reliability engineers")

    st.caption("Navigate using the sidebar → to predict units, run batch analysis, or monitor drift.")


main()
