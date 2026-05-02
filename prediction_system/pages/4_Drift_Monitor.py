"""
Concept drift monitoring dashboard.
Simulates production data batches with controlled drift and shows
PSI / KS-test alerts in real time.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(page_title="Drift Monitor", page_icon="🚨", layout="wide")
st.title("🚨 Concept Drift Monitor")
st.caption(
    "Tracks whether incoming manufacturing data has drifted from training distribution. "
    "Drift signals suggest model recalibration is needed."
)


@st.cache_resource(show_spinner="Loading model...")
def get_artifacts():
    from src.model.pipeline import load_artifacts, train_pipeline
    arts = load_artifacts()
    if arts is None:
        arts = train_pipeline(n_samples=3000)
        arts = load_artifacts()
    return arts


arts = get_artifacts()
drift_detector = arts["drift_detector"]
feature_cols = arts["feature_cols"]

# ── Drift simulation controls ─────────────────────────────────────────────────
with st.sidebar:
    st.header("Drift Simulation")
    drift_type = st.selectbox(
        "Drift scenario",
        ["No drift (stable process)", "Gradual moisture creep",
         "Sudden humidity spike", "Tool PM overdue drift",
         "Reflow temperature shift"],
    )
    drift_magnitude = st.slider("Drift magnitude", 0.0, 3.0, 1.0, step=0.1)
    n_batches = st.slider("Number of batches to simulate", 3, 12, 6)
    batch_size = st.slider("Units per batch", 50, 300, 100)

if st.button("▶ Simulate Production Batches", type="primary"):
    from src.data.dataset import generate_synthetic_dataset
    from src.features.lineage_engineering import engineer_lineage_features

    reports = []
    with st.spinner(f"Simulating {n_batches} production batches..."):
        for batch_idx in range(n_batches):
            df_batch = generate_synthetic_dataset(
                n_samples=batch_size, failure_rate=0.04 + batch_idx * 0.003,
                seed=200 + batch_idx,
            )
            df_batch = engineer_lineage_features(df_batch)

            # Inject drift based on scenario
            if drift_type == "Gradual moisture creep":
                shift = drift_magnitude * 50 * (batch_idx + 1)
                df_batch["moisture_content_ppm"] += shift
                df_batch["lot_p90_moisture"] += shift * 0.8

            elif drift_type == "Sudden humidity spike" and batch_idx >= n_batches // 2:
                df_batch["clean_room_humidity_pct"] += drift_magnitude * 12
                df_batch["floor_life_hours"] += drift_magnitude * 20

            elif drift_type == "Tool PM overdue drift":
                df_batch["days_since_last_pm"] += drift_magnitude * 8 * (batch_idx + 1)
                df_batch["tool_pm_urgency"] = (
                    df_batch["days_since_last_pm"] / 30
                ).clip(0, 2)

            elif drift_type == "Reflow temperature shift":
                shift = drift_magnitude * 4 * (batch_idx + 1)
                df_batch["reflow_peak_temp_c"] += shift
                df_batch["reflow_zone4_temp_c"] += shift * 0.9

            report = drift_detector.check(df_batch)
            reports.append(report)

    st.session_state["drift_reports"] = reports
    st.session_state["drift_type"] = drift_type

if "drift_reports" in st.session_state:
    reports = st.session_state["drift_reports"]
    hist_df = drift_detector.history_df().tail(len(reports))

    # ── Status banner ─────────────────────────────────────────────────────────
    latest = reports[-1]
    sev_color = {"stable": "success", "warning": "warning", "drift": "error"}[latest.severity]
    sev_icon = {"stable": "✅", "warning": "⚠️", "drift": "🔴"}[latest.severity]
    getattr(st, sev_color)(
        f"{sev_icon} Latest batch: **{latest.severity.upper()}** — "
        f"Overall PSI = {latest.overall_psi:.4f}, "
        f"{latest.n_drifted}/{latest.n_features_tested} features drifted"
    )

    # ── PSI trend ─────────────────────────────────────────────────────────────
    st.subheader("Population Stability Index (PSI) Over Time")
    st.caption("PSI < 0.10: stable | 0.10–0.25: monitor | > 0.25: drift detected")

    batch_labels = [f"Batch {i+1}" for i in range(len(reports))]
    psi_vals = [r.overall_psi for r in reports]
    severities = [r.severity for r in reports]
    bar_colors = [
        "#27ae60" if s == "stable" else "#f39c12" if s == "warning" else "#e74c3c"
        for s in severities
    ]

    fig_psi = go.Figure()
    fig_psi.add_trace(go.Bar(x=batch_labels, y=psi_vals,
                             marker_color=bar_colors, name="Overall PSI"))
    fig_psi.add_hline(y=0.10, line_dash="dot", line_color="#f39c12",
                      annotation_text="Monitor threshold (0.10)", annotation_position="right")
    fig_psi.add_hline(y=0.25, line_dash="dash", line_color="#e74c3c",
                      annotation_text="Drift threshold (0.25)", annotation_position="right")
    fig_psi.update_layout(height=360, yaxis_title="PSI",
                          margin=dict(l=10, r=10, t=20, b=20))
    st.plotly_chart(fig_psi, use_container_width=True)

    # ── Per-feature PSI heatmap ───────────────────────────────────────────────
    st.subheader("Per-Feature PSI Heatmap")
    all_psi = {f"Batch {i+1}": r.psi_scores for i, r in enumerate(reports)}
    psi_matrix = pd.DataFrame(all_psi).T.fillna(0)
    # Show only top-drifted features
    mean_psi = psi_matrix.mean(axis=0).sort_values(ascending=False)
    top_feats = mean_psi.head(20).index.tolist()
    psi_plot = psi_matrix[top_feats] if top_feats else psi_matrix

    fig_heat = px.imshow(
        psi_plot.T,
        color_continuous_scale=[[0, "#d5f5e3"], [0.4, "#fdebd0"], [1, "#fadbd8"]],
        zmin=0, zmax=0.5,
        labels={"color": "PSI"},
        aspect="auto",
    )
    fig_heat.update_layout(height=500, margin=dict(l=10, r=10, t=20, b=20))
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Drifted features list ─────────────────────────────────────────────────
    st.subheader("Drifted Features in Latest Batch")
    if latest.drifted_features:
        drift_detail = []
        for feat in latest.drifted_features:
            psi_val = latest.psi_scores.get(feat, 0)
            ks_p = latest.ks_pvalues.get(feat, 1)
            drift_detail.append({
                "Feature": feat.replace("_", " ").title(),
                "PSI": round(psi_val, 4),
                "KS p-value": round(ks_p, 4),
                "Status": "DRIFT" if psi_val > 0.25 else "MONITOR",
            })
        dd_df = pd.DataFrame(drift_detail).sort_values("PSI", ascending=False)

        def style_status(val):
            return "color: #e74c3c; font-weight: bold" if val == "DRIFT" else "color: #f39c12"

        st.dataframe(dd_df.style.applymap(style_status, subset=["Status"]),
                     use_container_width=True)
    else:
        st.success("No features showing significant drift in the latest batch.")

    # ── Recalibration recommendation ─────────────────────────────────────────
    st.divider()
    st.subheader("Recalibration Recommendation")
    if latest.severity == "drift":
        st.error(
            "**Model recalibration recommended.** "
            "Incoming data has drifted significantly from the training distribution. "
            "Collect fresh labelled data from the current process state and retrain. "
            "Continue using the current model with widened uncertainty bounds until recalibration is complete."
        )
    elif latest.severity == "warning":
        st.warning(
            "**Monitor closely.** "
            "Early drift signals detected. Schedule recalibration within the next production cycle. "
            "Increase inspection rate on flagged units until the model is updated."
        )
    else:
        st.success(
            "**Process stable.** "
            "Feature distributions are consistent with training data. No recalibration needed."
        )
