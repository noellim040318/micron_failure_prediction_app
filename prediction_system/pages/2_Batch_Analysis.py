"""
Batch prediction: upload a CSV of process data or use synthetic data.
Shows risk score distribution, top risky units, and lot-level heatmaps.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(page_title="Batch Analysis", page_icon="📊", layout="wide")
st.title("Batch Analysis")
st.caption("Score an entire production batch and identify the highest-risk units.")


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
if _stale and not st.session_state.get("_schema_cache_cleared"):
    st.session_state["_schema_cache_cleared"] = True
    st.cache_resource.clear()
    st.rerun()

# ── Data source selection ─────────────────────────────────────────────────────
st.subheader("Data Source")
source = st.radio("Choose input", ["Use synthetic batch (demo)", "Upload CSV"], horizontal=True)

if source == "Upload CSV":
    uploaded = st.file_uploader("Upload process data CSV", type="csv")
    if uploaded:
        df_input = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df_input)} units from upload.")
    else:
        st.info("Awaiting CSV upload. Using synthetic batch in the meantime.")
        source = "Use synthetic batch (demo)"

if source == "Use synthetic batch (demo)":
    n_batch = st.slider("Batch size (synthetic demo)", 100, 1000, 300, step=50)
    from src.data.dataset import generate_synthetic_dataset
    df_input = generate_synthetic_dataset(n_samples=n_batch, failure_rate=0.06, seed=99)
    st.info(f"Generated {len(df_input)} synthetic units (6% failure rate demo).")

# ── Run batch prediction ──────────────────────────────────────────────────────
if st.button("▶ Score Batch", type="primary"):
    with st.spinner("Scoring batch..."):
        from src.model.pipeline import predict_batch
        result_df = predict_batch(arts, df_input)
    st.session_state["batch_result"] = result_df
    st.success(f"Scored {len(result_df)} units.")

if "batch_result" in st.session_state:
    result_df = st.session_state["batch_result"]
    threshold = arts["optimal_threshold"]

    flagged = result_df[result_df["risk_score"] >= threshold]
    n_flagged = len(flagged)
    n_total = len(result_df)

    # ── KPI strip ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Units Scored", n_total)
    c2.metric("Units Flagged", n_flagged, f"{n_flagged/n_total*100:.1f}% of batch")
    c3.metric("High Risk (≥60%)", int((result_df["risk_score"] >= 0.6).sum()))
    c4.metric("Mean Risk Score", f"{result_df['risk_score'].mean():.3f}")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Risk Score Distribution")
        fig_hist = px.histogram(
            result_df, x="risk_score", nbins=40,
            color_discrete_sequence=["#3498db"],
            labels={"risk_score": "Failure Probability"},
        )
        fig_hist.add_vline(x=threshold, line_dash="dash", line_color="red",
                           annotation_text=f"Threshold={threshold:.2f}", annotation_position="top right")
        fig_hist.update_layout(height=350, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_right:
        st.subheader("Risk Level Breakdown")
        level_counts = result_df["risk_level"].value_counts()
        fig_pie = px.pie(
            values=level_counts.values, names=level_counts.index,
            color=level_counts.index,
            color_discrete_map={"High": "#e74c3c", "Medium": "#f39c12", "Low": "#27ae60"},
        )
        fig_pie.update_layout(height=350, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # ── Wafer map ─────────────────────────────────────────────────────────────
    if "die_position_x" in result_df.columns and "die_position_y" in result_df.columns:
        st.subheader("Wafer Risk Map")
        fig_wafer = px.scatter(
            result_df, x="die_position_x", y="die_position_y",
            color="risk_score",
            color_continuous_scale="RdYlGn_r",
            size="risk_score",
            size_max=20,
            hover_data=["unit_id", "risk_score", "risk_level"] if "unit_id" in result_df.columns else None,
            labels={"die_position_x": "Die X", "die_position_y": "Die Y",
                    "risk_score": "Risk Score"},
            title="Risk Score by Die Position",
        )
        fig_wafer.update_layout(height=450, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_wafer, use_container_width=True)

    st.divider()

    # ── Top 20 riskiest units ─────────────────────────────────────────────────
    st.subheader("Top 20 Highest-Risk Units")
    top20_cols = ["unit_id", "risk_score", "risk_level", "lot_id",
                  "moisture_content_ppm", "clean_room_humidity_pct",
                  "die_attach_void_pct", "mold_cure_time_s"]
    display_cols = [c for c in top20_cols if c in result_df.columns]

    top20 = result_df.nlargest(20, "risk_score")[display_cols].reset_index(drop=True)

    def colour_risk(val):
        if isinstance(val, float) and val >= 0.6:
            return "background-color: #fadbd8"
        elif isinstance(val, float) and val >= arts["optimal_threshold"]:
            return "background-color: #fdebd0"
        return ""

    st.dataframe(
        top20.style.format({"risk_score": "{:.3f}",
                             "moisture_content_ppm": "{:.0f}",
                             "clean_room_humidity_pct": "{:.1f}",
                             "die_attach_void_pct": "{:.1f}",
                             "mold_cure_time_s": "{:.0f}"}),
        use_container_width=True, height=380,
    )

    st.divider()

    # ── Download ──────────────────────────────────────────────────────────────
    csv = result_df[display_cols + ["prediction"]].to_csv(index=False)
    st.download_button(
        "⬇ Download Scored Batch CSV",
        data=csv,
        file_name="batch_risk_scores.csv",
        mime="text/csv",
        use_container_width=False,
    )
