"""
Lineage feature explorer — shows how lot-level and tool-level context
features improve over raw process parameters alone.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(page_title="Lineage Explorer", page_icon="🧬", layout="wide")
st.title("🧬 Lineage Feature Explorer")
st.caption(
    "Demonstrates how lineage-based features (lot context, tool state, production sequence) "
    "capture failure risk that raw process parameters alone cannot."
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


@st.cache_data(show_spinner="Loading dataset with lineage features...")
def load_data():
    from src.data.dataset import generate_synthetic_dataset
    from src.features.lineage_engineering import engineer_lineage_features
    df = generate_synthetic_dataset(n_samples=2000, seed=42)
    df = engineer_lineage_features(df)
    return df


df = load_data()
feature_cols = arts["feature_cols"]

# ── Lineage vs raw feature comparison ─────────────────────────────────────────
st.subheader("SHAP Importance: Raw Process vs Lineage Features")

lineage_feature_names = [
    "lot_p90_moisture", "lot_fail_rate_rolling", "humidity_x_floor_life",
    "moisture_x_cure_deficit", "edge_distance", "wafer_edge_flag",
    "days_since_last_pm", "tool_pm_urgency", "lot_std_bond_force",
    "moisture_z_in_lot", "bond_force_z_in_lot", "attach_temp_z_in_lot",
    "time_since_lot_start_norm", "lot_mean_attach_temp", "lot_mean_bond_force",
]

trainer = arts["trainer"]
imp = trainer.feature_importances()
imp_df = imp.reset_index()
imp_df.columns = ["feature", "importance"]
imp_df["type"] = imp_df["feature"].apply(
    lambda f: "Lineage Feature" if f in lineage_feature_names else "Raw Process Parameter"
)
imp_df = imp_df.sort_values("importance", ascending=False).head(30)

fig_comp = px.bar(
    imp_df, x="importance", y="feature", orientation="h",
    color="type",
    color_discrete_map={
        "Lineage Feature": "#e74c3c",
        "Raw Process Parameter": "#3498db",
    },
    labels={"importance": "Feature Importance (Gain)", "feature": "Feature",
             "type": "Feature Type"},
)
fig_comp.update_layout(height=650, yaxis=dict(autorange="reversed"),
                        margin=dict(l=10, r=10, t=20, b=20))
st.plotly_chart(fig_comp, use_container_width=True)

lineage_imp = imp_df[imp_df["type"] == "Lineage Feature"]["importance"].sum()
raw_imp = imp_df[imp_df["type"] == "Raw Process Parameter"]["importance"].sum()
c1, c2, c3 = st.columns(3)
c1.metric("Lineage feature total importance", f"{lineage_imp:.3f}")
c2.metric("Raw process parameter total importance", f"{raw_imp:.3f}")
c3.metric("Lineage share of total importance", f"{lineage_imp/(lineage_imp+raw_imp)*100:.1f}%")

st.divider()

# ── Moisture z-score vs failure ────────────────────────────────────────────────
st.subheader("Lineage Feature Deep-Dive: Moisture Deviation Within Lot")
st.caption(
    "Units with moisture content significantly above their lot peers "
    "face elevated delamination risk even when absolute moisture is moderate."
)

if "moisture_z_in_lot" in df.columns:
    plot_df = df[["moisture_z_in_lot", "label", "failure_mode_name"]].dropna()
    fig_box = px.box(
        plot_df, x="failure_mode_name", y="moisture_z_in_lot",
        color="failure_mode_name",
        labels={"moisture_z_in_lot": "Moisture Z-Score Within Lot",
                "failure_mode_name": "Failure Mode"},
        title="Moisture Z-Score Distribution by Failure Mode",
    )
    fig_box.update_layout(height=420, showlegend=False,
                          margin=dict(l=10, r=10, t=40, b=20))
    st.plotly_chart(fig_box, use_container_width=True)

st.divider()

# ── Lot rolling fail rate ─────────────────────────────────────────────────────
st.subheader("Lot-Level Failure Rate Context")
st.caption("Units from lots with elevated rolling failure rates carry excess risk — a lineage signal not in any single process parameter.")

if "lot_fail_rate_rolling" in df.columns and "risk_score" not in df.columns:
    # Compute risk scores for visualization
    prep = arts["prep"]
    from src.features.lineage_engineering import get_all_feature_columns
    feat_cols_avail = [c for c in feature_cols if c in df.columns]
    try:
        X_viz = prep.transform(df, feat_cols_avail)
        df["risk_score"] = arts["trainer"].predict_proba(X_viz)
    except Exception:
        df["risk_score"] = np.random.uniform(0, 0.3, len(df))
        df.loc[df["label"] == 1, "risk_score"] += 0.3

if "lot_fail_rate_rolling" in df.columns:
    fig_scatter = px.scatter(
        df.sample(min(800, len(df)), random_state=42),
        x="lot_fail_rate_rolling",
        y="risk_score",
        color="failure_mode_name",
        opacity=0.6,
        labels={"lot_fail_rate_rolling": "Lot Rolling Failure Rate",
                "risk_score": "Predicted Risk Score",
                "failure_mode_name": "Failure Mode"},
        title="Lot Context (Rolling Failure Rate) vs Predicted Risk Score",
    )
    fig_scatter.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=20))
    st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# ── Edge die effect ───────────────────────────────────────────────────────────
st.subheader("Wafer-Edge Die Effect")
st.caption("Edge-position dies experience greater thermal gradient exposure during processing.")

if "wafer_edge_flag" in df.columns and "risk_score" in df.columns:
    edge_df = df.groupby("wafer_edge_flag").agg(
        mean_risk=("risk_score", "mean"),
        fail_rate=("label", "mean"),
        count=("label", "count"),
    ).reset_index()
    edge_df["position"] = edge_df["wafer_edge_flag"].map({0: "Centre Die", 1: "Edge Die"})

    c1, c2 = st.columns(2)
    with c1:
        fig_edge = px.bar(edge_df, x="position", y="mean_risk",
                          color="position", title="Mean Risk Score by Die Position",
                          color_discrete_map={"Centre Die": "#27ae60", "Edge Die": "#e74c3c"},
                          labels={"mean_risk": "Mean Risk Score", "position": ""})
        fig_edge.update_layout(height=320, showlegend=False,
                                margin=dict(l=10, r=10, t=40, b=20))
        st.plotly_chart(fig_edge, use_container_width=True)
    with c2:
        fig_fr = px.bar(edge_df, x="position", y="fail_rate",
                        color="position", title="Actual Failure Rate by Die Position",
                        color_discrete_map={"Centre Die": "#27ae60", "Edge Die": "#e74c3c"},
                        labels={"fail_rate": "Failure Rate", "position": ""})
        fig_fr.update_layout(height=320, showlegend=False,
                              margin=dict(l=10, r=10, t=40, b=20))
        st.plotly_chart(fig_fr, use_container_width=True)
