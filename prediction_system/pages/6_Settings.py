"""
Settings: retrain the model with custom parameters,
adjust cost weights, and view system metadata.
"""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")
st.title("Model Settings & Retraining")


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

tab_info, tab_retrain, tab_cost = st.tabs(["System Info", "Retrain Model", "Cost Parameters"])

with tab_info:
    st.subheader("Current Model Metadata")
    import pandas as pd
    info = {
        "Training samples": arts["n_samples"],
        "Feature count": arts["n_features"],
        "Training failure rate": f"{arts['failure_rate']*100:.2f}%",
        "Optimal threshold": f"{arts['optimal_threshold']:.3f}",
        "Missed defect cost (FN)": f"${arts['cost_miss']:,.0f}",
        "False alarm cost (FP)": f"${arts['cost_fp']:,.0f}",
        "Validation AUC-ROC": f"{arts['eval_metrics']['roc_auc']:.4f}",
        "Validation Recall": f"{arts['eval_metrics']['recall']:.4f}",
        "Validation Precision": f"{arts['eval_metrics']['precision']:.4f}",
        "Validation F1": f"{arts['eval_metrics']['f1']:.4f}",
    }
    info_df = pd.DataFrame(list(info.items()), columns=["Parameter", "Value"])
    st.dataframe(info_df, use_container_width=True, hide_index=True)

    st.subheader("Feature List")
    feat_df = pd.DataFrame({"Feature": arts["feature_cols"],
                             "Index": range(len(arts["feature_cols"]))})
    st.dataframe(feat_df, use_container_width=True, height=400, hide_index=True)

with tab_retrain:
    st.subheader("Retrain with Custom Parameters")
    st.warning("Retraining will overwrite the current model artefacts.")

    c1, c2 = st.columns(2)
    n_samples = c1.number_input("Training samples", 500, 10000, 3000, step=500)
    smote_strat = c2.selectbox("Oversampling strategy", ["smote", "adasyn", "smotetomek"])

    c3, c4 = st.columns(2)
    cost_miss = c3.number_input("Missed defect cost ($)", 100, 5000, 850, step=50)
    cost_fp_val = c4.number_input("False alarm cost ($)", 10, 500, 45, step=5)

    if st.button("🔄 Retrain Model", type="primary"):
        with st.spinner("Retraining... this may take ~30 seconds"):
            from src.model.pipeline import train_pipeline
            st.cache_resource.clear()
            meta = train_pipeline(
                n_samples=n_samples,
                smote_strategy=smote_strat,
                cost_miss=cost_miss,
                cost_fp=cost_fp_val,
            )
        st.success(
            f"Retraining complete! "
            f"AUC-ROC={meta['eval_metrics']['roc_auc']:.4f}, "
            f"Recall={meta['eval_metrics']['recall']:.4f}, "
            f"Threshold={meta['optimal_threshold']:.3f}"
        )
        st.info("Refresh the page or navigate to another tab to use the new model.")

with tab_cost:
    st.subheader("Cost Parameter Sensitivity Analysis")
    st.markdown("""
    The cost-aligned threshold is derived from the extreme financial asymmetry in HBM failure costs:

    | Error Type | HBM Business Impact |
    |---|---|
    | **False Negative** (missed defect) | A single undetected latent defect in an HBM stack forces replacement of the co-packaged AI GPU (>\$30,000). Each hour of unplanned AI data-centre downtime averages **\$540,000**. |
    | **False Positive** (false alarm) | Unnecessary re-inspection or localised rework — typically **\$45–\$500** per unit. |

    The model threshold is deliberately shifted below 0.50 to aggressively minimise false negatives,
    accepting a marginally higher false-positive rate as a financially optimal trade-off.

    **Analytical threshold (Bayes decision boundary):**
    """)
    import numpy as np
    from src.utils.threshold_calibration import theoretical_optimal_threshold
    import plotly.graph_objects as go

    prevalence = arts["failure_rate"]
    cm_vals = np.linspace(100, 3000, 50)
    cf_vals = np.linspace(10, 300, 50)
    thresholds = np.array([
        theoretical_optimal_threshold(cm, arts["cost_fp"], prevalence)
        for cm in cm_vals
    ])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cm_vals, y=thresholds, mode="lines",
                             line=dict(color="#8e44ad", width=2.5),
                             name="Optimal threshold vs missed-defect cost"))
    fig.add_vline(x=arts["cost_miss"], line_dash="dash", line_color="red",
                  annotation_text=f"Current=${arts['cost_miss']}", annotation_position="top right")
    fig.update_layout(
        xaxis_title="Missed Defect Cost ($)",
        yaxis_title="Optimal Threshold",
        height=380, margin=dict(l=10, r=10, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Higher missed-defect cost → lower threshold (more conservative — catch more defects at cost of more false alarms). "
        "Adjust cost parameters in the Retrain tab to update the threshold."
    )
