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
if _stale:
    st.cache_resource.clear()
    st.rerun()

tab_info, tab_retrain, tab_cost = st.tabs(["System Info", "Retrain Model", "Cost Parameters"])

with tab_info:
    st.subheader("Current Model Metadata")
    import pandas as pd
    recall_ok = arts["eval_metrics"]["recall"] >= arts.get("recall_target", 0.90)
    info = {
        "Training samples": arts["n_samples"],
        "Feature count": arts["n_features"],
        "Target production failure rate": f"{arts.get('target_failure_rate', 0.04)*100:.1f}%",
        "Actual training failure rate": f"{arts['failure_rate']*100:.2f}%",
        "FN:FP training weight ratio": f"{arts.get('fn_train_weight', 10.0):.0f}:1",
        "Recall target": f"{arts.get('recall_target', 0.90)*100:.0f}%",
        "Recall target met": f"{'YES' if recall_ok else 'NO — consider raising FN weight'}",
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

    st.markdown("##### Dataset")
    c1, c2 = st.columns(2)
    n_samples = c1.number_input("Training samples", 500, 10000, 3000, step=500)
    smote_strat = c2.selectbox("Oversampling strategy", ["smote", "adasyn", "smotetomek"])

    st.markdown("##### Reliability Parameters")
    st.caption(
        "These control what the XGBoost model learns — they affect defect detection ability itself, "
        "independent of the cost-based threshold below."
    )
    r1, r2, r3 = st.columns(3)
    failure_rate_pct = r1.slider(
        "Expected production failure rate (%)",
        min_value=0.5, max_value=15.0,
        value=float(arts.get("target_failure_rate", 0.04)) * 100,
        step=0.5,
        help="Fraction of HBM units expected to fail in your fab. "
             "Sets the failure/pass balance in synthetic training data. "
             "Lower rate = harder learning problem → may need higher FN weight."
    )
    fn_weight = r2.slider(
        "FN:FP training weight ratio",
        min_value=2, max_value=30,
        value=int(arts.get("fn_train_weight", 10)),
        step=1,
        help="How aggressively XGBoost penalises missed failures DURING TRAINING "
             "(scale_pos_weight). Higher = more recall, less precision. "
             "This is separate from the cost-based threshold shift below."
    )
    recall_tgt_pct = r3.slider(
        "Minimum recall target (%)",
        min_value=70, max_value=99,
        value=int(arts.get("recall_target", 0.90) * 100),
        step=1,
        help="If the trained model's validation recall falls below this, "
             "a warning is shown. The system KPI target is 90%."
    )

    st.markdown("##### Cost-Based Threshold Calibration")
    st.caption(
        "These shift the decision threshold post-training to minimise expected financial cost. "
        "They do not change the model's underlying sensitivity — use the reliability parameters above for that."
    )
    c3, c4 = st.columns(2)
    cost_miss = c3.number_input("Missed defect cost ($)", 100, 5000, arts.get("cost_miss", 850), step=50)
    cost_fp_val = c4.number_input("False alarm cost ($)", 10, 500, arts.get("cost_fp", 45), step=5)

    if st.button("Retrain Model", type="primary"):
        with st.spinner("Retraining... this may take ~30 seconds"):
            from src.model.pipeline import train_pipeline
            st.cache_resource.clear()
            meta = train_pipeline(
                n_samples=n_samples,
                smote_strategy=smote_strat,
                failure_rate=failure_rate_pct / 100,
                fn_train_weight=float(fn_weight),
                recall_target=recall_tgt_pct / 100,
                cost_miss=cost_miss,
                cost_fp=cost_fp_val,
            )
        achieved_recall = meta["eval_metrics"]["recall"]
        if achieved_recall >= recall_tgt_pct / 100:
            st.success(
                f"Retraining complete! "
                f"AUC-ROC={meta['eval_metrics']['roc_auc']:.4f}, "
                f"Recall={achieved_recall:.4f} (target met), "
                f"Threshold={meta['optimal_threshold']:.3f}"
            )
        else:
            st.warning(
                f"Retraining complete but recall target not met: "
                f"achieved {achieved_recall:.4f} vs target {recall_tgt_pct/100:.2f}. "
                f"Try increasing the FN:FP training weight ratio."
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
