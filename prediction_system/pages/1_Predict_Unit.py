"""
Single-unit failure prediction with SHAP explanation and
engineer-readable risk report.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(page_title="Predict Unit", page_icon="🔮", layout="wide")
st.title("HBM Unit Risk Prediction")
st.caption("Enter HBM packaging process parameters for one unit and receive an instant risk score with full Trajectory Shapley Attribution.")


@st.cache_resource(show_spinner="Loading model...")
def get_artifacts():
    from src.model.pipeline import load_artifacts, train_pipeline
    arts = load_artifacts()
    if arts is None:
        arts = train_pipeline(n_samples=3000)
        arts = load_artifacts()
    return arts


def sample_unit(risk: str = "high") -> dict:
    """Return a pre-filled sample unit for demo purposes."""
    if risk == "high":
        return {
            # Die Attach — under-cured, high moisture exposure
            "die_attach_temp_c": 163.0,
            "die_attach_dwell_time_s": 1.8,
            "die_attach_force_n": 95.0,
            "epoxy_dispense_volume_ul": 5.0,
            "epoxy_pot_life_pct": 32.0,
            "die_attach_void_pct": 11.5,
            # TCB Bonding — low force + high temp → microbump cracking risk
            "tcb_bond_force_n": 215.0,
            "tcb_bond_temp_c": 271.0,
            "tcb_bond_time_s": 5.2,
            "tcb_misalignment_um": 3.4,
            "microbump_contact_res_mohm": 48.0,
            "tsv_chain_res_ohm": 1.7,
            # Encapsulation — under-cured with high moisture
            "mold_temp_c": 188.0,
            "mold_pressure_bar": 70.0,
            "mold_cure_time_s": 62.0,
            "filler_content_pct": 75.0,
            "moisture_content_ppm": 580.0,
            "gel_time_s": 38.0,
            # Reflow
            "reflow_peak_temp_c": 258.0,
            "reflow_zone4_temp_c": 245.0,
            "reflow_zone5_temp_c": 252.0,
            "reflow_dwell_time_above_liquidus_s": 45.0,
            "reflow_ramp_rate_c_per_s": 2.0,
            "reflow_cool_rate_c_per_s": 3.0,
            "solder_paste_volume_pct": 100.0,
            # Environment — high humidity, long floor life
            "clean_room_humidity_pct": 64.0,
            "clean_room_temp_c": 22.0,
            "time_to_reflow_min": 210.0,
            "floor_life_hours": 82.0,
            # Material
            "substrate_thickness_mm": 0.5,
            "die_thickness_um": 200.0,
            "underfill_viscosity_cps": 2500.0,
            "pcb_warpage_um": 50.0,
            "thermal_resistance_cjb": 8.0,
            # Lineage — edge die, PM overdue, high lot moisture
            "wafer_id": 12,
            "die_position_x": 1,
            "die_position_y": 2,
            "operator_shift": 2,
            "days_since_last_pm": 28.0,
            "lot_batch_position": 3,
            "wafer_edge_flag": 1,
            "cumulative_tool_cycles": 38000,
            "lot_mean_tcb_force": 218.0,
            "lot_std_tcb_force": 28.0,
            "lot_mean_attach_temp": 161.0,
            "lot_p90_moisture": 540.0,
            "lot_fail_rate_rolling": 0.06,
        }
    else:
        return {
            # Die Attach — nominal
            "die_attach_temp_c": 150.0,
            "die_attach_dwell_time_s": 3.0,
            "die_attach_force_n": 120.0,
            "epoxy_dispense_volume_ul": 5.0,
            "epoxy_pot_life_pct": 70.0,
            "die_attach_void_pct": 2.5,
            # TCB Bonding — nominal
            "tcb_bond_force_n": 302.0,
            "tcb_bond_temp_c": 251.0,
            "tcb_bond_time_s": 8.1,
            "tcb_misalignment_um": 0.9,
            "microbump_contact_res_mohm": 19.5,
            "tsv_chain_res_ohm": 0.81,
            # Encapsulation — nominal
            "mold_temp_c": 175.0,
            "mold_pressure_bar": 72.0,
            "mold_cure_time_s": 92.0,
            "filler_content_pct": 76.0,
            "moisture_content_ppm": 180.0,
            "gel_time_s": 24.0,
            # Reflow — nominal
            "reflow_peak_temp_c": 244.0,
            "reflow_zone4_temp_c": 229.0,
            "reflow_zone5_temp_c": 239.0,
            "reflow_dwell_time_above_liquidus_s": 44.0,
            "reflow_ramp_rate_c_per_s": 2.0,
            "reflow_cool_rate_c_per_s": 3.0,
            "solder_paste_volume_pct": 101.0,
            # Environment — nominal
            "clean_room_humidity_pct": 44.0,
            "clean_room_temp_c": 22.0,
            "time_to_reflow_min": 55.0,
            "floor_life_hours": 22.0,
            # Material — nominal
            "substrate_thickness_mm": 0.5,
            "die_thickness_um": 202.0,
            "underfill_viscosity_cps": 2490.0,
            "pcb_warpage_um": 48.0,
            "thermal_resistance_cjb": 8.0,
            # Lineage — centre die, recent PM, low lot moisture
            "wafer_id": 7,
            "die_position_x": 10,
            "die_position_y": 10,
            "operator_shift": 1,
            "days_since_last_pm": 8.0,
            "lot_batch_position": 12,
            "wafer_edge_flag": 0,
            "cumulative_tool_cycles": 12000,
            "lot_mean_tcb_force": 299.0,
            "lot_std_tcb_force": 8.0,
            "lot_mean_attach_temp": 150.2,
            "lot_p90_moisture": 195.0,
            "lot_fail_rate_rolling": 0.015,
        }


arts = get_artifacts()
_OLD_COLS = {"bond_force_gf", "wire_pull_strength_gf", "ball_shear_gf"}
_stale = arts is not None and (
    any(c in arts.get("feature_cols", []) for c in _OLD_COLS)
    or "tsa_engine" not in arts
)
if _stale:
    st.cache_resource.clear()
    st.rerun()

# ── Sidebar: load sample ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Quick Load")
    if st.button("Load HIGH-RISK sample unit", use_container_width=True):
        st.session_state["unit_data"] = sample_unit("high")
    if st.button("Load LOW-RISK sample unit", use_container_width=True):
        st.session_state["unit_data"] = sample_unit("low")
    st.markdown("---")
    st.caption("Or fill in parameters manually below.")

defaults = st.session_state.get("unit_data", sample_unit("high"))

# ── Parameter Input Form ──────────────────────────────────────────────────────
with st.expander("Process Parameters (expand to edit)", expanded=True):
    tabs = st.tabs(["Die Attach", "TCB Bonding", "Encapsulation", "Reflow", "Environment", "Material", "Lineage"])

    with tabs[0]:  # DIE ATTACH
        c1, c2, c3 = st.columns(3)
        die_attach_temp = c1.number_input("Die Attach Temp (°C)", 130.0, 185.0, float(defaults.get("die_attach_temp_c", 150)), step=0.5)
        die_attach_dwell = c2.number_input("Dwell Time (s)", 1.0, 6.0, float(defaults.get("die_attach_dwell_time_s", 3.0)), step=0.1)
        die_attach_force = c3.number_input("Bond Force (N)", 60.0, 180.0, float(defaults.get("die_attach_force_n", 120)), step=1.0)
        c4, c5, c6 = st.columns(3)
        epoxy_vol = c4.number_input("Epoxy Volume (μL)", 2.0, 9.0, float(defaults.get("epoxy_dispense_volume_ul", 5.0)), step=0.1)
        epoxy_pot = c5.number_input("Epoxy Pot Life (%)", 10.0, 100.0, float(defaults.get("epoxy_pot_life_pct", 60)), step=1.0)
        void_pct = c6.number_input("Void % in Die Attach", 0.0, 25.0, float(defaults.get("die_attach_void_pct", 3.0)), step=0.1)

    with tabs[1]:  # TCB BONDING — Thermal Compression Bonding (HBM stacking interconnect)
        st.caption("HBM3e/HBM4 uses microbumps at sub-40 µm pitch bonded via TCB. Deviations here drive Microbump/TSV Cracking.")
        c1, c2, c3 = st.columns(3)
        tcb_force   = c1.number_input("TCB Bond Force (N)", 150.0, 500.0, float(defaults.get("tcb_bond_force_n", 300)), step=5.0)
        tcb_temp    = c2.number_input("TCB Bond Temperature (°C)", 220.0, 290.0, float(defaults.get("tcb_bond_temp_c", 250)), step=1.0)
        tcb_time    = c3.number_input("TCB Bond Time (s)", 3.0, 15.0, float(defaults.get("tcb_bond_time_s", 8.0)), step=0.5)
        c4, c5, c6 = st.columns(3)
        tcb_mis     = c4.number_input("Die Misalignment (μm)", 0.0, 5.0, float(defaults.get("tcb_misalignment_um", 1.0)), step=0.1)
        mb_res      = c5.number_input("Microbump Contact Resistance (mΩ)", 5.0, 80.0, float(defaults.get("microbump_contact_res_mohm", 20.0)), step=0.5)
        tsv_res     = c6.number_input("TSV Chain Resistance (Ω)", 0.3, 3.0, float(defaults.get("tsv_chain_res_ohm", 0.8)), step=0.05)

    with tabs[2]:  # ENCAPSULATION
        c1, c2, c3 = st.columns(3)
        mold_temp = c1.number_input("Mold Temp (°C)", 155.0, 200.0, float(defaults.get("mold_temp_c", 175)), step=1.0)
        mold_pres = c2.number_input("Mold Pressure (bar)", 30.0, 120.0, float(defaults.get("mold_pressure_bar", 70)), step=1.0)
        cure_time = c3.number_input("Cure Time (s)", 30.0, 180.0, float(defaults.get("mold_cure_time_s", 90)), step=5.0)
        c4, c5, c6 = st.columns(3)
        filler = c4.number_input("Filler Content (%)", 55.0, 90.0, float(defaults.get("filler_content_pct", 75)), step=1.0)
        moisture = c5.number_input("Moisture Content (ppm)", 50.0, 1200.0, float(defaults.get("moisture_content_ppm", 200)), step=10.0)
        gel_t = c6.number_input("Gel Time (s)", 10.0, 60.0, float(defaults.get("gel_time_s", 25)), step=1.0)

    with tabs[3]:  # REFLOW
        c1, c2, c3 = st.columns(3)
        r_peak = c1.number_input("Peak Temp (°C)", 220.0, 275.0, float(defaults.get("reflow_peak_temp_c", 245)), step=1.0)
        r_z4 = c2.number_input("Zone 4 Temp (°C)", 200.0, 265.0, float(defaults.get("reflow_zone4_temp_c", 230)), step=1.0)
        r_z5 = c3.number_input("Zone 5 Temp (°C)", 210.0, 270.0, float(defaults.get("reflow_zone5_temp_c", 240)), step=1.0)
        c4, c5, c6 = st.columns(3)
        r_dwell = c4.number_input("Dwell Above Liquidus (s)", 20.0, 120.0, float(defaults.get("reflow_dwell_time_above_liquidus_s", 45)), step=1.0)
        r_ramp = c5.number_input("Ramp Rate (°C/s)", 0.5, 4.0, float(defaults.get("reflow_ramp_rate_c_per_s", 2.0)), step=0.1)
        r_cool = c6.number_input("Cool Rate (°C/s)", 1.0, 8.0, float(defaults.get("reflow_cool_rate_c_per_s", 3.0)), step=0.1)
        solder_vol = st.number_input("Solder Paste Volume (%)", 50.0, 150.0, float(defaults.get("solder_paste_volume_pct", 100)), step=1.0)

    with tabs[4]:  # ENVIRONMENT
        c1, c2 = st.columns(2)
        humidity = c1.number_input("Cleanroom Humidity (%RH)", 20.0, 75.0, float(defaults.get("clean_room_humidity_pct", 45)), step=0.5)
        cr_temp = c2.number_input("Cleanroom Temp (°C)", 18.0, 28.0, float(defaults.get("clean_room_temp_c", 22)), step=0.1)
        c3, c4 = st.columns(2)
        t2r = c3.number_input("Time to Reflow (min)", 10.0, 480.0, float(defaults.get("time_to_reflow_min", 60)), step=5.0)
        floor_life = c4.number_input("Floor Life Elapsed (hrs)", 0.0, 168.0, float(defaults.get("floor_life_hours", 24)), step=1.0)

    with tabs[5]:  # MATERIAL
        c1, c2, c3 = st.columns(3)
        sub_thick = c1.number_input("Substrate Thickness (mm)", 0.2, 0.8, float(defaults.get("substrate_thickness_mm", 0.5)), step=0.01)
        die_thick = c2.number_input("Die Thickness (μm)", 80.0, 300.0, float(defaults.get("die_thickness_um", 200)), step=5.0)
        underfill = c3.number_input("Underfill Viscosity (cPs)", 500.0, 6000.0, float(defaults.get("underfill_viscosity_cps", 2500)), step=50.0)
        c4, c5 = st.columns(2)
        pcb_warp = c4.number_input("PCB Warpage (μm)", 0.0, 300.0, float(defaults.get("pcb_warpage_um", 50)), step=1.0)
        therm_res = c5.number_input("Thermal Resistance Cjb", 3.0, 20.0, float(defaults.get("thermal_resistance_cjb", 8.0)), step=0.1)

    with tabs[6]:  # LINEAGE
        c1, c2, c3 = st.columns(3)
        wafer_id = c1.number_input("Wafer ID", 1, 25, int(defaults.get("wafer_id", 7)))
        die_x = c2.number_input("Die Position X", 0, 19, int(defaults.get("die_position_x", 10)))
        die_y = c3.number_input("Die Position Y", 0, 19, int(defaults.get("die_position_y", 10)))
        c4, c5, c6 = st.columns(3)
        op_shift = c4.selectbox("Operator Shift", [1, 2, 3], index=int(defaults.get("operator_shift", 1)) - 1)
        days_pm = c5.number_input("Days Since Last PM", 0.0, 90.0, float(defaults.get("days_since_last_pm", 15)), step=0.5)
        lot_pos = c6.number_input("Lot Batch Position", 1, 25, int(defaults.get("lot_batch_position", 12)))
        c7, c8 = st.columns(2)
        edge_flag = c7.selectbox("Wafer Edge Die?", [0, 1], index=int(defaults.get("wafer_edge_flag", 0)),
                                  format_func=lambda x: "Yes (edge)" if x else "No (centre)")
        tool_cycles = c8.number_input("Cumulative Tool Cycles", 1000, 50000,
                                       int(defaults.get("cumulative_tool_cycles", 12000)), step=500)
        c9, c10 = st.columns(2)
        lot_mean_bf = c9.number_input("Lot Mean TCB Force (N)", 150.0, 500.0,
                                       float(defaults.get("lot_mean_tcb_force", 300.0)), step=1.0)
        lot_std_bf = c10.number_input("Lot Std TCB Force (N)", 0.0, 60.0,
                                       float(defaults.get("lot_std_tcb_force", 10.0)), step=1.0)
        c11, c12 = st.columns(2)
        lot_mean_at = c11.number_input("Lot Mean Attach Temp (°C)", 130.0, 185.0,
                                        float(defaults.get("lot_mean_attach_temp", 150.0)), step=0.5)
        lot_p90m = c12.number_input("Lot P90 Moisture (ppm)", 50.0, 1200.0,
                                     float(defaults.get("lot_p90_moisture", 200.0)), step=10.0)
        lot_fail_rate = st.slider("Lot Rolling Failure Rate", 0.0, 0.15,
                                   float(defaults.get("lot_fail_rate_rolling", 0.02)), step=0.005,
                                   format="%.3f")

# ── Run prediction ────────────────────────────────────────────────────────────
unit_data = {
    "die_attach_temp_c": die_attach_temp,
    "die_attach_dwell_time_s": die_attach_dwell,
    "die_attach_force_n": die_attach_force,
    "epoxy_dispense_volume_ul": epoxy_vol,
    "epoxy_pot_life_pct": epoxy_pot,
    "die_attach_void_pct": void_pct,
    "tcb_bond_force_n": tcb_force,
    "tcb_bond_temp_c": tcb_temp,
    "tcb_bond_time_s": tcb_time,
    "tcb_misalignment_um": tcb_mis,
    "microbump_contact_res_mohm": mb_res,
    "tsv_chain_res_ohm": tsv_res,
    "mold_temp_c": mold_temp,
    "mold_pressure_bar": mold_pres,
    "mold_cure_time_s": cure_time,
    "filler_content_pct": filler,
    "moisture_content_ppm": moisture,
    "gel_time_s": gel_t,
    "reflow_peak_temp_c": r_peak,
    "reflow_zone4_temp_c": r_z4,
    "reflow_zone5_temp_c": r_z5,
    "reflow_dwell_time_above_liquidus_s": r_dwell,
    "reflow_ramp_rate_c_per_s": r_ramp,
    "reflow_cool_rate_c_per_s": r_cool,
    "solder_paste_volume_pct": solder_vol,
    "clean_room_humidity_pct": humidity,
    "clean_room_temp_c": cr_temp,
    "time_to_reflow_min": t2r,
    "floor_life_hours": floor_life,
    "substrate_thickness_mm": sub_thick,
    "die_thickness_um": die_thick,
    "underfill_viscosity_cps": underfill,
    "pcb_warpage_um": pcb_warp,
    "thermal_resistance_cjb": therm_res,
    "wafer_id": wafer_id,
    "die_position_x": die_x,
    "die_position_y": die_y,
    "operator_shift": op_shift,
    "days_since_last_pm": days_pm,
    "lot_batch_position": lot_pos,
    "wafer_edge_flag": edge_flag,
    "cumulative_tool_cycles": tool_cycles,
    "lot_mean_tcb_force": lot_mean_bf,
    "lot_std_tcb_force": lot_std_bf,
    "lot_mean_attach_temp": lot_mean_at,
    "lot_p90_moisture": lot_p90m,
    "lot_fail_rate_rolling": lot_fail_rate,
    "lot_id": "LOT-MANUAL",
    "tool_chamber_id": "CH-A",
}

predict_col, _ = st.columns([1, 3])
run = predict_col.button("▶ Run Prediction", type="primary", use_container_width=True)

if run:
    with st.spinner("Computing risk score and Trajectory Shapley Attribution..."):
        from src.model.pipeline import predict_unit
        result = predict_unit(arts, unit_data)

    st.divider()
    exp = result["explanation"]
    risk_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[exp["risk_level"]]

    # ── Risk headline ─────────────────────────────────────────────────────────
    if exp["risk_level"] == "HIGH":
        st.error(f"{risk_color} {exp['headline']}")
    elif exp["risk_level"] == "MEDIUM":
        st.warning(f"{risk_color} {exp['headline']}")
    else:
        st.success(f"{risk_color} {exp['headline']}")

    col_gauge, col_details = st.columns([1, 2])

    with col_gauge:
        import plotly.graph_objects as go
        score = result["risk_score"]
        color = "#e74c3c" if score >= 0.6 else "#f39c12" if score >= 0.3 else "#27ae60"
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score * 100,
            number={"suffix": "%", "font": {"size": 40}},
            delta={"reference": arts["optimal_threshold"] * 100,
                   "increasing": {"color": "#e74c3c"},
                   "decreasing": {"color": "#27ae60"},
                   "suffix": "% vs threshold"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 30], "color": "#d5f5e3"},
                    {"range": [30, 60], "color": "#fdebd0"},
                    {"range": [60, 100], "color": "#fadbd8"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "value": arts["optimal_threshold"] * 100,
                },
            },
            title={"text": f"Failure Probability<br><sub>Threshold = {arts['optimal_threshold']:.2f}</sub>"},
        ))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_details:
        st.subheader("Why this unit is flagged")
        for i, reason in enumerate(exp["reasons"], 1):
            st.markdown(f"{i}. {reason}")

    st.divider()

    # ── TSA waterfall ─────────────────────────────────────────────────────────
    st.subheader("Trajectory Shapley Attribution (Waterfall)")
    st.image(result["waterfall_bytes"], use_container_width=True)

    st.divider()

    # ── Top features bar chart ────────────────────────────────────────────────
    st.subheader("Top Risk Drivers")
    import plotly.express as px
    top_df = result["top_features"].head(10)
    top_df["color"] = top_df["tsa_value"].apply(lambda v: "Increases Risk" if v > 0 else "Decreases Risk")
    fig_bar = px.bar(
        top_df, x="tsa_value", y="feature", orientation="h",
        color="color",
        color_discrete_map={"Increases Risk": "#e74c3c", "Decreases Risk": "#27ae60"},
        labels={"tsa_value": "TSA Value", "feature": "Feature"},
        title="Trajectory Shapley Attribution — Top 10 Features",
    )
    fig_bar.update_layout(height=400, yaxis=dict(autorange="reversed"),
                          margin=dict(l=10, r=10, t=40, b=10), showlegend=True)
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Recommended actions ───────────────────────────────────────────────────
    st.subheader("Recommended Engineer Actions")
    for action in exp["recommended_actions"]:
        st.markdown(f"- {action}")
