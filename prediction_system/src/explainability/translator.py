"""
Engineer-readable SHAP translation layer.
Converts raw SHAP attributions into natural-language explanations
grounded in process engineering terminology and physics-of-failure concepts.
"""

import numpy as np
import pandas as pd

# Maps feature names to domain-specific readable names and physical failure links
FEATURE_CONTEXT = {
    # DIE ATTACH
    "die_attach_temp_c": {
        "name": "Die Attach Temperature",
        "unit": "°C",
        "nominal": 150,
        "failure_link": "CTE mismatch stress at die-substrate interface",
        "high_risk": "Elevated temperature increases thermal stress, promoting interface delamination.",
        "low_risk": "Below-nominal temperature may result in incomplete epoxy cure and weak adhesion.",
    },
    "die_attach_dwell_time_s": {
        "name": "Die Attach Dwell Time",
        "unit": "s",
        "nominal": 3.0,
        "failure_link": "epoxy cure completeness",
        "high_risk": "Excess dwell time indicates slow curing material, possible moisture contamination.",
        "low_risk": "Insufficient dwell time → under-cured epoxy → delamination under thermal cycling.",
    },
    "die_attach_force_n": {
        "name": "Die Attach Bonding Force",
        "unit": "N",
        "nominal": 120,
        "failure_link": "die-substrate adhesion strength",
        "high_risk": "Excess force can crack thin dies or damage the substrate.",
        "low_risk": "Insufficient force → poor epoxy spread → void formation and delamination risk.",
    },
    "epoxy_dispense_volume_ul": {
        "name": "Epoxy Dispense Volume",
        "unit": "μL",
        "nominal": 5.0,
        "failure_link": "void formation and coverage uniformity",
        "high_risk": "Excess epoxy causes bleed-out and voids at periphery.",
        "low_risk": "Insufficient epoxy → incomplete coverage → delamination initiation sites.",
    },
    "epoxy_pot_life_pct": {
        "name": "Epoxy Pot Life Remaining",
        "unit": "%",
        "nominal": 60,
        "failure_link": "adhesive material age and viscosity",
        "high_risk": "High pot life pct is nominal. Low is the risk driver.",
        "low_risk": "Aged epoxy (low pot life %) has elevated viscosity → poor wetting → delamination.",
    },
    "die_attach_void_pct": {
        "name": "Die Attach Void Percentage",
        "unit": "%",
        "nominal": 3.0,
        "failure_link": "thermal conduction path and delamination initiation",
        "high_risk": "High void % degrades thermal path and creates stress concentrations → delamination.",
        "low_risk": None,
    },
    # TCB BONDING (Thermal Compression Bonding — the interconnect process for HBM stacking)
    # HBM uses microbumps at sub-40 µm pitch bonded via TCB; wire bonds are absent in 3D stacks.
    "tcb_bond_force_n": {
        "name": "TCB Bond Force",
        "unit": "N",
        "nominal": 300,
        "failure_link": "microbump compression and IMC (intermetallic compound) formation",
        "high_risk": "Excess TCB force can crack the microbump or fracture the thin DRAM die stack.",
        "low_risk": "Insufficient TCB force → incomplete microbump compression → weak IMC → microbump cracking under thermal cycling.",
    },
    "tcb_bond_temp_c": {
        "name": "TCB Bond Temperature",
        "unit": "°C",
        "nominal": 250,
        "failure_link": "thermosonic bonding energy and CTE-driven stress at microbump interface",
        "high_risk": "Excess temperature → thermal shock to microbumps and TSV stress → interfacial cracking.",
        "low_risk": "Insufficient temperature → incomplete IMC formation → cold joint → early microbump fracture.",
    },
    "tcb_bond_time_s": {
        "name": "TCB Bond Time",
        "unit": "s",
        "nominal": 8.0,
        "failure_link": "IMC growth completeness at microbump interface",
        "high_risk": "Excess dwell time accelerates IMC thickening → brittle Ni₃Sn₄ layer → crack initiation.",
        "low_risk": "Short bond time → insufficient IMC → weak microbump joint → cracking under heat flux.",
    },
    "tcb_misalignment_um": {
        "name": "TCB Die Misalignment",
        "unit": "μm",
        "nominal": 1.0,
        "failure_link": "lateral shear stress on microbumps at sub-40 µm pitch",
        "high_risk": "Misalignment beyond spec → lateral shear loading on microbumps → crack initiation at bump base.",
        "low_risk": None,
    },
    "microbump_contact_res_mohm": {
        "name": "Microbump Contact Resistance",
        "unit": "mΩ",
        "nominal": 20,
        "failure_link": "post-bond electrical integrity of microbump joint",
        "high_risk": "Elevated contact resistance signals incomplete bonding or partial delamination — direct indicator of future field failure.",
        "low_risk": None,
    },
    "tsv_chain_res_ohm": {
        "name": "TSV Chain Resistance",
        "unit": "Ω",
        "nominal": 0.8,
        "failure_link": "Through-Silicon Via structural integrity across the HBM stack",
        "high_risk": "Elevated TSV chain resistance indicates cracked or delaminated TSV — catastrophic failure risk under AI data-centre heat flux.",
        "low_risk": None,
    },
    # ENCAPSULATION
    "mold_temp_c": {
        "name": "Mold Temperature",
        "unit": "°C",
        "nominal": 175,
        "failure_link": "compound crosslink density and package stress",
        "high_risk": "High mold temp increases shrinkage stress → delamination at mold compound interface.",
        "low_risk": "Low mold temp → incomplete crosslinking → soft compound, poor protection.",
    },
    "mold_pressure_bar": {
        "name": "Transfer Mold Pressure",
        "unit": "bar",
        "nominal": 70,
        "failure_link": "void entrapment in mold compound",
        "high_risk": None,
        "low_risk": "Low pressure → incomplete cavity fill → voids and wire sweep → early failure.",
    },
    "mold_cure_time_s": {
        "name": "Mold Cure Time",
        "unit": "s",
        "nominal": 90,
        "failure_link": "compound cure completeness",
        "high_risk": None,
        "low_risk": "Under-cured compound has reduced adhesion strength → delamination initiation.",
    },
    "filler_content_pct": {
        "name": "Filler Content",
        "unit": "%",
        "nominal": 75,
        "failure_link": "compound CTE and moisture absorption",
        "high_risk": None,
        "low_risk": "Lower filler content → higher CTE → greater mismatch stress → cracking.",
    },
    "moisture_content_ppm": {
        "name": "Package Moisture Content",
        "unit": "ppm",
        "nominal": 200,
        "failure_link": "moisture-induced delamination (popcorn effect)",
        "high_risk": "CRITICAL: High moisture + reflow → steam pressure at interfaces → delamination crack.",
        "low_risk": None,
    },
    "gel_time_s": {
        "name": "Compound Gel Time",
        "unit": "s",
        "nominal": 25,
        "failure_link": "moisture trap probability in compound",
        "high_risk": "Slow gel time allows moisture ingress before cure → trapped moisture → delamination.",
        "low_risk": None,
    },
    # REFLOW
    "reflow_peak_temp_c": {
        "name": "Reflow Peak Temperature",
        "unit": "°C",
        "nominal": 245,
        "failure_link": "solder joint integrity and package thermal stress",
        "high_risk": "Above nominal → excess thermal stress → warpage, delamination, component damage.",
        "low_risk": "Below liquidus → incomplete solder melting → cold joints.",
    },
    "reflow_zone4_temp_c": {
        "name": "Reflow Zone 4 Temperature",
        "unit": "°C",
        "nominal": 230,
        "failure_link": "pre-peak thermal profile (moisture volatilisation zone)",
        "high_risk": "High Zone 4 temp accelerates moisture volatilisation → steam-driven delamination.",
        "low_risk": None,
    },
    "reflow_cool_rate_c_per_s": {
        "name": "Reflow Cooling Rate",
        "unit": "°C/s",
        "nominal": 3.0,
        "failure_link": "package warpage and solder grain structure",
        "high_risk": "Fast cooling → large thermal gradients → package warpage, BGA joint cracking.",
        "low_risk": "Slow cooling → coarse solder grain structure → fatigue-prone joints.",
    },
    "reflow_dwell_time_above_liquidus_s": {
        "name": "Dwell Time Above Liquidus",
        "unit": "s",
        "nominal": 45,
        "failure_link": "IMC thickness and solder joint reliability",
        "high_risk": "Excess time above liquidus → thick Ni3Sn4/Cu6Sn5 IMC → brittle joint → cracking.",
        "low_risk": None,
    },
    # ENVIRONMENT
    "clean_room_humidity_pct": {
        "name": "Cleanroom Relative Humidity",
        "unit": "%RH",
        "nominal": 45,
        "failure_link": "ambient moisture absorption by package materials",
        "high_risk": "High humidity → accelerated moisture uptake in mold compound → delamination risk.",
        "low_risk": None,
    },
    "floor_life_hours": {
        "name": "Floor Life Elapsed",
        "unit": "hours",
        "nominal": 24,
        "failure_link": "cumulative moisture exposure before reflow",
        "high_risk": "Extended floor life → moisture saturation of package → IPC J-STD-033 violation.",
        "low_risk": None,
    },
    "time_to_reflow_min": {
        "name": "Time-to-Reflow After Unpacking",
        "unit": "min",
        "nominal": 60,
        "failure_link": "moisture ingress window",
        "high_risk": "Longer exposure window → more moisture absorbed → delamination under reflow.",
        "low_risk": None,
    },
    # MATERIAL
    "die_thickness_um": {
        "name": "Die Thickness",
        "unit": "μm",
        "nominal": 200,
        "failure_link": "mechanical stiffness and warpage susceptibility",
        "high_risk": None,
        "low_risk": "Thin die → low flexural rigidity → warpage, die cracking under thermal cycling.",
    },
    "pcb_warpage_um": {
        "name": "PCB Warpage",
        "unit": "μm",
        "nominal": 50,
        "failure_link": "stand-off height non-uniformity → BGA joint stress",
        "high_risk": "High PCB warpage → non-uniform BGA joint formation → opens on edge joints.",
        "low_risk": None,
    },
    "substrate_thickness_mm": {
        "name": "Substrate Thickness",
        "unit": "mm",
        "nominal": 0.5,
        "failure_link": "structural rigidity and CTE mismatch compensation",
        "high_risk": None,
        "low_risk": "Thin substrate → lower rigidity → warpage, poor flatness for BGA mounting.",
    },
    # LINEAGE
    "wafer_edge_flag": {
        "name": "Wafer Edge Die",
        "unit": "boolean",
        "nominal": 0,
        "failure_link": "thermal gradient and process uniformity at wafer edge",
        "high_risk": "Edge dies experience larger process non-uniformity → elevated failure risk.",
        "low_risk": None,
    },
    "days_since_last_pm": {
        "name": "Days Since Tool PM",
        "unit": "days",
        "nominal": 15,
        "failure_link": "tool condition and process reproducibility",
        "high_risk": "Long interval since PM → tool drift → process parameter excursion.",
        "low_risk": None,
    },
    "lot_p90_moisture": {
        "name": "Lot P90 Moisture (Lineage)",
        "unit": "ppm",
        "nominal": 220,
        "failure_link": "lot-level moisture exposure profile",
        "high_risk": "High P90 moisture in lot signals systematic humidity control issue → delamination cluster.",
        "low_risk": None,
    },
    "humidity_x_floor_life": {
        "name": "Humidity × Floor Life Interaction",
        "unit": "compound",
        "nominal": None,
        "failure_link": "compound moisture absorption driver (interaction feature)",
        "high_risk": "High compound value: package absorbed substantial moisture before reflow → delamination.",
        "low_risk": None,
    },
    "moisture_x_cure_deficit": {
        "name": "Moisture × Cure Deficit (Lineage)",
        "unit": "compound",
        "nominal": None,
        "failure_link": "synergistic delamination risk: high moisture + under-cured compound",
        "high_risk": "Critical interaction: moisture trapped under under-cured compound → steam delamination.",
        "low_risk": None,
    },
    "tool_pm_urgency": {
        "name": "Tool PM Urgency Index",
        "unit": "ratio",
        "nominal": 0.5,
        "failure_link": "tool maintenance state",
        "high_risk": "PM overdue → degraded tool condition → process drift → quality excursion.",
        "low_risk": None,
    },
    "edge_distance": {
        "name": "Die Edge Distance from Wafer Centre",
        "unit": "mm",
        "nominal": None,
        "failure_link": "radial process gradient exposure",
        "high_risk": "Far from centre → larger thermal gradient → non-uniform deposition, more defects.",
        "low_risk": None,
    },
    "lot_std_tcb_force": {
        "name": "Lot TCB Bond Force Variability",
        "unit": "N σ",
        "nominal": None,
        "failure_link": "within-lot TCB process stability",
        "high_risk": "High within-lot force variability signals TCB bonder instability → some units have insufficient microbump compression.",
        "low_risk": None,
    },
    "moisture_z_in_lot": {
        "name": "Moisture Deviation Within Lot",
        "unit": "σ",
        "nominal": 0,
        "failure_link": "anomalous moisture exposure relative to lot peers",
        "high_risk": "This unit has significantly higher moisture than its lot peers → elevated delamination risk.",
        "low_risk": None,
    },
    "tcb_force_z_in_lot": {
        "name": "TCB Force Deviation Within Lot",
        "unit": "σ",
        "nominal": 0,
        "failure_link": "anomalous TCB bonding force relative to lot peers",
        "high_risk": "Outlier TCB force within lot → possible bonder transient → microbump compression uncertainty.",
        "low_risk": None,
    },
}


def _sigma_label(val: float, nominal: float) -> str:
    """Describe deviation from nominal in plain language."""
    if nominal is None or nominal == 0:
        return f"{val:.2f}"
    pct = (val - nominal) / abs(nominal) * 100
    sign = "above" if pct > 0 else "below"
    return f"{abs(pct):.1f}% {sign} nominal"


def translate_shap_to_engineer(
    top_features_df: pd.DataFrame,
    feature_values: dict,
    risk_score: float,
    n_reasons: int = 5,
) -> dict:
    """
    Convert a SHAP top-features table into an engineer-readable report.

    Parameters
    ----------
    top_features_df : DataFrame with columns [feature, tsa_value, direction]
    feature_values  : dict mapping feature name → actual observed value
    risk_score      : model probability score (0–1)
    n_reasons       : number of top reasons to include in narrative

    Returns
    -------
    dict with 'headline', 'risk_level', 'reasons', 'recommended_actions'
    """
    risk_pct = risk_score * 100
    if risk_pct >= 70:
        risk_level = "HIGH"
        color = "red"
    elif risk_pct >= 40:
        risk_level = "MEDIUM"
        color = "orange"
    else:
        risk_level = "LOW"
        color = "green"

    headline = (
        f"Unit flagged as {risk_level} RISK ({risk_pct:.1f}% failure probability). "
    )

    reasons = []
    actions = []

    top = top_features_df[top_features_df["tsa_value"] > 0].head(n_reasons)
    for _, row in top.iterrows():
        feat = row["feature"]
        sv = row["tsa_value"]
        ctx = FEATURE_CONTEXT.get(feat, {})
        name = ctx.get("name", feat.replace("_", " ").title())
        nominal = ctx.get("nominal")
        val = feature_values.get(feat, None)
        unit = ctx.get("unit", "")
        risk_text = ctx.get("high_risk") or ctx.get("low_risk") or ""

        if val is not None:
            if nominal is not None:
                val_str = f"{val:.2f} {unit} ({_sigma_label(val, nominal)})"
            else:
                val_str = f"{val:.2f} {unit}"
        else:
            val_str = "N/A"

        reason_sentence = (
            f"**{name}** = {val_str} contributes +{sv:.3f} to risk score. "
            f"{risk_text}"
        )
        reasons.append(reason_sentence)

        # Generate action recommendation
        if "moisture" in feat.lower() or "humidity" in feat.lower() or "floor_life" in feat.lower():
            actions.append("Re-dry unit per IPC J-STD-033 before reflow; verify bake-out protocol.")
        elif "cure" in feat.lower():
            actions.append("Review mold cure cycle — extend dwell time or increase temperature per material spec.")
        elif "tcb_bond_force" in feat.lower() or "tcb_bond_temp" in feat.lower() or "tcb_bond_time" in feat.lower():
            actions.append("Recalibrate TCB bonder — verify force profile, temperature ramp, and dwell time against recipe spec.")
        elif "tcb_misalign" in feat.lower():
            actions.append("Inspect TCB alignment optics and stage calibration; verify die-pick accuracy before next lot release.")
        elif "microbump_contact" in feat.lower() or "tsv_chain" in feat.lower():
            actions.append("Route unit for post-bond electrical characterisation; cross-check with SAM (Scanning Acoustic Microscopy) for delamination voids.")
        elif "void" in feat.lower():
            actions.append("Inspect die attach dispense equipment; validate volume calibration and needle condition.")
        elif "pm_urgency" in feat.lower() or "days_since" in feat.lower():
            actions.append("Schedule preventive maintenance on this tool before next lot release.")
        elif "warpage" in feat.lower() or "warp" in feat.lower():
            actions.append("Adjust reflow cooling rate profile; evaluate substrate thickness specification.")
        elif "reflow" in feat.lower():
            actions.append("Review reflow oven temperature profile — validate zone setpoints against recipe.")
        elif "edge" in feat.lower():
            actions.append("Apply enhanced inspection to wafer-edge dies; consider lot segregation for critical applications.")

    # Deduplicate actions
    seen = set()
    unique_actions = []
    for a in actions:
        if a not in seen:
            seen.add(a)
            unique_actions.append(a)

    return {
        "headline": headline,
        "risk_level": risk_level,
        "risk_color": color,
        "risk_pct": round(risk_pct, 1),
        "reasons": reasons,
        "recommended_actions": unique_actions if unique_actions else ["Route unit for enhanced electrical characterisation."],
    }
