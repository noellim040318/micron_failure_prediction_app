"""
Lineage-based feature engineering.
Captures each unit's production context — lot statistics, tool-chamber
identity, time-since-PM, edge position — that raw process parameters alone
cannot represent.
"""

import numpy as np
import pandas as pd


# Human-readable description for each lineage feature (for SHAP translation)
LINEAGE_FEATURE_DESCRIPTIONS = {
    "wafer_edge_flag": "die is on wafer edge (higher stress zone)",
    "days_since_last_pm": "days since last tool preventive maintenance",
    "lot_mean_bond_force": "lot-average wire bond force",
    "lot_std_bond_force": "within-lot variability of wire bond force",
    "lot_mean_attach_temp": "lot-average die attach temperature",
    "lot_p90_moisture": "90th percentile moisture content for this lot",
    "lot_fail_rate_rolling": "rolling failure rate of preceding lots",
    "cumulative_tool_cycles": "cumulative tool operation cycles (tool age proxy)",
    "lot_batch_position": "unit position within production lot (first/last units more exposed)",
    "time_since_lot_start_norm": "normalised time since lot production start",
    "edge_distance": "Euclidean distance from wafer centre (thermal gradient exposure)",
    "bond_force_z_in_lot": "wire bond force deviation within lot (z-score)",
    "moisture_z_in_lot": "moisture content deviation within lot (z-score)",
    "attach_temp_z_in_lot": "die attach temperature deviation within lot (z-score)",
    "tool_pm_urgency": "tool PM urgency index (high = approaching PM interval)",
    "humidity_x_floor_life": "interaction: humidity exposure × floor life duration",
    "moisture_x_cure_deficit": "interaction: moisture × mold cure time deficit",
}


def engineer_lineage_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lineage-derived features to the dataframe in-place.
    Returns the enriched dataframe.
    """
    df = df.copy()

    # 1. Wafer position — edge dies experience larger thermal gradients
    if "die_position_x" in df.columns and "die_position_y" in df.columns:
        cx, cy = 10.0, 10.0  # assumed wafer centre
        df["edge_distance"] = np.sqrt(
            (df["die_position_x"] - cx) ** 2 + (df["die_position_y"] - cy) ** 2
        )

    # 2. Within-lot z-scores: deviation from lot mean
    for col, out_name in [
        ("bond_force_gf", "bond_force_z_in_lot"),
        ("moisture_content_ppm", "moisture_z_in_lot"),
        ("die_attach_temp_c", "attach_temp_z_in_lot"),
    ]:
        if col in df.columns and "lot_id" in df.columns:
            lot_mean = df.groupby("lot_id")[col].transform("mean")
            lot_std = df.groupby("lot_id")[col].transform("std").fillna(1).replace(0, 1)
            df[out_name] = (df[col] - lot_mean) / lot_std

    # 3. Tool PM urgency — higher = tool is approaching maintenance threshold
    if "days_since_last_pm" in df.columns:
        PM_INTERVAL = 30  # typical PM interval in days
        df["tool_pm_urgency"] = df["days_since_last_pm"] / PM_INTERVAL
        df["tool_pm_urgency"] = df["tool_pm_urgency"].clip(0, 2)  # cap at 2x overdue

    # 4. Interaction: humidity × floor_life (moisture absorption proxy)
    if "clean_room_humidity_pct" in df.columns and "floor_life_hours" in df.columns:
        df["humidity_x_floor_life"] = df["clean_room_humidity_pct"] * df["floor_life_hours"]

    # 5. Interaction: moisture × cure deficit (delamination risk compound)
    if "moisture_content_ppm" in df.columns and "mold_cure_time_s" in df.columns:
        nominal_cure = 90.0
        cure_deficit = np.maximum(0, nominal_cure - df["mold_cure_time_s"])
        df["moisture_x_cure_deficit"] = df["moisture_content_ppm"] * cure_deficit

    # 6. Normalised batch position (first and last units in lot often diverge)
    if "lot_batch_position" in df.columns:
        df["time_since_lot_start_norm"] = df["lot_batch_position"] / df["lot_batch_position"].max()

    return df


def get_all_feature_columns(df: pd.DataFrame) -> list:
    exclude = {"unit_id", "label", "failure_mode", "failure_mode_name",
               "lot_id", "tool_chamber_id"}
    return [c for c in df.columns if c not in exclude and df[c].dtype != object]
