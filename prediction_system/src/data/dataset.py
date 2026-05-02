"""
Semiconductor manufacturing dataset loader.
Generates realistic synthetic data modelling IC packaging processes
with delamination, warpage, and bonding failure modes.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import requests
import io
import warnings

warnings.filterwarnings("ignore")

FAILURE_MODES = {
    0: "Pass",
    1: "Delamination",
    2: "Warpage",
    3: "Bond Failure",
    4: "Voiding",
    5: "CTE Mismatch Crack",
}

# Mapping failure modes to binary (0=pass, 1=any failure)
BINARY_LABEL = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}

FEATURE_GROUPS = {
    "die_attach": [
        "die_attach_temp_c",
        "die_attach_dwell_time_s",
        "die_attach_force_n",
        "epoxy_dispense_volume_ul",
        "epoxy_pot_life_pct",
        "die_attach_void_pct",
    ],
    "wire_bond": [
        "bond_force_gf",
        "bond_power_mw",
        "bond_time_ms",
        "loop_height_um",
        "wire_pull_strength_gf",
        "ball_shear_gf",
    ],
    "encapsulation": [
        "mold_temp_c",
        "mold_pressure_bar",
        "mold_cure_time_s",
        "filler_content_pct",
        "moisture_content_ppm",
        "gel_time_s",
    ],
    "reflow": [
        "reflow_peak_temp_c",
        "reflow_zone4_temp_c",
        "reflow_zone5_temp_c",
        "reflow_dwell_time_above_liquidus_s",
        "reflow_ramp_rate_c_per_s",
        "reflow_cool_rate_c_per_s",
        "solder_paste_volume_pct",
    ],
    "environment": [
        "clean_room_humidity_pct",
        "clean_room_temp_c",
        "time_to_reflow_min",
        "floor_life_hours",
    ],
    "material": [
        "substrate_thickness_mm",
        "die_thickness_um",
        "underfill_viscosity_cps",
        "pcb_warpage_um",
        "thermal_resistance_cjb",
    ],
}

LINEAGE_FEATURES = [
    "lot_id",
    "wafer_id",
    "die_position_x",
    "die_position_y",
    "tool_chamber_id",
    "operator_shift",
    "days_since_last_pm",
    "lot_batch_position",
    "wafer_edge_flag",
    "cumulative_tool_cycles",
    "lot_mean_bond_force",
    "lot_std_bond_force",
    "lot_mean_attach_temp",
    "lot_p90_moisture",
    "lot_fail_rate_rolling",
]


def _normal(mean, std, size, clip_low=None, clip_high=None):
    vals = np.random.normal(mean, std, size)
    if clip_low is not None:
        vals = np.clip(vals, clip_low, None)
    if clip_high is not None:
        vals = np.clip(vals, None, clip_high)
    return vals


def generate_synthetic_dataset(n_samples: int = 3000, failure_rate: float = 0.04, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic semiconductor process data with physically-motivated
    failure modes. Each failure mode is linked to specific process parameters
    that deviate from nominal.
    """
    np.random.seed(seed)
    n_fail = int(n_samples * failure_rate)
    n_pass = n_samples - n_fail

    # --- Assign failure modes (weighted) ---
    fail_mode_weights = [0.35, 0.25, 0.20, 0.12, 0.08]  # delamination heaviest
    fail_modes = np.random.choice([1, 2, 3, 4, 5], size=n_fail, p=fail_mode_weights)
    labels = np.concatenate([np.zeros(n_pass, dtype=int), fail_modes])
    np.random.shuffle(labels)
    n = len(labels)

    df = pd.DataFrame()
    df["unit_id"] = [f"UNIT-{i:05d}" for i in range(n)]
    df["failure_mode"] = labels
    df["label"] = df["failure_mode"].map(BINARY_LABEL)
    df["failure_mode_name"] = df["failure_mode"].map(FAILURE_MODES)

    is_pass = df["failure_mode"] == 0
    is_delam = df["failure_mode"] == 1
    is_warp = df["failure_mode"] == 2
    is_bond = df["failure_mode"] == 3
    is_void = df["failure_mode"] == 4
    is_cte = df["failure_mode"] == 5

    # ---------------------------------------------------------------
    # DIE ATTACH parameters
    # ---------------------------------------------------------------
    # Nominal: attach_temp=150°C, dwell=3s
    # Delamination: high moisture + low force
    # Voiding: inconsistent dispense volume
    attach_temp = np.full(n, 150.0)
    attach_temp[is_delam] += np.random.normal(8, 3, is_delam.sum())   # thermal stress
    attach_temp[is_cte] += np.random.normal(12, 4, is_cte.sum())
    attach_temp += np.random.normal(0, 1.5, n)
    df["die_attach_temp_c"] = np.clip(attach_temp, 130, 185)

    dwell = np.full(n, 3.0)
    dwell[is_delam] -= np.random.uniform(0.5, 1.2, is_delam.sum())   # under-cured → delamination
    dwell += np.random.normal(0, 0.2, n)
    df["die_attach_dwell_time_s"] = np.clip(dwell, 1.0, 6.0)

    force = np.full(n, 120.0)
    force[is_delam] -= np.random.normal(20, 5, is_delam.sum())        # insufficient bonding
    force += np.random.normal(0, 5, n)
    df["die_attach_force_n"] = np.clip(force, 60, 180)

    dispense = np.full(n, 5.0)
    dispense[is_void] += np.random.normal(2.0, 0.8, is_void.sum())    # excess → voiding
    void_idx = np.where(is_void)[0]
    flip_mask = np.random.rand(len(void_idx)) > 0.5
    dispense[void_idx[flip_mask]] -= 1.5    # or deficit
    dispense += np.random.normal(0, 0.3, n)
    df["epoxy_dispense_volume_ul"] = np.clip(dispense, 2.0, 9.0)

    pot_life = np.full(n, 60.0)
    pot_life[is_delam] -= np.random.normal(15, 5, is_delam.sum())     # aged epoxy
    pot_life += np.random.normal(0, 4, n)
    df["epoxy_pot_life_pct"] = np.clip(pot_life, 10, 100)

    void_pct = np.full(n, 3.0)
    void_pct[is_delam] += np.random.normal(6, 2, is_delam.sum())
    void_pct[is_void] += np.random.normal(10, 3, is_void.sum())
    void_pct += np.random.normal(0, 0.5, n)
    df["die_attach_void_pct"] = np.clip(void_pct, 0, 25)

    # ---------------------------------------------------------------
    # WIRE BOND parameters
    # ---------------------------------------------------------------
    bond_force = np.full(n, 20.0)
    bond_force[is_bond] -= np.random.normal(5, 2, is_bond.sum())
    bond_force += np.random.normal(0, 1, n)
    df["bond_force_gf"] = np.clip(bond_force, 8, 35)

    bond_power = np.full(n, 250.0)
    bond_power[is_bond] += np.random.normal(40, 12, is_bond.sum())
    bond_power += np.random.normal(0, 10, n)
    df["bond_power_mw"] = np.clip(bond_power, 150, 400)

    bond_time = np.full(n, 12.0)
    bond_time[is_bond] -= np.random.normal(3, 1, is_bond.sum())
    bond_time += np.random.normal(0, 0.8, n)
    df["bond_time_ms"] = np.clip(bond_time, 5, 25)

    loop_h = np.full(n, 120.0)
    loop_h[is_bond] += np.random.normal(30, 10, is_bond.sum())
    loop_h += np.random.normal(0, 5, n)
    df["loop_height_um"] = np.clip(loop_h, 60, 220)

    pull_str = np.full(n, 8.0)
    pull_str[is_bond] -= np.random.normal(2.5, 0.8, is_bond.sum())
    pull_str += np.random.normal(0, 0.4, n)
    df["wire_pull_strength_gf"] = np.clip(pull_str, 2, 15)

    shear = np.full(n, 45.0)
    shear[is_bond] -= np.random.normal(12, 4, is_bond.sum())
    shear += np.random.normal(0, 2, n)
    df["ball_shear_gf"] = np.clip(shear, 10, 80)

    # ---------------------------------------------------------------
    # ENCAPSULATION / MOLD parameters
    # ---------------------------------------------------------------
    mold_temp = np.full(n, 175.0)
    mold_temp[is_delam] += np.random.normal(10, 3, is_delam.sum())    # thermal stress on interface
    mold_temp[is_warp] += np.random.normal(8, 3, is_warp.sum())
    mold_temp += np.random.normal(0, 2, n)
    df["mold_temp_c"] = np.clip(mold_temp, 155, 200)

    mold_pres = np.full(n, 70.0)
    mold_pres[is_void] -= np.random.normal(15, 5, is_void.sum())      # low pressure → voids
    mold_pres += np.random.normal(0, 3, n)
    df["mold_pressure_bar"] = np.clip(mold_pres, 30, 120)

    cure_time = np.full(n, 90.0)
    cure_time[is_delam] -= np.random.normal(20, 7, is_delam.sum())    # under-cure
    cure_time += np.random.normal(0, 5, n)
    df["mold_cure_time_s"] = np.clip(cure_time, 30, 180)

    filler = np.full(n, 75.0)
    filler[is_cte] -= np.random.normal(8, 3, is_cte.sum())            # lower filler → higher CTE
    filler += np.random.normal(0, 2, n)
    df["filler_content_pct"] = np.clip(filler, 55, 90)

    # KEY DELAMINATION DRIVER: moisture content
    moisture = np.full(n, 200.0)
    moisture[is_delam] += np.random.normal(300, 80, is_delam.sum())   # moisture → popcorn cracking
    moisture[is_cte] += np.random.normal(150, 50, is_cte.sum())
    moisture += np.random.normal(0, 30, n)
    df["moisture_content_ppm"] = np.clip(moisture, 50, 1200)

    gel_t = np.full(n, 25.0)
    gel_t[is_delam] += np.random.normal(8, 3, is_delam.sum())         # slow gel → moisture trap
    gel_t += np.random.normal(0, 2, n)
    df["gel_time_s"] = np.clip(gel_t, 10, 60)

    # ---------------------------------------------------------------
    # REFLOW parameters
    # ---------------------------------------------------------------
    reflow_peak = np.full(n, 245.0)
    reflow_peak[is_warp] += np.random.normal(12, 4, is_warp.sum())    # excess thermal stress
    reflow_peak[is_delam] += np.random.normal(8, 3, is_delam.sum())
    reflow_peak += np.random.normal(0, 3, n)
    df["reflow_peak_temp_c"] = np.clip(reflow_peak, 220, 275)

    z4_temp = np.full(n, 230.0)
    z4_temp[is_warp] += np.random.normal(15, 5, is_warp.sum())
    z4_temp += np.random.normal(0, 3, n)
    df["reflow_zone4_temp_c"] = np.clip(z4_temp, 200, 265)

    z5_temp = np.full(n, 240.0)
    z5_temp[is_warp] += np.random.normal(12, 4, is_warp.sum())
    z5_temp += np.random.normal(0, 3, n)
    df["reflow_zone5_temp_c"] = np.clip(z5_temp, 210, 270)

    dwell_liq = np.full(n, 45.0)
    dwell_liq[is_warp] += np.random.normal(20, 6, is_warp.sum())
    dwell_liq += np.random.normal(0, 4, n)
    df["reflow_dwell_time_above_liquidus_s"] = np.clip(dwell_liq, 20, 120)

    ramp = np.full(n, 2.0)
    ramp[is_warp] += np.random.normal(0.8, 0.3, is_warp.sum())
    ramp += np.random.normal(0, 0.15, n)
    df["reflow_ramp_rate_c_per_s"] = np.clip(ramp, 0.5, 4.0)

    cool = np.full(n, 3.0)
    cool[is_warp] += np.random.normal(1.5, 0.5, is_warp.sum())        # fast cool → warpage
    cool += np.random.normal(0, 0.2, n)
    df["reflow_cool_rate_c_per_s"] = np.clip(cool, 1.0, 8.0)

    solder_vol = np.full(n, 100.0)
    solder_vol[is_bond] -= np.random.normal(20, 7, is_bond.sum())
    solder_vol += np.random.normal(0, 5, n)
    df["solder_paste_volume_pct"] = np.clip(solder_vol, 50, 150)

    # ---------------------------------------------------------------
    # ENVIRONMENT
    # ---------------------------------------------------------------
    humidity = np.full(n, 45.0)
    humidity[is_delam] += np.random.normal(15, 5, is_delam.sum())     # humidity → delamination
    humidity += np.random.normal(0, 3, n)
    df["clean_room_humidity_pct"] = np.clip(humidity, 20, 75)

    cr_temp = np.full(n, 22.0)
    cr_temp[is_warp] += np.random.normal(3, 1, is_warp.sum())
    cr_temp += np.random.normal(0, 0.5, n)
    df["clean_room_temp_c"] = np.clip(cr_temp, 18, 28)

    t2reflow = np.full(n, 60.0)
    t2reflow[is_delam] += np.random.normal(90, 30, is_delam.sum())    # long exposure to humidity
    t2reflow += np.random.normal(0, 10, n)
    df["time_to_reflow_min"] = np.clip(t2reflow, 10, 480)

    floor_life = np.full(n, 24.0)
    floor_life[is_delam] += np.random.normal(48, 16, is_delam.sum())  # moisture exposure duration
    floor_life += np.random.normal(0, 4, n)
    df["floor_life_hours"] = np.clip(floor_life, 0, 168)

    # ---------------------------------------------------------------
    # MATERIAL properties
    # ---------------------------------------------------------------
    sub_thick = np.full(n, 0.5)
    sub_thick[is_warp] -= np.random.normal(0.08, 0.03, is_warp.sum())
    sub_thick += np.random.normal(0, 0.01, n)
    df["substrate_thickness_mm"] = np.clip(sub_thick, 0.2, 0.8)

    die_thick = np.full(n, 200.0)
    die_thick[is_warp] -= np.random.normal(30, 10, is_warp.sum())     # thin die → more warpage
    die_thick += np.random.normal(0, 8, n)
    df["die_thickness_um"] = np.clip(die_thick, 80, 300)

    underfill = np.full(n, 2500.0)
    underfill[is_void] += np.random.normal(800, 200, is_void.sum())   # high viscosity → voids
    underfill += np.random.normal(0, 100, n)
    df["underfill_viscosity_cps"] = np.clip(underfill, 500, 6000)

    pcb_warp = np.full(n, 50.0)
    pcb_warp[is_warp] += np.random.normal(80, 25, is_warp.sum())
    pcb_warp += np.random.normal(0, 8, n)
    df["pcb_warpage_um"] = np.clip(pcb_warp, 0, 300)

    therm_res = np.full(n, 8.0)
    therm_res[is_cte] += np.random.normal(3, 1, is_cte.sum())
    therm_res += np.random.normal(0, 0.4, n)
    df["thermal_resistance_cjb"] = np.clip(therm_res, 3, 20)

    # ---------------------------------------------------------------
    # LINEAGE features (production context)
    # ---------------------------------------------------------------
    n_lots = max(1, n // 25)
    lot_ids = np.random.randint(0, n_lots, n)
    df["lot_id"] = [f"LOT-{i:03d}" for i in lot_ids]
    df["wafer_id"] = np.random.randint(1, 26, n)
    df["die_position_x"] = np.random.randint(0, 20, n)
    df["die_position_y"] = np.random.randint(0, 20, n)
    df["tool_chamber_id"] = np.random.choice(["CH-A", "CH-B", "CH-C", "CH-D"], n)
    df["operator_shift"] = np.random.choice([1, 2, 3], n)
    df["days_since_last_pm"] = np.clip(np.random.exponential(15, n), 0, 90)
    df["lot_batch_position"] = np.random.randint(1, 26, n)
    df["wafer_edge_flag"] = ((df["die_position_x"] <= 1) | (df["die_position_x"] >= 18) |
                             (df["die_position_y"] <= 1) | (df["die_position_y"] >= 18)).astype(int)
    df["cumulative_tool_cycles"] = np.random.randint(1000, 50000, n)

    # Lot-level aggregate features (captured production context)
    lot_bond_force = df.groupby("lot_id")["bond_force_gf"].transform("mean")
    lot_bond_std = df.groupby("lot_id")["bond_force_gf"].transform("std").fillna(0)
    lot_attach_temp = df.groupby("lot_id")["die_attach_temp_c"].transform("mean")
    lot_p90_moisture = df.groupby("lot_id")["moisture_content_ppm"].transform(lambda x: np.percentile(x, 90))

    df["lot_mean_bond_force"] = lot_bond_force
    df["lot_std_bond_force"] = lot_bond_std
    df["lot_mean_attach_temp"] = lot_attach_temp
    df["lot_p90_moisture"] = lot_p90_moisture

    # Rolling lot failure rate (simulated — uses lot index as proxy)
    df["lot_fail_rate_rolling"] = lot_ids / n_lots * 0.04 + np.random.normal(0, 0.005, n)
    df["lot_fail_rate_rolling"] = np.clip(df["lot_fail_rate_rolling"], 0, 0.15)

    # Introduce ~5% random missing values (real fab data is never perfect)
    process_cols = [c for c in df.columns if c not in
                    ["unit_id", "label", "failure_mode", "failure_mode_name",
                     "lot_id", "tool_chamber_id", "operator_shift", "wafer_edge_flag"]]
    for col in process_cols:
        mask = np.random.rand(n) < 0.03
        df.loc[mask, col] = np.nan

    return df


def load_dataset(n_samples: int = 3000, seed: int = 42) -> pd.DataFrame:
    return generate_synthetic_dataset(n_samples=n_samples, seed=seed)


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude = ["unit_id", "label", "failure_mode", "failure_mode_name",
               "lot_id", "tool_chamber_id", "operator_shift"]
    return [c for c in df.columns if c not in exclude]


def get_categorical_columns() -> list:
    return ["tool_chamber_id", "operator_shift", "wafer_edge_flag"]
