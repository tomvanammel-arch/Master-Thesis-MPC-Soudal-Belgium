"""Patch notebook 10 Part 1 access power (HP). Run once then delete if desired."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "10_online_MPC_1_HP.ipynb"

INTRO_MD = """# Online HP-Only MPC 1 – Access Power, Simulation & Comparison

This notebook mirrors **notebook 09** (EV) for the HP-only online MPC study:

1. **Monthly access power (§1.1–§1.3)**
   - **§1.1:** worst-case HP @ −10°C stress on historical grid peaks (intuition only).
   - **§1.2 (grid-based):** **baseline HP access** = cum-max `grid_consumption` (M−1) + 20 kW + worst-case HP electrical peak; **flex-aware (online)** = cum-max `grid_consumption` (M−1) + 20 kW; **deterministic** from notebook 03 export.
   - **§1.3 (no-PV):** same rules on `grid + pv − injection` counterfactual (validation; not wired to Part 2).

2. **Rolling-horizon online MPC simulation (HP-only)** — uses §1.2 **flex-aware** access.

3. **Visualisation and comparison** — baseline / deterministic (notebook 03) / online.

**Run order:** notebook 03 export → §1.1 → §1.2 → (optional §1.3) → Part 2 → Part 3.
"""

MD_11 = """### 1.1 Grid consumption, worst-case HP peak, and buffer intuition

Inspect **Plant 1** `grid_consumption` (includes EV; no HP in meter history) and add a **worst-case HP electrical peak @ −10°C** for peak-duration and buffer sizing intuition only. This section does **not** set the Part 2 access contract.
"""

MD_12 = """### 1.2 Access power selection (grid-based)

Monthly access from **cum-max of metered `grid_consumption` peaks** (kWh/15 min → kW via ×4) on `plant1.csv` (2025) with January 2025 seeded from `plant1_2024_training.csv`, plus **+20 kW** margin.

- **Flex-aware (online MPC, Part 2):** \\(P_{\\mathrm{access},M}^{\\mathrm{flex}} = \\max_{m<M}(\\max_t 4\\cdot\\texttt{grid\\_consumption}) + 20\\) kW  
- **Baseline HP (Part 3.2):** flex-aware grid part **+** worst-case HP electrical peak @ COP(−10°C) (matches notebook 03 `baseline_hp_access_power_kw`).  
- **Deterministic:** `access_power_kw` from `output/notebooks/deterministic_hp_monthly_bills_notebook_03.csv` (notebook 03).

**Headroom validation (full calendar day):** playroom = access − \\(4\\times\\texttt{grid\\_consumption}\\); daily need = sum of `thermal_load / COP(−10°C)`; utilisation vs **70%** reference line.

**Exports:** `table`, `ACCESS_POWER_BASELINE_MONTHLY`, `ACCESS_POWER_ONLINE_MONTHLY`, `ACCESS_POWER_DICT`.
"""

MD_13 = """### 1.3 Access power selection (no-PV counterfactual)

Same structure as **§1.2**, but monthly peaks use **no-PV** site load:

- `grid_nopv = grid_consumption + pv_production − grid_injection` (2024 training: `inflex_load + ev` if no injection column).

Headroom playroom uses `grid_nopv` as baseline load (full day). **Not used in Part 2** unless you explicitly switch the online access mapping.
"""

SECTION_12 = r'''# §1.2 — Access power selection (grid-based, HP)

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("..").resolve()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

NOTEBOOKS_OUTPUT_DIR = PROJECT_ROOT / "output" / "notebooks"
MARGIN_KW = 20.0

months = pd.period_range("2025-01", "2025-12", freq="M")

# --- Plant 2025 (reuse §1.1 `plant` when available) ---
if "plant" not in globals():
    plant = pd.read_csv(PROJECT_ROOT / "data" / "plant1.csv")
    plant["timestamp"] = pd.to_datetime(plant["timestamp"], utc=True, errors="coerce")
    plant["timestamp"] = plant["timestamp"].dt.tz_convert("Europe/Brussels").dt.tz_localize(None)
    plant = plant.sort_values("timestamp").reset_index(drop=True)

plant_2025 = plant.loc[plant["timestamp"].dt.year == 2025].copy()
for _c in ["grid_consumption", "thermal_load", "pv_production", "grid_injection"]:
    if _c in plant_2025.columns:
        plant_2025[_c] = pd.to_numeric(plant_2025[_c], errors="coerce").fillna(0.0)

# --- Worst-case HP electrical peak @ -10°C (baseline access add-on) ---
if "hp_peak_elec_kw" in globals():
    hp_additional_peak_kw = float(hp_peak_elec_kw)
else:
    from heat_pump_load import load_hp_config, interpolate_cop

    hp_cfg = load_hp_config(str(PROJECT_ROOT / "config" / "hp.yaml"))
    cop_minus10 = float(interpolate_cop(-10.0, hp_cfg["COP_data"]))
    max_thermal_kwh = float(plant_2025["thermal_load"].max())
    thermal_max_kw = max_thermal_kwh * 4.0
    hp_additional_peak_kw = thermal_max_kw / cop_minus10

# --- 2024 training peak (grid_consumption only) ---
train_2024_path = PROJECT_ROOT / "data" / "plant1_2024_training.csv"
train_2024 = pd.read_csv(train_2024_path)
train_2024["month"] = pd.PeriodIndex(train_2024["timestamp"].astype(str).str.slice(0, 7), freq="M")
train_2024["grid_consumption"] = pd.to_numeric(train_2024["grid_consumption"], errors="coerce").fillna(0.0)
monthly_peak_2024_grid_kw = train_2024.groupby("month")["grid_consumption"].max() * 4.0
baseline_2024_peak_grid_kw = float(monthly_peak_2024_grid_kw.max())

# --- 2025 monthly peaks (grid_consumption, all hours) ---
tmp_2025 = plant_2025[["timestamp", "grid_consumption", "thermal_load"]].copy()
tmp_2025["month"] = tmp_2025["timestamp"].dt.to_period("M")
monthly_peak_2025_grid_kw = (
    tmp_2025.groupby("month")["grid_consumption"].max() * 4.0
).reindex(months).astype(float)

# --- Cum-max(M-1) + margin on grid_consumption ---
cummax_grid_Mm1_kw = monthly_peak_2025_grid_kw.cummax().shift(1)
cummax_grid_Mm1_kw.loc[months.min()] = baseline_2024_peak_grid_kw
cummax_grid_Mm1_kw = cummax_grid_Mm1_kw.fillna(baseline_2024_peak_grid_kw)

access_power_flex_aware_kw = cummax_grid_Mm1_kw + MARGIN_KW
access_power_baseline_hp_kw = access_power_flex_aware_kw + hp_additional_peak_kw

# 2024 hourly access for headroom plots (grid part only / baseline HP)
months_2024 = monthly_peak_2024_grid_kw.sort_index().index
cummax_2024_Mm1_kw = monthly_peak_2024_grid_kw.cummax().shift(1)
cummax_2024_Mm1_kw.loc[months_2024.min()] = float(monthly_peak_2024_grid_kw.iloc[0])
cummax_2024_Mm1_kw = cummax_2024_Mm1_kw.fillna(float(monthly_peak_2024_grid_kw.iloc[0]))
access_power_flex_2024_kw = cummax_2024_Mm1_kw + MARGIN_KW
access_power_baseline_hp_2024_kw = access_power_flex_2024_kw + hp_additional_peak_kw
access_power_by_month_flex_hr = pd.concat([access_power_flex_2024_kw, access_power_flex_aware_kw]).sort_index()
access_power_by_month_baseline_hp_hr = pd.concat(
    [access_power_baseline_hp_2024_kw, access_power_baseline_hp_kw]
).sort_index()

# --- Deterministic access (notebook 03) ---
DET_HP_BILLS_PATH = NOTEBOOKS_OUTPUT_DIR / "deterministic_hp_monthly_bills_notebook_03.csv"
if not DET_HP_BILLS_PATH.exists():
    raise FileNotFoundError(
        f"Missing {DET_HP_BILLS_PATH}. Run notebook 03 export cell first."
    )
det_bills_ap = pd.read_csv(DET_HP_BILLS_PATH)
access_power_deterministic_kw = (
    det_bills_ap.assign(month_key=det_bills_ap["month"].astype(str))
    .set_index("month_key")["access_power_kw"]
    .astype(float)
)

table = pd.DataFrame(
    {
        "monthly_peak_grid_kw": monthly_peak_2025_grid_kw.values,
        "cummax_grid_kw_M_minus_1": cummax_grid_Mm1_kw.values,
        "access_power_flex_aware": access_power_flex_aware_kw.values,
        "access_power_baseline_hp": access_power_baseline_hp_kw.values,
        "access_power_deterministic": access_power_deterministic_kw.reindex(months.astype(str)).values,
    },
    index=months.astype(str),
).T
display(table.round(1))

# --- plant_hr for headroom (2024–2025 grid; 2025 only for thermal / HP need) ---
# plant1_2024_training.csv has grid_consumption but no thermal_load column.
train_2024_ts = pd.to_datetime(train_2024["timestamp"], utc=True, errors="coerce")
train_2024_ts = train_2024_ts.dt.tz_convert("Europe/Brussels").dt.tz_localize(None)

plant_hr_grid = pd.concat(
    [
        train_2024.assign(timestamp=train_2024_ts)[["timestamp", "grid_consumption"]],
        plant_2025[["timestamp", "grid_consumption"]],
    ],
    ignore_index=True,
).sort_values("timestamp")
plant_hr_grid["grid_consumption"] = pd.to_numeric(
    plant_hr_grid["grid_consumption"], errors="coerce"
).fillna(0.0)

if "cop_minus10" not in globals():
    from heat_pump_load import load_hp_config, interpolate_cop

    hp_cfg = load_hp_config(str(PROJECT_ROOT / "config" / "hp.yaml"))
    cop_minus10 = float(interpolate_cop(-10.0, hp_cfg["COP_data"]))

plant_hr_hp = plant_2025[["timestamp", "thermal_load"]].copy()
plant_hr_hp["thermal_load"] = pd.to_numeric(plant_hr_hp["thermal_load"], errors="coerce").fillna(0.0)
plant_hr_hp["hp_need_kwh_15"] = plant_hr_hp["thermal_load"] / cop_minus10


def _daily_headroom_kwh(ts, load_kwh_15, access_by_month):
    access_kw = access_by_month.reindex(ts.dt.to_period("M")).to_numpy(dtype=float)
    headroom_kw = np.maximum(access_kw - 4.0 * np.asarray(load_kwh_15, dtype=float), 0.0)
    return (
        pd.DataFrame({"date": ts.dt.normalize(), "h": headroom_kw * 0.25})
        .groupby("date")["h"]
        .sum()
    )


ts_hr = plant_hr_grid["timestamp"]
grid_kwh = plant_hr_grid["grid_consumption"].to_numpy(dtype=float)
daily_h_base = _daily_headroom_kwh(ts_hr, grid_kwh, access_power_by_month_baseline_hp_hr)
daily_h_flex = _daily_headroom_kwh(ts_hr, grid_kwh, access_power_by_month_flex_hr)

daily_hp_need_kwh = (
    pd.DataFrame({"date": plant_hr_hp["timestamp"].dt.normalize(), "hp": plant_hr_hp["hp_need_kwh_15"]})
    .groupby("date")["hp"]
    .sum()
    .reindex(daily_h_base.index)
)
hr_dates = pd.to_datetime(daily_h_base.index)
util_base_pct = np.where(daily_h_base > 1e-9, 100.0 * daily_hp_need_kwh / daily_h_base, np.nan)
util_flex_pct = np.where(daily_h_flex > 1e-9, 100.0 * daily_hp_need_kwh / daily_h_flex, np.nan)

# --- §1.2 plots ---
C_BLACK = "#000000"
C_RED = "#d62728"
C_BLUE = "#1f77b4"
C_GREEN = "#2ca02c"
C_ORANGE = "#ff7f0e"
LW_ACCESS, LW_DAILY = 2.0, 1.5

month_ts = months.to_timestamp()
peak_grid_kw = monthly_peak_2025_grid_kw.values.astype(float)
_bar_w_days = 22

fig_acc, ax_acc = plt.subplots(figsize=(10, 5))
ax_acc.bar(
    month_ts,
    peak_grid_kw,
    width=_bar_w_days,
    align="center",
    color=C_GREEN,
    alpha=0.45,
    edgecolor=C_GREEN,
    linewidth=0.6,
    label="Actual monthly peak (grid)",
    zorder=1,
)
ax_acc.step(
    month_ts,
    access_power_baseline_hp_kw.values,
    where="post",
    color=C_BLUE,
    linewidth=LW_ACCESS,
    linestyle="-",
    label="Access baseline HP (+ HP @ -10°C)",
    zorder=3,
)
ax_acc.step(
    month_ts,
    access_power_deterministic_kw.reindex(months.astype(str)).values,
    where="post",
    color=C_ORANGE,
    linewidth=LW_ACCESS,
    linestyle="-",
    label="Access deterministic (notebook 03)",
    zorder=3,
)
ax_acc.step(
    month_ts,
    access_power_flex_aware_kw.values,
    where="post",
    color=C_BLACK,
    linewidth=LW_ACCESS,
    linestyle="-",
    label="Access online / flex-aware",
    zorder=3,
)
ax_acc.set_title("§1.2 Monthly access power (2025)")
ax_acc.set_ylabel("kW")
ax_acc.set_xlabel("Month")
ax_acc.set_ylim(
    2200,
    max(
        access_power_baseline_hp_kw.max(),
        access_power_flex_aware_kw.max(),
        access_power_deterministic_kw.max(),
        monthly_peak_2025_grid_kw.max(),
    )
    + 120,
)
ax_acc.grid(True, axis="y", linestyle="--", alpha=0.35)
ax_acc.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8, framealpha=0.95)
fig_acc.subplots_adjust(bottom=0.28)
plt.show()

fig_hr, ax_hr = plt.subplots(figsize=(14, 4.5))
ax_hr.step(hr_dates, daily_h_base.values, where="post", color=C_BLUE, linewidth=LW_DAILY, label="Headroom (baseline HP)")
ax_hr.step(hr_dates, daily_h_flex.values, where="post", color=C_GREEN, linewidth=LW_DAILY, label="Headroom (flex-aware)")
ax_hr.step(hr_dates, daily_hp_need_kwh.values, where="post", color=C_RED, linewidth=LW_DAILY, label="Daily HP need @ -10°C")
ax_hr.set_ylabel("Energy [kWh/day]")
ax_hr.set_xlabel("Date (2024–2025)")
ax_hr.set_title("§1.2 Daily headroom vs HP need (full day)")
ax_hr.grid(True, axis="y", linestyle="--", alpha=0.35)
ax_hr.legend(loc="upper right", fontsize=9, framealpha=0.95)
plt.tight_layout()
plt.show()

fig_u, ax_u = plt.subplots(figsize=(14, 4.5))
ax_u.step(hr_dates, util_base_pct, where="post", color=C_BLUE, linewidth=LW_DAILY, label="HP / H (baseline HP)")
ax_u.step(hr_dates, util_flex_pct, where="post", color=C_GREEN, linewidth=LW_DAILY, label="HP / H (flex-aware)")
ax_u.axhline(70.0, color=C_BLACK, linewidth=1.5, linestyle="-", label="70% reference")
ax_u.set_ylabel("Utilisation [%]")
ax_u.set_xlabel("Date (2024–2025)")
ax_u.set_title("§1.2 Daily headroom utilisation (full day)")
ax_u.grid(True, axis="y", linestyle="--", alpha=0.35)
ax_u.legend(loc="upper right", fontsize=9, framealpha=0.95)
plt.tight_layout()
plt.show()

# --- Exports for Parts 2–4 ---
ACCESS_POWER_BASELINE_MONTHLY = access_power_baseline_hp_kw.copy()
ACCESS_POWER_BASELINE_MONTHLY.index = ACCESS_POWER_BASELINE_MONTHLY.index.astype(str)
ACCESS_POWER_ONLINE_MONTHLY = access_power_flex_aware_kw.copy()
ACCESS_POWER_ONLINE_MONTHLY.index = ACCESS_POWER_ONLINE_MONTHLY.index.astype(str)
ACCESS_POWER_DICT = {str(k): float(v) for k, v in ACCESS_POWER_ONLINE_MONTHLY.items()}

print("ACCESS_POWER_ONLINE_MONTHLY (flex-aware, kW):")
display(ACCESS_POWER_ONLINE_MONTHLY.round(1))
print("ACCESS_POWER_BASELINE_MONTHLY (baseline HP, kW):")
display(ACCESS_POWER_BASELINE_MONTHLY.round(1))
'''

SECTION_13 = r'''# §1.3 — Access power selection (no-PV counterfactual, HP)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path("..").resolve()
MARGIN_KW = 20.0

months = pd.period_range("2025-01", "2025-12", freq="M")

if "hp_additional_peak_kw" not in globals():
    raise RuntimeError("Run §1.2 first (hp_additional_peak_kw).")


def _add_nopv_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["grid_consumption", "pv_production", "grid_injection", "inflex_load", "ev"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    if "grid_injection" in out.columns:
        out["grid_nopv"] = out["grid_consumption"] + out["pv_production"] - out["grid_injection"]
    else:
        out["grid_nopv"] = out["inflex_load"] + out["ev"]
    return out


if "plant" not in globals():
    raise RuntimeError("Run §1.1 / §1.2 first so `plant` is defined.")

train_2024_path = PROJECT_ROOT / "data" / "plant1_2024_training.csv"
train_2024 = _add_nopv_columns(pd.read_csv(train_2024_path))
train_2024["month"] = pd.PeriodIndex(train_2024["timestamp"].astype(str).str.slice(0, 7), freq="M")
monthly_peak_2024_grid_nopv_kw = train_2024.groupby("month")["grid_nopv"].max() * 4.0
baseline_2024_peak_grid_nopv_kw = float(monthly_peak_2024_grid_nopv_kw.max())

_pv_cols = [c for c in ["pv_production", "grid_injection"] if c in plant.columns]
_tmp_cols = ["timestamp", "grid_consumption"] + _pv_cols
if "inflex_load" in plant.columns and "ev" in plant.columns:
    _tmp_cols += ["inflex_load", "ev"]
tmp_2025 = _add_nopv_columns(plant.loc[plant["timestamp"].dt.year == 2025, _tmp_cols].copy())
tmp_2025["month"] = tmp_2025["timestamp"].dt.to_period("M")
monthly_peak_2025_grid_nopv_kw = (
    tmp_2025.groupby("month")["grid_nopv"].max() * 4.0
).reindex(months).astype(float)

cummax_grid_nopv_Mm1_kw = monthly_peak_2025_grid_nopv_kw.cummax().shift(1)
cummax_grid_nopv_Mm1_kw.loc[months.min()] = baseline_2024_peak_grid_nopv_kw
cummax_grid_nopv_Mm1_kw = cummax_grid_nopv_Mm1_kw.fillna(baseline_2024_peak_grid_nopv_kw)
access_power_flex_aware_nopv_kw = cummax_grid_nopv_Mm1_kw + MARGIN_KW
access_power_baseline_hp_nopv_kw = access_power_flex_aware_nopv_kw + hp_additional_peak_kw

months_2024 = monthly_peak_2024_grid_nopv_kw.sort_index().index
cummax_2024_nopv_Mm1_kw = monthly_peak_2024_grid_nopv_kw.cummax().shift(1)
cummax_2024_nopv_Mm1_kw.loc[months_2024.min()] = float(monthly_peak_2024_grid_nopv_kw.iloc[0])
cummax_2024_nopv_Mm1_kw = cummax_2024_nopv_Mm1_kw.fillna(float(monthly_peak_2024_grid_nopv_kw.iloc[0]))
access_power_flex_nopv_2024_kw = cummax_2024_nopv_Mm1_kw + MARGIN_KW
access_power_baseline_hp_nopv_2024_kw = access_power_flex_nopv_2024_kw + hp_additional_peak_kw
access_power_by_month_flex_nopv_hr = pd.concat(
    [access_power_flex_nopv_2024_kw, access_power_flex_aware_nopv_kw]
).sort_index()
access_power_by_month_baseline_hp_nopv_hr = pd.concat(
    [access_power_baseline_hp_nopv_2024_kw, access_power_baseline_hp_nopv_kw]
).sort_index()

table_nopv = pd.DataFrame(
    {
        "monthly_peak_grid_nopv_kw": monthly_peak_2025_grid_nopv_kw.values,
        "cummax_grid_nopv_M_minus_1": cummax_grid_nopv_Mm1_kw.values,
        "access_power_flex_aware_nopv": access_power_flex_aware_nopv_kw.values,
        "access_power_baseline_hp_nopv": access_power_baseline_hp_nopv_kw.values,
    },
    index=months.astype(str),
).T
display(table_nopv.round(1))

train_2024_ts = pd.to_datetime(train_2024["timestamp"], utc=True, errors="coerce")
train_2024_ts = train_2024_ts.dt.tz_convert("Europe/Brussels").dt.tz_localize(None)
plant_2025_nopv = _add_nopv_columns(plant.loc[plant["timestamp"].dt.year == 2025].copy())
plant_hr_nopv = pd.concat(
    [
        train_2024.assign(timestamp=train_2024_ts)[["timestamp", "grid_nopv", "thermal_load"]],
        plant_2025_nopv.assign(
            timestamp=pd.to_datetime(plant_2025_nopv["timestamp"], errors="coerce")
        )[["timestamp", "grid_nopv", "thermal_load"]],
    ],
    ignore_index=True,
).sort_values("timestamp")
for _c in ["grid_nopv", "thermal_load"]:
    plant_hr_nopv[_c] = pd.to_numeric(plant_hr_nopv[_c], errors="coerce").fillna(0.0)
plant_hr_nopv["hp_need_kwh_15"] = plant_hr_nopv["thermal_load"] / cop_minus10

ts_hr = plant_hr_nopv["timestamp"]
grid_nopv_kwh = plant_hr_nopv["grid_nopv"].to_numpy(dtype=float)
daily_h_base_nopv = _daily_headroom_kwh(ts_hr, grid_nopv_kwh, access_power_by_month_baseline_hp_nopv_hr)
daily_h_flex_nopv = _daily_headroom_kwh(ts_hr, grid_nopv_kwh, access_power_by_month_flex_nopv_hr)
daily_hp_need_kwh_nopv = (
    pd.DataFrame({"date": ts_hr.dt.normalize(), "hp": plant_hr_nopv["hp_need_kwh_15"]})
    .groupby("date")["hp"]
    .sum()
    .reindex(daily_h_base_nopv.index)
)
hr_dates_nopv = pd.to_datetime(daily_h_base_nopv.index)
util_base_nopv_pct = np.where(
    daily_h_base_nopv > 1e-9, 100.0 * daily_hp_need_kwh_nopv / daily_h_base_nopv, np.nan
)
util_flex_nopv_pct = np.where(
    daily_h_flex_nopv > 1e-9, 100.0 * daily_hp_need_kwh_nopv / daily_h_flex_nopv, np.nan
)

C_BLACK, C_RED, C_BLUE, C_GREEN = "#000000", "#d62728", "#1f77b4", "#2ca02c"
LW_ACCESS, LW_DAILY = 2.0, 1.5
month_ts = months.to_timestamp()

fig_acc_nopv, ax_acc_nopv = plt.subplots(figsize=(10, 5))
ax_acc_nopv.bar(
    month_ts,
    monthly_peak_2025_grid_nopv_kw.values,
    width=22,
    color=C_GREEN,
    alpha=0.45,
    label="Actual peak (grid no-PV)",
)
ax_acc_nopv.step(
    month_ts,
    access_power_baseline_hp_nopv_kw.values,
    where="post",
    color=C_BLUE,
    linewidth=LW_ACCESS,
    label="Access baseline HP (no-PV)",
)
ax_acc_nopv.step(
    month_ts,
    access_power_flex_aware_nopv_kw.values,
    where="post",
    color=C_BLACK,
    linewidth=LW_ACCESS,
    label="Access flex-aware (no-PV)",
)
ax_acc_nopv.set_title("§1.3 Monthly access power (2025, no-PV)")
ax_acc_nopv.set_ylabel("kW")
ax_acc_nopv.grid(True, axis="y", linestyle="--", alpha=0.35)
ax_acc_nopv.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)
fig_acc_nopv.subplots_adjust(bottom=0.22)
plt.show()

fig_hr_nopv, ax_hr_nopv = plt.subplots(figsize=(14, 4.5))
ax_hr_nopv.step(hr_dates_nopv, daily_h_base_nopv.values, where="post", color=C_BLUE, linewidth=LW_DAILY, label="Headroom (baseline HP)")
ax_hr_nopv.step(hr_dates_nopv, daily_h_flex_nopv.values, where="post", color=C_GREEN, linewidth=LW_DAILY, label="Headroom (flex-aware)")
ax_hr_nopv.step(hr_dates_nopv, daily_hp_need_kwh_nopv.values, where="post", color=C_RED, linewidth=LW_DAILY, label="Daily HP need @ -10°C")
ax_hr_nopv.set_title("§1.3 Daily headroom vs HP need (full day, no-PV)")
ax_hr_nopv.set_ylabel("Energy [kWh/day]")
ax_hr_nopv.grid(True, axis="y", linestyle="--", alpha=0.35)
ax_hr_nopv.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.show()

fig_u_nopv, ax_u_nopv = plt.subplots(figsize=(14, 4.5))
ax_u_nopv.step(hr_dates_nopv, util_base_nopv_pct, where="post", color=C_BLUE, linewidth=LW_DAILY, label="HP / H (baseline HP)")
ax_u_nopv.step(hr_dates_nopv, util_flex_nopv_pct, where="post", color=C_GREEN, linewidth=LW_DAILY, label="HP / H (flex-aware)")
ax_u_nopv.axhline(70.0, color=C_BLACK, linewidth=1.5, linestyle="-", label="70% reference")
ax_u_nopv.set_title("§1.3 Daily headroom utilisation (full day, no-PV)")
ax_u_nopv.set_ylabel("Utilisation [%]")
ax_u_nopv.grid(True, axis="y", linestyle="--", alpha=0.35)
ax_u_nopv.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.show()

# Part 2 still uses §1.2 grid-based ACCESS_POWER_ONLINE_MONTHLY (not overwritten here).
'''


def _cell(source: str, cell_type: str = "code") -> dict:
    lines = source.strip("\n").splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines,
        "id": uuid.uuid4().hex[:8],
    }


def _lines_to_str(lines: list[str]) -> str:
    return "".join(lines)


def patch_part2(source: str) -> str:
    old = (
        "# Requires ACCESS_POWER_DICT from Part 1 (month_key -> kW).\n"
    )
    new = (
        "# Requires §1.2: ACCESS_POWER_ONLINE_MONTHLY (flex-aware) and ACCESS_POWER_DICT.\n"
        "if \"ACCESS_POWER_ONLINE_MONTHLY\" not in globals():\n"
        "    raise RuntimeError(\"Run §1.2 first to build ACCESS_POWER_ONLINE_MONTHLY.\")\n"
        "ACCESS_POWER_DICT = {str(k): float(v) for k, v in ACCESS_POWER_ONLINE_MONTHLY.items()}\n"
    )
    if old in source:
        source = source.replace(old, new, 1)
    # Remove redundant self-assignments if present
    source = source.replace(
        "ACCESS_POWER_ONLINE_MONTHLY = ACCESS_POWER_ONLINE_MONTHLY  # Series (YYYY-MM -> kW)\n"
        "ACCESS_POWER_DICT = ACCESS_POWER_DICT                      # dict (YYYY-MM -> kW)\n",
        "# ACCESS_POWER_ONLINE_MONTHLY / ACCESS_POWER_DICT set in §1.2\n",
    )
    return source


def patch_baseline_access_block(source: str) -> str:
    """Replace duplicated cummax+HP access mapping with §1.2 export."""
    replacement = '''# Access power from §1.2 (baseline HP = grid cummax + margin + HP @ -10°C)
if "ACCESS_POWER_BASELINE_MONTHLY" not in globals():
    raise RuntimeError("Run §1.2 first to build ACCESS_POWER_BASELINE_MONTHLY.")

df_base["month_key"] = df_base["month"].astype(str)
df_base["access_kw"] = df_base["month_key"].map(ACCESS_POWER_BASELINE_MONTHLY.to_dict()).astype(float)

'''
    start = "# Conservative access power based on baseline grid peaks"
    end = 'df_base["month_key"] = df_base["month"].astype(str)\n'
    if start in source:
        i0 = source.index(start)
        i1 = source.index(end, i0) + len(end)
        return source[:i0] + replacement + source[i1:]

    marker = "MARGIN_KW = 20.0"
    end_marker = 'df_base["grid_consumption"] = df_base["grid_consumption_with_hp"]'
    i0 = source.find(marker)
    i1 = source.find(end_marker)
    if i0 >= 0 and i1 > i0:
        return source[:i0] + replacement + source[i1:]
    return source


patch_part32_baseline = patch_baseline_access_block


def _apply_post_patches(nb: dict) -> None:
    for i, c in enumerate(nb["cells"]):
        src = _lines_to_str(c.get("source", []))
        if c["cell_type"] == "code" and "Part 2 — Run HP-only Online MPC" in src:
            nb["cells"][i]["source"] = [ln + "\n" for ln in patch_part2(src).splitlines()]
        if c["cell_type"] == "code" and (
            "Part 3.2 — Billing comparison" in src or "Part 4D — Savings vs baseline" in src
        ):
            nb["cells"][i]["source"] = [
                ln + "\n" for ln in patch_baseline_access_block(src).splitlines()
            ]
        if c["cell_type"] == "markdown" and "## 2. Rolling-horizon online MPC" in src:
            extra = (
                "\n\n**Access power:** `ACCESS_POWER_ONLINE_MONTHLY` from §1.2 "
                "(flex-aware = cum-max `grid_consumption` + 20 kW). Run **§1.2** before Part 2.\n"
            )
            if "§1.2" not in src:
                nb["cells"][i]["source"] = [src.rstrip() + extra + "\n"]


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # Idempotent: if already restructured, only apply post-patches
    if any("§1.2 — Access power selection (grid-based, HP)" in _lines_to_str(c.get("source", [])) for c in cells):
        _apply_post_patches(nb)
        NB_PATH.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Post-patched {NB_PATH} ({len(nb['cells'])} cells)")
        return

    cells[0]["source"] = [line + "\n" for line in INTRO_MD.splitlines()]
    new_cells = [
        cells[0],
        _cell(MD_11, "markdown"),
        cells[1],
        _cell(MD_12, "markdown"),
        _cell(SECTION_12, "code"),
        _cell(MD_13, "markdown"),
        _cell(SECTION_13, "code"),
    ]
    new_cells.extend(cells[3:])
    nb["cells"] = new_cells
    _apply_post_patches(nb)
    NB_PATH.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Patched {NB_PATH} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
