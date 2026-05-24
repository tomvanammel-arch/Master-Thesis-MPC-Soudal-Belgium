"""§1.2: flex-aware on excl_ev; dual HP need (actual COP + worst-day -10°C constant)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "10_online_MPC_1_HP.ipynb"

OLD_BLOCK_START = "# --- 2024 training peak (grid_consumption only) ---"
OLD_BLOCK_END = "# --- Exports for Parts 2–4 ---"

NEW_BLOCK = r'''# --- 2024 training peaks (grid + grid excl. EV) ---
train_2024_path = PROJECT_ROOT / "data" / "plant1_2024_training.csv"
train_2024 = pd.read_csv(train_2024_path)
train_2024["month"] = pd.PeriodIndex(train_2024["timestamp"].astype(str).str.slice(0, 7), freq="M")
for _c in ["grid_consumption", "grid_consumption_excl_ev"]:
    train_2024[_c] = pd.to_numeric(train_2024[_c], errors="coerce").fillna(0.0)
monthly_peak_2024_grid_kw = train_2024.groupby("month")["grid_consumption"].max() * 4.0
baseline_2024_peak_grid_kw = float(monthly_peak_2024_grid_kw.max())
monthly_peak_2024_excl_ev_kw = train_2024.groupby("month")["grid_consumption_excl_ev"].max() * 4.0
baseline_2024_peak_excl_ev_kw = float(monthly_peak_2024_excl_ev_kw.max())

# --- 2025 monthly peaks ---
_tmp_cols = ["timestamp", "grid_consumption", "grid_consumption_excl_ev", "thermal_load", "outdoor_temperature"]
tmp_2025 = plant_2025[[c for c in _tmp_cols if c in plant_2025.columns]].copy()
tmp_2025["month"] = tmp_2025["timestamp"].dt.to_period("M")
monthly_peak_2025_grid_kw = (
    tmp_2025.groupby("month")["grid_consumption"].max() * 4.0
).reindex(months).astype(float)
monthly_peak_2025_excl_ev_kw = (
    tmp_2025.groupby("month")["grid_consumption_excl_ev"].max() * 4.0
).reindex(months).astype(float)

# --- Conservative: cum-max(grid, M-1) + margin ---
cummax_grid_Mm1_kw = monthly_peak_2025_grid_kw.cummax().shift(1)
cummax_grid_Mm1_kw.loc[months.min()] = baseline_2024_peak_grid_kw
cummax_grid_Mm1_kw = cummax_grid_Mm1_kw.fillna(baseline_2024_peak_grid_kw)
access_power_conservative_kw = cummax_grid_Mm1_kw + MARGIN_KW

# --- Flex-aware: cum-max(grid_excl_ev, M-1) + margin (notebook 09) ---
cummax_excl_ev_Mm1_kw = monthly_peak_2025_excl_ev_kw.cummax().shift(1)
cummax_excl_ev_Mm1_kw.loc[months.min()] = baseline_2024_peak_excl_ev_kw
cummax_excl_ev_Mm1_kw = cummax_excl_ev_Mm1_kw.fillna(baseline_2024_peak_excl_ev_kw)
access_power_flex_aware_kw = cummax_excl_ev_Mm1_kw + MARGIN_KW

# --- Baseline HP: conservative grid part + worst-case HP electrical peak (notebook 03) ---
access_power_baseline_hp_kw = access_power_conservative_kw + hp_additional_peak_kw

# 2024 hourly access for headroom plots
months_2024 = monthly_peak_2024_grid_kw.sort_index().index
cummax_2024_grid_Mm1_kw = monthly_peak_2024_grid_kw.cummax().shift(1)
cummax_2024_grid_Mm1_kw.loc[months_2024.min()] = float(monthly_peak_2024_grid_kw.iloc[0])
cummax_2024_grid_Mm1_kw = cummax_2024_grid_Mm1_kw.fillna(float(monthly_peak_2024_grid_kw.iloc[0]))
cummax_2024_excl_ev_Mm1_kw = monthly_peak_2024_excl_ev_kw.cummax().shift(1)
cummax_2024_excl_ev_Mm1_kw.loc[months_2024.min()] = float(monthly_peak_2024_excl_ev_kw.iloc[0])
cummax_2024_excl_ev_Mm1_kw = cummax_2024_excl_ev_Mm1_kw.fillna(float(monthly_peak_2024_excl_ev_kw.iloc[0]))
access_power_conservative_2024_kw = cummax_2024_grid_Mm1_kw + MARGIN_KW
access_power_flex_2024_kw = cummax_2024_excl_ev_Mm1_kw + MARGIN_KW
access_power_baseline_hp_2024_kw = access_power_conservative_2024_kw + hp_additional_peak_kw
access_power_by_month_conservative_hr = pd.concat(
    [access_power_conservative_2024_kw, access_power_conservative_kw]
).sort_index()
access_power_by_month_flex_hr = pd.concat(
    [access_power_flex_2024_kw, access_power_flex_aware_kw]
).sort_index()
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
        "monthly_peak_excl_ev_kw": monthly_peak_2025_excl_ev_kw.values,
        "cummax_grid_kw_M_minus_1": cummax_grid_Mm1_kw.values,
        "cummax_excl_ev_kw_M_minus_1": cummax_excl_ev_Mm1_kw.values,
        "access_power_conservative": access_power_conservative_kw.values,
        "access_power_flex_aware": access_power_flex_aware_kw.values,
        "access_power_baseline_hp": access_power_baseline_hp_kw.values,
        "access_power_deterministic": access_power_deterministic_kw.reindex(months.astype(str)).values,
    },
    index=months.astype(str),
).T
display(table.round(1))

# --- plant_hr for headroom (2024–2025 grid; 2025 HP need) ---
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

from heat_pump_load import load_hp_config, interpolate_cop

if "hp_cfg" not in globals():
    hp_cfg = load_hp_config(str(PROJECT_ROOT / "config" / "hp.yaml"))
if "cop_minus10" not in globals():
    cop_minus10 = float(interpolate_cop(-10.0, hp_cfg["COP_data"]))


def _cop_from_temp(temp_c: float, cop_data: dict) -> float:
    if temp_c is None or (isinstance(temp_c, float) and np.isnan(temp_c)):
        return cop_minus10
    return float(interpolate_cop(float(temp_c), cop_data))


plant_hr_hp = plant_2025[["timestamp", "thermal_load", "outdoor_temperature"]].copy()
plant_hr_hp["thermal_load"] = pd.to_numeric(plant_hr_hp["thermal_load"], errors="coerce").fillna(0.0)
plant_hr_hp["outdoor_temperature"] = pd.to_numeric(
    plant_hr_hp["outdoor_temperature"], errors="coerce"
)
plant_hr_hp["cop_actual"] = plant_hr_hp["outdoor_temperature"].map(
    lambda t: _cop_from_temp(t, hp_cfg["COP_data"])
)
plant_hr_hp["hp_need_actual_kwh_15"] = plant_hr_hp["thermal_load"] / plant_hr_hp["cop_actual"]
plant_hr_hp["hp_need_minus10_kwh_15"] = plant_hr_hp["thermal_load"] / cop_minus10


def _daily_headroom_kwh(ts, load_kwh_15, access_by_month):
    access_kw = access_by_month.reindex(ts.dt.to_period("M")).to_numpy(dtype=float)
    headroom_kw = np.maximum(access_kw - 4.0 * np.asarray(load_kwh_15, dtype=float), 0.0)
    return (
        pd.DataFrame({"date": ts.dt.normalize(), "h": headroom_kw * 0.25})
        .groupby("date")["h"]
        .sum()
    )


def _daily_hp_kwh(hp_df: pd.DataFrame, col: str) -> pd.Series:
    return (
        pd.DataFrame({"date": hp_df["timestamp"].dt.normalize(), "hp": hp_df[col]})
        .groupby("date")["hp"]
        .sum()
    )


ts_hr = plant_hr_grid["timestamp"]
grid_kwh = plant_hr_grid["grid_consumption"].to_numpy(dtype=float)
daily_h_cons = _daily_headroom_kwh(ts_hr, grid_kwh, access_power_by_month_conservative_hr)
daily_h_flex = _daily_headroom_kwh(ts_hr, grid_kwh, access_power_by_month_flex_hr)

daily_hp_need_actual_kwh = _daily_hp_kwh(plant_hr_hp, "hp_need_actual_kwh_15").reindex(
    daily_h_cons.index
)
daily_hp_need_minus10_kwh = _daily_hp_kwh(plant_hr_hp, "hp_need_minus10_kwh_15")
hp_need_worst_day_kwh = float(daily_hp_need_minus10_kwh.max())
_worst_day = daily_hp_need_minus10_kwh.idxmax()
print(
    f"Worst-case daily HP electrical need @ COP(-10°C): {hp_need_worst_day_kwh:.1f} kWh/d "
    f"(thermal peak day {_worst_day.date()})"
)

hr_dates = pd.to_datetime(daily_h_cons.index)
util_cons_pct = np.where(
    daily_h_cons > 1e-9, 100.0 * daily_hp_need_actual_kwh / daily_h_cons, np.nan
)
util_flex_pct = np.where(
    daily_h_flex > 1e-9, 100.0 * daily_hp_need_actual_kwh / daily_h_flex, np.nan
)

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
    access_power_conservative_kw.values,
    where="post",
    color=C_BLUE,
    linewidth=LW_ACCESS,
    linestyle="-",
    label="Access conservative (grid)",
    zorder=3,
)
ax_acc.step(
    month_ts,
    access_power_flex_aware_kw.values,
    where="post",
    color=C_BLACK,
    linewidth=LW_ACCESS,
    linestyle="-",
    label="Access flex-aware (grid excl. EV)",
    zorder=3,
)
ax_acc.step(
    month_ts,
    access_power_baseline_hp_kw.values,
    where="post",
    color=C_ORANGE,
    linewidth=LW_ACCESS,
    linestyle="--",
    label="Access baseline HP (+ HP @ -10°C)",
    zorder=3,
)
ax_acc.step(
    month_ts,
    access_power_deterministic_kw.reindex(months.astype(str)).values,
    where="post",
    color=C_GREEN,
    linewidth=LW_ACCESS,
    linestyle="-",
    label="Access deterministic (notebook 03)",
    zorder=3,
)
ax_acc.set_title("§1.2 Monthly access power (2025)")
ax_acc.set_ylabel("kW")
ax_acc.set_xlabel("Month")
ax_acc.set_ylim(
    2200,
    max(
        access_power_conservative_kw.max(),
        access_power_baseline_hp_kw.max(),
        access_power_flex_aware_kw.max(),
        access_power_deterministic_kw.max(),
        monthly_peak_2025_grid_kw.max(),
    )
    + 120,
)
ax_acc.grid(True, axis="y", linestyle="--", alpha=0.35)
ax_acc.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8, framealpha=0.95)
fig_acc.subplots_adjust(bottom=0.32)
plt.show()

fig_hr, ax_hr = plt.subplots(figsize=(14, 4.5))
ax_hr.step(
    hr_dates, daily_h_cons.values, where="post", color=C_BLUE, linewidth=LW_DAILY,
    label="Headroom (conservative)",
)
ax_hr.step(
    hr_dates, daily_h_flex.values, where="post", color=C_GREEN, linewidth=LW_DAILY,
    label="Headroom (flex-aware)",
)
ax_hr.step(
    hr_dates,
    daily_hp_need_actual_kwh.values,
    where="post",
    color=C_RED,
    linewidth=LW_DAILY,
    label="Daily HP need (actual COP)",
)
ax_hr.axhline(
    hp_need_worst_day_kwh,
    color=C_RED,
    linewidth=LW_DAILY,
    linestyle="--",
    label=f"Worst day @ COP(-10°C) ({hp_need_worst_day_kwh:.0f} kWh/d)",
)
ax_hr.set_ylabel("Energy [kWh/day]")
ax_hr.set_xlabel("Date (2024–2025)")
ax_hr.set_title("§1.2 Daily headroom vs HP need (full day, 2025 HP curves)")
ax_hr.grid(True, axis="y", linestyle="--", alpha=0.35)
ax_hr.legend(loc="upper right", fontsize=9, framealpha=0.95)
plt.tight_layout()
plt.show()

fig_u, ax_u = plt.subplots(figsize=(14, 4.5))
ax_u.step(
    hr_dates, util_cons_pct, where="post", color=C_BLUE, linewidth=LW_DAILY,
    label="Actual HP / H (conservative)",
)
ax_u.step(
    hr_dates, util_flex_pct, where="post", color=C_GREEN, linewidth=LW_DAILY, linestyle="--",
    label="Actual HP / H (flex-aware)",
)
ax_u.axhline(70.0, color=C_BLACK, linewidth=1.5, linestyle="-", label="70% reference")
ax_u.set_ylabel("Utilisation [%]")
ax_u.set_xlabel("Date (2024–2025)")
ax_u.set_title("§1.2 Daily headroom utilisation — actual HP electrical need")
ax_u.grid(True, axis="y", linestyle="--", alpha=0.35)
ax_u.legend(loc="upper right", fontsize=9, framealpha=0.95)
plt.tight_layout()
plt.show()

'''

MD_12 = r"""### 1.2 Access power selection (grid-based)

Two **cum-max + 20 kW** rules on metered data (`plant1.csv` 2025 + `plant1_2024_training.csv`), January 2025 seeded from 2024 — same structure as **notebook 09**.

**Conservative:** \(\max_{m<M}(\max_t 4\cdot\texttt{grid\_consumption}) + 20\) kW  

**Flex-aware** (Part 2 online contract): \(\max_{m<M}(\max_t 4\cdot\texttt{grid\_consumption\_excl\_ev}) + 20\) kW  

**Baseline HP** (Part 3.2, notebook 03): conservative grid part **+** worst-case HP electrical peak @ COP(−10°C).

**Headroom** (full day, playroom load = `grid_consumption`): \(H_d = \sum_t \max(P_{\mathrm{access}} - 4\cdot\texttt{grid}, 0)\times 0.25\) h.

**HP need (2025 only):**
- **Actual:** \(\sum_t \texttt{thermal\_load}_t / \mathrm{COP}(T^{\mathrm{out}}_t)\) per day (outdoor temperature from `plant1.csv`).
- **Worst-case reference (horizontal):** maximum daily sum with **COP(−10°C)** fixed — the single worst thermal day at −10°C COP.

Utilisation plots use **actual** daily HP need vs conservative / flex-aware headroom (70% reference).

**Exports:** `table`, `ACCESS_POWER_BASELINE_MONTHLY` (baseline HP), `ACCESS_POWER_ONLINE_MONTHLY` (flex-aware), `ACCESS_POWER_DICT`.
"""


def patch_cell_12(src: str) -> str:
    if OLD_BLOCK_START not in src:
        raise SystemExit("§1.2 block start not found")
    if OLD_BLOCK_END not in src:
        raise SystemExit("§1.2 block end not found")
    return src[: src.index(OLD_BLOCK_START)] + NEW_BLOCK + src[src.index(OLD_BLOCK_END) :]


def patch_cell_13(src: str) -> str:
    """Mirror HP need dual curves in §1.3; fix util denominator bug."""
    old = """plant_hr_nopv_hp = plant_2025_nopv[["timestamp", "thermal_load"]].copy()
plant_hr_nopv_hp["thermal_load"] = pd.to_numeric(
    plant_hr_nopv_hp["thermal_load"], errors="coerce"
).fillna(0.0)
plant_hr_nopv_hp["hp_need_kwh_15"] = plant_hr_nopv_hp["thermal_load"] / cop_minus10

ts_hr = plant_hr_nopv_grid["timestamp"]
grid_nopv_kwh = plant_hr_nopv_grid["grid_nopv"].to_numpy(dtype=float)
daily_h_cons_nopv = _daily_headroom_kwh(ts_hr, grid_nopv_kwh, access_power_by_month_conservative_nopv_hr)
daily_h_flex_nopv = _daily_headroom_kwh(ts_hr, grid_nopv_kwh, access_power_by_month_flex_nopv_hr)
daily_hp_need_kwh_nopv = (
    pd.DataFrame(
        {"date": plant_hr_nopv_hp["timestamp"].dt.normalize(), "hp": plant_hr_nopv_hp["hp_need_kwh_15"]}
    )
    .groupby("date")["hp"]
    .sum()
    .reindex(daily_h_cons_nopv.index)
)
hr_dates_nopv = pd.to_datetime(daily_h_cons_nopv.index)
util_cons_nopv_pct = np.where(
    daily_h_cons_nopv > 1e-9, 100.0 * daily_hp_need_kwh_nopv / daily_h_base_nopv, np.nan
)
util_flex_nopv_pct = np.where(
    daily_h_flex_nopv > 1e-9, 100.0 * daily_hp_need_kwh_nopv / daily_h_flex_nopv, np.nan
)"""

    new = """ts_hr = plant_hr_nopv_grid["timestamp"]
grid_nopv_kwh = plant_hr_nopv_grid["grid_nopv"].to_numpy(dtype=float)
daily_h_cons_nopv = _daily_headroom_kwh(ts_hr, grid_nopv_kwh, access_power_by_month_conservative_nopv_hr)
daily_h_flex_nopv = _daily_headroom_kwh(ts_hr, grid_nopv_kwh, access_power_by_month_flex_nopv_hr)

if "daily_hp_need_actual_kwh" not in globals() or "hp_need_worst_day_kwh" not in globals():
    raise RuntimeError("Run §1.2 first (daily_hp_need_actual_kwh, hp_need_worst_day_kwh).")
daily_hp_need_actual_kwh_nopv = daily_hp_need_actual_kwh.reindex(daily_h_cons_nopv.index)
hp_need_worst_day_kwh_nopv = hp_need_worst_day_kwh

hr_dates_nopv = pd.to_datetime(daily_h_cons_nopv.index)
util_cons_nopv_pct = np.where(
    daily_h_cons_nopv > 1e-9, 100.0 * daily_hp_need_actual_kwh_nopv / daily_h_cons_nopv, np.nan
)
util_flex_nopv_pct = np.where(
    daily_h_flex_nopv > 1e-9, 100.0 * daily_hp_need_actual_kwh_nopv / daily_h_flex_nopv, np.nan
)"""

    if old not in src:
        raise SystemExit("§1.3 HP need block not found")
    src = src.replace(old, new)

    # flex nopv from excl_ev_nopv peaks (notebook 09)
    old_acc = """access_power_conservative_nopv_kw = cummax_grid_nopv_Mm1_kw + MARGIN_KW
access_power_flex_aware_nopv_kw = cummax_grid_nopv_Mm1_kw + MARGIN_KW
access_power_baseline_hp_nopv_kw = access_power_flex_aware_nopv_kw + hp_additional_peak_kw"""
    new_acc = """access_power_conservative_nopv_kw = cummax_grid_nopv_Mm1_kw + MARGIN_KW

def _add_excl_ev_nopv(df: pd.DataFrame) -> pd.DataFrame:
    out = _add_nopv_columns(df)
    if "grid_consumption_excl_ev" in out.columns:
        if "grid_injection" in out.columns:
            out["grid_excl_ev_nopv"] = (
                out["grid_consumption_excl_ev"] + out["pv_production"] - out["grid_injection"]
            )
        else:
            out["grid_excl_ev_nopv"] = out["inflex_load"]
    return out

train_2024 = _add_excl_ev_nopv(train_2024)
monthly_peak_2024_excl_ev_nopv_kw = train_2024.groupby("month")["grid_excl_ev_nopv"].max() * 4.0
baseline_2024_peak_excl_ev_nopv_kw = float(monthly_peak_2024_excl_ev_nopv_kw.max())

_tmp_nopv = _add_excl_ev_nopv(plant.loc[plant["timestamp"].dt.year == 2025].copy())
_tmp_nopv["month"] = _tmp_nopv["timestamp"].dt.to_period("M")
monthly_peak_2025_excl_ev_nopv_kw = (
    _tmp_nopv.groupby("month")["grid_excl_ev_nopv"].max() * 4.0
).reindex(months).astype(float)

cummax_excl_ev_nopv_Mm1_kw = monthly_peak_2025_excl_ev_nopv_kw.cummax().shift(1)
cummax_excl_ev_nopv_Mm1_kw.loc[months.min()] = baseline_2024_peak_excl_ev_nopv_kw
cummax_excl_ev_nopv_Mm1_kw = cummax_excl_ev_nopv_Mm1_kw.fillna(baseline_2024_peak_excl_ev_nopv_kw)
access_power_flex_aware_nopv_kw = cummax_excl_ev_nopv_Mm1_kw + MARGIN_KW
access_power_baseline_hp_nopv_kw = access_power_conservative_nopv_kw + hp_additional_peak_kw"""

    if old_acc in src:
        src = src.replace(old_acc, new_acc)
    else:
        raise SystemExit("§1.3 access block not found")

    old_2024 = """access_power_flex_nopv_2024_kw = cummax_2024_nopv_Mm1_kw + MARGIN_KW
access_power_baseline_hp_nopv_2024_kw = access_power_flex_nopv_2024_kw + hp_additional_peak_kw
access_power_by_month_conservative_nopv_hr = pd.concat(
    [access_power_conservative_nopv_2024_kw, access_power_conservative_nopv_kw]
).sort_index()
access_power_by_month_flex_nopv_hr = pd.concat(
    [access_power_flex_nopv_2024_kw, access_power_flex_aware_nopv_kw]
).sort_index()"""

    new_2024 = """cummax_2024_excl_ev_nopv_Mm1_kw = monthly_peak_2024_excl_ev_nopv_kw.cummax().shift(1)
cummax_2024_excl_ev_nopv_Mm1_kw.loc[months_2024.min()] = float(
    monthly_peak_2024_excl_ev_nopv_kw.iloc[0]
)
cummax_2024_excl_ev_nopv_Mm1_kw = cummax_2024_excl_ev_nopv_Mm1_kw.fillna(
    float(monthly_peak_2024_excl_ev_nopv_kw.iloc[0])
)
access_power_flex_nopv_2024_kw = cummax_2024_excl_ev_nopv_Mm1_kw + MARGIN_KW
access_power_baseline_hp_nopv_2024_kw = access_power_conservative_nopv_2024_kw + hp_additional_peak_kw
access_power_by_month_conservative_nopv_hr = pd.concat(
    [access_power_conservative_nopv_2024_kw, access_power_conservative_nopv_kw]
).sort_index()
access_power_by_month_flex_nopv_hr = pd.concat(
    [access_power_flex_nopv_2024_kw, access_power_flex_aware_nopv_kw]
).sort_index()"""

    if old_2024 in src:
        src = src.replace(old_2024, new_2024)

    # fix duplicate conservative assignment if patch added conservative_nopv twice
    if "access_power_conservative_nopv_kw = cummax_grid_nopv_Mm1_kw" in src and src.count(
        "access_power_conservative_nopv_kw = cummax_grid_nopv_Mm1_kw"
    ) > 1:
        pass

    old_plot = """ax_hr_nopv.step(hr_dates_nopv, daily_h_cons_nopv.values, where="post", color=C_BLUE, linewidth=LW_DAILY, label="Headroom (conservative, no PV)")
ax_hr_nopv.step(hr_dates_nopv, daily_h_flex_nopv.values, where="post", color=C_GREEN, linewidth=LW_DAILY, label="Headroom (flex-aware, no PV)")
ax_hr_nopv.step(hr_dates_nopv, daily_hp_need_kwh_nopv.values, where="post", color=C_RED, linewidth=LW_DAILY, label="Daily HP need @ -10°C")"""
    new_plot = """ax_hr_nopv.step(hr_dates_nopv, daily_h_cons_nopv.values, where="post", color=C_BLUE, linewidth=LW_DAILY, label="Headroom (conservative, no PV)")
ax_hr_nopv.step(hr_dates_nopv, daily_h_flex_nopv.values, where="post", color=C_GREEN, linewidth=LW_DAILY, label="Headroom (flex-aware, no PV)")
ax_hr_nopv.step(hr_dates_nopv, daily_hp_need_actual_kwh_nopv.values, where="post", color=C_RED, linewidth=LW_DAILY, label="Daily HP need (actual COP)")
ax_hr_nopv.axhline(hp_need_worst_day_kwh_nopv, color=C_RED, linewidth=LW_DAILY, linestyle="--", label=f"Worst day @ COP(-10°C) ({hp_need_worst_day_kwh_nopv:.0f} kWh/d)")"""
    if old_plot in src:
        src = src.replace(old_plot, new_plot)

    return src


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))

    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        if c["cell_type"] == "markdown" and "### 1.2 Access power selection" in src:
            nb["cells"][i]["source"] = [MD_12]
        if c["cell_type"] == "code" and src.startswith("# §1.2 — Access power"):
            nb["cells"][i]["source"] = [patch_cell_12(src)]
        if c["cell_type"] == "code" and src.startswith("# §1.3 — Access power"):
            nb["cells"][i]["source"] = [patch_cell_13(src)]

    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Patched", NB)


if __name__ == "__main__":
    main()
