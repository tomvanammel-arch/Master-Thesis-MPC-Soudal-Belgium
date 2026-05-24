"""§1.2/§1.3: flex-aware on grid peaks; conservative = grid cummax + margin + HP peak."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "10_online_MPC_1_HP.ipynb"

OLD_ACCESS = """# --- 2024 training peaks (grid + grid excl. EV) ---
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
).sort_index()"""

NEW_ACCESS = """# --- 2024 training peaks (grid_consumption only) ---
train_2024_path = PROJECT_ROOT / "data" / "plant1_2024_training.csv"
train_2024 = pd.read_csv(train_2024_path)
train_2024["month"] = pd.PeriodIndex(train_2024["timestamp"].astype(str).str.slice(0, 7), freq="M")
train_2024["grid_consumption"] = pd.to_numeric(train_2024["grid_consumption"], errors="coerce").fillna(0.0)
monthly_peak_2024_grid_kw = train_2024.groupby("month")["grid_consumption"].max() * 4.0
baseline_2024_peak_grid_kw = float(monthly_peak_2024_grid_kw.max())

# --- 2025 monthly peaks (grid_consumption) ---
_tmp_cols = ["timestamp", "grid_consumption", "thermal_load", "outdoor_temperature"]
tmp_2025 = plant_2025[[c for c in _tmp_cols if c in plant_2025.columns]].copy()
tmp_2025["month"] = tmp_2025["timestamp"].dt.to_period("M")
monthly_peak_2025_grid_kw = (
    tmp_2025.groupby("month")["grid_consumption"].max() * 4.0
).reindex(months).astype(float)

# --- Cum-max(grid, M-1) + margin ---
cummax_grid_Mm1_kw = monthly_peak_2025_grid_kw.cummax().shift(1)
cummax_grid_Mm1_kw.loc[months.min()] = baseline_2024_peak_grid_kw
cummax_grid_Mm1_kw = cummax_grid_Mm1_kw.fillna(baseline_2024_peak_grid_kw)

# Flex-aware (Part 2 online): grid peaks + margin
access_power_flex_aware_kw = cummax_grid_Mm1_kw + MARGIN_KW

# Conservative (Part 3.2 baseline / notebook 03): grid peaks + margin + worst-case HP electrical peak
access_power_conservative_kw = access_power_flex_aware_kw + hp_additional_peak_kw

# 2024 hourly access for headroom plots
months_2024 = monthly_peak_2024_grid_kw.sort_index().index
cummax_2024_grid_Mm1_kw = monthly_peak_2024_grid_kw.cummax().shift(1)
cummax_2024_grid_Mm1_kw.loc[months_2024.min()] = float(monthly_peak_2024_grid_kw.iloc[0])
cummax_2024_grid_Mm1_kw = cummax_2024_grid_Mm1_kw.fillna(float(monthly_peak_2024_grid_kw.iloc[0]))
access_power_flex_2024_kw = cummax_2024_grid_Mm1_kw + MARGIN_KW
access_power_conservative_2024_kw = access_power_flex_2024_kw + hp_additional_peak_kw
access_power_by_month_flex_hr = pd.concat(
    [access_power_flex_2024_kw, access_power_flex_aware_kw]
).sort_index()
access_power_by_month_conservative_hr = pd.concat(
    [access_power_conservative_2024_kw, access_power_conservative_kw]
).sort_index()"""

OLD_TABLE = """table = pd.DataFrame(
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
).T"""

NEW_TABLE = """table = pd.DataFrame(
    {
        "monthly_peak_grid_kw": monthly_peak_2025_grid_kw.values,
        "cummax_grid_kw_M_minus_1": cummax_grid_Mm1_kw.values,
        "access_power_flex_aware": access_power_flex_aware_kw.values,
        "access_power_conservative": access_power_conservative_kw.values,
        "access_power_deterministic": access_power_deterministic_kw.reindex(months.astype(str)).values,
    },
    index=months.astype(str),
).T"""

OLD_PLOT_ACC = """ax_acc.step(
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
)"""

NEW_PLOT_ACC = """ax_acc.step(
    month_ts,
    access_power_flex_aware_kw.values,
    where="post",
    color=C_BLACK,
    linewidth=LW_ACCESS,
    linestyle="-",
    label="Access flex-aware (grid)",
    zorder=3,
)
ax_acc.step(
    month_ts,
    access_power_conservative_kw.values,
    where="post",
    color=C_BLUE,
    linewidth=LW_ACCESS,
    linestyle="-",
    label="Access conservative (grid + HP @ -10°C)",
    zorder=3,
)"""

OLD_EXPORT = """ACCESS_POWER_BASELINE_MONTHLY = access_power_baseline_hp_kw.copy()
ACCESS_POWER_BASELINE_MONTHLY.index = ACCESS_POWER_BASELINE_MONTHLY.index.astype(str)
ACCESS_POWER_ONLINE_MONTHLY = access_power_flex_aware_kw.copy()
ACCESS_POWER_ONLINE_MONTHLY.index = ACCESS_POWER_ONLINE_MONTHLY.index.astype(str)
ACCESS_POWER_DICT = {str(k): float(v) for k, v in ACCESS_POWER_ONLINE_MONTHLY.items()}

print("ACCESS_POWER_ONLINE_MONTHLY (flex-aware, kW):")
display(ACCESS_POWER_ONLINE_MONTHLY.round(1))
print("ACCESS_POWER_BASELINE_MONTHLY (baseline HP, kW):")
display(ACCESS_POWER_BASELINE_MONTHLY.round(1))"""

NEW_EXPORT = """ACCESS_POWER_BASELINE_MONTHLY = access_power_conservative_kw.copy()
ACCESS_POWER_BASELINE_MONTHLY.index = ACCESS_POWER_BASELINE_MONTHLY.index.astype(str)
ACCESS_POWER_ONLINE_MONTHLY = access_power_flex_aware_kw.copy()
ACCESS_POWER_ONLINE_MONTHLY.index = ACCESS_POWER_ONLINE_MONTHLY.index.astype(str)
ACCESS_POWER_DICT = {str(k): float(v) for k, v in ACCESS_POWER_ONLINE_MONTHLY.items()}

print("ACCESS_POWER_ONLINE_MONTHLY (flex-aware, grid, kW):")
display(ACCESS_POWER_ONLINE_MONTHLY.round(1))
print("ACCESS_POWER_BASELINE_MONTHLY (conservative, grid + HP peak, kW):")
display(ACCESS_POWER_BASELINE_MONTHLY.round(1))"""

OLD_YLIM = """        access_power_conservative_kw.max(),
        access_power_baseline_hp_kw.max(),
        access_power_flex_aware_kw.max(),"""

NEW_YLIM = """        access_power_conservative_kw.max(),
        access_power_flex_aware_kw.max(),"""

MD_12 = r"""### 1.2 Access power selection (grid-based)

Monthly access from **cum-max of `grid_consumption` peaks** (kWh/15 min → kW via ×4) on `plant1.csv` (2025), January 2025 seeded from `plant1_2024_training.csv`, plus **+20 kW** margin.

**Flex-aware** (Part 2 online MPC):

\[
P_{\mathrm{access},M}^{\mathrm{flex}} = \max_{m < M}\bigl(\max_t 4\cdot\texttt{grid\_consumption}_t^{(m)}\bigr) + 20\ \mathrm{kW}
\]

**Conservative** (Part 3.2 baseline / notebook 03 — same grid peaks **+** worst-case HP electrical peak @ COP(−10°C)):

\[
P_{\mathrm{access},M}^{\mathrm{cons}} = P_{\mathrm{access},M}^{\mathrm{flex}} + P_{\mathrm{HP,elec}}^{\max}
\]

**Deterministic:** `access_power_kw` from notebook 03 export.

**Headroom** (full calendar day; load = `grid_consumption` for both):

- **Conservative:** \(P_{\mathrm{playroom}}(t) = \max(P_{\mathrm{access}}^{\mathrm{cons}}(\mathrm{month}) - 4\cdot\texttt{grid\_consumption}_t,\ 0)\)
- **Flex-aware:** \(P_{\mathrm{playroom}}(t) = \max(P_{\mathrm{access}}^{\mathrm{flex}}(\mathrm{month}) - 4\cdot\texttt{grid\_consumption}_t,\ 0)\)

Daily headroom: \(H_d = \sum_t P_{\mathrm{playroom}}(t) \times 0.25\) h (kWh/day).

**HP need (2025 only):** actual electrical HP from thermal load / COP(\(T_\mathrm{out}\)); worst-case reference = max daily sum at COP(−10°C).

Utilisation (%): actual and worst-case HP vs conservative / flex-aware headroom (70% reference).

**Exports:** `table`, `ACCESS_POWER_BASELINE_MONTHLY` (conservative), `ACCESS_POWER_ONLINE_MONTHLY` (flex-aware), `ACCESS_POWER_DICT`.
"""


def patch_13(src: str) -> str:
    old = """access_power_conservative_nopv_kw = cummax_grid_nopv_Mm1_kw + MARGIN_KW

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

    new = """access_power_flex_aware_nopv_kw = cummax_grid_nopv_Mm1_kw + MARGIN_KW
access_power_conservative_nopv_kw = access_power_flex_aware_nopv_kw + hp_additional_peak_kw
"""

    if old not in src:
        raise SystemExit("§1.3 flex nopv block not found")
    src = src.replace(old, new)

    old2 = """access_power_conservative_nopv_2024_kw = cummax_2024_nopv_Mm1_kw + MARGIN_KW
access_power_flex_nopv_2024_kw = cummax_2024_nopv_Mm1_kw + MARGIN_KW
access_power_baseline_hp_nopv_2024_kw = access_power_conservative_nopv_2024_kw + hp_additional_peak_kw
access_power_by_month_conservative_nopv_hr = pd.concat(
    [access_power_conservative_nopv_2024_kw, access_power_conservative_nopv_kw]
).sort_index()
access_power_by_month_flex_nopv_hr = pd.concat(
    [access_power_flex_nopv_2024_kw, access_power_flex_aware_nopv_kw]
).sort_index()
access_power_by_month_baseline_hp_nopv_hr = pd.concat(
    [access_power_baseline_hp_nopv_2024_kw, access_power_baseline_hp_nopv_kw]
).sort_index()"""

    new2 = """access_power_flex_nopv_2024_kw = cummax_2024_nopv_Mm1_kw + MARGIN_KW
access_power_conservative_nopv_2024_kw = access_power_flex_nopv_2024_kw + hp_additional_peak_kw
access_power_by_month_flex_nopv_hr = pd.concat(
    [access_power_flex_nopv_2024_kw, access_power_flex_aware_nopv_kw]
).sort_index()
access_power_by_month_conservative_nopv_hr = pd.concat(
    [access_power_conservative_nopv_2024_kw, access_power_conservative_nopv_kw]
).sort_index()"""

    if old2 not in src:
        raise SystemExit("§1.3 2024 nopv block not found")
    src = src.replace(old2, new2)

    old3 = """        "access_power_conservative_nopv": access_power_conservative_nopv_kw.values,
        "access_power_flex_aware_nopv": access_power_flex_aware_nopv_kw.values,
        "access_power_baseline_hp_nopv": access_power_baseline_hp_nopv_kw.values,"""
    new3 = """        "access_power_flex_aware_nopv": access_power_flex_aware_nopv_kw.values,
        "access_power_conservative_nopv": access_power_conservative_nopv_kw.values,"""
    src = src.replace(old3, new3)

    # fix order: conservative_nopv was set before flex in old broken block - check cell for duplicate
    old4 = """access_power_conservative_nopv_kw = cummax_grid_nopv_Mm1_kw + MARGIN_KW
access_power_flex_aware_nopv_kw = cummax_grid_nopv_Mm1_kw + MARGIN_KW
access_power_baseline_hp_nopv_kw = access_power_flex_aware_nopv_kw + hp_additional_peak_kw"""
    if old4 in src:
        src = src.replace(old4, new)

    src = src.replace(
        'label="Access flex-aware (no-PV)"',
        'label="Access flex-aware (no-PV, grid)"',
    )
    src = src.replace(
        "access_power_baseline_hp_nopv_kw.values",
        "access_power_conservative_nopv_kw.values",
    )
    src = src.replace(
        'label="Access baseline HP (no-PV)"',
        'label="Access conservative (no-PV, grid + HP)"',
    )
    return src


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    src = "".join(nb["cells"][4]["source"])
    for old, new in [
        (OLD_ACCESS, NEW_ACCESS),
        (OLD_TABLE, NEW_TABLE),
        (OLD_PLOT_ACC, NEW_PLOT_ACC),
        (OLD_EXPORT, NEW_EXPORT),
        (OLD_YLIM, NEW_YLIM),
    ]:
        if old not in src:
            raise SystemExit(f"§1.2 block not found: {old[:60]!r}")
        src = src.replace(old, new)
    nb["cells"][4]["source"] = [src]
    nb["cells"][3]["source"] = [MD_12]

    src13 = "".join(nb["cells"][6]["source"])
    nb["cells"][6]["source"] = [patch_13(src13)]

    # intro cell
    intro = "".join(nb["cells"][0]["source"])
    intro = intro.replace(
        "**conservative** and **flex-aware** = cum-max `grid_consumption` (M−1) + 20 kW (identical for HP); "
        "**baseline HP** adds worst-case HP electrical peak",
        "**flex-aware** = cum-max `grid_consumption` (M−1) + 20 kW; **conservative** adds worst-case HP electrical peak",
    )
    nb["cells"][0]["source"] = [intro]

    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Patched", NB)


if __name__ == "__main__":
    main()
