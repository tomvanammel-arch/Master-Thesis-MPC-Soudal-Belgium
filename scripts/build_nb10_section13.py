"""Insert §1.3 no-PV cells into notebook 10 (mirror §1.2)."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "10_online_MPC_1_HP.ipynb"

MD_13 = r"""### 1.3 Access power selection (no-PV counterfactual)

Same structure as **§1.2**, but monthly peaks and headroom use **no-PV site load** (kWh/15 min):

\[
\texttt{grid\_nopv} = \texttt{grid\_consumption} + \texttt{pv\_production} - \texttt{grid\_injection}
\]

(2024 training CSV has no `grid_injection`; there we use `inflex_load + ev`.)

**Flex-aware:** \(\max_{m<M}(\max_t 4\cdot\texttt{grid\_nopv}) + 20\) kW  

**Conservative:** flex-aware **+** worst-case HP electrical peak @ COP(−10°C).

**Headroom** (full day; playroom load = `grid_nopv`): same formulas as §1.2 with \(P_{\mathrm{access}}\) from this section and load `grid_nopv`.

HP need curves reuse §1.2 daily electrical HP (same thermal profile; 2025 only).

**Not wired to Part 2** — validation only. Run **§1.2** before this cell.
"""

CODE_13 = r'''# §1.3 — Access power selection (no-PV counterfactual, HP)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path("..").resolve()
MARGIN_KW = 20.0
months = pd.period_range("2025-01", "2025-12", freq="M")

if "plant" not in globals():
    raise RuntimeError("Run §1.1 / §1.2 first so `plant` is defined.")
if "hp_additional_peak_kw" not in globals():
    raise RuntimeError("Run §1.2 first (hp_additional_peak_kw).")


def _add_grid_nopv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["grid_consumption", "pv_production", "grid_injection", "inflex_load", "ev"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    if "grid_injection" in out.columns:
        out["grid_nopv"] = (
            out["grid_consumption"] + out["pv_production"] - out["grid_injection"]
        )
    else:
        out["grid_nopv"] = out["inflex_load"] + out["ev"]
    return out


# --- 2024 training peaks (grid_nopv) ---
train_2024_path = PROJECT_ROOT / "data" / "plant1_2024_training.csv"
train_2024 = _add_grid_nopv(pd.read_csv(train_2024_path))
train_2024["month"] = pd.PeriodIndex(train_2024["timestamp"].astype(str).str.slice(0, 7), freq="M")
monthly_peak_2024_grid_nopv_kw = train_2024.groupby("month")["grid_nopv"].max() * 4.0
baseline_2024_peak_grid_nopv_kw = float(monthly_peak_2024_grid_nopv_kw.max())

# --- 2025 monthly peaks (grid_nopv) ---
_pv_cols = [c for c in ["pv_production", "grid_injection"] if c in plant.columns]
_tmp_cols = ["timestamp", "grid_consumption"] + _pv_cols
if "inflex_load" in plant.columns and "ev" in plant.columns:
    _tmp_cols += ["inflex_load", "ev"]
plant_2025_nopv = _add_grid_nopv(plant.loc[plant["timestamp"].dt.year == 2025, _tmp_cols].copy())
plant_2025_nopv["month"] = plant_2025_nopv["timestamp"].dt.to_period("M")
monthly_peak_2025_grid_nopv_kw = (
    plant_2025_nopv.groupby("month")["grid_nopv"].max() * 4.0
).reindex(months).astype(float)

cummax_grid_nopv_Mm1_kw = monthly_peak_2025_grid_nopv_kw.cummax().shift(1)
cummax_grid_nopv_Mm1_kw.loc[months.min()] = baseline_2024_peak_grid_nopv_kw
cummax_grid_nopv_Mm1_kw = cummax_grid_nopv_Mm1_kw.fillna(baseline_2024_peak_grid_nopv_kw)

access_power_flex_aware_nopv_kw = cummax_grid_nopv_Mm1_kw + MARGIN_KW
access_power_conservative_nopv_kw = access_power_flex_aware_nopv_kw + hp_additional_peak_kw

months_2024 = monthly_peak_2024_grid_nopv_kw.sort_index().index
cummax_2024_nopv_Mm1_kw = monthly_peak_2024_grid_nopv_kw.cummax().shift(1)
cummax_2024_nopv_Mm1_kw.loc[months_2024.min()] = float(monthly_peak_2024_grid_nopv_kw.iloc[0])
cummax_2024_nopv_Mm1_kw = cummax_2024_nopv_Mm1_kw.fillna(float(monthly_peak_2024_grid_nopv_kw.iloc[0]))
access_power_flex_nopv_2024_kw = cummax_2024_nopv_Mm1_kw + MARGIN_KW
access_power_conservative_nopv_2024_kw = access_power_flex_nopv_2024_kw + hp_additional_peak_kw
access_power_by_month_flex_nopv_hr = pd.concat(
    [access_power_flex_nopv_2024_kw, access_power_flex_aware_nopv_kw]
).sort_index()
access_power_by_month_conservative_nopv_hr = pd.concat(
    [access_power_conservative_nopv_2024_kw, access_power_conservative_nopv_kw]
).sort_index()

if "access_power_deterministic_kw" not in globals():
    raise RuntimeError("Run §1.2 first (access_power_deterministic_kw).")

table_nopv = pd.DataFrame(
    {
        "monthly_peak_grid_nopv_kw": monthly_peak_2025_grid_nopv_kw.values,
        "cummax_grid_nopv_M_minus_1": cummax_grid_nopv_Mm1_kw.values,
        "access_power_flex_aware_nopv": access_power_flex_aware_nopv_kw.values,
        "access_power_conservative_nopv": access_power_conservative_nopv_kw.values,
        "access_power_deterministic": access_power_deterministic_kw.reindex(months.astype(str)).values,
    },
    index=months.astype(str),
).T
display(table_nopv.round(1))

# --- Hourly series for headroom (2024–2025 grid_nopv) ---
train_2024_ts = pd.to_datetime(train_2024["timestamp"], utc=True, errors="coerce")
train_2024_ts = train_2024_ts.dt.tz_convert("Europe/Brussels").dt.tz_localize(None)
plant_hr_nopv_grid = pd.concat(
    [
        train_2024.assign(timestamp=train_2024_ts)[["timestamp", "grid_nopv"]],
        plant_2025_nopv[["timestamp", "grid_nopv"]],
    ],
    ignore_index=True,
).sort_values("timestamp")
plant_hr_nopv_grid["grid_nopv"] = pd.to_numeric(
    plant_hr_nopv_grid["grid_nopv"], errors="coerce"
).fillna(0.0)


def _daily_headroom_kwh(ts, load_kwh_15, access_by_month):
    access_kw = access_by_month.reindex(ts.dt.to_period("M")).to_numpy(dtype=float)
    headroom_kw = np.maximum(access_kw - 4.0 * np.asarray(load_kwh_15, dtype=float), 0.0)
    return (
        pd.DataFrame({"date": ts.dt.normalize(), "h": headroom_kw * 0.25})
        .groupby("date")["h"]
        .sum()
    )


if "daily_hp_need_actual_kwh" not in globals() or "hp_need_worst_day_kwh" not in globals():
    raise RuntimeError("Run §1.2 first (daily_hp_need_actual_kwh, hp_need_worst_day_kwh).")

ts_hr = plant_hr_nopv_grid["timestamp"]
grid_nopv_kwh = plant_hr_nopv_grid["grid_nopv"].to_numpy(dtype=float)

daily_h_cons_nopv = _daily_headroom_kwh(
    ts_hr, grid_nopv_kwh, access_power_by_month_conservative_nopv_hr
)
daily_h_flex_nopv = _daily_headroom_kwh(ts_hr, grid_nopv_kwh, access_power_by_month_flex_nopv_hr)

daily_hp_need_actual_kwh_nopv = daily_hp_need_actual_kwh.reindex(daily_h_cons_nopv.index)
hp_need_worst_day_kwh_nopv = float(hp_need_worst_day_kwh)

hr_dates_nopv = pd.to_datetime(daily_h_cons_nopv.index)
util_cons_nopv_pct = np.where(
    daily_h_cons_nopv > 1e-9,
    100.0 * daily_hp_need_actual_kwh_nopv / daily_h_cons_nopv,
    np.nan,
)
util_flex_nopv_pct = np.where(
    daily_h_flex_nopv > 1e-9,
    100.0 * daily_hp_need_actual_kwh_nopv / daily_h_flex_nopv,
    np.nan,
)
util_worst_cons_nopv_pct = np.where(
    daily_h_cons_nopv > 1e-9,
    100.0 * hp_need_worst_day_kwh_nopv / daily_h_cons_nopv,
    np.nan,
)
util_worst_flex_nopv_pct = np.where(
    daily_h_flex_nopv > 1e-9,
    100.0 * hp_need_worst_day_kwh_nopv / daily_h_flex_nopv,
    np.nan,
)

C_BLACK = "#000000"
C_RED = "#d62728"
C_BLUE = "#1f77b4"
C_GREEN = "#2ca02c"
LW_ACCESS, LW_DAILY = 2.0, 1.5
month_ts = months.to_timestamp()

fig_acc_nopv, ax_acc_nopv = plt.subplots(figsize=(10, 5))
ax_acc_nopv.bar(
    month_ts,
    monthly_peak_2025_grid_nopv_kw.values,
    width=22,
    align="center",
    color=C_GREEN,
    alpha=0.45,
    edgecolor=C_GREEN,
    linewidth=0.6,
    label="Actual monthly peak (grid no-PV)",
    zorder=1,
)
ax_acc_nopv.step(
    month_ts,
    access_power_flex_aware_nopv_kw.values,
    where="post",
    color=C_BLACK,
    linewidth=LW_ACCESS,
    linestyle="-",
    label="Access flex-aware (no-PV)",
    zorder=3,
)
ax_acc_nopv.step(
    month_ts,
    access_power_conservative_nopv_kw.values,
    where="post",
    color=C_BLUE,
    linewidth=LW_ACCESS,
    linestyle="-",
    label="Access conservative (no-PV + HP @ -10°C)",
    zorder=3,
)
ax_acc_nopv.step(
    month_ts,
    access_power_deterministic_kw.reindex(months.astype(str)).values,
    where="post",
    color=C_GREEN,
    linewidth=LW_ACCESS,
    linestyle="-",
    label="Access deterministic (notebook 03, grid)",
    zorder=3,
)
ax_acc_nopv.set_title("§1.3 Monthly access power (2025, no-PV)")
ax_acc_nopv.set_ylabel("kW")
ax_acc_nopv.set_xlabel("Month")
ax_acc_nopv.set_ylim(
    2200,
    max(
        access_power_conservative_nopv_kw.max(),
        access_power_flex_aware_nopv_kw.max(),
        access_power_deterministic_kw.max(),
        monthly_peak_2025_grid_nopv_kw.max(),
    )
    + 120,
)
ax_acc_nopv.grid(True, axis="y", linestyle="--", alpha=0.35)
ax_acc_nopv.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8, framealpha=0.95)
fig_acc_nopv.subplots_adjust(bottom=0.32)
plt.show()

fig_hr_nopv, ax_hr_nopv = plt.subplots(figsize=(14, 4.5))
ax_hr_nopv.step(
    hr_dates_nopv,
    daily_h_cons_nopv.values,
    where="post",
    color=C_BLUE,
    linewidth=LW_DAILY,
    label="Headroom (conservative, no-PV)",
)
ax_hr_nopv.step(
    hr_dates_nopv,
    daily_h_flex_nopv.values,
    where="post",
    color=C_GREEN,
    linewidth=LW_DAILY,
    label="Headroom (flex-aware, no-PV)",
)
ax_hr_nopv.step(
    hr_dates_nopv,
    daily_hp_need_actual_kwh_nopv.values,
    where="post",
    color=C_RED,
    linewidth=LW_DAILY,
    label="Daily HP electrical need (actual COP)",
)
ax_hr_nopv.axhline(
    hp_need_worst_day_kwh_nopv,
    color=C_RED,
    linewidth=LW_DAILY,
    linestyle="--",
    label=f"Worst day @ COP(-10°C) ({hp_need_worst_day_kwh_nopv:.0f} kWh/d)",
)
ax_hr_nopv.set_ylabel("Energy [kWh/day]")
ax_hr_nopv.set_xlabel("Date (2024–2025)")
ax_hr_nopv.set_title("§1.3 Daily headroom vs HP need (full day, no-PV)")
ax_hr_nopv.grid(True, axis="y", linestyle="--", alpha=0.35)
ax_hr_nopv.legend(loc="upper right", fontsize=9, framealpha=0.95)
plt.tight_layout()
plt.show()

fig_u_nopv, ax_u_nopv = plt.subplots(figsize=(14, 4.5))
ax_u_nopv.step(
    hr_dates_nopv,
    util_cons_nopv_pct,
    where="post",
    color=C_BLUE,
    linewidth=LW_DAILY,
    label="Actual HP / H (conservative, no-PV)",
)
ax_u_nopv.step(
    hr_dates_nopv,
    util_flex_nopv_pct,
    where="post",
    color=C_GREEN,
    linewidth=LW_DAILY,
    linestyle="--",
    label="Actual HP / H (flex-aware, no-PV)",
)
ax_u_nopv.step(
    hr_dates_nopv,
    util_worst_cons_nopv_pct,
    where="post",
    color=C_BLUE,
    linewidth=LW_DAILY,
    linestyle=":",
    label=f"Worst-case HP / H (conservative, {hp_need_worst_day_kwh_nopv:.0f} kWh/d)",
)
ax_u_nopv.step(
    hr_dates_nopv,
    util_worst_flex_nopv_pct,
    where="post",
    color=C_GREEN,
    linewidth=LW_DAILY,
    linestyle=":",
    label=f"Worst-case HP / H (flex-aware, {hp_need_worst_day_kwh_nopv:.0f} kWh/d)",
)
ax_u_nopv.axhline(70.0, color=C_BLACK, linewidth=1.5, linestyle="-", label="70% reference")
ax_u_nopv.set_ylabel("Utilisation [%]")
ax_u_nopv.set_xlabel("Date (2024–2025)")
ax_u_nopv.set_title("§1.3 Daily headroom utilisation (no-PV)")
ax_u_nopv.grid(True, axis="y", linestyle="--", alpha=0.35)
ax_u_nopv.legend(loc="upper right", fontsize=9, framealpha=0.95)
plt.tight_layout()
plt.show()
'''


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))

    # Update intro
    intro = "".join(nb["cells"][0]["source"])
    intro = intro.replace(
        "   - **§1.2 (grid-based):** **conservative** and **flex-aware** = cum-max `grid_consumption` (M−1) + 20 kW (identical for HP); **baseline HP** adds worst-case HP electrical peak; **deterministic** from notebook 03 export.",
        "   - **§1.2 (grid-based):** **flex-aware** = cum-max `grid_consumption` (M−1) + 20 kW; **conservative** adds worst-case HP electrical peak; **deterministic** from notebook 03 export.",
    )
    nb["cells"][0]["source"] = [intro]

    md_cell = {
        "cell_type": "markdown",
        "id": str(uuid.uuid4()),
        "metadata": {},
        "source": [MD_13],
    }
    code_cell = {
        "cell_type": "code",
        "id": str(uuid.uuid4()),
        "metadata": {},
        "outputs": [],
        "source": [CODE_13],
        "execution_count": None,
    }

    # Insert after §1.2 (index 5)
    if any("§1.3 — Access power" in "".join(c.get("source", [])) for c in nb["cells"]):
        print("§1.3 already present; updating cells in place")
        for i, c in enumerate(nb["cells"]):
            s = "".join(c.get("source", []))
            if c["cell_type"] == "markdown" and "### 1.3 Access power" in s:
                nb["cells"][i] = md_cell
                nb["cells"][i]["id"] = c.get("id", md_cell["id"])
            if c["cell_type"] == "code" and s.startswith("# §1.3 — Access power"):
                nb["cells"][i] = code_cell
                nb["cells"][i]["id"] = c.get("id", code_cell["id"])
    else:
        nb["cells"].insert(5, md_cell)
        nb["cells"].insert(6, code_cell)
        print("Inserted §1.3 at indices 5–6")

    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Wrote", NB)


if __name__ == "__main__":
    main()
