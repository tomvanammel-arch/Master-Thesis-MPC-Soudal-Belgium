"""Align notebook 09 §1.2 plots with notebook 10 thesis style."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "09_online_MPC_1_EV.ipynb"

MD_12 = r"""### 1.2 Access power selection (grid-based)

Two **cum-max + 20 kW** rules on **metered grid** data (`plant1.csv` 2025 + `plant1_2024_training.csv`). January 2025 is seeded from the **2024** peak of the same series (no 2025 history yet at contract time).

**Conservative** (baseline / notebook 02 — matches `baseline_access_power_kw` in the deterministic export):

\[
P_{\mathrm{access},M}^{\mathrm{conservative}} = \max_{m < M}\bigl(\max_t 4 \cdot \texttt{grid\_consumption}_t^{(m)}\bigr) + 20\ \mathrm{kW}
\]

**Flex-aware** (online MPC contract in Part 2):

\[
P_{\mathrm{access},M}^{\mathrm{flex}} = \max_{m < M}\bigl(\max_t 4 \cdot \texttt{grid\_consumption\_excl\_ev}_t^{(m)}\bigr) + 20\ \mathrm{kW}
\]

**Deterministic (plot only):** `optimized_access_kw` from `output/notebooks/deterministic_ev_monthly_notebook_02.csv` (notebook 02 export).

**Exports after this cell:** `table`, `ACCESS_POWER_BASELINE_MONTHLY` (conservative), `ACCESS_POWER_ONLINE_MONTHLY` (flex-aware).

**Headroom validation** (weekdays 07:00–17:00, from first day with EV > 1 kWh/day; playroom load = `grid_consumption_excl_ev`):

\[
P_{\mathrm{hr},t} = \max\bigl(P_{\mathrm{access},M(t)} - 4 \cdot \texttt{grid\_consumption\_excl\_ev}_t,\ 0\bigr),\quad
H_d = \sum_{t \in \text{window}} P_{\mathrm{hr},t} \cdot 0.25\ \mathrm{kWh}
\]

**Thesis figures** (see [`STYLE_GUIDE_PLOTS.md`](../STYLE_GUIDE_PLOTS.md); caption-ready titles, no § prefix):

1. **Monthly access power (2025)** — stacked bars (peak excl. EV + EV increment) + step lines: **Online** (flex-aware), **Baseline** (conservative), **Offline** (deterministic).
2. **Daily electrical headroom and EV demand (2025)** — flex-aware headroom and actual daily EV need (weekdays 07:00–17:00).
3. **Daily headroom utilisation (2025)** — actual utilisation vs flex headroom and **70% limit**.

No-PV counterfactual is in **§1.3**.
"""

PLOTS_BLOCK = r'''# --- §1.2 plots (STYLE_GUIDE_PLOTS.md) ---
import matplotlib as mpl

THESIS_STYLE = {
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "lines.linewidth": 2.2,
    "savefig.dpi": 300,
}
mpl.rcParams.update(THESIS_STYLE)

C_BLACK = "#000000"
C_KUL_RED = "#b30000"
C_BAR_PEAK = "#666666"
LW_ACCESS = 2.2
LW_DAILY = 1.8
_LGND_BOTTOM = dict(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
_X_START_2025 = pd.Timestamp("2025-01-01")
_X_END_2025 = pd.Timestamp("2026-01-01")

month_ts = months.to_timestamp()
month_ts_step = pd.DatetimeIndex([*month_ts, _X_END_2025])


def _extend_step_y(y_month) -> np.ndarray:
    y = np.asarray(y_month, dtype=float)
    return np.r_[y, y[-1]]


peak_excl_kw = monthly_peak_2025_excl_ev_kw.values.astype(float)
peak_ev_incr_kw = np.maximum(
    (monthly_peak_2025_grid_kw - monthly_peak_2025_excl_ev_kw).values.astype(float),
    0.0,
)
_bar_w_days = 22

fig_acc, ax_acc = plt.subplots(figsize=(10, 5))
ax_acc.bar(
    month_ts,
    peak_excl_kw,
    width=_bar_w_days,
    align="edge",
    color=C_BAR_PEAK,
    alpha=0.35,
    edgecolor=C_BLACK,
    linewidth=0.5,
    label="Actual peak excl. EV",
    zorder=1,
)
ax_acc.bar(
    month_ts,
    peak_ev_incr_kw,
    width=_bar_w_days,
    bottom=peak_excl_kw,
    align="edge",
    color=C_KUL_RED,
    alpha=0.45,
    edgecolor=C_KUL_RED,
    linewidth=0.5,
    label="EV peak increment",
    zorder=1,
)
ax_acc.step(
    month_ts_step,
    _extend_step_y(access_power_flex_aware_kw.values),
    where="post",
    color=C_BLACK,
    linewidth=LW_ACCESS,
    linestyle="-",
    label="Online access power",
    zorder=3,
)
ax_acc.step(
    month_ts_step,
    _extend_step_y(access_power_conservative_kw.values),
    where="post",
    color=C_KUL_RED,
    linewidth=LW_ACCESS,
    linestyle="-",
    label="Baseline access power",
    zorder=3,
)
ax_acc.step(
    month_ts_step,
    _extend_step_y(access_power_deterministic_kw.reindex(months.astype(str)).values),
    where="post",
    color=C_BLACK,
    linewidth=LW_ACCESS,
    linestyle="--",
    label="Offline access power",
    zorder=3,
)
ax_acc.set_title("Monthly access power (2025)")
ax_acc.set_ylabel("kW")
ax_acc.set_xlabel("Month")
_y_max_acc = float(
    np.nanmax(
        [
            np.nanmax(peak_excl_kw + peak_ev_incr_kw),
            access_power_conservative_kw.max(),
            access_power_flex_aware_kw.max(),
            access_power_deterministic_kw.reindex(months.astype(str)).max(),
        ]
    )
)
ax_acc.set_ylim(2000, _y_max_acc + 80)
ax_acc.set_xlim(_X_START_2025, _X_END_2025)
ax_acc.margins(x=0)
ax_acc.legend(**_LGND_BOTTOM)
fig_acc.subplots_adjust(bottom=0.26)
plt.tight_layout()
plt.show()

# Daily headroom / utilisation — 2025 only (weekdays 07:00–17:00)
idx_2025 = (
    pd.DatetimeIndex(pd.to_datetime(daily_h_flex.index))
    .normalize()
    .unique()
    .sort_values()
)
idx_2025 = idx_2025[idx_2025.year == 2025]

df_hr_2025 = pd.DataFrame(
    {
        "headroom": daily_h_flex.reindex(idx_2025),
        "ev_actual": daily_ev_kwh.reindex(idx_2025),
    },
    index=idx_2025,
).astype(float)
df_hr_2025["ev_actual"] = df_hr_2025["ev_actual"].replace([np.inf, -np.inf], np.nan)

util_actual_2025 = pd.Series(np.asarray(util_flex_pct, dtype=float), index=hr_dates).reindex(
    idx_2025
)

fig_hr, ax_hr = plt.subplots(figsize=(14, 4.5))
ax_hr.plot(
    df_hr_2025.index,
    df_hr_2025["headroom"],
    drawstyle="steps-post",
    color=C_BLACK,
    linewidth=LW_DAILY,
    label="Headroom",
    zorder=1,
)
ax_hr.plot(
    df_hr_2025.index,
    df_hr_2025["ev_actual"],
    drawstyle="steps-post",
    color=C_KUL_RED,
    linewidth=LW_DAILY,
    linestyle="-",
    label="Actual EV usage",
    zorder=2,
)
ax_hr.set_title("Daily electrical headroom and EV demand (2025)")
ax_hr.set_ylabel("Energy [kWh/day]")
ax_hr.set_xlabel("Date")
ax_hr.set_xlim(_X_START_2025, _X_END_2025)
ax_hr.margins(x=0)
ax_hr.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
fig_hr.subplots_adjust(bottom=0.26)
plt.tight_layout()
plt.show()

fig_u, ax_u = plt.subplots(figsize=(14, 4.5))
ax_u.plot(
    idx_2025,
    util_actual_2025,
    drawstyle="steps-post",
    color=C_BLACK,
    linewidth=LW_DAILY,
    linestyle="-",
    label="Actual utilisation rate",
    zorder=2,
)
ax_u.axhline(70.0, color=C_KUL_RED, linewidth=1.2, linestyle=":", label="70% limit")
ax_u.set_title("Daily headroom utilisation (2025)")
ax_u.set_ylabel("Utilisation [%]")
ax_u.set_xlabel("Date")
ax_u.set_xlim(_X_START_2025, _X_END_2025)
ax_u.margins(x=0)
ax_u.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
fig_u.subplots_adjust(bottom=0.24)
plt.tight_layout()
plt.show()
'''


def patch_code(src: str) -> str:
    # months column fallback
    old_months = (
        'months = (\n'
        '    DET_EV_MONTHLY.set_index("month")["optimized_access_power_kw"]\n'
        "    .sort_index()\n"
        "    .index\n"
        ")"
    )
    new_months = (
        '_det_ap_col = (\n'
        '    "optimized_access_power_kw"\n'
        '    if "optimized_access_power_kw" in DET_EV_MONTHLY.columns\n'
        '    else "optimized_access_kw"\n'
        ")\n"
        "months = DET_EV_MONTHLY.set_index(\"month\")[_det_ap_col].sort_index().index"
    )
    if old_months not in src:
        raise SystemExit("months block not found")
    src = src.replace(old_months, new_months)

    # deterministic access + table row (before table)
    insert_before_table = (
        "access_power_by_month_flex_hr = pd.concat(\n"
        '    [access_power_flex_aware_2024_kw, access_power_flex_aware_kw]\n'
        ").sort_index()\n"
        "\n"
        "table = pd.DataFrame("
    )
    det_block = (
        "access_power_by_month_flex_hr = pd.concat(\n"
        '    [access_power_flex_aware_2024_kw, access_power_flex_aware_kw]\n'
        ").sort_index()\n"
        "\n"
        "access_power_deterministic_kw = (\n"
        '    DET_EV_MONTHLY.assign(month_key=DET_EV_MONTHLY["month"].astype(str))\n'
        "    .set_index(\"month_key\")[_det_ap_col]\n"
        "    .astype(float)\n"
        ")\n"
        "\n"
        "table = pd.DataFrame("
    )
    if insert_before_table not in src:
        raise SystemExit("insert_before_table not found")
    src = src.replace(insert_before_table, det_block)

    old_table_rows = (
        '        "access_power_conservative": access_power_conservative_kw.values,\n'
        '        "access_power_flex_aware": access_power_flex_aware_kw.values,\n'
        "    },\n"
        "    index=months,\n"
        ").T"
    )
    new_table_rows = (
        '        "access_power_conservative": access_power_conservative_kw.values,\n'
        '        "access_power_flex_aware": access_power_flex_aware_kw.values,\n'
        '        "access_power_deterministic": access_power_deterministic_kw.reindex(\n'
        "            months.astype(str)\n"
        "        ).values,\n"
        "    },\n"
        "    index=months,\n"
        ").T"
    )
    if old_table_rows not in src:
        raise SystemExit("table rows not found")
    src = src.replace(old_table_rows, new_table_rows)

    # replace entire old plots section
    start = src.index("# --- §1.2 plots (ZOH")
    end = src.index("# --- Monthly access power for Parts 2–4")
    src = src[:start] + PLOTS_BLOCK + "\n\n" + src[end:]
    return src


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        if c["cell_type"] == "markdown" and "### 1.2 Access power selection" in src:
            nb["cells"][i]["source"] = [line + "\n" for line in MD_12.split("\n")]
            if nb["cells"][i]["source"] and nb["cells"][i]["source"][-1] == "\n":
                nb["cells"][i]["source"].pop()
            print(f"patched markdown cell {i}")
        if c["cell_type"] == "code" and src.startswith("# §1.2 — Access power"):
            nb["cells"][i]["source"] = [
                line + "\n" for line in patch_code(src).split("\n")
            ]
            if nb["cells"][i]["source"] and nb["cells"][i]["source"][-1] == "\n":
                nb["cells"][i]["source"].pop()
            print(f"patched code cell {i}")

    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()
