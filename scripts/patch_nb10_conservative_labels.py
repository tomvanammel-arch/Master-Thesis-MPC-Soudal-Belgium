"""Relabel notebook 10 §1.2/§1.3 headroom to conservative/flex-aware (notebook 09 naming)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "10_online_MPC_1_HP.ipynb"

MD_12 = r"""### 1.2 Access power selection (grid-based)

Monthly access from **cum-max of metered `grid_consumption` peaks** (kWh/15 min → kW via ×4) on `plant1.csv` (2025) with January 2025 seeded from `plant1_2024_training.csv`, plus **+20 kW** margin (same as notebooks 02/03/09).

**Conservative** (grid peaks — notebook 09 analogue; for HP there is no separate `grid_excl_hp`, so this equals the flex-aware grid rule):

\[
P_{\mathrm{access},M}^{\mathrm{conservative}} = \max_{m < M}\bigl(\max_t 4 \cdot \texttt{grid\_consumption}_t^{(m)}\bigr) + 20\ \mathrm{kW}
\]

**Flex-aware** (online MPC contract, Part 2 — same peak history as conservative for HP):

\[
P_{\mathrm{access},M}^{\mathrm{flex}} = \max_{m < M}\bigl(\max_t 4 \cdot \texttt{grid\_consumption}_t^{(m)}\bigr) + 20\ \mathrm{kW}
\]

**Baseline HP** (Part 3.2 billing — matches notebook 03 `baseline_hp_access_power_kw`):

\[
P_{\mathrm{access},M}^{\mathrm{baseline\,HP}} = P_{\mathrm{access},M}^{\mathrm{flex}} + P_{\mathrm{HP,elec}}^{\max}
\]

with \(P_{\mathrm{HP,elec}}^{\max} = \max_t \bigl(4 \cdot \texttt{thermal\_load}_t / \mathrm{COP}(-10\,^{\circ}\mathrm{C})\bigr)\) on 2025 `plant1.csv`.

**Deterministic:** `access_power_kw` from `output/notebooks/deterministic_hp_monthly_bills_notebook_03.csv`.

**Headroom validation** (full calendar day; playroom load = `grid_consumption`):

\[
P_{\mathrm{playroom}}(t) = \max\bigl(P_{\mathrm{access}}(\mathrm{month}(t)) - 4\cdot \texttt{grid\_consumption}_t,\ 0\bigr)\ \mathrm{kW}
\]

\[
H_d = \sum_{t \in \mathrm{day}} P_{\mathrm{playroom}}(t)\cdot 0.25\ \mathrm{h}\quad\text{(kWh/day)}
\]

Two curves: **conservative** and **flex-aware** use their respective monthly \(P_{\mathrm{access}}\) (identical kW for HP). **Daily HP need @ −10°C** (2025 only — 2024 training has no `thermal_load`):

\[
E_{\mathrm{ref},d} = \sum_{t \in \mathrm{day}} \frac{\texttt{thermal\_load}_t}{\mathrm{COP}(-10\,^{\circ}\mathrm{C})}\quad\text{(kWh/day)}
\]

Utilisation: \(100 \times E_{\mathrm{ref},d} / H_d\) vs **70%** reference.

**Exports:** `table`, `ACCESS_POWER_BASELINE_MONTHLY` (baseline HP), `ACCESS_POWER_ONLINE_MONTHLY` (flex-aware), `ACCESS_POWER_DICT`.
"""


def patch_code_12(src: str) -> str:
    repls = [
        (
            "access_power_flex_aware_kw = cummax_grid_Mm1_kw + MARGIN_KW\naccess_power_baseline_hp_kw = access_power_flex_aware_kw + hp_additional_peak_kw",
            "access_power_conservative_kw = cummax_grid_Mm1_kw + MARGIN_KW\naccess_power_flex_aware_kw = cummax_grid_Mm1_kw + MARGIN_KW\naccess_power_baseline_hp_kw = access_power_flex_aware_kw + hp_additional_peak_kw",
        ),
        (
            "access_power_flex_2024_kw = cummax_2024_Mm1_kw + MARGIN_KW\naccess_power_baseline_hp_2024_kw = access_power_flex_2024_kw + hp_additional_peak_kw\naccess_power_by_month_flex_hr = pd.concat([access_power_flex_2024_kw, access_power_flex_aware_kw]).sort_index()\naccess_power_by_month_baseline_hp_hr = pd.concat(\n    [access_power_baseline_hp_2024_kw, access_power_baseline_hp_kw]\n).sort_index()",
            "access_power_conservative_2024_kw = cummax_2024_Mm1_kw + MARGIN_KW\naccess_power_flex_2024_kw = cummax_2024_Mm1_kw + MARGIN_KW\naccess_power_baseline_hp_2024_kw = access_power_flex_2024_kw + hp_additional_peak_kw\naccess_power_by_month_conservative_hr = pd.concat(\n    [access_power_conservative_2024_kw, access_power_conservative_kw]\n).sort_index()\naccess_power_by_month_flex_hr = pd.concat([access_power_flex_2024_kw, access_power_flex_aware_kw]).sort_index()\naccess_power_by_month_baseline_hp_hr = pd.concat(\n    [access_power_baseline_hp_2024_kw, access_power_baseline_hp_kw]\n).sort_index()",
        ),
        (
            '        "access_power_flex_aware": access_power_flex_aware_kw.values,\n        "access_power_baseline_hp": access_power_baseline_hp_kw.values,',
            '        "access_power_conservative": access_power_conservative_kw.values,\n        "access_power_flex_aware": access_power_flex_aware_kw.values,\n        "access_power_baseline_hp": access_power_baseline_hp_kw.values,',
        ),
        (
            "daily_h_base = _daily_headroom_kwh(ts_hr, grid_kwh, access_power_by_month_baseline_hp_hr)\ndaily_h_flex = _daily_headroom_kwh(ts_hr, grid_kwh, access_power_by_month_flex_hr)",
            "daily_h_cons = _daily_headroom_kwh(ts_hr, grid_kwh, access_power_by_month_conservative_hr)\ndaily_h_flex = _daily_headroom_kwh(ts_hr, grid_kwh, access_power_by_month_flex_hr)",
        ),
        (
            "    .reindex(daily_h_base.index)\n)\nhr_dates = pd.to_datetime(daily_h_base.index)\nutil_base_pct = np.where(daily_h_base > 1e-9, 100.0 * daily_hp_need_kwh / daily_h_base, np.nan)",
            "    .reindex(daily_h_cons.index)\n)\nhr_dates = pd.to_datetime(daily_h_cons.index)\nutil_cons_pct = np.where(daily_h_cons > 1e-9, 100.0 * daily_hp_need_kwh / daily_h_cons, np.nan)",
        ),
        (
            'ax_acc.step(\n    month_ts,\n    access_power_baseline_hp_kw.values,\n    where="post",\n    color=C_BLUE,\n    linewidth=LW_ACCESS,\n    linestyle="-",\n    label="Access baseline HP (+ HP @ -10°C)",\n    zorder=3,\n)\nax_acc.step(\n    month_ts,\n    access_power_deterministic_kw.reindex(months.astype(str)).values,',
            'ax_acc.step(\n    month_ts,\n    access_power_conservative_kw.values,\n    where="post",\n    color=C_BLUE,\n    linewidth=LW_ACCESS,\n    linestyle="-",\n    label="Access conservative (grid)",\n    zorder=3,\n)\nax_acc.step(\n    month_ts,\n    access_power_baseline_hp_kw.values,\n    where="post",\n    color=C_ORANGE,\n    linewidth=LW_ACCESS,\n    linestyle="--",\n    label="Access baseline HP (+ HP @ -10°C)",\n    zorder=3,\n)\nax_acc.step(\n    month_ts,\n    access_power_deterministic_kw.reindex(months.astype(str)).values,',
        ),
        (
            '    color=C_ORANGE,\n    linewidth=LW_ACCESS,\n    linestyle="-",\n    label="Access deterministic (notebook 03)",',
            '    color=C_GREEN,\n    linewidth=LW_ACCESS,\n    linestyle="-",\n    label="Access deterministic (notebook 03)",',
        ),
        (
            '    label="Access online / flex-aware",',
            '    label="Access flex-aware (grid)",',
        ),
        (
            "        access_power_baseline_hp_kw.max(),\n        access_power_flex_aware_kw.max(),",
            "        access_power_conservative_kw.max(),\n        access_power_baseline_hp_kw.max(),\n        access_power_flex_aware_kw.max(),",
        ),
        (
            'ax_hr.step(hr_dates, daily_h_base.values, where="post", color=C_BLUE, linewidth=LW_DAILY, label="Headroom (baseline HP)")',
            'ax_hr.step(hr_dates, daily_h_cons.values, where="post", color=C_BLUE, linewidth=LW_DAILY, label="Headroom (conservative)")',
        ),
        (
            'ax_u.step(hr_dates, util_base_pct, where="post", color=C_BLUE, linewidth=LW_DAILY, label="HP / H (baseline HP)")',
            'ax_u.step(hr_dates, util_cons_pct, where="post", color=C_BLUE, linewidth=LW_DAILY, label="HP / H (conservative)")',
        ),
        (
            'ax_u.step(hr_dates, util_flex_pct, where="post", color=C_GREEN, linewidth=LW_DAILY, label="HP / H (flex-aware)")',
            'ax_u.step(hr_dates, util_flex_pct, where="post", color=C_GREEN, linewidth=LW_DAILY, linestyle="--", label="HP / H (flex-aware)")',
        ),
    ]
    out = src
    for old, new in repls:
        if old not in out:
            raise SystemExit(f"§1.2 patch miss: {old[:70]!r}")
        out = out.replace(old, new)
    return out


def patch_code_13(src: str) -> str:
    repls = [
        (
            "access_power_flex_aware_nopv_kw = cummax_grid_nopv_Mm1_kw + MARGIN_KW\naccess_power_baseline_hp_nopv_kw = access_power_flex_aware_nopv_kw + hp_additional_peak_kw",
            "access_power_conservative_nopv_kw = cummax_grid_nopv_Mm1_kw + MARGIN_KW\naccess_power_flex_aware_nopv_kw = cummax_grid_nopv_Mm1_kw + MARGIN_KW\naccess_power_baseline_hp_nopv_kw = access_power_flex_aware_nopv_kw + hp_additional_peak_kw",
        ),
        (
            "access_power_flex_nopv_2024_kw = cummax_2024_nopv_Mm1_kw + MARGIN_KW\naccess_power_baseline_hp_nopv_2024_kw = access_power_flex_nopv_2024_kw + hp_additional_peak_kw\naccess_power_by_month_flex_nopv_hr = pd.concat(\n    [access_power_flex_nopv_2024_kw, access_power_flex_aware_nopv_kw]\n).sort_index()\naccess_power_by_month_baseline_hp_nopv_hr = pd.concat(\n    [access_power_baseline_hp_nopv_2024_kw, access_power_baseline_hp_nopv_kw]\n).sort_index()",
            "access_power_conservative_nopv_2024_kw = cummax_2024_nopv_Mm1_kw + MARGIN_KW\naccess_power_flex_nopv_2024_kw = cummax_2024_nopv_Mm1_kw + MARGIN_KW\naccess_power_baseline_hp_nopv_2024_kw = access_power_flex_nopv_2024_kw + hp_additional_peak_kw\naccess_power_by_month_conservative_nopv_hr = pd.concat(\n    [access_power_conservative_nopv_2024_kw, access_power_conservative_nopv_kw]\n).sort_index()\naccess_power_by_month_flex_nopv_hr = pd.concat(\n    [access_power_flex_nopv_2024_kw, access_power_flex_aware_nopv_kw]\n).sort_index()\naccess_power_by_month_baseline_hp_nopv_hr = pd.concat(\n    [access_power_baseline_hp_nopv_2024_kw, access_power_baseline_hp_nopv_kw]\n).sort_index()",
        ),
        (
            '        "access_power_flex_aware_nopv": access_power_flex_aware_nopv_kw.values,\n        "access_power_baseline_hp_nopv": access_power_baseline_hp_nopv_kw.values,',
            '        "access_power_conservative_nopv": access_power_conservative_nopv_kw.values,\n        "access_power_flex_aware_nopv": access_power_flex_aware_nopv_kw.values,\n        "access_power_baseline_hp_nopv": access_power_baseline_hp_nopv_kw.values,',
        ),
        (
            "daily_h_base_nopv = _daily_headroom_kwh(ts_hr, grid_nopv_kwh, access_power_by_month_baseline_hp_nopv_hr)\ndaily_h_flex_nopv = _daily_headroom_kwh(ts_hr, grid_nopv_kwh, access_power_by_month_flex_nopv_hr)",
            "daily_h_cons_nopv = _daily_headroom_kwh(ts_hr, grid_nopv_kwh, access_power_by_month_conservative_nopv_hr)\ndaily_h_flex_nopv = _daily_headroom_kwh(ts_hr, grid_nopv_kwh, access_power_by_month_flex_nopv_hr)",
        ),
    ]
    out = src
    for old, new in repls:
        if old not in out:
            raise SystemExit(f"§1.3 patch miss: {old[:70]!r}")
        out = out.replace(old, new)
    out = out.replace(".reindex(daily_h_base_nopv.index)", ".reindex(daily_h_cons_nopv.index)")
    out = out.replace(
        "hr_dates_nopv = pd.to_datetime(daily_h_base_nopv.index)",
        "hr_dates_nopv = pd.to_datetime(daily_h_cons_nopv.index)",
    )
    out = out.replace(
        "util_base_nopv_pct = np.where(\n    daily_h_base_nopv > 1e-9",
        "util_cons_nopv_pct = np.where(\n    daily_h_cons_nopv > 1e-9",
    )
    out = out.replace("daily_h_base_nopv.values", "daily_h_cons_nopv.values")
    out = out.replace('label="Headroom (baseline HP, no-PV)"', 'label="Headroom (conservative, no PV)"')
    out = out.replace('label="Headroom (flex-aware, no-PV)"', 'label="Headroom (flex-aware, no PV)"')
    out = out.replace("util_base_nopv_pct", "util_cons_nopv_pct")
    out = out.replace('label="HP / H (baseline HP, no-PV)"', 'label="HP / H (conservative, no PV)"')
    out = out.replace('label="Access baseline HP (no-PV)"', 'label="Access conservative (no PV)"')
    out = out.replace('label="Access flex-aware (no-PV)"', 'label="Access flex-aware (no PV)"')
    return out


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))

    intro = "".join(nb["cells"][0]["source"])
    old_intro = (
        "   - **§1.2 (grid-based):** **baseline HP access** = cum-max `grid_consumption` (M−1) + 20 kW + worst-case HP electrical peak; "
        "**flex-aware (online)** = cum-max `grid_consumption` (M−1) + 20 kW; **deterministic** from notebook 03 export."
    )
    new_intro = (
        "   - **§1.2 (grid-based):** **conservative** and **flex-aware** = cum-max `grid_consumption` (M−1) + 20 kW (identical for HP); "
        "**baseline HP** adds worst-case HP electrical peak; **deterministic** from notebook 03 export."
    )
    if old_intro in intro:
        nb["cells"][0]["source"] = [intro.replace(old_intro, new_intro)]

    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        if c["cell_type"] == "markdown" and "### 1.2 Access power selection" in src:
            nb["cells"][i]["source"] = [MD_12]
            print(f"md §1.2 cell {i}")
        if c["cell_type"] == "code" and src.startswith("# §1.2 — Access power"):
            nb["cells"][i]["source"] = [patch_code_12(src)]
            print(f"code §1.2 cell {i}")
        if c["cell_type"] == "code" and src.startswith("# §1.3 — Access power"):
            nb["cells"][i]["source"] = [patch_code_13(src)]
            print(f"code §1.3 cell {i}")

    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Wrote", NB)


if __name__ == "__main__":
    main()
