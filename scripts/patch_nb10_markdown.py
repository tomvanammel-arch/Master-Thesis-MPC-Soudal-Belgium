"""Update all markdown cells in notebook 10 to match current implementation."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "10_online_MPC_1_HP.ipynb"


def md(*lines: str) -> list[str]:
    return [line + "\n" for line in lines]


MARKDOWN_BY_KEY = {
    "intro": md(
        "# Online HP-Only MPC 1 – Access Power, Simulation & Comparison",
        "",
        "This notebook matches **Chapter 3** of the thesis (methodology): **§3.7.1** (monthly access before online operation), **§3.7.3** (HP-only online MPC), plus **§3.5.3** (offline HP exports from notebook 03), **§3.6** (rolling forecasts from notebooks 05–08), and **§3.3** (billing).",
        "",
        "Structured like **notebook 09** (EV) for a later joint case (notebook 11):",
        "",
        "1. **Monthly access power (§1.1–§1.3)**",
        "   - **§1.2 (grid-based):** **flex-aware** = cum-max `grid_consumption` (M−1) + 20 kW → Part 2 MPC and online billing; **conservative** = flex-aware + worst-case HP electrical peak @ COP(−10°C) → Part 3.2 / 4D baseline; **deterministic** from notebook 03 export (plot only).",
        "   - **§1.3 (no-PV):** same rules on `grid + pv − injection` (validation plots; not wired to Part 2).",
        "",
        "2. **Rolling-horizon online MPC (HP-only)** — `run_hp_online_mpc_1`; clipper **Eq. (3.50)** with `monthly_peak_plan` (same as notebook 09).",
        "",
        "3. **Visualisation and comparison** — baseline (`uncontrolled_hp.csv`) / deterministic (notebook 03) / online (Part 2).",
        "",
        "4. **Scenario analysis** — forecast-stress / inflex-quantile grid (Part 4A–4D).",
        "",
        "**Run order:** notebook 03 export → §1.1 → **§1.2** → (optional §1.3) → Part 2 → Part 3 → (optional Part 4).",
    ),
    "s11": md(
        "### 1.1 Grid consumption, worst-case HP peak, and buffer intuition",
        "",
        "Inspect **Plant 1** `grid_consumption` (includes EV; no HP in meter history) and overlay a **worst-case HP electrical peak @ −10°C** for peak-duration and buffer-sizing intuition only.",
        "",
        "This section does **not** set the Part 2 access contract (that is **§1.2**).",
    ),
    "s12": md(
        "### 1.2 Access power selection (grid-based)",
        "",
        "Monthly access from **cum-max of `grid_consumption` peaks** (kWh/15 min → kW via ×4) on `plant1.csv` (2025), January 2025 seeded from `plant1_2024_training.csv`, plus **+20 kW** margin.",
        "",
        "**Flex-aware** (Part 2 online MPC and online billing):",
        "",
        "\\[",
        "P_{\\mathrm{access},M}^{\\mathrm{flex}} = \\max_{m < M}\\bigl(\\max_t 4\\cdot\\texttt{grid\\_consumption}_t^{(m)}\\bigr) + 20\\ \\mathrm{kW}",
        "\\]",
        "",
        "**Conservative** (Part 3.2 / 4D baseline — same grid peaks **+** worst-case HP electrical peak @ COP(−10°C); aligns with notebook 03 baseline access):",
        "",
        "\\[",
        "P_{\\mathrm{access},M}^{\\mathrm{cons}} = P_{\\mathrm{access},M}^{\\mathrm{flex}} + P_{\\mathrm{HP,elec}}^{\\max}",
        "\\]",
        "",
        "**Deterministic (plot):** `access_power_kw` per month from notebook 03 export (`deterministic_hp_monthly_bills_notebook_03.csv`).",
        "",
        "**Headroom** (full calendar day; playroom load = `grid_consumption` for both access series):",
        "",
        "- **Conservative:** \\(P_{\\mathrm{playroom}}(t) = \\max(P_{\\mathrm{access}}^{\\mathrm{cons}}(\\mathrm{month}) - 4\\cdot\\texttt{grid\\_consumption}_t,\\ 0)\\)",
        "- **Flex-aware:** \\(P_{\\mathrm{playroom}}(t) = \\max(P_{\\mathrm{access}}^{\\mathrm{flex}}(\\mathrm{month}) - 4\\cdot\\texttt{grid\\_consumption}_t,\\ 0)\\)",
        "",
        "Daily headroom: \\(H_d = \\sum_t P_{\\mathrm{playroom}}(t) \\times 0.25\\) h (kWh/day).",
        "",
        "**HP need (2025 only):** actual electrical HP from `thermal_load / COP(T_outdoor)`; worst-case reference = max daily sum at COP(−10°C).",
        "",
        "Utilisation (%): actual and worst-case HP vs conservative / flex-aware headroom (70% reference).",
        "",
        "**Exports:** `table`, `ACCESS_POWER_BASELINE_MONTHLY` (conservative), `ACCESS_POWER_ONLINE_MONTHLY` (flex-aware), `ACCESS_POWER_DICT` (= flex-aware dict for `run_hp_online_mpc_1`).",
    ),
    "s13": md(
        "### 1.3 Access power selection (no-PV counterfactual)",
        "",
        "Same structure as **§1.2**, but monthly peaks and headroom use **no-PV site load** (kWh/15 min):",
        "",
        "\\[",
        "\\texttt{grid\\_nopv} = \\texttt{grid\\_consumption} + \\texttt{pv\\_production} - \\texttt{grid\\_injection}",
        "\\]",
        "",
        "(2024 training CSV has no `grid_injection`; there we use `inflex_load + ev`.)",
        "",
        "**Flex-aware:** \\(\\max_{m<M}(\\max_t 4\\cdot\\texttt{grid\\_nopv}) + 20\\) kW  ",
        "",
        "**Conservative:** flex-aware **+** worst-case HP electrical peak @ COP(−10°C).",
        "",
        "**Headroom** (full day; playroom load = `grid_nopv`): same formulas as §1.2 with the no-PV access series.",
        "",
        "HP need curves reuse §1.2 daily electrical HP (same thermal profile; 2025 only).",
        "",
        "**Not wired to Part 2** — validation only. Run **§1.2** before this cell (does not overwrite `ACCESS_POWER_*` exports).",
    ),
    "part2": md(
        "## 2. Rolling-horizon online MPC simulation (HP-only)",
        "",
        "Full-year simulation via `run_hp_online_mpc_1` in `src/online_MPC_1_HP.py` (thesis §3.7.3; same six-step pattern as notebook 09):",
        "",
        "- At each 15 min step: 24 h HP+buffer MPC (`mpc_hp_24h`) on **forecast** inflex, EV, PV, thermal load, and outdoor temperature.",
        "- **Real-time clipper (Eq. 3.50, aligned with `online_MPC_1_EV`):**",
        "  - \\(P^{\\mathrm{target}}_k\\) = planner `monthly_peak_plan` for the current month (`window_summary[\"monthly_peak_plan\"][month_key]`).",
        "  - \\(P^{\\lim}_k = \\min(P^{\\mathrm{access}}_m,\\ \\max(P^{\\mathrm{peak,sofar}}_{m,k},\\ P^{\\mathrm{target}}_k))\\).",
        "  - \\(P^{\\mathrm{grid,plan}}\\) uses **actual** inflex, EV, PV and planned HP; if above \\(P^{\\lim}\\), reduce HP power by \\(\\Delta P\\) (same as EV Eq. 3.56 / HP Eq. 3.69–3.70).",
        "- **HP-only after clip:** access-aware cap (when SOC allows), physical SOC-min floor, optional SOC-max, then **PLC safeguard** (may exceed access to protect buffer).",
        "- Optional **forecast-stress SOC floor** (`enable_forecast_stress_soc_floor`): raises planner SOC minimum before forecasted access stress.",
        "- Rolling-12 exceedance state passed into each `mpc_hp_24h` solve (Eq. 3.48–3.49 in planner).",
        "",
        "**Access power:** `ACCESS_POWER_DICT` / `ACCESS_POWER_ONLINE_MONTHLY` from §1.2 (**flex-aware**). Run **§1.2** first.",
        "",
        "After editing `src/online_MPC_1_HP.py`, **reload the module** (kernel restart or `importlib.reload`) before Part 2.",
        "",
        "**Outputs:** `res_hp_online`, `summ_hp_online`; export cell writes `output/notebooks/online_hp_15min_notebook_10_part2.csv` (includes `monthly_peak_plan_kw`, `current_peak_opt_kw`, `p_limit_kw`, `was_clipped`, PLC columns).",
    ),
    "part3": md(
        "## 3. Visualisation and comparison of results",
        "",
        "Compare **baseline HP** (`output/uncontrolled_hp.csv` + conservative access), **deterministic HP MPC** (notebook 03 / §3.5.3), and **online MPC** (Part 2) on profiles, peaks, thermal service, and costs.",
        "",
        "**Shadow billing (§3.2):**",
        "- **Baseline:** uncontrolled HP schedule on `plant1.csv` with `ACCESS_POWER_BASELINE_MONTHLY` (§1.2 conservative).",
        "- **Deterministic:** monthly bills / 15-min export from notebook 03 (`deterministic_hp_monthly_bills_notebook_03.csv`, etc.).",
        "- **Online:** bills from Part 2 results (`access_kw` = flex-aware contract; `p_limit_kw` from Eq. 3.50).",
        "- **Access in peak/access bar charts:** baseline uses conservative series; online uses flex-aware / realized `access_kw` from simulation.",
        "- Re-run **Part 2** after changing §1.2 access rules or the clipper in `src/online_MPC_1_HP.py`.",
    ),
    "part31": md(
        "### 3.1 Optimized volumes",
        "",
        "HP plan vs applied power, buffer SOC, grid power vs access / `p_limit_kw`, forecast-stress shading, unmet thermal (`unmet_thermal_kwh_th`), and optional single-window MPC debug.",
        "",
        "Online series: in-memory `res_hp_online` from Part 2, or CSV `output/notebooks/online_hp_15min_notebook_10_part2.csv`.",
        "Deterministic overlay: `deterministic_hp_15min_notebook_03.csv` when notebook 03 export exists.",
    ),
    "part32": md(
        "### 3.2 The bill",
        "",
        "Shadow billing table and cost breakdown: baseline vs deterministic vs online.",
        "",
        "Stacked monthly savings vs baseline (spot / access / peak components). Baseline access = §1.2 **conservative**; online access = §1.2 **flex-aware** (same split as notebook 09).",
    ),
    "part4": md(
        "## 4. Scenario analysis",
        "",
        "Batch HP online MPC runs over a grid of **forecast-stress SOC floor**, **inflex stress quantiles**, and **SOC floor strength** (same core wrapper as Part 2).",
        "",
        "| Part | Purpose | Access power |",
        "|------|---------|----------------|",
        "| **4A** | Define `SCENARIOS` (`HpOnlineScenario`) | — |",
        "| **4B** | Run full-year MPC per scenario (`RUN_SCENARIOS`) | §1.2 **flex-aware** (`ACCESS_POWER_DICT`) |",
        "| **4C** | Viewer: Part 3.1 + 3.2 plots for one export | flex online |",
        "| **4D** | Stacked savings vs baseline (all scenarios) | Baseline: §1.2 **conservative**; online: `access_kw` in each scenario CSV |",
        "",
        "**Prerequisites**",
        "",
        "- §1.2: `ACCESS_POWER_DICT` (month key `YYYY-MM` → kW).",
        "- Forecast CSVs + `config/hp.yaml` as in Part 2.",
        "- Notebook 03 exports for deterministic reference and Part 4C/4D comparisons.",
        "- `output/uncontrolled_hp.csv` for baseline net cost (same definition as Part 3.2).",
        "",
        "**Run order:** §1.2 → Part 2 (reference scenario) → set `RUN_SCENARIOS=True` in 4B only when recomputing the batch → 4C–4D.",
        "",
        "Rerunning 4B overwrites `online_hp_15min_notebook_10_scenario_*.csv` and `online_hp_scenario_analysis_summary_notebook_10.csv`.",
    ),
}


def cell_text(cell: dict) -> str:
    return "".join(cell.get("source", []))


ID_TO_KEY = {
    "e7a3ef92": "intro",
    "ccc716f1": "s11",
    "3715b01a": "s12",
    "09672015-54cf-4add-babd-af02dda5e83d": "s13",
    "213ddff7": "part2",
    "c7b98adc": "part3",
    "09e27a23": "part4",
    "nb10-md-31": "part31",
    "nb10-md-32": "part32",
}


def classify_markdown(cell: dict) -> str | None:
    cid = cell.get("id")
    if cid in ID_TO_KEY:
        return ID_TO_KEY[cid]
    t = cell_text(cell)
    if t.startswith("# Online HP-Only"):
        return "intro"
    if t.startswith("### 1.1"):
        return "s11"
    if t.startswith("### 1.2"):
        return "s12"
    if t.startswith("### 1.3"):
        return "s13"
    if t.startswith("## 2."):
        return "part2"
    if t.startswith("## 3."):
        return "part3"
    if t.startswith("### 3.1"):
        return "part31"
    if t.startswith("### 3.2"):
        return "part32"
    if t.startswith("## 4.") or t.startswith("### Part 4"):
        return "part4"
    return None


def ensure_part31_part32(cells: list) -> None:
    """Insert §3.1 / §3.2 markdown before their code cells if missing."""
    texts = [cell_text(c) for c in cells]
    has_31 = any(t.startswith("### 3.1") for t in texts)
    has_32 = any(t.startswith("### 3.2") for t in texts)

    if has_31 and has_32:
        return

    idx_31_code = next(
        i for i, c in enumerate(cells) if c["cell_type"] == "code" and "# Part 3.1" in cell_text(c)
    )
    idx_32_code = next(
        i for i, c in enumerate(cells) if c["cell_type"] == "code" and "# Part 3.2" in cell_text(c)
    )

    insertions: list[tuple[int, dict]] = []
    if not has_32:
        insertions.append(
            (
                idx_32_code,
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "id": "nb10-md-32",
                    "source": MARKDOWN_BY_KEY["part32"],
                },
            )
        )
    if not has_31:
        insertions.append(
            (
                idx_31_code,
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "id": "nb10-md-31",
                    "source": MARKDOWN_BY_KEY["part31"],
                },
            )
        )

    for idx, cell in sorted(insertions, key=lambda x: x[0], reverse=True):
        cells.insert(idx, cell)


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    ensure_part31_part32(cells)

    for cell in cells:
        if cell["cell_type"] != "markdown":
            continue
        key = classify_markdown(cell)
        if key is not None:
            cell["source"] = MARKDOWN_BY_KEY[key]

    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
    print(f"Updated {n_md} markdown cells in {NB_PATH} ({len(cells)} cells total)")


if __name__ == "__main__":
    main()
