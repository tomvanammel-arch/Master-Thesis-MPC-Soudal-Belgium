"""
Notebook 11 Part 3.1 / 3.2 / 4D code paths for Part 4 scenario viewing.
Keep in sync with `notebooks/11_online_MPC_1_EV+HP.ipynb` Part 3 and Part 4C–4D.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

try:
    from IPython.display import display
except ImportError:

    def display(obj):  # type: ignore[no-redef]
        print(obj)


PROJECT_ROOT_PKG = Path(__file__).resolve().parents[2]
_PART31_SRC = PROJECT_ROOT_PKG / "scripts" / "nb11_part31_cell.py"
_PART32_SRC = PROJECT_ROOT_PKG / "scripts" / "nb11_part32_cell.py"


def _main_namespace() -> Dict[str, Any]:
    main = sys.modules.get("__main__")
    return vars(main) if main is not None else {}


def _exec_cell_source(
    src_path: Path,
    *,
    project_root: Path,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    if not src_path.exists():
        raise FileNotFoundError(f"Missing viewer source: {src_path}")
    ns = _main_namespace()
    ns["PROJECT_ROOT"] = project_root
    ns["SRC_DIR"] = project_root / "src"
    if str(ns["SRC_DIR"]) not in sys.path:
        sys.path.insert(0, str(ns["SRC_DIR"]))
    if extra:
        ns.update(extra)
    code = src_path.read_text(encoding="utf-8")
    code = code.replace('Path("..").resolve()', f"Path({repr(str(project_root))})")
    exec(compile(code, str(src_path), "exec"), ns, ns)


def run_notebook11_part31_optimized_volumes(
    res_evhp_online: pd.DataFrame,
    project_root: Path,
    *,
    week_start: pd.Timestamp,
    day_of_week: int,
    debug_ts: Optional[pd.Timestamp] = None,
    run_mpc_debug: Optional[bool] = None,
    run_day_replay: Optional[bool] = None,
    ev_deadline_slack_min_plot: Optional[int] = None,
) -> None:
    knobs: Dict[str, Any] = {
        "WEEK_START": pd.Timestamp(week_start),
        "DAY_OF_WEEK": int(day_of_week),
    }
    if debug_ts is not None:
        knobs["DEBUG_TS"] = pd.Timestamp(debug_ts)
    if run_mpc_debug is not None:
        knobs["RUN_MPC_DEBUG"] = bool(run_mpc_debug)
    if run_day_replay is not None:
        knobs["RUN_DAY_REPLAY"] = bool(run_day_replay)
    if ev_deadline_slack_min_plot is not None:
        knobs["EV_DEADLINE_SLACK_MIN"] = int(ev_deadline_slack_min_plot)

    extra: Dict[str, Any] = {
        "res_evhp_online": res_evhp_online.copy(),
        "_NB11_PART31_KNOBS": knobs,
    }
    _exec_cell_source(_PART31_SRC, project_root=project_root, extra=extra)


def run_notebook11_part32_billing_comparison(
    res_evhp_online: pd.DataFrame,
    project_root: Path,
    *,
    summ_evhp_online: Optional[dict] = None,
    online_access_power_mode: Optional[str] = None,
) -> None:
    # Reload scenario driver so new helpers are visible without restarting the kernel.
    import importlib
    import online_MPC_1_EV_HP_scenario_analysis as _evhp_scen_reload

    importlib.reload(_evhp_scen_reload)

    extra: Dict[str, Any] = {"res_evhp_online": res_evhp_online.copy()}
    if summ_evhp_online is not None:
        extra["summ_evhp_online"] = summ_evhp_online
    if online_access_power_mode is not None:
        extra["ONLINE_ACCESS_POWER_MODE"] = online_access_power_mode
    _exec_cell_source(_PART32_SRC, project_root=project_root, extra=extra)


_AP_MODE_PLOT_LABELS = {
    "flex_aware": "Offline AP",
    "deterministic": "Online AP",
}


def _access_power_mode_plot_label(mode: str) -> str:
    key = str(mode).strip().lower()
    return _AP_MODE_PLOT_LABELS.get(key, str(mode).replace("_", " "))


def _scenario_axis_label(row: pd.Series) -> str:
    mode = _access_power_mode_plot_label(str(row.get("access_power_mode", "")))
    inflex = str(row.get("forecast_strategy_inflex", "")).replace("c_", "")
    strength = row.get("forecast_stress_soc_floor_strength", "")
    try:
        s = f"{float(strength):g}"
    except (TypeError, ValueError):
        s = str(strength)
    sid = row.get("scenario_id", "")
    return f"{sid}: {mode} inflex {inflex} f{s}"


def run_notebook11_part4d_scenario_savings(project_root: Path) -> None:
    """Part 4D — annual savings vs baseline (thesis-style, joint EV+HP)."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    from billing import calculate_monthly_bills, calculate_monthly_injection_bills, load_billing_config

    _main = _main_namespace()
    access_baseline = _main.get("ACCESS_POWER_BASELINE_MONTHLY")
    if access_baseline is None:
        raise RuntimeError("Run §1.2 first to build ACCESS_POWER_BASELINE_MONTHLY.")

    PROJECT_ROOT = project_root
    NOTEBOOKS_OUTPUT_DIR = PROJECT_ROOT / "output" / "notebooks"
    SUMMARY_CSV = NOTEBOOKS_OUTPUT_DIR / "online_ev_hp_scenario_analysis_summary_notebook_11.csv"
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Run Part 4B first. Missing {SUMMARY_CSV}")

    df_sum = pd.read_csv(SUMMARY_CSV)
    err = df_sum["error"].fillna("").astype(str).str.strip()
    ok = err.eq("")
    on_net = pd.to_numeric(df_sum["online_net_cost_eur"], errors="coerce")
    ok = ok & on_net.notna()
    df_ok = df_sum.loc[ok].copy()
    if df_ok.empty:
        raise ValueError("No successful scenarios in summary CSV.")

    billing_cfg = load_billing_config(str(PROJECT_ROOT / "config" / "billing.yaml"))
    plant = pd.read_csv(PROJECT_ROOT / "data" / "plant1.csv")
    plant_ts = pd.to_datetime(plant["timestamp"], utc=True, errors="coerce")
    plant["timestamp"] = plant_ts.dt.tz_convert("Europe/Brussels").dt.tz_localize(None)
    plant = plant.sort_values("timestamp").reset_index(drop=True)
    plant = plant[
        (plant["timestamp"] >= pd.Timestamp("2025-01-01"))
        & (plant["timestamp"] < pd.Timestamp("2026-01-01"))
    ].copy()

    hp_un_path = PROJECT_ROOT / "output" / "uncontrolled_hp.csv"
    if not hp_un_path.exists():
        raise FileNotFoundError(f"Missing {hp_un_path} (notebook 03).")
    hp_un = pd.read_csv(hp_un_path)

    plant_seq_b = plant[
        ["timestamp", "inflex_load", "ev", "pv_production", "price"]
    ].copy().reset_index(drop=True)
    hp_seq = (
        hp_un[["hp_electrical_load"]]
        .rename(columns={"hp_electrical_load": "hp_kwh"})
        .reset_index(drop=True)
    )
    n_b = min(len(plant_seq_b), len(hp_seq))
    df_base = pd.concat([plant_seq_b.iloc[:n_b], hp_seq.iloc[:n_b]], axis=1)
    net_base = (
        df_base["inflex_load"].fillna(0.0)
        + df_base["ev"].fillna(0.0)
        + df_base["hp_kwh"].fillna(0.0)
        - df_base["pv_production"].fillna(0.0)
    )
    df_base["grid_consumption"] = net_base.clip(lower=0.0)
    df_base["grid_injection"] = (-net_base).clip(lower=0.0)
    df_base["month_key"] = df_base["timestamp"].dt.to_period("M").astype(str)
    df_base["access_kw"] = df_base["month_key"].map(access_baseline.astype(float).to_dict()).astype(float)

    a_base = calculate_monthly_bills(
        df_base,
        billing_cfg,
        volume_col="grid_consumption",
        price_col="price",
        timestamp_col="timestamp",
        access_power_col="access_kw",
    )
    inj_base = calculate_monthly_injection_bills(
        df_base,
        billing_cfg,
        injection_col="grid_injection",
        price_col="price",
        timestamp_col="timestamp",
    )
    base_energy = float(a_base["energy_cost_eur"].sum())
    base_spot = float(a_base["spot_cost_eur"].sum())
    base_access = float(a_base["access_cost_eur"].sum())
    base_monthly_peak = float(a_base["monthly_peak_cost_eur"].sum())
    base_over_usage = float(a_base["over_usage_cost_eur"].sum())
    base_inj_rev = float(inj_base["injection_net_revenue_eur"].sum())
    baseline_net = float(a_base["total_cost_eur"].sum() - base_inj_rev)

    det_bills_path = NOTEBOOKS_OUTPUT_DIR / "deterministic_ev_hp_monthly_bills_notebook_04.csv"
    det_inj_path = NOTEBOOKS_OUTPUT_DIR / "deterministic_ev_hp_monthly_injection_notebook_04.csv"
    if not det_bills_path.exists() or not det_inj_path.exists():
        raise FileNotFoundError("Run notebook 04 export first.")
    det_bills = pd.read_csv(det_bills_path)
    det_inj = pd.read_csv(det_inj_path)
    det_energy = float(det_bills["energy_cost_eur"].sum())
    det_spot = float(det_bills["spot_cost_eur"].sum())
    det_access = float(det_bills["access_cost_eur"].sum())
    det_monthly_peak = float(det_bills["monthly_peak_cost_eur"].sum())
    det_over_usage = float(det_bills["over_usage_cost_eur"].sum())
    det_inj_rev = float(det_inj["injection_net_revenue_eur"].sum())
    deterministic_net = float(det_bills["total_cost_eur"].sum() - det_inj_rev)

    def _online_components(results_csv_path: Path) -> dict:
        res = pd.read_csv(results_csv_path)
        ev_col = "ev_applied" if "ev_applied" in res.columns else "ev_online_mpc"
        hp_col = "hp_applied" if "hp_applied" in res.columns else "hp_applied_kwh"
        plant_seq = plant[["timestamp", "inflex_load", "pv_production", "price"]].reset_index(drop=True)
        res_seq = res[[ev_col, hp_col, "access_kw"]].reset_index(drop=True)
        n = min(len(plant_seq), len(res_seq))
        df_on = pd.concat([plant_seq.iloc[:n], res_seq.iloc[:n]], axis=1)
        net = (
            df_on["inflex_load"].fillna(0.0)
            + df_on[ev_col].fillna(0.0)
            + df_on[hp_col].fillna(0.0)
            - df_on["pv_production"].fillna(0.0)
        )
        df_on["grid_consumption"] = net.clip(lower=0.0)
        df_on["grid_injection"] = (-net).clip(lower=0.0)
        bills = calculate_monthly_bills(
            df_on,
            billing_cfg,
            volume_col="grid_consumption",
            price_col="price",
            timestamp_col="timestamp",
            access_power_col="access_kw",
        )
        inj = calculate_monthly_injection_bills(
            df_on,
            billing_cfg,
            injection_col="grid_injection",
            price_col="price",
            timestamp_col="timestamp",
        )
        return {
            "energy": float(bills["energy_cost_eur"].sum()),
            "spot": float(bills["spot_cost_eur"].sum()),
            "access": float(bills["access_cost_eur"].sum()),
            "monthly_peak": float(bills["monthly_peak_cost_eur"].sum()),
            "over_usage": float(bills["over_usage_cost_eur"].sum()),
            "inj_rev": float(inj["injection_net_revenue_eur"].sum()),
            "net": float(bills["total_cost_eur"].sum() - float(inj["injection_net_revenue_eur"].sum())),
        }

    labels = ["Offline"]
    energy_sav = [base_energy - det_energy]
    spot_sav = [base_spot - det_spot]
    access_sav = [base_access - det_access]
    monthly_peak_sav = [base_monthly_peak - det_monthly_peak]
    over_usage_sav = [base_over_usage - det_over_usage]
    inj_rev_sav = [det_inj_rev - base_inj_rev]
    rows_print = [
        ("Baseline", baseline_net, 0.0),
        ("Offline", deterministic_net, baseline_net - deterministic_net),
    ]

    comp_rows = []
    for _, row in df_ok.iterrows():
        res_path = Path(str(row["results_15min_path"]))
        if not res_path.is_absolute():
            res_path = NOTEBOOKS_OUTPUT_DIR / res_path.name
        if not res_path.exists():
            res_path = PROJECT_ROOT / str(row["results_15min_path"])
        if not res_path.exists():
            print(f"WARNING: missing {row['results_15min_path']}")
            continue
        comps = _online_components(res_path)
        ax_label = _scenario_axis_label(row)
        labels.append(ax_label)
        energy_sav.append(base_energy - comps["energy"])
        spot_sav.append(base_spot - comps["spot"])
        access_sav.append(base_access - comps["access"])
        monthly_peak_sav.append(base_monthly_peak - comps["monthly_peak"])
        over_usage_sav.append(base_over_usage - comps["over_usage"])
        inj_rev_sav.append(comps["inj_rev"] - base_inj_rev)
        rows_print.append((ax_label, comps["net"], baseline_net - comps["net"]))
        comp_rows.append(
            {
                "scenario_id": row["scenario_id"],
                "scenario_name": row["scenario_name"],
                "access_power_mode": row.get("access_power_mode", ""),
                "forecast_strategy_inflex": row.get("forecast_strategy_inflex", ""),
                "forecast_strategy_inflex_stress": row.get("forecast_strategy_inflex_stress", ""),
                "forecast_stress_soc_floor_strength": row.get(
                    "forecast_stress_soc_floor_strength", np.nan
                ),
                "online_net_cost_eur": comps["net"],
                "savings_vs_baseline_eur": baseline_net - comps["net"],
                "savings_vs_deterministic_eur": deterministic_net - comps["net"],
            }
        )

    net_costs = [deterministic_net] + [r[1] for r in rows_print[2:]]
    total_savings = baseline_net - np.array(net_costs[: len(labels)], dtype=float)
    comp_series = {
        "Energy": np.array(energy_sav, dtype=float),
        "Spot": np.array(spot_sav, dtype=float),
        "Access power": np.array(access_sav, dtype=float),
        "Monthly peak": np.array(monthly_peak_sav, dtype=float),
        "Over-usage": np.array(over_usage_sav, dtype=float),
        "Injection revenue": np.array(inj_rev_sav, dtype=float),
    }
    table = pd.DataFrame({**comp_series, "Total savings": total_savings}, index=labels).T

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    _C_BLACK = "#000000"
    _C_KUL_RED = "#b30000"
    n = len(labels)
    x = np.arange(n)
    fig_w = max(12.0, 0.85 * n)

    fig1, ax1 = plt.subplots(figsize=(fig_w, 5))
    for i, sav in enumerate(total_savings):
        if i == 0:
            ax1.bar(i, sav, width=0.72, facecolor="white", edgecolor=_C_BLACK, hatch="///")
        else:
            ax1.bar(i, sav, width=0.72, color=_C_KUL_RED, alpha=0.55, edgecolor=_C_KUL_RED)
    ax1.axhline(0.0, color=_C_BLACK, linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=90, ha="center", fontsize=8)
    ax1.set_ylabel("Savings vs baseline [EUR/year]")
    ax1.set_title("Joint online MPC — annual net savings vs baseline (2025)")
    plt.tight_layout()
    plt.show()

    stack_spec = [
        ("Energy", "#666666"),
        ("Spot", _C_BLACK),
        ("Access power", _C_KUL_RED),
        ("Monthly peak", "#aaaaaa"),
        ("Over-usage", "#cccccc"),
        ("Injection revenue", "#2166ac"),
    ]
    fig2, ax2 = plt.subplots(figsize=(fig_w, 5))
    bottom = np.zeros(n, dtype=float)
    for comp_name, comp_color in stack_spec:
        vals = comp_series[comp_name]
        ax2.bar(x, vals, bottom=bottom, width=0.72, label=comp_name, color=comp_color, edgecolor=_C_BLACK, linewidth=0.35)
        bottom = bottom + vals
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=90, ha="center", fontsize=8)
    ax2.set_ylabel("Savings vs baseline [EUR/year]")
    ax2.set_title("Annual savings by cost component vs baseline (2025)")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    plt.tight_layout()
    plt.show()

    cmp_df = pd.DataFrame(comp_rows)
    print("\nScenario comparison table:")
    display(
        cmp_df.sort_values(
            ["access_power_mode", "forecast_stress_soc_floor_strength", "forecast_strategy_inflex"]
        )
    )

    if not cmp_df.empty:
        fig3, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, mode in zip(axes, ("flex_aware", "deterministic")):
            sub = cmp_df[cmp_df["access_power_mode"] == mode]
            if sub.empty:
                continue
            strengths = sorted(sub["forecast_stress_soc_floor_strength"].unique())
            x_s = np.arange(len(strengths))
            w = 0.35
            for j, q in enumerate(("c_p50", "c_p90")):
                y = []
                for st in strengths:
                    row_q = sub[
                        (sub["forecast_stress_soc_floor_strength"] == st)
                        & (sub["forecast_strategy_inflex"] == q)
                    ]
                    y.append(
                        float(row_q["savings_vs_baseline_eur"].iloc[0])
                        if len(row_q)
                        else 0.0
                    )
                ax.bar(x_s + (j - 0.5) * w, y, width=w, label=q.replace("c_", ""))
            ax.set_xticks(x_s)
            ax.set_xticklabels([f"{s:g}" for s in strengths])
            ax.set_xlabel("SOC floor strength")
            ax.set_title(_access_power_mode_plot_label(mode))
            ax.axhline(0, color="k", lw=0.6)
            ax.legend()
        axes[0].set_ylabel("Savings vs baseline [EUR/year]")
        fig3.suptitle("Scenario grid: savings by access power (Offline AP vs Online AP)")
        plt.tight_layout()
        plt.show()

    print("\nSavings vs baseline [EUR/year] — component table:")
    display(table.style.format("{:,.0f}"))
