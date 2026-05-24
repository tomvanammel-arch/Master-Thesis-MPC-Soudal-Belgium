from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from online_MPC_1_EV import run_ev_online_mpc_1


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SlackSensitivityOutputs:
    results_15min_path: Path
    summary_path: Path


def _compute_online_net_cost_eur(summary: Dict) -> float:
    bills = summary.get("bills", None)
    inj = summary.get("injection_bills", None)
    if bills is None or inj is None:
        raise KeyError("summary must contain 'bills' and 'injection_bills'.")

    bills_df = pd.DataFrame(bills).copy()
    inj_df = pd.DataFrame(inj).copy()

    for df in (bills_df, inj_df):
        if "month" not in df.columns:
            raise KeyError("Expected 'month' column in bills DataFrames.")
        df["month_key"] = df["month"].astype(str)

    if "total_cost_eur" not in bills_df.columns:
        raise KeyError("Expected 'total_cost_eur' in bills.")
    if "injection_net_revenue_eur" not in inj_df.columns:
        raise KeyError("Expected 'injection_net_revenue_eur' in injection_bills.")

    net = bills_df[["month_key", "total_cost_eur"]].merge(
        inj_df[["month_key", "injection_net_revenue_eur"]],
        on="month_key",
        how="left",
    )
    net["online_net_cost_eur"] = (
        pd.to_numeric(net["total_cost_eur"], errors="coerce").fillna(0.0)
        - pd.to_numeric(net["injection_net_revenue_eur"], errors="coerce").fillna(0.0)
    )
    return float(net["online_net_cost_eur"].sum())


def _baseline_net_cost_eur_from_det_monthly(det_monthly_path: Path) -> float:
    det = pd.read_csv(det_monthly_path)
    if "baseline_net_cost_eur" not in det.columns:
        raise KeyError(
            f"Expected 'baseline_net_cost_eur' in deterministic monthly export: {det_monthly_path}"
        )
    return float(pd.to_numeric(det["baseline_net_cost_eur"], errors="coerce").fillna(0.0).sum())


def run_online_mpc_slack_sensitivity(
    *,
    slacks_min: Iterable[int] = tuple(range(0, 91, 15)),
    forecast_strategy_ev: str = "b5",
    forecast_strategy_inflex: str = "b8",
    forecast_strategy_pv: str = "actual",
    enforce_daily_ev_demand: bool = True,
    access_power_by_month: Dict[str, float],
    out_dir: Optional[Path] = None,
    results_15min_filename: str = "online_ev_15min_notebook_09_slack_sensitivity.csv",
    summary_filename: str = "online_ev_slack_sensitivity_summary_notebook_09.csv",
    det_monthly_export_path: Optional[Path] = None,
    per_slack_save: bool = True,
    per_slack_15min_stem: str = "online_ev_15min_notebook_09",
    per_slack_debug_stem: str = "online_ev_mpc_debug_notebook_09",
    verbose: bool = True,
) -> SlackSensitivityOutputs:
    """
    Run the full-year online MPC for multiple deadline slack values and store:
    - a combined 15-min results CSV with column `slack_min`
    - a per-slack summary CSV for plotting (unmet energy, net cost, savings)

    Notes
    -----
    - If `per_slack_save=True`, we store two CSVs per slack:
      - `{per_slack_15min_stem}_{xx}_min_slack.csv`
      - `{per_slack_debug_stem}_{xx}_min_slack.csv`
    - We set a custom print prefix, e.g.:
        [Online MPC - 30 min slack] Simulating day 2025-01-12 (step 1057/35040, month 2025-01)
    """
    slacks_min_list = [int(s) for s in slacks_min]
    if any(s < 0 for s in slacks_min_list):
        raise ValueError("slacks_min must be non-negative minutes.")
    if any(s % 15 != 0 for s in slacks_min_list):
        raise ValueError("This sensitivity expects 15-min grid slacks (multiples of 15).")

    if out_dir is None:
        out_dir = PROJECT_ROOT / "output" / "notebooks"
    out_dir.mkdir(parents=True, exist_ok=True)

    if det_monthly_export_path is None:
        det_monthly_export_path = (
            PROJECT_ROOT / "output" / "notebooks" / "deterministic_ev_monthly_notebook_02.csv"
        )
    if not det_monthly_export_path.exists():
        raise FileNotFoundError(f"Deterministic monthly export not found: {det_monthly_export_path}")
    baseline_net_cost_eur = _baseline_net_cost_eur_from_det_monthly(det_monthly_export_path)

    all_results: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, float]] = []

    for slack in slacks_min_list:
        if verbose:
            print("=" * 80)
            print(
                f"[Sensitivity] Running online MPC for slack={slack} min, "
                f"enforce_daily_ev_demand={enforce_daily_ev_demand}"
            )
            print("=" * 80)

        slack_tag = f"{int(slack)}_min_slack"
        debug_path = None
        if per_slack_save:
            debug_path = str(out_dir / f"{per_slack_debug_stem}_{slack_tag}.csv")

        res, summ = run_ev_online_mpc_1(
            forecast_strategy_ev=forecast_strategy_ev,
            forecast_strategy_inflex=forecast_strategy_inflex,
            forecast_strategy_pv=forecast_strategy_pv,
            ev_deadline_slack_minutes=int(slack),
            enforce_daily_ev_demand=enforce_daily_ev_demand,
            access_power_by_month=access_power_by_month,
            verbose=verbose,
            log_prefix=f"[Online MPC - {slack} min slack]",
            enable_mpc_window_debug=bool(per_slack_save),
            mpc_window_debug_csv_path=debug_path,
        )

        if per_slack_save:
            per_slack_15min_path = out_dir / f"{per_slack_15min_stem}_{slack_tag}.csv"
            res.to_csv(per_slack_15min_path, index=False)
            if verbose:
                print(f"[Sensitivity] Saved 15-min results to: {per_slack_15min_path}")

        res_out = res.copy()
        res_out["slack_min"] = int(slack)
        res_out["enforce_daily_ev_demand"] = bool(enforce_daily_ev_demand)
        all_results.append(res_out)

        uncharged = summ.get("uncharged_kwh_by_day", {})
        if not isinstance(uncharged, dict):
            raise TypeError("summary['uncharged_kwh_by_day'] must be a dict day -> kWh.")
        unmet_kwh = float(sum(max(float(v), 0.0) for v in uncharged.values()))
        unmet_mwh = unmet_kwh / 1000.0

        online_net_cost_eur = _compute_online_net_cost_eur(summ)
        savings_vs_baseline_eur = baseline_net_cost_eur - online_net_cost_eur

        summary_rows.append(
            {
                "slack_min": float(slack),
                "enforce_daily_ev_demand": bool(enforce_daily_ev_demand),
                "unmet_mwh": float(unmet_mwh),
                "online_net_cost_eur": float(online_net_cost_eur),
                "baseline_net_cost_eur": float(baseline_net_cost_eur),
                "online_savings_vs_baseline_eur": float(savings_vs_baseline_eur),
                "ev_enforce_steps": int(summ.get("ev_enforce_steps", 0)),
                "ev_enforce_extra_kwh_total": float(
                    summ.get("ev_enforce_extra_kwh_total", 0.0)
                ),
            }
        )

    results_15min = pd.concat(all_results, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows).sort_values("slack_min").reset_index(drop=True)

    results_15min_path = out_dir / results_15min_filename
    summary_path = out_dir / summary_filename

    results_15min.to_csv(results_15min_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    if verbose:
        print("=" * 80)
        print("[Sensitivity] Finished.")
        print(f"[Sensitivity] Saved combined 15-min results to: {results_15min_path}")
        print(f"[Sensitivity] Saved per-slack summary to:      {summary_path}")
        print("=" * 80)

    return SlackSensitivityOutputs(
        results_15min_path=results_15min_path,
        summary_path=summary_path,
    )

