from __future__ import annotations

import json
import re
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from billing import calculate_monthly_bills, calculate_monthly_injection_bills, load_billing_config
from heat_pump_load import interpolate_cop, load_hp_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class HpOnlineScenario:
    """One HP online MPC scenario (metadata + kwargs for `run_hp_online_mpc_1`)."""

    scenario_id: Union[int, str]
    name: str
    forecast_strategy_inflex: str = "c"
    forecast_strategy_inflex_stress: Optional[str] = None
    forecast_strategy_pv: str = "actual"
    forecast_strategy_thermal: str = "c"
    forecast_strategy_ev: str = "actual"
    forecast_strategy_temperature: str = "actual"
    hp_config_path: Optional[str] = None
    enforce_soc_min: bool = True
    enforce_soc_max: bool = False
    soc_slack_penalty_eur_per_soc: Optional[float] = None
    soc_min_slack_penalty_eur_per_soc: float = 1.0e6
    monthly_peak_price_multiplier: float = 1.0
    horizon_len: int = 96
    enable_mpc_window_debug: bool = False
    enable_forecast_stress_soc_floor: bool = False
    forecast_stress_soc_floor_strength: float = 1.0

    def to_run_kwargs(self) -> Dict[str, Any]:
        return {
            "forecast_strategy_inflex": self.forecast_strategy_inflex,
            "forecast_strategy_inflex_stress": self.forecast_strategy_inflex_stress,
            "forecast_strategy_pv": self.forecast_strategy_pv,
            "forecast_strategy_thermal": self.forecast_strategy_thermal,
            "forecast_strategy_ev": self.forecast_strategy_ev,
            "forecast_strategy_temperature": self.forecast_strategy_temperature,
            "hp_config_path": self.hp_config_path,
            "enforce_soc_min": self.enforce_soc_min,
            "enforce_soc_max": self.enforce_soc_max,
            "soc_slack_penalty_eur_per_soc": self.soc_slack_penalty_eur_per_soc,
            "soc_min_slack_penalty_eur_per_soc": self.soc_min_slack_penalty_eur_per_soc,
            "monthly_peak_price_multiplier": self.monthly_peak_price_multiplier,
            "horizon_len": self.horizon_len,
            "enable_mpc_window_debug": self.enable_mpc_window_debug,
            "enable_forecast_stress_soc_floor": self.enable_forecast_stress_soc_floor,
            "forecast_stress_soc_floor_strength": self.forecast_stress_soc_floor_strength,
        }


@dataclass(frozen=True)
class HpScenarioRunOutputs:
    scenario_id: Union[int, str]
    name: str
    results_15min_path: Path
    summary_json_path: Path
    mpc_debug_csv_path: Optional[Path] = None


@dataclass(frozen=True)
class HpScenarioAnalysisOutputs:
    master_summary_path: Path
    per_scenario: Tuple[HpScenarioRunOutputs, ...]


def sanitize_scenario_name_for_filename(name: str) -> str:
    s = name.strip().lower().replace(" ", "_")
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._-")
    return s or "scenario"


def scenario_id_tag_for_filename(scenario_id: Union[int, str]) -> str:
    if isinstance(scenario_id, int):
        return f"{int(scenario_id):02d}"
    sid = str(scenario_id).strip()
    return re.sub(r"[^A-Za-z0-9._-]+", "_", sid)[:48] or "id"


def _load_plant_2025(project_root: Path) -> pd.DataFrame:
    plant_path = project_root / "data" / "plant1.csv"
    plant = pd.read_csv(plant_path)
    plant_ts = pd.to_datetime(plant["timestamp"], utc=True, errors="coerce")
    plant["timestamp"] = plant_ts.dt.tz_convert("Europe/Brussels").dt.tz_localize(None)
    plant = plant.sort_values("timestamp").reset_index(drop=True)
    plant = plant[
        (plant["timestamp"] >= pd.Timestamp("2025-01-01"))
        & (plant["timestamp"] < pd.Timestamp("2026-01-01"))
    ].copy()
    return plant


def compute_online_annual_net_cost_eur(project_root: Path, res: pd.DataFrame) -> float:
    """
    Same annual net (consumption bills minus injection revenue) as Notebook 10 Part 3.2
    for the online branch: plant + hp_applied_kwh + access_kw from `res`.
    """
    billing_cfg = load_billing_config(str(project_root / "config" / "billing.yaml"))
    plant = _load_plant_2025(project_root)

    plant_seq = plant[
        ["timestamp", "inflex_load", "ev", "pv_production", "price", "outdoor_temperature"]
    ].copy().reset_index(drop=True)
    res_b = res.copy()
    res_seq = res_b[["hp_applied_kwh", "access_kw"]].copy().reset_index(drop=True)

    n = min(len(plant_seq), len(res_seq))
    if len(plant_seq) != len(res_seq):
        print(
            "WARNING: plant and online results lengths differ in billing helper. "
            f"Using first n={n} rows (plant={len(plant_seq)}, res={len(res_seq)})."
        )
    plant_seq = plant_seq.iloc[:n].copy()
    res_seq = res_seq.iloc[:n].copy()

    df_online = pd.concat([plant_seq, res_seq], axis=1)
    net_kwh_online = (
        df_online["inflex_load"].fillna(0.0)
        + df_online["ev"].fillna(0.0)
        + df_online["hp_applied_kwh"].fillna(0.0)
        - df_online["pv_production"].fillna(0.0)
    )
    df_online["grid_consumption"] = net_kwh_online.clip(lower=0.0)
    df_online["grid_injection"] = (-net_kwh_online).clip(lower=0.0)

    online_bills = calculate_monthly_bills(
        df_online,
        billing_cfg,
        volume_col="grid_consumption",
        price_col="price",
        timestamp_col="timestamp",
        access_power_col="access_kw",
    )
    online_inj = calculate_monthly_injection_bills(
        df_online,
        billing_cfg,
        injection_col="grid_injection",
        price_col="price",
        timestamp_col="timestamp",
    )
    online_bills["month_key"] = online_bills["month"].astype(str)
    online_inj["month_key"] = online_inj["month"].astype(str)
    online_net = online_bills[["month_key", "total_cost_eur"]].merge(
        online_inj[["month_key", "injection_net_revenue_eur"]],
        on="month_key",
        how="left",
    )
    online_net["online_net_cost_eur"] = (
        online_net["total_cost_eur"] - online_net["injection_net_revenue_eur"].fillna(0.0)
    )
    return float(online_net["online_net_cost_eur"].sum())


def compute_baseline_annual_net_cost_eur(project_root: Path) -> float:
    """Notebook 10 Part 3.2 baseline (uncontrolled HP + heuristic access power)."""
    billing_cfg = load_billing_config(str(project_root / "config" / "billing.yaml"))
    plant = _load_plant_2025(project_root)

    uncontrolled_hp_path = project_root / "output" / "uncontrolled_hp.csv"
    if not uncontrolled_hp_path.exists():
        raise FileNotFoundError(
            f"Baseline uncontrolled HP not found at {uncontrolled_hp_path}. "
            "Run notebook 03 uncontrolled HP generation first."
        )
    hp_un = pd.read_csv(uncontrolled_hp_path)

    plant_seq_b = plant[
        [
            "timestamp",
            "inflex_load",
            "ev",
            "pv_production",
            "price",
            "grid_consumption",
            "thermal_load",
            "outdoor_temperature",
        ]
    ].copy().reset_index(drop=True)
    hp_seq = hp_un[["hp_electrical_load"]].rename(columns={"hp_electrical_load": "hp_kwh"}).copy().reset_index(
        drop=True
    )

    n_b = min(len(plant_seq_b), len(hp_seq))
    if len(plant_seq_b) != len(hp_seq):
        print(
            "WARNING: plant and uncontrolled_hp lengths differ in baseline billing helper. "
            f"Using first n={n_b} rows (plant={len(plant_seq_b)}, uncontrolled_hp={len(hp_seq)})."
        )
    plant_seq_b = plant_seq_b.iloc[:n_b].copy()
    hp_seq = hp_seq.iloc[:n_b].copy()

    df_base = pd.concat([plant_seq_b, hp_seq], axis=1)
    net_kwh_base = (
        df_base["inflex_load"].fillna(0.0)
        + df_base["ev"].fillna(0.0)
        + df_base["hp_kwh"].fillna(0.0)
        - df_base["pv_production"].fillna(0.0)
    )
    df_base["grid_consumption_with_hp"] = net_kwh_base.clip(lower=0.0)
    df_base["grid_injection_with_hp"] = (-net_kwh_base).clip(lower=0.0)

    _naive = pd.to_datetime(
        df_base["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        format="%Y-%m-%d %H:%M:%S",
    )
    df_base["month"] = _naive.dt.to_period("M")

    MARGIN_KW = 20.0
    BASELINE_2024_PEAK_GRID_KW = 2663.5
    months_2025 = pd.period_range("2025-01", "2025-12", freq="M")

    monthly_peak_baseline_kw = (
        (df_base.groupby("month")["grid_consumption"].max() * 4.0).reindex(months_2025).fillna(0.0)
    )
    cummax_M_minus_1_kw = monthly_peak_baseline_kw.cummax().shift(1)
    cummax_M_minus_1_kw.loc[months_2025.min()] = BASELINE_2024_PEAK_GRID_KW
    cummax_M_minus_1_kw = cummax_M_minus_1_kw.fillna(BASELINE_2024_PEAK_GRID_KW)
    access_power_conservative = cummax_M_minus_1_kw + MARGIN_KW

    hp_cfg = load_hp_config(str(project_root / "config" / "hp.yaml"))
    cop_at_minus10 = interpolate_cop(-10.0, hp_cfg["COP_data"])
    max_thermal_kwh = float(df_base["thermal_load"].max())
    thermal_max_kw = max_thermal_kwh * 4.0
    hp_additional_peak_kw = thermal_max_kw / cop_at_minus10
    access_power_hp_monthly = access_power_conservative + hp_additional_peak_kw

    df_base["access_kw"] = df_base["month"].map(access_power_hp_monthly.to_dict()).astype(float)
    df_base["grid_consumption"] = df_base["grid_consumption_with_hp"]
    df_base["grid_injection"] = df_base["grid_injection_with_hp"]
    df_base["month_key"] = df_base["month"].astype(str)

    baseline_bills = calculate_monthly_bills(
        df_base,
        billing_cfg,
        volume_col="grid_consumption",
        price_col="price",
        timestamp_col="timestamp",
        access_power_col="access_kw",
    )
    baseline_inj = calculate_monthly_injection_bills(
        df_base,
        billing_cfg,
        injection_col="grid_injection",
        price_col="price",
        timestamp_col="timestamp",
    )
    baseline_bills["month_key"] = baseline_bills["month"].astype(str)
    baseline_inj["month_key"] = baseline_inj["month"].astype(str)
    baseline_net = baseline_bills[["month_key", "total_cost_eur"]].merge(
        baseline_inj[["month_key", "injection_net_revenue_eur"]],
        on="month_key",
        how="left",
    )
    baseline_net["baseline_net_cost_eur"] = (
        baseline_net["total_cost_eur"] - baseline_net["injection_net_revenue_eur"].fillna(0.0)
    )
    return float(baseline_net["baseline_net_cost_eur"].sum())


def compute_deterministic_annual_net_cost_eur(project_root: Path) -> float:
    """Sum of deterministic HP monthly net from notebook 03 exports."""
    out_dir = project_root / "output" / "notebooks"
    det_bills_path = out_dir / "deterministic_hp_monthly_bills_notebook_03.csv"
    det_inj_path = out_dir / "deterministic_hp_monthly_injection_notebook_03.csv"
    if not det_bills_path.exists() or not det_inj_path.exists():
        raise FileNotFoundError(
            "Missing deterministic HP exports from notebook 03. Expected:\n"
            f"- {det_bills_path}\n"
            f"- {det_inj_path}\n"
            "Run notebook 03 export cell first."
        )
    det_bills = pd.read_csv(det_bills_path)
    det_inj = pd.read_csv(det_inj_path)
    for _df in (det_bills, det_inj):
        if "month" in _df.columns:
            _df["month_key"] = _df["month"].astype(str)
    det_net = det_bills[["month_key", "total_cost_eur"]].merge(
        det_inj[["month_key", "injection_net_revenue_eur"]],
        on="month_key",
        how="left",
    )
    det_net["deterministic_net_cost_eur"] = (
        det_net["total_cost_eur"] - det_net["injection_net_revenue_eur"].fillna(0.0)
    )
    return float(det_net["deterministic_net_cost_eur"].sum())


def _scenario_json_ready(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def run_hp_online_scenario_analysis(
    *,
    scenarios: Sequence[HpOnlineScenario],
    access_power_by_month: Dict[str, float],
    out_dir: Optional[Path] = None,
    per_scenario_15min_stem: str = "online_hp_15min_notebook_10_scenario",
    per_scenario_summary_stem: str = "online_hp_summary_notebook_10_scenario",
    master_summary_filename: str = "online_hp_scenario_analysis_summary_notebook_10.csv",
    per_scenario_save_debug: bool = False,
    stop_on_error: bool = False,
    verbose: bool = True,
    mpc_verbose: bool = True,
    project_root: Optional[Path] = None,
) -> HpScenarioAnalysisOutputs:
    """
    Run `run_hp_online_mpc_1` for each scenario, write per-scenario 15-min CSV + summary JSON,
    and a master summary CSV (annual net costs aligned with Notebook 10 Part 3.2).

    Reruns overwrite outputs at the same paths. If a scenario fails, the error is logged and
    recorded in the master summary row unless `stop_on_error=True`.

    Baseline and deterministic annual net costs are computed once **after** all MPC runs so
    per-scenario progress logs appear immediately (same as Part 2), then the master CSV rows
    are filled with reference totals and savings columns.

    Parameters
    ----------
    verbose:
        Print batch-level banners from this wrapper.
    mpc_verbose:
        Passed explicitly to ``run_hp_online_mpc_1(..., verbose=mpc_verbose, log_prefix=...)``
        (same idea as notebook 09 / ``run_ev_online_mpc_1(..., verbose=verbose)``). Scenario
        definitions must not carry MPC ``verbose``; it is controlled only here.
    """
    if project_root is None:
        project_root = PROJECT_ROOT
    if out_dir is None:
        out_dir = project_root / "output" / "notebooks"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve at call time (same pattern as EV slack sensitivity) so notebook `importlib.reload`
    # of `online_MPC_1_HP` is respected without restarting the kernel.
    from online_MPC_1_HP import run_hp_online_mpc_1 as _run_hp_online_mpc_1

    ids_seen = set()
    for sc in scenarios:
        sid = sc.scenario_id
        if sid in ids_seen:
            raise ValueError(f"Duplicate scenario_id: {sid!r}")
        ids_seen.add(sid)

    per_outputs: List[HpScenarioRunOutputs] = []
    summary_rows: List[Dict[str, Any]] = []

    for sc in scenarios:
        sid_tag = scenario_id_tag_for_filename(sc.scenario_id)
        safe_name = sanitize_scenario_name_for_filename(sc.name)
        base_name = f"{per_scenario_15min_stem}_{sid_tag}_{safe_name}"
        summ_name = f"{per_scenario_summary_stem}_{sid_tag}_{safe_name}"
        results_path = out_dir / f"{base_name}.csv"
        summary_path = out_dir / f"{summ_name}.json"
        debug_path: Optional[Path] = None
        if per_scenario_save_debug and sc.enable_mpc_window_debug:
            debug_path = out_dir / f"online_hp_mpc_debug_notebook_10_scenario_{sid_tag}_{safe_name}.csv"

        log_prefix = f"[Online MPC - scenario {sc.scenario_id}: {sc.name}]"
        mpc_kw = dict(sc.to_run_kwargs())
        # Never pass these via **mpc_kw: always set explicitly (avoids duplicate-keyword
        # errors if an older `HpOnlineScenario` / cached module still put `verbose` in kwargs).
        mpc_kw.pop("verbose", None)
        mpc_kw.pop("log_prefix", None)
        mpc_kw.pop("access_power_by_month", None)

        row: Dict[str, Any] = {
            "scenario_id": sc.scenario_id,
            "scenario_name": sc.name,
            "results_15min_path": str(results_path),
            "summary_json_path": str(summary_path),
            "mpc_debug_csv_path": str(debug_path) if debug_path else "",
            "baseline_net_cost_eur": float("nan"),
            "deterministic_net_cost_eur": float("nan"),
            "scenario_kwargs_json": json.dumps(_scenario_json_ready(asdict(sc)), sort_keys=True),
            "error": "",
            "online_net_cost_eur": float("nan"),
            "online_savings_vs_baseline_eur": float("nan"),
            "online_savings_vs_deterministic_eur": float("nan"),
        }

        try:
            if verbose:
                print("=" * 80, flush=True)
                print(
                    f"[HP scenario analysis] Running scenario_id={sc.scenario_id!r} name={sc.name!r}",
                    flush=True,
                )
                print("=" * 80, flush=True)

            res, summ = _run_hp_online_mpc_1(
                access_power_by_month=access_power_by_month,
                verbose=mpc_verbose,
                log_prefix=log_prefix,
                **mpc_kw,
            )

            res.to_csv(results_path, index=False)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summ, f, indent=2)

            if verbose:
                print(
                    "[HP scenario analysis] Computing online annual net cost (billing)...",
                    flush=True,
                )
            online_net = compute_online_annual_net_cost_eur(project_root, res)
            row["online_net_cost_eur"] = float(online_net)

            per_outputs.append(
                HpScenarioRunOutputs(
                    scenario_id=sc.scenario_id,
                    name=sc.name,
                    results_15min_path=results_path,
                    summary_json_path=summary_path,
                    mpc_debug_csv_path=debug_path,
                )
            )
            if verbose:
                print(f"[HP scenario analysis] Saved: {results_path}", flush=True)
                print(f"[HP scenario analysis] Saved: {summary_path}", flush=True)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            row["error"] = err
            tb = traceback.format_exc()
            print(
                f"[HP scenario analysis] FAILED scenario_id={sc.scenario_id!r} name={sc.name!r}: {err}",
                flush=True,
            )
            print(tb)

        summary_rows.append(row)
        if row["error"] and stop_on_error:
            break

    if verbose:
        print(
            "[HP scenario analysis] Computing baseline + deterministic reference annual nets "
            "(billing; runs once after all MPC simulations)...",
            flush=True,
        )
    baseline_net = compute_baseline_annual_net_cost_eur(project_root)
    deterministic_net = compute_deterministic_annual_net_cost_eur(project_root)
    if verbose:
        print(
            f"[HP scenario analysis] Reference nets — baseline: {baseline_net:,.2f} EUR, "
            f"deterministic: {deterministic_net:,.2f} EUR",
            flush=True,
        )

    for row in summary_rows:
        row["baseline_net_cost_eur"] = float(baseline_net)
        row["deterministic_net_cost_eur"] = float(deterministic_net)
        err_s = str(row.get("error", "")).strip()
        if err_s:
            continue
        on_net = float(pd.to_numeric(row["online_net_cost_eur"], errors="coerce"))
        if not np.isfinite(on_net):
            continue
        row["online_savings_vs_baseline_eur"] = float(baseline_net - on_net)
        row["online_savings_vs_deterministic_eur"] = float(deterministic_net - on_net)

    master_path = out_dir / master_summary_filename
    pd.DataFrame(summary_rows).to_csv(master_path, index=False)
    if verbose:
        print("=" * 80, flush=True)
        print(f"[HP scenario analysis] Master summary: {master_path}", flush=True)
        print("=" * 80, flush=True)

    return HpScenarioAnalysisOutputs(master_summary_path=master_path, per_scenario=tuple(per_outputs))
