"""Batch scenario driver for joint EV+HP online MPC (notebook 11 Part 4)."""

from __future__ import annotations

import json
import re
import traceback
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from billing import calculate_monthly_bills, calculate_monthly_injection_bills, load_billing_config
from heat_pump_load import interpolate_cop, load_hp_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class EvHpOnlineScenario:
    scenario_id: Union[int, str]
    name: str
    forecast_strategy_ev: str = "c_p90"
    forecast_strategy_inflex: str = "c"
    forecast_strategy_inflex_stress: Optional[str] = None
    forecast_strategy_pv: str = "chronos2_elia_p50"
    forecast_strategy_thermal: str = "c2t_p50"
    forecast_strategy_temperature: str = "open_meteo_day_ahead"
    ev_deadline_slack_minutes: int = 105
    enforce_daily_ev_demand: bool = True
    hp_config_path: Optional[str] = None
    enforce_soc_min: bool = True
    enforce_soc_max: bool = True
    soc_slack_penalty_eur_per_soc: Optional[float] = None
    soc_min_slack_penalty_eur_per_soc: float = 1.0e6
    monthly_peak_price_multiplier: float = 1.0
    horizon_len: int = 96
    enable_mpc_window_debug: bool = False
    enable_forecast_stress_soc_floor: bool = False
    forecast_stress_soc_floor_strength: float = 0.5
    access_power_mode: str = "flex_aware"

    def to_run_kwargs(self) -> Dict[str, Any]:
        return {
            "forecast_strategy_ev": self.forecast_strategy_ev,
            "forecast_strategy_inflex": self.forecast_strategy_inflex,
            "forecast_strategy_inflex_stress": self.forecast_strategy_inflex_stress,
            "forecast_strategy_pv": self.forecast_strategy_pv,
            "forecast_strategy_thermal": self.forecast_strategy_thermal,
            "forecast_strategy_temperature": self.forecast_strategy_temperature,
            "ev_deadline_slack_minutes": self.ev_deadline_slack_minutes,
            "enforce_daily_ev_demand": self.enforce_daily_ev_demand,
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
class EvHpScenarioRunOutputs:
    scenario_id: Union[int, str]
    name: str
    results_15min_path: Path
    summary_json_path: Path


@dataclass(frozen=True)
class EvHpScenarioAnalysisOutputs:
    master_summary_path: Path
    per_scenario: Tuple[EvHpScenarioRunOutputs, ...]


def sanitize_scenario_name_for_filename(name: str) -> str:
    s = name.strip().lower().replace(" ", "_")
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    return re.sub(r"_+", "_", s).strip("._-") or "scenario"


def scenario_id_tag_for_filename(scenario_id: Union[int, str]) -> str:
    if isinstance(scenario_id, int):
        return f"{int(scenario_id):02d}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(scenario_id).strip())[:48] or "id"


def _load_plant_2025(project_root: Path) -> pd.DataFrame:
    plant = pd.read_csv(project_root / "data" / "plant1.csv")
    plant_ts = pd.to_datetime(plant["timestamp"], utc=True, errors="coerce")
    plant["timestamp"] = plant_ts.dt.tz_convert("Europe/Brussels").dt.tz_localize(None)
    return plant[
        (plant["timestamp"] >= pd.Timestamp("2025-01-01"))
        & (plant["timestamp"] < pd.Timestamp("2026-01-01"))
    ].sort_values("timestamp").reset_index(drop=True)


def compute_online_annual_net_cost_eur(project_root: Path, res: pd.DataFrame) -> float:
    billing_cfg = load_billing_config(str(project_root / "config" / "billing.yaml"))
    plant = _load_plant_2025(project_root)
    ev_col = "ev_applied" if "ev_applied" in res.columns else "ev_online_mpc"
    hp_col = "hp_applied_kwh" if "hp_applied_kwh" in res.columns else "hp_applied"
    plant_seq = plant[["timestamp", "inflex_load", "pv_production", "price"]].reset_index(drop=True)
    res_seq = res[[ev_col, hp_col, "access_kw"]].reset_index(drop=True)
    n = min(len(plant_seq), len(res_seq))
    df = pd.concat([plant_seq.iloc[:n], res_seq.iloc[:n]], axis=1)
    net = (
        df["inflex_load"].fillna(0.0)
        + df[ev_col].fillna(0.0)
        + df[hp_col].fillna(0.0)
        - df["pv_production"].fillna(0.0)
    )
    df["grid_consumption"] = net.clip(lower=0.0)
    df["grid_injection"] = (-net).clip(lower=0.0)
    bills = calculate_monthly_bills(
        df, billing_cfg, volume_col="grid_consumption", access_power_col="access_kw"
    )
    inj = calculate_monthly_injection_bills(df, billing_cfg, injection_col="grid_injection")
    bills["month_key"] = bills["month"].astype(str)
    inj["month_key"] = inj["month"].astype(str)
    merged = bills[["month_key", "total_cost_eur"]].merge(
        inj[["month_key", "injection_net_revenue_eur"]], on="month_key", how="left"
    )
    return float(
        (
            merged["total_cost_eur"]
            - merged["injection_net_revenue_eur"].fillna(0.0)
        ).sum()
    )


def compute_baseline_annual_net_cost_eur(
    project_root: Path,
    access_power_by_month: Optional[Dict[str, float]] = None,
) -> float:
    """Uncontrolled EV + uncontrolled HP; prefer notebook 04 baseline exports if present."""
    out_dir = project_root / "output" / "notebooks"
    base_bills_path = out_dir / "deterministic_ev_hp_monthly_baseline_bills_notebook_04.csv"
    base_inj_path = out_dir / "deterministic_ev_hp_monthly_baseline_injection_notebook_04.csv"
    if base_bills_path.exists() and base_inj_path.exists():
        base_bills = pd.read_csv(base_bills_path)
        base_inj = pd.read_csv(base_inj_path)
        for _df in (base_bills, base_inj):
            if "month" in _df.columns:
                _df["month_key"] = _df["month"].astype(str)
        base_net = base_bills[["month_key", "total_cost_eur"]].merge(
            base_inj[["month_key", "injection_net_revenue_eur"]],
            on="month_key",
            how="left",
        )
        return float(
            (
                base_net["total_cost_eur"] - base_net["injection_net_revenue_eur"].fillna(0.0)
            ).sum()
        )

    billing_cfg = load_billing_config(str(project_root / "config" / "billing.yaml"))
    plant = _load_plant_2025(project_root)
    hp_path = project_root / "output" / "uncontrolled_hp.csv"
    if not hp_path.exists():
        raise FileNotFoundError(f"Run notebook 03 first: {hp_path}")
    hp_un = pd.read_csv(hp_path)
    n = min(len(plant), len(hp_un))
    df = plant.iloc[:n].copy()
    df["hp_kwh"] = pd.to_numeric(
        hp_un["hp_electrical_load"].iloc[:n], errors="coerce"
    ).fillna(0.0)
    net = (
        df["inflex_load"].fillna(0.0)
        + df["ev"].fillna(0.0)
        + df["hp_kwh"]
        - df["pv_production"].fillna(0.0)
    )
    df["grid_consumption"] = net.clip(lower=0.0)
    df["grid_injection"] = (-net).clip(lower=0.0)
    df["month_key"] = df["timestamp"].dt.to_period("M").astype(str)
    if access_power_by_month is not None:
        df["access_kw"] = df["month_key"].map(access_power_by_month).astype(float)
    else:
        MARGIN_KW = 20.0
        BASELINE_2024_PEAK_GRID_KW = 2663.5
        months = pd.period_range("2025-01", "2025-12", freq="M")
        monthly_peak = (df.groupby(df["timestamp"].dt.to_period("M"))["grid_consumption"].max() * 4.0).reindex(
            months
        ).fillna(0.0)
        cummax = monthly_peak.cummax().shift(1)
        cummax.loc[months.min()] = BASELINE_2024_PEAK_GRID_KW
        cummax = cummax.fillna(BASELINE_2024_PEAK_GRID_KW)
        hp_cfg = load_hp_config(str(project_root / "config" / "hp.yaml"))
        cop_m10 = interpolate_cop(-10.0, hp_cfg["COP_data"])
        hp_add = float(df["thermal_load"].max()) * 4.0 / cop_m10
        access = cummax + MARGIN_KW + hp_add
        df["access_kw"] = df["month_key"].map(access.to_dict()).astype(float)

    bills = calculate_monthly_bills(
        df, billing_cfg, volume_col="grid_consumption", access_power_col="access_kw"
    )
    inj = calculate_monthly_injection_bills(df, billing_cfg, injection_col="grid_injection")
    bills["month_key"] = bills["month"].astype(str)
    inj["month_key"] = inj["month"].astype(str)
    merged = bills[["month_key", "total_cost_eur"]].merge(
        inj[["month_key", "injection_net_revenue_eur"]], on="month_key", how="left"
    )
    return float(
        (
            merged["total_cost_eur"]
            - merged["injection_net_revenue_eur"].fillna(0.0)
        ).sum()
    )


def compute_deterministic_joint_annual_net_cost_eur(
    project_root: Path,
    access_power_by_month: Optional[Dict[str, float]] = None,
) -> float:
    """Prefer notebook 04 monthly exports; fall back to ``deterministic_ev_hp.csv``."""
    out_dir = project_root / "output" / "notebooks"
    det_bills_path = out_dir / "deterministic_ev_hp_monthly_bills_notebook_04.csv"
    det_inj_path = out_dir / "deterministic_ev_hp_monthly_injection_notebook_04.csv"
    if det_bills_path.exists() and det_inj_path.exists():
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

    det_path = project_root / "output" / "optimised_ts" / "deterministic_ev_hp.csv"
    if not det_path.exists():
        raise FileNotFoundError(
            "Run notebook 04 first (export cell or deterministic_ev_hp.csv). "
            f"Missing: {det_bills_path} and {det_path}"
        )
    billing_cfg = load_billing_config(str(project_root / "config" / "billing.yaml"))
    plant = _load_plant_2025(project_root)
    det = pd.read_csv(det_path)
    ev_c = "ev_deterministic" if "ev_deterministic" in det.columns else "ev_charge"
    hp_c = "hp_deterministic" if "hp_deterministic" in det.columns else "hp_electrical_input"
    n = min(len(plant), len(det))
    df = plant.iloc[:n][["timestamp", "inflex_load", "pv_production", "price"]].copy()
    df["ev_kwh"] = pd.to_numeric(det[ev_c].iloc[:n], errors="coerce").fillna(0.0)
    df["hp_kwh"] = pd.to_numeric(det[hp_c].iloc[:n], errors="coerce").fillna(0.0)
    net = (
        df["inflex_load"]
        + df["ev_kwh"]
        + df["hp_kwh"]
        - df["pv_production"]
    )
    df["grid_consumption"] = net.clip(lower=0.0)
    df["grid_injection"] = (-net).clip(lower=0.0)
    df["month_key"] = df["timestamp"].dt.to_period("M").astype(str)
    if access_power_by_month is not None:
        df["access_kw"] = df["month_key"].map(access_power_by_month).astype(float)
    else:
        df["access_kw"] = 2700.0
    bills = calculate_monthly_bills(
        df, billing_cfg, volume_col="grid_consumption", access_power_col="access_kw"
    )
    inj = calculate_monthly_injection_bills(df, billing_cfg, injection_col="grid_injection")
    bills["month_key"] = bills["month"].astype(str)
    inj["month_key"] = inj["month"].astype(str)
    merged = bills[["month_key", "total_cost_eur"]].merge(
        inj[["month_key", "injection_net_revenue_eur"]], on="month_key", how="left"
    )
    return float(
        (
            merged["total_cost_eur"]
            - merged["injection_net_revenue_eur"].fillna(0.0)
        ).sum()
    )


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


def _json_key(key: Any) -> str:
    if isinstance(key, (str, int, float, bool)):
        return str(key)
    if isinstance(key, (date, datetime)):
        return key.isoformat()
    if isinstance(key, (np.integer,)):
        return str(int(key))
    return str(key)


def _summary_json_ready(obj: Any) -> Any:
    """Recursively convert summary dicts to strict-JSON-safe structures."""
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {_json_key(k): _summary_json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_summary_json_ready(x) for x in obj]
    return obj


def summary_json_path_for_scenario(
    out_dir: Path,
    sc: EvHpOnlineScenario,
    *,
    per_scenario_summary_stem: str = "online_ev_hp_summary_notebook_11_scenario",
) -> Path:
    sid_tag = scenario_id_tag_for_filename(sc.scenario_id)
    safe_name = sanitize_scenario_name_for_filename(sc.name)
    return out_dir / f"{per_scenario_summary_stem}_{sid_tag}_{safe_name}.json"


def write_ev_hp_summary_json(path: Path, summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_summary_json_ready(summary), f, indent=2)
    tmp.replace(path)


NB01_MARGIN_KW = 20.0
NB01_BASELINE_2024_PEAK_GRID_KW = 2663.5


def nb01_conservative_access_kw_by_month(plant: pd.DataFrame) -> Dict[str, float]:
    """
    Notebook 01 EV billing: monthly cummax on full-site ``grid_consumption`` peaks
    (M−1) + margin; January seeded from 2024 peak.
    """
    df = plant.copy()
    df["month"] = df["timestamp"].dt.to_period("M")
    months = pd.period_range("2025-01", "2025-12", freq="M")
    monthly_peak_kw = (
        (df.groupby("month")["grid_consumption"].max() * 4.0)
        .reindex(months)
        .fillna(0.0)
    )
    cummax_m_minus_1_kw = monthly_peak_kw.cummax().shift(1)
    cummax_m_minus_1_kw.loc[months.min()] = NB01_BASELINE_2024_PEAK_GRID_KW
    cummax_m_minus_1_kw = cummax_m_minus_1_kw.fillna(NB01_BASELINE_2024_PEAK_GRID_KW)
    access_kw = cummax_m_minus_1_kw + NB01_MARGIN_KW
    return {str(k): float(v) for k, v in access_kw.to_dict().items()}


def compute_nb01_site_reference_bills(project_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Notebook 01 site reference R: ``grid_consumption_excl_ev`` (no EV on meter),
    conservative access from full-site ``grid_consumption`` peaks (same AP for both
    nb01 baseline and with_ev rows). Matches nb01 ``total_cost_eur`` (offtake only).
    """
    billing_cfg = load_billing_config(str(project_root / "config" / "billing.yaml"))
    plant = _load_plant_2025(project_root)
    plant.columns = plant.columns.str.strip()
    if "grid_consumption_excl_ev" not in plant.columns:
        raise KeyError(
            "plant1.csv missing column grid_consumption_excl_ev (notebook 01 benchmark)"
        )
    access_by_month = nb01_conservative_access_kw_by_month(plant)
    df = plant[["timestamp", "price", "grid_consumption_excl_ev"]].copy()
    df["month_key"] = df["timestamp"].dt.to_period("M").astype(str)
    df["access_kw"] = df["month_key"].map(access_by_month).astype(float)
    bills = calculate_monthly_bills(
        df,
        billing_cfg,
        volume_col="grid_consumption_excl_ev",
        price_col="price",
        timestamp_col="timestamp",
        access_power_col="access_kw",
    )
    if "month" in bills.columns:
        bills["month_key"] = bills["month"].astype(str)
    inj = bills[["month_key"]].copy() if "month_key" in bills.columns else pd.DataFrame()
    inj["injection_net_revenue_eur"] = 0.0
    return bills, inj


def compute_nb01_site_reference_annual_offtake_eur(project_root: Path) -> float:
    """Notebook 01 baseline row (offtake ``total_cost_eur``, no injection subtracted)."""
    bills, _ = compute_nb01_site_reference_bills(project_root)
    return float(bills["total_cost_eur"].sum())


def compute_inflex_site_bills(
    project_root: Path,
    access_power_by_month: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Alias for :func:`compute_nb01_site_reference_bills` (``access_power_by_month`` ignored)."""
    del access_power_by_month
    return compute_nb01_site_reference_bills(project_root)


def compute_online_bills_from_results(
    project_root: Path, res: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly offtake + injection bills from a Part 4B 15-min export."""
    billing_cfg = load_billing_config(str(project_root / "config" / "billing.yaml"))
    plant = _load_plant_2025(project_root)
    ev_col = "ev_applied" if "ev_applied" in res.columns else "ev_online_mpc"
    hp_col = "hp_applied_kwh" if "hp_applied_kwh" in res.columns else "hp_applied"
    plant_seq = plant[["timestamp", "inflex_load", "pv_production", "price"]].reset_index(drop=True)
    res_seq = res[[ev_col, hp_col, "access_kw"]].reset_index(drop=True)
    n = min(len(plant_seq), len(res_seq))
    df = pd.concat([plant_seq.iloc[:n], res_seq.iloc[:n]], axis=1)
    net = (
        df["inflex_load"].fillna(0.0)
        + df[ev_col].fillna(0.0)
        + df[hp_col].fillna(0.0)
        - df["pv_production"].fillna(0.0)
    )
    df["grid_consumption"] = net.clip(lower=0.0)
    df["grid_injection"] = (-net).clip(lower=0.0)
    bills = calculate_monthly_bills(
        df, billing_cfg, volume_col="grid_consumption", access_power_col="access_kw"
    )
    inj = calculate_monthly_injection_bills(df, billing_cfg, injection_col="grid_injection")
    for _df in (bills, inj):
        if "month" in _df.columns:
            _df["month_key"] = _df["month"].astype(str)
    return bills, inj


def build_summary_dict_from_results(
    project_root: Path,
    res: pd.DataFrame,
    sc: EvHpOnlineScenario,
) -> Dict[str, Any]:
    """Rebuild Part 4 summary metadata from a scenario CSV (for Part 4C JSON export)."""
    res = res.copy()
    if "timestamp" in res.columns:
        res["timestamp"] = pd.to_datetime(res["timestamp"], errors="coerce")
    bills, inj_bills = compute_online_bills_from_results(project_root, res)

    uncharged_by_day: Dict[str, float] = {}
    if "uncharged_kwh" in res.columns:
        day_col = "date" if "date" in res.columns else None
        if day_col is None and "timestamp" in res.columns:
            res["_day"] = res["timestamp"].dt.date.astype(str)
            day_col = "_day"
        if day_col is not None:
            for day, grp in res.groupby(day_col, sort=False):
                uncharged_by_day[str(day)] = float(
                    pd.to_numeric(grp["uncharged_kwh"], errors="coerce").max()
                )

    monthly_peak: Dict[str, float] = {}
    if "month_key" in res.columns and "p_grid_actual_kw" in res.columns:
        gc = pd.to_numeric(res["p_grid_actual_kw"], errors="coerce").fillna(0.0).clip(lower=0.0)
        tmp = res.assign(_gc_kw=gc)
        monthly_peak = (
            tmp.groupby("month_key", sort=False)["_gc_kw"]
            .max()
            .astype(float)
            .to_dict()
        )
        monthly_peak = {str(k): float(v) for k, v in monthly_peak.items()}

    ev_enforce_steps = 0
    ev_enforce_extra = 0.0
    if "ev_enforce_active" in res.columns:
        ev_enforce_steps = int((pd.to_numeric(res["ev_enforce_active"], errors="coerce").fillna(0) > 0).sum())
    if "ev_enforce_extra_kwh" in res.columns:
        ev_enforce_extra = float(pd.to_numeric(res["ev_enforce_extra_kwh"], errors="coerce").fillna(0).sum())

    return {
        "n_steps": int(len(res)),
        "monthly_peak_so_far": monthly_peak,
        "uncharged_kwh_by_day": uncharged_by_day,
        "bills": bills,
        "injection_bills": inj_bills,
        "forecast_strategy_ev": sc.forecast_strategy_ev,
        "forecast_strategy_inflex": sc.forecast_strategy_inflex,
        "forecast_strategy_inflex_stress": sc.forecast_strategy_inflex_stress,
        "forecast_strategy_pv": sc.forecast_strategy_pv,
        "forecast_strategy_thermal": sc.forecast_strategy_thermal,
        "ev_deadline_slack_minutes": sc.ev_deadline_slack_minutes,
        "enforce_daily_ev_demand": sc.enforce_daily_ev_demand,
        "ev_enforce_steps": ev_enforce_steps,
        "ev_enforce_extra_kwh_total": ev_enforce_extra,
        "enable_forecast_stress_soc_floor": sc.enable_forecast_stress_soc_floor,
        "forecast_stress_soc_floor_strength": sc.forecast_stress_soc_floor_strength,
        "access_power_mode": sc.access_power_mode,
    }


def resolve_scenario_for_results_path(
    scenarios: Sequence[EvHpOnlineScenario],
    results_path: Path,
) -> Optional[EvHpOnlineScenario]:
    stem = results_path.stem
    prefix = "online_ev_hp_15min_notebook_11_scenario_"
    if not stem.startswith(prefix):
        return None
    name_part = stem[len(prefix) :]
    if "_" in name_part:
        _, name_part = name_part.split("_", 1)
    for sc in scenarios:
        if sanitize_scenario_name_for_filename(sc.name) == name_part:
            return sc
    return None


def _resolve_access_power_by_month(
    sc: EvHpOnlineScenario,
    *,
    access_power_flex_by_month: Dict[str, float],
    access_power_deterministic_by_month: Dict[str, float],
) -> Dict[str, float]:
    mode = (sc.access_power_mode or "flex_aware").strip().lower()
    if mode == "flex_aware":
        return access_power_flex_by_month
    if mode == "deterministic":
        return access_power_deterministic_by_month
    raise ValueError(
        f"scenario {sc.scenario_id}: access_power_mode must be 'flex_aware' or 'deterministic', got {sc.access_power_mode!r}"
    )


def run_ev_hp_online_scenario_analysis(
    *,
    scenarios: Sequence[EvHpOnlineScenario],
    access_power_flex_by_month: Optional[Dict[str, float]] = None,
    access_power_deterministic_by_month: Optional[Dict[str, float]] = None,
    access_power_by_month: Optional[Dict[str, float]] = None,
    out_dir: Optional[Path] = None,
    per_scenario_15min_stem: str = "online_ev_hp_15min_notebook_11_scenario",
    per_scenario_summary_stem: str = "online_ev_hp_summary_notebook_11_scenario",
    master_summary_filename: str = "online_ev_hp_scenario_analysis_summary_notebook_11.csv",
    stop_on_error: bool = False,
    verbose: bool = True,
    mpc_verbose: bool = True,
    project_root: Optional[Path] = None,
    baseline_access_power_by_month: Optional[Dict[str, float]] = None,
) -> EvHpScenarioAnalysisOutputs:
    if project_root is None:
        project_root = PROJECT_ROOT
    if out_dir is None:
        out_dir = project_root / "output" / "notebooks"
    out_dir.mkdir(parents=True, exist_ok=True)

    if access_power_by_month is not None:
        access_power_flex_by_month = access_power_flex_by_month or access_power_by_month
        access_power_deterministic_by_month = (
            access_power_deterministic_by_month or access_power_by_month
        )
    if access_power_flex_by_month is None or access_power_deterministic_by_month is None:
        raise ValueError(
            "Provide access_power_flex_by_month and access_power_deterministic_by_month "
            "(or legacy access_power_by_month for both)."
        )

    from online_MPC_1_EV_HP import run_ev_hp_online_mpc_1 as _run

    per_outputs: List[EvHpScenarioRunOutputs] = []
    summary_rows: List[Dict[str, Any]] = []

    for sc in scenarios:
        sid_tag = scenario_id_tag_for_filename(sc.scenario_id)
        safe_name = sanitize_scenario_name_for_filename(sc.name)
        results_path = out_dir / f"{per_scenario_15min_stem}_{sid_tag}_{safe_name}.csv"
        summary_path = summary_json_path_for_scenario(
            out_dir, sc, per_scenario_summary_stem=per_scenario_summary_stem
        )
        mpc_kw = dict(sc.to_run_kwargs())
        mpc_kw.pop("verbose", None)
        mpc_kw.pop("log_prefix", None)
        mpc_kw.pop("access_power_by_month", None)

        log_prefix = f"[Online MPC - scenario {sc.scenario_id}: {sc.name}]"

        row: Dict[str, Any] = {
            "scenario_id": sc.scenario_id,
            "scenario_name": sc.name,
            "access_power_mode": sc.access_power_mode,
            "forecast_strategy_inflex": sc.forecast_strategy_inflex,
            "forecast_strategy_inflex_stress": sc.forecast_strategy_inflex_stress or "",
            "forecast_stress_soc_floor_strength": sc.forecast_stress_soc_floor_strength,
            "results_15min_path": str(results_path),
            "summary_json_path": str(summary_path),
            "scenario_kwargs_json": json.dumps(_scenario_json_ready(asdict(sc)), sort_keys=True),
            "error": "",
            "online_net_cost_eur": float("nan"),
            "baseline_net_cost_eur": float("nan"),
            "deterministic_net_cost_eur": float("nan"),
            "online_savings_vs_baseline_eur": float("nan"),
            "online_savings_vs_deterministic_eur": float("nan"),
        }

        try:
            ap = _resolve_access_power_by_month(
                sc,
                access_power_flex_by_month=access_power_flex_by_month,
                access_power_deterministic_by_month=access_power_deterministic_by_month,
            )
            if verbose:
                print("=" * 80, flush=True)
                print(
                    f"[EV+HP scenario analysis] Running scenario_id={sc.scenario_id!r} "
                    f"name={sc.name!r} access={sc.access_power_mode!r}",
                    flush=True,
                )
                print("=" * 80, flush=True)

            res, _summ = _run(
                access_power_by_month=ap,
                verbose=mpc_verbose,
                log_prefix=log_prefix,
                **mpc_kw,
            )

            res.to_csv(results_path, index=False)

            if verbose:
                print(
                    "[EV+HP scenario analysis] Computing online annual net cost (billing)...",
                    flush=True,
                )
            online_net = compute_online_annual_net_cost_eur(project_root, res)
            row["online_net_cost_eur"] = float(online_net)

            per_outputs.append(
                EvHpScenarioRunOutputs(
                    scenario_id=sc.scenario_id,
                    name=sc.name,
                    results_15min_path=results_path,
                    summary_json_path=summary_path,
                )
            )
            if verbose:
                print(f"[EV+HP scenario analysis] Saved: {results_path}", flush=True)
                print(
                    f"[EV+HP scenario analysis] Summary JSON deferred to Part 4C: {summary_path}",
                    flush=True,
                )
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            row["error"] = err
            tb = traceback.format_exc()
            print(
                f"[EV+HP scenario analysis] FAILED scenario_id={sc.scenario_id!r} "
                f"name={sc.name!r}: {err}",
                flush=True,
            )
            print(tb, flush=True)

        summary_rows.append(row)
        if row["error"] and stop_on_error:
            break

    if verbose:
        print(
            "[EV+HP scenario analysis] Computing baseline + deterministic reference annual nets "
            "(billing; runs once after all MPC simulations)...",
            flush=True,
        )
    baseline_net = compute_baseline_annual_net_cost_eur(
        project_root, access_power_by_month=baseline_access_power_by_month
    )
    deterministic_net = compute_deterministic_joint_annual_net_cost_eur(project_root)
    if verbose:
        print(
            f"[EV+HP scenario analysis] Reference nets — baseline: {baseline_net:,.2f} EUR, "
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
        print(f"[EV+HP scenario analysis] Master summary: {master_path}", flush=True)
        print("=" * 80, flush=True)

    return EvHpScenarioAnalysisOutputs(
        master_summary_path=master_path, per_scenario=tuple(per_outputs)
    )


def build_heating_charging_cost_table(
    *,
    baseline_net_eur: float,
    offline_net_eur: float,
    online_net_eur: float,
    inflex_site_net_eur: float,
) -> pd.DataFrame:
    """Annual flex-load (EV + HP) cost = full net minus notebook 01 site reference R."""
    return pd.DataFrame(
        {
            "Baseline (uncontrolled)": [
                baseline_net_eur,
                inflex_site_net_eur,
                baseline_net_eur - inflex_site_net_eur,
            ],
            "Offline (deterministic joint)": [
                offline_net_eur,
                inflex_site_net_eur,
                offline_net_eur - inflex_site_net_eur,
            ],
            "Online (scenario MPC)": [
                online_net_eur,
                inflex_site_net_eur,
                online_net_eur - inflex_site_net_eur,
            ],
        },
        index=[
            "Full annual net cost [EUR]",
            "Site only — grid_consumption_excl_ev (notebook 01) [EUR]",
            "Heating & charging cost [EUR]",
        ],
    )


def backfill_master_summary_online_costs(
    project_root: Optional[Path] = None,
    *,
    master_summary_filename: str = "online_ev_hp_scenario_analysis_summary_notebook_11.csv",
    baseline_access_power_by_month: Optional[Dict[str, float]] = None,
    verbose: bool = True,
) -> Path:
    """
    Recompute ``online_net_cost_eur`` (and savings columns) from existing Part 4B CSVs
    without re-running MPC. Clears stale JSON-export errors in the master summary.
    """
    if project_root is None:
        project_root = PROJECT_ROOT
    master_path = project_root / "output" / "notebooks" / master_summary_filename
    if not master_path.exists():
        raise FileNotFoundError(master_path)

    df = pd.read_csv(master_path)
    if "error" in df.columns:
        df["error"] = df["error"].astype("object")
    baseline_net = compute_baseline_annual_net_cost_eur(
        project_root, access_power_by_month=baseline_access_power_by_month
    )
    deterministic_net = compute_deterministic_joint_annual_net_cost_eur(project_root)

    for idx, row in df.iterrows():
        csv_path = Path(str(row.get("results_15min_path", "")))
        if not csv_path.is_file():
            if verbose:
                print(f"[backfill] skip row {row.get('scenario_id')}: missing {csv_path}")
            continue
        res = pd.read_csv(csv_path)
        on_net = compute_online_annual_net_cost_eur(project_root, res)
        df.at[idx, "online_net_cost_eur"] = float(on_net)
        df.at[idx, "error"] = ""
        df.at[idx, "baseline_net_cost_eur"] = float(baseline_net)
        df.at[idx, "deterministic_net_cost_eur"] = float(deterministic_net)
        df.at[idx, "online_savings_vs_baseline_eur"] = float(baseline_net - on_net)
        df.at[idx, "online_savings_vs_deterministic_eur"] = float(deterministic_net - on_net)
        if verbose:
            print(
                f"[backfill] scenario {row.get('scenario_id')}: "
                f"online_net={on_net:,.2f} EUR"
            )

    df.to_csv(master_path, index=False)
    if verbose:
        print(f"[backfill] Updated: {master_path}")
    return master_path
