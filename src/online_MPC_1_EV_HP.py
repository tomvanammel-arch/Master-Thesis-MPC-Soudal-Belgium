"""
Joint EV + HP online myopic MPC (thesis §3.7.4).

Full-year rolling simulation: ``mpc_ev_hp_24h`` planner + joint real-time clipping
(shared proportional clip on flexible EV/HP plan power above HP SOC-min floor), optional headroom-aware enforce daily EV demand (12:00-17:00; defer when envelope headroom suffices, else minimal restore), EV catch-up enforce,
HP PLC safeguard.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).parent
PROJECT_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from optimization import mpc_ev_hp_24h  # type: ignore
from online_MPC_1_EV import (  # type: ignore
    _apply_ev_enforce_minimal_after_clip,
    _ev_envelope_energy_remaining_kwh,
    _ev_envelope_headroom_after_step_kwh,
)
from heat_pump_load import load_hp_config, interpolate_cop  # type: ignore
from billing import (  # type: ignore
    load_billing_config,
    calculate_monthly_bills,
    calculate_monthly_injection_bills,
)


def _parse_plant_data(plant_path: Path) -> pd.DataFrame:
    df = pd.read_csv(plant_path)
    ts_utc = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["timestamp"] = ts_utc.dt.tz_convert("Europe/Brussels").dt.tz_localize(None)
    df = df.sort_values("timestamp").reset_index(drop=True)
    start_2025 = pd.Timestamp("2025-01-01 00:00:00")
    end_2025 = pd.Timestamp("2026-01-01 00:00:00")
    df = df[(df["timestamp"] >= start_2025) & (df["timestamp"] < end_2025)].copy()
    return df.sort_values("timestamp").reset_index(drop=True)


def _load_forecast_column(
    path: Path,
    strategy: str,
    prefix: str,
    hint_prefix: str,
) -> np.ndarray:
    df = pd.read_csv(path)
    col = strategy if strategy.startswith(prefix) else f"{prefix}{strategy}"
    if col not in df.columns:
        if strategy.endswith("_p50"):
            alt = strategy[: -len("_p50")]
            alt_col = alt if alt.startswith(prefix) else f"{prefix}{alt}"
            if alt_col in df.columns:
                col = alt_col
            else:
                hint = [c for c in df.columns if c.startswith(hint_prefix)]
                raise KeyError(
                    f"No column for {strategy!r} in {path.name}. Expected {col!r}. "
                    f"Available: {hint[:14]}"
                )
        else:
            hint = [c for c in df.columns if c.startswith(hint_prefix)]
            raise KeyError(f"No column for {strategy!r} in {path.name}. Available: {hint[:14]}")
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)


def _month_key(ts: pd.Timestamp) -> str:
    return ts.to_period("M").strftime("%Y-%m")


def _build_dynamic_ev_envelope(ev_kwh_arr: np.ndarray, timestamps: pd.Series) -> np.ndarray:
    ev_kwh = np.asarray(ev_kwh_arr, dtype=float)
    power_bench = ev_kwh * 4.0
    ts_local = pd.to_datetime(timestamps)
    dates = ts_local.dt.date.values
    tod = ts_local.dt.hour.values + ts_local.dt.minute.values / 60.0
    env = np.zeros(len(ts_local), dtype=float)
    for d in sorted(np.unique(dates)):
        mask = dates == d
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        day_power = power_bench[idxs]
        day_tod = tod[idxs]
        cum_max = np.maximum.accumulate(day_power)
        mask_1530 = day_tod <= 15.5
        i_1530 = np.where(mask_1530)[0][-1] if np.any(mask_1530) else 0
        p_max_1530 = cum_max[i_1530] if len(cum_max) else 0.0
        for j, idx in enumerate(idxs):
            t = day_tod[j]
            if t <= 15.5:
                env[idx] = cum_max[j]
            elif 15.5 < t < 17.0:
                env[idx] = p_max_1530 * (17.0 - t) / 1.5
            else:
                env[idx] = 0.0
    return env


def _joint_clip_first_step(
    *,
    inflex_act_kwh: float,
    pv_act_kwh: float,
    ev_plan_kwh: float,
    hp_plan_kwh: float,
    cop_act: float,
    soc: float,
    soc_min_phys: float,
    buffer_capacity_kwh_th: float,
    thermal_act_kwh_th: float,
    loss_kwh: float,
    p_limit_kw: float,
) -> Tuple[float, float, float, float, bool]:
    """
    Joint grid clip (thesis Eq. 3.50 limit): share overload proportionally between
    flexible EV plan power and HP plan power above the SOC-min electrical floor.

    Returns (ev_applied_kwh, hp_applied_kwh, p_grid_plan_kw, p_grid_after_clip_kw, was_clipped).
    """
    ev_plan_kw = 4.0 * ev_plan_kwh
    hp_plan_kw = 4.0 * hp_plan_kwh
    p_grid_plan = 4.0 * (inflex_act_kwh + ev_plan_kwh + hp_plan_kwh - pv_act_kwh)

    if p_grid_plan <= p_limit_kw + 1e-9:
        return ev_plan_kwh, hp_plan_kwh, p_grid_plan, p_grid_plan, False

    p_clip_req = p_grid_plan - p_limit_kw

    if cop_act > 1e-9:
        e_hp_socmin = max(
            ((soc_min_phys - soc) * buffer_capacity_kwh_th + thermal_act_kwh_th + loss_kwh)
            / cop_act,
            0.0,
        )
    else:
        e_hp_socmin = 0.0
    p_hp_socmin = 4.0 * e_hp_socmin
    hp_flex = max(hp_plan_kw - p_hp_socmin, 0.0)
    ev_flex = ev_plan_kw
    total_flex = hp_flex + ev_flex

    if total_flex <= 1e-9:
        p_ev_clip = 0.0
        p_hp_clip = min(p_clip_req, hp_flex)
    else:
        p_ev_clip = p_clip_req * (ev_flex / total_flex)
        p_hp_clip = p_clip_req * (hp_flex / total_flex)
        p_ev_clip = min(p_ev_clip, ev_flex)
        p_hp_clip = min(p_hp_clip, hp_flex)
        rem = p_clip_req - p_ev_clip - p_hp_clip
        while rem > 1e-6:
            ev_head = ev_flex - p_ev_clip
            hp_head = hp_flex - p_hp_clip
            if ev_head <= 1e-9 and hp_head <= 1e-9:
                break
            if ev_head >= hp_head:
                add = min(rem, ev_head)
                p_ev_clip += add
            else:
                add = min(rem, hp_head)
                p_hp_clip += add
            rem = p_clip_req - p_ev_clip - p_hp_clip

    p_ev_applied = ev_plan_kw - p_ev_clip
    p_hp_applied = hp_plan_kw - p_hp_clip
    p_grid_clip = 4.0 * (
        inflex_act_kwh + p_ev_applied / 4.0 + p_hp_applied / 4.0 - pv_act_kwh
    )
    return (
        p_ev_applied / 4.0,
        p_hp_applied / 4.0,
        p_grid_plan,
        p_grid_clip,
        True,
    )


def _hp_shave_after_ev_enforce(
    *,
    ev_app_kwh: float,
    hp_kwh: float,
    inflex_act_kwh: float,
    pv_act_kwh: float,
    p_limit_kw: float,
    cop_act: float,
    soc: float,
    soc_min_phys: float,
    buffer_capacity_kwh_th: float,
    thermal_act_kwh_th: float,
    loss_kwh: float,
) -> Tuple[float, float]:
    """
    After EV enforce raises charging, reduce HP (drain buffer) so grid stays at p_limit,
    but not below the electrical input needed to keep SOC >= soc_min_phys.
    Returns (hp_kwh_after, shaved_kwh).
    """
    p_after_kw = 4.0 * (inflex_act_kwh + ev_app_kwh + hp_kwh - pv_act_kwh)
    if p_after_kw <= p_limit_kw + 1e-9:
        return float(hp_kwh), 0.0
    e_shave_kwh = (p_after_kw - p_limit_kw) / 4.0
    if cop_act > 1e-9:
        e_hp_socmin = max(
            (
                (soc_min_phys - soc) * buffer_capacity_kwh_th
                + thermal_act_kwh_th
                + loss_kwh
            )
            / cop_act,
            0.0,
        )
    else:
        e_hp_socmin = 0.0
    hp_new = float(max(hp_kwh - e_shave_kwh, e_hp_socmin, 0.0))
    return hp_new, float(max(hp_kwh - hp_new, 0.0))


def run_ev_hp_online_mpc_1(
    forecast_strategy_ev: str = "c_p90",
    forecast_strategy_inflex: str = "c",
    forecast_strategy_inflex_stress: Optional[str] = None,
    forecast_strategy_pv: str = "chronos2_elia_p50",
    forecast_strategy_thermal: str = "c2t_p50",
    forecast_strategy_temperature: str = "open_meteo_day_ahead",
    ev_deadline_slack_minutes: int = 105,
    enforce_daily_ev_demand: bool = True,
    access_power_by_month: Dict[str, float] = None,
    hp_config_path: Optional[str] = None,
    enforce_soc_min: bool = True,
    enforce_soc_max: bool = True,
    soc_slack_penalty_eur_per_soc: Optional[float] = None,
    soc_min_slack_penalty_eur_per_soc: float = 1.0e6,
    monthly_peak_price_multiplier: float = 1.0,
    horizon_len: int = 96,
    enable_mpc_window_debug: bool = False,
    enable_forecast_stress_soc_floor: bool = False,
    forecast_stress_soc_floor_strength: float = 0.5,
    verbose: bool = True,
    log_prefix: str = "[Online MPC]",
    mpc_window_debug_csv_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict]:
    plant_path = PROJECT_ROOT / "data" / "plant1.csv"
    forecast_ev_path = PROJECT_ROOT / "output" / "forecast" / "forecast_ev_rolling_horizon.csv"
    forecast_inflex_path = (
        PROJECT_ROOT / "output" / "forecast" / "forecast_inflex_load_rolling_horizon.csv"
    )
    forecast_pv_path = PROJECT_ROOT / "output" / "forecast" / "forecast_pv_rolling_horizon.csv"
    forecast_thermal_path = (
        PROJECT_ROOT / "output" / "forecast" / "forecast_thermal_load_rolling_horizon.csv"
    )
    temperature_forecast_path = (
        PROJECT_ROOT
        / "data"
        / "temperature_forecast_day_ahead_open_meteo_Turnhout_15min.csv"
    )
    billing_path = PROJECT_ROOT / "config" / "billing.yaml"
    if hp_config_path is None:
        hp_config_path = str(PROJECT_ROOT / "config" / "hp.yaml")

    effective_inflex_stress = (
        forecast_strategy_inflex
        if forecast_strategy_inflex_stress is None
        else forecast_strategy_inflex_stress
    )

    if verbose:
        print("=" * 80, flush=True)
        print("Joint EV+HP Online MPC 1 – Full-year simulation", flush=True)
        print("=" * 80, flush=True)
        print(f"  EV forecast strategy:       {forecast_strategy_ev}", flush=True)
        print(f"  Inflex forecast strategy:   {forecast_strategy_inflex}", flush=True)
        print(f"  Inflex strategy (stress):   {effective_inflex_stress}", flush=True)
        print(f"  PV forecast strategy:       {forecast_strategy_pv}", flush=True)
        print(f"  Thermal forecast strategy:  {forecast_strategy_thermal}", flush=True)
        print(f"  Temperature strategy:       {forecast_strategy_temperature}", flush=True)
        print(f"  EV deadline slack (min):      {ev_deadline_slack_minutes}", flush=True)
        print(f"  Enforce daily EV demand:    {enforce_daily_ev_demand}", flush=True)
        print(f"  Enforce SOC min:            {enforce_soc_min}", flush=True)
        print(f"  Enforce SOC max:            {enforce_soc_max}", flush=True)
        _pen = (
            soc_min_slack_penalty_eur_per_soc
            if soc_slack_penalty_eur_per_soc is None
            else soc_slack_penalty_eur_per_soc
        )
        print(f"  Planner SOC slack penalty (€/SOC): {_pen}", flush=True)
        print(
            "  Planner monthly peak price multiplier: "
            f"{float(monthly_peak_price_multiplier)}",
            flush=True,
        )
        print(
            f"  Forecast stress SOC floor:  {enable_forecast_stress_soc_floor} "
            f"(strength={forecast_stress_soc_floor_strength})",
            flush=True,
        )
        print(f"  Plant data:            {plant_path}", flush=True)
        print(f"  EV forecast:           {forecast_ev_path}", flush=True)
        print(f"  Inflex forecast:       {forecast_inflex_path}", flush=True)
        print(f"  PV forecast:           {forecast_pv_path}", flush=True)
        print(f"  Thermal forecast:      {forecast_thermal_path}", flush=True)
        print(f"  Temperature forecast:  {temperature_forecast_path}", flush=True)
        print(f"  Billing config:        {billing_path}", flush=True)
        print(f"  HP config:             {hp_config_path}", flush=True)
        print("-" * 80, flush=True)

    if access_power_by_month is None:
        raise ValueError("run_ev_hp_online_mpc_1 requires access_power_by_month.")

    plant_df = _parse_plant_data(plant_path)
    n = len(plant_df)
    month_keys_in_data = sorted({_month_key(ts) for ts in plant_df["timestamp"]})
    missing = [m for m in month_keys_in_data if m not in access_power_by_month]
    if missing:
        raise KeyError(f"access_power_by_month missing: {', '.join(missing)}")

    ev_fc = _load_forecast_column(
        forecast_ev_path, forecast_strategy_ev, "forecast_ev_", "forecast_ev_"
    )
    inflex_fc = _load_forecast_column(
        forecast_inflex_path,
        forecast_strategy_inflex,
        "forecast_inflex_",
        "forecast_inflex_",
    )
    inflex_fc_stress = (
        _load_forecast_column(
            forecast_inflex_path,
            effective_inflex_stress,
            "forecast_inflex_",
            "forecast_inflex_",
        )
        if enable_forecast_stress_soc_floor
        else inflex_fc
    )

    if forecast_strategy_pv == "actual":
        raise ValueError("PV strategy 'actual' disabled for online MPC.")
    pv_df = pd.read_csv(forecast_pv_path)
    pv_col = (
        forecast_strategy_pv
        if forecast_strategy_pv.startswith("pv_forecast_kWh_15min_")
        else f"pv_forecast_kWh_15min_{forecast_strategy_pv}"
    )
    if pv_col not in pv_df.columns:
        raise KeyError(f"Missing PV column {pv_col!r}")
    pv_fc = pd.to_numeric(pv_df[pv_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    thermal_fc = _load_forecast_column(
        forecast_thermal_path,
        forecast_strategy_thermal,
        "forecast_thermal_",
        "forecast_thermal_",
    )

    if forecast_strategy_temperature not in {"open_meteo_day_ahead", "day_ahead_open_meteo"}:
        raise ValueError("Use temperature strategy 'open_meteo_day_ahead'.")
    temp_df = pd.read_csv(temperature_forecast_path)
    temp_cols = [c for c in temp_df.columns if c != "timestamp"]
    temp_fc = pd.to_numeric(temp_df[temp_cols[0]], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    for name, arr in (
        ("ev", ev_fc),
        ("inflex", inflex_fc),
        ("pv", pv_fc),
        ("thermal", thermal_fc),
        ("temp", temp_fc),
    ):
        if len(arr) != n:
            raise ValueError(f"{name} forecast length {len(arr)} != plant {n}")

    plant_df = plant_df.copy()
    plant_df["inflex_forecast"] = inflex_fc
    plant_df["pv_for_mpc"] = pv_fc
    plant_df["thermal_forecast"] = thermal_fc
    plant_df["ev_forecast"] = ev_fc
    plant_df["outdoor_temperature_for_mpc"] = temp_fc
    plant_df["date"] = plant_df["timestamp"].dt.date

    ts = plant_df["timestamp"]
    ev_actual = plant_df["ev"].astype(float).values
    ev_power_envelope_actual_kw = _build_dynamic_ev_envelope(ev_actual, ts)
    ev_power_envelope_forecast_kw = _build_dynamic_ev_envelope(ev_fc, ts)
    ev_power_envelope_base_kw = ev_power_envelope_actual_kw.copy()

    hp_cfg = load_hp_config(hp_config_path)
    soc_min_phys = float(hp_cfg["buffer"]["soc_min"])
    soc_max_phys = float(hp_cfg["buffer"]["soc_max"])
    soc = float(hp_cfg["buffer"]["soc_initial"])
    loss_coeff_per_hour = float(hp_cfg["buffer"]["loss_coefficient_per_hour"])
    buf = hp_cfg["buffer"]
    buffer_capacity_kwh_th = (
        float(buf["size_m3"])
        * float(buf["water_density_kg_per_m3"])
        * float(buf["cp_kj_per_kg_k"])
        * float(buf["usable_delta_t_k"])
    ) / 3600.0
    loss_rate_per_interval = loss_coeff_per_hour / 4.0

    access_kw_full = np.array(
        [float(access_power_by_month[_month_key(t)]) for t in plant_df["timestamp"]],
        dtype=float,
    )
    temp_fc_arr = plant_df["outdoor_temperature_for_mpc"].to_numpy(dtype=float)
    cop_fc = np.array(
        [
            float(interpolate_cop(float(t), hp_cfg["COP_data"])) if not np.isnan(t) else 2.5
            for t in temp_fc_arr
        ],
        dtype=float,
    )
    thermal_fc_kwh = plant_df["thermal_forecast"].to_numpy(dtype=float)
    hp_est_kwh = np.where(cop_fc > 1e-9, thermal_fc_kwh / cop_fc, 0.0)
    forecast_grid_kw_full = 4.0 * (
        np.array(inflex_fc_stress, dtype=float)
        + ev_fc
        + hp_est_kwh
        - pv_fc
    )
    stress_active = forecast_grid_kw_full > access_kw_full
    _strength = float(np.clip(forecast_stress_soc_floor_strength, 0.0, 1.0))
    soc_min_target = float(
        min(max(soc_min_phys + _strength * (soc_max_phys - soc_min_phys), soc_min_phys), soc_max_phys)
    )
    soc_min_schedule = np.full(n, soc_min_phys, dtype=float)
    if enable_forecast_stress_soc_floor:
        soc_min_schedule[stress_active] = soc_min_target
        rising = stress_active & (~np.roll(stress_active, 1))
        rising[0] = False
        ramp_steps = 12
        for t0 in np.where(rising)[0].tolist():
            for j in range(1, ramp_steps + 1):
                t_prev = t0 - j
                if t_prev < 0:
                    break
                frac = (ramp_steps - j) / ramp_steps
                val = soc_min_phys + frac * (soc_min_target - soc_min_phys)
                soc_min_schedule[t_prev] = max(soc_min_schedule[t_prev], val)

    daily_ev_forecast_total = (
        pd.Series(ev_fc, index=plant_df["date"]).groupby(level=0).sum().to_dict()
    )
    daily_ev_actual_total = plant_df.groupby("date")["ev"].sum().astype(float).to_dict()
    charged_ev_by_date: Dict[object, float] = {d: 0.0 for d in plant_df["date"].unique()}

    monthly_peak_so_far: Dict[str, float] = {}
    finalized_exceedance_last12: deque = deque(maxlen=12)
    exceedance_month_so_far: Dict[str, float] = {}
    prev_month_key: Optional[str] = None
    prev_date = None

    ev_applied = np.zeros(n)
    hp_applied = np.zeros(n)
    hp_plc_extra = np.zeros(n)
    p_grid_actual = np.zeros(n)
    ev_plan_kwh_ts = np.full(n, np.nan)
    hp_plan_kwh_ts = np.full(n, np.nan)
    p_grid_plan_kw_ts = np.full(n, np.nan)
    p_limit_kw_ts = np.full(n, np.nan)
    access_kw_ts = np.full(n, np.nan)
    was_clipped_ts = np.zeros(n)
    ev_was_clipped_ts = np.zeros(n)
    ev_envelope_remaining_kwh_ts = np.full(n, np.nan)
    ev_envelope_headroom_after_kwh_ts = np.full(n, np.nan)
    ev_envelope_feasible_ts = np.full(n, np.nan)
    ev_enforce_extra_kwh_ts = np.zeros(n)
    ev_enforce_active_ts = np.zeros(n)
    ev_enforce_deferred_ts = np.zeros(n)
    hp_ev_enforce_shave_kwh_ts = np.zeros(n)
    ev_to_deliver_ts = np.zeros(n, dtype=float)
    ev_daily_demand_ts = np.zeros(n, dtype=float)
    soc_before_ts = np.full(n, np.nan)
    soc_after_ts = np.full(n, np.nan)
    billing_cfg = load_billing_config(str(billing_path))

    def _invoke_mpc_ev_hp_24h(
        df_window: pd.DataFrame,
        k: int,
        k_end: int,
        daily_ev_remaining: Dict[object, float],
        rolling12_map: Dict[str, float],
        t_k,
        month_key: str,
        peak_sofar_kw: float,
    ) -> Tuple[pd.DataFrame, Dict]:
        try:
            return mpc_ev_hp_24h(
                df_window=df_window,
                config_path=str(billing_path),
                hp_config_path=str(hp_config_path),
                monthly_peak_so_far=dict(monthly_peak_so_far),
                rolling12_max_exceedance_so_far_by_month=rolling12_map,
                soc_initial=float(soc),
                daily_ev_remaining=daily_ev_remaining,
                ev_deadline_slack_minutes=ev_deadline_slack_minutes,
                access_power_by_month=access_power_by_month,
                soc_slack_penalty_eur_per_soc=soc_slack_penalty_eur_per_soc,
                soc_min_slack_penalty_eur_per_soc=soc_min_slack_penalty_eur_per_soc,
                monthly_peak_price_multiplier=monthly_peak_price_multiplier,
                buffer_soc_min_profile=(
                    [float(x) for x in soc_min_schedule[k:k_end]]
                    if enable_forecast_stress_soc_floor
                    else None
                ),
            )
        except Exception:
            print(
                f"{log_prefix} MPC failed at k={k} / {n-1} (t={t_k}, month={month_key}).",
                flush=True,
            )
            print(
                f"{log_prefix} soc_initial={soc:.4f}, access_kw={access_kw:.1f}, "
                f"peak_sofar_kw={peak_sofar_kw:.1f}",
                flush=True,
            )
            print(
                f"{log_prefix} Forecast strategies: ev={forecast_strategy_ev!r}, "
                f"inflex={forecast_strategy_inflex!r}, inflex_stress={effective_inflex_stress!r}, "
                f"pv={forecast_strategy_pv!r}, thermal={forecast_strategy_thermal!r}, "
                f"temp={forecast_strategy_temperature!r}",
                flush=True,
            )
            try:
                print(f"{log_prefix} df_window head:\n{df_window.head(6)}", flush=True)
                print(f"{log_prefix} df_window tail:\n{df_window.tail(6)}", flush=True)
                print(
                    f"{log_prefix} Window stats: "
                    f"inflex_kWh={float(df_window['inflex_load'].sum()):.2f}, "
                    f"ev_kWh={float(df_window['ev'].sum()):.2f}, "
                    f"pv_kWh={float(df_window['pv_production'].sum()):.2f}, "
                    f"thermal_kWhth={float(df_window['thermal_load'].sum()):.2f}, "
                    f"T_out_min={float(pd.to_numeric(df_window['outdoor_temperature'], errors='coerce').min()):.2f}, "
                    f"T_out_max={float(pd.to_numeric(df_window['outdoor_temperature'], errors='coerce').max()):.2f}",
                    flush=True,
                )
            except Exception as inner:
                print(f"{log_prefix} (debug print failed: {inner})", flush=True)

            if enable_mpc_window_debug:
                try:
                    out_dir = Path("output/online_mpc")
                    out_dir.mkdir(parents=True, exist_ok=True)
                    debug_path = (
                        Path(mpc_window_debug_csv_path)
                        if mpc_window_debug_csv_path
                        else out_dir / f"debug_ev_hp_window_k{k}_{month_key}.csv"
                    )
                    df_debug = df_window.copy()
                    ts_dbg = pd.to_datetime(df_debug["timestamp"], errors="coerce")
                    df_debug["access_kw_window"] = [
                        float(access_power_by_month[_month_key(ts_i)])
                        if not pd.isna(ts_i)
                        else np.nan
                        for ts_i in ts_dbg
                    ]
                    df_debug["forecast_grid_kw_window"] = forecast_grid_kw_full[k:k_end]
                    df_debug["forecast_stress_active_window"] = stress_active[k:k_end].astype(float)
                    df_debug["soc_min_planner_floor_window"] = soc_min_schedule[k:k_end]
                    df_debug["dbg_k0"] = int(k)
                    df_debug["dbg_soc_initial"] = float(soc)
                    df_debug["dbg_access_kw_at_k0"] = float(access_kw)
                    df_debug["dbg_realized_monthly_peak_kw_before_step"] = float(peak_sofar_kw)
                    df_debug.to_csv(debug_path, index=False)
                    print(f"{log_prefix} Saved failing df_window to {debug_path}", flush=True)
                except Exception as inner2:
                    print(f"{log_prefix} (debug csv save failed: {inner2})", flush=True)
            raise

    if verbose:
        print(
            f"{log_prefix} Prepared timestep data (n={n}). Entering main simulation loop...",
            flush=True,
        )

    for k in range(n):
        t_k = plant_df["timestamp"].iloc[k]
        date_k = t_k.date()
        month_key = _month_key(t_k)

        if prev_month_key is None:
            prev_month_key = month_key
        elif month_key != prev_month_key:
            finalized_exceedance_last12.append(
                float(exceedance_month_so_far.get(prev_month_key, 0.0))
            )
            prev_month_key = month_key

        if verbose and (k == 0 or date_k != prev_date):
            print(
                f"{log_prefix} Simulating day {date_k} (step {k+1}/{n}, month {month_key})",
                flush=True,
            )
        prev_date = date_k

        tod = t_k.hour + t_k.minute / 60.0
        dow = t_k.dayofweek
        is_weekday = dow < 5
        in_ev_window = 7.0 <= tod < 17.0
        opt_active = is_weekday and in_ev_window
        slack_hours = ev_deadline_slack_minutes / 60.0
        mpc_ramp_end = 17.0 - slack_hours
        in_mpc_region = opt_active and (tod >= 7.0) and (tod < mpc_ramp_end)
        in_catchup_region = opt_active and (tod >= mpc_ramp_end) and (tod < 17.0)

        access_kw = float(access_power_by_month[month_key])
        access_kw_ts[k] = access_kw
        peak_sofar_kw = float(monthly_peak_so_far.get(month_key, 0.0))
        soc_before_ts[k] = soc

        inflex_act = float(plant_df["inflex_load"].iloc[k])
        pv_act = float(plant_df["pv_production"].iloc[k])
        thermal_act = float(plant_df["thermal_load"].iloc[k])
        temp_act = float(plant_df["outdoor_temperature"].iloc[k])
        cop_act = (
            float(interpolate_cop(temp_act, hp_cfg["COP_data"]))
            if not np.isnan(temp_act)
            else 2.5
        )
        buffer_energy_prev = soc * buffer_capacity_kwh_th
        losses_kwh = buffer_energy_prev * loss_rate_per_interval

        if not in_catchup_region:
            # Joint MPC: HP 24/7; EV planner active only in weekday 07:00–(17:00−slack)
            required_forecast = float(daily_ev_forecast_total.get(date_k, 0.0))
            required_actual = float(daily_ev_actual_total.get(date_k, 0.0))
            if tod <= 7.0:
                w_actual = 0.0
            elif tod >= 12.0:
                w_actual = 1.0
            else:
                w_actual = (tod - 7.0) / 5.0
            ev_daily_demand = (1.0 - w_actual) * required_forecast + w_actual * required_actual
            charged_so_far = charged_ev_by_date.get(date_k, 0.0)
            ev_to_deliver = max(ev_daily_demand - charged_so_far, 0.0) if opt_active else 0.0
            ev_daily_demand_ts[k] = ev_daily_demand if opt_active else 0.0
            ev_to_deliver_ts[k] = ev_to_deliver

            k_end = min(k + horizon_len, n)
            df_window = plant_df.loc[
                k : k_end - 1,
                [
                    "timestamp",
                    "pv_for_mpc",
                    "inflex_forecast",
                    "price",
                    "ev_forecast",
                    "thermal_forecast",
                    "outdoor_temperature_for_mpc",
                ],
            ].copy()
            wlen = k_end - k
            env_blend = np.zeros(wlen)
            env_forecast = ev_power_envelope_forecast_kw[k:k_end].copy()
            env_actual = ev_power_envelope_actual_kw[k:k_end].copy()
            w_unfold = w_actual if opt_active else 0.0
            env_blend = w_unfold * env_actual + (1.0 - w_unfold) * env_forecast
            if k > 0 and wlen > 0:
                env_blend[0] = (
                    ev_power_envelope_actual_kw[k - 1]
                    if tod < 12.0
                    else ev_power_envelope_actual_kw[k]
                )
            if not opt_active:
                env_blend[:] = 0.0

            df_window["ev_power_envelope_fixed_kw"] = env_blend
            df_window["ev_power_envelope_forecast_kw"] = env_forecast
            df_window.rename(
                columns={
                    "pv_for_mpc": "pv_production",
                    "inflex_forecast": "inflex_load",
                    "ev_forecast": "ev",
                    "thermal_forecast": "thermal_load",
                    "outdoor_temperature_for_mpc": "outdoor_temperature",
                },
                inplace=True,
            )

            dates_in_window = pd.to_datetime(df_window["timestamp"]).dt.date.unique().tolist()
            daily_ev_remaining: Dict[object, float] = {}
            for d in dates_in_window:
                if not opt_active or d != date_k:
                    daily_ev_remaining[d] = 0.0
                elif d == date_k:
                    daily_ev_remaining[d] = float(ev_to_deliver)
                else:
                    daily_ev_remaining[d] = float(daily_ev_forecast_total.get(d, 0.0))

            roll12_completed = (
                float(max(finalized_exceedance_last12)) if finalized_exceedance_last12 else 0.0
            )
            window_month_keys = sorted({_month_key(ts_i) for ts_i in df_window["timestamp"]})
            rolling12_map: Dict[str, float] = {}
            for mk in window_month_keys:
                if mk == month_key:
                    rolling12_map[mk] = max(
                        roll12_completed, float(exceedance_month_so_far.get(mk, 0.0))
                    )
                else:
                    rolling12_map[mk] = roll12_completed

            window_res, window_summary = _invoke_mpc_ev_hp_24h(
                df_window,
                k,
                k_end,
                daily_ev_remaining,
                rolling12_map,
                t_k,
                month_key,
                peak_sofar_kw,
            )

            ev_plan_kwh = float(window_res["ev_charge"].iloc[0])
            hp_plan_kwh = float(window_res["hp_electrical_input"].iloc[0])
            ev_plan_kwh_ts[k] = ev_plan_kwh
            hp_plan_kwh_ts[k] = hp_plan_kwh

            monthly_peak_plan_kw = float(
                window_summary.get("monthly_peak_plan", {}).get(month_key, peak_sofar_kw)
            )
            p_target_kw = monthly_peak_plan_kw
            inner = max(peak_sofar_kw, p_target_kw)
            p_limit = min(access_kw, inner) if access_kw > 0 else inner
            p_limit_kw_ts[k] = p_limit

            ev_app_kwh, hp_app_kwh, p_plan, _, clipped = _joint_clip_first_step(
                inflex_act_kwh=inflex_act,
                pv_act_kwh=pv_act,
                ev_plan_kwh=ev_plan_kwh,
                hp_plan_kwh=hp_plan_kwh,
                cop_act=cop_act,
                soc=soc,
                soc_min_phys=soc_min_phys,
                buffer_capacity_kwh_th=buffer_capacity_kwh_th,
                thermal_act_kwh_th=thermal_act,
                loss_kwh=losses_kwh,
                p_limit_kw=p_limit,
            )
            was_clipped_ts[k] = 1.0 if clipped else 0.0
            p_grid_plan_kw_ts[k] = p_plan

            ev_enforce_extra_kwh = 0.0
            ev_enforce_active = 0.0
            ev_enforce_deferred = 0.0
            if in_mpc_region and opt_active:
                ev_was_clipped = ev_app_kwh < ev_plan_kwh - 1e-6
                ev_was_clipped_ts[k] = 1.0 if ev_was_clipped else 0.0
                headroom_after_kwh = 0.0
                if tod >= 12.0:
                    headroom_kwh = _ev_envelope_energy_remaining_kwh(
                        k, n, plant_df["timestamp"], ev_power_envelope_actual_kw
                    )
                    headroom_after_kwh = _ev_envelope_headroom_after_step_kwh(
                        k, n, plant_df["timestamp"], ev_power_envelope_actual_kw
                    )
                    ev_envelope_remaining_kwh_ts[k] = headroom_kwh
                    ev_envelope_headroom_after_kwh_ts[k] = headroom_after_kwh
                    ev_envelope_feasible_ts[k] = (
                        1.0 if ev_to_deliver <= headroom_kwh + 1e-6 else 0.0
                    )
                if enforce_daily_ev_demand and tod >= 12.0 and ev_was_clipped:
                    ev_app_kwh, ev_enforce_extra_kwh, ev_enforce_active, ev_enforce_deferred = (
                        _apply_ev_enforce_minimal_after_clip(
                            ev_to_deliver,
                            ev_app_kwh,
                            float(ev_power_envelope_actual_kw[k]),
                            headroom_after_kwh,
                        )
                    )
                    if ev_enforce_active > 0.5:
                        hp_app_kwh, hp_ev_enforce_shave_kwh_ts[k] = _hp_shave_after_ev_enforce(
                            ev_app_kwh=ev_app_kwh,
                            hp_kwh=hp_app_kwh,
                            inflex_act_kwh=inflex_act,
                            pv_act_kwh=pv_act,
                            p_limit_kw=p_limit,
                            cop_act=cop_act,
                            soc=soc,
                            soc_min_phys=soc_min_phys,
                            buffer_capacity_kwh_th=buffer_capacity_kwh_th,
                            thermal_act_kwh_th=thermal_act,
                            loss_kwh=losses_kwh,
                        )

            ev_enforce_extra_kwh_ts[k] = ev_enforce_extra_kwh
            ev_enforce_active_ts[k] = ev_enforce_active
            ev_enforce_deferred_ts[k] = ev_enforce_deferred

            if opt_active:
                charged_ev_by_date[date_k] = charged_ev_by_date.get(date_k, 0.0) + ev_app_kwh

        else:
            # EV catch-up region: planner EV envelope forced to zero; HP still from joint MPC
            k_end = min(k + horizon_len, n)
            df_window = plant_df.loc[
                k : k_end - 1,
                [
                    "timestamp",
                    "pv_for_mpc",
                    "inflex_forecast",
                    "price",
                    "ev_forecast",
                    "thermal_forecast",
                    "outdoor_temperature_for_mpc",
                ],
            ].copy()
            wlen = k_end - k
            env_forecast = ev_power_envelope_forecast_kw[k:k_end].copy()
            df_window["ev_power_envelope_fixed_kw"] = np.zeros(wlen)
            df_window["ev_power_envelope_forecast_kw"] = env_forecast
            df_window.rename(
                columns={
                    "pv_for_mpc": "pv_production",
                    "inflex_forecast": "inflex_load",
                    "ev_forecast": "ev",
                    "thermal_forecast": "thermal_load",
                    "outdoor_temperature_for_mpc": "outdoor_temperature",
                },
                inplace=True,
            )
            required_forecast = float(daily_ev_forecast_total.get(date_k, 0.0))
            required_actual = float(daily_ev_actual_total.get(date_k, 0.0))
            w_actual = 1.0 if tod >= 12.0 else (tod - 7.0) / 5.0
            ev_daily_demand = (1.0 - w_actual) * required_forecast + w_actual * required_actual
            ev_to_deliver = max(ev_daily_demand - charged_ev_by_date.get(date_k, 0.0), 0.0)
            ev_daily_demand_ts[k] = ev_daily_demand
            ev_to_deliver_ts[k] = ev_to_deliver
            dates_in_window = pd.to_datetime(df_window["timestamp"]).dt.date.unique().tolist()
            daily_ev_remaining = {
                d: (float(ev_to_deliver) if d == date_k else 0.0) for d in dates_in_window
            }
            roll12_completed = (
                float(max(finalized_exceedance_last12)) if finalized_exceedance_last12 else 0.0
            )
            window_month_keys = sorted({_month_key(ts_i) for ts_i in df_window["timestamp"]})
            rolling12_map = {
                mk: (
                    max(roll12_completed, float(exceedance_month_so_far.get(mk, 0.0)))
                    if mk == month_key
                    else roll12_completed
                )
                for mk in window_month_keys
            }
            window_res, window_summary = _invoke_mpc_ev_hp_24h(
                df_window,
                k,
                k_end,
                daily_ev_remaining,
                rolling12_map,
                t_k,
                month_key,
                peak_sofar_kw,
            )
            hp_plan_kwh = float(window_res["hp_electrical_input"].iloc[0])
            hp_plan_kwh_ts[k] = hp_plan_kwh
            monthly_peak_plan_kw = float(
                window_summary.get("monthly_peak_plan", {}).get(month_key, peak_sofar_kw)
            )
            p_limit = (
                min(access_kw, max(peak_sofar_kw, monthly_peak_plan_kw))
                if access_kw > 0
                else max(peak_sofar_kw, monthly_peak_plan_kw)
            )
            p_limit_kw_ts[k] = p_limit

            ev_enforce_extra_kwh = 0.0
            ev_enforce_active = 0.0
            ev_enforce_deferred = 0.0
            p_ev_set_kw = 0.0

            if ev_to_deliver <= 0.0:
                ev_app_kwh = 0.0
            else:
                p_ev_ideal = ev_to_deliver / 0.25
                p_ev_env = float(ev_power_envelope_base_kw[k])
                p_ev_set_kw = min(p_ev_ideal, p_ev_env)
                p_grid_plan_ev = 4.0 * (inflex_act + p_ev_set_kw / 4.0 + hp_plan_kwh - pv_act)
                p_lim_cu = min(access_kw, peak_sofar_kw) if access_kw > 0 else peak_sofar_kw
                p_limit_kw_ts[k] = p_lim_cu
                ev_cu_clipped = p_grid_plan_ev > p_lim_cu
                if not ev_cu_clipped:
                    p_ev_new = p_ev_set_kw
                else:
                    p_ev_new = max(p_ev_set_kw - (p_grid_plan_ev - p_lim_cu), 0.0)
                ev_app_kwh = p_ev_new / 4.0
                if ev_cu_clipped:
                    ev_was_clipped_ts[k] = 1.0
                ev_clipped_kwh = p_ev_new / 4.0
                headroom_after_kwh = 0.0
                if tod >= 12.0:
                    headroom_kwh = _ev_envelope_energy_remaining_kwh(
                        k, n, plant_df["timestamp"], ev_power_envelope_actual_kw
                    )
                    headroom_after_kwh = _ev_envelope_headroom_after_step_kwh(
                        k, n, plant_df["timestamp"], ev_power_envelope_actual_kw
                    )
                    ev_envelope_remaining_kwh_ts[k] = headroom_kwh
                    ev_envelope_headroom_after_kwh_ts[k] = headroom_after_kwh
                    ev_envelope_feasible_ts[k] = (
                        1.0 if ev_to_deliver <= headroom_kwh + 1e-6 else 0.0
                    )
                if enforce_daily_ev_demand and tod >= 12.0 and ev_cu_clipped:
                    ev_clipped_kwh, ev_enforce_extra_kwh, ev_enforce_active, ev_enforce_deferred = (
                        _apply_ev_enforce_minimal_after_clip(
                            ev_to_deliver,
                            ev_clipped_kwh,
                            float(ev_power_envelope_base_kw[k]),
                            headroom_after_kwh,
                        )
                    )
                ev_app_kwh = ev_clipped_kwh
                if ev_enforce_active > 0.5:
                    hp_plan_kwh, hp_ev_enforce_shave_kwh_ts[k] = _hp_shave_after_ev_enforce(
                        ev_app_kwh=ev_app_kwh,
                        hp_kwh=hp_plan_kwh,
                        inflex_act_kwh=inflex_act,
                        pv_act_kwh=pv_act,
                        p_limit_kw=p_limit,
                        cop_act=cop_act,
                        soc=soc,
                        soc_min_phys=soc_min_phys,
                        buffer_capacity_kwh_th=buffer_capacity_kwh_th,
                        thermal_act_kwh_th=thermal_act,
                        loss_kwh=losses_kwh,
                    )

            ev_enforce_extra_kwh_ts[k] = ev_enforce_extra_kwh
            ev_enforce_active_ts[k] = ev_enforce_active
            ev_enforce_deferred_ts[k] = ev_enforce_deferred

            _, hp_app_kwh, p_plan, _, clipped = _joint_clip_first_step(
                inflex_act_kwh=inflex_act,
                pv_act_kwh=pv_act,
                ev_plan_kwh=ev_app_kwh,
                hp_plan_kwh=hp_plan_kwh,
                cop_act=cop_act,
                soc=soc,
                soc_min_phys=soc_min_phys,
                buffer_capacity_kwh_th=buffer_capacity_kwh_th,
                thermal_act_kwh_th=thermal_act,
                loss_kwh=losses_kwh,
                p_limit_kw=p_limit,
            )
            was_clipped_ts[k] = 1.0 if clipped else 0.0
            p_grid_plan_kw_ts[k] = p_plan
            ev_enforce_extra_kwh_ts[k] = ev_enforce_extra_kwh
            ev_enforce_active_ts[k] = ev_enforce_active
            charged_ev_by_date[date_k] = charged_ev_by_date.get(date_k, 0.0) + ev_app_kwh

        # HP PLC safeguard (thesis Eq. 3.98–3.99)
        plc_extra = 0.0
        hp_thermal_out = hp_app_kwh * cop_act
        soc_next_raw = soc + (hp_thermal_out - thermal_act - losses_kwh) / buffer_capacity_kwh_th

        if enforce_soc_min and soc_next_raw < soc_min_phys:
            deficit_th = (soc_min_phys - soc_next_raw) * buffer_capacity_kwh_th
            plc_extra = deficit_th / cop_act if cop_act > 1e-9 else 0.0
            hp_app_kwh += plc_extra

        hp_plc_extra[k] = plc_extra
        ev_applied[k] = ev_app_kwh
        hp_applied[k] = hp_app_kwh

        p_grid_act = 4.0 * (inflex_act + ev_app_kwh + hp_app_kwh - pv_act)
        p_grid_actual[k] = p_grid_act
        monthly_peak_so_far[month_key] = max(peak_sofar_kw, p_grid_act)
        exceedance_month_so_far[month_key] = max(
            0.0, float(monthly_peak_so_far[month_key] - access_kw)
        )

        hp_thermal_out = hp_app_kwh * cop_act
        soc = soc + (hp_thermal_out - thermal_act - losses_kwh) / buffer_capacity_kwh_th
        if enforce_soc_max and soc > soc_max_phys and cop_act > 1e-9:
            hp_cap = (
                (soc_max_phys - soc_before_ts[k]) * buffer_capacity_kwh_th
                + thermal_act
                + losses_kwh
            ) / cop_act
            hp_app_kwh = float(max(min(hp_app_kwh, hp_cap), 0.0))
            hp_applied[k] = hp_app_kwh
            hp_thermal_out = hp_app_kwh * cop_act
            soc = soc_before_ts[k] + (hp_thermal_out - thermal_act - losses_kwh) / buffer_capacity_kwh_th

        soc_after_ts[k] = soc

    res = plant_df.copy()
    res["ev_applied"] = ev_applied
    res["ev_online_mpc"] = ev_applied
    res["hp_applied"] = hp_applied
    res["hp_applied_kwh"] = hp_applied
    res["hp_plc_extra_kwh"] = hp_plc_extra
    res["grid_power_online"] = p_grid_actual
    res["p_grid_actual_kw"] = p_grid_actual
    res["ev_plan_kwh"] = ev_plan_kwh_ts
    res["hp_plan_kwh"] = hp_plan_kwh_ts
    res["p_grid_plan_kw"] = p_grid_plan_kw_ts
    res["p_limit_kw"] = p_limit_kw_ts
    res["access_kw"] = access_kw_ts
    res["was_clipped"] = was_clipped_ts
    res["ev_was_clipped"] = ev_was_clipped_ts
    res["ev_envelope_remaining_kwh"] = ev_envelope_remaining_kwh_ts
    res["ev_envelope_headroom_after_kwh"] = ev_envelope_headroom_after_kwh_ts
    res["ev_envelope_feasible"] = ev_envelope_feasible_ts
    res["ev_enforce_extra_kwh"] = ev_enforce_extra_kwh_ts
    res["ev_enforce_active"] = ev_enforce_active_ts
    res["ev_enforce_deferred"] = ev_enforce_deferred_ts
    res["hp_ev_enforce_shave_kwh"] = hp_ev_enforce_shave_kwh_ts
    res["ev_to_deliver_kwh"] = ev_to_deliver_ts
    res["ev_daily_demand_kwh"] = ev_daily_demand_ts
    res["soc_before"] = soc_before_ts
    res["soc_after"] = soc_after_ts
    res["forecast_grid_kw"] = forecast_grid_kw_full.astype(float)
    res["forecast_access_exceedance_active"] = stress_active.astype(float)
    res["soc_min_planner_floor"] = soc_min_schedule.astype(float)
    res["month_key"] = res["timestamp"].apply(_month_key)

    ev_actual_by_day = plant_df.groupby("date")["ev"].sum().to_dict()
    delivered_by_day = res.groupby("date")["ev_applied"].sum().to_dict()
    uncharged_kwh = {
        d: max(ev_actual_by_day.get(d, 0.0) - delivered_by_day.get(d, 0.0), 0.0)
        for d in set(ev_actual_by_day) | set(delivered_by_day)
    }
    res["uncharged_kwh"] = res["date"].map(uncharged_kwh)

    res_bill = res.copy()
    res_bill["grid_consumption"] = np.maximum(res["p_grid_actual_kw"], 0.0) / 4.0
    res_bill["grid_injection"] = np.maximum(-res["p_grid_actual_kw"], 0.0) / 4.0
    res_bill["access_power_online_kw"] = res_bill["month_key"].map(access_power_by_month)

    bills = calculate_monthly_bills(
        res_bill, billing_cfg, access_power_col="access_power_online_kw"
    )
    inj_bills = calculate_monthly_injection_bills(res_bill, billing_cfg)

    summary = {
        "n_steps": n,
        "monthly_peak_so_far": dict(monthly_peak_so_far),
        "uncharged_kwh_by_day": {str(d): float(v) for d, v in uncharged_kwh.items()},
        "bills": bills,
        "injection_bills": inj_bills,
        "forecast_strategy_ev": forecast_strategy_ev,
        "forecast_strategy_inflex": forecast_strategy_inflex,
        "forecast_strategy_pv": forecast_strategy_pv,
        "forecast_strategy_thermal": forecast_strategy_thermal,
        "ev_deadline_slack_minutes": ev_deadline_slack_minutes,
        "enforce_daily_ev_demand": enforce_daily_ev_demand,
        "ev_enforce_steps": int(np.sum(ev_enforce_active_ts > 0)),
        "ev_enforce_extra_kwh_total": float(np.sum(ev_enforce_extra_kwh_ts)),
        "enable_forecast_stress_soc_floor": enable_forecast_stress_soc_floor,
        "forecast_stress_soc_floor_strength": forecast_stress_soc_floor_strength,
    }

    if verbose:
        total_uncharged = sum(max(v, 0.0) for v in uncharged_kwh.values())
        print(f"{log_prefix} Done. Total uncharged EV: {total_uncharged:.2f} kWh", flush=True)
        if enforce_daily_ev_demand:
            n_enforce = int(np.sum(ev_enforce_active_ts > 0))
            extra_kwh = float(np.sum(ev_enforce_extra_kwh_ts))
            print(
                f"{log_prefix} EV enforce: {n_enforce} steps, "
                f"+{extra_kwh:.2f} kWh restored after clip",
                flush=True,
            )

    return res, summary


if __name__ == "__main__":
    raise RuntimeError("Call run_ev_hp_online_mpc_1 from notebook 11 or a script.")
