import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from collections import deque

# Ensure src is on path when run as a script
THIS_DIR = Path(__file__).parent
PROJECT_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from optimization import mpc_hp_24h  # type: ignore
from heat_pump_load import load_hp_config, interpolate_cop  # type: ignore


def _parse_plant_data(plant_path: Path) -> pd.DataFrame:
    df = pd.read_csv(plant_path)
    ts_utc = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["timestamp"] = ts_utc.dt.tz_convert("Europe/Brussels").dt.tz_localize(None)
    df = df.sort_values("timestamp").reset_index(drop=True)

    start_2025 = pd.Timestamp("2025-01-01 00:00:00")
    end_2025 = pd.Timestamp("2026-01-01 00:00:00")
    df = df[(df["timestamp"] >= start_2025) & (df["timestamp"] < end_2025)].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _load_forecast_column(
    path: Path,
    strategy: str,
    prefix: str,
    hint_prefix: str,
) -> np.ndarray:
    df = pd.read_csv(path)
    col = strategy if strategy.startswith(prefix) else f"{prefix}{strategy}"
    if col not in df.columns:
        # Common alias: some exports store p50 as the base series without "_p50"
        # e.g. EV: "forecast_ev_c" is p50, while users may request "c_p50".
        if strategy.endswith("_p50"):
            alt_strategy = strategy[: -len("_p50")]
            alt_col = (
                alt_strategy if alt_strategy.startswith(prefix) else f"{prefix}{alt_strategy}"
            )
            if alt_col in df.columns:
                col = alt_col
            else:
                hint_cols = [c for c in df.columns if c.startswith(hint_prefix)]
                raise KeyError(
                    f"No forecast column found for strategy {strategy!r} in {path.name}. "
                    f"Expected {col!r} (or alias {alt_col!r}). Available include: {hint_cols[:14]}"
                )
        else:
            hint_cols = [c for c in df.columns if c.startswith(hint_prefix)]
            raise KeyError(
                f"No forecast column found for strategy {strategy!r} in {path.name}. "
                f"Expected {col!r}. Available include: {hint_cols[:14]}"
            )
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)


def _month_key(ts: pd.Timestamp) -> str:
    return ts.to_period("M").strftime("%Y-%m")


def run_hp_online_mpc_1(
    forecast_strategy_inflex: str = "c",
    forecast_strategy_inflex_stress: Optional[str] = None,
    forecast_strategy_pv: str = "actual",
    forecast_strategy_thermal: str = "c",
    forecast_strategy_ev: str = "actual",
    forecast_strategy_temperature: str = "actual",
    access_power_by_month: Dict[str, float] = None,
    hp_config_path: Optional[str] = None,
    enforce_soc_min: bool = True,
    enforce_soc_max: bool = False,
    soc_slack_penalty_eur_per_soc: Optional[float] = None,
    soc_min_slack_penalty_eur_per_soc: float = 1.0e6,
    monthly_peak_price_multiplier: float = 1.0,
    verbose: bool = True,
    log_prefix: str = "[Online MPC]",
    horizon_len: int = 96,
    enable_mpc_window_debug: bool = False,
    enable_forecast_stress_soc_floor: bool = False,
    forecast_stress_soc_floor_strength: float = 1.0,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run a full-year HP-only online myopic MPC simulation (24h rolling horizon).

    Core mechanics (mirrors EV online MPC 1):
    - At each 15-min step k, solve `mpc_hp_24h` on a 24h window using *forecast inputs*.
    - Apply real-time clipping with thesis Eq. (3.50): P_lim = min(P_access, max(P_peak,sofar, P_target)),
      P_target = planner monthly_peak_plan for the current month (same as `online_MPC_1_EV`).
    - Update buffer SOC using *actual* thermal load and actual COP, with feedback to the next MPC call.

    Forecast-access stress (optional): before each mpc_hp_24h solve, expected grid kW is formed from
    forecast inflex/EV/PV and an HP electrical estimate (thermal_forecast / COP from forecast temperature).
    If that exceeds access power, a time-varying SOC-min schedule is precomputed for the full year and
    sliced into each 24h MPC window; the optimizer sees upcoming stress periods in advance.
    PLC and unmet-thermal paths always use soc_min_phys so the buffer can discharge to the physical
    minimum after clipping.
    """
    plant_path = PROJECT_ROOT / "data" / "plant1.csv"
    forecast_inflex_path = PROJECT_ROOT / "output" / "forecast" / "forecast_inflex_load_rolling_horizon.csv"
    forecast_ev_path = PROJECT_ROOT / "output" / "forecast" / "forecast_ev_rolling_horizon.csv"
    forecast_pv_path = PROJECT_ROOT / "output" / "forecast" / "forecast_pv_rolling_horizon.csv"
    forecast_thermal_path = PROJECT_ROOT / "output" / "forecast" / "forecast_thermal_load_rolling_horizon.csv"
    temperature_forecast_path = PROJECT_ROOT / "data" / "temperature_forecast_day_ahead_open_meteo_Turnhout_15min.csv"
    billing_path = PROJECT_ROOT / "config" / "billing.yaml"
    if hp_config_path is None:
        hp_config_path = str(PROJECT_ROOT / "config" / "hp.yaml")

    if verbose:
        effective_inflex_stress = (
            forecast_strategy_inflex
            if forecast_strategy_inflex_stress is None
            else forecast_strategy_inflex_stress
        )
        print("=" * 80, flush=True)
        print("HP-only Online MPC 1 – Full-year simulation", flush=True)
        print("=" * 80, flush=True)
        print(f"  Inflex forecast strategy:   {forecast_strategy_inflex}", flush=True)
        print(f"  Inflex strategy (stress):   {effective_inflex_stress}", flush=True)
        print(f"  PV forecast strategy:       {forecast_strategy_pv}", flush=True)
        print(f"  Thermal forecast strategy:  {forecast_strategy_thermal}", flush=True)
        print(f"  EV strategy (uncontrollable): {forecast_strategy_ev}", flush=True)
        print(f"  Temperature strategy:       {forecast_strategy_temperature}", flush=True)
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

    plant_df = _parse_plant_data(plant_path)
    n = len(plant_df)

    if access_power_by_month is None:
        raise ValueError(
            "run_hp_online_mpc_1 requires 'access_power_by_month' mapping month key 'YYYY-MM' -> access power in kW."
        )

    month_keys_in_data = sorted({_month_key(ts_i) for ts_i in plant_df["timestamp"]})
    missing_months = [m for m in month_keys_in_data if m not in access_power_by_month]
    if missing_months:
        raise KeyError(
            "access_power_by_month is missing entries for months: " + ", ".join(missing_months)
        )

    # Forecast arrays (same length as plant assumed)
    inflex_fc = _load_forecast_column(
        forecast_inflex_path,
        strategy=forecast_strategy_inflex,
        prefix="forecast_inflex_",
        hint_prefix="forecast_inflex_",
    )
    effective_inflex_stress = (
        forecast_strategy_inflex
        if forecast_strategy_inflex_stress is None
        else forecast_strategy_inflex_stress
    )
    inflex_fc_stress = (
        _load_forecast_column(
            forecast_inflex_path,
            strategy=effective_inflex_stress,
            prefix="forecast_inflex_",
            hint_prefix="forecast_inflex_",
        )
        if enable_forecast_stress_soc_floor
        else inflex_fc
    )

    if forecast_strategy_pv == "actual":
        raise ValueError(
            "PV strategy 'actual' is disabled for online MPC comparability. "
            "Provide a forecast strategy (e.g. 'chronos2_elia_p50', 'k1'..'k10')."
        )
    pv_df = pd.read_csv(forecast_pv_path)
    if len(pv_df) != len(plant_df):
        raise ValueError("PV forecast and plant data must have same length.")
    pv_col = (
        forecast_strategy_pv
        if forecast_strategy_pv.startswith("pv_forecast_kWh_15min_")
        else f"pv_forecast_kWh_15min_{forecast_strategy_pv}"
    )
    if pv_col not in pv_df.columns:
        hint = [c for c in pv_df.columns if c.startswith("pv_forecast_kWh_15min_")]
        raise KeyError(f"Missing PV forecast col {pv_col!r}. Available: {hint[:14]}")
    pv_fc = pd.to_numeric(pv_df[pv_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    thermal_fc = _load_forecast_column(
        forecast_thermal_path,
        strategy=forecast_strategy_thermal,
        prefix="forecast_thermal_",
        hint_prefix="forecast_thermal_",
    )

    # Temperature series used by the planner (for COP computation).
    # Align by sequence index (ignore tz/DST) to match the "works before" approach.
    if forecast_strategy_temperature == "actual":
        raise ValueError(
            "Temperature strategy 'actual' is disabled for online MPC comparability. "
            "Use 'open_meteo_day_ahead'."
        )
    if forecast_strategy_temperature in {"open_meteo_day_ahead", "day_ahead_open_meteo"}:
        temp_df = pd.read_csv(temperature_forecast_path)
        if len(temp_df) != n:
            raise ValueError("Temperature forecast and plant data must have same length.")
        # Pick the first non-timestamp column as the temperature column.
        temp_cols = [c for c in temp_df.columns if c != "timestamp"]
        if not temp_cols:
            raise KeyError(
                f"No temperature column found in {temperature_forecast_path.name!r}."
            )
        temp_for_mpc = pd.to_numeric(temp_df[temp_cols[0]], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        raise ValueError(
            "Unknown forecast_strategy_temperature. Use 'open_meteo_day_ahead'. "
            f"Got {forecast_strategy_temperature!r}."
        )

    # EV is physically uncontrollable in this HP-only setting; require a non-actual strategy
    # for comparability (planner should not use measured EV directly).
    if forecast_strategy_ev == "actual":
        raise ValueError(
            "EV strategy 'actual' is disabled for online MPC comparability. "
            "Provide a forecast strategy (e.g. 'c_p50')."
        )
    ev_fc = _load_forecast_column(
        forecast_ev_path,
        strategy=forecast_strategy_ev,
        prefix="forecast_ev_",
        hint_prefix="forecast_ev_",
    )
    if len(ev_fc) != n:
        raise ValueError("EV forecast array must match plant length (35040).")
    ev_seen = ev_fc

    if len(inflex_fc) != n or len(pv_fc) != n or len(thermal_fc) != n or len(ev_seen) != n:
        raise ValueError("Forecast arrays must match plant length (35040).")

    plant_df = plant_df.copy()
    plant_df["inflex_forecast"] = inflex_fc
    plant_df["pv_for_mpc"] = pv_fc
    plant_df["thermal_forecast"] = thermal_fc
    plant_df["ev_for_mpc"] = ev_seen
    plant_df["outdoor_temperature_for_mpc"] = temp_for_mpc

    # HP config for SOC params and COP
    hp_cfg = load_hp_config(hp_config_path)
    soc_min_phys = float(hp_cfg["buffer"]["soc_min"])
    soc_max_phys = float(hp_cfg["buffer"]["soc_max"])
    soc = float(hp_cfg["buffer"]["soc_initial"])
    loss_coeff_per_hour = float(hp_cfg["buffer"]["loss_coefficient_per_hour"])
    buffer_size_m3 = float(hp_cfg["buffer"]["size_m3"])
    rho = float(hp_cfg["buffer"]["water_density_kg_per_m3"])
    cp = float(hp_cfg["buffer"]["cp_kj_per_kg_k"])
    dT = float(hp_cfg["buffer"]["usable_delta_t_k"])
    buffer_capacity_kwh_th = (buffer_size_m3 * rho * cp * dT) / 3600.0
    loss_rate_per_interval = loss_coeff_per_hour / 4.0

    # ------------------------------------------------------------------
    # Forecast-stress SOC-min schedule (precompute full year)
    # ------------------------------------------------------------------
    # stress_active[t] := (forecast_grid_kw[t] > access_kw(month(t))) where forecast_grid_kw uses
    # forecast inflex/EV/PV and HP estimate (thermal_forecast/COP(forecast_temp)).
    access_kw_full = np.array(
        [float(access_power_by_month[_month_key(ts)]) for ts in plant_df["timestamp"]],
        dtype=float,
    )
    temp_fc = pd.to_numeric(plant_df["outdoor_temperature_for_mpc"], errors="coerce").to_numpy(dtype=float)
    cop_fc = np.array(
        [
            float(interpolate_cop(float(t), hp_cfg["COP_data"])) if not np.isnan(t) else 2.5
            for t in temp_fc
        ],
        dtype=float,
    )
    thermal_fc_kwh_th = pd.to_numeric(plant_df["thermal_forecast"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    hp_est_kwh = np.where(cop_fc > 1e-9, thermal_fc_kwh_th / cop_fc, 0.0)
    # Use potentially different inflex forecast for stress detection vs optimization.
    inflex_fc_kwh = np.array(inflex_fc_stress, dtype=float)
    ev_fc_kwh = pd.to_numeric(plant_df["ev_for_mpc"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    pv_fc_kwh = pd.to_numeric(plant_df["pv_for_mpc"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    forecast_grid_kw_full = 4.0 * (inflex_fc_kwh + ev_fc_kwh + hp_est_kwh - pv_fc_kwh)
    stress_active_full = forecast_grid_kw_full > access_kw_full

    # Map stress to SOC-min target.
    _strength = float(np.clip(forecast_stress_soc_floor_strength, 0.0, 1.0))
    soc_min_target = float(soc_min_phys + _strength * (soc_max_phys - soc_min_phys))
    soc_min_target = float(min(max(soc_min_target, soc_min_phys), soc_max_phys))
    soc_min_schedule_full = np.full(n, soc_min_phys, dtype=float)
    soc_min_schedule_full[stress_active_full] = soc_min_target

    # Optional ramp-in at rising edges to reduce discontinuity:
    # if stress becomes active at t0, ramp up the floor over the preceding steps so the jump is anticipatable.
    rising = stress_active_full & (~np.roll(stress_active_full, 1))
    rising[0] = False
    if rising.any():
        ramp_steps = 12  # 3 hours (12 × 15-min) before t0 used for linear ramp
        idx = np.where(rising)[0]
        for t0 in idx.tolist():
            for j in range(1, ramp_steps + 1):
                t_prev = int(t0 - j)
                if t_prev < 0:
                    break
                frac = float((ramp_steps - j) / ramp_steps)  # t0-ramp_steps -> 0, t0-1 -> (ramp_steps-1)/ramp_steps
                val = float(soc_min_phys + frac * (soc_min_target - soc_min_phys))
                soc_min_schedule_full[t_prev] = max(float(soc_min_schedule_full[t_prev]), val)

    # Peak states (realized grid max per month, after clipping + PLC)
    monthly_peak_so_far: Dict[str, float] = {}

    # Rolling 12-month exceedance state (tariff logic)
    # Store finalized monthly exceedances (kW) for the last completed months.
    finalized_exceedance_last12 = deque(maxlen=12)
    exceedance_month_so_far: Dict[str, float] = {}
    prev_month_key: Optional[str] = None

    # Outputs
    hp_plan_kwh_ts = np.full(n, np.nan, dtype=float)
    hp_applied_kwh_ts = np.full(n, np.nan, dtype=float)
    hp_applied_kwh_nominal_ts = np.full(n, np.nan, dtype=float)
    hp_plc_extra_kwh_ts = np.full(n, np.nan, dtype=float)
    plc_active_ts = np.full(n, np.nan, dtype=float)
    thermal_served_kwh_th_ts = np.full(n, np.nan, dtype=float)
    unmet_thermal_kwh_th_ts = np.full(n, np.nan, dtype=float)
    soc_after_raw_ts = np.full(n, np.nan, dtype=float)
    soc_before_ts = np.full(n, np.nan, dtype=float)
    soc_after_ts = np.full(n, np.nan, dtype=float)
    p_grid_plan_kw_ts = np.full(n, np.nan, dtype=float)
    p_limit_kw_ts = np.full(n, np.nan, dtype=float)
    p_grid_actual_kw_ts = np.full(n, np.nan, dtype=float)
    was_clipped_ts = np.full(n, np.nan, dtype=float)
    access_kw_ts = np.full(n, np.nan, dtype=float)
    monthly_peak_so_far_ts = np.full(n, np.nan, dtype=float)
    mpc_first_interval_grid_kw_ts = np.full(n, np.nan, dtype=float)
    monthly_peak_plan_kw_ts = np.full(n, np.nan, dtype=float)
    grid_clip_limit_kw_ts = np.full(n, np.nan, dtype=float)
    realized_monthly_peak_kw_before_step_ts = np.full(n, np.nan, dtype=float)
    forecast_grid_kw_ts = forecast_grid_kw_full.astype(float).copy()
    forecast_access_exceedance_active_ts = stress_active_full.astype(float).copy()
    soc_min_planner_floor_ts = soc_min_schedule_full.astype(float).copy()

    prev_date = None

    if verbose:
        print(
            f"{log_prefix} Prepared timestep data (n={n}). Entering main simulation loop...",
            flush=True,
        )

    for k in range(n):
        t_k = plant_df["timestamp"].iloc[k]
        date_k = t_k.date()
        month_key = _month_key(t_k)

        # Month rollover: finalize previous month's exceedance into the rolling window.
        if prev_month_key is None:
            prev_month_key = month_key
        elif month_key != prev_month_key:
            prev_ex = float(exceedance_month_so_far.get(prev_month_key, 0.0))
            finalized_exceedance_last12.append(prev_ex)
            prev_month_key = month_key

        if verbose and (k == 0 or date_k != prev_date):
            print(
                f"{log_prefix} Simulating day {date_k} (step {k+1}/{n}, month {month_key})",
                flush=True,
            )
        prev_date = date_k

        # SOC feedback at k (measurement)
        soc_before_ts[k] = soc

        access_kw = float(access_power_by_month[month_key])
        access_kw_ts[k] = access_kw
        peak_sofar_kw = float(monthly_peak_so_far.get(month_key, 0.0))
        realized_monthly_peak_kw_before_step_ts[k] = peak_sofar_kw

        # MPC window [k, k+96)
        k_end = min(k + horizon_len, n)
        df_window = plant_df.loc[
            k : k_end - 1,
            [
                "timestamp",
                "pv_for_mpc",
                "inflex_forecast",
                "price",
                "ev_for_mpc",
                "thermal_forecast",
                "outdoor_temperature_for_mpc",
            ],
        ].copy()
        df_window.rename(
            columns={
                "pv_for_mpc": "pv_production",
                "inflex_forecast": "inflex_load",
                "ev_for_mpc": "ev",
                "thermal_forecast": "thermal_load",
                "outdoor_temperature_for_mpc": "outdoor_temperature",
            },
            inplace=True,
        )

        # Rolling 12-month max exceedance "locked in" so far for months in this window.
        # Typically 1 month; can be 2 at month boundary. Align by month key (not timestamps).
        roll12_completed = float(
            max(finalized_exceedance_last12) if len(finalized_exceedance_last12) else 0.0
        )
        window_month_keys = sorted({_month_key(ts_i) for ts_i in df_window["timestamp"]})
        rolling12_max_exceedance_so_far_by_month: Dict[str, float] = {}
        for mk in window_month_keys:
            if mk == month_key:
                ex_so_far = float(exceedance_month_so_far.get(mk, 0.0))
                rolling12_max_exceedance_so_far_by_month[mk] = max(roll12_completed, ex_so_far)
            else:
                rolling12_max_exceedance_so_far_by_month[mk] = roll12_completed

        # Planner call (debug on first failure)
        try:
            window_res, window_summary = mpc_hp_24h(
                df_window=df_window,
                config_path=str(billing_path),
                hp_config_path=str(hp_config_path),
                monthly_peak_so_far=dict(monthly_peak_so_far),
                rolling12_max_exceedance_so_far_by_month=dict(
                    rolling12_max_exceedance_so_far_by_month
                ),
                soc_initial=float(soc),
                soc_slack_penalty_eur_per_soc=(
                    float(soc_min_slack_penalty_eur_per_soc)
                    if soc_slack_penalty_eur_per_soc is None
                    else float(soc_slack_penalty_eur_per_soc)
                ),
                monthly_peak_price_multiplier=float(monthly_peak_price_multiplier),
                timestamp_col="timestamp",
                pv_col="pv_production",
                inflex_load_col="inflex_load",
                price_col="price",
                ev_col="ev",
                thermal_load_col="thermal_load",
                outdoor_temp_col="outdoor_temperature",
                access_power_by_month=access_power_by_month,
                buffer_soc_min_profile=(
                    [float(x) for x in soc_min_schedule_full[k:k_end]]
                    if enable_forecast_stress_soc_floor
                    else None
                ),
            )
        except Exception as e:
            print(
                f"{log_prefix} MPC failed at k={k} / {n-1} (t={t_k}, month={month_key})."
            )
            print(
                f"{log_prefix} soc_initial={soc:.4f}, access_kw={access_kw:.1f}, peak_sofar_kw={peak_sofar_kw:.1f}"
            )
            print(
                f"{log_prefix} Forecast strategies: inflex={forecast_strategy_inflex!r}, pv={forecast_strategy_pv!r}, "
                f"thermal={forecast_strategy_thermal!r}, ev={forecast_strategy_ev!r}, "
                f"temp={forecast_strategy_temperature!r}"
            )
            try:
                print(f"{log_prefix} df_window head:\n{df_window.head(6)}")
                print(f"{log_prefix} df_window tail:\n{df_window.tail(6)}")
                print(
                    f"{log_prefix} Window stats: "
                    f"inflex_kWh={float(df_window['inflex_load'].sum()):.2f}, "
                    f"ev_kWh={float(df_window['ev'].sum()):.2f}, "
                    f"pv_kWh={float(df_window['pv_production'].sum()):.2f}, "
                    f"thermal_kWhth={float(df_window['thermal_load'].sum()):.2f}, "
                    f"T_out_min={float(pd.to_numeric(df_window['outdoor_temperature'], errors='coerce').min()):.2f}, "
                    f"T_out_max={float(pd.to_numeric(df_window['outdoor_temperature'], errors='coerce').max()):.2f}"
                )
            except Exception as inner:
                print(f"{log_prefix} (debug print failed: {inner})")

            if enable_mpc_window_debug:
                try:
                    out_dir = Path("output/online_mpc")
                    out_dir.mkdir(parents=True, exist_ok=True)
                    debug_path = out_dir / f"debug_hp_window_k{k}_{month_key}.csv"
                    # Enrich the debug window with the additional context needed to diagnose infeasibility
                    # without re-running the full simulation.
                    df_debug = df_window.copy()
                    ts_dbg = pd.to_datetime(df_debug["timestamp"], errors="coerce")
                    df_debug["access_kw_window"] = [
                        float(access_power_by_month[_month_key(ts_i)]) if not pd.isna(ts_i) else np.nan
                        for ts_i in ts_dbg
                    ]
                    df_debug["forecast_grid_kw_window"] = forecast_grid_kw_ts[k:k_end]
                    df_debug["forecast_stress_active_window"] = forecast_access_exceedance_active_ts[
                        k:k_end
                    ]
                    df_debug["soc_min_planner_floor_window"] = (
                        soc_min_planner_floor_ts[k:k_end]
                        if enable_forecast_stress_soc_floor
                        else np.full(k_end - k, soc_min_phys, dtype=float)
                    )

                    # Scalar context (repeated for every row)
                    df_debug["dbg_k0"] = int(k)
                    df_debug["dbg_soc_initial"] = float(soc)
                    df_debug["dbg_access_kw_at_k0"] = float(access_kw)
                    df_debug["dbg_realized_monthly_peak_kw_before_step"] = float(peak_sofar_kw)
                    df_debug["dbg_forecast_stress_enabled"] = (
                        1.0 if enable_forecast_stress_soc_floor else 0.0
                    )
                    df_debug["dbg_forecast_stress_strength"] = float(
                        forecast_stress_soc_floor_strength
                    )
                    df_debug["dbg_soc_min_phys"] = float(soc_min_phys)
                    df_debug["dbg_soc_max_phys"] = float(soc_max_phys)

                    df_debug.to_csv(debug_path, index=False)
                    print(f"{log_prefix} Saved failing df_window to {debug_path}")
                except Exception as inner2:
                    print(f"{log_prefix} (debug csv save failed: {inner2})")

            raise

        hp_plan_kwh = float(window_res["hp_electrical_input"].iloc[0])
        hp_plan_kw = 4.0 * hp_plan_kwh
        hp_plan_kwh_ts[k] = hp_plan_kwh

        mpc_first_interval_grid_kw = float(window_res["grid_power"].iloc[0])
        mpc_first_interval_grid_kw_ts[k] = mpc_first_interval_grid_kw

        monthly_peak_plan_dict = window_summary.get("monthly_peak_plan", {})
        monthly_peak_plan_kw = float(monthly_peak_plan_dict.get(month_key, peak_sofar_kw))
        monthly_peak_plan_kw_ts[k] = monthly_peak_plan_kw

        # Planned grid power with actual inflex/ev/pv at k, planned HP
        inflex_act_kwh = float(plant_df["inflex_load"].iloc[k])
        ev_act_kwh = float(plant_df["ev"].iloc[k])
        pv_act_kwh = float(plant_df["pv_production"].iloc[k])
        p_grid_plan_kw = 4.0 * (inflex_act_kwh + ev_act_kwh + hp_plan_kwh - pv_act_kwh)
        p_grid_plan_kw_ts[k] = p_grid_plan_kw

        # Real-time peak limit (thesis Eq. 3.50; aligned with online_MPC_1_EV)
        p_target_kw = monthly_peak_plan_kw
        inner = max(peak_sofar_kw, p_target_kw)
        p_limit = min(access_kw, inner) if access_kw > 0 else inner
        p_limit_kw_ts[k] = p_limit
        grid_clip_limit_kw_ts[k] = p_limit

        was_clipped = p_grid_plan_kw > p_limit
        was_clipped_ts[k] = 1.0 if was_clipped else 0.0

        if not was_clipped:
            hp_new_kw = hp_plan_kw
        else:
            delta_p = p_grid_plan_kw - p_limit
            hp_new_kw = max(hp_plan_kw - delta_p, 0.0)

        hp_applied_kwh = hp_new_kw / 4.0
        hp_applied_kwh_nominal_ts[k] = hp_applied_kwh
        plc_extra_kwh = 0.0
        plc_active = 0.0
        hp_applied_kwh_ts[k] = hp_applied_kwh

        # Buffer SOC update using actual thermal load and actual COP at k
        thermal_demand_kwh_th = float(plant_df["thermal_load"].iloc[k])
        temp_act = float(plant_df["outdoor_temperature"].iloc[k])
        cop_k = float(interpolate_cop(temp_act, hp_cfg["COP_data"])) if not np.isnan(temp_act) else 2.5

        buffer_energy_prev = soc * buffer_capacity_kwh_th
        losses_kwh = buffer_energy_prev * loss_rate_per_interval
        thermal_served_kwh_th = thermal_demand_kwh_th
        unmet_thermal_kwh_th = 0.0

        # ------------------------------------------------------------------
        # Access-aware actuator (always on):
        # Reduce HP to avoid p_grid_actual_kw > access_kw when possible,
        # but prioritize physical SOC-min (soc_min_phys). If meeting SOC-min
        # requires exceeding access, we allow the exceedance.
        # ------------------------------------------------------------------
        hp_access_cap_kwh = (float(access_kw) / 4.0) - (inflex_act_kwh + ev_act_kwh - pv_act_kwh)
        hp_access_cap_kwh = float(max(hp_access_cap_kwh, 0.0))
        if cop_k > 1e-9:
            hp_socmin_req_kwh = (
                (soc_min_phys - soc) * buffer_capacity_kwh_th
                + thermal_demand_kwh_th
                + losses_kwh
            ) / cop_k
        else:
            hp_socmin_req_kwh = 0.0
        hp_socmin_req_kwh = float(max(hp_socmin_req_kwh, 0.0))

        hp_applied_kwh = float(
            max(hp_socmin_req_kwh, min(hp_applied_kwh, hp_access_cap_kwh))
        )
        hp_new_kw = 4.0 * hp_applied_kwh

        hp_thermal_out_kwh = hp_applied_kwh * cop_k
        soc_next_raw = soc + (hp_thermal_out_kwh - thermal_served_kwh_th - losses_kwh) / buffer_capacity_kwh_th
        soc_after_raw_ts[k] = soc_next_raw

        # SOC-max safeguard: if SOC would exceed physical maximum, reduce HP to meet SOC_max_phys.
        if enforce_soc_max and soc_next_raw > soc_max_phys:
            # Solve for hp_applied_kwh such that soc_next_raw == soc_max_phys:
            # soc_max_phys = soc + (hp_applied_kwh * cop_k - thermal_act_kwh - losses_kwh) / Cb
            # => hp_applied_kwh = ((soc_max_phys - soc)*Cb + thermal_act_kwh + losses_kwh) / cop_k
            if cop_k > 1e-9:
                hp_cap_kwh = (
                    (soc_max_phys - soc) * buffer_capacity_kwh_th
                    + thermal_demand_kwh_th
                    + losses_kwh
                ) / cop_k
            else:
                hp_cap_kwh = 0.0
            hp_applied_kwh = float(max(min(hp_applied_kwh, hp_cap_kwh), 0.0))
            hp_new_kw = 4.0 * hp_applied_kwh
            hp_thermal_out_kwh = hp_applied_kwh * cop_k
            soc_next_raw = soc + (hp_thermal_out_kwh - thermal_served_kwh_th - losses_kwh) / buffer_capacity_kwh_th
            soc_after_raw_ts[k] = soc_next_raw

        # SOC-min / unmet: use physical SOC min only so the buffer can discharge to soc_min_phys after clipping (PLC does not enforce the planner stress floor).
        if enforce_soc_min and soc_next_raw < soc_min_phys:
            deficit_kwh_th = (soc_min_phys - soc_next_raw) * buffer_capacity_kwh_th
            extra_kwh_el = deficit_kwh_th / cop_k if cop_k > 1e-9 else 0.0
            if extra_kwh_el > 0.0:
                plc_extra_kwh = float(extra_kwh_el)
                plc_active = 1.0
                hp_applied_kwh = float(hp_applied_kwh + extra_kwh_el)
                hp_new_kw = 4.0 * hp_applied_kwh
                # Recompute thermal output and raw SOC after PLC action
                hp_thermal_out_kwh = hp_applied_kwh * cop_k
                soc_next_raw = soc + (hp_thermal_out_kwh - thermal_served_kwh_th - losses_kwh) / buffer_capacity_kwh_th
                soc_after_raw_ts[k] = soc_next_raw
        elif (not enforce_soc_min) and soc_next_raw < soc_min_phys:
            # If we do NOT enforce SOC-min by adding HP energy, keep SOC >= soc_min_phys by allowing unmet thermal demand.
            served_max = (
                hp_thermal_out_kwh
                - losses_kwh
                + (soc - soc_min_phys) * buffer_capacity_kwh_th
            )
            thermal_served_kwh_th = float(
                min(thermal_demand_kwh_th, max(0.0, served_max))
            )
            unmet_thermal_kwh_th = float(max(thermal_demand_kwh_th - thermal_served_kwh_th, 0.0))
            soc_next_raw = soc + (hp_thermal_out_kwh - thermal_served_kwh_th - losses_kwh) / buffer_capacity_kwh_th
            # Guard against numerical noise:
            soc_next_raw = float(max(soc_next_raw, soc_min_phys))
            soc_after_raw_ts[k] = soc_next_raw

        # Final applied signal (includes PLC safeguard)
        hp_plc_extra_kwh_ts[k] = plc_extra_kwh
        plc_active_ts[k] = plc_active
        hp_applied_kwh_ts[k] = hp_applied_kwh
        thermal_served_kwh_th_ts[k] = thermal_served_kwh_th
        unmet_thermal_kwh_th_ts[k] = unmet_thermal_kwh_th

        # Actual grid power after clipping + PLC safeguard (may exceed access power if safeguard needed)
        p_grid_act_kw = 4.0 * (inflex_act_kwh + ev_act_kwh + hp_applied_kwh - pv_act_kwh)
        p_grid_actual_kw_ts[k] = p_grid_act_kw

        new_peak = max(peak_sofar_kw, p_grid_act_kw)
        monthly_peak_so_far[month_key] = new_peak
        monthly_peak_so_far_ts[k] = new_peak
        exceedance_month_so_far[month_key] = max(0.0, float(new_peak - access_kw))

        # No unphysical SOC clipping: keep physics-based SOC (>= soc_min_phys if PLC enabled)
        soc = float(soc_next_raw)
        soc_after_ts[k] = soc

    res = pd.DataFrame(
        {
            "timestamp": plant_df["timestamp"],
            "hp_plan_kwh": hp_plan_kwh_ts,
            "hp_applied_kwh": hp_applied_kwh_ts,
            "hp_applied_kwh_nominal": hp_applied_kwh_nominal_ts,
            "hp_plc_extra_kwh": hp_plc_extra_kwh_ts,
            "plc_active": plc_active_ts,
            "thermal_served_kwh_th": thermal_served_kwh_th_ts,
            "unmet_thermal_kwh_th": unmet_thermal_kwh_th_ts,
            "soc_before": soc_before_ts,
            "soc_after": soc_after_ts,
            "soc_after_raw": soc_after_raw_ts,
            "p_grid_plan_kw": p_grid_plan_kw_ts,
            "grid_clip_limit_kw": grid_clip_limit_kw_ts,
            "p_limit_kw": p_limit_kw_ts,
            "p_grid_actual_kw": p_grid_actual_kw_ts,
            "was_clipped": was_clipped_ts,
            "access_kw": access_kw_ts,
            "mpc_first_interval_grid_kw": mpc_first_interval_grid_kw_ts,
            "monthly_peak_plan_kw": monthly_peak_plan_kw_ts,
            "current_peak_opt_kw": monthly_peak_plan_kw_ts,
            "realized_monthly_peak_kw_before_step": realized_monthly_peak_kw_before_step_ts,
            "forecast_grid_kw": forecast_grid_kw_ts,
            "forecast_access_exceedance_active": forecast_access_exceedance_active_ts,
            "soc_min_planner_floor": soc_min_planner_floor_ts,
            "monthly_peak_so_far_kw": monthly_peak_so_far_ts,
        }
    )

    summary = {
        "monthly_peak_so_far": dict(monthly_peak_so_far),
        "soc_min_phys": float(soc_min_phys),
        "soc_max_phys": float(soc_max_phys),
        "forecast_strategy_inflex": str(forecast_strategy_inflex),
        "forecast_strategy_inflex_stress": str(effective_inflex_stress),
        "forecast_strategy_temperature": str(forecast_strategy_temperature),
        "enforce_soc_min": bool(enforce_soc_min),
        "enforce_soc_max": bool(enforce_soc_max),
        "soc_slack_penalty_eur_per_soc": float(
            soc_min_slack_penalty_eur_per_soc
            if soc_slack_penalty_eur_per_soc is None
            else soc_slack_penalty_eur_per_soc
        ),
        "monthly_peak_price_multiplier": float(monthly_peak_price_multiplier),
        "enable_forecast_stress_soc_floor": bool(enable_forecast_stress_soc_floor),
        "forecast_stress_soc_floor_strength": float(forecast_stress_soc_floor_strength),
    }
    return res, summary

