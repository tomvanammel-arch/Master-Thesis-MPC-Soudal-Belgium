import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure src is on path when run as a script
THIS_DIR = Path(__file__).parent
PROJECT_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from optimization import mpc_ev_24h  # type: ignore
from billing import (
    load_billing_config,
    calculate_monthly_bills,
    calculate_monthly_injection_bills,
)


def _parse_plant_data(plant_path: Path) -> pd.DataFrame:
    df = pd.read_csv(plant_path)
    # Parse timestamps as timezone-aware, then convert to Europe/Brussels and drop tzinfo,
    # so we work consistently in Belgian local (naive) time.
    ts_utc = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["timestamp"] = ts_utc.dt.tz_convert("Europe/Brussels").dt.tz_localize(None)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Filter to 2025 as in deterministic notebooks (Belgian local time)
    start_2025 = pd.Timestamp("2025-01-01 00:00:00")
    end_2025 = pd.Timestamp("2026-01-01 00:00:00")
    df = df[(df["timestamp"] >= start_2025) & (df["timestamp"] < end_2025)].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _load_forecast_series(
    ev_path: Path,
    inflex_path: Path,
    strategy_ev: str,
    strategy_inflex: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load EV and inflex forecasts strictly by row order (no timestamp alignment).

    Assumes:
    - plant1.csv, forecast_ev_*.csv and forecast_inflex_load_*.csv
      all have the same number of rows (35040 for 2025, 15-min).
    """
    ev_df = pd.read_csv(ev_path)
    inflex_df = pd.read_csv(inflex_path)

    ev_col = f"forecast_ev_{strategy_ev}"
    inflex_col = (
        strategy_inflex
        if strategy_inflex.startswith("forecast_inflex_")
        else f"forecast_inflex_{strategy_inflex}"
    )

    if ev_col not in ev_df.columns:
        raise KeyError(f"Column {ev_col!r} not found in {ev_path.name}")
    if inflex_col not in inflex_df.columns:
        inflex_hint_cols = [c for c in inflex_df.columns if c.startswith("forecast_inflex_")]
        raise KeyError(
            f"No inflex forecast column found for strategy {strategy_inflex!r} in "
            f"{inflex_path.name}. Expected column {inflex_col!r}. "
            f"Available inflex columns include: {inflex_hint_cols[:12]}"
        )

    if len(ev_df) != len(inflex_df):
        raise ValueError(
            f"EV and inflex forecast files have different lengths: "
            f"{len(ev_df)} vs {len(inflex_df)}"
        )

    ev_arr = pd.to_numeric(ev_df[ev_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    inflex_arr = (
        pd.to_numeric(inflex_df[inflex_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    )
    return ev_arr, inflex_arr


def _build_inflex_forecast_on_grid(
    plant_df: pd.DataFrame,
    inflex_forecast: pd.Series,
    inflex_col_plant: str = "inflex_load",
) -> pd.Series:
    ts = plant_df["timestamp"]
    return (
        inflex_forecast.reindex(ts)
        .fillna(method="ffill")
        .fillna(plant_df[inflex_col_plant].fillna(0.0))
        .rename("inflex_forecast")
    )


def _month_key(ts: pd.Timestamp) -> str:
    """Return month key like '2025-01'."""
    return ts.to_period("M").strftime("%Y-%m")


def _ev_envelope_energy_remaining_kwh(
    k: int,
    n: int,
    timestamps: pd.Series,
    envelope_kw: np.ndarray,
    deadline_tod: float = 17.0,
) -> float:
    """Sum envelope-deliverable kWh from step k through same day before deadline_tod."""
    date_k = pd.Timestamp(timestamps[k]).date()
    total = 0.0
    for i in range(k, n):
        ti = pd.Timestamp(timestamps[i])
        if ti.date() != date_k:
            break
        tod_i = ti.hour + ti.minute / 60.0
        if tod_i >= deadline_tod:
            break
        total += float(envelope_kw[i]) / 4.0
    return total


def _ev_envelope_headroom_after_step_kwh(
    k: int,
    n: int,
    timestamps: pd.Series,
    envelope_kw: np.ndarray,
    deadline_tod: float = 17.0,
) -> float:
    """Envelope-deliverable kWh from step k+1 through deadline_tod on the same day."""
    if k + 1 >= n:
        return 0.0
    return _ev_envelope_energy_remaining_kwh(
        k + 1, n, timestamps, envelope_kw, deadline_tod=deadline_tod
    )


def _apply_ev_enforce_minimal_after_clip(
    ev_to_deliver_kwh: float,
    ev_clipped_kwh: float,
    envelope_kw_k: float,
    headroom_after_kwh: float,
) -> Tuple[float, float, bool, bool]:
    """
    Defer enforce when remaining after clip fits in future envelope headroom; otherwise
    add the minimum kWh needed so the blended daily target can still be met by 17:00.

    Returns (ev_applied_kwh, ev_enforce_extra_kwh, enforce_active, enforce_deferred).
    """
    ev_clipped_kwh = max(0.0, float(ev_clipped_kwh))
    envelope_cap_kwh = max(0.0, float(envelope_kw_k)) / 4.0
    remaining_after_clip = max(0.0, float(ev_to_deliver_kwh) - ev_clipped_kwh)

    if remaining_after_clip <= float(headroom_after_kwh) + 1e-6:
        return ev_clipped_kwh, 0.0, False, True

    extra_needed = float(ev_to_deliver_kwh) - ev_clipped_kwh - float(headroom_after_kwh)
    ev_applied_kwh = min(envelope_cap_kwh, ev_clipped_kwh + max(0.0, extra_needed))
    extra_kwh = max(0.0, ev_applied_kwh - ev_clipped_kwh)
    return ev_applied_kwh, extra_kwh, extra_kwh > 1e-6, False


def run_ev_online_mpc_1(
    forecast_strategy_ev: str = "a",
    forecast_strategy_inflex: str = "a",
    forecast_strategy_pv: str = "actual",
    ev_deadline_slack_minutes: int = 0,
    enforce_daily_ev_demand: bool = True,
    access_power_by_month: Dict[str, float] = None,
    verbose: bool = True,
    log_prefix: str = "[Online MPC]",
    enable_mpc_window_debug: bool = True,
    mpc_window_debug_csv_path: Optional[str] = None,
    actuator_mode: Literal["full", "planner_only"] = "full",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run a full-year EV-only online myopic MPC simulation (thesis §3.7.2).

    - Plant actuals: `data/plant1.csv`.
    - Forecasts: `output/forecast/forecast_ev_rolling_horizon.csv`,
      `forecast_inflex_load_rolling_horizon.csv`, optional PV forecast CSV.
    - Planner: `mpc_ev_24h` each step (access soft via rolling-12 over-usage; exogenous
      EV column: first horizon step `ev_actual[k-1]`, remaining steps flat forecast at k).
    - Actuator (`actuator_mode="full"`): clipping with **Equation (3.50)** in the MPC
      region; optional **headroom-aware enforce** (12:00–17:00); catch-up after slack.
    - Actuator (`actuator_mode="planner_only"`): thesis steps **0, 1, 2, 6** only —
      apply first-step planner output with actual inflex/PV (no clip, catch-up, enforce;
      `ev_deadline_slack_minutes=0`; MPC on full weekday 07:00–17:00 window).

    Returns
    -------
    results_df : pd.DataFrame
        Full-year simulation results at 15 min resolution.
    summary : dict
        Aggregate summary (e.g. monthly peaks, uncharged energy stats, bills).
    """
    planner_only = actuator_mode == "planner_only"
    if planner_only:
        if ev_deadline_slack_minutes != 0 and verbose:
            print(
                f"{log_prefix} planner_only: ignoring ev_deadline_slack_minutes="
                f"{ev_deadline_slack_minutes} (forced to 0)"
            )
        if enforce_daily_ev_demand and verbose:
            print(f"{log_prefix} planner_only: ignoring enforce_daily_ev_demand=True")
        ev_deadline_slack_minutes = 0
        enforce_daily_ev_demand = False
        enable_mpc_window_debug = False
    # Paths
    plant_path = PROJECT_ROOT / "data" / "plant1.csv"
    # Use the rolling-horizon EV forecast exported by notebook 05
    forecast_ev_path = PROJECT_ROOT / "output" / "forecast" / "forecast_ev_rolling_horizon.csv"
    forecast_inflex_path = PROJECT_ROOT / "output" / "forecast" / "forecast_inflex_load_rolling_horizon.csv"
    forecast_pv_path = PROJECT_ROOT / "output" / "forecast" / "forecast_pv_rolling_horizon.csv"
    billing_path = PROJECT_ROOT / "config" / "billing.yaml"

    if verbose:
        print("=" * 80)
        title = (
            "EV-only Online MPC 1 – Planner-only (thesis steps 0,1,2,6)"
            if planner_only
            else "EV-only Online MPC 1 – Full-year simulation"
        )
        print(title)
        print("=" * 80)
        print(f"  EV forecast strategy:        {forecast_strategy_ev}")
        print(f"  Inflex forecast strategy:    {forecast_strategy_inflex}")
        print(f"  PV forecast strategy:        {forecast_strategy_pv}")
        print(f"  EV deadline slack (min): {ev_deadline_slack_minutes}")
        print(f"  Enforce daily EV demand: {enforce_daily_ev_demand}")
        print(f"  Plant data:             {plant_path}")
        print(f"  EV forecast:            {forecast_ev_path}")
        print(f"  Inflex forecast:        {forecast_inflex_path}")
        print(f"  PV forecast:            {forecast_pv_path}")
        print(f"  Billing config:         {billing_path}")
        print("-" * 80)

    plant_df = _parse_plant_data(plant_path)
    ev_forecast_arr, inflex_forecast_arr = _load_forecast_series(
        forecast_ev_path,
        forecast_inflex_path,
        strategy_ev=forecast_strategy_ev,
        strategy_inflex=forecast_strategy_inflex,
    )

    # EV and inflex series on plant grid (assume same row order as forecasts)
    ts = plant_df["timestamp"].copy()
    ev_actual = plant_df["ev"].astype(float).values

    # Optional PV forecast: strictly by row order (no timestamp alignment).
    # When forecast_strategy_pv == "actual", the optimiser sees actual PV.
    if forecast_strategy_pv == "actual":
        pv_for_mpc_arr = plant_df["pv_production"].astype(float).to_numpy()
    else:
        # Accept either:
        # - short strategy token (e.g. "k7", "chronos2_elia_p50"), mapped to
        #   "pv_forecast_kWh_15min_<token>"
        # - full column name already prefixed with "pv_forecast_kWh_15min_"
        pv_forecast_df = pd.read_csv(forecast_pv_path)
        if len(pv_forecast_df) != len(plant_df):
            raise ValueError(
                f"PV forecast and plant data must have same length; "
                f"got pv_forecast={len(pv_forecast_df)}, plant={len(plant_df)}"
            )
        pv_col_name = (
            forecast_strategy_pv
            if forecast_strategy_pv.startswith("pv_forecast_kWh_15min_")
            else f"pv_forecast_kWh_15min_{forecast_strategy_pv}"
        )
        if pv_col_name not in pv_forecast_df.columns:
            pv_hint_cols = [c for c in pv_forecast_df.columns if c.startswith("pv_forecast_kWh_15min_")]
            raise KeyError(
                f"No PV forecast column found for strategy {forecast_strategy_pv!r} in "
                f"{forecast_pv_path.name}. Expected column {pv_col_name!r}. "
                f"Available PV columns include: {pv_hint_cols[:14]}"
            )
        pv_for_mpc_arr = (
            pd.to_numeric(pv_forecast_df[pv_col_name], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

    if (
        len(ev_forecast_arr) != len(plant_df)
        or len(inflex_forecast_arr) != len(plant_df)
    ):
        raise ValueError(
            f"Forecast and plant data must have same length; "
            f"got ev={len(ev_forecast_arr)}, inflex={len(inflex_forecast_arr)}, "
            f"plant={len(plant_df)}"
        )
    ev_forecast_aligned = ev_forecast_arr.astype(float)
    inflex_forecast_aligned = inflex_forecast_arr.astype(float)

    plant_df = plant_df.copy()
    plant_df["inflex_forecast"] = inflex_forecast_aligned
    # PV series used by the optimiser (forecast or actual depending on strategy)
    plant_df["pv_for_mpc"] = pv_for_mpc_arr

    if access_power_by_month is None:
        raise ValueError(
            "run_ev_online_mpc_1 requires 'access_power_by_month' "
            "mapping month key 'YYYY-MM' -> access power in kW."
        )

    # Validate that we have an access power entry for every month in the plant data
    month_keys_in_data = sorted({_month_key(ts_i) for ts_i in plant_df["timestamp"]})
    missing_months = [m for m in month_keys_in_data if m not in access_power_by_month]
    if missing_months:
        raise KeyError(
            "access_power_by_month is missing entries for months: "
            + ", ".join(missing_months)
        )

    # Initialize state
    n = len(plant_df)

    # Monthly peak so far (actual) – keys like '2025-01'
    monthly_peak_so_far: Dict[str, float] = {}

    # Rolling 12-month exceedance state (for mpc_ev_24h, aligned with online HP logic)
    finalized_exceedance_last12: deque = deque(maxlen=12)
    exceedance_month_so_far: Dict[str, float] = {}
    prev_month_key: Optional[str] = None

    # Preload billing config (for energy/peak prices, later billing)
    billing_cfg = load_billing_config(str(billing_path))

    # Outputs
    ev_applied = np.zeros(n, dtype=float)
    p_grid_actual = np.zeros(n, dtype=float)
    monthly_peak_time_series = np.zeros(n, dtype=float)
    effective_peak_time_series = np.zeros(n, dtype=float)
    # Tracking daily EV energy signals for analysis/plotting
    ev_daily_demand_ts = np.zeros(n, dtype=float)
    ev_charged_so_far_ts = np.zeros(n, dtype=float)
    ev_to_deliver_ts = np.zeros(n, dtype=float)
    # Per-step MPC/clip diagnostics (NaN where not applicable)
    ev_plan_kwh_ts = np.full(n, np.nan, dtype=float)
    p_grid_plan_kw_ts = np.full(n, np.nan, dtype=float)
    p_limit_kw_ts = np.full(n, np.nan, dtype=float)
    access_kw_ts = np.full(n, np.nan, dtype=float)
    current_peak_opt_kw_ts = np.full(n, np.nan, dtype=float)
    was_clipped_ts = np.zeros(n, dtype=float)
    ev_envelope_remaining_kwh_ts = np.full(n, np.nan, dtype=float)
    ev_envelope_headroom_after_kwh_ts = np.full(n, np.nan, dtype=float)
    ev_envelope_feasible_ts = np.full(n, np.nan, dtype=float)
    ev_enforce_extra_kwh_ts = np.zeros(n, dtype=float)
    ev_enforce_active_ts = np.zeros(n, dtype=float)
    ev_enforce_deferred_ts = np.zeros(n, dtype=float)
    # Physical EV power envelope (kW) derived from actual EV presence (plant_df["ev"])
    # Used to cap catch-up setpoints before grid clipping.
    ev_power_envelope_base_kw_ts = np.zeros(n, dtype=float)

    # Per-day metadata for dynamic remaining EV computation (heuristic blend)
    plant_df["date"] = ts.dt.date
    # Day start/end indices
    day_start_idx: Dict[object, int] = {}
    day_end_idx: Dict[object, int] = {}
    for d, idxs in plant_df.groupby("date").indices.items():
        idx_list = list(sorted(idxs))
        day_start_idx[d] = int(idx_list[0])
        day_end_idx[d] = int(idx_list[-1])
    # Daily totals for forecasted and actual EV (kWh/day)
    daily_ev_forecast_total = (
        pd.Series(ev_forecast_aligned, index=plant_df["date"])
        .groupby(level=0)
        .sum()
        .to_dict()
    )
    daily_ev_actual_total = (
        plant_df.groupby("date")["ev"].sum().astype(float).to_dict()
    )
    # Track how much EV we have actually charged per day so far (kWh)
    charged_ev_by_date: Dict[object, float] = {
        d: 0.0 for d in plant_df["date"].unique()
    }

    horizon_len = 96  # 24 h

    # Optional detailed debug logging for each MPC window (inputs + outputs).
    # This can generate a large CSV (~millions of rows) for a full-year run.
    debug_rows: List[Dict[str, object]] = [] if enable_mpc_window_debug else []

    # Track previous date for daily progress printing
    prev_date = None

    # ------------------------------------------------------------
    # Helper: deterministic dynamic EV power envelope (kW) for full year
    # ------------------------------------------------------------
    def _build_dynamic_ev_envelope(ev_kwh_arr: np.ndarray, timestamps: pd.Series) -> np.ndarray:
        """
        Build a deterministic-style dynamic EV power envelope (kW) for the full year.

        For each day d:
          - P_bench(t) = ev_kwh_arr(t) * 4 (kWh/15min -> kW)
          - P_cum(d,t) = max_{τ≤t} P_bench(τ)
          - P_env(d,t) = P_cum(d,t) for t ≤ 15:30
                        P_cum(d,15:30) * (17 - t) / 1.5 for 15:30 < t < 17:00
                        0 for t ≥ 17:00
        """
        ev_kwh = np.asarray(ev_kwh_arr, dtype=float)
        power_bench = ev_kwh * 4.0

        ts_local = pd.to_datetime(timestamps)
        dates = ts_local.dt.date.values
        tod = ts_local.dt.hour.values + ts_local.dt.minute.values / 60.0

        env = np.zeros(len(ts_local), dtype=float)
        unique_dates = sorted(np.unique(dates))

        for d in unique_dates:
            mask = dates == d
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                continue

            day_power = power_bench[idxs]
            day_tod = tod[idxs]

            cum_max = np.maximum.accumulate(day_power)

            mask_1530 = day_tod <= 15.5
            if np.any(mask_1530):
                i_1530 = np.where(mask_1530)[0][-1]
                p_max_1530 = cum_max[i_1530]
            else:
                p_max_1530 = cum_max[0] if len(cum_max) > 0 else 0.0

            for j, idx in enumerate(idxs):
                t = day_tod[j]
                if t <= 15.5:
                    env[idx] = cum_max[j]
                elif 15.5 < t < 17.0:
                    env[idx] = p_max_1530 * (17.0 - t) / 1.5
                else:
                    env[idx] = 0.0

        return env

    # ------------------------------------------------------------
    # Full-year deterministic envelopes: actual vs forecast, then blended
    # ------------------------------------------------------------
    # "Actual" envelope from plant_df["ev"] (kWh/15-min)
    ev_power_envelope_actual_kw_ts = _build_dynamic_ev_envelope(ev_actual, ts)
    # Forecast envelope from ev_forecast_aligned (kWh/15-min)
    ev_power_envelope_forecast_kw_ts = _build_dynamic_ev_envelope(
        ev_forecast_aligned, ts
    )
    # For catch-up region we still use the actual envelope as the physical cap
    ev_power_envelope_base_kw_ts = ev_power_envelope_actual_kw_ts.copy()

    for k in range(n):
        t_k = ts.iloc[k]
        date_k = plant_df["date"].iloc[k]
        month_key = _month_key(t_k)

        # Month rollover: finalize previous month's exceedance into rolling-12 deque.
        if prev_month_key is None:
            prev_month_key = month_key
        elif month_key != prev_month_key:
            prev_ex = float(exceedance_month_so_far.get(prev_month_key, 0.0))
            finalized_exceedance_last12.append(prev_ex)
            prev_month_key = month_key

        # Progress printouts: once per (new) calendar day
        if verbose and (k == 0 or date_k != prev_date):
            print(f"{log_prefix} Simulating day {date_k} (step {k+1}/{n}, month {month_key})")
        prev_date = date_k

        # Time-of-day and weekday information (Belgium local time, naive timestamps)
        tod = t_k.hour + t_k.minute / 60.0
        dow = t_k.dayofweek  # Monday=0 ... Sunday=6

        # Define EV optimization activity window: weekdays 07:00–17:00
        is_weekday = dow < 5
        in_ev_window = 7.0 <= tod < 17.0
        opt_active = is_weekday and in_ev_window

        # Time-of-day and regions for MPC vs catch-up within the EV window.
        # The slack does NOT shift the physical envelope; instead, the MPC problem
        # is constructed with a forced-to-zero EV envelope tail during the last
        # `ev_deadline_slack_minutes` before the true 17:00 deadline. That tail is
        # handled by the catch-up logic below.
        base_start = 15.5  # 15:30
        base_end = 17.0
        slack_hours = ev_deadline_slack_minutes / 60.0
        mpc_ramp_start = base_start - slack_hours
        mpc_ramp_end = base_end - slack_hours

        # MPC is only meaningful once the EV window starts at 07:00.
        # Use [7:00, mpc_ramp_end) for MPC, [mpc_ramp_end, base_end) for catch-up,
        # so the final slack period before the true deadline is treated as catch-up.
        if planner_only:
            in_mpc_region = bool(in_ev_window)
            in_catchup_region = False
        else:
            in_mpc_region = (tod >= 7.0) and (tod < mpc_ramp_end)
            in_catchup_region = (tod >= mpc_ramp_end) and (tod < base_end)

        # (No routine debug prints here; only on MPC failure in the try/except below.)

        # Heuristic daily EV energy demand blending forecast and actual
        required_forecast = float(daily_ev_forecast_total.get(date_k, 0.0))
        required_actual = float(daily_ev_actual_total.get(date_k, 0.0))
        if tod <= 7.0:
            w_actual = 0.0
        elif tod >= 12.0:
            w_actual = 1.0
        else:
            w_actual = (tod - 7.0) / 5.0  # linear from 07:00 to 12:00
        ev_daily_demand = (1.0 - w_actual) * required_forecast + w_actual * required_actual
        charged_so_far = charged_ev_by_date.get(date_k, 0.0)
        ev_to_deliver = max(ev_daily_demand - charged_so_far, 0.0)

        ev_daily_demand_ts[k] = ev_daily_demand
        ev_charged_so_far_ts[k] = charged_so_far
        ev_to_deliver_ts[k] = ev_to_deliver

        # Access power & peak so far (must be provided explicitly by caller)
        try:
            access_kw = float(access_power_by_month[month_key])
        except KeyError as e:
            raise KeyError(f"Missing access power for month_key={month_key}") from e
        peak_sofar_kw = monthly_peak_so_far.get(month_key, 0.0)
        access_kw_ts[k] = access_kw

        if not opt_active:
            # Outside optimization window (night or weekend): no EV from online MPC.
            inflex_act_kwh = float(plant_df["inflex_load"].iloc[k])
            pv_act_kwh = float(plant_df["pv_production"].iloc[k])
            ev_applied_kwh = 0.0
            p_grid_act = 4.0 * (inflex_act_kwh + ev_applied_kwh - pv_act_kwh)

            ev_applied[k] = ev_applied_kwh
            p_grid_actual[k] = p_grid_act
            monthly_peak_so_far[month_key] = max(
                monthly_peak_so_far.get(month_key, 0.0), p_grid_act
            )
            exceedance_month_so_far[month_key] = max(
                0.0, float(monthly_peak_so_far[month_key] - access_kw)
            )
            monthly_peak_time_series[k] = monthly_peak_so_far[month_key]
            effective_peak_time_series[k] = monthly_peak_so_far[month_key]

            # Skip MPC / catch-up logic entirely for this step
            continue

        if in_mpc_region:
            # Determine 24 h window [k, k+96) only when running MPC
            k_end = min(k + horizon_len, n)
            df_window = plant_df.loc[
                k : k_end - 1,
                ["timestamp", "pv_for_mpc", "inflex_forecast", "price"],
            ].copy()

            # Build EV horizon according to rule:
            #   h[0] = ev_actual[k - delay_steps]
            #   h[j>=1] = ev_forecast_aligned[k]  (flat forecast over horizon)
            window_len = k_end - k
            ev_horizon = np.zeros(window_len, dtype=float)
            delay_steps = 1
            idx0 = max(0, k - delay_steps)
            ev_horizon[0] = ev_actual[idx0]
            if window_len > 1:
                ev_horizon[1:] = ev_forecast_aligned[k]

            df_window["ev"] = ev_horizon
            # Provide actual EV for envelope blending inside mpc_ev_24h
            df_window["ev_actual"] = plant_df["ev"].iloc[k:k_end].to_numpy()
            # Provide a fixed blended EV power envelope slice for this window (kW).
            # At current step k, use the deterministic actual-envelope from the
            # previous step as the cap. For all future steps in the horizon,
            # use the blended deterministic envelope:
            #   env_blend(t) = w(k) * env_actual(t) + (1 - w(k)) * env_forecast(t),
            # where w(k) is the "unfolded fraction" based on current time-of-day.
            env_actual_window = ev_power_envelope_actual_kw_ts[k:k_end]
            env_forecast_window = ev_power_envelope_forecast_kw_ts[k:k_end]
            # Reuse w_actual computed above as the unfolded fraction at time k
            w_unfold = w_actual
            env_blend_window = (
                w_unfold * env_actual_window + (1.0 - w_unfold) * env_forecast_window
            )
            # For early-day MPC runs (up to 12:00), cap the current step
            # using the previous-step actual envelope to reflect realistic
            # use of information during the forecast-driven period.
            # From 12:00 onward (when actual EV behaviour is fully unfolded),
            # cap using the current-step actual envelope so that online
            # power cannot exceed the deterministic envelope at the same time.
            if k > 0:
                if tod < 12.0:
                    env_blend_window[0] = ev_power_envelope_actual_kw_ts[k - 1]
                else:
                    env_blend_window[0] = ev_power_envelope_actual_kw_ts[k]
            # Export both the forecast envelope and the blended envelope to df_window
            # so that they appear in the detailed MPC debug CSV (as inputs).
            df_window["ev_power_envelope_forecast_kw"] = env_forecast_window
            df_window["ev_power_envelope_fixed_kw"] = env_blend_window
            df_window.rename(
                columns={
                    "pv_for_mpc": "pv_production",
                    "inflex_forecast": "inflex_load",
                },
                inplace=True,
            )

            # Monthly peak so far input for this window (copy)
            monthly_peak_input = dict(monthly_peak_so_far)

            # --- 24 h planner call ---
            try:
                # Remaining EV energy requirements (kWh) for all calendar days
                # that appear in this 24h window.
                # - For the *current* day (date_k), use the blended remaining
                #   energy ev_to_deliver (forecast→actual blend minus what has
                #   actually been charged so far, post-clipping).
                # - For *future* days in the window, use the forecast-only
                #   daily total from daily_ev_forecast_total (no energy
                #   delivered yet at this point in the simulation).
                dates_in_window = (
                    pd.to_datetime(df_window["timestamp"]).dt.date.unique().tolist()
                )
                daily_ev_remaining: Dict[object, float] = {}
                for d in dates_in_window:
                    if d == date_k:
                        daily_ev_remaining[d] = float(ev_to_deliver)
                    else:
                        daily_ev_remaining[d] = float(
                            daily_ev_forecast_total.get(d, 0.0)
                        )

                roll12_completed = (
                    float(max(finalized_exceedance_last12))
                    if len(finalized_exceedance_last12)
                    else 0.0
                )
                window_month_keys = sorted(
                    {_month_key(ts_i) for ts_i in df_window["timestamp"]}
                )
                rolling12_max_exceedance_so_far_by_month: Dict[str, float] = {}
                for mk in window_month_keys:
                    if mk == month_key:
                        rolling12_max_exceedance_so_far_by_month[mk] = max(
                            roll12_completed,
                            float(exceedance_month_so_far.get(mk, 0.0)),
                        )
                    else:
                        rolling12_max_exceedance_so_far_by_month[mk] = roll12_completed

                window_res, window_summary = mpc_ev_24h(
                    df_window=df_window,
                    config_path=str(billing_path),
                    monthly_peak_so_far=monthly_peak_input,
                    timestamp_col="timestamp",
                    pv_col="pv_production",
                    inflex_load_col="inflex_load",
                    price_col="price",
                    ev_col="ev",
                    ev_deadline_slack_minutes=ev_deadline_slack_minutes,
                    daily_ev_remaining=daily_ev_remaining,
                    access_power_by_month=access_power_by_month,
                    rolling12_max_exceedance_so_far_by_month=rolling12_max_exceedance_so_far_by_month,
                )
            except Exception as e:
                # Debug dump of the window that caused infeasibility/failure
                print("=" * 80)
                print("[DEBUG] MPC window failed at k="
                      f"{k}, t_k={t_k}, month={month_key}")
                print("[DEBUG] df_window.head():")
                print(df_window.head(12))
                try:
                    daily_ev_window = (
                        df_window.assign(date=df_window["timestamp"].dt.date)
                        .groupby("date")["ev"]
                        .sum()
                    )
                    print("[DEBUG] Daily EV in window (kWh):")
                    print(daily_ev_window)
                except Exception:
                    pass
                print("[DEBUG] access_kw=", access_kw,
                      "peak_sofar_kw=", peak_sofar_kw)
                print("[DEBUG] ev_horizon (first 12 steps) =", ev_horizon[:12])
                print("=" * 80)
                raise

            if enable_mpc_window_debug:
                # --- Detailed debug logging for this MPC window ---
                try:
                    window_len_logged = len(df_window)
                    for j in range(window_len_logged):
                        row: Dict[str, object] = {}
                        # Global simulation context
                        row["k_global"] = k
                        row["horizon_idx"] = j
                        row["t_k"] = t_k
                        row["timestamp_window"] = df_window["timestamp"].iloc[j]
                        row["date_k"] = date_k
                        row["month_key"] = month_key
                        row["tod_k"] = tod
                        row["dow_k"] = dow
                        row["is_weekday"] = is_weekday
                        row["in_ev_window"] = in_ev_window
                        row["in_mpc_region"] = in_mpc_region
                        row["in_catchup_region"] = in_catchup_region
                        row["opt_active"] = opt_active

                        # Daily EV energy blending at k
                        row["required_forecast_kwh"] = required_forecast
                        row["required_actual_kwh"] = required_actual
                        row["ev_daily_demand_kwh"] = ev_daily_demand
                        row["charged_so_far_kwh"] = charged_so_far
                        row["ev_to_deliver_kwh"] = ev_to_deliver

                        # Access power and peak state at k
                        row["access_kw_at_k"] = access_kw
                        row["peak_sofar_kw_at_k"] = peak_sofar_kw
                        row["p_grid_plan_kw_at_k"] = p_grid_plan
                        row["p_limit_kw_at_k"] = p_limit
                        row["was_clipped_at_k"] = was_clipped

                        # Window inputs (what goes into MPC)
                        for col in df_window.columns:
                            row[f"in_{col}"] = df_window[col].iloc[j]

                        # Window outputs (what comes out of MPC)
                        for col in window_res.columns:
                            row[f"opt_{col}"] = window_res[col].iloc[j]

                        debug_rows.append(row)
                except Exception:
                    # Debug logging is best-effort; do not break main simulation
                    pass

            ev_plan_kwh = float(window_res["ev_charge"].iloc[0])
            ev_plan_kw = 4.0 * ev_plan_kwh
            ev_plan_kwh_ts[k] = ev_plan_kwh

            monthly_peak_plan_dict = window_summary.get("monthly_peak_plan", {})
            current_peak_opt_kw = float(
                monthly_peak_plan_dict.get(month_key, peak_sofar_kw)
            )
            current_peak_opt_kw_ts[k] = current_peak_opt_kw

            # Planned grid power with actual inflex & PV
            inflex_act_kwh = float(plant_df["inflex_load"].iloc[k])
            pv_act_kwh = float(plant_df["pv_production"].iloc[k])
            p_grid_plan = 4.0 * (inflex_act_kwh + ev_plan_kwh - pv_act_kwh)
            p_grid_plan_kw_ts[k] = p_grid_plan

            if planner_only:
                # Thesis steps 3+6: apply planner first step; actual inflex/PV for realised grid.
                p_limit = float("nan")
                p_limit_kw_ts[k] = p_limit
                was_clipped = False
                was_clipped_ts[k] = 0.0
                ev_enforce_extra_kwh = 0.0
                ev_enforce_active = 0.0
                ev_enforce_deferred = 0.0
                ev_enforce_extra_kwh_ts[k] = 0.0
                ev_enforce_active_ts[k] = 0.0
                ev_enforce_deferred_ts[k] = 0.0
                ev_applied_kwh = ev_plan_kwh
            else:
                # Real-time peak limit (thesis Eq. 3.50): P_lim = min(P_access, max(P_peak,sofar, P_target))
                p_target_kw = current_peak_opt_kw
                inner = max(peak_sofar_kw, p_target_kw)
                p_limit = min(access_kw, inner) if access_kw > 0 else inner
                p_limit_kw_ts[k] = p_limit

                # Clipping flag: True when planned grid power exceeds allowed limit
                was_clipped = p_grid_plan > p_limit

                if not was_clipped:
                    p_ev_new = ev_plan_kw
                else:
                    delta_p = p_grid_plan - p_limit
                    p_ev_new = max(ev_plan_kw - delta_p, 0.0)

                was_clipped_ts[k] = 1.0 if was_clipped else 0.0
                ev_enforce_extra_kwh = 0.0
                ev_enforce_active = 0.0
                ev_enforce_deferred = 0.0
                ev_clipped_kwh = p_ev_new / 4.0

                headroom_after_kwh = 0.0
                if tod >= 12.0:
                    headroom_kwh = _ev_envelope_energy_remaining_kwh(
                        k, n, ts, ev_power_envelope_actual_kw_ts
                    )
                    headroom_after_kwh = _ev_envelope_headroom_after_step_kwh(
                        k, n, ts, ev_power_envelope_actual_kw_ts
                    )
                    ev_envelope_remaining_kwh_ts[k] = headroom_kwh
                    ev_envelope_headroom_after_kwh_ts[k] = headroom_after_kwh
                    ev_envelope_feasible_ts[k] = (
                        1.0 if ev_to_deliver <= headroom_kwh + 1e-6 else 0.0
                    )

                if enforce_daily_ev_demand and tod >= 12.0 and was_clipped:
                    ev_clipped_kwh, ev_enforce_extra_kwh, ev_enforce_active, ev_enforce_deferred = (
                        _apply_ev_enforce_minimal_after_clip(
                            ev_to_deliver,
                            ev_clipped_kwh,
                            float(ev_power_envelope_actual_kw_ts[k]),
                            headroom_after_kwh,
                        )
                    )
                    p_ev_new = 4.0 * ev_clipped_kwh

                ev_enforce_extra_kwh_ts[k] = ev_enforce_extra_kwh
                ev_enforce_active_ts[k] = ev_enforce_active
                ev_enforce_deferred_ts[k] = ev_enforce_deferred

                ev_applied_kwh = ev_clipped_kwh
            ev_applied[k] = ev_applied_kwh
            charged_ev_by_date[date_k] = charged_ev_by_date.get(date_k, 0.0) + ev_applied_kwh

            p_grid_act = 4.0 * (inflex_act_kwh + ev_applied_kwh - pv_act_kwh)
            p_grid_actual[k] = p_grid_act

            new_peak = max(peak_sofar_kw, p_grid_act)
            monthly_peak_so_far[month_key] = new_peak
            exceedance_month_so_far[month_key] = max(0.0, float(new_peak - access_kw))

            monthly_peak_time_series[k] = float(monthly_peak_so_far[month_key])
            effective_peak_time_series[k] = float(monthly_peak_so_far[month_key])

        elif in_catchup_region and not planner_only:
            # --- Catch-up window: try to deliver remaining energy now, capped by clipping ---
            # Use ev_to_deliver from the blended daily demand minus charged_so_far.
            remaining = ev_to_deliver
            ev_enforce_extra_kwh = 0.0
            ev_enforce_active = 0.0
            ev_enforce_deferred = 0.0
            was_clipped = False

            if remaining <= 0.0:
                inflex_act_kwh = float(plant_df["inflex_load"].iloc[k])
                pv_act_kwh = float(plant_df["pv_production"].iloc[k])
                ev_applied_kwh = 0.0
                p_grid_act = 4.0 * (inflex_act_kwh + ev_applied_kwh - pv_act_kwh)
            else:
                p_ev_ideal = remaining / 0.25  # kW
                p_ev_env = float(ev_power_envelope_base_kw_ts[k])
                p_ev_set = min(p_ev_ideal, p_ev_env)

                inflex_act_kwh = float(plant_df["inflex_load"].iloc[k])
                pv_act_kwh = float(plant_df["pv_production"].iloc[k])
                p_grid_plan = 4.0 * (inflex_act_kwh + p_ev_set / 4.0 - pv_act_kwh)
                p_grid_plan_kw_ts[k] = p_grid_plan

                peak_sofar_kw = monthly_peak_so_far.get(month_key, 0.0)
                p_limit = min(access_kw, peak_sofar_kw) if access_kw > 0 else peak_sofar_kw
                p_limit_kw_ts[k] = p_limit

                was_clipped = p_grid_plan > p_limit
                if not was_clipped:
                    p_ev_new = p_ev_set
                else:
                    delta_p = p_grid_plan - p_limit
                    p_ev_new = max(p_ev_set - delta_p, 0.0)

                ev_clipped_kwh = p_ev_new / 4.0
                headroom_after_kwh = 0.0
                if tod >= 12.0:
                    headroom_kwh = _ev_envelope_energy_remaining_kwh(
                        k, n, ts, ev_power_envelope_actual_kw_ts
                    )
                    headroom_after_kwh = _ev_envelope_headroom_after_step_kwh(
                        k, n, ts, ev_power_envelope_actual_kw_ts
                    )
                    ev_envelope_remaining_kwh_ts[k] = headroom_kwh
                    ev_envelope_headroom_after_kwh_ts[k] = headroom_after_kwh
                    ev_envelope_feasible_ts[k] = (
                        1.0 if ev_to_deliver <= headroom_kwh + 1e-6 else 0.0
                    )

                if enforce_daily_ev_demand and tod >= 12.0 and was_clipped:
                    ev_clipped_kwh, ev_enforce_extra_kwh, ev_enforce_active, ev_enforce_deferred = (
                        _apply_ev_enforce_minimal_after_clip(
                            ev_to_deliver,
                            ev_clipped_kwh,
                            float(ev_power_envelope_base_kw_ts[k]),
                            headroom_after_kwh,
                        )
                    )

                ev_applied_kwh = ev_clipped_kwh
                p_grid_act = 4.0 * (inflex_act_kwh + ev_applied_kwh - pv_act_kwh)

            was_clipped_ts[k] = 1.0 if was_clipped else 0.0
            ev_enforce_extra_kwh_ts[k] = ev_enforce_extra_kwh
            ev_enforce_active_ts[k] = ev_enforce_active
            ev_enforce_deferred_ts[k] = ev_enforce_deferred

            ev_applied[k] = ev_applied_kwh
            charged_ev_by_date[date_k] = charged_ev_by_date.get(date_k, 0.0) + ev_applied_kwh
            p_grid_actual[k] = p_grid_act
            monthly_peak_so_far[month_key] = max(
                monthly_peak_so_far.get(month_key, 0.0), p_grid_act
            )
            exceedance_month_so_far[month_key] = max(
                0.0, float(monthly_peak_so_far[month_key] - access_kw)
            )
            monthly_peak_time_series[k] = monthly_peak_so_far[month_key]
            effective_peak_time_series[k] = monthly_peak_so_far[month_key]

        else:
            # Outside EV window: no EV from online MPC (could be extended).
            inflex_act_kwh = float(plant_df["inflex_load"].iloc[k])
            pv_act_kwh = float(plant_df["pv_production"].iloc[k])
            ev_applied_kwh = 0.0
            p_grid_act = 4.0 * (inflex_act_kwh + ev_applied_kwh - pv_act_kwh)

            ev_applied[k] = ev_applied_kwh
            p_grid_actual[k] = p_grid_act
            monthly_peak_so_far[month_key] = max(
                monthly_peak_so_far.get(month_key, 0.0), p_grid_act
            )
            exceedance_month_so_far[month_key] = max(
                0.0, float(monthly_peak_so_far[month_key] - access_kw)
            )
            monthly_peak_time_series[k] = monthly_peak_so_far[month_key]
            effective_peak_time_series[k] = monthly_peak_so_far[month_key]

    # Build result DataFrame
    res = plant_df.copy()
    res["ev_online_mpc"] = ev_applied
    res["grid_power_online"] = p_grid_actual
    res["monthly_peak_so_far"] = [monthly_peak_so_far[_month_key(t)] for t in res["timestamp"]]
    res["monthly_peak_plan_series"] = monthly_peak_time_series
    res["effective_peak_series"] = effective_peak_time_series
    res["ev_daily_demand_kwh"] = ev_daily_demand_ts
    res["ev_charged_so_far_kwh"] = ev_charged_so_far_ts
    res["ev_to_deliver_kwh"] = ev_to_deliver_ts
    res["ev_plan_kwh"] = ev_plan_kwh_ts
    res["p_grid_plan_kw"] = p_grid_plan_kw_ts
    res["p_limit_kw"] = p_limit_kw_ts
    res["access_kw"] = access_kw_ts
    res["current_peak_opt_kw"] = current_peak_opt_kw_ts
    res["ev_power_envelope_base_kw"] = ev_power_envelope_base_kw_ts
    res["was_clipped"] = was_clipped_ts
    res["ev_envelope_remaining_kwh"] = ev_envelope_remaining_kwh_ts
    res["ev_envelope_headroom_after_kwh"] = ev_envelope_headroom_after_kwh_ts
    res["ev_envelope_feasible"] = ev_envelope_feasible_ts
    res["ev_enforce_extra_kwh"] = ev_enforce_extra_kwh_ts
    res["ev_enforce_active"] = ev_enforce_active_ts
    res["ev_enforce_deferred"] = ev_enforce_deferred_ts

    # Uncharged EV per day (relative to actual EV demand)
    ev_actual_by_day = plant_df.groupby("date")["ev"].sum().to_dict()
    delivered_by_day = res.groupby("date")["ev_online_mpc"].sum().to_dict()
    all_dates = set(ev_actual_by_day.keys()) | set(delivered_by_day.keys())
    uncharged_kwh = {
        d: max(ev_actual_by_day.get(d, 0.0) - delivered_by_day.get(d, 0.0), 0.0)
        for d in all_dates
    }
    res["uncharged_kwh"] = res["date"].map(uncharged_kwh)

    # Billing (online result)
    res_for_billing = res.copy()
    # Simple approximation: grid_consumption/grid_injection from grid_power_online
    res_for_billing["grid_consumption"] = np.maximum(res["grid_power_online"], 0.0) / 4.0
    res_for_billing["grid_injection"] = np.maximum(-res["grid_power_online"], 0.0) / 4.0
    # Map monthly access power used for online MPC into billing DataFrame
    res_for_billing["month_key"] = res_for_billing["timestamp"].apply(_month_key)
    res_for_billing["access_power_online_kw"] = res_for_billing["month_key"].map(access_power_by_month)
    if res_for_billing["access_power_online_kw"].isna().any():
        bad_months = sorted(
            set(res_for_billing.loc[res_for_billing["access_power_online_kw"].isna(), "month_key"])
        )
        raise KeyError(
            "access_power_by_month is missing entries for months in billing: "
            + ", ".join(bad_months)
        )

    bills = calculate_monthly_bills(
        res_for_billing,
        billing_cfg,
        access_power_col="access_power_online_kw",
    )
    inj_bills = calculate_monthly_injection_bills(res_for_billing, billing_cfg)

    summary = {
        "n_steps": n,
        "monthly_peak_so_far": dict(monthly_peak_so_far),
        "uncharged_kwh_by_day": uncharged_kwh,
        "bills": bills,
        "injection_bills": inj_bills,
        "forecast_strategy_ev": forecast_strategy_ev,
        "forecast_strategy_inflex": forecast_strategy_inflex,
        "forecast_strategy_pv": forecast_strategy_pv,
        "ev_deadline_slack_minutes": ev_deadline_slack_minutes,
        "enforce_daily_ev_demand": enforce_daily_ev_demand,
        "ev_enforce_steps": int(np.sum(ev_enforce_active_ts > 0)),
        "ev_enforce_extra_kwh_total": float(np.sum(ev_enforce_extra_kwh_ts)),
        "actuator_mode": actuator_mode,
        "thesis_steps": "0,1,2,6" if planner_only else "0-6",
    }

    if enable_mpc_window_debug and debug_rows:
        try:
            debug_df = pd.DataFrame(debug_rows)
            if mpc_window_debug_csv_path is None:
                debug_path = (
                    PROJECT_ROOT
                    / "output"
                    / "notebooks"
                    / "online_ev_mpc_debug_notebook_09.csv"
                )
            else:
                debug_path = Path(mpc_window_debug_csv_path)
                if not debug_path.is_absolute():
                    debug_path = PROJECT_ROOT / debug_path
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            debug_df.to_csv(debug_path, index=False)
            if verbose:
                print(f"[DEBUG] Saved detailed MPC window log to: {debug_path}")
        except Exception as e:
            if verbose:
                print(f"[DEBUG] Failed to write MPC debug CSV: {e}")

    if verbose:
        print("=" * 80)
        total_uncharged = sum(max(v, 0.0) for v in uncharged_kwh.values())
        print(f"Simulation finished. Total uncharged EV energy: {total_uncharged:.2f} kWh")
        if enforce_daily_ev_demand:
            n_enforce = int(np.sum(ev_enforce_active_ts > 0))
            extra_kwh = float(np.sum(ev_enforce_extra_kwh_ts))
            print(
                f"  Enforce daily EV demand: {n_enforce} steps, "
                f"{extra_kwh:.2f} kWh restored after clipping"
            )
        print("=" * 80)

    return res, summary


def run_ev_online_mpc_planner_only(
    forecast_strategy_ev: str = "a",
    forecast_strategy_inflex: str = "a",
    forecast_strategy_pv: str = "actual",
    access_power_by_month: Dict[str, float] = None,
    verbose: bool = True,
    log_prefix: str = "[Planner-only MPC]",
) -> Tuple[pd.DataFrame, Dict]:
    """Full-year EV online MPC with thesis §3.7.2 steps 0, 1, 2, 6 only (no actuator layer)."""
    return run_ev_online_mpc_1(
        forecast_strategy_ev=forecast_strategy_ev,
        forecast_strategy_inflex=forecast_strategy_inflex,
        forecast_strategy_pv=forecast_strategy_pv,
        ev_deadline_slack_minutes=0,
        enforce_daily_ev_demand=False,
        access_power_by_month=access_power_by_month,
        verbose=verbose,
        log_prefix=log_prefix,
        enable_mpc_window_debug=False,
        actuator_mode="planner_only",
    )


if __name__ == "__main__":
    raise RuntimeError(
        "CLI entry point is disabled. Call run_ev_online_mpc_1 from a notebook or script "
        "and provide 'access_power_by_month' explicitly."
    )

