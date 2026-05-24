"""
Thesis §3.7.1 — access power before online operation (Equations (3.43)–(3.45)).

Headroom over the EV charging window on historical weekdays; utilisation
u = E_ref / H_d(P); pick the lowest candidate P such that P95_d(u) <= u_head,max.
Then add a fixed kW margin and apply a 12-month lock-in after increases (§3.5.1).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Defaults for Equation (3.45) discrete search (optional / legacy §3.7.1 path).
# Notebook 09 §1.2 uses inline cum-max + MARGIN_KW; these apply if this module is called.
U_HEAD_MAX_DEFAULT: float = 0.95
P_ACCESS_CANDIDATE_KW_MIN: float = 2000.0
P_ACCESS_CANDIDATE_KW_MAX: float = 3300.0
P_ACCESS_CANDIDATE_STEP_KW: float = 10.0
ACCESS_POWER_MARGIN_KW: float = 20.0
U_UTIL_CAP_DEFAULT: float = 0.70


def _ev_weekday_window_mask(ts: pd.Series) -> np.ndarray:
    """Weekdays 07:00–17:00 local (naive timestamps), same as online EV MPC."""
    t = pd.to_datetime(ts, errors="coerce")
    dow = t.dt.dayofweek.to_numpy()
    tod = (t.dt.hour + t.dt.minute / 60.0).to_numpy()
    return (dow < 5) & (tod >= 7.0) & (tod < 17.0)


def max_inflex_power_kw_ev_window(
    inflex_kwh_15: np.ndarray,
    timestamps: pd.Series,
) -> float:
    """
    Maximum P_inflex,t = 4*E_inflex,t over weekdays 07:00–17:00 (kW).

    Contracted access must not be below the largest inflexible power seen in the
    same reference window, otherwise the site would be in permanent exceedance
    on inflexible load alone.
    """
    win = _ev_weekday_window_mask(timestamps)
    if not np.any(win):
        return 0.0
    p_inflex_kw = 4.0 * np.asarray(inflex_kwh_15, dtype=float)
    return float(np.max(p_inflex_kw[win]))


def daily_headroom_kwh_for_day(
    p_access_kw: float,
    inflex_kwh_15: np.ndarray,
    mask_day: np.ndarray,
    dt_h: float = 0.25,
) -> float:
    """
    H_d(P_access) = sum_t max(P_access - P_inflex,t, 0) * Δt  [kWh/day]
    P_inflex,t = 4 * E_inflex,t  (kW from kWh/15min).
    """
    p_inflex_kw = 4.0 * inflex_kwh_15
    head_kw = np.maximum(p_access_kw - p_inflex_kw, 0.0)
    return float(np.sum(head_kw[mask_day] * dt_h))


def select_lowest_access_for_reference_days(
    inflex_kwh_15: np.ndarray,
    timestamps: pd.Series,
    day_ids: np.ndarray,
    e_ref_kwh_per_day: float,
    u_head_max: float = U_HEAD_MAX_DEFAULT,
    p_candidates_kw: Optional[np.ndarray] = None,
    p_floor_kw: float = 0.0,
) -> float:
    """
    Lowest P_access (kW) such that P95 over reference days of u_d <= u_head_max.
    Candidates are restricted to P >= p_floor_kw (e.g. max historical inflex in EV window).

    If no candidate passes, return the maximum candidate (caller should validate).
    """
    step = float(P_ACCESS_CANDIDATE_STEP_KW)
    p_lo = max(float(P_ACCESS_CANDIDATE_KW_MIN), float(np.ceil(p_floor_kw / step) * step))
    p_hi = float(P_ACCESS_CANDIDATE_KW_MAX) + 0.5 * step

    if p_candidates_kw is None:
        p_candidates_kw = np.arange(p_lo, p_hi, step, dtype=float)
    else:
        p_candidates_kw = np.asarray(p_candidates_kw, dtype=float)
        p_candidates_kw = p_candidates_kw[p_candidates_kw + 1e-9 >= p_lo]
        if len(p_candidates_kw) == 0:
            p_candidates_kw = np.array([p_lo], dtype=float)

    win = _ev_weekday_window_mask(timestamps)
    unique_days = np.unique(day_ids[win])
    if len(unique_days) == 0:
        raise ValueError("No EV-window timesteps in reference data for headroom selection.")

    best_p: Optional[float] = None
    for p_kw in p_candidates_kw:
        utils: List[float] = []
        for d in unique_days:
            mday = (day_ids == d) & win
            if not np.any(mday):
                continue
            h_kwh = daily_headroom_kwh_for_day(float(p_kw), inflex_kwh_15, mday)
            if h_kwh <= 1e-9:
                utils.append(float("inf"))
            else:
                utils.append(float(e_ref_kwh_per_day / h_kwh))
        if not utils:
            continue
        uarr = np.asarray(utils, dtype=float)
        uarr = uarr[np.isfinite(uarr)]
        if len(uarr) == 0:
            continue
        p95 = float(np.percentile(uarr, 95, method="linear"))
        if p95 <= u_head_max + 1e-9:
            best_p = float(p_kw)
            break
    if best_p is None:
        return float(p_candidates_kw[-1])
    return best_p


def daily_ev_kwh_weekday_window_series(
    ev_kwh_15: np.ndarray,
    timestamps: pd.Series,
    day_ids: np.ndarray,
) -> pd.Series:
    """Daily EV energy (kWh) summed over weekdays 07:00–17:00; index = day (normalized)."""
    win = _ev_weekday_window_mask(timestamps)
    df = pd.DataFrame({"day": day_ids, "ev": ev_kwh_15, "w": win.astype(int)})
    daily = df.loc[df["w"] == 1].groupby("day")["ev"].sum()
    return daily.astype(float)


def max_daily_ev_kwh_ev_window(
    ev_kwh_15: np.ndarray,
    timestamps: pd.Series,
    day_ids: np.ndarray,
) -> float:
    """Maximum daily EV energy (kWh) on EV-window weekdays in the given series."""
    daily = daily_ev_kwh_weekday_window_series(ev_kwh_15, timestamps, day_ids)
    if daily.empty:
        return 0.0
    return float(daily.max())


def compute_monthly_e_ref_max_daily_ev_so_far(
    train_2024_df: pd.DataFrame,
    plant_df: pd.DataFrame,
    month_periods: pd.PeriodIndex,
    *,
    ev_col: str = "ev",
    timestamp_col: str = "timestamp",
) -> pd.Series:
    """
    For each month M: E_ref = max daily EV energy (kWh, weekday 07:00–17:00)
    over all days strictly before the start of M (2024 training + 2025 actuals).
    """
    frames: List[pd.DataFrame] = []
    for df in (train_2024_df, plant_df):
        if df is None or len(df) == 0:
            continue
        sub = df[[timestamp_col, ev_col]].copy()
        frames.append(sub)
    if not frames:
        return pd.Series(0.0, index=month_periods, dtype=float)

    combined = pd.concat(frames, ignore_index=True)
    ts = pd.to_datetime(combined[timestamp_col], utc=True, errors="coerce")
    ts = ts.dt.tz_convert("Europe/Brussels").dt.tz_localize(None)
    ev = pd.to_numeric(combined[ev_col], errors="coerce").fillna(0.0).to_numpy()
    day_ids = ts.dt.normalize().to_numpy()
    daily = daily_ev_kwh_weekday_window_series(ev, ts, day_ids)
    if daily.empty:
        return pd.Series(0.0, index=month_periods, dtype=float)

    daily.index = pd.to_datetime(daily.index)
    out: List[float] = []
    for m in month_periods:
        cutoff = pd.Timestamp(m.start_time)
        past = daily[daily.index < cutoff]
        out.append(float(past.max()) if len(past) else 0.0)
    return pd.Series(out, index=month_periods, dtype=float)


def p95_headroom_utilisation(
    inflex_kwh_15: np.ndarray,
    timestamps: pd.Series,
    day_ids: np.ndarray,
    e_ref_kwh_per_day: float,
    p_access_kw: float,
) -> float:
    """P95 over EV-window weekdays of u_d = E_ref / H_d(P_access) (Eq. 3.44–3.45)."""
    win = _ev_weekday_window_mask(timestamps)
    unique_days = np.unique(day_ids[win])
    utils: List[float] = []
    for d in unique_days:
        mday = (day_ids == d) & win
        if not np.any(mday):
            continue
        h_kwh = daily_headroom_kwh_for_day(float(p_access_kw), inflex_kwh_15, mday)
        if h_kwh <= 1e-9:
            utils.append(float("inf"))
        else:
            utils.append(float(e_ref_kwh_per_day / h_kwh))
    if not utils:
        return float("nan")
    uarr = np.asarray(utils, dtype=float)
    uarr = uarr[np.isfinite(uarr)]
    if len(uarr) == 0:
        return float("nan")
    return float(np.percentile(uarr, 95, method="linear"))


def monthly_headroom_utilisation_p95_pct_series(
    train_2024_df: pd.DataFrame,
    month_periods: pd.PeriodIndex,
    access_kw_by_month: pd.Series,
    *,
    e_ref_by_month: Optional[pd.Series] = None,
    inflex_col: str = "inflex_load",
    ev_col: str = "ev",
) -> pd.Series:
    """P95 headroom utilisation (%) at the given monthly access levels (kW)."""
    ts4 = pd.to_datetime(train_2024_df["timestamp"], utc=True, errors="coerce")
    ts4 = ts4.dt.tz_convert("Europe/Brussels").dt.tz_localize(None)
    inf4 = pd.to_numeric(train_2024_df[inflex_col], errors="coerce").fillna(0.0).to_numpy()
    ev4 = pd.to_numeric(train_2024_df[ev_col], errors="coerce").fillna(0.0).to_numpy()
    day4 = ts4.dt.normalize().to_numpy()

    out: List[float] = []
    for m in month_periods:
        p_kw = float(access_kw_by_month.reindex([m]).iloc[0])
        month_num = int(m.month)
        bi = (ts4.dt.month == month_num).to_numpy(dtype=bool)
        if not np.any(bi):
            ref_ts = ts4
            ref_inflex = inf4
            ref_ev = ev4
            ref_day = day4
        else:
            ref_ts = ts4[bi].reset_index(drop=True)
            ref_inflex = inf4[bi]
            ref_ev = ev4[bi]
            ref_day = day4[bi]

        if e_ref_by_month is not None:
            e_ref = float(e_ref_by_month.reindex([m]).iloc[0])
        else:
            e_ref = max_daily_ev_kwh_ev_window(ref_ev, ref_ts, ref_day)
        if e_ref <= 1e-6:
            out.append(float("nan"))
            continue
        u95 = p95_headroom_utilisation(ref_inflex, ref_ts, ref_day, e_ref, p_kw)
        out.append(float(u95 * 100.0) if np.isfinite(u95) else float("nan"))

    return pd.Series(out, index=month_periods, dtype=float)


def apply_twelve_month_lock_in(raw_monthly_kw: List[float]) -> List[float]:
    """
    After any strict increase P[m] > P[m-1], access cannot drop below P[m]
    for the next 11 months (12-month window including month m), §3.5.1 style.
    First month has no prior contractual level (no lock from m=-1).
    """
    out: List[float] = []
    locks: List[Tuple[int, float]] = []  # (start_month_index, level)
    for m, raw in enumerate(raw_monthly_kw):
        floor = 0.0
        for sm, lev in locks:
            if sm <= m < sm + 12:
                floor = max(floor, lev)
        p = max(float(raw), floor)
        out.append(p)
        if m > 0 and p > out[m - 1] + 1e-6:
            locks.append((m, p))
    return out


def monthly_access_power_series_thesis(
    train_2024_df: pd.DataFrame,
    month_periods: pd.PeriodIndex,
    *,
    e_ref_by_month: Optional[pd.Series] = None,
    inflex_col: str = "inflex_load",
    ev_col: str = "ev",
    u_head_max: float = U_HEAD_MAX_DEFAULT,
    margin_kw: float = ACCESS_POWER_MARGIN_KW,
    p_candidates_kw: Optional[np.ndarray] = None,
) -> pd.Series:
    """
    For each calendar month, select access using 2024 same-month weekdays as
    reference inflexible load. E_ref defaults to max daily EV (kWh) in that slice;
    pass ``e_ref_by_month`` for max-daily-EV-so-far (e.g. from 2024 + 2025 actuals).

    Returns PeriodIndex -> kW (after +margin_kw and 12-month lock-in).
    """
    # --- 2024 reference: timestamps local naive ---
    ts4 = pd.to_datetime(train_2024_df["timestamp"], utc=True, errors="coerce")
    ts4 = ts4.dt.tz_convert("Europe/Brussels").dt.tz_localize(None)
    inf4 = pd.to_numeric(train_2024_df[inflex_col], errors="coerce").fillna(0.0).to_numpy()
    ev4 = pd.to_numeric(train_2024_df[ev_col], errors="coerce").fillna(0.0).to_numpy()
    day4 = ts4.dt.normalize().to_numpy()

    raw: List[float] = []
    for m in month_periods:
        month_num = int(m.month)
        bi = (ts4.dt.month == month_num).to_numpy(dtype=bool)
        if not np.any(bi):
            ref_ts = ts4
            ref_inflex = inf4
            ref_ev = ev4
            ref_day = day4
        else:
            ref_ts = ts4[bi].reset_index(drop=True)
            ref_inflex = inf4[bi]
            ref_ev = ev4[bi]
            ref_day = day4[bi]

        win = _ev_weekday_window_mask(ref_ts)
        if not np.any(win):
            raw.append(float(P_ACCESS_CANDIDATE_KW_MAX))
            continue

        if e_ref_by_month is not None:
            e_ref = float(e_ref_by_month.reindex([m]).iloc[0])
        else:
            e_ref = max_daily_ev_kwh_ev_window(ref_ev, ref_ts, ref_day)
        inflex_peak_kw = max_inflex_power_kw_ev_window(ref_inflex, ref_ts)
        p_head = select_lowest_access_for_reference_days(
            ref_inflex,
            ref_ts,
            ref_day,
            e_ref_kwh_per_day=e_ref,
            u_head_max=u_head_max,
            p_candidates_kw=p_candidates_kw,
            p_floor_kw=inflex_peak_kw,
        )
        # Thesis headroom + margin, but never below (max inflex in EV window) + margin:
        # otherwise contracted access could sit below the site's own inflexible peak.
        raw.append(float(max(p_head + margin_kw, inflex_peak_kw + margin_kw)))

    locked = apply_twelve_month_lock_in(raw)
    return pd.Series(locked, index=month_periods, dtype=float)


def month_period_index_to_str_keys(series: pd.Series) -> dict:
    """{'2025-01': float, ...} for run_ev_online_mpc_1."""
    return {p.strftime("%Y-%m"): float(v) for p, v in series.items()}


def max_inflex_power_kw_full_day(
    inflex_kwh_15: np.ndarray,
) -> float:
    """Maximum P_inflex = 4*E_inflex over all 15-min steps (kW)."""
    if len(inflex_kwh_15) == 0:
        return 0.0
    return float(np.max(4.0 * np.asarray(inflex_kwh_15, dtype=float)))


def daily_headroom_kwh_full_day(
    p_access_kw: float,
    inflex_kwh_15: np.ndarray,
    mask_day: np.ndarray,
    dt_h: float = 0.25,
) -> float:
    """H_d(P) over full calendar day W_joint (all intervals in mask_day)."""
    return daily_headroom_kwh_for_day(p_access_kw, inflex_kwh_15, mask_day, dt_h=dt_h)


def select_lowest_access_for_reference_days_full_day(
    inflex_kwh_15: np.ndarray,
    timestamps: pd.Series,
    day_ids: np.ndarray,
    e_ref_kwh_per_day: float,
    u_head_max: float = U_HEAD_MAX_DEFAULT,
    p_candidates_kw: Optional[np.ndarray] = None,
    p_floor_kw: float = 0.0,
) -> float:
    """
    Lowest P_access (kW) such that P95 over reference calendar days of
    u_d = E_ref / H_d(P) <= u_head_max, using full-day headroom (thesis §3.7.4).
    """
    step = float(P_ACCESS_CANDIDATE_STEP_KW)
    p_lo = max(float(P_ACCESS_CANDIDATE_KW_MIN), float(np.ceil(p_floor_kw / step) * step))
    p_hi = float(P_ACCESS_CANDIDATE_KW_MAX) + 0.5 * step

    if p_candidates_kw is None:
        p_candidates_kw = np.arange(p_lo, p_hi, step, dtype=float)
    else:
        p_candidates_kw = np.asarray(p_candidates_kw, dtype=float)
        p_candidates_kw = p_candidates_kw[p_candidates_kw + 1e-9 >= p_lo]
        if len(p_candidates_kw) == 0:
            p_candidates_kw = np.array([p_lo], dtype=float)

    unique_days = np.unique(day_ids)
    if len(unique_days) == 0:
        raise ValueError("No timesteps in reference data for joint headroom selection.")

    best_p: Optional[float] = None
    for p_kw in p_candidates_kw:
        utils: List[float] = []
        for d in unique_days:
            mday = day_ids == d
            if not np.any(mday):
                continue
            h_kwh = daily_headroom_kwh_full_day(float(p_kw), inflex_kwh_15, mday)
            if h_kwh <= 1e-9:
                utils.append(float("inf"))
            else:
                utils.append(float(e_ref_kwh_per_day / h_kwh))
        if not utils:
            continue
        uarr = np.asarray(utils, dtype=float)
        uarr = uarr[np.isfinite(uarr)]
        if len(uarr) == 0:
            continue
        p95 = float(np.percentile(uarr, 95, method="linear"))
        if p95 <= u_head_max + 1e-9:
            best_p = float(p_kw)
            break
    if best_p is None:
        return float(p_candidates_kw[-1])
    return best_p


def p95_headroom_utilisation_full_day(
    inflex_kwh_15: np.ndarray,
    timestamps: pd.Series,
    day_ids: np.ndarray,
    e_ref_kwh_per_day: float,
    p_access_kw: float,
) -> float:
    """P95 over calendar days of u_d = E_ref / H_d(P) using full-day headroom."""
    unique_days = np.unique(day_ids)
    utils: List[float] = []
    for d in unique_days:
        mday = day_ids == d
        if not np.any(mday):
            continue
        h_kwh = daily_headroom_kwh_full_day(float(p_access_kw), inflex_kwh_15, mday)
        if h_kwh <= 1e-9:
            utils.append(float("inf"))
        else:
            utils.append(float(e_ref_kwh_per_day / h_kwh))
    if not utils:
        return float("nan")
    uarr = np.asarray(utils, dtype=float)
    uarr = uarr[np.isfinite(uarr)]
    if len(uarr) == 0:
        return float("nan")
    return float(np.percentile(uarr, 95, method="linear"))


def monthly_access_power_series_joint(
    train_2024_df: pd.DataFrame,
    month_periods: pd.PeriodIndex,
    *,
    e_ev_ref_by_month: pd.Series,
    e_hp_ref_kwh_per_day: float,
    inflex_col: str = "inflex_load",
    u_head_max: float = U_HEAD_MAX_DEFAULT,
    margin_kw: float = ACCESS_POWER_MARGIN_KW,
    p_candidates_kw: Optional[np.ndarray] = None,
) -> pd.Series:
    """
    Thesis §3.7.4 Step 0: monthly access from full-day headroom with
    E_joint,ref = E_ev,ref + E_hp,ref (kWh/day).

    Uses 2024 same-calendar-month slices as reference inflexible load.
    """
    ts4 = pd.to_datetime(train_2024_df["timestamp"], utc=True, errors="coerce")
    ts4 = ts4.dt.tz_convert("Europe/Brussels").dt.tz_localize(None)
    inf4 = pd.to_numeric(train_2024_df[inflex_col], errors="coerce").fillna(0.0).to_numpy()
    day4 = ts4.dt.normalize().to_numpy()

    e_hp = float(e_hp_ref_kwh_per_day)
    raw: List[float] = []
    for m in month_periods:
        month_num = int(m.month)
        bi = (ts4.dt.month == month_num).to_numpy(dtype=bool)
        if not np.any(bi):
            ref_inflex = inf4
            ref_day = day4
        else:
            ref_inflex = inf4[bi]
            ref_day = day4[bi]

        e_ev = float(e_ev_ref_by_month.reindex([m]).iloc[0])
        e_joint = e_ev + e_hp
        inflex_peak_kw = max_inflex_power_kw_full_day(ref_inflex)
        p_head = select_lowest_access_for_reference_days_full_day(
            ref_inflex,
            ts4[bi] if np.any(bi) else ts4,
            ref_day,
            e_ref_kwh_per_day=e_joint,
            u_head_max=u_head_max,
            p_candidates_kw=p_candidates_kw,
            p_floor_kw=inflex_peak_kw,
        )
        raw.append(float(max(p_head + margin_kw, inflex_peak_kw + margin_kw)))

    locked = apply_twelve_month_lock_in(raw)
    return pd.Series(locked, index=month_periods, dtype=float)


def _prepare_hr_dataframe(
    train_2024_df: pd.DataFrame,
    plant_2025_df: pd.DataFrame,
    *,
    grid_col: str = "grid_consumption",
    ev_col: str = "ev",
    thermal_col: str = "thermal_load",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Concatenate 2024 training + 2025 plant rows with naive local timestamps."""
    frames: List[pd.DataFrame] = []
    for df in (train_2024_df, plant_2025_df):
        if df is None or len(df) == 0:
            continue
        part = df.copy()
        ts = pd.to_datetime(part[timestamp_col], utc=True, errors="coerce")
        part[timestamp_col] = ts.dt.tz_convert("Europe/Brussels").dt.tz_localize(None)
        frames.append(part)
    if not frames:
        raise ValueError("No plant data for headroom access selection.")
    out = pd.concat(frames, ignore_index=True).sort_values(timestamp_col)
    for col in (grid_col, ev_col, thermal_col):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        else:
            out[col] = 0.0
    out["month"] = out[timestamp_col].dt.to_period("M")
    return out


def _daily_sum_kwh_by_day(
    kwh_15: np.ndarray,
    day_ids: np.ndarray,
    interval_mask: Optional[np.ndarray] = None,
) -> pd.Series:
    days = np.asarray(day_ids)
    v = np.asarray(kwh_15, dtype=float)
    if interval_mask is not None:
        m = np.asarray(interval_mask, dtype=bool)
        days = days[m]
        v = v[m]
    return pd.DataFrame({"day": days, "v": v}).groupby("day")["v"].sum()


def max_daily_flex_headroom_utilisation(
    grid_kwh_15: np.ndarray,
    ev_kwh_15: np.ndarray,
    day_ids: np.ndarray,
    p_access_kw: float,
    hp_add_kwh_per_day: float = 0.0,
    interval_mask: Optional[np.ndarray] = None,
) -> float:
    """
    Max daily (EV + HP_add) / H_flex with H from constant P_access and grid load.

    hp_add_kwh_per_day is a scalar added to each calendar day (worst-case HP reference).
    """
    p_acc = float(p_access_kw)
    grid = np.asarray(grid_kwh_15, dtype=float)
    headroom_kw = np.maximum(p_acc - 4.0 * grid, 0.0)
    h_kwh_15 = headroom_kw * 0.25
    h_daily = _daily_sum_kwh_by_day(h_kwh_15, day_ids, interval_mask)
    ev_daily = _daily_sum_kwh_by_day(ev_kwh_15, day_ids, interval_mask)
    need = ev_daily + float(hp_add_kwh_per_day)
    util = need / h_daily.replace(0.0, np.nan)
    util = util[np.isfinite(util) & (h_daily.values > 1e-9)]
    if len(util) == 0:
        return float("inf")
    return float(util.max())


def select_lowest_access_flex_util_cap(
    grid_kwh_15: np.ndarray,
    ev_kwh_15: np.ndarray,
    day_ids: np.ndarray,
    *,
    hp_add_kwh_per_day: float = 0.0,
    util_cap: float = U_UTIL_CAP_DEFAULT,
    step_kw: float = P_ACCESS_CANDIDATE_STEP_KW,
    p_candidates_kw: Optional[np.ndarray] = None,
    interval_mask: Optional[np.ndarray] = None,
) -> float:
    """Lowest P_access (10 kW steps) with max daily flex utilisation <= util_cap."""
    grid = np.asarray(grid_kwh_15, dtype=float)
    p_floor = float(np.max(4.0 * grid)) if len(grid) else 0.0
    step = float(step_kw)
    p_lo = max(float(P_ACCESS_CANDIDATE_KW_MIN), float(np.ceil(p_floor / step) * step))
    p_hi = float(P_ACCESS_CANDIDATE_KW_MAX) + 0.5 * step

    if p_candidates_kw is None:
        p_candidates_kw = np.arange(p_lo, p_hi, step, dtype=float)
    else:
        p_candidates_kw = np.asarray(p_candidates_kw, dtype=float)
        p_candidates_kw = p_candidates_kw[p_candidates_kw + 1e-9 >= p_lo]
        if len(p_candidates_kw) == 0:
            p_candidates_kw = np.array([p_lo], dtype=float)

    for p_kw in p_candidates_kw:
        u_max = max_daily_flex_headroom_utilisation(
            grid,
            ev_kwh_15,
            day_ids,
            float(p_kw),
            hp_add_kwh_per_day=hp_add_kwh_per_day,
            interval_mask=interval_mask,
        )
        if u_max <= float(util_cap) + 1e-9:
            return float(p_kw)
    return float(p_candidates_kw[-1])


def monthly_access_power_flex_hybrid(
    train_2024_df: pd.DataFrame,
    plant_2025_df: pd.DataFrame,
    month_periods: pd.PeriodIndex,
    *,
    monthly_peak_excl_ev_kw: pd.Series,
    baseline_2024_peak_excl_ev_kw: float,
    hp_worst_day_kwh: float,
    margin_kw: float = ACCESS_POWER_MARGIN_KW,
    util_cap: float = U_UTIL_CAP_DEFAULT,
    step_kw: float = P_ACCESS_CANDIDATE_STEP_KW,
    grid_col: str = "grid_consumption",
    ev_col: str = "ev",
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Flex-aware access = max(rule_cummax, rule_70pct_cummax) per month.

    Rule 1: cum-max(grid_excl_ev, M-1) + margin (Jan 2025 seeded from 2024).

    Rule 2: for contract month M, search lowest P (step_kw) so that reference
    calendar month M-1 has max daily utilisation (EV + hp_worst_day) / H_flex <= util_cap;
    then cum-max lock-in of those P over months seen so far.
    """
    hr = _prepare_hr_dataframe(train_2024_df, plant_2025_df)

    cummax_Mm1 = monthly_peak_excl_ev_kw.reindex(month_periods).astype(float).cummax().shift(1)
    cummax_Mm1.loc[month_periods.min()] = float(baseline_2024_peak_excl_ev_kw)
    cummax_Mm1 = cummax_Mm1.fillna(float(baseline_2024_peak_excl_ev_kw))
    access_cummax_kw = cummax_Mm1 + float(margin_kw)

    p70_raw: List[float] = []
    p70_cummax: List[float] = []
    p70_running_max = 0.0

    for m in month_periods:
        ref_month = m - 1
        ref = hr.loc[hr["month"] == ref_month]
        if ref.empty:
            p_sel = float(P_ACCESS_CANDIDATE_KW_MAX)
        else:
            day_ids = ref["timestamp"].dt.normalize().to_numpy()
            p_sel = select_lowest_access_flex_util_cap(
                ref[grid_col].to_numpy(dtype=float),
                ref[ev_col].to_numpy(dtype=float),
                day_ids,
                hp_add_kwh_per_day=float(hp_worst_day_kwh),
                util_cap=util_cap,
                step_kw=step_kw,
            )
        p70_raw.append(p_sel)
        p70_running_max = max(p70_running_max, p_sel)
        p70_cummax.append(p70_running_max)

    access_70_cummax_kw = pd.Series(p70_cummax, index=month_periods, dtype=float)
    access_flex_kw = np.maximum(access_cummax_kw.values, access_70_cummax_kw.values)

    breakdown = pd.DataFrame(
        {
            "access_power_flex_cummax": access_cummax_kw.values,
            "access_power_flex_70pct_raw": p70_raw,
            "access_power_flex_70pct_cummax": access_70_cummax_kw.values,
            "access_power_flex_aware": access_flex_kw,
        },
        index=month_periods,
    )
    return (
        pd.Series(access_flex_kw, index=month_periods, dtype=float),
        breakdown,
    )


def monthly_joint_e_ref_series(
    e_ev_ref_by_month: pd.Series,
    e_hp_ref_kwh_per_day: float,
) -> pd.Series:
    """E_joint,ref (kWh/day) per month = E_ev,ref + E_hp,ref."""
    return e_ev_ref_by_month.astype(float) + float(e_hp_ref_kwh_per_day)
