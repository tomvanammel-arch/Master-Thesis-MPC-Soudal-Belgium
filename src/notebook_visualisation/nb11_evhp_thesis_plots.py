"""
Thesis-style volume plots for notebook 11 (joint EV+HP), aligned with nb09 EV and nb10 HP Part 4C.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from notebook_visualisation.nb09_ev_part34_viewer import (
    C_ACCESS,
    C_BASELINE,
    C_OFFLINE,
    C_ONLINE,
    C_STRESS,
    LS_CONSTRAINT,
    LS_OFFLINE,
    LS_ONLINE,
    _LAB_EV_BASELINE,
    _LAB_EV_OFFLINE,
    _LAB_EV_ONLINE,
    _LAB_GRID_ACCESS,
    _LAB_GRID_BASELINE,
    _LAB_GRID_OFFLINE,
    _LAB_GRID_ONLINE,
    _ev_window_lines,
    _legend_ordered_below,
    _style_day_xaxis,
    _zoh,
    apply_thesis_rc,
    baseline_ev_kw,
    plot_thesis_day_ev_power,
    plot_thesis_day_spot_price,
)

_LAB_HP_BASELINE = "HP power (baseline)"
_LAB_HP_OFFLINE = "HP power (offline)"
_LAB_HP_ONLINE = "HP power (online)"
_LAB_SOC_BASELINE = "SOC (baseline)"
_LAB_SOC_OFFLINE = "SOC (offline)"
_LAB_SOC_ONLINE = "SOC (online)"


def baseline_grid_kw_nb11(df: pd.DataFrame) -> pd.Series:
    """Baseline grid power (kW): plant1 ``grid_consumption`` + uncontrolled HP."""
    gc_kw = pd.to_numeric(df.get("grid_consumption"), errors="coerce").fillna(0.0) * 4.0
    hp_kw = pd.to_numeric(df.get("hp_baseline_kw"), errors="coerce").fillna(0.0)
    return gc_kw + hp_kw


def enrich_plot_frame(
    df: pd.DataFrame,
    plant_slice: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Add column aliases expected by nb09 grid/EV thesis helpers."""
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    if "ev_applied_kw" in out.columns:
        out["ev_power_kw_online"] = pd.to_numeric(out["ev_applied_kw"], errors="coerce")
    elif "ev_applied" in out.columns:
        out["ev_power_kw_online"] = pd.to_numeric(out["ev_applied"], errors="coerce") * 4.0
    if "p_grid_actual_kw" in out.columns:
        out["grid_power_online"] = pd.to_numeric(out["p_grid_actual_kw"], errors="coerce")
    if plant_slice is not None and not plant_slice.empty:
        pc = plant_slice.copy()
        pc["timestamp"] = pd.to_datetime(pc["timestamp"], errors="coerce")
        cols = ["timestamp"]
        for c in ("grid_consumption", "grid_consumption_excl_ev", "ev"):
            if c in pc.columns:
                cols.append(c)
        out = out.merge(pc[cols], on="timestamp", how="left")
    return out


def _style_week_xaxis(ax, t_start: pd.Timestamp, t_end: pd.Timestamp) -> None:
    ax.set_xlim(t_start, t_end)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    ax.grid(True, which="major", axis="both")
    ax.grid(True, which="minor", axis="x", alpha=0.15)


def _shade_stress(ax, ts: pd.Series, stress_active: pd.Series) -> None:
    sa = pd.to_numeric(stress_active, errors="coerce").fillna(0.0) > 0.5
    if not sa.any():
        return
    sp = sa.to_numpy()
    t = pd.to_datetime(ts, errors="coerce")
    i, first = 0, True
    while i < len(sp):
        if not sp[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(sp) and sp[j + 1]:
            j += 1
        ax.axvspan(
            t.iloc[i],
            t.iloc[j] + pd.Timedelta(minutes=15),
            facecolor=C_STRESS,
            alpha=0.55,
            zorder=0,
            label="Forecast stress" if first else None,
        )
        first = False
        i = j + 1


def plot_thesis_week_ev_power(
    week_df: pd.DataFrame,
    week_start: pd.Timestamp,
    week_end: pd.Timestamp,
    *,
    show_window: bool = True,
) -> plt.Axes:
    """Week: baseline / offline / online EV power (kW), nb09-style."""
    apply_thesis_rc()
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ts = week_df["timestamp"]
    _zoh(ax, ts, baseline_ev_kw(week_df), color=C_BASELINE, linestyle=LS_ONLINE, label=_LAB_EV_BASELINE)
    if "ev_charge_power" in week_df.columns:
        _zoh(
            ax,
            ts,
            week_df["ev_charge_power"],
            color=C_OFFLINE,
            linestyle=LS_OFFLINE,
            label=_LAB_EV_OFFLINE,
        )
    if "ev_power_kw_online" in week_df.columns:
        _zoh(
            ax,
            ts,
            week_df["ev_power_kw_online"],
            color=C_ONLINE,
            linestyle=LS_ONLINE,
            label=_LAB_EV_ONLINE,
        )
    if show_window:
        d = week_start.normalize()
        while d < week_end.normalize():
            if d.weekday() < 5:
                _ev_window_lines(ax, d)
            d += pd.Timedelta(days=1)
    ax.set_ylabel("kW")
    ax.set_xlabel("Time")
    ax.set_title(
        f"EV power — week {week_start.strftime('%Y-%m-%d')} to "
        f"{(week_end - pd.Timedelta(days=1)).strftime('%Y-%m-%d')}"
    )
    _style_week_xaxis(ax, week_start, week_end)
    _legend_ordered_below(fig, ax, [_LAB_EV_BASELINE, _LAB_EV_OFFLINE, _LAB_EV_ONLINE], ncol=3)
    plt.show()
    return ax


def plot_thesis_week_hp_electrical(
    week_df: pd.DataFrame,
    week_det: pd.DataFrame,
    week_start: pd.Timestamp,
    week_end: pd.Timestamp,
) -> plt.Axes:
    """Week: HP baseline (uncontrolled) / offline / online (kW), nb10-style."""
    apply_thesis_rc()
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ts = week_df["timestamp"]
    if "forecast_access_exceedance_active" in week_df.columns:
        _shade_stress(ax, ts, week_df["forecast_access_exceedance_active"])
    if "hp_baseline_kw" in week_df.columns:
        _zoh(
            ax,
            ts,
            week_df["hp_baseline_kw"],
            color=C_BASELINE,
            linestyle=LS_ONLINE,
            label=_LAB_HP_BASELINE,
        )
    if not week_det.empty and "hp_kw_deterministic" in week_det.columns:
        _zoh(
            ax,
            week_det["timestamp"],
            week_det["hp_kw_deterministic"],
            color=C_OFFLINE,
            linestyle=LS_OFFLINE,
            label=_LAB_HP_OFFLINE,
        )
    if "hp_applied_kw" in week_df.columns:
        _zoh(
            ax,
            ts,
            week_df["hp_applied_kw"],
            color=C_ONLINE,
            linestyle=LS_ONLINE,
            label=_LAB_HP_ONLINE,
        )
    ax.set_ylabel("kW")
    ax.set_xlabel("Time")
    ax.set_title(
        f"HP electrical power — week {week_start.strftime('%Y-%m-%d')} to "
        f"{(week_end - pd.Timedelta(days=1)).strftime('%Y-%m-%d')}"
    )
    _style_week_xaxis(ax, week_start, week_end)
    _legend_hp(fig, ax)
    plt.show()
    return ax


def _legend_hp(fig, ax) -> None:
    h, lab = ax.get_legend_handles_labels()
    order = [_LAB_HP_BASELINE, _LAB_HP_OFFLINE, _LAB_HP_ONLINE]
    if "Forecast stress" in lab:
        order = ["Forecast stress"] + order
    _legend_ordered_below(fig, ax, order, ncol=3)


def plot_thesis_week_buffer_soc(
    week_df: pd.DataFrame,
    week_det: pd.DataFrame,
    week_start: pd.Timestamp,
    week_end: pd.Timestamp,
    *,
    soc_min: float = 0.1,
    soc_max: float = 0.9,
) -> plt.Axes:
    """Week: buffer SOC fraction (online vs offline), nb10-style."""
    apply_thesis_rc()
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ts = week_df["timestamp"]
    if "forecast_access_exceedance_active" in week_df.columns:
        _shade_stress(ax, ts, week_df["forecast_access_exceedance_active"])
    if "soc_after" in week_df.columns:
        _zoh(
            ax,
            ts,
            pd.to_numeric(week_df["soc_after"], errors="coerce"),
            color=C_ONLINE,
            linestyle=LS_ONLINE,
            label=_LAB_SOC_ONLINE,
        )
    if not week_det.empty and "buffer_soc" in week_det.columns:
        _zoh(
            ax,
            week_det["timestamp"],
            week_det["buffer_soc"],
            color=C_OFFLINE,
            linestyle=LS_OFFLINE,
            label=_LAB_SOC_OFFLINE,
        )
    ax.axhline(soc_min, color=C_ACCESS, linestyle=LS_CONSTRAINT, linewidth=1.0, alpha=0.8)
    ax.axhline(soc_max, color=C_ACCESS, linestyle=LS_CONSTRAINT, linewidth=1.0, alpha=0.8)
    ax.set_ylabel("SOC")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Time")
    ax.set_title(
        f"Buffer SOC — week {week_start.strftime('%Y-%m-%d')} to "
        f"{(week_end - pd.Timedelta(days=1)).strftime('%Y-%m-%d')}"
    )
    _style_week_xaxis(ax, week_start, week_end)
    _legend_ordered_below(fig, ax, [_LAB_SOC_ONLINE, _LAB_SOC_OFFLINE], ncol=2)
    plt.show()
    return ax


def plot_thesis_week_grid_power(
    week_df: pd.DataFrame,
    week_det: pd.DataFrame,
    week_start: pd.Timestamp,
    week_end: pd.Timestamp,
) -> plt.Axes:
    """Week: grid baseline / online / offline / access (kW), nb09-style."""
    apply_thesis_rc()
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ts = week_df["timestamp"]
    if "forecast_access_exceedance_active" in week_df.columns:
        _shade_stress(ax, ts, week_df["forecast_access_exceedance_active"])
    _zoh(ax, ts, baseline_grid_kw_nb11(week_df), color=C_BASELINE, linestyle=LS_ONLINE, label=_LAB_GRID_BASELINE)
    if "grid_power_online" in week_df.columns:
        online = pd.to_numeric(week_df["grid_power_online"], errors="coerce")
        _zoh(ax, ts, online, color=C_ONLINE, linestyle=LS_ONLINE, label=_LAB_GRID_ONLINE)
        if "access_kw" in week_df.columns:
            access = pd.to_numeric(week_df["access_kw"], errors="coerce")
            viol = (online > access).fillna(False)
            if viol.any():
                ax.fill_between(ts, access, online, where=viol, color=C_ONLINE, alpha=0.15, step="post")
    if not week_det.empty and "grid_power" in week_det.columns:
        _zoh(
            ax,
            week_det["timestamp"],
            week_det["grid_power"],
            color=C_OFFLINE,
            linestyle=LS_OFFLINE,
            label=_LAB_GRID_OFFLINE,
        )
    if "access_kw" in week_df.columns:
        _zoh(
            ax,
            ts,
            week_df["access_kw"],
            color=C_ACCESS,
            linestyle=LS_CONSTRAINT,
            label=_LAB_GRID_ACCESS,
        )
    ax.set_ylabel("kW")
    ax.set_xlabel("Time")
    ax.set_title(
        f"Grid power — week {week_start.strftime('%Y-%m-%d')} to "
        f"{(week_end - pd.Timedelta(days=1)).strftime('%Y-%m-%d')}"
    )
    _style_week_xaxis(ax, week_start, week_end)
    order = [
        "Forecast stress",
        _LAB_GRID_BASELINE,
        _LAB_GRID_ONLINE,
        _LAB_GRID_OFFLINE,
        _LAB_GRID_ACCESS,
    ]
    _legend_ordered_below(fig, ax, order, ncol=3)
    plt.show()
    return ax


def plot_thesis_day_hp_electrical(
    day_df: pd.DataFrame,
    day_det: pd.DataFrame,
    day_start: pd.Timestamp,
    *,
    xlim: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
) -> plt.Axes:
    """Day: HP baseline / offline / online (kW), nb10 Part 4C style."""
    apply_thesis_rc()
    day_title = day_start.strftime("%d/%m/%Y")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ts = day_df["timestamp"]
    if "forecast_access_exceedance_active" in day_df.columns:
        _shade_stress(ax, ts, day_df["forecast_access_exceedance_active"])
    if "hp_baseline_kw" in day_df.columns:
        _zoh(
            ax,
            ts,
            day_df["hp_baseline_kw"],
            color=C_BASELINE,
            linestyle=LS_ONLINE,
            label=_LAB_HP_BASELINE,
        )
    if day_det is not None and not day_det.empty and "hp_kw_deterministic" in day_det.columns:
        _zoh(
            ax,
            day_det["timestamp"],
            day_det["hp_kw_deterministic"],
            color=C_OFFLINE,
            linestyle=LS_OFFLINE,
            label=_LAB_HP_OFFLINE,
        )
    if "hp_applied_kw" in day_df.columns:
        _zoh(
            ax,
            ts,
            day_df["hp_applied_kw"],
            color=C_ONLINE,
            linestyle=LS_ONLINE,
            label=_LAB_HP_ONLINE,
        )
    ax.set_ylabel("kW")
    ax.set_xlabel("Time")
    ax.set_title(f"HP electrical power — {day_title}")
    _style_day_xaxis(ax, day_start, xlim=xlim)
    _legend_hp(fig, ax)
    plt.show()
    return ax


def plot_thesis_day_buffer_soc(
    day_df: pd.DataFrame,
    day_det: pd.DataFrame,
    day_start: pd.Timestamp,
    *,
    soc_min: float = 0.1,
    soc_max: float = 0.9,
    xlim: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
) -> plt.Axes:
    """Day: buffer SOC fraction (online vs offline), nb10 Part 4C style."""
    apply_thesis_rc()
    day_title = day_start.strftime("%d/%m/%Y")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ts = day_df["timestamp"]
    if "forecast_access_exceedance_active" in day_df.columns:
        _shade_stress(ax, ts, day_df["forecast_access_exceedance_active"])
    if "soc_after" in day_df.columns:
        _zoh(
            ax,
            ts,
            pd.to_numeric(day_df["soc_after"], errors="coerce"),
            color=C_ONLINE,
            linestyle=LS_ONLINE,
            label=_LAB_SOC_ONLINE,
        )
    if day_det is not None and not day_det.empty and "buffer_soc" in day_det.columns:
        _zoh(
            ax,
            day_det["timestamp"],
            day_det["buffer_soc"],
            color=C_OFFLINE,
            linestyle=LS_OFFLINE,
            label=_LAB_SOC_OFFLINE,
        )
    ax.axhline(soc_min, color=C_ACCESS, linestyle=LS_CONSTRAINT, linewidth=1.0, alpha=0.8)
    ax.axhline(soc_max, color=C_ACCESS, linestyle=LS_CONSTRAINT, linewidth=1.0, alpha=0.8)
    ax.set_ylabel("SOC")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Time")
    ax.set_title(f"Buffer SOC — {day_title}")
    _style_day_xaxis(ax, day_start, xlim=xlim)
    _legend_ordered_below(fig, ax, [_LAB_SOC_ONLINE, _LAB_SOC_OFFLINE], ncol=2)
    plt.show()
    return ax


def plot_thesis_day_ev_power_joint(
    day_df: pd.DataFrame,
    day_start: pd.Timestamp,
    *,
    show_window: bool = True,
    show_enforce_markers: bool = True,
    xlim: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
) -> plt.Axes:
    """Wrapper around nb09 EV day plot (enforce markers optional)."""
    return plot_thesis_day_ev_power(
        day_df,
        day_start,
        show_window=show_window,
        show_enforce_markers=show_enforce_markers,
        standalone=True,
        xlim=xlim,
    )


def plot_thesis_day_grid_power_joint(
    day_df: pd.DataFrame,
    day_start: pd.Timestamp,
    *,
    xlim: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
) -> plt.Axes:
    """Day: grid baseline (plant consumption + uncontrolled HP) / online / offline / access."""
    apply_thesis_rc()
    day_title = day_start.strftime("%d/%m/%Y")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ts = day_df["timestamp"]
    if "forecast_access_exceedance_active" in day_df.columns:
        _shade_stress(ax, ts, day_df["forecast_access_exceedance_active"])
    _zoh(ax, ts, baseline_grid_kw_nb11(day_df), color=C_BASELINE, linestyle=LS_ONLINE, label=_LAB_GRID_BASELINE)
    if "grid_power_online" in day_df.columns:
        online = pd.to_numeric(day_df["grid_power_online"], errors="coerce")
        _zoh(ax, ts, online, color=C_ONLINE, linestyle=LS_ONLINE, label=_LAB_GRID_ONLINE)
        if "access_kw" in day_df.columns:
            access = pd.to_numeric(day_df["access_kw"], errors="coerce")
            viol = (online > access).fillna(False)
            if viol.any():
                ax.fill_between(ts, access, online, where=viol, color=C_ONLINE, alpha=0.15, step="post")
    if "grid_power" in day_df.columns:
        _zoh(
            ax,
            ts,
            day_df["grid_power"],
            color=C_OFFLINE,
            linestyle=LS_OFFLINE,
            label=_LAB_GRID_OFFLINE,
        )
    if "access_kw" in day_df.columns:
        _zoh(
            ax,
            ts,
            day_df["access_kw"],
            color=C_ACCESS,
            linestyle=LS_CONSTRAINT,
            label=_LAB_GRID_ACCESS,
        )
    ax.set_ylabel("kW")
    ax.set_xlabel("Time")
    ax.set_title(f"Grid power — {day_title}")
    _style_day_xaxis(ax, day_start, xlim=xlim)
    order = [
        "Forecast stress",
        _LAB_GRID_BASELINE,
        _LAB_GRID_ONLINE,
        _LAB_GRID_OFFLINE,
        _LAB_GRID_ACCESS,
    ]
    _legend_ordered_below(fig, ax, order, ncol=2)
    plt.show()
    return ax


def plot_thesis_volume_suite_week(
    week_df: pd.DataFrame,
    week_det: pd.DataFrame,
    week_start: pd.Timestamp,
    week_end: pd.Timestamp,
) -> None:
    """All four thesis week figures for Part 4C."""
    plot_thesis_week_ev_power(week_df, week_start, week_end)
    plot_thesis_week_hp_electrical(week_df, week_det, week_start, week_end)
    plot_thesis_week_buffer_soc(week_df, week_det, week_start, week_end)
    plot_thesis_week_grid_power(week_df, week_det, week_start, week_end)


def plot_thesis_volume_suite_day(
    day_df: pd.DataFrame,
    day_det: pd.DataFrame,
    day_start: pd.Timestamp,
    *,
    soc_min: float = 0.1,
    soc_max: float = 0.9,
    show_enforce_markers: bool = True,
    xlim: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
) -> None:
    """All five thesis day figures for Part 4C (spot price below buffer SOC)."""
    plot_thesis_day_ev_power_joint(
        day_df,
        day_start,
        show_enforce_markers=show_enforce_markers,
        xlim=xlim,
    )
    plot_thesis_day_hp_electrical(day_df, day_det, day_start, xlim=xlim)
    plot_thesis_day_buffer_soc(
        day_df, day_det, day_start, soc_min=soc_min, soc_max=soc_max, xlim=xlim
    )
    _spot_df = day_df
    if "price" not in _spot_df.columns and "spot_price" in _spot_df.columns:
        _spot_df = day_df.copy()
        _spot_df["price"] = _spot_df["spot_price"]
    plot_thesis_day_spot_price(_spot_df, day_start, xlim=xlim)
    plot_thesis_day_grid_power_joint(day_df, day_start, xlim=xlim)
