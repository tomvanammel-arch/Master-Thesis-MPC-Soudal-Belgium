"""
Notebook 09 Part 4.3 thesis-style day plots (EV power, grid power, spot price).
Aligned with nb10 §4C day figures in nb10_hp_part34_viewer.py.
"""
from __future__ import annotations

from typing import Any

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THESIS_STYLE: dict[str, Any] = {
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "lines.linewidth": 2.2,
    "savefig.dpi": 300,
}

C_ONLINE = "#b30000"
C_OFFLINE = "#000000"
C_BLACK = "#000000"
C_BASELINE = "#2166ac"
C_ACCESS = "#444444"
C_STRESS = "#c8d8e8"

LS_ONLINE = "-"
LS_OFFLINE = "--"
LS_CONSTRAINT = ":"

LW = 2.2

_LAB_EV_BASELINE = "EV power (baseline)"
_LAB_EV_OFFLINE = "EV power (offline)"
_LAB_EV_ONLINE = "EV power (online)"

_LAB_GRID_BASELINE = "Grid power (baseline)"
_LAB_GRID_ONLINE = "Grid power (online)"
_LAB_GRID_OFFLINE = "Grid power (offline)"
_LAB_GRID_ACCESS = "Access power (online)"

_LAB_SPOT = "Spot price (€/MWh)"


def apply_thesis_rc() -> None:
    mpl.rcParams.update(THESIS_STYLE)


def _zoh(ax, ts, y, **kwargs) -> None:
    plot_kw: dict[str, Any] = {"linewidth": LW, "drawstyle": "steps-post"}
    plot_kw.update(kwargs)
    ax.plot(
        pd.to_datetime(ts, errors="coerce"),
        pd.to_numeric(y, errors="coerce"),
        **plot_kw,
    )


def _style_day_xaxis(ax, day_start: pd.Timestamp, *, xlim: tuple[pd.Timestamp, pd.Timestamp] | None = None) -> None:
    day_end = day_start + pd.Timedelta(days=1)
    ax.set_xlim(day_start if xlim is None else xlim[0], day_end if xlim is None else xlim[1])
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    ax.grid(True, which="major", axis="both")
    ax.grid(True, which="minor", axis="x", alpha=0.15)


def _ev_window_lines(ax, day_start: pd.Timestamp) -> None:
    """07:00 / 17:00 markers without legend entries."""
    start_line = day_start.replace(hour=7, minute=0, second=0, microsecond=0)
    end_line = day_start.replace(hour=17, minute=0, second=0, microsecond=0)
    ax.axvline(start_line, color=C_ONLINE, linestyle="--", linewidth=1.2, alpha=0.65)
    ax.axvline(end_line, color=C_ONLINE, linestyle="--", linewidth=1.2, alpha=0.65)


def _day_view_xlim(day_start: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Shared thesis day window (06:00–18:00) for EV and grid figures."""
    return (
        day_start.replace(hour=6, minute=0, second=0, microsecond=0),
        day_start.replace(hour=18, minute=0, second=0, microsecond=0),
    )


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


def _legend_ordered_below(
    fig,
    ax,
    order: list[str],
    *,
    ncol: int = 2,
) -> None:
    h, lab = ax.get_legend_handles_labels()
    oh, ol = [], []
    for name in order:
        if name in lab:
            i = lab.index(name)
            oh.append(h[i])
            ol.append(lab[i])
    if not oh:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=ncol, frameon=False)
    else:
        ax.legend(oh, ol, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=ncol, frameon=False)
    fig.subplots_adjust(bottom=0.28, top=0.90)


def baseline_ev_kw(day_df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(day_df["ev_demand_actual"], errors="coerce").fillna(0.0) * 4.0


def baseline_grid_kw(day_df: pd.DataFrame) -> pd.Series:
    excl = pd.to_numeric(day_df.get("grid_consumption_excl_ev"), errors="coerce").fillna(0.0)
    ev = pd.to_numeric(day_df.get("ev_demand_actual"), errors="coerce").fillna(0.0)
    return (excl + ev) * 4.0


def spot_price_eur_per_mwh(day_df: pd.DataFrame) -> pd.Series:
    """Day-ahead spot price (€/MWh) from ``price``, ``spot_price_eur_per_mwh``, or ``spot_price``."""
    for col in ("spot_price_eur_per_mwh", "price", "spot_price"):
        if col in day_df.columns:
            return pd.to_numeric(day_df[col], errors="coerce")
    raise KeyError(
        "Day frame missing spot price column (expected one of: "
        "spot_price_eur_per_mwh, price, spot_price)"
    )


def _enforce_markers(ax, day_df: pd.DataFrame) -> None:
    if "ev_enforce_active" in day_df.columns:
        _enf = day_df[pd.to_numeric(day_df["ev_enforce_active"], errors="coerce").fillna(0.0) > 0.5]
        if len(_enf):
            ax.scatter(
                _enf["timestamp"],
                pd.to_numeric(_enf["ev_power_kw_online"], errors="coerce"),
                s=36,
                c=C_ONLINE,
                edgecolors="white",
                linewidths=0.4,
                zorder=6,
                label="Enforce active",
            )
    if "ev_enforce_deferred" in day_df.columns:
        _def = day_df[pd.to_numeric(day_df["ev_enforce_deferred"], errors="coerce").fillna(0.0) > 0.5]
        if len(_def):
            ax.scatter(
                _def["timestamp"],
                pd.to_numeric(_def["ev_power_kw_online"], errors="coerce"),
                s=36,
                c="#c4a000",
                edgecolors="white",
                linewidths=0.4,
                zorder=6,
                label="Enforce deferred",
            )


def plot_thesis_day_ev_power(
    day_df: pd.DataFrame,
    day_start: pd.Timestamp,
    *,
    ax=None,
    show_window: bool = True,
    show_enforce_markers: bool = False,
    standalone: bool = True,
    title: str | None = None,
    xlim: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> plt.Axes:
    """Baseline / offline / online EV power (kW), nb10 thesis day style."""
    apply_thesis_rc()
    day_title = day_start.strftime("%d/%m/%Y")
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=(10, 4.2))
    elif standalone:
        created_fig = ax.figure

    if xlim is None and standalone:
        xlim = _day_view_xlim(day_start)

    ts = day_df["timestamp"]
    _zoh(ax, ts, baseline_ev_kw(day_df), color=C_BASELINE, linestyle=LS_ONLINE, label=_LAB_EV_BASELINE)
    if "ev_charge_power" in day_df.columns:
        _zoh(
            ax,
            ts,
            day_df["ev_charge_power"],
            color=C_OFFLINE,
            linestyle=LS_OFFLINE,
            label=_LAB_EV_OFFLINE,
        )
    _zoh(
        ax,
        ts,
        day_df["ev_power_kw_online"],
        color=C_ONLINE,
        linestyle=LS_ONLINE,
        label=_LAB_EV_ONLINE,
    )

    if show_window:
        _ev_window_lines(ax, day_start)
    if show_enforce_markers:
        _enforce_markers(ax, day_df)

    ax.set_ylabel("kW")
    ax.set_xlabel("Time")
    ax.set_title(title if title is not None else f"EV power — {day_title}")
    _style_day_xaxis(ax, day_start, xlim=xlim)

    if standalone and created_fig is not None:
        order = [
            _LAB_EV_BASELINE,
            _LAB_EV_OFFLINE,
            _LAB_EV_ONLINE,
        ]
        _legend_ordered_below(created_fig, ax, order, ncol=2)
        plt.show()
    return ax


def plot_thesis_day_grid_power(
    day_df: pd.DataFrame,
    day_start: pd.Timestamp,
    *,
    ax=None,
    standalone: bool = True,
    title: str | None = None,
    xlim: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> plt.Axes:
    """Baseline / online / offline / access grid power (kW), nb10 thesis day style."""
    apply_thesis_rc()
    day_title = day_start.strftime("%d/%m/%Y")
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=(10, 4.2))
    elif standalone:
        created_fig = ax.figure

    if xlim is None and standalone:
        xlim = _day_view_xlim(day_start)

    ts = day_df["timestamp"]

    if "forecast_access_exceedance_active" in day_df.columns:
        _shade_stress(ax, ts, day_df["forecast_access_exceedance_active"])

    _zoh(ax, ts, baseline_grid_kw(day_df), color=C_BASELINE, linestyle=LS_ONLINE, label=_LAB_GRID_BASELINE)

    if "grid_power_online" in day_df.columns:
        online = pd.to_numeric(day_df["grid_power_online"], errors="coerce")
        _zoh(ax, ts, online, color=C_ONLINE, linestyle=LS_ONLINE, label=_LAB_GRID_ONLINE)
        if "access_kw" in day_df.columns:
            access = pd.to_numeric(day_df["access_kw"], errors="coerce")
            viol = (online > access).fillna(False)
            if viol.any():
                ax.fill_between(
                    ts,
                    access,
                    online,
                    where=viol,
                    color=C_ONLINE,
                    alpha=0.15,
                    step="post",
                )

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
    ax.set_title(title if title is not None else f"Grid power — {day_title}")
    _style_day_xaxis(ax, day_start, xlim=xlim)

    if standalone and created_fig is not None:
        order = [
            "Forecast stress",
            _LAB_GRID_BASELINE,
            _LAB_GRID_ONLINE,
            _LAB_GRID_OFFLINE,
            _LAB_GRID_ACCESS,
        ]
        _legend_ordered_below(created_fig, ax, order, ncol=2)
        plt.show()
    return ax


def plot_thesis_day_spot_price(
    day_df: pd.DataFrame,
    day_start: pd.Timestamp,
    *,
    ax=None,
    standalone: bool = True,
    title: str | None = None,
    xlim: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> plt.Axes:
    """Electricity spot price (€/MWh), nb10 thesis day style."""
    apply_thesis_rc()
    day_title = day_start.strftime("%d/%m/%Y")
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=(10, 4.2))
    elif standalone:
        created_fig = ax.figure

    if xlim is None and standalone:
        xlim = _day_view_xlim(day_start)

    ts = day_df["timestamp"]
    _zoh(
        ax,
        ts,
        spot_price_eur_per_mwh(day_df),
        color=C_BASELINE,
        linestyle=LS_ONLINE,
        label=_LAB_SPOT,
    )

    ax.set_ylabel("€/MWh")
    ax.set_xlabel("Time")
    ax.set_title(title if title is not None else f"Electricity spot price — {day_title}")
    _style_day_xaxis(ax, day_start, xlim=xlim)

    if standalone and created_fig is not None:
        _legend_ordered_below(created_fig, ax, [_LAB_SPOT], ncol=1)
        plt.show()
    return ax


def _thesis_slack_line(
    ax,
    slack_min,
    values,
    *,
    label: str,
    color: str = C_ONLINE,
) -> None:
    ax.plot(
        slack_min,
        values,
        color=color,
        linestyle=LS_ONLINE,
        linewidth=LW,
        marker="o",
        markersize=5,
        markerfacecolor=color,
        markeredgecolor=color,
        label=label,
    )


def _style_slack_xaxis(ax, slack_min) -> None:
    ticks = sorted(pd.to_numeric(slack_min, errors="coerce").dropna().unique())
    ax.set_xticks(ticks)
    ax.set_xlim(min(ticks) - 5, max(ticks) + 5)


def plot_thesis_slack_unmet(sens: pd.DataFrame) -> plt.Axes:
    """Unmet EV energy (MWh/year) vs deadline slack — thesis line plot."""
    apply_thesis_rc()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_axisbelow(True)

    slack = pd.to_numeric(sens["slack_min"], errors="coerce")
    unmet = pd.to_numeric(sens["unmet_mwh"], errors="coerce")
    _thesis_slack_line(ax, slack, unmet, label="Unmet energy (online)")
    ax.axhline(0.0, color=C_BLACK, linewidth=0.8, linestyle="-", alpha=0.35, zorder=1)

    ax.set_xlabel("Deadline slack (min)")
    ax.set_ylabel("Unmet energy (MWh/year)")
    ax.set_title("Unmet EV energy vs deadline slack")
    _style_slack_xaxis(ax, slack)
    _legend_ordered_below(fig, ax, ["Unmet energy (online)"], ncol=1)
    plt.show()
    return ax


def plot_thesis_monthly_peaks_ev(
    monthly_df: pd.DataFrame,
    *,
    month_col: str = "month",
    baseline_col: str = "baseline_peak_kw",
    offline_col: str = "deterministic_peak_kw",
    online_col: str = "online_planner_peak_kw",
    title: str = "Monthly peak power — baseline vs offline vs online (planner-only)",
    ylim_bottom: float = 2000.0,
) -> plt.Axes:
    """Grouped bar chart of monthly peaks (thesis style, notebook 09 Part 5)."""
    apply_thesis_rc()
    df = monthly_df.copy()
    months = df[month_col].astype(str).tolist()
    x = np.arange(len(months))
    width = 0.26

    base = pd.to_numeric(df[baseline_col], errors="coerce")
    off = pd.to_numeric(df[offline_col], errors="coerce")
    on = pd.to_numeric(df[online_col], errors="coerce")
    peak_vals = np.concatenate([base.values, off.values, on.values])
    ymax = float(np.nanmax(peak_vals)) * 1.05

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_axisbelow(True)
    ax.bar(
        x - width,
        base,
        width,
        label="Baseline",
        color=C_BASELINE,
        alpha=0.85,
        edgecolor="none",
    )
    ax.bar(
        x,
        off,
        width,
        label="Offline",
        color=C_OFFLINE,
        alpha=0.85,
        edgecolor="none",
    )
    ax.bar(
        x + width,
        on,
        width,
        label="Online (planner-only)",
        color=C_ONLINE,
        alpha=0.85,
        edgecolor="none",
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly peak power (kW)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    _legend_ordered_below(
        fig,
        ax,
        ["Baseline", "Offline", "Online (planner-only)"],
        ncol=3,
    )
    plt.tight_layout()
    ax.set_autoscaley_on(False)
    ax.set_ylim(ylim_bottom, ymax)
    plt.show()
    return ax


def plot_thesis_slack_savings(sens: pd.DataFrame) -> plt.Axes:
    """Online net savings vs baseline (EUR/year) vs deadline slack — thesis line plot."""
    apply_thesis_rc()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_axisbelow(True)

    slack = pd.to_numeric(sens["slack_min"], errors="coerce")
    savings = pd.to_numeric(sens["online_savings_vs_baseline_eur"], errors="coerce")
    _thesis_slack_line(ax, slack, savings, label="Savings vs baseline (online)")

    ax.set_xlabel("Deadline slack (min)")
    ax.set_ylabel("Savings vs baseline (EUR/year)")
    ax.set_title("Online MPC savings vs baseline vs deadline slack")
    _style_slack_xaxis(ax, slack)
    _legend_ordered_below(fig, ax, ["Savings vs baseline (online)"], ncol=1)
    plt.show()
    return ax
