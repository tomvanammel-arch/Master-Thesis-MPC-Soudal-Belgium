"""Single-day replay diagnostics for joint online EV+HP MPC (notebook 11)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _zoh(ax, ts, y, **kwargs) -> None:
    ax.step(pd.to_datetime(ts, errors="coerce"), pd.to_numeric(y, errors="coerce"), where="post", **kwargs)


def mpc_region_bounds(
    day_start: pd.Timestamp, *, slack_minutes: int = 105
) -> Dict[str, pd.Timestamp]:
    """Weekday MPC / catch-up / EV-window boundaries (matches online_MPC_1_EV_HP)."""
    d0 = pd.Timestamp(day_start).normalize()
    slack_h = float(slack_minutes) / 60.0
    return {
        "ev_window_start": d0 + pd.Timedelta(hours=7),
        "mpc_region_end": d0 + pd.Timedelta(hours=17.0 - slack_h),
        "ev_window_end": d0 + pd.Timedelta(hours=17),
    }


def shade_mpc_regions(
    axes: Sequence,
    day_start: pd.Timestamp,
    *,
    slack_minutes: int = 105,
    weekday_only: bool = True,
) -> None:
    """Light bands: MPC planner region vs catch-up enforce region (weekday)."""
    if weekday_only and pd.Timestamp(day_start).weekday() >= 5:
        return
    b = mpc_region_bounds(day_start, slack_minutes=slack_minutes)
    t_mpc0, t_mpc1 = b["ev_window_start"], b["mpc_region_end"]
    t_cu0, t_cu1 = b["mpc_region_end"], b["ev_window_end"]
    first_mpc, first_cu = True, True
    for ax in axes:
        ax.axvspan(
            t_mpc0,
            t_mpc1,
            facecolor="tab:blue",
            alpha=0.06,
            zorder=0,
            label="MPC region (07:00–17:00−slack)" if first_mpc else None,
        )
        ax.axvspan(
            t_cu0,
            t_cu1,
            facecolor="tab:orange",
            alpha=0.10,
            zorder=0,
            label="Catch-up / enforce (17:00−slack → 17:00)" if first_cu else None,
        )
        first_mpc = first_cu = False


def vlines_mpc_deadline(
    ax,
    day_start: pd.Timestamp,
    *,
    slack_minutes: int = 105,
    weekday_only: bool = True,
) -> None:
    """Vertical markers for MPC end (17:00−slack) and EV window end (17:00)."""
    if weekday_only and pd.Timestamp(day_start).weekday() >= 5:
        return
    b = mpc_region_bounds(day_start, slack_minutes=slack_minutes)
    ax.axvline(
        b["mpc_region_end"],
        color="tab:blue",
        linestyle="-.",
        linewidth=1.4,
        alpha=0.9,
        label=f"MPC ends ({b['mpc_region_end'].strftime('%H:%M')})",
    )
    ax.axvline(
        b["ev_window_end"],
        color="red",
        linestyle="--",
        linewidth=1.2,
        alpha=0.85,
        label="EV window ends (17:00)",
    )


def summarize_day(
    df_day: pd.DataFrame,
    *,
    day: pd.Timestamp,
    slack_minutes: int = 105,
    clip_col: str = "p_limit_kw",
) -> Dict[str, Any]:
    """Text diagnostics for one calendar day."""
    d = df_day.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d = d.dropna(subset=["timestamp"]).sort_values("timestamp")
    if d.empty:
        return {"day": str(day.date()), "error": "no rows"}

    slack_h = slack_minutes / 60.0
    tod = d["timestamp"].dt.hour + d["timestamp"].dt.minute / 60.0
    wd = d["timestamp"].dt.dayofweek < 5
    in_mpc = wd & (tod >= 7.0) & (tod < 17.0 - slack_h)
    in_cu = wd & (tod >= 17.0 - slack_h) & (tod < 17.0)

    p_grid = pd.to_numeric(d["p_grid_actual_kw"], errors="coerce")
    access = pd.to_numeric(d["access_kw"], errors="coerce")
    p_lim = pd.to_numeric(d.get(clip_col, d.get("p_limit_kw")), errors="coerce")
    ev_app = pd.to_numeric(d.get("ev_applied", d.get("ev_applied_kw", 0) / 4.0), errors="coerce")
    if "ev_applied_kw" in d.columns and "ev_applied" not in d.columns:
        ev_app = pd.to_numeric(d["ev_applied_kw"], errors="coerce") / 4.0

    exceed_ap = (p_grid - access).clip(lower=0.0)
    viol_ap = p_grid > access + 0.5
    viol_clip = p_grid > p_lim + 0.5

    out: Dict[str, Any] = {
        "day": str(day.date()),
        "slack_minutes": slack_minutes,
        "mpc_region_end_clock": (pd.Timestamp(day).normalize() + pd.Timedelta(hours=17.0 - slack_h)).strftime(
            "%H:%M"
        ),
        "max_p_grid_kw": float(p_grid.max()),
        "access_kw": float(access.iloc[0]) if access.notna().any() else np.nan,
        "min_p_limit_kw": float(p_lim.min()),
        "max_p_limit_kw": float(p_lim.max()),
        "access_violation_steps": int(viol_ap.sum()),
        "max_access_exceed_kw": float(exceed_ap.max()),
        "clip_violation_steps": int(viol_clip.sum()),
        "ev_applied_kwh": float(ev_app.sum()),
        "ev_enforce_active_steps": int((pd.to_numeric(d.get("ev_enforce_active"), errors="coerce") > 0.5).sum()),
        "ev_enforce_deferred_steps": int((pd.to_numeric(d.get("ev_enforce_deferred"), errors="coerce") > 0.5).sum()),
        "ev_enforce_extra_kwh": float(pd.to_numeric(d.get("ev_enforce_extra_kwh"), errors="coerce").sum()),
        "mpc_region_ev_kwh": float(ev_app[in_mpc].sum()),
        "catchup_region_ev_kwh": float(ev_app[in_cu].sum()),
        "catchup_max_exceed_kw": float(exceed_ap[in_cu].max()) if in_cu.any() else 0.0,
    }
    if "uncharged_kwh" in d.columns:
        out["uncharged_kwh_eod"] = float(pd.to_numeric(d["uncharged_kwh"], errors="coerce").iloc[-1])
    if "ev_to_deliver_kwh" in d.columns:
        out["ev_to_deliver_at_mpc_end"] = float(
            pd.to_numeric(d.loc[in_mpc, "ev_to_deliver_kwh"], errors="coerce").iloc[-1]
            if in_mpc.any()
            else pd.to_numeric(d["ev_to_deliver_kwh"], errors="coerce").max()
        )
    return out


def print_day_replay_report(summary: Dict[str, Any]) -> None:
    """Human-readable replay report."""
    if summary.get("error"):
        print(summary)
        return
    print("\n" + "=" * 80)
    print(f"Day replay — {summary['day']}  (slack {summary['slack_minutes']} min → MPC ends {summary['mpc_region_end_clock']})")
    print("=" * 80)
    print(
        f"Grid peak {summary['max_p_grid_kw']:.1f} kW | access {summary['access_kw']:.1f} kW | "
        f"p_limit [{summary['min_p_limit_kw']:.1f}, {summary['max_p_limit_kw']:.1f}] kW"
    )
    print(
        f"Access violations: {summary['access_violation_steps']} steps, max exceed {summary['max_access_exceed_kw']:.1f} kW"
    )
    print(
        f"EV delivered {summary['ev_applied_kwh']:.1f} kWh "
        f"(MPC region {summary['mpc_region_ev_kwh']:.1f}, catch-up {summary['catchup_region_ev_kwh']:.1f})"
    )
    print(
        f"Enforce: {summary['ev_enforce_active_steps']} active steps, "
        f"{summary['ev_enforce_extra_kwh']:.1f} kWh extra | "
        f"Deferred: {summary['ev_enforce_deferred_steps']} steps"
    )
    if "ev_to_deliver_at_mpc_end" in summary:
        print(f"EV still to deliver at MPC region end: {summary['ev_to_deliver_at_mpc_end']:.1f} kWh")
    print(
        f"Catch-up max access exceed: {summary['catchup_max_exceed_kw']:.1f} kW\n"
        "Mechanism: p_limit ≈ min(access, max(peak_so_far, monthly_peak_plan)) in MPC region; "
        "low monthly_peak_plan holds clip below access until catch-up enforce pushes EV."
    )


def plot_day_replay(
    df_day: pd.DataFrame,
    *,
    day: pd.Timestamp,
    slack_minutes: int = 105,
    clip_col: str = "p_limit_kw",
    xlim: Optional[tuple] = None,
) -> plt.Figure:
    """Four-panel replay: grid limits, EV energy, remaining EV, SOC."""
    d = df_day.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d = d.dropna(subset=["timestamp"]).sort_values("timestamp")
    ev_kw = pd.to_numeric(d.get("ev_applied_kw"), errors="coerce")
    if ev_kw.isna().all() and "ev_applied" in d.columns:
        ev_kw = pd.to_numeric(d["ev_applied"], errors="coerce") * 4.0

    day0 = pd.Timestamp(day).normalize()
    if xlim is None:
        xlim = (day0 + pd.Timedelta(hours=5), day0 + pd.Timedelta(hours=19))

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    shade_mpc_regions(axes, day0, slack_minutes=slack_minutes)

    ax = axes[0]
    _zoh(ax, d["timestamp"], d["p_grid_actual_kw"], label="Grid actual", color="black")
    _zoh(ax, d["timestamp"], d["access_kw"], label="Access power", linestyle="--", color="tab:red")
    if clip_col in d.columns or "p_limit_kw" in d.columns:
        _zoh(ax, d["timestamp"], d.get(clip_col, d["p_limit_kw"]), label="Clip limit", linestyle=":", color="tab:green")
    viol = pd.to_numeric(d["p_grid_actual_kw"], errors="coerce") > pd.to_numeric(d["access_kw"], errors="coerce")
    if viol.any():
        ax.fill_between(
            d["timestamp"],
            d["access_kw"],
            d["p_grid_actual_kw"],
            where=viol,
            color="tab:red",
            alpha=0.15,
            label="Access violation",
        )
    vlines_mpc_deadline(ax, day0, slack_minutes=slack_minutes)
    ax.set_ylabel("kW")
    ax.set_title(f"Day replay — grid vs access vs clip ({day0.strftime('%Y-%m-%d')})")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if "ev_plan_kwh" in d.columns:
        _zoh(ax, d["timestamp"], pd.to_numeric(d["ev_plan_kwh"], errors="coerce") * 4.0, label="EV plan", alpha=0.7)
    _zoh(ax, d["timestamp"], ev_kw, label="EV applied", color="tab:blue", linewidth=2)
    if "ev_enforce_active" in d.columns:
        m = d[pd.to_numeric(d["ev_enforce_active"], errors="coerce") > 0.5]
        if len(m):
            ax.scatter(m["timestamp"], ev_kw.loc[m.index], c="red", s=45, zorder=5, label="Enforce active")
    if "ev_enforce_deferred" in d.columns:
        m = d[pd.to_numeric(d["ev_enforce_deferred"], errors="coerce") > 0.5]
        if len(m):
            ax.scatter(m["timestamp"], ev_kw.loc[m.index], c="gold", s=45, zorder=5, label="Enforce deferred")
    vlines_mpc_deadline(ax, day0, slack_minutes=slack_minutes)
    ax.set_ylabel("kW")
    ax.set_title("EV power")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    if "ev_to_deliver_kwh" in d.columns:
        _zoh(ax, d["timestamp"], d["ev_to_deliver_kwh"], label="EV to deliver", color="tab:purple")
    if "ev_envelope_remaining_kwh" in d.columns:
        _zoh(
            ax,
            d["timestamp"],
            d["ev_envelope_remaining_kwh"],
            label="Envelope headroom to 17:00",
            linestyle="--",
            color="tab:orange",
        )
    vlines_mpc_deadline(ax, day0, slack_minutes=slack_minutes)
    ax.set_ylabel("kWh")
    ax.set_title("Remaining EV obligation")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    if "soc_after" in d.columns:
        _zoh(ax, d["timestamp"], pd.to_numeric(d["soc_after"], errors="coerce") * 100.0, label="SOC after %")
    if "soc_min_planner_floor" in d.columns:
        _zoh(
            ax,
            d["timestamp"],
            pd.to_numeric(d["soc_min_planner_floor"], errors="coerce") * 100.0,
            label="Planner SOC floor",
            linestyle=":",
            alpha=0.8,
        )
    vlines_mpc_deadline(ax, day0, slack_minutes=slack_minutes)
    ax.set_ylabel("SOC %")
    ax.set_xlabel("Timestamp")
    ax.set_title("Buffer SOC")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlim(*xlim)
    fig.tight_layout()
    return fig
