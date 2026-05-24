"""
Notebook 10 Part 3.1 / 3.2 / 4D code paths extracted for Part 4 scenario viewing.
Keep in sync when editing `notebooks/10_online_MPC_1_HP.ipynb` cells 7–8 and Part 4D.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

try:
    from IPython.display import display
except ImportError:

    def display(obj):  # type: ignore[no-redef]
        print(obj)


def run_notebook10_part31_optimized_volumes(res_hp_online: pd.DataFrame, project_root: Path) -> None:
    # Part 3.1 — Optimized volumes (plots)

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from pathlib import Path

    # ------------------------------------------------------------------
    # Defaults for notebook-level knobs (Part 2) so this viewer can run
    # standalone in Part 4 without requiring earlier cells.
    # If the notebook already defined these names, those values win.
    # ------------------------------------------------------------------
    # These are typically defined in the *notebook kernel* (i.e. __main__), not inside this module.
    # So we look them up from __main__ first, and fall back to safe defaults.
    import sys as _sys

    _main = _sys.modules.get("__main__")
    _main_ns = vars(_main) if _main is not None else {}

    INFLEX_FORECAST_STRATEGY = _main_ns.get("INFLEX_FORECAST_STRATEGY", "c_p50")
    EV_FORECAST_STRATEGY = _main_ns.get("EV_FORECAST_STRATEGY", "c_p50")
    THERMAL_FORECAST_STRATEGY = _main_ns.get("THERMAL_FORECAST_STRATEGY", "c2t_p50")
    PV_FORECAST_STRATEGY = _main_ns.get("PV_FORECAST_STRATEGY", "chronos2_elia_p50")
    ENABLE_FORECAST_STRESS_SOC_FLOOR = _main_ns.get("ENABLE_FORECAST_STRESS_SOC_FLOOR", True)
    ACCESS_POWER_DICT = _main_ns.get("ACCESS_POWER_DICT", None)

    # Load deterministic HP MPC exports from notebook 03
    PROJECT_ROOT = project_root
    NOTEBOOKS_OUTPUT_DIR = PROJECT_ROOT / "output" / "notebooks"
    DET_HP_15MIN_PATH = NOTEBOOKS_OUTPUT_DIR / "deterministic_hp_15min_notebook_03.csv"

    # Pick one representative week + one representative day
    WEEK_START = pd.Timestamp("2025-01-20")
    WEEK_END = WEEK_START + pd.Timedelta(days=7)
    DAY = pd.Timestamp("2025-01-20")

    DEBUG_TS = pd.Timestamp("2025-01-20 08:00:00")

    det_hp_15min = None
    if DET_HP_15MIN_PATH.exists():
        det_hp_15min = pd.read_csv(DET_HP_15MIN_PATH)
        det_hp_15min["timestamp"] = pd.to_datetime(det_hp_15min["timestamp"], errors="coerce")
        if "hp_electrical_input" in det_hp_15min.columns:
            det_hp_15min["hp_kw_deterministic"] = 4.0 * det_hp_15min["hp_electrical_input"].fillna(0.0)
    else:
        print(f"NOTE: Deterministic export not found at {DET_HP_15MIN_PATH}. Run notebook 03 export cell first.")

    # Ensure timestamp dtype
    res_plot = res_hp_online.copy()
    res_plot["timestamp"] = pd.to_datetime(res_plot["timestamp"], errors="coerce")

    # Convenience (kW)
    res_plot["hp_plan_kw"] = 4.0 * res_plot["hp_plan_kwh"]
    res_plot["hp_applied_kw"] = 4.0 * res_plot["hp_applied_kwh"]

    # --- SOC violation summary (counts + consecutive durations) ---
    import yaml

    hp_cfg_path = PROJECT_ROOT / "config" / "hp.yaml"
    with open(hp_cfg_path, "r", encoding="utf-8") as f:
        hp_cfg = yaml.safe_load(f)

    SOC_MIN = float(hp_cfg["buffer"]["soc_min"])
    SOC_MAX = float(hp_cfg["buffer"]["soc_max"])


    def _summarize_soc_violations(df_soc: pd.DataFrame, *, soc_col: str, label: str, dt_minutes: int = 15):
        if df_soc is None or df_soc.empty or soc_col not in df_soc.columns:
            print(f"SOC violations ({label}): no data / missing column '{soc_col}'.")
            return

        _d = df_soc[["timestamp", soc_col]].copy()
        _d["timestamp"] = pd.to_datetime(_d["timestamp"], errors="coerce")
        _d = _d.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        _soc = pd.to_numeric(_d[soc_col], errors="coerce")

        mask = (_soc < SOC_MIN) | (_soc > SOC_MAX)
        n_points = int(mask.sum())

        runs = []
        in_run = False
        start_i = None
        for i, flag in enumerate(mask.fillna(False).tolist()):
            if flag and not in_run:
                in_run = True
                start_i = i
            elif (not flag) and in_run:
                end_i = i - 1
                runs.append((start_i, end_i))
                in_run = False
                start_i = None
        if in_run and start_i is not None:
            runs.append((start_i, len(mask) - 1))

        durations_steps = [(b - a + 1) for a, b in runs]
        durations_hours = [d * dt_minutes / 60.0 for d in durations_steps]

        print("\n" + "=" * 80)
        print(f"SOC violations summary ({label})")
        print("=" * 80)
        print(f"SOC bounds: [{SOC_MIN:.3f}, {SOC_MAX:.3f}]")
        print(f"Violation points: {n_points} (each = {dt_minutes} min)")
        print(f"Violation episodes (consecutive runs): {len(runs)}")
        if durations_hours:
            print(f"Total violated time: {sum(durations_hours):.2f} h")
            print(f"Max consecutive episode: {max(durations_hours):.2f} h")
            print(f"Consecutive durations (h): {[round(x, 2) for x in durations_hours]}")

            # Show first few episodes with timestamps
            max_show = 10
            print(f"\nFirst {min(max_show, len(runs))} episodes:")
            for k, (a, b) in enumerate(runs[:max_show], start=1):
                t0 = _d.loc[a, "timestamp"]
                t1 = _d.loc[b, "timestamp"]
                dur_h = (b - a + 1) * dt_minutes / 60.0
                print(f"{k:>2}. {t0} -> {t1}  ({dur_h:.2f} h)")


    # Online MPC SOC (soc_after) -- NOTE: soc_after is CLIPPED to [SOC_MIN, SOC_MAX] in the online simulation
    _summarize_soc_violations(res_plot, soc_col="soc_after", label="online MPC (soc_after, clipped)")

    # Deterministic MPC SOC (buffer_soc)
    if det_hp_15min is not None:
        _summarize_soc_violations(det_hp_15min, soc_col="buffer_soc", label="deterministic MPC (buffer_soc)")

    # Reconstruct *raw* (unclipped) SOC for online MPC to detect hidden infeasibilities
    buf = hp_cfg["buffer"]
    rho = float(buf["water_density_kg_per_m3"])
    cp = float(buf["cp_kj_per_kg_k"])
    dT = float(buf["usable_delta_t_k"])
    V = float(buf["size_m3"])
    loss_coeff_per_h = float(buf["loss_coefficient_per_hour"])

    buffer_capacity_kwh_th = (V * rho * cp * dT) / 3600.0
    loss_rate_per_interval = loss_coeff_per_h * 0.25  # 15-min interval

    # Load plant actual thermal_load + outdoor_temperature (sequential alignment)
    plant_path = PROJECT_ROOT / "data" / "plant1.csv"
    plant = pd.read_csv(plant_path)
    plant_ts = pd.to_datetime(plant["timestamp"], utc=True, errors="coerce")
    plant["timestamp"] = plant_ts.dt.tz_convert("Europe/Brussels").dt.tz_localize(None)
    plant = plant.sort_values("timestamp").reset_index(drop=True)
    plant = plant[(plant["timestamp"] >= pd.Timestamp("2025-01-01")) & (plant["timestamp"] < pd.Timestamp("2026-01-01"))].copy()

    n = min(len(plant), len(res_plot))
    plant = plant.iloc[:n].copy().reset_index(drop=True)
    res_al = res_plot.iloc[:n].copy().reset_index(drop=True)

    from heat_pump_load import load_hp_config, interpolate_cop
    hp_cfg2 = load_hp_config(str(PROJECT_ROOT / "config" / "hp.yaml"))

    def _cop(temp_c):
        try:
            return float(interpolate_cop(float(temp_c), hp_cfg2["COP_data"]))
        except Exception:
            return 2.5

    soc_raw = []
    unmet_thermal_kwh = []

    soc = float(buf["soc_initial"])
    for k in range(n):
        thermal_act_kwh = float(pd.to_numeric(plant.loc[k, "thermal_load"], errors="coerce") or 0.0)
        temp_act = pd.to_numeric(plant.loc[k, "outdoor_temperature"], errors="coerce")
        cop_k = _cop(temp_act) if not pd.isna(temp_act) else 2.5

        hp_kwh = float(pd.to_numeric(res_al.loc[k, "hp_applied_kwh"], errors="coerce") or 0.0)
        hp_th_out_kwh = hp_kwh * cop_k

        buffer_energy_prev = soc * buffer_capacity_kwh_th
        losses_kwh = buffer_energy_prev * loss_rate_per_interval

        # Raw SOC update (no clipping)
        soc_next_raw = soc + (hp_th_out_kwh - thermal_act_kwh - losses_kwh) / buffer_capacity_kwh_th
        soc_raw.append(soc_next_raw)

        # If raw SOC would go below SOC_MIN, compute implied unmet thermal (kWh_th)
        if soc_next_raw < SOC_MIN:
            deficit_kwh_th = (SOC_MIN - soc_next_raw) * buffer_capacity_kwh_th
            unmet_thermal_kwh.append(deficit_kwh_th)
            soc = SOC_MIN
        else:
            unmet_thermal_kwh.append(0.0)
            soc = min(max(soc_next_raw, SOC_MIN), SOC_MAX)

    # Unmet thermal is now tracked directly by the online simulation wrapper.
    # - If ENFORCE_SOC_MIN=True: unmet_thermal_kwh_th should stay ~0.
    # - If ENFORCE_SOC_MIN=False: unmet_thermal_kwh_th captures shed thermal demand needed to keep SOC >= SOC_min_phys.

    if "unmet_thermal_kwh_th" in res_plot.columns:
        total_unmet_mwh_th = float(
            pd.to_numeric(res_plot["unmet_thermal_kwh_th"], errors="coerce").fillna(0.0).sum() / 1000.0
        )
        print(f"\nOnline MPC unmet thermal (from wrapper): {total_unmet_mwh_th:,.2f} MWh_th")
    else:
        print("\nColumn 'unmet_thermal_kwh_th' not found. Re-run Part 2 with updated wrapper.")

    _summarize_soc_violations(res_plot.dropna(subset=["soc_after_raw"]), soc_col="soc_after_raw", label="online MPC (soc_after_raw)")


    def _summarize_binary_runs(mask: pd.Series, ts: pd.Series, *, label: str, dt_minutes: int = 15):
        _mask = mask.fillna(False).astype(bool).tolist()
        _ts = pd.to_datetime(ts, errors="coerce")

        runs = []
        in_run = False
        start_i = None
        for i, flag in enumerate(_mask):
            if flag and not in_run:
                in_run = True
                start_i = i
            elif (not flag) and in_run:
                end_i = i - 1
                runs.append((start_i, end_i))
                in_run = False
                start_i = None
        if in_run and start_i is not None:
            runs.append((start_i, len(_mask) - 1))

        durations_steps = [(b - a + 1) for a, b in runs]
        durations_hours = [d * dt_minutes / 60.0 for d in durations_steps]

        print("\n" + "=" * 80)
        print(f"Runs summary ({label})")
        print("=" * 80)
        print(f"Points (condition True): {int(sum(_mask))} (each = {dt_minutes} min)")
        print(f"Episodes (consecutive runs): {len(runs)}")
        if durations_hours:
            print(f"Total duration: {sum(durations_hours):.2f} h")
            print(f"Max consecutive episode: {max(durations_hours):.2f} h")
            print(f"Consecutive durations (h): {[round(x, 2) for x in durations_hours]}")

            max_show = 10
            print(f"\nFirst {min(max_show, len(runs))} episodes:")
            for k, (a, b) in enumerate(runs[:max_show], start=1):
                t0 = _ts.iloc[a] if hasattr(_ts, "iloc") else _ts[a]
                t1 = _ts.iloc[b] if hasattr(_ts, "iloc") else _ts[b]
                dur_h = (b - a + 1) * dt_minutes / 60.0
                print(f"{k:>2}. {t0} -> {t1}  ({dur_h:.2f} h)")


    # --- PLC safeguard energy (hp_plc_extra_kwh) ---
    if "hp_plc_extra_kwh" in res_plot.columns:
        _plc_extra = pd.to_numeric(res_plot["hp_plc_extra_kwh"], errors="coerce").fillna(0.0)
        _plc_active = _plc_extra > 0.0
        print("\n" + "=" * 80)
        print("PLC safeguard summary (online MPC)")
        print("=" * 80)
        print(f"Active intervals: {int(_plc_active.sum())} (each = 15 min)")
        print(f"Total extra HP energy: {float(_plc_extra.sum() / 1000.0):,.3f} MWh_el")

        # Episode/duration summary
        _summarize_binary_runs(_plc_active, res_plot["timestamp"], label="PLC active (hp_plc_extra_kwh > 0)")
    else:
        print("\nNOTE: Column 'hp_plc_extra_kwh' not found in res_hp_online. Re-run Part 2 with PLC safeguard enabled.")


    # --- Access power violations (counts + consecutive durations) ---

    # Online: p_grid_actual_kw > access_kw
    if "p_grid_actual_kw" in res_plot.columns and "access_kw" in res_plot.columns:
        _viol_online = (res_plot["p_grid_actual_kw"] > res_plot["access_kw"]).fillna(False)
        _summarize_binary_runs(_viol_online, res_plot["timestamp"], label="online MPC (p_grid_actual_kw > access_kw)")

    # Deterministic: grid_power > access_power
    if det_hp_15min is not None and "grid_power" in det_hp_15min.columns and "access_power" in det_hp_15min.columns:
        _viol_det = (det_hp_15min["grid_power"] > det_hp_15min["access_power"]).fillna(False)
        _summarize_binary_runs(_viol_det, det_hp_15min["timestamp"], label="deterministic MPC (grid_power > access_power)")


    # --- Forecast-stress peak periods (forecast grid > access) ---
    if "forecast_access_exceedance_active" in res_plot.columns:
        _stress = (pd.to_numeric(res_plot["forecast_access_exceedance_active"], errors="coerce").fillna(0.0) > 0.5)
        stress_hours_total = float(_stress.sum()) * 0.25
        print("\n" + "=" * 80)
        print("Forecast-stress periods summary")
        print("=" * 80)
        print(f"Total stress time (year): {stress_hours_total:.2f} h")
        if "soc_min_planner_floor" in res_plot.columns:
            _floor = pd.to_numeric(res_plot["soc_min_planner_floor"], errors="coerce")
            print(f"Planner SOC floor range: [{float(_floor.min()):.3f}, {float(_floor.max()):.3f}]")
        _summarize_binary_runs(_stress, res_plot["timestamp"], label="forecast stress (forecast_grid_kw > access_kw)")
    else:
        print("\nNOTE: Column 'forecast_access_exceedance_active' not found. Re-run Part 2 with updated wrapper.")


    week = res_plot[(res_plot["timestamp"] >= WEEK_START) & (res_plot["timestamp"] < WEEK_END)].copy()
    day = res_plot[(res_plot["timestamp"] >= DAY) & (res_plot["timestamp"] < DAY + pd.Timedelta(days=1))].copy()

    if week.empty or day.empty:
        raise ValueError("Selected WEEK_START/DAY not found in res_hp_online timestamps.")

    _col_clip = "grid_clip_limit_kw" if "grid_clip_limit_kw" in res_plot.columns else "p_limit_kw"


    def _shade_forecast_access_stress(axes_list, ts: pd.Series, stress_active: pd.Series) -> None:
        """Light vertical bands when forecast grid power (MPC inputs) exceeds access — same x as week/day plots."""
        if stress_active is None or len(stress_active) == 0:
            return
        sa = pd.to_numeric(stress_active, errors="coerce").fillna(0.0) > 0.5
        if not sa.any():
            return
        sp = sa.to_numpy()
        t = pd.to_datetime(ts, errors="coerce")
        n = len(sp)
        i = 0
        first = True
        while i < n:
            if not sp[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and sp[j + 1]:
                j += 1
            t0 = t.iloc[i]
            t1 = t.iloc[j] + pd.Timedelta(minutes=15)
            lab = "Forecast grid > access (stress)" if first else None
            first = False
            axes_list[0].axvspan(t0, t1, facecolor="tab:purple", alpha=0.12, zorder=0, label=lab)
            for ax in axes_list[1:]:
                ax.axvspan(t0, t1, facecolor="tab:purple", alpha=0.12, zorder=0)
            i = j + 1


    # --- Plot: one week ---
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

    # 1) Grid power vs limits
    axes[0].plot(week["timestamp"], week["p_grid_actual_kw"], label="Grid power actual (kW)", alpha=0.8)
    axes[0].plot(week["timestamp"], week[_col_clip], label="Grid clip limit (max of realized peak, MPC u0)", alpha=0.9)
    axes[0].plot(week["timestamp"], week["access_kw"], label="Access power (kW)", linestyle="--", alpha=0.7)

    # Highlight access power violations (actual > access)
    _week_viol = (week["p_grid_actual_kw"] > week["access_kw"]).fillna(False)
    if _week_viol.any():
        axes[0].fill_between(
            week["timestamp"],
            week["access_kw"],
            week["p_grid_actual_kw"],
            where=_week_viol,
            color="tab:red",
            alpha=0.15,
            label="Access violation area",
        )

    if det_hp_15min is not None:
        week_det = det_hp_15min[(det_hp_15min["timestamp"] >= WEEK_START) & (det_hp_15min["timestamp"] < WEEK_END)].copy()
        if not week_det.empty and "grid_power" in week_det.columns:
            axes[0].plot(week_det["timestamp"], week_det["grid_power"], label="Grid power deterministic (kW)", alpha=0.6)

    axes[0].set_ylabel("kW")
    axes[0].set_title("Week: grid power and limits")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.4)
    axes[0].legend(loc="upper right")

    # 2) HP electrical power (planned vs applied + deterministic)
    axes[1].plot(week["timestamp"], week["hp_plan_kw"], label="HP plan (kW)", alpha=0.8)
    axes[1].plot(week["timestamp"], week["hp_applied_kw"], label="HP applied (kW)", alpha=0.8)

    if det_hp_15min is not None:
        week_det = det_hp_15min[(det_hp_15min["timestamp"] >= WEEK_START) & (det_hp_15min["timestamp"] < WEEK_END)].copy()
        if not week_det.empty and "hp_kw_deterministic" in week_det.columns:
            axes[1].plot(week_det["timestamp"], week_det["hp_kw_deterministic"], label="HP deterministic (kW)", alpha=0.7)

    axes[1].set_ylabel("kW")
    axes[1].set_title("Week: HP electrical power")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.4)
    axes[1].legend(loc="upper right")

    # 3) SOC
    axes[2].plot(week["timestamp"], week["soc_before"], label="SOC before", alpha=0.8)
    axes[2].plot(week["timestamp"], week["soc_after"], label="SOC after", alpha=0.8)
    if "soc_min_planner_floor" in week.columns:
        axes[2].plot(
            week["timestamp"],
            week["soc_min_planner_floor"],
            label="Planner SOC floor (applied)",
            alpha=0.65,
            linestyle=":",
        )
        _soc_floor_target = float(pd.to_numeric(week["soc_min_planner_floor"], errors="coerce").max())
        axes[2].axhline(
            _soc_floor_target,
            color="tab:gray",
            linestyle="--",
            alpha=0.6,
            label=f"Planner SOC floor target ({_soc_floor_target:.2f})",
        )

    # Optional: unmet thermal (kWh_th per 15 min) tracked by wrapper when SOC-min enforcement is disabled
    ax2b = None
    if "unmet_thermal_kwh_th" in week.columns:
        ax2b = axes[2].twinx()
        ax2b.plot(
            week["timestamp"],
            pd.to_numeric(week["unmet_thermal_kwh_th"], errors="coerce").fillna(0.0),
            color="tab:red",
            alpha=0.35,
            label="Unmet thermal (kWh_th/15min)",
        )
        ax2b.set_ylabel("Unmet thermal (kWh_th/15min)")

    if det_hp_15min is not None:
        week_det = det_hp_15min[(det_hp_15min["timestamp"] >= WEEK_START) & (det_hp_15min["timestamp"] < WEEK_END)].copy()
        if not week_det.empty and "buffer_soc" in week_det.columns:
            axes[2].plot(week_det["timestamp"], week_det["buffer_soc"], label="SOC deterministic", alpha=0.7)

    axes[2].set_ylabel("SOC (fraction)")
    axes[2].set_title("Week: buffer SOC")
    axes[2].grid(True, axis="y", linestyle="--", alpha=0.4)

    lines, labels = axes[2].get_legend_handles_labels()
    if ax2b is not None:
        lines2, labels2 = ax2b.get_legend_handles_labels()
        lines += lines2
        labels += labels2
    axes[2].legend(lines, labels, loc="upper right")

    # 4) Planner exceedance signal (rolling max exceedance from MPC) + clipping indicator
    if "access_overage_kw" in week.columns:
        axes[3].plot(week["timestamp"], week["access_overage_kw"], label="MPC rolling max exceedance (kW)", alpha=0.8)
    axes[3].plot(week["timestamp"], 50.0 * week["was_clipped"], label="Clipped (scaled)", alpha=0.6)

    # Access power violation magnitude (kW): max(0, actual - access)
    axes[3].plot(
        week["timestamp"],
        (week["p_grid_actual_kw"] - week["access_kw"]).clip(lower=0.0),
        label="Access violation (kW)",
        alpha=0.7,
    )

    if det_hp_15min is not None:
        week_det = det_hp_15min[(det_hp_15min["timestamp"] >= WEEK_START) & (det_hp_15min["timestamp"] < WEEK_END)].copy()
        if not week_det.empty and "rolling_max_exceedance" in week_det.columns:
            axes[3].plot(week_det["timestamp"], week_det["rolling_max_exceedance"], label="Deterministic rolling max exceedance (kW)", alpha=0.7)

    axes[3].set_ylabel("kW")
    axes[3].set_title("Week: exceedance signal (MPC) + clipping")
    axes[3].grid(True, axis="y", linestyle="--", alpha=0.4)
    axes[3].legend(loc="upper right")

    _st_week = (
        week["forecast_access_exceedance_active"]
        if "forecast_access_exceedance_active" in week.columns
        else pd.Series(0.0, index=week.index)
    )
    _shade_forecast_access_stress(list(axes), week["timestamp"], _st_week)
    axes[0].legend(loc="upper right")

    plt.tight_layout()
    plt.show()

    # --- Plot: one day (ZOH, thesis style, three separate figures) ---
    import matplotlib as mpl
    import matplotlib.dates as mdates

    mpl.rcParams.update(
        {
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
    )
    _C_ONLINE = "#b30000"
    _C_OFFLINE = "#000000"
    _C_ACCESS = "#2166ac"
    _C_STRESS = "#c8d8e8"
    _LS_ONLINE = "-"       # red: solid
    _LS_OFFLINE = "--"     # black: dashed
    _LS_CONSTRAINT = ":"   # black: dotted (limits)
    _LW = 2.2
    _day_end = DAY + pd.Timedelta(days=1)
    _day_title = DAY.strftime("%d/%m/%Y")

    def _zoh(ax, ts, y, **kwargs):
        plot_kw = {"linewidth": _LW, "drawstyle": "steps-post"}
        plot_kw.update(kwargs)
        ax.plot(
            pd.to_datetime(ts, errors="coerce"),
            pd.to_numeric(y, errors="coerce"),
            **plot_kw,
        )

    def _style_day_xaxis(ax):
        ax.set_xlim(DAY, _day_end)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
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
                facecolor=_C_STRESS,
                alpha=0.55,
                zorder=0,
                label="Forecast stress" if first else None,
            )
            first = False
            i = j + 1

    def _legend_below(fig, ax, *, ncol: int = 3):
        h, lab = ax.get_legend_handles_labels()
        ax.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=ncol, frameon=False)
        fig.subplots_adjust(bottom=0.28, top=0.90)

    def _legend_grid_below(fig, ax):
        """Row 1: stress, online, offline — row 2: access power (left to right)."""
        h, lab = ax.get_legend_handles_labels()
        order = [
            "Forecast stress",
            "Grid power (online)",
            "Grid power (offline)",
            "Access power (online)",
        ]
        oh, ol = [], []
        for name in order:
            if name in lab:
                i = lab.index(name)
                oh.append(h[i])
                ol.append(lab[i])
        ncol = 3 if len(ol) >= 4 else 2
        ax.legend(oh, ol, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=ncol, frameon=False)
        fig.subplots_adjust(bottom=0.28, top=0.90)

    day_det = None
    if det_hp_15min is not None:
        day_det = det_hp_15min[(det_hp_15min["timestamp"] >= DAY) & (det_hp_15min["timestamp"] < _day_end)].copy()

    _st_day = (
        day["forecast_access_exceedance_active"]
        if "forecast_access_exceedance_active" in day.columns
        else pd.Series(0.0, index=day.index)
    )

    # 1) Grid power
    fig_g, ax_g = plt.subplots(figsize=(10, 4.2))
    _shade_stress(ax_g, day["timestamp"], _st_day)
    _zoh(ax_g, day["timestamp"], day["p_grid_actual_kw"], color=_C_ONLINE, linestyle=_LS_ONLINE, label="Grid power (online)")
    if day_det is not None and not day_det.empty and "grid_power" in day_det.columns:
        _zoh(ax_g, day_det["timestamp"], day_det["grid_power"], color=_C_OFFLINE, linestyle=_LS_OFFLINE, label="Grid power (offline)")
    _day_viol = (day["p_grid_actual_kw"] > day["access_kw"]).fillna(False)
    if _day_viol.any():
        ax_g.fill_between(
            day["timestamp"],
            day["access_kw"],
            day["p_grid_actual_kw"],
            where=_day_viol,
            color=_C_ONLINE,
            alpha=0.15,
            step="post",
        )
    _zoh(ax_g, day["timestamp"], day["access_kw"], color=_C_ACCESS, linestyle=_LS_CONSTRAINT, label="Access power (online)")
    ax_g.set_ylabel("kW")
    ax_g.set_xlabel("Time")
    ax_g.set_title(f"Grid power — {_day_title}")
    _style_day_xaxis(ax_g)
    _legend_grid_below(fig_g, ax_g)
    plt.show()

    # 2) HP electrical power
    fig_h, ax_h = plt.subplots(figsize=(10, 4.2))
    _shade_stress(ax_h, day["timestamp"], _st_day)
    _zoh(ax_h, day["timestamp"], day["hp_applied_kw"], color=_C_ONLINE, linestyle=_LS_ONLINE, label="HP applied (online)")
    if day_det is not None and not day_det.empty and "hp_kw_deterministic" in day_det.columns:
        _zoh(ax_h, day_det["timestamp"], day_det["hp_kw_deterministic"], color=_C_OFFLINE, linestyle=_LS_OFFLINE, label="HP (offline)")
    ax_h.set_ylabel("kW")
    ax_h.set_xlabel("Time")
    ax_h.set_title(f"HP electrical power — {_day_title}")
    _style_day_xaxis(ax_h)
    _legend_below(fig_h, ax_h)
    plt.show()

    # 3) Buffer SOC
    fig_s, ax_s = plt.subplots(figsize=(10, 4.2))
    _shade_stress(ax_s, day["timestamp"], _st_day)
    _zoh(ax_s, day["timestamp"], day["soc_after"], color=_C_ONLINE, linestyle=_LS_ONLINE, label="SOC (online)")
    if day_det is not None and not day_det.empty and "buffer_soc" in day_det.columns:
        _zoh(ax_s, day_det["timestamp"], day_det["buffer_soc"], color=_C_OFFLINE, linestyle=_LS_OFFLINE, label="SOC (offline)")
    ax_s.set_ylabel("SOC")
    ax_s.set_xlabel("Time")
    ax_s.set_title(f"Buffer SOC — {_day_title}")
    _style_day_xaxis(ax_s)
    _legend_below(fig_s, ax_s)
    plt.show()

    # --- Debug: what the optimizer sees at a specific timestamp (24h MPC window) ---
    # Mirrors the EV debug idea from notebook 09, but for HP.

    from online_MPC_1_HP import _parse_plant_data, _load_forecast_column  # uses sequential alignment
    from optimization import mpc_hp_24h

    plant_df_dbg = _parse_plant_data(PROJECT_ROOT / "data" / "plant1.csv")

    # Load forecast arrays (same logic as the online wrapper)
    forecast_inflex_path = PROJECT_ROOT / "output" / "forecast" / "forecast_inflex_load_rolling_horizon.csv"
    forecast_ev_path = PROJECT_ROOT / "output" / "forecast" / "forecast_ev_rolling_horizon.csv"
    forecast_pv_path = PROJECT_ROOT / "output" / "forecast" / "forecast_pv_rolling_horizon.csv"
    forecast_thermal_path = PROJECT_ROOT / "output" / "forecast" / "forecast_thermal_load_rolling_horizon.csv"
    temperature_forecast_path = PROJECT_ROOT / "data" / "temperature_forecast_day_ahead_open_meteo_Turnhout_15min.csv"

    inflex_fc = _load_forecast_column(
        forecast_inflex_path,
        strategy=INFLEX_FORECAST_STRATEGY,
        prefix="forecast_inflex_",
        hint_prefix="forecast_inflex_",
    )
    ev_fc = _load_forecast_column(
        forecast_ev_path,
        strategy=EV_FORECAST_STRATEGY,
        prefix="forecast_ev_",
        hint_prefix="forecast_ev_",
    )
    thermal_fc = _load_forecast_column(
        forecast_thermal_path,
        strategy=THERMAL_FORECAST_STRATEGY,
        prefix="forecast_thermal_",
        hint_prefix="forecast_thermal_",
    )

    pv_df = pd.read_csv(forecast_pv_path)
    _pv_col = (
        PV_FORECAST_STRATEGY
        if str(PV_FORECAST_STRATEGY).startswith("pv_forecast_kWh_15min_")
        else f"pv_forecast_kWh_15min_{PV_FORECAST_STRATEGY}"
    )
    if _pv_col not in pv_df.columns:
        raise KeyError(f"Missing PV forecast col {_pv_col!r} in {forecast_pv_path.name!r}.")
    pv_fc = pd.to_numeric(pv_df[_pv_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    # Temperature (planner COP input)
    temp_df = pd.read_csv(temperature_forecast_path)
    temp_cols = [c for c in temp_df.columns if c != "timestamp"]
    if not temp_cols:
        raise KeyError(f"No temperature column found in {temperature_forecast_path.name!r}.")
    temp_fc = pd.to_numeric(temp_df[temp_cols[0]], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    if not (len(plant_df_dbg) == len(inflex_fc) == len(ev_fc) == len(pv_fc) == len(thermal_fc) == len(temp_fc)):
        raise ValueError("Plant and forecast arrays must have equal length.")

    plant_df_dbg = plant_df_dbg.copy()
    plant_df_dbg["inflex_forecast"] = inflex_fc
    plant_df_dbg["ev_for_mpc"] = ev_fc
    plant_df_dbg["pv_for_mpc"] = pv_fc
    plant_df_dbg["thermal_forecast"] = thermal_fc
    plant_df_dbg["outdoor_temperature_for_mpc"] = temp_fc

    # Find index k for chosen timestamp
    mask_k = plant_df_dbg["timestamp"] == DEBUG_TS
    if not mask_k.any():
        raise ValueError(f"DEBUG_TS {DEBUG_TS} not found in plant timestamps.")
    k0 = int(plant_df_dbg.index[mask_k][0])

    # Build 24h window exactly as the wrapper passes into mpc_hp_24h
    k_end = min(k0 + 96, len(plant_df_dbg))
    df_window_dbg = plant_df_dbg.loc[
        k0 : k_end - 1,
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
    df_window_dbg.rename(
        columns={
            "pv_for_mpc": "pv_production",
            "inflex_forecast": "inflex_load",
            "ev_for_mpc": "ev",
            "thermal_forecast": "thermal_load",
            "outdoor_temperature_for_mpc": "outdoor_temperature",
        },
        inplace=True,
    )

    # Planner initial SOC = measured SOC at this timestep (after all previous-step post-processing)
    mask_res = pd.to_datetime(res_plot["timestamp"], errors="coerce") == DEBUG_TS
    if not mask_res.any():
        raise ValueError(f"DEBUG_TS {DEBUG_TS} not found in res_hp_online timestamps.")
    soc_init_dbg = float(pd.to_numeric(res_plot.loc[mask_res, "soc_before"], errors="coerce").iloc[0])

    month_key_dbg = DEBUG_TS.to_period("M").strftime("%Y-%m")
    peak_so_far_dbg = float(pd.to_numeric(res_plot.loc[mask_res, "monthly_peak_so_far_kw"], errors="coerce").iloc[0])
    monthly_peak_so_far_dbg = {month_key_dbg: peak_so_far_dbg}

    # Reconstruct rolling-12-month exceedance state from realized history up to DEBUG_TS
    hist = res_plot.copy()
    hist["timestamp"] = pd.to_datetime(hist["timestamp"], errors="coerce")
    hist = hist[hist["timestamp"] < DEBUG_TS].copy()
    hist["month_key"] = hist["timestamp"].dt.to_period("M").astype(str)
    ex_hist = (pd.to_numeric(hist["p_grid_actual_kw"], errors="coerce").fillna(0.0) - pd.to_numeric(hist["access_kw"], errors="coerce").fillna(0.0)).clip(lower=0.0)
    hist["exceed_kw"] = ex_hist

    # Finalized months = months strictly before current month
    finalized = hist[hist["month_key"] < month_key_dbg].groupby("month_key")["exceed_kw"].max().sort_index()
    last12 = finalized.tail(12)
    roll12_completed = float(last12.max()) if len(last12) else 0.0

    curr_month_so_far = float(hist[hist["month_key"] == month_key_dbg]["exceed_kw"].max()) if (hist["month_key"] == month_key_dbg).any() else 0.0
    roll12_so_far_current = max(roll12_completed, curr_month_so_far)

    window_month_keys = sorted({ts.to_period("M").strftime("%Y-%m") for ts in pd.to_datetime(df_window_dbg["timestamp"], errors="coerce")})
    rolling12_so_far_by_month_dbg = {}
    for mk in window_month_keys:
        rolling12_so_far_by_month_dbg[mk] = roll12_so_far_current if mk == month_key_dbg else roll12_completed

    # Solve one 24h MPC window (what the optimizer sees)
    # Include time-varying planner SOC floor if enabled.
    soc_min_profile_dbg = None
    if ENABLE_FORECAST_STRESS_SOC_FLOOR and "soc_min_planner_floor" in res_plot.columns:
        soc_min_profile_dbg = (
            pd.to_numeric(res_plot.loc[k0 : k_end - 1, "soc_min_planner_floor"], errors="coerce")
            .fillna(float(SOC_MIN))
            .to_list()
        )

    print("\n=== MPC window debug ===")
    print("DEBUG_TS:", DEBUG_TS)
    print("k0:", k0)
    print("soc_initial passed to MPC:", soc_init_dbg)
    if soc_min_profile_dbg is not None:
        print(
            f"planner SOC floor in window: min={min(soc_min_profile_dbg):.3f}, max={max(soc_min_profile_dbg):.3f}"
        )

    try:
        if ACCESS_POWER_DICT is None:
            raise NameError(
                "ACCESS_POWER_DICT is not defined (needed only for MPC debug window). "
                "Run Part 1 or skip the debug window section."
            )

        win_res, win_sum = mpc_hp_24h(
            df_window=df_window_dbg,
            config_path=str(PROJECT_ROOT / "config" / "billing.yaml"),
            hp_config_path=str(PROJECT_ROOT / "config" / "hp.yaml"),
            monthly_peak_so_far=monthly_peak_so_far_dbg,
            rolling12_max_exceedance_so_far_by_month=rolling12_so_far_by_month_dbg,
            soc_initial=soc_init_dbg,
            timestamp_col="timestamp",
            pv_col="pv_production",
            inflex_load_col="inflex_load",
            price_col="price",
            ev_col="ev",
            thermal_load_col="thermal_load",
            outdoor_temp_col="outdoor_temperature",
            access_power_by_month=ACCESS_POWER_DICT,
            buffer_soc_min_profile=soc_min_profile_dbg,
        )
        print("months_in_window:", win_sum.get("months_in_window"))
        print("objective_value:", win_sum.get("objective_value"))

        # Plots: inputs and optimizer outputs over the 24h horizon
        win = win_res.copy()
        win["hp_kw"] = 4.0 * pd.to_numeric(win["hp_electrical_input"], errors="coerce").fillna(0.0)

        fig, axes = plt.subplots(4, 1, figsize=(16, 11), sharex=True)
    except Exception as e:
        print("MPC solve failed for debug window:", repr(e))
        # Still plot the inputs + the planner SOC floor schedule for infeasibility diagnosis.
        win = df_window_dbg.copy()
        win["timestamp"] = pd.to_datetime(win["timestamp"], errors="coerce")
        win["hp_kw"] = 0.0
        win["grid_power" ] = 4.0 * (
            pd.to_numeric(win["inflex_load"], errors="coerce").fillna(0.0)
            + pd.to_numeric(win["ev"], errors="coerce").fillna(0.0)
            - pd.to_numeric(win["pv_production"], errors="coerce").fillna(0.0)
        )
        # Provide missing columns expected by the plots below.
        if "spot_price_eur_per_mwh" not in win.columns:
            if "price" in win.columns:
                win["spot_price_eur_per_mwh"] = pd.to_numeric(win["price"], errors="coerce")
            elif "spot_price" in win.columns:
                win["spot_price_eur_per_mwh"] = pd.to_numeric(win["spot_price"], errors="coerce")
            else:
                win["spot_price_eur_per_mwh"] = np.nan
        if "ev_baseline" not in win.columns and "ev" in win.columns:
            win["ev_baseline"] = pd.to_numeric(win["ev"], errors="coerce").fillna(0.0)
        if "hp_thermal_output" not in win.columns:
            win["hp_thermal_output"] = 0.0
        if "buffer_soc" not in win.columns:
            win["buffer_soc"] = np.nan
        for _c in (
            "monthly_peak_plan",
            "effective_peak",
            "access_power_fixed",
            "rolling12_max_exceedance_kw",
            "rolling12_increment_kw",
            "access_overage_kw",
        ):
            if _c not in win.columns:
                win[_c] = np.nan

        # Keep the same axes layout as the success case so the plot code stays shared.
        fig, axes = plt.subplots(4, 1, figsize=(16, 11), sharex=True)

    # 1) Price
    axes[0].plot(
        win["timestamp"],
        pd.to_numeric(win["spot_price_eur_per_mwh"], errors="coerce"),
        label="Price (€/MWh)",
    )
    axes[0].set_ylabel("€/MWh")
    axes[0].set_title("MPC 24h window: price")
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.35)
    axes[0].legend(loc="upper right")

    # 2) Electrical power balance (kW)
    axes[1].plot(win["timestamp"], 4.0 * win["inflex_load"], label="Inflex forecast (kW)", alpha=0.8)
    axes[1].plot(win["timestamp"], 4.0 * win["ev_baseline"], label="EV baseline forecast (kW)", alpha=0.8)
    axes[1].plot(win["timestamp"], 4.0 * win["pv_production"], label="PV forecast (kW)", alpha=0.8)
    axes[1].plot(win["timestamp"], win["hp_kw"], label="HP planned (kW)", lw=1.2)
    axes[1].plot(win["timestamp"], win["grid_power"], label="Grid power (kW)", lw=1.2, color="black", alpha=0.85)
    axes[1].set_ylabel("kW")
    axes[1].set_title("MPC 24h window: electrical powers")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.35)
    axes[1].legend(loc="upper right")

    # 3) Thermal and SOC
    axes[2].plot(win["timestamp"], win["thermal_load"], label="Thermal load forecast (kWh_th/15min)", alpha=0.8)
    axes[2].plot(win["timestamp"], win["hp_thermal_output"], label="HP thermal output (kWh_th/15min)", alpha=0.8)
    ax2b = axes[2].twinx()
    ax2b.plot(win["timestamp"], win["buffer_soc"], label="Buffer SOC", color="tab:green", lw=1.2)

    # Plot planner SOC floor if provided
    if soc_min_profile_dbg is not None:
        ax2b.plot(
            win["timestamp"],
            soc_min_profile_dbg,
            label="Planner SOC floor (applied)",
            color="tab:gray",
            linestyle=":",
            alpha=0.9,
        )

    axes[2].set_ylabel("kWh_th/15min")
    ax2b.set_ylabel("SOC")
    axes[2].set_title("MPC 24h window: thermal balance + SOC")
    axes[2].grid(True, axis="y", linestyle="--", alpha=0.35)
    lines, labels = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    axes[2].legend(lines + lines2, labels + labels2, loc="upper right")

    # 4) Peak / tariff state in the window
    axes[3].plot(win["timestamp"], win["monthly_peak_plan"], label="Monthly peak plan (kW)", alpha=0.9)
    axes[3].plot(win["timestamp"], win["effective_peak"], label="Effective peak (kW)", alpha=0.9)
    axes[3].plot(win["timestamp"], win["access_power_fixed"], label="Access power (kW)", linestyle="--", alpha=0.7)
    if "rolling12_max_exceedance_kw" in win.columns:
        axes[3].plot(win["timestamp"], win["rolling12_max_exceedance_kw"], label="Rolling12 max exceedance (kW)", alpha=0.9)
    if "rolling12_increment_kw" in win.columns:
        axes[3].plot(win["timestamp"], win["rolling12_increment_kw"], label="Rolling12 increment (kW)", alpha=0.9)
    axes[3].set_ylabel("kW")
    axes[3].set_title("MPC 24h window: peak / access / rolling-12 exceedance")
    axes[3].grid(True, axis="y", linestyle="--", alpha=0.35)
    axes[3].legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def run_notebook10_part32_billing_comparison(res_hp_online: pd.DataFrame, project_root: Path) -> None:
    # Part 3.2 — Billing comparison (Notebook 09 style)

    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from billing import load_billing_config, calculate_monthly_bills, calculate_monthly_injection_bills

    PROJECT_ROOT = project_root
    SRC_DIR = PROJECT_ROOT / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.append(str(SRC_DIR))

    NOTEBOOKS_OUTPUT_DIR = PROJECT_ROOT / "output" / "notebooks"

    # -----------------------------
    # Load deterministic exports (notebook 03)
    # -----------------------------
    DET_HP_BILLS_PATH = NOTEBOOKS_OUTPUT_DIR / "deterministic_hp_monthly_bills_notebook_03.csv"
    DET_HP_INJ_PATH = NOTEBOOKS_OUTPUT_DIR / "deterministic_hp_monthly_injection_notebook_03.csv"

    det_bills = None
    det_inj = None
    if DET_HP_BILLS_PATH.exists():
        det_bills = pd.read_csv(DET_HP_BILLS_PATH)
    if DET_HP_INJ_PATH.exists():
        det_inj = pd.read_csv(DET_HP_INJ_PATH)

    if det_bills is None or det_inj is None:
        raise FileNotFoundError(
            "Missing deterministic HP exports from notebook 03. Expected:\n"
            f"- {DET_HP_BILLS_PATH}\n"
            f"- {DET_HP_INJ_PATH}\n"
            "Run notebook 03 export cell first."
        )

    # Ensure month_key for deterministic tables
    for _df in (det_bills, det_inj):
        if "month" in _df.columns:
            _df["month_key"] = _df["month"].astype(str)

    # -----------------------------
    # Billing config
    # -----------------------------
    billing_cfg_path = PROJECT_ROOT / "config" / "billing.yaml"
    billing_cfg = load_billing_config(str(billing_cfg_path))

    # -----------------------------
    # Plant (meter) data
    # -----------------------------
    plant_path = PROJECT_ROOT / "data" / "plant1.csv"
    plant = pd.read_csv(plant_path)
    plant_ts = pd.to_datetime(plant["timestamp"], utc=True, errors="coerce")
    plant["timestamp"] = plant_ts.dt.tz_convert("Europe/Brussels").dt.tz_localize(None)
    plant = plant.sort_values("timestamp").reset_index(drop=True)
    plant = plant[(plant["timestamp"] >= pd.Timestamp("2025-01-01")) & (plant["timestamp"] < pd.Timestamp("2026-01-01"))].copy()

    # -----------------------------
    # Scenario: ONLINE MPC (this notebook)
    # -----------------------------

    res_b = res_hp_online.copy()

    # IMPORTANT: avoid timestamp merges (DST issues). Treat series as sequential.
    plant_seq = plant[["timestamp", "inflex_load", "ev", "pv_production", "price", "outdoor_temperature"]].copy().reset_index(drop=True)
    res_seq = res_b[["hp_applied_kwh", "access_kw"]].copy().reset_index(drop=True)

    n = min(len(plant_seq), len(res_seq))
    if len(plant_seq) != len(res_seq):
        print(
            "WARNING: plant and res_hp_online lengths differ. "
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

    # -----------------------------
    # Scenario: BASELINE (uncontrolled HP)
    # -----------------------------

    uncontrolled_hp_path = PROJECT_ROOT / "output" / "uncontrolled_hp.csv"
    if not uncontrolled_hp_path.exists():
        raise FileNotFoundError(
            f"Baseline uncontrolled HP not found at {uncontrolled_hp_path}. Run notebook 03 uncontrolled HP generation first."
        )

    hp_un = pd.read_csv(uncontrolled_hp_path)

    # Build baseline df: same plant signals + uncontrolled HP kWh
    # Access power heuristic (same as notebook 03):
    # - access_power_conservative[m] = max(observed monthly peak up to m-1) + 20 kW
    # - access_power_hp[m] = access_power_conservative[m] + worst-case HP electrical peak @ COP(-10°C)
    #
    # Avoid timestamp merges (DST issues). Treat series as sequential.
    plant_seq_b = plant[["timestamp", "inflex_load", "ev", "pv_production", "price", "grid_consumption", "thermal_load", "outdoor_temperature"]].copy().reset_index(drop=True)
    hp_seq = hp_un[["hp_electrical_load"]].rename(columns={"hp_electrical_load": "hp_kwh"}).copy().reset_index(drop=True)

    n_b = min(len(plant_seq_b), len(hp_seq))
    if len(plant_seq_b) != len(hp_seq):
        print(
            "WARNING: plant and uncontrolled_hp lengths differ. "
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

    # Month for access power mapping (avoid DST logic; timestamps already naive Brussels)
    _naive = pd.to_datetime(
        df_base["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        format="%Y-%m-%d %H:%M:%S",
    )
    df_base["month"] = _naive.dt.to_period("M")

    # Conservative access power based on baseline grid peaks (no HP), notebook 03 values
    MARGIN_KW = 20.0
    BASELINE_2024_PEAK_GRID_KW = 2663.5
    months_2025 = pd.period_range("2025-01", "2025-12", freq="M")

    monthly_peak_baseline_kw = (
        (df_base.groupby("month")["grid_consumption"].max() * 4.0)
        .reindex(months_2025)
        .fillna(0.0)
    )

    cummax_M_minus_1_kw = monthly_peak_baseline_kw.cummax().shift(1)
    cummax_M_minus_1_kw.loc[months_2025.min()] = BASELINE_2024_PEAK_GRID_KW
    cummax_M_minus_1_kw = cummax_M_minus_1_kw.fillna(BASELINE_2024_PEAK_GRID_KW)

    access_power_conservative = cummax_M_minus_1_kw + MARGIN_KW

    # Worst-case HP additional peak @ COP(-10°C)
    from heat_pump_load import load_hp_config, interpolate_cop

    hp_cfg = load_hp_config(str(PROJECT_ROOT / "config" / "hp.yaml"))
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

    # -----------------------------
    # Scenario: DETERMINISTIC (net)
    # -----------------------------

    det_net = det_bills[["month_key", "total_cost_eur"]].merge(
        det_inj[["month_key", "injection_net_revenue_eur"]],
        on="month_key",
        how="left",
    )
    det_net["deterministic_net_cost_eur"] = (
        det_net["total_cost_eur"] - det_net["injection_net_revenue_eur"].fillna(0.0)
    )

    # -----------------------------
    # Shadow billing table (monthly)
    # -----------------------------

    months = (
        pd.Index(
            sorted(
                set(baseline_bills["month_key"])
                | set(det_bills["month_key"])
                | set(online_bills["month_key"])
            )
        )
        .astype(str)
        .tolist()
    )

    shadow = pd.DataFrame({"month_key": months}).set_index("month_key")
    shadow["baseline_net_cost_eur"] = baseline_net.set_index("month_key")["baseline_net_cost_eur"].reindex(shadow.index).values
    shadow["deterministic_net_cost_eur"] = det_net.set_index("month_key")["deterministic_net_cost_eur"].reindex(shadow.index).values
    shadow["online_net_cost_eur"] = online_net.set_index("month_key")["online_net_cost_eur"].reindex(shadow.index).values

    shadow["deterministic_savings_eur"] = shadow["baseline_net_cost_eur"] - shadow["deterministic_net_cost_eur"]
    shadow["online_savings_eur"] = shadow["baseline_net_cost_eur"] - shadow["online_net_cost_eur"]

    shadow_billing_hp = shadow.reset_index().rename(columns={"month_key": "month"})

    print("Shadow billing monthly comparison (HP)")
    display(
        shadow_billing_hp[[
            "month",
            "baseline_net_cost_eur",
            "deterministic_net_cost_eur",
            "online_net_cost_eur",
            "deterministic_savings_eur",
            "online_savings_eur",
        ]]
    )

    # -----------------------------
    # Component-wise cost comparison (annual)
    # -----------------------------

    def _net_total(bills: pd.DataFrame, inj: pd.DataFrame) -> float:
        return float(bills["total_cost_eur"].sum() - inj["injection_net_revenue_eur"].sum())

    baseline_total = _net_total(baseline_bills, baseline_inj)
    deterministic_total = _net_total(det_bills, det_inj)
    online_total = _net_total(online_bills, online_inj)

    print("\n" + "=" * 80)
    print("COST COMPARISON: BASELINE vs DETERMINISTIC vs ONLINE MPC (HP)")
    print("=" * 80)
    print(f"{'Cost Component':<30} {'Baseline (EUR)':>18} {'Deterministic (EUR)':>20} {'Online (EUR)':>18}")
    print("-" * 90)

    rows = [
        ("Energy Cost", baseline_bills["energy_cost_eur"].sum(), det_bills["energy_cost_eur"].sum(), online_bills["energy_cost_eur"].sum()),
        ("Spot Cost", baseline_bills["spot_cost_eur"].sum(), det_bills["spot_cost_eur"].sum(), online_bills["spot_cost_eur"].sum()),
        ("Access Power Cost", baseline_bills["access_cost_eur"].sum(), det_bills["access_cost_eur"].sum(), online_bills["access_cost_eur"].sum()),
        ("Monthly Peak Cost", baseline_bills["monthly_peak_cost_eur"].sum(), det_bills["monthly_peak_cost_eur"].sum(), online_bills["monthly_peak_cost_eur"].sum()),
        ("Over-usage Cost", baseline_bills["over_usage_cost_eur"].sum(), det_bills["over_usage_cost_eur"].sum(), online_bills["over_usage_cost_eur"].sum()),
        ("Injection Revenue", -baseline_inj["injection_net_revenue_eur"].sum(), -det_inj["injection_net_revenue_eur"].sum(), -online_inj["injection_net_revenue_eur"].sum()),
    ]

    for name, b, d, o in rows:
        print(f"{name:<30} {b:>18,.2f} {d:>20,.2f} {o:>18,.2f}")

    print("-" * 90)
    print(f"{'NET TOTAL':<30} {baseline_total:>18,.2f} {deterministic_total:>20,.2f} {online_total:>18,.2f}")
    print("=" * 80)

    # -----------------------------
    # Plot: HP thermal energy delivered (to load) per month
    # -----------------------------
    # Baseline (uncontrolled): reconstruct thermal output from hp_kwh * COP(actual outdoor temp)
    # Online: same but with hp_applied_kwh
    # Deterministic: use exported hp_thermal_output if available; fallback to hp_electrical_input * COP

    # COP series (sequential, same length as plant)
    from heat_pump_load import load_hp_config, interpolate_cop
    hp_cfg2 = load_hp_config(str(PROJECT_ROOT / "config" / "hp.yaml"))

    def _cop_series(outdoor_temp_s: pd.Series) -> pd.Series:
        t = pd.to_numeric(outdoor_temp_s, errors="coerce")
        return t.apply(lambda x: float(interpolate_cop(float(x), hp_cfg2["COP_data"])) if not np.isnan(x) else 2.5)

    # Baseline thermal out (kWh_th/15min)
    base_temp = pd.to_numeric(plant_seq_b["outdoor_temperature"], errors="coerce")
    base_cop = _cop_series(base_temp)
    base_th_out_kwh = (pd.to_numeric(hp_seq["hp_kwh"], errors="coerce").fillna(0.0) * base_cop).fillna(0.0)

    # Online thermal out (kWh_th/15min)
    on_temp = pd.to_numeric(plant_seq["outdoor_temperature"], errors="coerce") if "plant_seq" in globals() else pd.to_numeric(plant_seq_b["outdoor_temperature"], errors="coerce")
    on_cop = _cop_series(on_temp)
    on_hp_kwh = pd.to_numeric(df_online["hp_applied_kwh"], errors="coerce").fillna(0.0)
    on_th_out_kwh = (on_hp_kwh * on_cop).fillna(0.0)

    # Deterministic thermal out (kWh_th/15min)
    # Try to load deterministic 15-min export if available (from Part 3.1 import)
    det_th_out_kwh = None
    try:
        det_15_path = NOTEBOOKS_OUTPUT_DIR / "deterministic_hp_15min_notebook_03.csv"
        if det_15_path.exists():
            det_15 = pd.read_csv(det_15_path)
            if "hp_thermal_output" in det_15.columns:
                det_th_out_kwh = pd.to_numeric(det_15["hp_thermal_output"], errors="coerce").fillna(0.0)
            elif "hp_electrical_input" in det_15.columns and "outdoor_temperature" in det_15.columns:
                det_cop = _cop_series(det_15["outdoor_temperature"])
                det_th_out_kwh = pd.to_numeric(det_15["hp_electrical_input"], errors="coerce").fillna(0.0) * det_cop
    except Exception as e:
        print(f"NOTE: could not compute deterministic thermal output series: {e}")

    # Monthly aggregation (use plant months; sequential)
    months_plot = shadow_billing_hp["month"].astype(str).tolist()

    base_month = df_base["month"].astype(str)
    on_month = pd.to_datetime(df_online["timestamp"]).dt.to_period("M").astype(str)

    base_th_mwh = (pd.DataFrame({"month": base_month, "th_kwh": base_th_out_kwh}).groupby("month")["th_kwh"].sum() / 1000.0)
    on_th_mwh = (pd.DataFrame({"month": on_month, "th_kwh": on_th_out_kwh}).groupby("month")["th_kwh"].sum() / 1000.0)

    if det_th_out_kwh is not None:
        # deterministic months from its timestamps
        det_15["timestamp"] = pd.to_datetime(det_15["timestamp"], errors="coerce")
        det_month = det_15["timestamp"].dt.to_period("M").astype(str)
        det_th_mwh = (pd.DataFrame({"month": det_month, "th_kwh": det_th_out_kwh}).groupby("month")["th_kwh"].sum() / 1000.0)
    else:
        det_th_mwh = pd.Series(dtype=float)

    # Plot bars
    x = np.arange(len(months_plot))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width, base_th_mwh.reindex(months_plot).values, width, label="Baseline thermal out (MWh_th)", color="red", alpha=0.7)
    ax.bar(x, det_th_mwh.reindex(months_plot).values if len(det_th_mwh) else np.zeros(len(months_plot)), width, label="Deterministic thermal out (MWh_th)", color="green", alpha=0.7)
    ax.bar(x + width, on_th_mwh.reindex(months_plot).values, width, label="Online thermal out (MWh_th)", color="blue", alpha=0.7)
    ax.set_xlabel("Month")
    ax.set_ylabel("Thermal delivered (MWh_th)")
    ax.set_title("HP thermal energy output per month (not load served)")
    ax.set_xticks(x)
    ax.set_xticklabels(months_plot, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    plt.tight_layout()
    plt.show()

    print("\nHP thermal output totals (2025):")
    print(f"- Baseline: {float(base_th_mwh.sum()):,.2f} MWh_th")
    if len(det_th_mwh):
        print(f"- Deterministic: {float(det_th_mwh.sum()):,.2f} MWh_th")
    print(f"- Online: {float(on_th_mwh.sum()):,.2f} MWh_th")

    # Thermal load (demand) totals from plant (same for all scenarios unless unmet)
    thermal_demand_mwh = float(pd.to_numeric(plant["thermal_load"], errors="coerce").fillna(0.0).sum() / 1000.0)
    print(f"\nThermal load demand total (2025, from plant): {thermal_demand_mwh:,.2f} MWh_th")

    # -----------------------------
    # Plot: monthly peak + access power (Notebook 09 style)
    # -----------------------------

    base_idx = baseline_bills.set_index("month_key")
    det_idx = det_bills.set_index("month_key")
    on_idx = online_bills.set_index("month_key")

    months_plot = shadow_billing_hp["month"].astype(str).tolist()
    x = np.arange(len(months_plot))
    width = 0.25

    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    ax1 = axes[0]
    ax1.bar(x - width, base_idx.reindex(months_plot)["monthly_peak_kw"].values, width, label="Baseline monthly peak", color="red", alpha=0.7)
    ax1.bar(x, det_idx.reindex(months_plot)["monthly_peak_kw"].values, width, label="Deterministic monthly peak", color="green", alpha=0.7)
    ax1.bar(x + width, on_idx.reindex(months_plot)["monthly_peak_kw"].values, width, label="Online monthly peak", color="blue", alpha=0.7)
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Monthly peak (kW)")
    ax1.set_title("Monthly peak power: baseline vs deterministic vs online (HP)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(months_plot, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    ax2 = axes[1]
    ax2.bar(x - width, base_idx.reindex(months_plot)["access_power_kw"].values, width, label="Baseline access power", color="orange", alpha=0.7)
    ax2.bar(x, det_idx.reindex(months_plot)["access_power_kw"].values, width, label="Deterministic access power", color="tab:green", alpha=0.7)
    ax2.bar(x + width, on_idx.reindex(months_plot)["access_power_kw"].values, width, label="Online access power", color="tab:blue", alpha=0.7)
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Access power (kW)")
    ax2.set_title("Access power: baseline vs deterministic vs online (HP)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(months_plot, rotation=45, ha="right")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Savings breakdown vs baseline (stacked bars, Notebook 09 style)
    # -----------------------------

    base_spot = base_idx.reindex(months_plot)["spot_cost_eur"].values
    base_access = base_idx.reindex(months_plot)["access_cost_eur"].values
    base_peak = (
        base_idx.reindex(months_plot)["monthly_peak_cost_eur"].values
        + base_idx.reindex(months_plot)["over_usage_cost_eur"].values
    )

    det_spot = det_idx.reindex(months_plot)["spot_cost_eur"].values
    det_access = det_idx.reindex(months_plot)["access_cost_eur"].values
    det_peak = (
        det_idx.reindex(months_plot)["monthly_peak_cost_eur"].values
        + det_idx.reindex(months_plot)["over_usage_cost_eur"].values
    )

    on_spot = on_idx.reindex(months_plot)["spot_cost_eur"].values
    on_access = on_idx.reindex(months_plot)["access_cost_eur"].values
    on_peak = (
        on_idx.reindex(months_plot)["monthly_peak_cost_eur"].values
        + on_idx.reindex(months_plot)["over_usage_cost_eur"].values
    )

    savings = pd.DataFrame({
        "month": months_plot,
        "spot_savings_det": base_spot - det_spot,
        "access_power_savings_det": base_access - det_access,
        "peak_savings_det": base_peak - det_peak,
        "spot_savings_online": base_spot - on_spot,
        "access_power_savings_online": base_access - on_access,
        "peak_savings_online": base_peak - on_peak,
    })

    savings["total_savings_det"] = savings[["spot_savings_det", "access_power_savings_det", "peak_savings_det"]].sum(axis=1)
    savings["total_savings_online"] = savings[["spot_savings_online", "access_power_savings_online", "peak_savings_online"]].sum(axis=1)

    x = np.arange(len(savings))
    width = 0.35

    fig, ax = plt.subplots(figsize=(16, 6))

    ax.bar(x - width / 2, savings["spot_savings_det"], width, label="Deterministic — Spot", color="tab:blue", alpha=0.8)
    ax.bar(x - width / 2, savings["access_power_savings_det"], width, bottom=savings["spot_savings_det"], label="Deterministic — Access power", color="tab:green", alpha=0.8)
    ax.bar(
        x - width / 2,
        savings["peak_savings_det"],
        width,
        bottom=savings["spot_savings_det"] + savings["access_power_savings_det"],
        label="Deterministic — Peak",
        color="tab:orange",
        alpha=0.8,
    )

    ax.bar(x + width / 2, savings["spot_savings_online"], width, label="Online — Spot", color="tab:blue", alpha=0.4)
    ax.bar(x + width / 2, savings["access_power_savings_online"], width, bottom=savings["spot_savings_online"], label="Online — Access power", color="tab:green", alpha=0.4)
    ax.bar(
        x + width / 2,
        savings["peak_savings_online"],
        width,
        bottom=savings["spot_savings_online"] + savings["access_power_savings_online"],
        label="Online — Peak",
        color="tab:orange",
        alpha=0.4,
    )

    ax.set_xlabel("Month")
    ax.set_ylabel("Savings vs baseline (EUR)")
    ax.set_title("Monthly savings breakdown vs baseline: deterministic vs online (HP)")
    ax.set_xticks(x)
    ax.set_xticklabels(savings["month"], rotation=45, ha="right")
    ax.legend(loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()

    print("\nSavings breakdown (first rows):")
    display(
        savings[[
            "month",
            "spot_savings_det",
            "access_power_savings_det",
            "peak_savings_det",
            "total_savings_det",
            "spot_savings_online",
            "access_power_savings_online",
            "peak_savings_online",
            "total_savings_online",
        ]].head()
    )


def _decode_strength_tag(tag: str) -> str:
    """Filename tag from notebook 10 (e.g. 0p75) -> plot label (0.75)."""
    s = str(tag).strip()
    if "p" in s and not s.startswith("p"):
        return s.replace("p", ".", 1)
    return s


def _scenario_axis_label(
    scenario_id,
    scenario_name: str = "",
    scenario_kwargs_json: str | None = None,
) -> str:
    """Compact assessor-friendly x-axis label, e.g. ``S4 p50 soc floor 0.75``."""
    sid = int(scenario_id) if str(scenario_id).strip().lstrip("-").isdigit() else scenario_id
    prefix = f"S{sid}"

    kw = None
    if scenario_kwargs_json is not None and str(scenario_kwargs_json).strip():
        try:
            kw = json.loads(scenario_kwargs_json)
        except json.JSONDecodeError:
            kw = None

    if kw is not None:
        if not bool(kw.get("enable_forecast_stress_soc_floor", False)):
            return f"{prefix} no stress floor"
        qraw = str(kw.get("forecast_strategy_inflex_stress", "c_p50"))
        q = qraw.replace("c_", "") if qraw.startswith("c_") else qraw
        if not q.startswith("p"):
            q = f"p{q}"
        floor_s = f"{float(kw.get('forecast_stress_soc_floor_strength', 0.5)):g}"
        return f"{prefix} {q} soc floor {floor_s}"

    name = str(scenario_name).strip()
    if "base_no_forecast" in name or name.startswith("base_"):
        return f"{prefix} no stress floor"

    m = re.match(r"inflexstress_(p\d+)_stressfloor_([^_]+)_mult_\d+", name)
    if m:
        q = m.group(1)
        floor_s = _decode_strength_tag(m.group(2))
        return f"{prefix} {q} soc floor {floor_s}"

    short = name.replace("_", " ")
    return f"{prefix} {short}" if short else prefix


def run_notebook10_part4d_scenario_savings(project_root: Path) -> None:
    """Part 4D — annual savings vs baseline (thesis-style plots for assessors)."""
    import sys

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from billing import calculate_monthly_bills, calculate_monthly_injection_bills, load_billing_config

    _main = sys.modules.get("__main__")
    _main_ns = vars(_main) if _main is not None else {}
    access_baseline = _main_ns.get("ACCESS_POWER_BASELINE_MONTHLY")
    if access_baseline is None:
        raise RuntimeError("Run §1.2 first to build ACCESS_POWER_BASELINE_MONTHLY.")

    PROJECT_ROOT = project_root
    NOTEBOOKS_OUTPUT_DIR = PROJECT_ROOT / "output" / "notebooks"
    SUMMARY_CSV = NOTEBOOKS_OUTPUT_DIR / "online_hp_scenario_analysis_summary_notebook_10.csv"
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Run Part 4B first. Missing {SUMMARY_CSV}")

    df_sum = pd.read_csv(SUMMARY_CSV)
    if df_sum.empty:
        raise ValueError("Scenario summary CSV is empty.")

    billing_cfg = load_billing_config(str(PROJECT_ROOT / "config" / "billing.yaml"))

    plant_path = PROJECT_ROOT / "data" / "plant1.csv"
    plant = pd.read_csv(plant_path)
    plant_ts = pd.to_datetime(plant["timestamp"], utc=True, errors="coerce")
    plant["timestamp"] = plant_ts.dt.tz_convert("Europe/Brussels").dt.tz_localize(None)
    plant = plant.sort_values("timestamp").reset_index(drop=True)
    plant = plant[
        (plant["timestamp"] >= pd.Timestamp("2025-01-01"))
        & (plant["timestamp"] < pd.Timestamp("2026-01-01"))
    ].copy()

    uncontrolled_hp_path = PROJECT_ROOT / "output" / "uncontrolled_hp.csv"
    if not uncontrolled_hp_path.exists():
        raise FileNotFoundError(
            f"Baseline uncontrolled HP not found at {uncontrolled_hp_path}. "
            "Run notebook 03 uncontrolled HP generation first."
        )

    hp_un = pd.read_csv(uncontrolled_hp_path)
    plant_seq_b = plant[
        ["timestamp", "inflex_load", "ev", "pv_production", "price", "grid_consumption", "thermal_load", "outdoor_temperature"]
    ].copy().reset_index(drop=True)
    hp_seq = hp_un[["hp_electrical_load"]].rename(columns={"hp_electrical_load": "hp_kwh"}).copy().reset_index(drop=True)
    n_b = min(len(plant_seq_b), len(hp_seq))
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
    _naive = pd.to_datetime(df_base["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"), format="%Y-%m-%d %H:%M:%S")
    df_base["month"] = _naive.dt.to_period("M")
    df_base["month_key"] = df_base["month"].astype(str)
    df_base["access_kw"] = df_base["month_key"].map(access_baseline.to_dict()).astype(float)
    df_base["grid_consumption"] = df_base["grid_consumption_with_hp"]
    df_base["grid_injection"] = df_base["grid_injection_with_hp"]

    a_base = calculate_monthly_bills(
        df_base, billing_cfg,
        volume_col="grid_consumption", price_col="price", timestamp_col="timestamp", access_power_col="access_kw",
    )
    inj_base = calculate_monthly_injection_bills(
        df_base, billing_cfg,
        injection_col="grid_injection", price_col="price", timestamp_col="timestamp",
    )

    base_energy = float(a_base["energy_cost_eur"].sum())
    base_spot = float(a_base["spot_cost_eur"].sum())
    base_access = float(a_base["access_cost_eur"].sum())
    base_monthly_peak = float(a_base["monthly_peak_cost_eur"].sum())
    base_over_usage = float(a_base["over_usage_cost_eur"].sum())
    base_inj_rev = float(inj_base["injection_net_revenue_eur"].sum())
    baseline_net = float(a_base["total_cost_eur"].sum() - base_inj_rev)

    DET_HP_BILLS_PATH = NOTEBOOKS_OUTPUT_DIR / "deterministic_hp_monthly_bills_notebook_03.csv"
    DET_HP_INJ_PATH = NOTEBOOKS_OUTPUT_DIR / "deterministic_hp_monthly_injection_notebook_03.csv"
    if not DET_HP_BILLS_PATH.exists() or not DET_HP_INJ_PATH.exists():
        raise FileNotFoundError("Missing deterministic HP exports from notebook 03.")

    det_bills = pd.read_csv(DET_HP_BILLS_PATH)
    det_inj = pd.read_csv(DET_HP_INJ_PATH)
    det_energy = float(det_bills["energy_cost_eur"].sum())
    det_spot = float(det_bills["spot_cost_eur"].sum())
    det_access = float(det_bills["access_cost_eur"].sum())
    det_monthly_peak = float(det_bills["monthly_peak_cost_eur"].sum())
    det_over_usage = float(det_bills["over_usage_cost_eur"].sum())
    det_inj_rev = float(det_inj["injection_net_revenue_eur"].sum())
    deterministic_net = float(det_bills["total_cost_eur"].sum() - det_inj_rev)

    def _online_components_from_results_csv(results_csv_path: Path) -> dict:
        res = pd.read_csv(results_csv_path)
        plant_seq = plant[["timestamp", "inflex_load", "ev", "pv_production", "price", "outdoor_temperature"]].copy().reset_index(drop=True)
        res_seq = res[["hp_applied_kwh", "access_kw"]].copy().reset_index(drop=True)
        n = min(len(plant_seq), len(res_seq))
        df_online = pd.concat([plant_seq.iloc[:n].copy(), res_seq.iloc[:n].copy()], axis=1)
        net_kwh = (
            df_online["inflex_load"].fillna(0.0)
            + df_online["ev"].fillna(0.0)
            + df_online["hp_applied_kwh"].fillna(0.0)
            - df_online["pv_production"].fillna(0.0)
        )
        df_online["grid_consumption"] = net_kwh.clip(lower=0.0)
        df_online["grid_injection"] = (-net_kwh).clip(lower=0.0)
        bills = calculate_monthly_bills(
            df_online, billing_cfg,
            volume_col="grid_consumption", price_col="price", timestamp_col="timestamp", access_power_col="access_kw",
        )
        inj = calculate_monthly_injection_bills(
            df_online, billing_cfg,
            injection_col="grid_injection", price_col="price", timestamp_col="timestamp",
        )
        return {
            "energy": float(bills["energy_cost_eur"].sum()),
            "spot": float(bills["spot_cost_eur"].sum()),
            "access": float(bills["access_cost_eur"].sum()),
            "monthly_peak": float(bills["monthly_peak_cost_eur"].sum()),
            "over_usage": float(bills["over_usage_cost_eur"].sum()),
            "inj_rev": float(inj["injection_net_revenue_eur"].sum()),
            "net": float(bills["total_cost_eur"].sum() - float(inj["injection_net_revenue_eur"].sum())),
        }

    labels = ["Offline"]
    energy_sav = [base_energy - det_energy]
    spot_sav = [base_spot - det_spot]
    access_sav = [base_access - det_access]
    monthly_peak_sav = [base_monthly_peak - det_monthly_peak]
    over_usage_sav = [base_over_usage - det_over_usage]
    inj_rev_sav = [det_inj_rev - base_inj_rev]
    rows_print = [("Baseline", baseline_net, 0.0), ("Offline", deterministic_net, baseline_net - deterministic_net)]

    for _, row in df_sum.iterrows():
        err_val = row.get("error", "")
        if (not pd.isna(err_val)) and bool(str(err_val).strip()):
            continue
        res_path = Path(str(row["results_15min_path"]))
        if not res_path.exists():
            res_path = NOTEBOOKS_OUTPUT_DIR / Path(str(row["results_15min_path"])).name
        if not res_path.exists():
            print(f"WARNING: scenario results CSV not found: {row['results_15min_path']}")
            continue
        comps = _online_components_from_results_csv(res_path)
        sid = row.get("scenario_id", "")
        name = str(row.get("scenario_name", sid))
        ax_label = _scenario_axis_label(
            sid,
            name,
            row.get("scenario_kwargs_json"),
        )
        labels.append(ax_label)
        energy_sav.append(base_energy - comps["energy"])
        spot_sav.append(base_spot - comps["spot"])
        access_sav.append(base_access - comps["access"])
        monthly_peak_sav.append(base_monthly_peak - comps["monthly_peak"])
        over_usage_sav.append(base_over_usage - comps["over_usage"])
        inj_rev_sav.append(comps["inj_rev"] - base_inj_rev)
        rows_print.append((ax_label, comps["net"], baseline_net - comps["net"]))

    net_costs = [float(deterministic_net)]
    for _name, net, _sav in rows_print[2:]:
        net_costs.append(float(net))
    net_costs = net_costs[: len(labels)]
    total_savings = baseline_net - np.array(net_costs, dtype=float)

    comp_series = {
        "Energy": np.array(energy_sav, dtype=float),
        "Spot": np.array(spot_sav, dtype=float),
        "Access power": np.array(access_sav, dtype=float),
        "Monthly peak": np.array(monthly_peak_sav, dtype=float),
        "Over-usage": np.array(over_usage_sav, dtype=float),
        "Injection revenue": np.array(inj_rev_sav, dtype=float),
    }
    table = pd.DataFrame({**comp_series, "Total savings": total_savings}, index=labels).T

    # --- Thesis style (match §1.2 / Part 3.2 monthly peak plots) ---
    mpl.rcParams.update(
        {
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
    )
    _C_BLACK = "#000000"
    _C_KUL_RED = "#b30000"
    _C_GRAY = "#666666"
    _C_LIGHT = "#aaaaaa"
    _C_INJ = "#2166ac"
    mpl.rcParams["hatch.color"] = _C_BLACK

    x = np.arange(len(labels))
    n = len(labels)
    _bar_w = 0.72
    fig_w = max(12.0, 0.9 * n)
    _xlabel_fs = 8 if n > 12 else 9

    # Figure 1: total annual net savings vs baseline
    fig1, ax1 = plt.subplots(figsize=(fig_w, 5))
    ax1.set_axisbelow(True)
    for i, sav in enumerate(total_savings):
        if i == 0:
            ax1.bar(
                i,
                sav,
                width=_bar_w,
                facecolor="white",
                edgecolor=_C_BLACK,
                hatch="///",
                linewidth=0.8,
                zorder=3,
            )
        else:
            ax1.bar(
                i,
                sav,
                width=_bar_w,
                color=_C_KUL_RED,
                alpha=0.55,
                edgecolor=_C_KUL_RED,
                linewidth=0.6,
                zorder=3,
            )
    ax1.axhline(0.0, color=_C_BLACK, linewidth=0.8, zorder=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=90, ha="center")
    ax1.set_xlim(-0.5, n - 0.5)
    ax1.margins(x=0)
    ax1.tick_params(axis="x", pad=4, labelsize=_xlabel_fs)
    ax1.set_ylabel("Savings vs baseline [EUR/year]")
    ax1.set_xlabel("Scenario", labelpad=6)
    ax1.set_title("Annual net savings vs baseline (2025)")
    ax1.yaxis.grid(True, alpha=0.25, linewidth=0.6)
    ax1.xaxis.grid(False)
    fig1.subplots_adjust(bottom=0.32, top=0.90)
    plt.show()

    # Figure 2: savings split by cost component (stacked)
    stack_spec = [
        ("Energy", _C_GRAY),
        ("Spot", _C_BLACK),
        ("Access power", _C_KUL_RED),
        ("Monthly peak", _C_LIGHT),
        ("Over-usage", "#cccccc"),
        ("Injection revenue", _C_INJ),
    ]
    fig2, ax2 = plt.subplots(figsize=(fig_w, 5))
    ax2.set_axisbelow(True)
    bottom = np.zeros(n, dtype=float)
    for comp_name, comp_color in stack_spec:
        vals = comp_series[comp_name]
        ax2.bar(
            x,
            vals,
            bottom=bottom,
            width=_bar_w,
            label=comp_name,
            color=comp_color,
            edgecolor=_C_BLACK,
            linewidth=0.35,
        )
        bottom = bottom + vals
    ax2.axhline(0.0, color=_C_BLACK, linewidth=0.8, zorder=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=90, ha="center")
    ax2.set_xlim(-0.5, n - 0.5)
    ax2.margins(x=0)
    ax2.tick_params(axis="x", pad=4, labelsize=_xlabel_fs)
    ax2.set_ylabel("Savings vs baseline [EUR/year]")
    ax2.set_xlabel("Scenario", labelpad=6)
    ax2.set_title("Annual savings by cost component vs baseline (2025)")
    ax2.yaxis.grid(True, alpha=0.25, linewidth=0.6)
    ax2.xaxis.grid(False)
    ax2.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=3,
        frameon=False,
    )
    fig2.subplots_adjust(bottom=0.42, top=0.90)
    plt.show()

    print("\nScenarios (annual net cost + savings vs baseline):")
    for name, net, sav in rows_print:
        print(f"- {name}: net cost = {net:,.0f} EUR, savings = {sav:,.0f} EUR")

    print("\nSavings vs baseline [EUR/year] — full component table:")
    display(table.style.format("{:,.0f}"))


def run_notebook10_forecast_stress_hours_by_quantile(project_root: Path) -> None:
    """
    Annual forecast-stress hours for inflex quantiles p50–p99.

    Stress flag (same as `run_hp_online_mpc_1` pre-pass):
      P_forecast_grid = 4*(inflex_stress + EV_fc + HP_est - PV_fc) > access_kw(month)
    with flex-aware access from §1.2 (`ACCESS_POWER_DICT` when present).
    Reports stress hours and share of 8760 h; thesis-style bar chart of %.
    """
    import sys as _sys

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from heat_pump_load import interpolate_cop, load_hp_config
    from online_MPC_1_HP import _load_forecast_column, _month_key, _parse_plant_data

    HOURS_YEAR = 8760.0
    DT_H = 0.25
    QUANTILES = ("p50", "p90", "p95", "p99")

    PROJECT_ROOT = project_root
    _main = _sys.modules.get("__main__")
    _main_ns = vars(_main) if _main is not None else {}

    EV_FORECAST_STRATEGY = _main_ns.get("EV_FORECAST_STRATEGY", "c_p50")
    THERMAL_FORECAST_STRATEGY = _main_ns.get("THERMAL_FORECAST_STRATEGY", "c2t_p50")
    PV_FORECAST_STRATEGY = _main_ns.get("PV_FORECAST_STRATEGY", "chronos2_elia_p50")
    ACCESS_POWER_DICT = _main_ns.get("ACCESS_POWER_DICT", None)
    MARGIN_KW = float(_main_ns.get("MARGIN_KW", 20.0))

    plant_df = _parse_plant_data(PROJECT_ROOT / "data" / "plant1.csv")
    n = len(plant_df)

    if ACCESS_POWER_DICT is None:
        months = pd.period_range("2025-01", "2025-12", freq="M")
        plant_2025 = plant_df.loc[plant_df["timestamp"].dt.year == 2025].copy()
        plant_2025["month"] = plant_2025["timestamp"].dt.to_period("M")
        monthly_peak_2025_grid_kw = (
            plant_2025.groupby("month")["grid_consumption"].max() * 4.0
        ).reindex(months).astype(float)
        train_2024 = pd.read_csv(PROJECT_ROOT / "data" / "plant1_2024_training.csv")
        train_2024["month"] = pd.PeriodIndex(
            train_2024["timestamp"].astype(str).str.slice(0, 7), freq="M"
        )
        train_2024["grid_consumption"] = pd.to_numeric(
            train_2024["grid_consumption"], errors="coerce"
        ).fillna(0.0)
        baseline_2024_peak_grid_kw = float(
            (train_2024.groupby("month")["grid_consumption"].max() * 4.0).max()
        )
        cummax_grid_Mm1_kw = monthly_peak_2025_grid_kw.cummax().shift(1)
        cummax_grid_Mm1_kw.loc[months.min()] = baseline_2024_peak_grid_kw
        cummax_grid_Mm1_kw = cummax_grid_Mm1_kw.fillna(baseline_2024_peak_grid_kw)
        access_power_flex_aware_kw = cummax_grid_Mm1_kw + MARGIN_KW
        ACCESS_POWER_DICT = {
            str(k): float(v) for k, v in access_power_flex_aware_kw.items()
        }
        print(
            "NOTE: ACCESS_POWER_DICT not in kernel — rebuilt flex-aware access from plant1.csv."
        )

    access_kw_full = np.array(
        [
            float(ACCESS_POWER_DICT[_month_key(ts)])
            for ts in plant_df["timestamp"]
        ],
        dtype=float,
    )

    forecast_dir = PROJECT_ROOT / "output" / "forecast"
    inflex_path = forecast_dir / "forecast_inflex_load_rolling_horizon.csv"
    ev_path = forecast_dir / "forecast_ev_rolling_horizon.csv"
    thermal_path = forecast_dir / "forecast_thermal_load_rolling_horizon.csv"
    pv_path = forecast_dir / "forecast_pv_rolling_horizon.csv"
    temp_path = (
        PROJECT_ROOT / "data" / "temperature_forecast_day_ahead_open_meteo_Turnhout_15min.csv"
    )

    ev_fc = _load_forecast_column(
        ev_path, EV_FORECAST_STRATEGY, "forecast_ev_", "forecast_ev_"
    )
    thermal_fc = _load_forecast_column(
        thermal_path,
        THERMAL_FORECAST_STRATEGY,
        "forecast_thermal_",
        "forecast_thermal_",
    )
    pv_col = (
        PV_FORECAST_STRATEGY
        if str(PV_FORECAST_STRATEGY).startswith("pv_forecast_kWh_15min_")
        else f"pv_forecast_kWh_15min_{PV_FORECAST_STRATEGY}"
    )
    pv_fc = (
        pd.to_numeric(pd.read_csv(pv_path)[pv_col], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    if len(pv_fc) != n:
        raise ValueError("PV forecast length must match plant data.")

    temp_df = pd.read_csv(temp_path)
    temp_cols = [c for c in temp_df.columns if c != "timestamp"]
    if not temp_cols:
        raise KeyError(f"No temperature column in {temp_path.name}")
    temp_fc = (
        pd.to_numeric(temp_df[temp_cols[0]], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    if len(temp_fc) != n:
        raise ValueError("Temperature forecast length must match plant data.")

    hp_cfg = load_hp_config(str(PROJECT_ROOT / "config" / "hp.yaml"))
    cop_fc = np.array(
        [
            float(interpolate_cop(float(t), hp_cfg["COP_data"]))
            if not np.isnan(t)
            else 2.5
            for t in temp_fc
        ],
        dtype=float,
    )
    hp_est_kwh = np.where(cop_fc > 1e-9, thermal_fc / cop_fc, 0.0)
    ev_fc_kwh = np.asarray(ev_fc, dtype=float)
    pv_fc_kwh = np.asarray(pv_fc, dtype=float)

    rows = []
    for q in QUANTILES:
        inflex_stress = _load_forecast_column(
            inflex_path, f"c_{q}", "forecast_inflex_", "forecast_inflex_"
        )
        forecast_grid_kw = 4.0 * (
            np.asarray(inflex_stress, dtype=float) + ev_fc_kwh + hp_est_kwh - pv_fc_kwh
        )
        stress_mask = forecast_grid_kw > access_kw_full
        n_steps = int(stress_mask.sum())
        stress_h = float(n_steps) * DT_H
        stress_pct = 100.0 * stress_h / HOURS_YEAR
        rows.append(
            {
                "inflex_quantile": q,
                "stress_steps_15min": n_steps,
                "stress_hours": stress_h,
                "stress_pct_of_8760": stress_pct,
            }
        )

    table = pd.DataFrame(rows)
    print("\nForecast access exceedance — inflex stress quantile sweep (2025)")
    print(
        "Definition: 4*(inflex_q + EV_p50 + HP_est_p50 - PV_p50) > flex-aware access_kw; "
        f"EV={EV_FORECAST_STRATEGY}, thermal={THERMAL_FORECAST_STRATEGY}, PV={PV_FORECAST_STRATEGY}"
    )
    display(
        table.style.format(
            {
                "stress_hours": "{:.2f}",
                "stress_pct_of_8760": "{:.3f}",
            }
        )
    )

    # --- Thesis-style bar chart (% of year) ---
    mpl = plt
    mpl.rcParams.update(
        {
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
            "savefig.dpi": 300,
        }
    )
    _C_BLACK = "#000000"
    _C_KUL_RED = "#b30000"

    labels = [f"p{q[1:]}" if q.startswith("p") else q for q in QUANTILES]
    pct_vals = table["stress_pct_of_8760"].to_numpy(dtype=float)
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.set_axisbelow(True)
    bars = ax.bar(
        x,
        pct_vals,
        width=0.55,
        color=_C_KUL_RED,
        alpha=0.55,
        edgecolor=_C_KUL_RED,
        linewidth=0.6,
        zorder=3,
    )
    ax.axhline(0.0, color=_C_BLACK, linewidth=0.8, zorder=2)
    for bar, pct, hrs in zip(bars, pct_vals, table["stress_hours"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.08,
            f"{pct:.2f}%\n({hrs:.1f} h)",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.margins(x=0)
    ax.set_ylabel("Stress hours [% of 8760 h]")
    ax.set_xlabel("Inflex load forecast quantile (stress detection)")
    ax.set_title("Forecast access exceedance — share of year (2025)")
    ax.yaxis.grid(True, alpha=0.25, linewidth=0.6)
    ax.xaxis.grid(False)
    ymax = float(np.max(pct_vals)) if len(pct_vals) else 1.0
    ax.set_ylim(0.0, ymax * 1.25 + 0.5)
    fig.subplots_adjust(top=0.90, bottom=0.14)
    plt.show()
