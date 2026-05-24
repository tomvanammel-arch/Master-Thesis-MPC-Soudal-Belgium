# §3.1 — Joint online optimised volumes (plots)

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path("..").resolve()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from notebook_visualisation.nb11_evhp_day_replay import (
    plot_day_replay,
    print_day_replay_report,
    summarize_day,
    vlines_mpc_deadline,
)
from notebook_visualisation import nb11_evhp_thesis_plots as _thesis_plt
NOTEBOOKS_OUTPUT_DIR = PROJECT_ROOT / "output" / "notebooks"
RESULTS_15MIN_PATH = NOTEBOOKS_OUTPUT_DIR / "online_ev_hp_15min_notebook_11_part2.csv"
DET_EV_HP_15MIN_PATH = NOTEBOOKS_OUTPUT_DIR / "deterministic_ev_hp_15min_notebook_04.csv"
UNCONTROLLED_HP_PATH = PROJECT_ROOT / "output" / "uncontrolled_hp.csv"

# --- Plot knobs ---
# Part 4C injects ``_NB11_PART31_KNOBS`` (no default dates). Direct §3.1: set WEEK_START / DAY_OF_WEEK
# in this cell above this block before running.
_knobs = globals().pop("_NB11_PART31_KNOBS", None)
if _knobs is not None:
    WEEK_START = pd.Timestamp(_knobs["WEEK_START"])
    DAY_OF_WEEK = int(_knobs["DAY_OF_WEEK"])
    DEBUG_TS = (
        pd.Timestamp(_knobs["DEBUG_TS"])
        if _knobs.get("DEBUG_TS") is not None
        else WEEK_START + pd.Timedelta(hours=8)
    )
    RUN_MPC_DEBUG = bool(_knobs.get("RUN_MPC_DEBUG", False))
    RUN_DAY_REPLAY = bool(_knobs.get("RUN_DAY_REPLAY", True))
    EV_DEADLINE_SLACK_MIN_PLOT = int(_knobs.get("EV_DEADLINE_SLACK_MIN", 105))
elif "WEEK_START" in globals() and "DAY_OF_WEEK" in globals():
    WEEK_START = pd.Timestamp(WEEK_START)
    DAY_OF_WEEK = int(DAY_OF_WEEK)
    DEBUG_TS = pd.Timestamp(
        globals()["DEBUG_TS"] if "DEBUG_TS" in globals() else WEEK_START + pd.Timedelta(hours=8)
    )
    RUN_MPC_DEBUG = bool(globals().get("RUN_MPC_DEBUG", False))
    RUN_DAY_REPLAY = bool(globals().get("RUN_DAY_REPLAY", True))
    EV_DEADLINE_SLACK_MIN_PLOT = int(globals().get("EV_DEADLINE_SLACK_MIN", 105))
else:
    raise ValueError(
        "Set WEEK_START and DAY_OF_WEEK (Part 4C cell, or at the top of this §3.1 cell). "
        "No default week is applied."
    )

WEEK_END = WEEK_START + pd.Timedelta(days=7)
year_start = pd.Timestamp("2025-01-01 00:00:00")
week_number = int((WEEK_START - year_start).days // 7) + 1
selected_day_start = WEEK_START + pd.Timedelta(days=DAY_OF_WEEK - 1)
selected_day_end = selected_day_start + pd.Timedelta(days=1)
print(
    f"§3.1 plot window: week {WEEK_START.date()} – "
    f"{(WEEK_END - pd.Timedelta(days=1)).date()}, "
    f"daily day {DAY_OF_WEEK} -> {selected_day_start.date()}"
)


def _ts_plot_local(s: pd.Series) -> pd.Series:
    """Naive Europe/Brussels timestamps for §3.1 (nb09 exports + offset CSVs like uncontrolled_hp)."""
    ts = pd.to_datetime(s, errors="coerce")
    if isinstance(ts.dtype, pd.DatetimeTZDtype):
        ts = ts.dt.tz_convert("Europe/Brussels").dt.tz_localize(None)
    return pd.Series(pd.to_datetime(ts.to_numpy(), errors="coerce"), index=s.index, dtype="datetime64[ns]")


# --- Load online results ---
if "res_evhp_online" in globals():
    res_plot = res_evhp_online.copy()
else:
    if not RESULTS_15MIN_PATH.exists():
        raise FileNotFoundError(
            f"Run Part 2 first or place results at {RESULTS_15MIN_PATH}"
        )
    res_plot = pd.read_csv(RESULTS_15MIN_PATH)

res_plot["timestamp"] = _ts_plot_local(res_plot["timestamp"])
res_plot = res_plot.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

_plant_path = PROJECT_ROOT / "data" / "plant1.csv"
if _plant_path.exists():
    _plant_need = [
        c
        for c in ("grid_consumption", "grid_consumption_excl_ev")
        if c not in res_plot.columns
    ]
    if _plant_need:
        _plant_m = pd.read_csv(_plant_path, usecols=["timestamp"] + _plant_need)
        _plant_m["timestamp"] = _ts_plot_local(_plant_m["timestamp"])
        res_plot = res_plot.merge(_plant_m, on="timestamp", how="left")

# kW convenience columns
res_plot["ev_applied_kw"] = pd.to_numeric(res_plot["ev_applied"], errors="coerce").fillna(0.0) * 4.0
res_plot["hp_applied_kw"] = pd.to_numeric(res_plot["hp_applied"], errors="coerce").fillna(0.0) * 4.0
res_plot["ev_plan_kw"] = pd.to_numeric(res_plot["ev_plan_kwh"], errors="coerce").fillna(0.0) * 4.0
res_plot["hp_plan_kw"] = pd.to_numeric(res_plot["hp_plan_kwh"], errors="coerce").fillna(0.0) * 4.0
res_plot["ev_baseline_kw"] = pd.to_numeric(res_plot["ev"], errors="coerce").fillna(0.0) * 4.0
res_plot["spot_price"] = pd.to_numeric(res_plot.get("price"), errors="coerce")
_diag_cols = ["was_clipped", "ev_enforce_deferred", "ev_envelope_headroom_after_kwh"]
_missing_diag = [c for c in _diag_cols if c not in res_plot.columns]
if _missing_diag:
    raise KeyError(
        f"Results missing {_missing_diag}. Re-run Part 2 after updating src/online_MPC_1_EV_HP.py."
    )

_col_clip = "grid_clip_limit_kw" if "grid_clip_limit_kw" in res_plot.columns else "p_limit_kw"

# Uncontrolled HP baseline (kW) — align by row index (nb10 / notebook 03; avoids timestamp merge dtypes)
if UNCONTROLLED_HP_PATH.exists():
    hp_un = pd.read_csv(UNCONTROLLED_HP_PATH)
    hp_baseline_kw = (
        pd.to_numeric(hp_un["hp_electrical_load"], errors="coerce").fillna(0.0) * 4.0
    )
    n_hp = min(len(res_plot), len(hp_baseline_kw))
    if len(res_plot) != len(hp_baseline_kw):
        print(
            f"WARNING: res_plot ({len(res_plot)}) vs uncontrolled_hp ({len(hp_baseline_kw)}); "
            f"using first n={n_hp} rows for HP baseline."
        )
    res_plot["hp_baseline_kw"] = 0.0
    res_plot.loc[res_plot.index[:n_hp], "hp_baseline_kw"] = hp_baseline_kw.iloc[:n_hp].to_numpy()
else:
    print(f"NOTE: {UNCONTROLLED_HP_PATH} not found — HP baseline trace omitted.")
    res_plot["hp_baseline_kw"] = 0.0

# Deterministic joint export (notebook 04)
if "DET_EV_HP_15MIN" in globals():
    det_evhp = DET_EV_HP_15MIN.copy()
else:
    if not DET_EV_HP_15MIN_PATH.exists():
        raise FileNotFoundError(
            f"Run notebook 04 export first: {DET_EV_HP_15MIN_PATH}"
        )
    det_evhp = pd.read_csv(DET_EV_HP_15MIN_PATH)

det_evhp["timestamp"] = _ts_plot_local(det_evhp["timestamp"])
det_evhp["hp_kw_deterministic"] = (
    pd.to_numeric(det_evhp.get("hp_electrical_input"), errors="coerce").fillna(0.0) * 4.0
)
if "buffer_soc" in det_evhp.columns:
    det_evhp["buffer_soc_pct"] = pd.to_numeric(det_evhp["buffer_soc"], errors="coerce") * 100.0

# --- SOC violation summary ---
hp_cfg_path = PROJECT_ROOT / "config" / "hp.yaml"
with open(hp_cfg_path, "r", encoding="utf-8") as f:
    hp_cfg = yaml.safe_load(f)
SOC_MIN = float(hp_cfg["buffer"]["soc_min"])
SOC_MAX = float(hp_cfg["buffer"]["soc_max"])


def _summarize_soc_violations(df_soc, *, soc_col, label, dt_minutes=15):
    if df_soc is None or df_soc.empty or soc_col not in df_soc.columns:
        print(f"SOC violations ({label}): no data / missing '{soc_col}'.")
        return
    _d = df_soc[["timestamp", soc_col]].copy()
    _d["timestamp"] = pd.to_datetime(_d["timestamp"], errors="coerce")
    _d = _d.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    _soc = pd.to_numeric(_d[soc_col], errors="coerce")
    mask = (_soc < SOC_MIN) | (_soc > SOC_MAX)
    n_points = int(mask.sum())
    runs = []
    in_run, start_i = False, None
    for i, flag in enumerate(mask.fillna(False).tolist()):
        if flag and not in_run:
            in_run, start_i = True, i
        elif (not flag) and in_run:
            runs.append((start_i, i - 1))
            in_run, start_i = False, None
    if in_run and start_i is not None:
        runs.append((start_i, len(mask) - 1))
    print("\n" + "=" * 80)
    print(f"SOC violations summary ({label})")
    print("=" * 80)
    print(f"SOC bounds: [{SOC_MIN:.3f}, {SOC_MAX:.3f}]")
    print(f"Violation points: {n_points} (each = {dt_minutes} min)")
    print(f"Violation episodes: {len(runs)}")
    if runs:
        durs_h = [(b - a + 1) * dt_minutes / 60.0 for a, b in runs]
        print(f"Total violated time: {sum(durs_h):.2f} h; max episode: {max(durs_h):.2f} h")


_summarize_soc_violations(res_plot, soc_col="soc_after", label="online MPC (soc_after)")
_summarize_soc_violations(det_evhp, soc_col="buffer_soc", label="deterministic MPC (buffer_soc)")


def _add_ev_remaining(df_on, df_det):
    """Merge det + online remaining EV energy (kWh) onto df_on by timestamp."""
    det = df_det.copy()
    det["date"] = det["timestamp"].dt.date
    if "ev_demand_actual" in det.columns and "ev_charge" in det.columns:
        det["det_remaining_kwh"] = det.groupby("date")["ev_demand_actual"].transform("sum") - det.groupby(
            "date"
        )["ev_charge"].transform(lambda s: s.cumsum())
    on = df_on.copy()
    on["date"] = on["timestamp"].dt.date
    daily_need = on.groupby("date")["ev"].transform("sum")
    on["online_charged_cum_kwh"] = on.groupby("date")["ev_applied"].transform(lambda s: s.cumsum())
    on["online_remaining_kwh"] = daily_need - on["online_charged_cum_kwh"]
    cols = ["timestamp", "det_remaining_kwh"] if "det_remaining_kwh" in det.columns else ["timestamp"]
    if len(cols) > 1:
        on = on.merge(det[cols], on="timestamp", how="left")
    return on


def _zoh(ax, ts, y, **kwargs) -> None:
    """Zero-order hold (15-min piecewise constant) — steps-post."""
    ax.step(
        pd.to_datetime(ts, errors="coerce"),
        pd.to_numeric(y, errors="coerce"),
        where="post",
        **kwargs,
    )


def _shade_forecast_stress(axes_list, ts: pd.Series, stress_active: pd.Series) -> None:
    """Light vertical bands when forecast grid (MPC inputs) exceeds access — nb10 style."""
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
        for ax_i, ax in enumerate(axes_list):
            ax.axvspan(
                t0,
                t1,
                facecolor="tab:purple",
                alpha=0.12,
                zorder=0,
                label=lab if ax_i == 0 else None,
            )
        i = j + 1


def _vlines_ev_window(ax, t0, t1):
    first = True
    for d in pd.date_range(t0.normalize(), t1.normalize() - pd.Timedelta(days=1), freq="D"):
        if d.weekday() >= 5:
            continue
        s = d.replace(hour=7, minute=0, second=0)
        e = d.replace(hour=17, minute=0, second=0)
        lab = "EV window (weekdays 07:00–17:00)" if first else None
        first = False
        ax.axvline(s, color="red", linestyle="--", linewidth=1.2, alpha=0.85, label=lab)
        ax.axvline(e, color="red", linestyle="--", linewidth=1.2, alpha=0.85)


# --- Weekly slice ---
week = res_plot[
    (res_plot["timestamp"] >= WEEK_START) & (res_plot["timestamp"] < WEEK_END)
].copy()
week_det = det_evhp[
    (det_evhp["timestamp"] >= WEEK_START) & (det_evhp["timestamp"] < WEEK_END)
].copy()
week_merge = week.merge(
    week_det[
        [
            c
            for c in [
                "timestamp",
                "ev_demand_actual",
                "ev_charge_power",
                "ev_power_envelope",
                "hp_kw_deterministic",
                "hp_thermal_power",
                "buffer_soc_pct",
                "grid_power",
            ]
            if c in week_det.columns
        ]
    ],
    on="timestamp",
    how="left",
)
week_merge = _add_ev_remaining(week_merge, week_det)

# === Weekly thesis-style volume plots (nb09 EV + nb10 HP) ===
week_view = _thesis_plt.enrich_plot_frame(week_merge)
if "forecast_access_exceedance_active" in week.columns:
    week_view["forecast_access_exceedance_active"] = week[
        "forecast_access_exceedance_active"
    ].to_numpy()
else:
    print(
        "NOTE: 'forecast_access_exceedance_active' not in Part 2 export — "
        "re-run Part 2 to shade forecast-stress periods."
    )
_thesis_plt.plot_thesis_week_ev_power(week_view, WEEK_START, WEEK_END)
_thesis_plt.plot_thesis_week_hp_electrical(week_view, week_det, WEEK_START, WEEK_END)
_thesis_plt.plot_thesis_week_buffer_soc(
    week, week_det, WEEK_START, WEEK_END, soc_min=SOC_MIN, soc_max=SOC_MAX
)
_thesis_plt.plot_thesis_week_grid_power(week_view, week_det, WEEK_START, WEEK_END)

# === Daily deep-dive ===
day = res_plot[
    (res_plot["timestamp"] >= selected_day_start)
    & (res_plot["timestamp"] < selected_day_end)
].copy()
day_det = det_evhp[
    (det_evhp["timestamp"] >= selected_day_start)
    & (det_evhp["timestamp"] < selected_day_end)
].copy()
day_merge = day.merge(
    day_det[
        [
            c
            for c in [
                "timestamp",
                "ev_demand_actual",
                "ev_charge_power",
                "ev_power_envelope",
                "hp_kw_deterministic",
                "hp_thermal_power",
                "buffer_soc_pct",
                "grid_power",
            ]
            if c in day_det.columns
        ]
    ],
    on="timestamp",
    how="left",
)
day_merge = _add_ev_remaining(day_merge, day_det)

xlim_lo = selected_day_start.replace(hour=5, minute=0)
xlim_hi = selected_day_start.replace(hour=19, minute=0)

# === Daily thesis-style volume plots (nb09 EV + nb10 HP) ===
day_view = _thesis_plt.enrich_plot_frame(day_merge)
for _c in ("ev_enforce_active", "ev_enforce_deferred", "forecast_access_exceedance_active"):
    if _c in day.columns and _c not in day_view.columns:
        day_view = day_view.merge(day[["timestamp", _c]], on="timestamp", how="left")
_day_xlim = (xlim_lo, xlim_hi)
_thesis_plt.plot_thesis_volume_suite_day(
    day_view,
    day_det,
    selected_day_start,
    soc_min=SOC_MIN,
    soc_max=SOC_MAX,
    show_enforce_markers=False,
    xlim=_day_xlim,
)

if RUN_DAY_REPLAY:
    _replay_sum = summarize_day(
        day,
        day=selected_day_start,
        slack_minutes=EV_DEADLINE_SLACK_MIN_PLOT,
        clip_col=_col_clip,
    )
    print_day_replay_report(_replay_sum)
    plot_day_replay(
        day,
        day=selected_day_start,
        slack_minutes=EV_DEADLINE_SLACK_MIN_PLOT,
        clip_col=_col_clip,
        xlim=(xlim_lo, xlim_hi),
    )
    plt.show()

_n_def = int((day["ev_enforce_deferred"] > 0.5).sum()) if "ev_enforce_deferred" in day.columns else 0
_n_enf = int((day["ev_enforce_active"] > 0.5).sum()) if "ev_enforce_active" in day.columns else 0
print(
    f"\nDay summary ({selected_day_start.strftime('%Y-%m-%d')}): "
    f"EV online {day['ev_applied'].sum():.1f} kWh, "
    f"HP online {day['hp_applied'].sum():.1f} kWh, "
    f"mean SOC {pd.to_numeric(day['soc_after'], errors='coerce').mean() * 100:.1f}%, "
    f"enforce active {_n_enf} steps, deferred {_n_def} steps"
)

# === Yearly summaries ===
res_y = res_plot.copy()
res_y["date"] = res_y["timestamp"].dt.date

if "uncharged_kwh" in res_y.columns:
    daily_unmet = res_y.groupby("date")["uncharged_kwh"].max()
else:
    daily_unmet = (
        res_y.groupby("date")["ev"].sum() - res_y.groupby("date")["ev_applied"].sum()
    ).clip(lower=0.0)

daily_ev_online = res_y.groupby("date")["ev_applied"].sum()
det_ts = det_evhp.copy()
det_ts["date"] = det_ts["timestamp"].dt.date
daily_ev_det = det_ts.groupby("date")["ev_charge"].sum() if "ev_charge" in det_ts.columns else None

fig_y1, ax_u = plt.subplots(figsize=(16, 4))
_zoh(
    ax_u,
    pd.to_datetime(daily_unmet.index),
    daily_unmet.values,
    color="tab:orange",
    alpha=0.85,
    linewidth=1.5,
)
ax_u.set_ylabel("kWh / day")
ax_u.set_title("Daily uncharged EV energy (online joint MPC)")
ax_u.grid(True, axis="y", alpha=0.35)
plt.tight_layout()
plt.show()

fig_y2, ax_e = plt.subplots(figsize=(16, 4))
_zoh(
    ax_e,
    pd.to_datetime(daily_ev_online.index),
    daily_ev_online.values,
    label="EV delivered (online)",
    color="tab:blue",
)
if daily_ev_det is not None:
    _zoh(
        ax_e,
        pd.to_datetime(daily_ev_det.index),
        daily_ev_det.reindex(daily_ev_online.index).values,
        label="EV delivered (deterministic)",
        color="tab:green",
        alpha=0.8,
    )
ax_e.set_ylabel("kWh / day")
ax_e.set_title("Daily EV energy delivered")
ax_e.legend()
ax_e.grid(True, alpha=0.35)
plt.tight_layout()
plt.show()

res_y["grid_online_kw"] = pd.to_numeric(res_y["p_grid_actual_kw"], errors="coerce")
res_y["grid_baseline_kw"] = (
    pd.to_numeric(res_y["grid_consumption"], errors="coerce").fillna(0.0) * 4.0
    + pd.to_numeric(res_y["hp_baseline_kw"], errors="coerce").fillna(0.0)
)

fig_y3, ax_g = plt.subplots(figsize=(14, 5))
_zoh(
    ax_g,
    res_y["timestamp"],
    res_y["grid_online_kw"],
    label="Grid with joint online MPC",
    alpha=0.75,
    linewidth=0.8,
)
_zoh(
    ax_g,
    res_y["timestamp"],
    res_y["grid_baseline_kw"],
    label="Grid baseline (plant consumption + uncontrolled HP)",
    alpha=0.6,
    linewidth=0.8,
)
ax_g.set_ylabel("kW")
ax_g.set_title("Full-year grid power — online MPC vs baseline")
ax_g.legend()
ax_g.grid(True, alpha=0.35)
plt.tight_layout()
plt.show()

# === Optional MPC 24h window debug (joint) ===
if RUN_MPC_DEBUG:
    if "ACCESS_POWER_DICT" not in globals():
        raise RuntimeError("Run §1.2 before MPC debug (need ACCESS_POWER_DICT).")

    SRC_DIR = PROJECT_ROOT / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    from online_MPC_1_EV_HP import _load_forecast_column, _parse_plant_data
    from optimization import mpc_ev_hp_24h

    _ev_strat = globals().get("EV_FORECAST_STRATEGY", "c_p90")
    _inflex_strat = globals().get("INFLEX_FORECAST_STRATEGY", "c")
    _pv_strat = globals().get("PV_FORECAST_STRATEGY", "chronos2_elia_p50")
    _th_strat = globals().get("THERMAL_FORECAST_STRATEGY", "c2t_p50")
    _temp_path = PROJECT_ROOT / "data" / "temperature_forecast_day_ahead_open_meteo_Turnhout_15min.csv"
    _slack = int(globals().get("EV_DEADLINE_SLACK_MIN", 105))

    plant_dbg = _parse_plant_data(PROJECT_ROOT / "data" / "plant1.csv")
    inflex_fc = _load_forecast_column(
        PROJECT_ROOT / "output" / "forecast" / "forecast_inflex_load_rolling_horizon.csv",
        strategy=_inflex_strat,
        prefix="forecast_inflex_",
        hint_prefix="forecast_inflex_",
    )
    ev_fc = _load_forecast_column(
        PROJECT_ROOT / "output" / "forecast" / "forecast_ev_rolling_horizon.csv",
        strategy=_ev_strat,
        prefix="forecast_ev_",
        hint_prefix="forecast_ev_",
    )
    th_fc = _load_forecast_column(
        PROJECT_ROOT / "output" / "forecast" / "forecast_thermal_load_rolling_horizon.csv",
        strategy=_th_strat,
        prefix="forecast_thermal_",
        hint_prefix="forecast_thermal_",
    )
    pv_df = pd.read_csv(PROJECT_ROOT / "output" / "forecast" / "forecast_pv_rolling_horizon.csv")
    _pv_col = (
        _pv_strat
        if str(_pv_strat).startswith("pv_forecast_kWh_15min_")
        else f"pv_forecast_kWh_15min_{_pv_strat}"
    )
    pv_fc = pd.to_numeric(pv_df[_pv_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    temp_df = pd.read_csv(_temp_path)
    _tc = [c for c in temp_df.columns if c != "timestamp"][0]
    temp_fc = pd.to_numeric(temp_df[_tc], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    plant_dbg = plant_dbg.copy()
    plant_dbg["inflex_forecast"] = inflex_fc
    plant_dbg["ev_forecast"] = ev_fc
    plant_dbg["pv_for_mpc"] = pv_fc
    plant_dbg["thermal_forecast"] = th_fc
    plant_dbg["outdoor_temperature_for_mpc"] = temp_fc

    mask_k = plant_dbg["timestamp"] == DEBUG_TS
    if not mask_k.any():
        raise ValueError(f"DEBUG_TS {DEBUG_TS} not in plant data.")
    k0 = int(plant_dbg.index[mask_k][0])
    k_end = min(k0 + 96, len(plant_dbg))
    df_w = plant_dbg.loc[k0 : k_end - 1].copy()
    wlen = k_end - k0
    df_w["ev_power_envelope_fixed_kw"] = 0.0
    df_w["ev_power_envelope_forecast_kw"] = 0.0
    df_w.rename(
        columns={
            "pv_for_mpc": "pv_production",
            "inflex_forecast": "inflex_load",
            "ev_forecast": "ev",
            "thermal_forecast": "thermal_load",
            "outdoor_temperature_for_mpc": "outdoor_temperature",
        },
        inplace=True,
    )
    mk = DEBUG_TS.to_period("M").strftime("%Y-%m")
    hist = res_plot[res_plot["timestamp"] < DEBUG_TS].copy()
    ex_hist = (
        pd.to_numeric(hist["p_grid_actual_kw"], errors="coerce").fillna(0.0)
        - pd.to_numeric(hist["access_kw"], errors="coerce").fillna(0.0)
    ).clip(lower=0.0)
    hist["exceed_kw"] = ex_hist
    hist["month_key"] = hist["timestamp"].dt.to_period("M").astype(str)
    finalized = hist[hist["month_key"] < mk].groupby("month_key")["exceed_kw"].max()
    roll12_done = float(finalized.tail(12).max()) if len(finalized) else 0.0
    curr_ex = float(hist[hist["month_key"] == mk]["exceed_kw"].max()) if (hist["month_key"] == mk).any() else 0.0
    roll12_so_far = max(roll12_done, curr_ex)
    peak_so_far = float(
        hist[hist["month_key"] == mk]["p_grid_actual_kw"].max()
        if (hist["month_key"] == mk).any()
        else 0.0
    )
    mask_res = res_plot["timestamp"] == DEBUG_TS
    soc_init = float(pd.to_numeric(res_plot.loc[mask_res, "soc_before"], errors="coerce").iloc[0])
    dates_w = pd.to_datetime(df_w["timestamp"]).dt.date.unique().tolist()
    daily_rem = {d: 0.0 for d in dates_w}
    if dates_w:
        daily_rem[dates_w[0]] = float(
            res_plot.loc[mask_res, "online_remaining_kwh"].iloc[0]
            if "online_remaining_kwh" in res_plot.columns
            else 0.0
        )
    roll12_map = {
        ts.to_period("M").strftime("%Y-%m"): roll12_so_far
        for ts in pd.to_datetime(df_w["timestamp"])
    }
    win_res, win_sum = mpc_ev_hp_24h(
        df_window=df_w,
        config_path=str(PROJECT_ROOT / "config" / "billing.yaml"),
        hp_config_path=str(PROJECT_ROOT / "config" / "hp.yaml"),
        monthly_peak_so_far={mk: peak_so_far},
        rolling12_max_exceedance_so_far_by_month=roll12_map,
        soc_initial=soc_init,
        daily_ev_remaining=daily_rem,
        ev_deadline_slack_minutes=_slack,
        access_power_by_month=ACCESS_POWER_DICT,
    )
    win = win_res.copy()
    win["ev_kw"] = 4.0 * pd.to_numeric(win["ev_charge"], errors="coerce").fillna(0.0)
    win["hp_kw"] = 4.0 * pd.to_numeric(win["hp_electrical_input"], errors="coerce").fillna(0.0)
    fig_dbg, ax_dbg = plt.subplots(4, 1, figsize=(16, 11), sharex=True)
    _zoh(ax_dbg[0], win["timestamp"], win["ev_kw"], label="EV plan (kW)")
    _zoh(ax_dbg[0], win["timestamp"], win["hp_kw"], label="HP plan (kW)")
    ax_dbg[0].set_ylabel("kW")
    ax_dbg[0].legend()
    ax_dbg[0].set_title(f"Joint MPC 24h window @ {DEBUG_TS}")
    _zoh(ax_dbg[1], win["timestamp"], win["grid_power"], label="Grid power (kW)", color="black")
    ax_dbg[1].set_ylabel("kW")
    ax_dbg[1].legend()
    if "buffer_soc" in win.columns:
        _zoh(ax_dbg[2], win["timestamp"], win["buffer_soc"] * 100, label="SOC (%)")
        ax_dbg[2].set_ylabel("SOC %")
        ax_dbg[2].legend()
    if "spot_price_eur_per_mwh" in win.columns:
        _zoh(ax_dbg[3], win["timestamp"], win["spot_price_eur_per_mwh"], label="Price")
        ax_dbg[3].set_ylabel("EUR/MWh")
    plt.tight_layout()
    plt.show()
    print("MPC debug objective:", win_sum.get("objective_value"))
else:
    print("\nMPC 24h window debug skipped (RUN_MPC_DEBUG=False).")
