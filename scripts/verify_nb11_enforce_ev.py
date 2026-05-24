"""Quick False vs True check for enforce_daily_ev_demand in joint online MPC."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

import online_MPC_1_EV_HP as evhp  # noqa: E402

N_DAYS = 21
N_STEPS = 96 * N_DAYS
_orig_parse = evhp._parse_plant_data
_orig_load = evhp._load_forecast_column
_orig_read_csv = pd.read_csv


def _limited_plant(plant_path: Path):
    df = _orig_parse(plant_path)
    return df.iloc[:N_STEPS].copy().reset_index(drop=True)


def _limited_load(*args, **kwargs):
    return _orig_load(*args, **kwargs)[:N_STEPS]


def _limited_read_csv(path, *args, **kwargs):
    df = _orig_read_csv(path, *args, **kwargs)
    if len(df) > N_STEPS:
        return df.iloc[:N_STEPS].copy()
    return df


evhp._parse_plant_data = _limited_plant
evhp._load_forecast_column = _limited_load
pd.read_csv = _limited_read_csv

ACCESS = {f"2025-{m:02d}": 600.0 for m in range(1, 13)}
RUN_KWARGS = dict(
    forecast_strategy_ev="c_p90",
    forecast_strategy_inflex="c",
    forecast_strategy_pv="chronos2_elia_p50",
    forecast_strategy_thermal="c2t_p50",
    forecast_strategy_temperature="open_meteo_day_ahead",
    ev_deadline_slack_minutes=105,
    access_power_by_month=ACCESS,
    enforce_soc_min=True,
    enforce_soc_max=True,
    enable_forecast_stress_soc_floor=False,
    verbose=False,
)

ENFORCE_COLS = (
    "ev_enforce_active",
    "ev_enforce_extra_kwh",
    "ev_envelope_remaining_kwh",
    "ev_was_clipped",
)

results: dict[bool, dict] = {}
for enforce in (False, True):
    res, summ = evhp.run_ev_hp_online_mpc_1(
        enforce_daily_ev_demand=enforce, **RUN_KWARGS
    )
    uncharged = sum(max(v, 0.0) for v in summ["uncharged_kwh_by_day"].values())
    results[enforce] = {
        "uncharged_total_kwh": uncharged,
        "ev_enforce_steps": summ["ev_enforce_steps"],
        "ev_enforce_extra_kwh_total": summ["ev_enforce_extra_kwh_total"],
        "enforce_cols_ok": all(c in res.columns for c in ENFORCE_COLS),
    }

print(f"verify_nb11_enforce_ev ({N_DAYS} days)")
for enforce, r in results.items():
    print(f"  enforce_daily_ev_demand={enforce}: {r}")

assert results[False]["enforce_cols_ok"] and results[True]["enforce_cols_ok"]
assert results[False]["ev_enforce_steps"] == 0
assert results[False]["ev_enforce_extra_kwh_total"] == 0.0
assert results[True]["ev_enforce_extra_kwh_total"] >= 0.0
assert results[True]["uncharged_total_kwh"] <= results[False]["uncharged_total_kwh"] + 1e-6
print("OK")
