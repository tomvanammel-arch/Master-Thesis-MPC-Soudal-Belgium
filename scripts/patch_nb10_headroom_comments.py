import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "10_online_MPC_1_HP.ipynb"
nb = json.loads(NB.read_text(encoding="utf-8"))
src = "".join(nb["cells"][4]["source"])
old = (
    'ts_hr = plant_hr_grid["timestamp"]\n'
    'grid_kwh = plant_hr_grid["grid_consumption"].to_numpy(dtype=float)\n'
    "daily_h_cons = _daily_headroom_kwh(ts_hr, grid_kwh, access_power_by_month_conservative_hr)\n"
    "daily_h_flex = _daily_headroom_kwh(ts_hr, grid_kwh, access_power_by_month_flex_hr)"
)
new = (
    'ts_hr = plant_hr_grid["timestamp"]\n'
    'grid_kwh = plant_hr_grid["grid_consumption"].to_numpy(dtype=float)\n'
    "\n"
    "# Headroom playroom (full day): max(P_access - 4*grid_consumption, 0) kW; daily H_d = sum * 0.25 h\n"
    "# Conservative: P_access^cons; flex-aware: P_access^flex (each its own monthly step series)\n"
    "daily_h_cons = _daily_headroom_kwh(ts_hr, grid_kwh, access_power_by_month_conservative_hr)\n"
    "daily_h_flex = _daily_headroom_kwh(ts_hr, grid_kwh, access_power_by_month_flex_hr)"
)
if old not in src:
    raise SystemExit("block not found in §1.2")
nb["cells"][4]["source"] = [src.replace(old, new)]
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Patched", NB)
