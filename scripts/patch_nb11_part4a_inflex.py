"""Update Part 4 intro + 4A cell: sweep regular inflex forecast (c_p50/c_p90), not stress."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_nb11_online_ev_hp as b11  # noqa: E402

NB = ROOT / "notebooks" / "11_online_MPC_1_EV+HP.ipynb"


def _src(cell: dict) -> str:
    return "".join(cell.get("source", []))


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    for i, c in enumerate(nb["cells"]):
        s = _src(c)
        if s.startswith("## 4. Scenario analysis"):
            nb["cells"][i]["source"] = [line + "\n" for line in b11.PART4_INTRO.splitlines()]
        elif s.startswith("# Part 4A"):
            nb["cells"][i]["source"] = [line + "\n" for line in b11.PART4A.splitlines()]
        elif "forecast_strategy_inflex_stress" in s and "INFLEX_FORECAST_STRATEGY_STRESS" in s and "Part 4C" in s:
            # 4C knob sync block
            old = """    for _k, _g in (
        ("forecast_strategy_inflex_stress", "INFLEX_FORECAST_STRATEGY_STRESS"),
        ("forecast_stress_soc_floor_strength", "FORECAST_STRESS_SOC_FLOOR_STRENGTH"),
    ):"""
            new = """    for _k, _g in (
        ("forecast_strategy_inflex", "INFLEX_FORECAST_STRATEGY"),
        ("forecast_strategy_inflex_stress", "INFLEX_FORECAST_STRATEGY_STRESS"),
        ("forecast_stress_soc_floor_strength", "FORECAST_STRESS_SOC_FLOOR_STRENGTH"),
    ):"""
            src = s.replace(old, new)
            nb["cells"][i]["source"] = [line + "\n" for line in src.splitlines()]

    NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("Patched Part 4 intro + 4A (+ 4C forecast knobs if present)")


if __name__ == "__main__":
    main()
