"""Append §4E forecast-stress quantile plot cells to notebook 10."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "10_online_MPC_1_HP.ipynb"

MARKDOWN = """### 4E — Forecast stress hours by inflex quantile (2025)

Counts **forecast access exceedance** steps for inflex stress quantiles **p50, p90, p95, p99** (same rule as Part 2 `forecast_access_exceedance_active`):

\\[
P_\\mathrm{forecast,grid} = 4\\,(E_\\mathrm{inflex,q} + E_\\mathrm{EV} + E_\\mathrm{HP,est} - E_\\mathrm{PV}) > P_\\mathrm{access}(m)
\\]

EV / thermal / PV forecasts stay at Part 2 defaults (p50). **Flex-aware** `ACCESS_POWER_DICT` from §1.2. Reported metric: stress hours as **% of 8760 h**.
"""

CODE = """# Part 4E — Forecast stress hours vs inflex quantile (thesis style)
# Prerequisites: §1.2 (ACCESS_POWER_DICT). Reuses Part 2 forecast strategy knobs when set.

from pathlib import Path
import importlib.util
import sys

PROJECT_ROOT = Path("..").resolve()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if "ACCESS_POWER_DICT" not in globals():
    raise RuntimeError("Run §1.2 first to build ACCESS_POWER_DICT.")

_viewer_path = SRC_DIR / "notebook_visualisation" / "nb10_hp_part34_viewer.py"
_spec = importlib.util.spec_from_file_location("nb10_hp_viewer_4e", _viewer_path)
_viewer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_viewer)

print("4E stress quantile plot from:", _viewer_path)
_viewer.run_notebook10_forecast_stress_hours_by_quantile(PROJECT_ROOT)
"""


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    src_joined = "".join(
        line for cell in nb["cells"] for line in cell.get("source", [])
    )
    if "run_notebook10_forecast_stress_hours_by_quantile" in src_joined:
        print("Already patched:", NB_PATH)
        return

    nb["cells"].append(
        {
            "cell_type": "markdown",
            "id": "nb10-part4e-stress-quantile-md",
            "metadata": {},
            "source": [line + "\n" for line in MARKDOWN.split("\n")],
        }
    )
    nb["cells"].append(
        {
            "cell_type": "code",
            "id": "nb10-part4e-stress-quantile-code",
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in CODE.split("\n")],
        }
    )
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Patched", NB_PATH, "->", len(nb["cells"]), "cells")


if __name__ == "__main__":
    main()
