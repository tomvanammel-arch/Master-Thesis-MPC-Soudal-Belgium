"""Patch notebook 09 Part 4.2 cell to use thesis-style slack sensitivity plots."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "09_online_MPC_1_EV.ipynb"

NEW_CELL = '''# Part 4.2 — Plot unmet energy vs slack, and online savings vs slack (thesis style)

import sys
from pathlib import Path

import importlib
import pandas as pd

PROJECT_ROOT = Path("..").resolve()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from notebook_visualisation import nb09_ev_part34_viewer as _ev_thesis

importlib.reload(_ev_thesis)

summary_path = PROJECT_ROOT / "output" / "notebooks" / "online_ev_slack_sensitivity_summary_notebook_09.csv"
if not summary_path.exists():
    raise FileNotFoundError(f"Sensitivity summary not found: {summary_path}. Run Part 4.1 first.")

sens = pd.read_csv(summary_path)
sens = sens.sort_values("slack_min").reset_index(drop=True)

_ev_thesis.plot_thesis_slack_unmet(sens)
_ev_thesis.plot_thesis_slack_savings(sens)

display(sens)
'''


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    for c in nb["cells"]:
        src = "".join(c.get("source", []))
        if "Part 4.2" in src and ("unmet" in src.lower() or "slack" in src.lower()):
            c["source"] = [line + "\n" for line in NEW_CELL.splitlines()]
            if c["source"] and not c["source"][-1].endswith("\n"):
                c["source"][-1] += "\n"
            c["outputs"] = []
            c["execution_count"] = None
            NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
            print("Patched Part 4.2 in", NB_PATH)
            return
    raise RuntimeError("Part 4.2 cell not found")


if __name__ == "__main__":
    main()
