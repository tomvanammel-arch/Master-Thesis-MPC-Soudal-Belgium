"""Append Part 4 cells to notebooks/11_online_MPC_1_EV+HP.ipynb (if missing)."""
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
    if any("## 4. Scenario analysis" in _src(c) for c in nb["cells"]):
        print("Part 4 already present — updating Part 4 code cells only.")
        # Remove old Part 4 block from first ## 4. marker
        start = None
        for i, c in enumerate(nb["cells"]):
            if _src(c).startswith("## 4. Scenario analysis"):
                start = i
                break
        if start is not None:
            nb["cells"] = nb["cells"][:start]
    new_cells = [
        b11.cell(md=b11.PART4_INTRO),
        b11.cell(md="### 4A — Scenario definitions"),
        b11.cell(code=b11.PART4A),
        b11.cell(md="### 4B — Batch run"),
        b11.cell(code=b11.PART4B),
        b11.cell(md="### 4C — Viewer (Part 3.1 + 3.2)"),
        b11.cell(code=b11.PART4C),
        b11.cell(md="### 4D — Savings comparison"),
        b11.cell(code=b11.PART4D),
    ]
    # Update intro bullet if present
    for c in nb["cells"]:
        s = _src(c)
        if s.startswith("# Notebook 11"):
            c["source"] = [
                line.replace(
                    "inflex stress × SOC floor",
                    "12 scenarios: access × SOC floor × inflex stress",
                )
                for line in c["source"]
            ]
            break
    nb["cells"].extend(new_cells)
    NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Patched {NB} — added {len(new_cells)} Part 4 cells")


if __name__ == "__main__":
    main()
