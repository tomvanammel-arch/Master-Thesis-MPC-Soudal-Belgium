"""Patch notebook 11 §3.2 cells from build_nb11_online_ev_hp.PART32 (does not touch §3.1)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

PART32 = (ROOT / "scripts" / "nb11_part32_cell.py").read_text(encoding="utf-8")

NB = ROOT / "notebooks" / "11_online_MPC_1_EV+HP.ipynb"
MD_32 = """### §3.2 The bill

Shadow billing: **baseline** (uncontrolled EV+HP, notebook 04 export) vs **deterministic joint** vs **online** (Part 2).

- **Site reference R** = notebook 01: `grid_consumption_excl_ev`, conservative AP from full-site grid peaks (~1,631,929 EUR offtake)
- **Heating & charging** = full annual net − R (uncontrolled EV+HP incremental over nb01 site)

Re-run Part 2 after changing access rules or `online_MPC_1_EV_HP.py`."""


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    md_idx = code_idx = None
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "markdown":
            continue
        src = "".join(c.get("source", []))
        if "### §3.2" in src or "### 3.2" in src:
            md_idx = i
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c.get("source", []))
        if src.startswith("# §3.2") or src.startswith("# Part 3.2"):
            code_idx = i
            break
    if md_idx is None or code_idx is None:
        raise RuntimeError(f"Could not find §3.2 cells (md={md_idx}, code={code_idx})")

    nb["cells"][md_idx]["source"] = [line + "\n" for line in MD_32.splitlines()]
    nb["cells"][code_idx]["source"] = [line + "\n" for line in PART32.splitlines()]
    nb["cells"][code_idx]["outputs"] = []
    nb["cells"][code_idx]["execution_count"] = None

    # Part 4B baseline_access fix
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c.get("source", []))
        if "month_period_index_to_str_keys(ACCESS_POWER_BASELINE_MONTHLY)" in src:
            c["source"] = [
                ln.replace(
                    "month_period_index_to_str_keys(ACCESS_POWER_BASELINE_MONTHLY)",
                    "ACCESS_POWER_BASELINE_MONTHLY.astype(float).to_dict()",
                )
                for ln in c["source"]
            ]

    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Patched {NB} §3.2 markdown cell {md_idx}, code cell {code_idx}")


if __name__ == "__main__":
    main()
