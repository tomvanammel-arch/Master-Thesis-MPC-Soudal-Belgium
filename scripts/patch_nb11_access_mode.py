"""Patch notebook 11: access power mode selector (§1.2, Part 2, §3.1, §3.2)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_nb11_online_ev_hp as b11  # noqa: E402

NB = ROOT / "notebooks" / "11_online_MPC_EV+HP.ipynb"

MD_PART2 = """## Part 2 — Rolling-horizon joint simulation

In the code cell below, set **`ONLINE_ACCESS_POWER_MODE`** before running:

| Mode | Dictionary | Description |
|------|------------|-------------|
| `"flex_aware"` | `ACCESS_POWER_FLEX_DICT` | §1.2 hybrid rule (default) |
| `"deterministic"` | `ACCESS_POWER_DETERMINISTIC_DICT` | Notebook 04 optimized monthly access |

Part 3 (§3.1 plots and §3.2 billing) uses the same selection via `access_kw` in the Part 2 export. **Re-run Part 2** after changing the mode, then Part 3."""

MD_31_EXTRA = (
    "**Knobs** (top of code cell): `WEEK_START`, `DAY_OF_WEEK` (1–7 within that week), "
    "`DEBUG_TS`, `RUN_MPC_DEBUG` (default `False`). Online `access_kw` limits follow "
    "**`ONLINE_ACCESS_POWER_MODE`** from Part 2."
)

MD_32_ONLINE = (
    "Shadow billing: **baseline** (uncontrolled EV+HP, conservative access from §1.2) vs "
    "**deterministic joint** (notebook 04 monthly exports) vs **online** (Part 2 bills; "
    "access contract = **`ONLINE_ACCESS_POWER_MODE`** from Part 2: flex-aware or deterministic)."
)

S12_AP_BLOCK_OLD = """ACCESS_POWER_ONLINE_MONTHLY = access_power_flex_aware_kw.copy()
ACCESS_POWER_BASELINE_MONTHLY = access_power_conservative_kw.copy()
ACCESS_POWER_DICT = month_period_index_to_str_keys(ACCESS_POWER_ONLINE_MONTHLY)"""

S12_AP_BLOCK_NEW = """ACCESS_POWER_ONLINE_MONTHLY = access_power_flex_aware_kw.copy()
ACCESS_POWER_BASELINE_MONTHLY = access_power_conservative_kw.copy()
ACCESS_POWER_FLEX_DICT = month_period_index_to_str_keys(access_power_flex_aware_kw)
ACCESS_POWER_DETERMINISTIC_DICT = access_power_deterministic_kw.astype(float).to_dict()
ACCESS_POWER_DICT = ACCESS_POWER_FLEX_DICT  # active online contract; set in Part 2"""


def _src(cell: dict) -> str:
    return "".join(cell.get("source", []))


def _set_src(cell: dict, text: str) -> None:
    cell["source"] = [line + "\n" for line in text.splitlines()]
    cell["outputs"] = []
    cell["execution_count"] = None


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]

    for c in cells:
        if c["cell_type"] == "code" and S12_AP_BLOCK_OLD in _src(c):
            _set_src(c, _src(c).replace(S12_AP_BLOCK_OLD, S12_AP_BLOCK_NEW))
            print("Patched §1.2 access dicts")

    for i, c in enumerate(cells):
        if c["cell_type"] == "markdown" and _src(c).strip() == "## Part 2 — Rolling-horizon joint simulation":
            _set_src(c, MD_PART2)
            print(f"Patched Part 2 markdown at cell {i}")

    for c in cells:
        if c["cell_type"] == "code" and _src(c).startswith("# Part 2 — Joint online MPC"):
            _set_src(c, b11.PART2)
            print("Patched Part 2 code")

    for c in cells:
        if c["cell_type"] == "code" and _src(c).startswith("# §3.1 — Joint online optimised"):
            _set_src(c, b11.PART31)
            print("Patched §3.1 code")

    for c in cells:
        if c["cell_type"] == "markdown" and "### §3.1 Optimised volumes" in _src(c):
            s = _src(c)
            if "ONLINE_ACCESS_POWER_MODE" not in s:
                s = s.replace(
                    "`RUN_MPC_DEBUG` (default `False`)",
                    "`RUN_MPC_DEBUG` (default `False`). Online `access_kw` limits follow "
                    "**`ONLINE_ACCESS_POWER_MODE`** from Part 2.",
                )
            _set_src(c, s)
            print("Patched §3.1 markdown")

    for c in cells:
        if c["cell_type"] == "markdown" and "### §3.2 The bill" in _src(c):
            s = _src(c)
            if "ONLINE_ACCESS_POWER_MODE" not in s or s.count("Shadow billing:") > 1:
                # Rebuild §3.2 intro from template
                lines = s.splitlines()
                out = [lines[0], "", MD_32_ONLINE, ""]
                for line in lines[1:]:
                    if line.strip().startswith("Shadow billing:"):
                        continue
                    out.append(line)
                s = "\n".join(out).strip() + "\n"
            _set_src(c, s)
            print("Patched §3.2 markdown")

    for c in cells:
        if c["cell_type"] == "code" and _src(c).startswith("# §3.2 — Shadow billing"):
            _set_src(c, b11.PART32)
            print("Patched §3.2 code")

    # Part 4B: baseline dict fix if still using month_period_index_to_str_keys
    for c in cells:
        if c["cell_type"] == "code" and "month_period_index_to_str_keys(ACCESS_POWER_BASELINE_MONTHLY)" in _src(c):
            c["source"] = [
                ln.replace(
                    "month_period_index_to_str_keys(ACCESS_POWER_BASELINE_MONTHLY)",
                    "ACCESS_POWER_BASELINE_MONTHLY.astype(float).to_dict()",
                )
                for ln in c["source"]
            ]

    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("Wrote", NB)


if __name__ == "__main__":
    main()
