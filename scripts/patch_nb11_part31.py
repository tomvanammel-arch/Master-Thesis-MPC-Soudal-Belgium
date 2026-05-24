"""Inject §3.1 markdown + code into notebook 11 and build_nb11_online_ev_hp.py."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "11_online_MPC_1_EV+HP.ipynb"
BUILD = ROOT / "scripts" / "build_nb11_online_ev_hp.py"
CELL_SRC = ROOT / "scripts" / "nb11_part31_cell.py"

MD_31 = """### §3.1 Optimised volumes

Joint **online MPC** (Part 2) vs **deterministic joint** (notebook 04) vs baselines, in the same spirit as notebook 09 (EV) and notebook 10 (HP), with EV+HP on shared axes like notebook 04.

**Inputs**

- Online: `res_evhp_online` or [`output/notebooks/online_ev_hp_15min_notebook_11_part2.csv`](../output/notebooks/online_ev_hp_15min_notebook_11_part2.csv)
- Deterministic: `DET_EV_HP_15MIN` from §1.2 or [`deterministic_ev_hp_15min_notebook_04.csv`](../output/notebooks/deterministic_ev_hp_15min_notebook_04.csv)
- Baseline EV: column `ev` in the online trace; baseline HP: [`output/uncontrolled_hp.csv`](../output/uncontrolled_hp.csv)

**Knobs** — set `WEEK_START` and `DAY_OF_WEEK` in **Part 4C** (or at the top of this cell for a direct §3.1 run). No default week is applied.

**Enforce diagnostics** (nb09-style; requires Part 2 re-run with current `src`): `ev_to_deliver_kwh`, envelope headroom columns, `ev_enforce_active` / `ev_enforce_deferred` markers on daily EV plot.

**Figures**

1. SOC violation summary (online + deterministic)
2. **Weekly** (4 thesis figures): EV power; HP electrical; buffer SOC; grid power (nb09/nb10 style; forecast-stress shading when Part 2 export includes it)
3. **Daily** (4 thesis figures, 05:00–19:00): EV power (+ enforce markers), HP electrical, buffer SOC, grid power
4. **Yearly**: daily uncharged EV; daily EV delivered (online vs det.); full-year grid comparison
5. Optional **MPC 24h window** at `DEBUG_TS` when `RUN_MPC_DEBUG=True` (requires §1.2 + forecasts)
"""


def patch_notebook() -> None:
    code = CELL_SRC.read_text(encoding="utf-8")
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        if c["cell_type"] == "markdown" and src.strip().startswith("### §3.1"):
            cells[i]["source"] = [line + "\n" for line in MD_31.splitlines()]
        if c["cell_type"] == "code" and src.strip().startswith("# §3.1"):
            cells[i]["source"] = [line + "\n" for line in code.splitlines()]
            cells[i]["outputs"] = []
            cells[i]["execution_count"] = None
    NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("Patched", NB)


def patch_build() -> None:
    text = BUILD.read_text(encoding="utf-8")
    code = CELL_SRC.read_text(encoding="utf-8")
    # Escape for raw string in triple quotes
    escaped = code.replace("\\", "\\\\")
    new_part31 = 'PART31 = r\'\'\'' + code + "'''"
    text = re.sub(r"PART31 = r'''[\s\S]*?'''", new_part31, text, count=1)
    old_md = 'cell(md="### §3.1 Optimised volumes"),'
    new_md = 'cell(md="""' + MD_31.replace('"""', '\\"\\"\\"') + '"""),'
    if old_md in text:
        text = text.replace(old_md, new_md, 1)
    else:
        text = text.replace(
            'cell(md="### §3.1 Optimised volumes\\n"),',
            new_md,
            1,
        )
    BUILD.write_text(text, encoding="utf-8")
    print("Patched", BUILD)


if __name__ == "__main__":
    patch_notebook()
    patch_build()
