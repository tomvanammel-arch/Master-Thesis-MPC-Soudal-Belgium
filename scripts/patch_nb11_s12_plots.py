"""Patch notebook 11 §1.2 code + markdown from build_nb11_online_ev_hp.S12."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_nb11_online_ev_hp import S12  # noqa: E402

NB = ROOT / "notebooks" / "11_online_MPC_1_EV+HP.ipynb"
MD_12 = """### §1.2 Access power selection (grid-based)

Same three plots as notebook 10 §1.2:

1. **Monthly access power (2025)** — stacked actual peak excl. EV + EV peak increment + HP worst-case; step lines for online, baseline, offline access power.
2. **Daily electrical headroom, EV and HP demand (2025)** — flex headroom (full day); daily EV; actual HP; HP design reference.
3. **Daily headroom, EV and HP demand (weekdays 07:00–17:00, 2025)** — flex headroom in EV window; actual HP; daily EV; HP design reference.
4. **Daily headroom utilisation (2025)** — full day; actual (EV + HP) vs design-reference (EV + worst HP); 70% limit.
5. **Daily headroom utilisation (weekdays 07:00–17:00, 2025)** — EV window; same definitions; 70% limit.

**Flex-aware** = max(cum-max `grid_consumption_excl_ev` (M−1) + 20 kW, 70%-cap search on M−1) → `ACCESS_POWER_FLEX_DICT`. **Baseline** = notebook 04 `baseline_access_power_kw`. **Offline** = notebook 04 `access_power_kw`.

Run notebook 04 export cell first."""


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    md_idx = code_idx = None
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "markdown":
            continue
        src = "".join(c.get("source", []))
        if "### §1.2" in src or "### 1.2" in src:
            md_idx = i
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c.get("source", []))
        if src.startswith("# §1.2") or src.startswith("# Part 1.2"):
            code_idx = i
            break
    if md_idx is None or code_idx is None:
        raise RuntimeError(f"Could not find §1.2 cells (md={md_idx}, code={code_idx})")

    nb["cells"][md_idx]["source"] = [line + "\n" for line in MD_12.splitlines()]
    nb["cells"][code_idx]["source"] = [line + "\n" for line in S12.splitlines()]
    nb["cells"][code_idx]["outputs"] = []
    nb["cells"][code_idx]["execution_count"] = None

    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Patched {NB} §1.2 markdown cell {md_idx}, code cell {code_idx}")


if __name__ == "__main__":
    main()
