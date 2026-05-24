"""Force-update Part 4 cells in notebook 11 from build_nb11_online_ev_hp.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_nb11_online_ev_hp as b11  # noqa: E402

NB = ROOT / "notebooks" / "11_online_MPC_1_EV+HP.ipynb"


def _to_source(text: str) -> list[str]:
    lines = text.splitlines()
    return [ln + "\n" for ln in lines[:-1]] + ([lines[-1] + "\n"] if lines else [])


def _src(cell: dict) -> str:
    return "".join(cell.get("source", []))


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    n_4a = n_intro = n_4c = 0
    for c in nb["cells"]:
        s = _src(c)
        if c["cell_type"] == "markdown" and s.strip().startswith("## 4. Scenario analysis"):
            c["source"] = _to_source(b11.PART4_INTRO)
            n_intro += 1
        elif c["cell_type"] == "code" and s.lstrip().startswith("# Part 4A"):
            c["source"] = _to_source(b11.PART4A)
            c["outputs"] = []
            c["execution_count"] = None
            n_4a += 1
        elif c["cell_type"] == "code" and s.lstrip().startswith("# Part 4C"):
            c["source"] = _to_source(b11.PART4C)
            n_4c += 1

    for c in nb["cells"]:
        if c["cell_type"] == "markdown":
            src = _src(c)
            if src.startswith("# Notebook 11"):
                c["source"] = [
                    ln.replace(
                        "access × SOC floor × inflex stress",
                        "access × SOC floor × inflex forecast p50/p90",
                    )
                    for ln in c["source"]
                ]

    if n_4a == 0:
        raise SystemExit("No Part 4A code cell found")
    NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Updated intro={n_intro}, 4A={n_4a}, 4C={n_4c} -> {NB}")


if __name__ == "__main__":
    main()
