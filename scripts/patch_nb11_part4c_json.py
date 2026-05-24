"""Patch notebook 11 Part 4C: JSON export in 4C, not 4B."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "11_online_MPC_1_EV+HP.ipynb"
BUILD = ROOT / "scripts" / "build_nb11_online_ev_hp.py"

part4c_start = "PART4C = r'''"
part4c_end = "'''\n\nPART4D"

text = BUILD.read_text(encoding="utf-8")
i0 = text.index(part4c_start) + len(part4c_start)
i1 = text.index(part4c_end, i0)
part4c_body = text[i0:i1]

nb = json.loads(NB.read_text(encoding="utf-8"))
for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if "Part 4C" in src and "run_notebook11_part31" in src:
        lines = part4c_body.split("\n")
        cell["source"] = [ln + "\n" for ln in lines[:-1]] + ([lines[-1] + "\n"] if lines[-1] else [])
        NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        print("Patched Part 4C in notebook 11")
        break
else:
    raise RuntimeError("Part 4C cell not found")
