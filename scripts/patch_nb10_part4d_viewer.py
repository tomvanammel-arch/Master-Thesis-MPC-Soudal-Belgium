"""Point notebook 10 Part 4D at nb10_hp_part34_viewer with forced disk reload."""
import json
from pathlib import Path

nb_path = Path(__file__).resolve().parents[1] / "notebooks" / "10_online_MPC_1_HP.ipynb"
nb = json.loads(nb_path.read_text(encoding="utf-8"))

new_src = '''# Part 4D — Savings vs baseline (thesis style)
#
# Prerequisites: §1.2 (ACCESS_POWER_BASELINE_MONTHLY), Part 4B summary CSV.

from pathlib import Path
import importlib.util
import sys

PROJECT_ROOT = Path("..").resolve()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

_viewer_path = SRC_DIR / "notebook_visualisation" / "nb10_hp_part34_viewer.py"
_spec = importlib.util.spec_from_file_location("nb10_hp_viewer_4d", _viewer_path)
_viewer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_viewer)

print("4D plots from:", _viewer_path)
_viewer.run_notebook10_part4d_scenario_savings(PROJECT_ROOT)
'''

for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if "Part 4D" in src and "run_notebook10_part4d" in src:
        cell["source"] = [new_src]
        cell["outputs"] = []
        cell["execution_count"] = None
        nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print("Patched Part 4D cell (forced reload from disk)")
        break
else:
    raise RuntimeError("Part 4D cell not found")
