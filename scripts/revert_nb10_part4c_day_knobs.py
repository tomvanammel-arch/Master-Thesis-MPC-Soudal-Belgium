import json
from pathlib import Path

nb_path = Path(__file__).resolve().parents[1] / "notebooks" / "10_online_MPC_1_HP.ipynb"
nb = json.loads(nb_path.read_text(encoding="utf-8"))

old = (
    "# --- knobs: pick one scenario from Part 4A ---\n"
    "# --- plot window (Part 3.1 day plots in viewer) ---\n"
    'WEEK_START = pd.Timestamp("2025-01-20 00:00:00")\n'
    "DAY_OF_WEEK = 3  # 1 = first day of WEEK_START; 3 = Wednesday when week starts Monday\n"
    "\n"
)
new = "# --- knobs: pick one scenario from Part 4A ---\n"

for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if "Part 4C" in src and "run_notebook10_part31" in src:
        if old in src:
            cell["source"] = [src.replace(old, new)]
            nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
            print("Reverted Part 4C plot knobs")
        else:
            print("Part 4C cell: plot knobs not found (already reverted?)")
        break
else:
    raise RuntimeError("Part 4C cell not found")
