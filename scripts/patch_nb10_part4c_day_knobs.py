"""Add WEEK_START / DAY_OF_WEEK knobs to notebook 10 Part 4C cell."""
import json
from pathlib import Path

nb_path = Path(__file__).resolve().parents[1] / "notebooks" / "10_online_MPC_1_HP.ipynb"
nb = json.loads(nb_path.read_text(encoding="utf-8"))

for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if "Part 4C" not in src or "run_notebook10_part31" not in src:
        continue
    needle = "# --- knobs: pick one scenario from Part 4A ---\n"
    if "DAY_OF_WEEK" in src:
        print("Already patched")
        break
    knob = (
        "# --- plot window (Part 3.1 day plots in viewer) ---\n"
        'WEEK_START = pd.Timestamp("2025-01-20 00:00:00")\n'
        "DAY_OF_WEEK = 3  # 1 = first day of WEEK_START; 3 = Wednesday when week starts Monday\n"
        "\n"
    )
    if needle not in src:
        raise RuntimeError("needle not found in Part 4C cell")
    cell["source"] = [src.replace(needle, needle + knob)]
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Patched Part 4C cell")
    break
else:
    raise RuntimeError("Part 4C cell not found")
