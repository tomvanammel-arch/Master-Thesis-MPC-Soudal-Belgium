"""Extract PART32 from build_nb11_online_ev_hp.py to scripts/nb11_part32_cell.py."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
text = (ROOT / "scripts/build_nb11_online_ev_hp.py").read_text(encoding="utf-8")
m = re.search(r"PART32 = r'''(.+?)'''\n\nPART4A", text, re.DOTALL)
if not m:
    raise SystemExit("PART32 not found")
(ROOT / "scripts/nb11_part32_cell.py").write_text(m.group(1).strip() + "\n", encoding="utf-8")
print("Wrote nb11_part32_cell.py")
