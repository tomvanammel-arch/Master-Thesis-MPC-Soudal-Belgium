"""Add worst-case HP/headroom utilisation lines to §1.2."""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "10_online_MPC_1_HP.ipynb"
nb = json.loads(NB.read_text(encoding="utf-8"))
src = "".join(nb["cells"][4]["source"])

old = """util_cons_pct = np.where(
    daily_h_cons > 1e-9, 100.0 * daily_hp_need_actual_kwh / daily_h_cons, np.nan
)
util_flex_pct = np.where(
    daily_h_flex > 1e-9, 100.0 * daily_hp_need_actual_kwh / daily_h_flex, np.nan
)"""

new = """util_cons_pct = np.where(
    daily_h_cons > 1e-9, 100.0 * daily_hp_need_actual_kwh / daily_h_cons, np.nan
)
util_flex_pct = np.where(
    daily_h_flex > 1e-9, 100.0 * daily_hp_need_actual_kwh / daily_h_flex, np.nan
)
util_worst_cons_pct = np.where(
    daily_h_cons > 1e-9, 100.0 * hp_need_worst_day_kwh / daily_h_cons, np.nan
)
util_worst_flex_pct = np.where(
    daily_h_flex > 1e-9, 100.0 * hp_need_worst_day_kwh / daily_h_flex, np.nan
)"""

old_plot = """ax_u.step(
    hr_dates, util_cons_pct, where="post", color=C_BLUE, linewidth=LW_DAILY,
    label="Actual HP / H (conservative)",
)
ax_u.step(
    hr_dates, util_flex_pct, where="post", color=C_GREEN, linewidth=LW_DAILY, linestyle="--",
    label="Actual HP / H (flex-aware)",
)
ax_u.axhline(70.0, color=C_BLACK, linewidth=1.5, linestyle="-", label="70% reference")
ax_u.set_ylabel("Utilisation [%]")
ax_u.set_xlabel("Date (2024–2025)")
ax_u.set_title("§1.2 Daily headroom utilisation — actual HP electrical need")"""

new_plot = """ax_u.step(
    hr_dates, util_cons_pct, where="post", color=C_BLUE, linewidth=LW_DAILY,
    label="Actual HP / H (conservative)",
)
ax_u.step(
    hr_dates, util_flex_pct, where="post", color=C_GREEN, linewidth=LW_DAILY, linestyle="--",
    label="Actual HP / H (flex-aware)",
)
ax_u.step(
    hr_dates, util_worst_cons_pct, where="post", color=C_BLUE, linewidth=LW_DAILY, linestyle=":",
    label=f"Worst-case HP / H (conservative, {hp_need_worst_day_kwh:.0f} kWh/d)",
)
ax_u.step(
    hr_dates, util_worst_flex_pct, where="post", color=C_GREEN, linewidth=LW_DAILY, linestyle=":",
    label=f"Worst-case HP / H (flex-aware, {hp_need_worst_day_kwh:.0f} kWh/d)",
)
ax_u.axhline(70.0, color=C_BLACK, linewidth=1.5, linestyle="-", label="70% reference")
ax_u.set_ylabel("Utilisation [%]")
ax_u.set_xlabel("Date (2024–2025)")
ax_u.set_title("§1.2 Daily headroom utilisation")"""

if old not in src:
    raise SystemExit("util block not found")
if old_plot not in src:
    raise SystemExit("plot block not found")
src = src.replace(old, new).replace(old_plot, new_plot)
nb["cells"][4]["source"] = [src]

# markdown cell 3
md = "".join(nb["cells"][3]["source"])
old_md = "Utilisation plots use **actual** daily HP need vs conservative / flex-aware headroom (70% reference)."
new_md = (
    "Utilisation (%): actual HP / headroom and worst-case HP / headroom "
    "(conservative & flex-aware), vs **70%** reference."
)
if old_md in md:
    nb["cells"][3]["source"] = [md.replace(old_md, new_md)]

NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Patched", NB)
