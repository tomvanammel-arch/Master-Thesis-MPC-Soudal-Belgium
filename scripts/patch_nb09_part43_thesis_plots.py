"""Patch notebook 09 Part 4.3 cell: thesis EV/grid day plots + fix week _day_merge bug."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "09_online_MPC_1_EV.ipynb"

IMPORT_BLOCK = """
from notebook_visualisation import nb09_ev_part34_viewer as _ev_thesis
import importlib
importlib.reload(_ev_thesis)
"""

# Remove buggy week enforce block
OLD_WEEK_ENFORCE = """
if "ev_enforce_active" in _day_merge.columns:
    _enf = _day_merge[_day_merge["ev_enforce_active"] > 0.5]
    if len(_enf):
        ax1.scatter(
            _enf["timestamp"],
            _enf["ev_power_kw_online"],
            s=40,
            c="red",
            zorder=5,
            label="Enforce active",
        )
if "ev_enforce_deferred" in _day_merge.columns:
    _def = _day_merge[_day_merge["ev_enforce_deferred"] > 0.5]
    if len(_def):
        ax1.scatter(
            _def["timestamp"],
            _def["ev_power_kw_online"],
            s=40,
            c="gold",
            zorder=5,
            label="Enforce deferred",
        )

"""

INSERT_AFTER_DAY_MERGE = """
# --- Thesis-style standalone day figures (nb10 §4C pattern) ---
""" + IMPORT_BLOCK + """
_ev_thesis.plot_thesis_day_ev_power(_day_merge, selected_day_start, show_window=True)
_ev_thesis.plot_thesis_day_grid_power(_day_merge, selected_day_start)

"""

OLD_AX1 = """ax1.step(_day_merge["timestamp"], _day_merge["ev_demand_actual"] * 4.0, label="Uncontrolled EV demand (kW)",
         color="red", linewidth=1.5, where="post")
ax1.step(_day_merge["timestamp"], _day_merge["ev_charge_power"], label="Deterministic EV power (kW)",
         color="tab:green", linewidth=2, where="post")
ax1.step(_day_merge["timestamp"], _day_merge["ev_power_kw_online"], label="Online MPC EV power (kW)",
         color="tab:blue", linewidth=2, where="post")
ax1.step(_day_merge["timestamp"], _day_merge["ev_power_envelope"], label="Deterministic EV envelope (kW)",
         color="tab:orange", linewidth=1.5, linestyle="--", where="post")
ax1.set_ylabel("Power (kW)", fontweight="bold")
ax1.set_title(f"[1] EV charging power (slack={SLACK_SELECT_MIN} min) — {selected_day_start:%Y-%m-%d}", fontweight="bold")
if "was_clipped" in _day_merge.columns:
    _clip = _day_merge[_day_merge["was_clipped"] > 0.5]
    if len(_clip):
        ax1.scatter(_clip["timestamp"], _clip["ev_power_kw_online"], s=35, c="black", marker="x", zorder=6, label="Clipped")
if "ev_enforce_active" in _day_merge.columns:
    _enf = _day_merge[_day_merge["ev_enforce_active"] > 0.5]
    if len(_enf):
        ax1.scatter(_enf["timestamp"], _enf["ev_power_kw_online"], s=40, c="red", zorder=5, label="Enforce active")
if "ev_enforce_deferred" in _day_merge.columns:
    _def = _day_merge[_day_merge["ev_enforce_deferred"] > 0.5]
    if len(_def):
        ax1.scatter(_def["timestamp"], _def["ev_power_kw_online"], s=40, c="gold", zorder=5, label="Enforce deferred")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)
"""

NEW_AX1 = f"""_ev_thesis.plot_thesis_day_ev_power(
    _day_merge,
    selected_day_start,
    ax=ax1,
    standalone=False,
    show_window=True,
    show_enforce_markers=True,
    title=f"[1] EV charging power (slack={{SLACK_SELECT_MIN}} min) — {{selected_day_start:%Y-%m-%d}}",
    xlim=(x_start, x_end),
)
if "was_clipped" in _day_merge.columns:
    _clip = _day_merge[_day_merge["was_clipped"] > 0.5]
    if len(_clip):
        ax1.scatter(
            _clip["timestamp"],
            _clip["ev_power_kw_online"],
            s=35,
            c="black",
            marker="x",
            zorder=6,
            label="Clipped",
        )
        ax1.legend(loc="upper left", fontsize=9)
"""

OLD_AX5 = """ax5.step(_day_merge["timestamp"], _day_merge["p_grid_plan_kw"], label="Planned grid power (kW) – p_grid_plan_kw",
         color="tab:cyan", linewidth=2, where="post")
ax5.step(_day_merge["timestamp"], _day_merge["grid_power_online"], label="Actual grid power (kW) – grid_power_online",
         color="tab:blue", linewidth=2, linestyle="--", where="post")
ax5.step(_day_merge["timestamp"], _day_merge["p_limit_kw"], label="Clipping limit (kW) – p_limit_kw",
         color="tab:red", linewidth=1.5, linestyle=":", where="post")
ax5.set_ylabel("Power (kW)", fontweight="bold")
ax5.set_title("[5] Grid power vs clipping limit", fontweight="bold")
ax5.legend(loc="lower left")
ax5.grid(True, alpha=0.3)
"""

NEW_AX5 = """_ev_thesis.plot_thesis_day_grid_power(
    _day_merge,
    selected_day_start,
    ax=ax5,
    standalone=False,
    title="[5] Grid power — baseline / online / offline / access",
    xlim=(x_start, x_end),
)
ax5.legend(loc="lower left", fontsize=9)
"""


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cell = None
    for c in nb["cells"]:
        src = "".join(c.get("source", []))
        if "Part 4.3" in src and "One-cell" in src:
            cell = c
            break
    if cell is None:
        raise RuntimeError("Part 4.3 cell not found")

    src = "".join(cell["source"])
    if OLD_WEEK_ENFORCE in src:
        src = src.replace(OLD_WEEK_ENFORCE, "\n")
    else:
        print("NOTE: week enforce block already removed or changed")

    marker = "fig, axes = plt.subplots(8, 1, figsize=(16, 26), sharex=True)"
    if INSERT_AFTER_DAY_MERGE.strip() not in src:
        if "_ev_thesis.plot_thesis_day_ev_power" not in src:
            src = src.replace(marker, INSERT_AFTER_DAY_MERGE + marker)
        else:
            print("NOTE: thesis standalone block already present")
    else:
        pass

    if "_ev_thesis.plot_thesis_day_ev_power(\n    _day_merge,\n    selected_day_start,\n    ax=ax1" not in src:
        if OLD_AX1 not in src:
            raise RuntimeError("ax1 block not found")
        src = src.replace(OLD_AX1, NEW_AX1)

    if "_ev_thesis.plot_thesis_day_grid_power(\n    _day_merge,\n    selected_day_start,\n    ax=ax5" not in src:
        if OLD_AX5 not in src:
            raise RuntimeError("ax5 block not found")
        src = src.replace(OLD_AX5, NEW_AX5)

  # Import must appear before 8-panel if we only inserted before fig,axes - ensure import before week is NOT needed
  # Import is in INSERT_AFTER_DAY_MERGE which is before 8-panel - good

    cell["source"] = [line + "\n" for line in src.splitlines()]
    if cell["source"] and not cell["source"][-1].endswith("\n"):
        cell["source"][-1] += "\n"

    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Patched", NB_PATH)


if __name__ == "__main__":
    main()
