"""Patch notebook 09: §2 markdown + 3.1/4.3 headroom diagnostic plots."""
from __future__ import annotations

import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "09_online_MPC_1_EV.ipynb"

OLD_MD = (
    "- Actuator chain: **clipper** (Eq. 3.50 in MPC; `min(access, peak_sofar)` in catch-up) → optional "
    "**enforce daily EV demand** (from 12:00 in MPC; in catch-up when slack > 0) to restore clipped EV "
    "up to the physical envelope, even if **access** or **peak_sofar** are exceeded.\n"
)

NEW_MD = (
    "- Actuator chain: **clipper** (Eq. 3.50 in MPC; `min(access, peak_sofar)` in catch-up) → optional "
    "**enforce daily EV demand** (12:00–17:00): after a clip, add only the **minimum** kWh needed when "
    "`ev_to_deliver` cannot still be met using envelope headroom through **17:00**; otherwise defer to "
    "catch-up (`[17:00−slack, 17:00)`). Diagnostics: `ev_envelope_remaining_kwh`, "
    "`ev_envelope_headroom_after_kwh`, `ev_envelope_feasible`, `ev_enforce_deferred`.\n"
)

WEEK_SNIP = (
    'ax3.plot(_week_merge["timestamp"], _week_merge["ev_to_deliver_kwh"], label="Online remaining (kWh)",\n'
    '         color="tab:blue", linewidth=2, linestyle="--")\n'
)

WEEK_SNIP_NEW = WEEK_SNIP + (
    'if "ev_envelope_remaining_kwh" in _week_merge.columns:\n'
    '    ax3.plot(_week_merge["timestamp"], _week_merge["ev_envelope_remaining_kwh"],\n'
    '             label="Envelope headroom to 17:00 (kWh)", color="tab:orange", linewidth=1.5, linestyle=":")\n'
)

DAY_HEADROOM_BLOCK = """
if "ev_envelope_remaining_kwh" in _day_merge.columns:
    ax3.plot(
        _day_merge["timestamp"],
        _day_merge["ev_envelope_remaining_kwh"],
        label="Envelope headroom to 17:00 (kWh)",
        color="tab:orange",
        linewidth=1.5,
        linestyle=":",
    )
if "ev_envelope_headroom_after_kwh" in _day_merge.columns:
    ax3.plot(
        _day_merge["timestamp"],
        _day_merge["ev_envelope_headroom_after_kwh"],
        label="Headroom after step (kWh)",
        color="tab:cyan",
        linewidth=1.2,
        linestyle="-.",
    )
"""

AX1_MARKERS = """
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


def patch_source(src: str) -> str:
    out = src.replace(OLD_MD, NEW_MD)
    if WEEK_SNIP in out and "Envelope headroom to 17:00" not in out:
        out = out.replace(WEEK_SNIP, WEEK_SNIP_NEW)
    if (
        '_day_merge["ev_to_deliver_kwh"]' in out
        and "Envelope headroom to 17:00" not in out
        and "det_remaining_kwh vs ev_to_deliver" in out
    ):
        marker = 'ax3.set_ylabel("Remaining energy (kWh)"'
        idx = out.find(marker)
        if idx != -1:
            out = out[:idx] + DAY_HEADROOM_BLOCK + "\n" + out[idx:]
    if "ev_power_kw_online" in out and "Enforce deferred" not in out and "fig, axes = plt.subplots(8" in out:
        marker = "ax1.legend(loc=\"upper left\")"
        idx = out.find(marker)
        if idx != -1:
            out = out[:idx] + AX1_MARKERS + "\n" + out[idx:]
    # Part 4.3: ax2 has remaining energy (ax3 is spot price)
    ax2_marker = (
        'ax2.plot(_day_merge["timestamp"], ev_charged_cum_kwh, label="Online charged (cum, kWh)",\n'
        '         color="tab:orange", linewidth=1.8)\n'
    )
    ax2_headroom = ax2_marker + DAY_HEADROOM_BLOCK.replace("_day_merge", "_day_merge").replace(
        "ax3.", "ax2."
    )
    if ax2_marker in out and "Headroom after step" not in out and "[3] Electricity spot price" in out:
        out = out.replace(ax2_marker, ax2_headroom)
    return out


def to_cell_source(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    if not lines:
        return []
    if not text.endswith("\n"):
        lines[-1] = lines[-1].rstrip("\n") + "\n" if lines else [text + "\n"]
    return lines


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    n = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            src = "".join(cell.get("source", []))
            if OLD_MD in src:
                cell["source"] = to_cell_source(src.replace(OLD_MD, NEW_MD))
                n += 1
            continue
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        new = patch_source(src)
        if new != src:
            cell["source"] = to_cell_source(new)
            n += 1
    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Patched {n} cells in {NB}")


if __name__ == "__main__":
    main()
