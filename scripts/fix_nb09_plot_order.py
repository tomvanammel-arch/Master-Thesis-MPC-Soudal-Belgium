"""Fix Part 3.1 / 4.3 plots: use correct DataFrames and load slack CSV for Part 3."""
from __future__ import annotations

import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "09_online_MPC_1_EV.ipynb"

# --- Part 3: optional slack CSV load ---
OLD_LOAD = (
    "# Allow plotting without re-running the lengthy online MPC simulation:\n"
    "# - If `results_ev_online` exists in memory, use it.\n"
    "# - Otherwise load the exported 15-min CSV from Part 2.\n"
    "if \"results_ev_online\" in globals():\n"
    "    results_df = results_ev_online.copy()\n"
    "else:\n"
    "    from pathlib import Path\n"
    "\n"
    "    csv_path = Path(\"../output/notebooks/online_ev_15min_notebook_09.csv\")\n"
)

NEW_LOAD = (
    "# Allow plotting without re-running the lengthy online MPC simulation.\n"
    "# Set PLOT_SLACK_MIN (e.g. 105) to load Part 4 per-slack CSV; None uses Part 2 export.\n"
    "PLOT_SLACK_MIN = 105  # None | 0 | 15 | ... | 105\n"
    "\n"
    "from pathlib import Path\n"
    "\n"
    "if PLOT_SLACK_MIN is not None:\n"
    "    csv_path = Path(\n"
    "        f\"../output/notebooks/online_ev_15min_notebook_09_{int(PLOT_SLACK_MIN)}_min_slack.csv\"\n"
    "    )\n"
    "    if not csv_path.exists():\n"
    "        raise FileNotFoundError(\n"
    "            f\"Slack CSV not found: {csv_path}. Run Part 4.1 sensitivity first.\"\n"
    "        )\n"
    "    results_df = pd.read_csv(csv_path)\n"
    "    print(f\"[Part 3] Loaded slack={PLOT_SLACK_MIN} min from {csv_path.name}\")\n"
    "elif \"results_ev_online\" in globals():\n"
    "    results_df = results_ev_online.copy()\n"
    "    print(\"[Part 3] Using in-memory results_ev_online (Part 2 slack)\")\n"
    "else:\n"
    "    csv_path = Path(\"../output/notebooks/online_ev_15min_notebook_09.csv\")\n"
)

# Remove misplaced weekly markers (before _day_merge exists)
WEEKLY_MARKER_BLOCK = '''if "ev_enforce_active" in _day_merge.columns:
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

'''

DAILY_MARKER_BLOCK = '''
# Clip / enforce diagnostics (daily plot)
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

'''

WEEKLY_HEADROOM_DAY = '''if "ev_envelope_remaining_kwh" in _day_merge.columns:
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

'''

WEEKLY_HEADROOM_WEEK = '''if "ev_envelope_remaining_kwh" in _week_merge.columns:
    ax3.plot(
        _week_merge["timestamp"],
        _week_merge["ev_envelope_remaining_kwh"],
        label="Envelope headroom to 17:00 (kWh)",
        color="tab:orange",
        linewidth=1.5,
        linestyle=":",
    )

'''

DAILY_HEADROOM_AX3 = '''if "ev_envelope_remaining_kwh" in _day_merge.columns:
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

'''

DAILY_AX1_ANCHOR = (
    'ax1.set_ylabel("Power (kW)", fontsize=11, fontweight="bold")\n'
    'ax1.set_title(\n'
    '    f"[1] EV charging power – uncontrolled vs deterministic vs online vs envelope\\n"\n'
)

PART2_RELOAD = (
    "from online_MPC_1_EV import run_ev_online_mpc_1\n"
    "\n"
    "results_ev_online, summary_ev_online = run_ev_online_mpc_1(\n"
)

PART2_RELOAD_NEW = (
    "import importlib\n"
    "import online_MPC_1_EV\n"
    "importlib.reload(online_MPC_1_EV)\n"
    "from online_MPC_1_EV import run_ev_online_mpc_1\n"
    "\n"
    "results_ev_online, summary_ev_online = run_ev_online_mpc_1(\n"
)


def patch_cell_source(src: str) -> str:
    out = src
    if OLD_LOAD in out and "PLOT_SLACK_MIN" not in out:
        out = out.replace(OLD_LOAD, NEW_LOAD)
    out = out.replace(WEEKLY_MARKER_BLOCK, "")
    out = out.replace(WEEKLY_HEADROOM_DAY, WEEKLY_HEADROOM_WEEK)
    # Daily 8-panel: add markers once (before first ax1.legend after envelope step)
    daily_anchor = (
        '    label="Deterministic EV envelope (kW)",\n'
        '    color="tab:orange",\n'
        '    linewidth=1.5,\n'
        '    linestyle="--",\n'
        '    where="post",\n'
        ")\n"
        'ax1.set_ylabel("Power (kW)", fontsize=11, fontweight="bold")\n'
        'ax1.set_title(\n'
        '    f"[1] EV charging power – uncontrolled vs deterministic vs online vs envelope\\n"'
    )
    daily_part = out.split("_day_merge = day_data.merge", 1)
    daily_tail = daily_part[1] if len(daily_part) > 1 else ""
    if daily_anchor in out and "Enforce deferred" not in daily_tail:
        out = out.replace(
            daily_anchor,
            daily_anchor.replace(
                'ax1.set_ylabel("Power (kW)"',
                DAILY_MARKER_BLOCK + 'ax1.set_ylabel("Power (kW)"',
                1,
            ),
            1,
        )
    # Daily ax2: envelope headroom on remaining-energy panel
    ax2_anchor = (
        '    label="Online charged (cum, kWh)",\n'
        '    color="tab:orange",\n'
        '    linewidth=1.8,\n'
        ")\n\n"
        'ax2.set_ylabel("Remaining (kWh)"'
    )
    if ax2_anchor in daily_tail and "Headroom after step" not in daily_tail:
        headroom_ax2 = (
            ")\n\n"
            'if "ev_envelope_remaining_kwh" in _day_merge.columns:\n'
            '    ax2.plot(_day_merge["timestamp"], _day_merge["ev_envelope_remaining_kwh"],\n'
            '             label="Envelope headroom to 17:00 (kWh)", color="tab:orange", linewidth=1.5, linestyle=":")\n'
            'if "ev_envelope_headroom_after_kwh" in _day_merge.columns:\n'
            '    ax2.plot(_day_merge["timestamp"], _day_merge["ev_envelope_headroom_after_kwh"],\n'
            '             label="Headroom after step (kWh)", color="tab:cyan", linewidth=1.2, linestyle="-.")\n\n'
            'ax2.set_ylabel("Remaining (kWh)"'
        )
        out = out.replace(ax2_anchor, headroom_ax2, 1)
    # Daily ax3 headroom if missing in 8-panel section (ax3 is remaining energy in 3.1 daily)
    if DAILY_AX1_ANCHOR in out and "Headroom after step" not in out.split("_day_merge = day_data.merge")[1]:
        pass  # handled by marker block location
    return out


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    n = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        new = patch_cell_source(src)
        if PART2_RELOAD in new and PART2_RELOAD_NEW not in new:
            new = new.replace(PART2_RELOAD, PART2_RELOAD_NEW)
        if new != src:
            cell["source"] = [line + "\n" for line in new.splitlines()]
            if new and not new.endswith("\n"):
                cell["source"][-1] = cell["source"][-1].rstrip("\n")
            n += 1
    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Patched {n} cells")


if __name__ == "__main__":
    main()
