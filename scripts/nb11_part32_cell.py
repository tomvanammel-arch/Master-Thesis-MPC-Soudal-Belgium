# §3.2 — Shadow billing (baseline / deterministic joint / online)

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

PROJECT_ROOT = Path("..").resolve()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from billing import load_billing_config, calculate_monthly_bills, calculate_monthly_injection_bills

import importlib
import online_MPC_1_EV_HP_scenario_analysis as _evhp_scen_mod

importlib.reload(_evhp_scen_mod)

build_heating_charging_cost_table = _evhp_scen_mod.build_heating_charging_cost_table
compute_baseline_annual_net_cost_eur = _evhp_scen_mod.compute_baseline_annual_net_cost_eur
compute_deterministic_joint_annual_net_cost_eur = (
    _evhp_scen_mod.compute_deterministic_joint_annual_net_cost_eur
)
compute_nb01_site_reference_bills = _evhp_scen_mod.compute_nb01_site_reference_bills
compute_nb01_site_reference_annual_offtake_eur = (
    _evhp_scen_mod.compute_nb01_site_reference_annual_offtake_eur
)
compute_online_annual_net_cost_eur = _evhp_scen_mod.compute_online_annual_net_cost_eur

NOTEBOOKS_OUTPUT_DIR = PROJECT_ROOT / "output" / "notebooks"


def _load_monthly_exports(bills_global, inj_global, bills_path, inj_path):
    if bills_global in globals() and inj_global in globals():
        bills = globals()[bills_global].copy()
        inj = globals()[inj_global].copy()
    else:
        if not bills_path.exists() or not inj_path.exists():
            raise FileNotFoundError(
                f"Run notebook 04 export + §1.2 first.\n  {bills_path}\n  {inj_path}"
            )
        bills = pd.read_csv(bills_path)
        inj = pd.read_csv(inj_path)
    for _df in (bills, inj):
        if "month" in _df.columns:
            _df["month_key"] = _df["month"].astype(str)
    return bills, inj


base_bills, base_inj = _load_monthly_exports(
    "DET_EV_HP_MONTHLY_BASELINE_BILLS",
    "DET_EV_HP_MONTHLY_BASELINE_INJ",
    NOTEBOOKS_OUTPUT_DIR / "deterministic_ev_hp_monthly_baseline_bills_notebook_04.csv",
    NOTEBOOKS_OUTPUT_DIR / "deterministic_ev_hp_monthly_baseline_injection_notebook_04.csv",
)
det_bills, det_inj = _load_monthly_exports(
    "DET_EV_HP_MONTHLY_BILLS",
    "DET_EV_HP_MONTHLY_INJ",
    NOTEBOOKS_OUTPUT_DIR / "deterministic_ev_hp_monthly_bills_notebook_04.csv",
    NOTEBOOKS_OUTPUT_DIR / "deterministic_ev_hp_monthly_injection_notebook_04.csv",
)

if "table" not in globals():
    raise RuntimeError("Run §1.2 first to build `table`.")

_online_ap_mode = str(globals().get("ONLINE_ACCESS_POWER_MODE", "flex_aware"))
print(f"§3.2 — online scenario uses Part 2 access mode: {_online_ap_mode}")

# --- Online monthly bills (Part 2 summary or recompute) ---
if "summ_evhp_online" in globals() and isinstance(summ_evhp_online, dict):
    online_bills = summ_evhp_online["bills"].copy()
    online_inj = summ_evhp_online["injection_bills"].copy()
else:
    billing_cfg = load_billing_config(str(PROJECT_ROOT / "config" / "billing.yaml"))
    if "res_evhp_online" in globals():
        res_b = res_evhp_online.copy()
    else:
        _csv = NOTEBOOKS_OUTPUT_DIR / "online_ev_hp_15min_notebook_11_part2.csv"
        if not _csv.exists():
            raise FileNotFoundError(f"Run Part 2 first: {_csv}")
        res_b = pd.read_csv(_csv)
    res_b["timestamp"] = pd.to_datetime(res_b["timestamp"], errors="coerce")
    plant = pd.read_csv(PROJECT_ROOT / "data" / "plant1.csv")
    plant["timestamp"] = (
        pd.to_datetime(plant["timestamp"], utc=True, errors="coerce")
        .dt.tz_convert("Europe/Brussels")
        .dt.tz_localize(None)
    )
    plant = plant.sort_values("timestamp").reset_index(drop=True)
    plant = plant[
        (plant["timestamp"] >= pd.Timestamp("2025-01-01"))
        & (plant["timestamp"] < pd.Timestamp("2026-01-01"))
    ].copy()
    ev_col = "ev_applied" if "ev_applied" in res_b.columns else "ev_online_mpc"
    hp_col = "hp_applied" if "hp_applied" in res_b.columns else "hp_applied_kwh"
    plant_seq = plant[["timestamp", "inflex_load", "pv_production", "price"]].reset_index(drop=True)
    res_seq = res_b[[ev_col, hp_col, "access_kw"]].reset_index(drop=True)
    n = min(len(plant_seq), len(res_seq))
    if len(plant_seq) != len(res_seq):
        print(
            f"WARNING: plant vs online length mismatch; using n={n} "
            f"(plant={len(plant_seq)}, res={len(res_seq)})."
        )
    df_online = pd.concat([plant_seq.iloc[:n], res_seq.iloc[:n]], axis=1)
    net_kwh = (
        df_online["inflex_load"].fillna(0.0)
        + df_online[ev_col].fillna(0.0)
        + df_online[hp_col].fillna(0.0)
        - df_online["pv_production"].fillna(0.0)
    )
    df_online["grid_consumption"] = net_kwh.clip(lower=0.0)
    df_online["grid_injection"] = (-net_kwh).clip(lower=0.0)
    online_bills = calculate_monthly_bills(
        df_online,
        billing_cfg,
        volume_col="grid_consumption",
        price_col="price",
        timestamp_col="timestamp",
        access_power_col="access_kw",
    )
    online_inj = calculate_monthly_injection_bills(
        df_online,
        billing_cfg,
        injection_col="grid_injection",
        price_col="price",
        timestamp_col="timestamp",
    )

for _df in (online_bills, online_inj):
    if "month" in _df.columns:
        _df["month_key"] = _df["month"].astype(str)


def _net_monthly(bills, inj, net_col):
    out = bills[["month_key", "total_cost_eur"]].merge(
        inj[["month_key", "injection_net_revenue_eur"]], on="month_key", how="left"
    )
    out[net_col] = out["total_cost_eur"] - out["injection_net_revenue_eur"].fillna(0.0)
    return out


baseline_net_m = _net_monthly(base_bills, base_inj, "baseline_net_cost_eur")
det_net_m = _net_monthly(det_bills, det_inj, "deterministic_net_cost_eur")
online_net_m = _net_monthly(online_bills, online_inj, "online_net_cost_eur")

months = sorted(
    set(base_bills["month_key"]) | set(det_bills["month_key"]) | set(online_bills["month_key"])
)
shadow = pd.DataFrame({"month_key": months}).set_index("month_key")
shadow["baseline_net_cost_eur"] = (
    baseline_net_m.set_index("month_key")["baseline_net_cost_eur"].reindex(shadow.index).values
)
shadow["deterministic_net_cost_eur"] = (
    det_net_m.set_index("month_key")["deterministic_net_cost_eur"].reindex(shadow.index).values
)
shadow["online_net_cost_eur"] = (
    online_net_m.set_index("month_key")["online_net_cost_eur"].reindex(shadow.index).values
)
shadow["deterministic_savings_eur"] = (
    shadow["baseline_net_cost_eur"] - shadow["deterministic_net_cost_eur"]
)
shadow["online_savings_eur"] = shadow["baseline_net_cost_eur"] - shadow["online_net_cost_eur"]

shadow_billing_evhp = shadow.reset_index().rename(columns={"month_key": "month"})

print("Shadow billing monthly comparison (joint EV+HP)")
display(
    shadow_billing_evhp[
        [
            "month",
            "baseline_net_cost_eur",
            "deterministic_net_cost_eur",
            "online_net_cost_eur",
            "deterministic_savings_eur",
            "online_savings_eur",
        ]
    ]
)

def _net_total(bills, inj):
    return float(bills["total_cost_eur"].sum() - inj["injection_net_revenue_eur"].sum())


baseline_total = _net_total(base_bills, base_inj)
deterministic_total = _net_total(det_bills, det_inj)
online_total = _net_total(online_bills, online_inj)

print("\n" + "=" * 80)
print("COST COMPARISON: BASELINE vs DETERMINISTIC vs ONLINE MPC (joint EV+HP)")
print("=" * 80)
print(
    f"\n{'Cost Component':<30} "
    f"{'Baseline (EUR)':>20} {'Deterministic (EUR)':>20} {'Online (EUR)':>20} "
    f"{'Savings det (EUR)':>20} {'Savings online (EUR)':>22}"
)
print("-" * 132)

rows = [
    (
        "Energy Cost",
        base_bills["energy_cost_eur"].sum(),
        det_bills["energy_cost_eur"].sum(),
        online_bills["energy_cost_eur"].sum(),
    ),
    (
        "Spot Cost",
        base_bills["spot_cost_eur"].sum(),
        det_bills["spot_cost_eur"].sum(),
        online_bills["spot_cost_eur"].sum(),
    ),
    (
        "Access Power Cost",
        base_bills["access_cost_eur"].sum(),
        det_bills["access_cost_eur"].sum(),
        online_bills["access_cost_eur"].sum(),
    ),
    (
        "Monthly Peak Cost",
        base_bills["monthly_peak_cost_eur"].sum(),
        det_bills["monthly_peak_cost_eur"].sum(),
        online_bills["monthly_peak_cost_eur"].sum(),
    ),
    (
        "Over-usage Cost",
        base_bills["over_usage_cost_eur"].sum(),
        det_bills["over_usage_cost_eur"].sum(),
        online_bills["over_usage_cost_eur"].sum(),
    ),
    (
        "Injection Revenue",
        -base_inj["injection_net_revenue_eur"].sum(),
        -det_inj["injection_net_revenue_eur"].sum(),
        -online_inj["injection_net_revenue_eur"].sum(),
    ),
]
for name, b, d, o in rows:
    print(
        f"{name:<30} "
        f"{b:>20,.2f} {d:>20,.2f} {o:>20,.2f} "
        f"{b - d:>20,.2f} {b - o:>22,.2f}"
    )

print("-" * 132)
print(
    f"{'NET TOTAL':<30} "
    f"{baseline_total:>20,.2f} {deterministic_total:>20,.2f} {online_total:>20,.2f} "
    f"{baseline_total - deterministic_total:>20,.2f} {baseline_total - online_total:>22,.2f}"
)
print("=" * 80)

savings_det = baseline_total - deterministic_total
savings_online = baseline_total - online_total
if baseline_total > 0:
    print(
        f"Savings vs baseline — deterministic: {savings_det:,.2f} EUR "
        f"({100.0 * savings_det / baseline_total:.2f}%)"
    )
    print(
        f"Savings vs baseline — online:       {savings_online:,.2f} EUR "
        f"({100.0 * savings_online / baseline_total:.2f}%)"
    )

nb01_site_bills, nb01_site_inj = compute_nb01_site_reference_bills(PROJECT_ROOT)
site_ref_eur = float(nb01_site_bills["total_cost_eur"].sum())
_nb01_check = compute_nb01_site_reference_annual_offtake_eur(PROJECT_ROOT)
assert abs(site_ref_eur - _nb01_check) < 0.01

flex_cost_table = build_heating_charging_cost_table(
    baseline_net_eur=baseline_total,
    offline_net_eur=deterministic_total,
    online_net_eur=online_total,
    inflex_site_net_eur=site_ref_eur,
)
heating_charging_cost_evhp = flex_cost_table.copy()

print("\n" + "=" * 80)
print("HEATING & CHARGING COST — incremental over notebook 01 site reference")
print("=" * 80)
print(
    "Reference R: grid_consumption_excl_ev (no EV on meter), conservative access "
    "from full-site grid_consumption peaks (notebook 01 EV billing section)."
)
print(f"Notebook 01 site offtake total_cost_eur: {site_ref_eur:,.2f} EUR")
print(
    "Heating & charging cost = full annual net − R  (EV + HP flex loads, all tariff components)."
)
if _online_ap_mode == "deterministic":
    print(
        "Online column: selected scenario MPC with Online AP (deterministic access); "
        "Offline column: deterministic joint export (notebook 04)."
    )
else:
    print(
        "Online column: selected scenario MPC with Offline AP (flex-aware access); "
        "Offline column: deterministic joint export (notebook 04)."
    )
display(flex_cost_table.style.format("{:,.2f}"))

flex_baseline = float(flex_cost_table.loc["Heating & charging cost [EUR]", "Baseline (uncontrolled)"])
flex_offline = float(flex_cost_table.loc["Heating & charging cost [EUR]", "Offline (deterministic joint)"])
flex_online = float(flex_cost_table.loc["Heating & charging cost [EUR]", "Online (scenario MPC)"])
print(
    f"\nFlex-load savings vs baseline — offline: {flex_baseline - flex_offline:,.2f} EUR; "
    f"online: {flex_baseline - flex_online:,.2f} EUR"
)
print("=" * 80)
if "res_evhp_online" in globals():
    _res_chk = res_evhp_online
elif (NOTEBOOKS_OUTPUT_DIR / "online_ev_hp_15min_notebook_11_part2.csv").exists():
    _res_chk = pd.read_csv(NOTEBOOKS_OUTPUT_DIR / "online_ev_hp_15min_notebook_11_part2.csv")
else:
    _res_chk = None
if _res_chk is not None:
    print("\nAnnual net cost cross-check (helpers):")
    baseline_access = ACCESS_POWER_BASELINE_MONTHLY.astype(float).to_dict()
    print(f"  Baseline:       {compute_baseline_annual_net_cost_eur(PROJECT_ROOT, baseline_access):,.0f} EUR")
    print(
        f"  Deterministic:  "
        f"{compute_deterministic_joint_annual_net_cost_eur(PROJECT_ROOT, None):,.0f} EUR"
    )
    print(f"  Online:         {compute_online_annual_net_cost_eur(PROJECT_ROOT, _res_chk):,.0f} EUR")

baseline_net = baseline_total
det_net = deterministic_total
online_net = online_total

base_idx = base_bills.set_index("month_key")
det_idx = det_bills.set_index("month_key")
on_idx = online_bills.set_index("month_key")
months_plot = shadow_billing_evhp["month"].astype(str).tolist()
x = np.arange(len(months_plot))
width = 0.25

import matplotlib as mpl

THESIS_STYLE = {
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "lines.linewidth": 2.2,
    "savefig.dpi": 300,
}
mpl.rcParams.update(THESIS_STYLE)
C_BLACK = "#000000"
C_KUL_RED = "#b30000"

_peak_base = base_idx.reindex(months_plot)["monthly_peak_kw"].values.astype(float)
_peak_off = det_idx.reindex(months_plot)["monthly_peak_kw"].values.astype(float)
_peak_on = on_idx.reindex(months_plot)["monthly_peak_kw"].values.astype(float)

fig_peak, ax_peak = plt.subplots(figsize=(10, 5))
mpl.rcParams["hatch.color"] = C_BLACK
ax_peak.bar(
    x - width,
    _peak_base,
    width,
    label="Baseline monthly peak",
    color=C_KUL_RED,
    alpha=0.55,
    edgecolor=C_KUL_RED,
    linewidth=0.6,
)
ax_peak.bar(
    x,
    _peak_off,
    width,
    label="Offline monthly peak",
    facecolor="white",
    edgecolor=C_BLACK,
    hatch="///",
    linewidth=0.8,
)
ax_peak.bar(
    x + width,
    _peak_on,
    width,
    label="Online monthly peak",
    color=C_BLACK,
    alpha=0.5,
    edgecolor=C_BLACK,
    linewidth=0.6,
)
ax_peak.set_ylabel("kW")
_peak_ymax = float(np.nanmax([_peak_base, _peak_off, _peak_on]))
ax_peak.set_ylim(2000, max(3000.0, _peak_ymax + 80.0))
ax_peak.set_title("Monthly peak power (2025)")
ax_peak.set_xticks(x)
ax_peak.set_xticklabels(months_plot, rotation=45, ha="right")
ax_peak.set_xlim(-0.5, len(months_plot) - 0.5)
ax_peak.margins(x=0)
ax_peak.tick_params(axis="x", pad=4)
ax_peak.set_xlabel("Month", labelpad=6)
_peak_handles, _peak_labels = ax_peak.get_legend_handles_labels()
ax_peak.legend(
    _peak_handles,
    _peak_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.22),
    ncol=3,
    frameon=False,
)
fig_peak.subplots_adjust(bottom=0.28, top=0.90)
plt.tight_layout()
plt.show()

fig_acc_cmp, ax2 = plt.subplots(figsize=(10, 5))
# `table` is indexed by month; access powers are columns (see §1.2)
_cons = table["access_power_conservative"].astype(float)
ax2.bar(
    x - width,
    _cons.reindex(months_plot).values,
    width,
    label="Baseline access power",
    color=C_KUL_RED,
    alpha=0.55,
    edgecolor=C_KUL_RED,
    linewidth=0.6,
)
ax2.bar(
    x,
    det_idx.reindex(months_plot)["access_power_kw"].values,
    width,
    label="Offline access power",
    facecolor="white",
    edgecolor=C_BLACK,
    hatch="///",
    linewidth=0.8,
)
ax2.bar(
    x + width,
    on_idx.reindex(months_plot)["access_power_kw"].values,
    width,
    label="Online access power",
    color=C_BLACK,
    alpha=0.5,
    edgecolor=C_BLACK,
    linewidth=0.6,
)
ax2.set_ylabel("kW")
_acc_ymax = float(
    np.nanmax(
        [
            _cons.reindex(months_plot).values,
            det_idx.reindex(months_plot)["access_power_kw"].values,
            on_idx.reindex(months_plot)["access_power_kw"].values,
        ]
    )
)
ax2.set_ylim(2000, max(3200.0, _acc_ymax + 80.0))
ax2.set_title("Monthly access power (2025)")
ax2.set_xticks(x)
ax2.set_xticklabels(months_plot, rotation=45, ha="right")
ax2.set_xlim(-0.5, len(months_plot) - 0.5)
ax2.margins(x=0)
ax2.set_xlabel("Month", labelpad=6)
_acc_handles, _acc_labels = ax2.get_legend_handles_labels()
ax2.legend(
    _acc_handles,
    _acc_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.22),
    ncol=3,
    frameon=False,
)
fig_acc_cmp.subplots_adjust(bottom=0.28, top=0.90)
plt.tight_layout()
plt.show()

base_spot = base_idx.reindex(months_plot)["spot_cost_eur"].values
base_access = base_idx.reindex(months_plot)["access_cost_eur"].values
base_peak = (
    base_idx.reindex(months_plot)["monthly_peak_cost_eur"].values
    + base_idx.reindex(months_plot)["over_usage_cost_eur"].values
)
det_spot = det_idx.reindex(months_plot)["spot_cost_eur"].values
det_access = det_idx.reindex(months_plot)["access_cost_eur"].values
det_peak = (
    det_idx.reindex(months_plot)["monthly_peak_cost_eur"].values
    + det_idx.reindex(months_plot)["over_usage_cost_eur"].values
)
on_spot = on_idx.reindex(months_plot)["spot_cost_eur"].values
on_access = on_idx.reindex(months_plot)["access_cost_eur"].values
on_peak = (
    on_idx.reindex(months_plot)["monthly_peak_cost_eur"].values
    + on_idx.reindex(months_plot)["over_usage_cost_eur"].values
)

savings = pd.DataFrame(
    {
        "month": months_plot,
        "spot_savings_det": base_spot - det_spot,
        "access_power_savings_det": base_access - det_access,
        "peak_savings_det": base_peak - det_peak,
        "spot_savings_online": base_spot - on_spot,
        "access_power_savings_online": base_access - on_access,
        "peak_savings_online": base_peak - on_peak,
    }
)
savings["total_savings_det"] = savings[
    ["spot_savings_det", "access_power_savings_det", "peak_savings_det"]
].sum(axis=1)
savings["total_savings_online"] = savings[
    ["spot_savings_online", "access_power_savings_online", "peak_savings_online"]
].sum(axis=1)

fig, ax = plt.subplots(figsize=(16, 6))
w = 0.35
ax.bar(x - w / 2, savings["spot_savings_det"], w, label="Det — Spot", color="tab:blue", alpha=0.8)
ax.bar(
    x - w / 2,
    savings["access_power_savings_det"],
    w,
    bottom=savings["spot_savings_det"],
    label="Det — Access",
    color="tab:green",
    alpha=0.8,
)
ax.bar(
    x - w / 2,
    savings["peak_savings_det"],
    w,
    bottom=savings["spot_savings_det"] + savings["access_power_savings_det"],
    label="Det — Peak",
    color="tab:orange",
    alpha=0.8,
)
ax.bar(x + w / 2, savings["spot_savings_online"], w, label="Online — Spot", color="tab:blue", alpha=0.4)
ax.bar(
    x + w / 2,
    savings["access_power_savings_online"],
    w,
    bottom=savings["spot_savings_online"],
    label="Online — Access",
    color="tab:green",
    alpha=0.4,
)
ax.bar(
    x + w / 2,
    savings["peak_savings_online"],
    w,
    bottom=savings["spot_savings_online"] + savings["access_power_savings_online"],
    label="Online — Peak",
    color="tab:orange",
    alpha=0.4,
)
ax.set_xlabel("Month")
ax.set_ylabel("Savings vs baseline (EUR)")
ax.set_title("Monthly savings breakdown: deterministic vs online (joint EV+HP)")
ax.set_xticks(x)
ax.set_xticklabels(savings["month"], rotation=45, ha="right")
ax.legend(loc="upper left", ncol=2)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.show()

print("\nSavings breakdown (first rows):")
display(
    savings[
        [
            "month",
            "spot_savings_det",
            "access_power_savings_det",
            "peak_savings_det",
            "total_savings_det",
            "spot_savings_online",
            "access_power_savings_online",
            "peak_savings_online",
            "total_savings_online",
        ]
    ].head()
)
