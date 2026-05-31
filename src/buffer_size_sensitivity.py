"""
Buffer Size Sensitivity Analysis

This script runs deterministic MPC optimization for different buffer sizes
and compares the cost of heating for each scenario.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Optional
from optimization import deterministic_mpc_hp
from billing import load_billing_config, calculate_monthly_bills, calculate_monthly_injection_bills
from heat_pump_load import calculate_heat_pump_load, interpolate_cop, load_hp_config
import copy


def run_buffer_size_sensitivity(
    df: pd.DataFrame,
    billing_config_path: str,
    hp_config_path: str,
    buffer_sizes: list = [50, 100, 150, 200, 250, 300, 350, 400],
    output_dir: Optional[str] = None,
    timestamp_col: str = "timestamp",
    pv_col: str = "pv_production",
    inflex_load_col: str = "inflex_load",
    price_col: str = "price",
    ev_col: str = "ev",
    thermal_load_col: str = "thermal_load",
    outdoor_temp_col: str = "outdoor_temperature"
):
    """
    Run optimization for different buffer sizes and compare results.
    
    Args:
        df: Input dataframe with plant data
        billing_config_path: Path to billing.yaml config file
        hp_config_path: Path to hp.yaml config file
        buffer_sizes: List of buffer sizes in m³ to test
        output_dir: Directory to save results
        timestamp_col: Column name for timestamp
        pv_col: Column name for PV production
        inflex_load_col: Column name for inflexible load
        price_col: Column name for price
        ev_col: Column name for EV charging
        thermal_load_col: Column name for thermal load
        outdoor_temp_col: Column name for outdoor temperature
    
    Returns:
        Dictionary with results for each buffer size including:
        - results_df: Optimization results dataframe
        - bills: Monthly bills dataframe
        - injection: Monthly injection dataframe
        - net_total: Total net cost
        - hp_volume_mwh: Total HP electrical volume
        - cost_of_heating_eur_per_mwh: Cost of heating
    """
    # Load billing config (same for all runs)
    billing_config = load_billing_config(billing_config_path)
    
    # Load base HP config
    with open(hp_config_path, 'r', encoding='utf-8') as f:
        base_hp_config = yaml.safe_load(f)
    
    # Create output directory
    if output_dir is None:
        project_root = Path(__file__).resolve().parents[1]
        output_path = project_root / "output" / "optimised_ts"
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Store results for each buffer size
    results = {}
    
    # Baseline (no HP): same conservative access power as notebook 03
    # (previous max peak up to M-1 + 20 kW, January seeded from 2024 spillover).
    print("Calculating baseline costs (no HP)...")
    MARGIN_KW = 20.0
    BASELINE_2024_PEAK_GRID_KW = 2663.5

    df_baseline = df.copy()
    ts_raw = df_baseline[timestamp_col]
    if ts_raw.dtype == "object" or isinstance(
        ts_raw.iloc[0] if len(ts_raw) else None, str
    ):
        ts_naive = ts_raw.astype(str).str.replace(r"[+-]\d{2}:\d{2}$", "", regex=True)
        ts_naive = pd.to_datetime(ts_naive, errors="coerce")
    else:
        ts_naive = pd.to_datetime(ts_raw, errors="coerce")
        if ts_naive.dt.tz is not None:
            ts_naive = ts_naive.dt.tz_localize(None)

    naive_timestamps = pd.to_datetime(
        ts_naive.dt.strftime("%Y-%m-%d %H:%M:%S"),
        format="%Y-%m-%d %H:%M:%S",
    )
    df_baseline["month"] = naive_timestamps.dt.to_period("M")

    months_2025 = pd.period_range("2025-01", "2025-12", freq="M")
    monthly_peak_baseline_kw = (
        (df_baseline.groupby("month")["grid_consumption"].max() * 4.0)
        .reindex(months_2025)
        .fillna(0.0)
    )
    cummax_M_minus_1_kw = monthly_peak_baseline_kw.cummax().shift(1)
    cummax_M_minus_1_kw.loc[months_2025.min()] = BASELINE_2024_PEAK_GRID_KW
    cummax_M_minus_1_kw = cummax_M_minus_1_kw.fillna(BASELINE_2024_PEAK_GRID_KW)
    access_power_conservative = cummax_M_minus_1_kw + MARGIN_KW

    df_baseline["baseline_access_power_conservative"] = (
        df_baseline["month"].map(access_power_conservative.to_dict()).astype(float)
    )

    baseline_bills = calculate_monthly_bills(
        df_baseline,
        billing_config,
        access_power_col="baseline_access_power_conservative",
    )
    baseline_injection = calculate_monthly_injection_bills(df_baseline, billing_config)
    baseline_net_total = baseline_bills['total_cost_eur'].sum() - baseline_injection['injection_net_revenue_eur'].sum()
    
    # Calculate uncontrolled HP costs
    print("Calculating uncontrolled HP costs...")
    import tempfile
    import os
    
    # Save df to temporary CSV for uncontrolled HP calculation
    # Need to save timestamp as string to avoid issues
    df_temp = df.copy()
    if timestamp_col in df_temp.columns:
        df_temp[timestamp_col] = df_temp[timestamp_col].astype(str)
    temp_data_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df_temp.to_csv(temp_data_file.name, index=False)
    temp_data_file.close()
    temp_data_path = temp_data_file.name
    
    temp_hp_output = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    temp_hp_output.close()
    temp_hp_output_path = temp_hp_output.name
    
    try:
        # Calculate uncontrolled HP load
        calculate_heat_pump_load(
            data_path=temp_data_path,
            config_path=hp_config_path,
            output_path=temp_hp_output_path
        )
        
        # Load uncontrolled HP data
        hp_load_df = pd.read_csv(temp_hp_output_path)
        # Parse hp_load_df timestamps if they're strings (matching notebook pattern)
        if hp_load_df['timestamp'].dtype == 'object':
            hp_load_df['timestamp'] = pd.to_datetime(hp_load_df['timestamp'], utc=True)
            hp_load_df['timestamp'] = hp_load_df['timestamp'].dt.tz_localize(None)
        
        # Create dataframe with uncontrolled HP consumption
        df_with_hp = df.copy()
        # Ensure df timestamps are also timezone-naive for consistent merging (matching notebook pattern exactly)
        if df_with_hp['timestamp'].dtype != 'datetime64[ns]':
            df_with_hp['timestamp'] = pd.to_datetime(df_with_hp['timestamp'], utc=True)
        if df_with_hp['timestamp'].dt.tz is not None:
            df_with_hp['timestamp'] = df_with_hp['timestamp'].dt.tz_localize(None)
        
        df_with_hp = df_with_hp.merge(hp_load_df[['timestamp', 'hp_electrical_load']], on='timestamp', how='left')
        df_with_hp['hp_electrical_load'] = df_with_hp['hp_electrical_load'].fillna(0.0)
        
        # Calculate grid consumption with HP
        df_with_hp['total_consumption_with_hp'] = df_with_hp['inflex_load'] + df_with_hp['ev'] + df_with_hp['hp_electrical_load']
        df_with_hp['grid_consumption_with_hp'] = np.maximum(0, df_with_hp['total_consumption_with_hp'] - df_with_hp['pv_production'])
        df_with_hp['grid_injection_with_hp'] = np.maximum(0, df_with_hp['pv_production'] - df_with_hp['total_consumption_with_hp'])
        
        # Create billing dataframe
        df_billing_hp = df_with_hp.copy()
        df_billing_hp['grid_consumption'] = df_billing_hp['grid_consumption_with_hp']
        df_billing_hp['grid_injection'] = df_billing_hp['grid_injection_with_hp']
        
        # Baseline + uncontrolled HP: same heuristic as notebook 03
        # (conservative monthly AP + max thermal at COP(-10°C), no hp.yaml fixed AP table)
        hp_cfg = load_hp_config(hp_config_path)
        cop_at_minus10 = interpolate_cop(-10.0, hp_cfg["COP_data"])
        max_thermal_kwh = float(df_with_hp[thermal_load_col].max())
        thermal_max_kw = max_thermal_kwh * 4.0
        hp_additional_peak_kw = thermal_max_kw / cop_at_minus10

        _ts = df_billing_hp[timestamp_col]
        if _ts.dtype == "object" or isinstance(
            _ts.iloc[0] if len(_ts) else None, str
        ):
            _ts_naive = _ts.astype(str).str.replace(r"[+-]\d{2}:\d{2}$", "", regex=True)
            _ts_naive = pd.to_datetime(_ts_naive, errors="coerce")
        else:
            _ts_naive = pd.to_datetime(_ts, errors="coerce")
            if _ts_naive.dt.tz is not None:
                _ts_naive = _ts_naive.dt.tz_localize(None)
        _naive_bill = pd.to_datetime(
            _ts_naive.dt.strftime("%Y-%m-%d %H:%M:%S"),
            format="%Y-%m-%d %H:%M:%S",
        )
        df_billing_hp["month"] = _naive_bill.dt.to_period("M")

        access_power_hp_monthly = access_power_conservative + hp_additional_peak_kw
        df_billing_hp["access_power_hp"] = df_billing_hp["month"].map(
            access_power_hp_monthly.to_dict()
        ).astype(float)

        print(
            f"  Heuristic HP AP add-on (worst case @ -10°C): +{hp_additional_peak_kw:.2f} kW "
            f"(max thermal {max_thermal_kwh:.2f} kWh/15min → {thermal_max_kw:.1f} kW thermal / COP {cop_at_minus10:.2f})"
        )
        
        # Calculate bills for uncontrolled HP
        baseline_hp_bills = calculate_monthly_bills(df_billing_hp, billing_config, access_power_col='access_power_hp')
        baseline_hp_injection = calculate_monthly_injection_bills(df_billing_hp, billing_config)
        baseline_hp_net_total = baseline_hp_bills['total_cost_eur'].sum() - baseline_hp_injection['injection_net_revenue_eur'].sum()
        
        # Calculate uncontrolled HP volume and cost of heating
        hp_volume_uncontrolled_mwh = hp_load_df['hp_electrical_load'].sum() / 1000.0
        delta_uncontrolled = baseline_hp_net_total - baseline_net_total
        cost_of_heating_uncontrolled = delta_uncontrolled / hp_volume_uncontrolled_mwh if hp_volume_uncontrolled_mwh > 0 else np.nan
        
        # Store uncontrolled HP results
        results['uncontrolled'] = {
            'net_total': baseline_hp_net_total,
            'hp_volume_mwh': hp_volume_uncontrolled_mwh,
            'cost_of_heating_eur_per_mwh': cost_of_heating_uncontrolled,
            'delta_cost': delta_uncontrolled
        }
        
        print(f"  ✓ Uncontrolled HP cost of heating: {cost_of_heating_uncontrolled:.2f} EUR/MWh")
        
    finally:
        # Clean up temporary files
        if os.path.exists(temp_data_path):
            os.remove(temp_data_path)
        if os.path.exists(temp_hp_output_path):
            os.remove(temp_hp_output_path)
    
    print("="*80)
    print("BUFFER SIZE SENSITIVITY ANALYSIS")
    print("="*80)
    print(f"Testing buffer sizes: {buffer_sizes} m³")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    for buffer_size in buffer_sizes:
        print(f"\nOptimizing {buffer_size}m³...")
        
        # Create modified HP config with new buffer size
        hp_config = copy.deepcopy(base_hp_config)
        hp_config['buffer']['size_m3'] = buffer_size
        
        # Save modified config to temporary file
        temp_config_path = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(hp_config, temp_config_path, default_flow_style=False)
        temp_config_path.close()
        temp_config_file = temp_config_path.name
        
        try:
            # Run optimization
            results_df, summary = deterministic_mpc_hp(
                df=df,
                config_path=billing_config_path,
                hp_config_path=temp_config_file,
                timestamp_col=timestamp_col,
                pv_col=pv_col,
                inflex_load_col=inflex_load_col,
                price_col=price_col,
                ev_col=ev_col,
                thermal_load_col=thermal_load_col,
                outdoor_temp_col=outdoor_temp_col
            )
            
            print(f"Billing {buffer_size}m³...")
            
            # Calculate billing
            optimized_bills = calculate_monthly_bills(
                results_df.rename(columns={"spot_price": "price"}),
                billing_config,
                access_power_col="access_power"
            )
            
            optimized_injection = calculate_monthly_injection_bills(
                results_df.rename(columns={"spot_price": "price"}),
                billing_config
            )
            
            optimized_net_total = optimized_bills['total_cost_eur'].sum() - optimized_injection['injection_net_revenue_eur'].sum()
            
            # Calculate HP volume
            hp_volume_mwh = results_df['hp_electrical_input'].sum() / 1000.0
            
            # Calculate cost of heating (delta from baseline)
            delta_cost = optimized_net_total - baseline_net_total
            cost_of_heating_eur_per_mwh = delta_cost / hp_volume_mwh if hp_volume_mwh > 0 else np.nan
            
            # Save results
            output_filename = f"deterministic_hp_{buffer_size}m3.csv"
            output_filepath = output_path / output_filename
            results_df.to_csv(output_filepath, index=False)
            
            # Store results
            results[buffer_size] = {
                'results_df': results_df,
                'bills': optimized_bills,
                'injection': optimized_injection,
                'net_total': optimized_net_total,
                'hp_volume_mwh': hp_volume_mwh,
                'cost_of_heating_eur_per_mwh': cost_of_heating_eur_per_mwh,
                'delta_cost': delta_cost,
                'output_file': str(output_filepath)
            }
            
            print(f"  ✓ Saved to {output_filename}")
            print(f"  ✓ Cost of heating: {cost_of_heating_eur_per_mwh:.2f} EUR/MWh")
            
        finally:
            # Clean up temporary config file
            if os.path.exists(temp_config_file):
                os.remove(temp_config_file)
    
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("="*80)
    
    return results, baseline_net_total


def default_sensitivity_summary_path(project_root: Optional[Path] = None) -> Path:
    """CSV summary written by notebook 03 (comparison table + economics)."""
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    return root / "output" / "notebooks" / "buffer_size_sensitivity_summary_notebook_03.csv"


def optimised_csv_path(output_dir: Path, buffer_size: int) -> Path:
    return Path(output_dir) / f"deterministic_hp_{int(buffer_size)}m3.csv"


def sensitivity_optimisations_on_disk(
    buffer_sizes: list,
    output_dir: str | Path,
) -> bool:
    """True when every buffer-size MPC result CSV is present on disk."""
    opt = Path(output_dir)
    return all(optimised_csv_path(opt, bs).exists() for bs in buffer_sizes)


def sensitivity_cache_ready(
    buffer_sizes: list,
    output_dir: str | Path,
    summary_path: str | Path,
) -> bool:
    """True when optimised CSVs and the summary table are both cached."""
    return sensitivity_optimisations_on_disk(buffer_sizes, output_dir) and Path(summary_path).exists()


def load_sensitivity_summary(summary_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(summary_path)


def save_sensitivity_summary(
    comparison_df: pd.DataFrame,
    summary_path: str | Path,
) -> Path:
    path = Path(summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(path, index=False)
    return path


def discover_optimised_buffer_sizes(output_dir: str | Path) -> list[int]:
    """Buffer sizes (m³) found as ``deterministic_hp_{n}m3.csv`` under *output_dir*."""
    import re

    pattern = re.compile(r"deterministic_hp_(\d+)m3\.csv$")
    sizes: list[int] = []
    for path in Path(output_dir).glob("deterministic_hp_*m3.csv"):
        match = pattern.match(path.name)
        if match:
            sizes.append(int(match.group(1)))
    return sorted(set(sizes))


def calculate_payback_time(capex: float, annual_savings: float) -> float:
    if annual_savings <= 0:
        return np.inf
    return capex / annual_savings


def calculate_npv(
    annual_savings: float,
    capex: float,
    discount_rate: float,
    lifetime: int,
) -> float:
    if annual_savings <= 0:
        return -capex
    npv = -capex
    for year in range(1, lifetime + 1):
        npv += annual_savings / ((1 + discount_rate) ** year)
    return npv


def calculate_irr(
    annual_savings: float,
    capex: float,
    lifetime: int,
    max_iterations: int = 100,
) -> float:
    if annual_savings <= 0:
        return -np.inf
    irr = 0.1
    for _ in range(max_iterations):
        npv = -capex
        npv_derivative = 0.0
        for year in range(1, lifetime + 1):
            discount_factor = (1 + irr) ** year
            npv += annual_savings / discount_factor
            npv_derivative -= year * annual_savings / (discount_factor * (1 + irr))
        if abs(npv_derivative) < 1e-10:
            break
        irr_new = irr - npv / npv_derivative
        if abs(irr_new - irr) < 1e-8:
            irr = irr_new
            break
        irr = irr_new
    return irr


def build_comparison_df_from_results(
    results: dict,
    buffer_sizes: list,
    *,
    cost_of_capital: float,
    buffer_lifetime: int,
    buffer_capex_per_m3: float,
) -> pd.DataFrame:
    """Build notebook-03 comparison table (cost of heating, NPV, IRR, …)."""
    comparison_data: list[dict] = []
    uncontrolled_cost = None
    uncontrolled_hp_volume = None

    if "uncontrolled" in results:
        uncontrolled = results["uncontrolled"]
        uncontrolled_cost = uncontrolled["cost_of_heating_eur_per_mwh"]
        uncontrolled_hp_volume = uncontrolled["hp_volume_mwh"]
        comparison_data.append(
            {
                "Buffer Size (m³)": "Uncontrolled",
                "Cost of Heating (EUR/MWh)": uncontrolled_cost,
                "Total Cost (EUR)": uncontrolled["delta_cost"],
                "HP Volume (MWh)": uncontrolled_hp_volume,
                "CAPEX (EUR)": np.nan,
                "Annual Savings (EUR)": np.nan,
                "Payback Time (years)": np.nan,
                "NPV (EUR)": np.nan,
                "IRR (%)": np.nan,
            }
        )

    for buffer_size in sorted(buffer_sizes):
        if buffer_size not in results:
            continue
        result = results[buffer_size]
        optimized_cost = result["cost_of_heating_eur_per_mwh"]
        hp_volume = result["hp_volume_mwh"]

        if uncontrolled_cost is not None and uncontrolled_hp_volume is not None:
            annual_savings = (uncontrolled_cost - optimized_cost) * hp_volume
        else:
            annual_savings = np.nan

        capex = buffer_size * buffer_capex_per_m3
        if not np.isnan(annual_savings) and annual_savings > 0:
            payback_time = calculate_payback_time(capex, annual_savings)
            npv = calculate_npv(annual_savings, capex, cost_of_capital, buffer_lifetime)
            irr = calculate_irr(annual_savings, capex, buffer_lifetime)
        else:
            payback_time = np.inf
            npv = -capex
            irr = -np.inf

        comparison_data.append(
            {
                "Buffer Size (m³)": f"{buffer_size}",
                "Cost of Heating (EUR/MWh)": optimized_cost,
                "Total Cost (EUR)": result["delta_cost"],
                "HP Volume (MWh)": hp_volume,
                "CAPEX (EUR)": capex,
                "Annual Savings (EUR)": annual_savings,
                "Payback Time (years)": payback_time,
                "NPV (EUR)": npv,
                "IRR (%)": irr * 100,
            }
        )

    return pd.DataFrame(comparison_data)


def load_hp_economics(hp_config_path: str | Path) -> dict:
    with open(hp_config_path, "r", encoding="utf-8") as f:
        econ = yaml.safe_load(f)["economics"]
    return {
        "cost_of_capital": float(econ["cost_of_capital"]),
        "buffer_lifetime": int(econ["buffer_lifetime_years"]),
        "buffer_capex_per_m3": float(econ["buffer_capex_per_m3"]),
    }


def load_comparison_df_from_csvs(
    project_root: str | Path,
    *,
    plant_csv_path: str | Path | None = None,
    billing_config_path: str | Path | None = None,
    hp_config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    save_summary: bool = False,
) -> pd.DataFrame:
    """
    Build comparison table from ``deterministic_hp_*m3.csv`` only (no MPC, no summary cache).
    """
    root = Path(project_root)
    opt_dir = Path(output_dir) if output_dir else root / "output" / "optimised_ts"
    buffer_sizes = discover_optimised_buffer_sizes(opt_dir)
    if not buffer_sizes:
        raise FileNotFoundError(
            "No optimised HP CSVs found.\n"
            f"  Looked in: {opt_dir.resolve()}\n"
            f"  Expected files: deterministic_hp_<n>m3.csv\n"
            "Run notebook 03 buffer sensitivity (cell 21) once to generate them."
        )

    plant_csv = Path(plant_csv_path) if plant_csv_path else root / "data" / "plant1.csv"
    billing_path = (
        Path(billing_config_path) if billing_config_path else root / "config" / "billing.yaml"
    )
    hp_path = Path(hp_config_path) if hp_config_path else root / "config" / "hp.yaml"

    print(
        f"Loading comparison table from {len(buffer_sizes)} CSV(s) in {opt_dir.resolve()} "
        "(billing only, no MPC)..."
    )
    df = pd.read_csv(plant_csv, parse_dates=["timestamp"])
    df.columns = df.columns.str.strip()

    results, _ = load_results_from_optimised_csv(
        df=df,
        billing_config_path=str(billing_path),
        hp_config_path=str(hp_path),
        buffer_sizes=buffer_sizes,
        output_dir=str(opt_dir),
    )
    econ = load_hp_economics(hp_path)
    comparison_df = build_comparison_df_from_results(
        results,
        buffer_sizes,
        **econ,
    )
    if save_summary:
        summary_path = default_sensitivity_summary_path(root)
        save_sensitivity_summary(comparison_df, summary_path)
        print(f"Saved comparison table to: {summary_path}")
    return comparison_df


def load_comparison_df(
    project_root: str | Path,
    *,
    plant_csv_path: str | Path | None = None,
    billing_config_path: str | Path | None = None,
    hp_config_path: str | Path | None = None,
    save_summary: bool = True,
) -> pd.DataFrame:
    """
    Load buffer sensitivity comparison table for plotting (no MPC re-run).

    Uses, in order: saved summary CSV → rebuild from ``deterministic_hp_*m3.csv``.
    """
    root = Path(project_root)
    summary_path = default_sensitivity_summary_path(root)
    if summary_path.exists():
        print(f"Loaded comparison table from cache: {summary_path}")
        return load_sensitivity_summary(summary_path)

    return load_comparison_df_from_csvs(
        root,
        plant_csv_path=plant_csv_path,
        billing_config_path=billing_config_path,
        hp_config_path=hp_config_path,
        save_summary=save_summary,
    )


def _baseline_net_total_from_df(
    df: pd.DataFrame,
    billing_config,
    timestamp_col: str = "timestamp",
) -> float:
    """Baseline (no HP) net annual cost — same conservative access power as notebook 03."""
    MARGIN_KW = 20.0
    BASELINE_2024_PEAK_GRID_KW = 2663.5

    df_baseline = df.copy()
    ts_raw = df_baseline[timestamp_col]
    if ts_raw.dtype == "object" or isinstance(
        ts_raw.iloc[0] if len(ts_raw) else None, str
    ):
        ts_naive = ts_raw.astype(str).str.replace(r"[+-]\d{2}:\d{2}$", "", regex=True)
        ts_naive = pd.to_datetime(ts_naive, errors="coerce")
    else:
        ts_naive = pd.to_datetime(ts_raw, errors="coerce")
        if ts_naive.dt.tz is not None:
            ts_naive = ts_naive.dt.tz_localize(None)

    naive_timestamps = pd.to_datetime(
        ts_naive.dt.strftime("%Y-%m-%d %H:%M:%S"),
        format="%Y-%m-%d %H:%M:%S",
    )
    df_baseline["month"] = naive_timestamps.dt.to_period("M")

    months_2025 = pd.period_range("2025-01", "2025-12", freq="M")
    monthly_peak_baseline_kw = (
        (df_baseline.groupby("month")["grid_consumption"].max() * 4.0)
        .reindex(months_2025)
        .fillna(0.0)
    )
    cummax_M_minus_1_kw = monthly_peak_baseline_kw.cummax().shift(1)
    cummax_M_minus_1_kw.loc[months_2025.min()] = BASELINE_2024_PEAK_GRID_KW
    cummax_M_minus_1_kw = cummax_M_minus_1_kw.fillna(BASELINE_2024_PEAK_GRID_KW)
    access_power_conservative = cummax_M_minus_1_kw + MARGIN_KW

    df_baseline["baseline_access_power_conservative"] = (
        df_baseline["month"].map(access_power_conservative.to_dict()).astype(float)
    )

    baseline_bills = calculate_monthly_bills(
        df_baseline,
        billing_config,
        access_power_col="baseline_access_power_conservative",
    )
    baseline_injection = calculate_monthly_injection_bills(df_baseline, billing_config)
    return float(
        baseline_bills["total_cost_eur"].sum()
        - baseline_injection["injection_net_revenue_eur"].sum()
    )


def load_results_from_optimised_csv(
    df: pd.DataFrame,
    billing_config_path: str,
    hp_config_path: str,
    buffer_sizes: list,
    output_dir: str | Path,
    timestamp_col: str = "timestamp",
    thermal_load_col: str = "thermal_load",
) -> tuple[dict, float]:
    """
    Rebuild the sensitivity ``results`` dict from cached ``deterministic_hp_*m3.csv``
    files (no MPC re-run). Still computes baseline + uncontrolled HP once.
    """
    import os
    import tempfile

    billing_config = load_billing_config(billing_config_path)
    output_path = Path(output_dir)
    baseline_net_total = _baseline_net_total_from_df(df, billing_config, timestamp_col)
    results: dict = {}

    # --- uncontrolled HP (same logic as run_buffer_size_sensitivity) ---
    print("Rebuilding uncontrolled HP metrics from plant data...")
    df_temp = df.copy()
    if timestamp_col in df_temp.columns:
        df_temp[timestamp_col] = df_temp[timestamp_col].astype(str)
    temp_data_file = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    df_temp.to_csv(temp_data_file.name, index=False)
    temp_data_file.close()
    temp_data_path = temp_data_file.name

    temp_hp_output = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    temp_hp_output.close()
    temp_hp_output_path = temp_hp_output.name

    try:
        calculate_heat_pump_load(
            data_path=temp_data_path,
            config_path=hp_config_path,
            output_path=temp_hp_output_path,
        )
        hp_load_df = pd.read_csv(temp_hp_output_path)
        if hp_load_df["timestamp"].dtype == "object":
            hp_load_df["timestamp"] = pd.to_datetime(hp_load_df["timestamp"], utc=True)
            hp_load_df["timestamp"] = hp_load_df["timestamp"].dt.tz_localize(None)

        df_with_hp = df.copy()
        if df_with_hp["timestamp"].dtype != "datetime64[ns]":
            df_with_hp["timestamp"] = pd.to_datetime(df_with_hp["timestamp"], utc=True)
        if df_with_hp["timestamp"].dt.tz is not None:
            df_with_hp["timestamp"] = df_with_hp["timestamp"].dt.tz_localize(None)

        df_with_hp = df_with_hp.merge(
            hp_load_df[["timestamp", "hp_electrical_load"]], on="timestamp", how="left"
        )
        df_with_hp["hp_electrical_load"] = df_with_hp["hp_electrical_load"].fillna(0.0)
        df_with_hp["total_consumption_with_hp"] = (
            df_with_hp["inflex_load"] + df_with_hp["ev"] + df_with_hp["hp_electrical_load"]
        )
        df_with_hp["grid_consumption_with_hp"] = np.maximum(
            0, df_with_hp["total_consumption_with_hp"] - df_with_hp["pv_production"]
        )
        df_with_hp["grid_injection_with_hp"] = np.maximum(
            0, df_with_hp["pv_production"] - df_with_hp["total_consumption_with_hp"]
        )

        df_billing_hp = df_with_hp.copy()
        df_billing_hp["grid_consumption"] = df_billing_hp["grid_consumption_with_hp"]
        df_billing_hp["grid_injection"] = df_billing_hp["grid_injection_with_hp"]

        hp_cfg = load_hp_config(hp_config_path)
        cop_at_minus10 = interpolate_cop(-10.0, hp_cfg["COP_data"])
        max_thermal_kwh = float(df_with_hp[thermal_load_col].max())
        thermal_max_kw = max_thermal_kwh * 4.0
        hp_additional_peak_kw = thermal_max_kw / cop_at_minus10

        _ts = df_billing_hp[timestamp_col]
        if _ts.dtype == "object" or isinstance(_ts.iloc[0] if len(_ts) else None, str):
            _ts_naive = _ts.astype(str).str.replace(r"[+-]\d{2}:\d{2}$", "", regex=True)
            _ts_naive = pd.to_datetime(_ts_naive, errors="coerce")
        else:
            _ts_naive = pd.to_datetime(_ts, errors="coerce")
            if _ts_naive.dt.tz is not None:
                _ts_naive = _ts_naive.dt.tz_localize(None)
        _naive_bill = pd.to_datetime(
            _ts_naive.dt.strftime("%Y-%m-%d %H:%M:%S"),
            format="%Y-%m-%d %H:%M:%S",
        )
        df_billing_hp["month"] = _naive_bill.dt.to_period("M")

        MARGIN_KW = 20.0
        BASELINE_2024_PEAK_GRID_KW = 2663.5
        months_2025 = pd.period_range("2025-01", "2025-12", freq="M")
        df_peak = df.copy()
        df_peak["month"] = _naive_bill.dt.to_period("M")
        monthly_peak_baseline_kw = (
            (df_peak.groupby("month")["grid_consumption"].max() * 4.0)
            .reindex(months_2025)
            .fillna(0.0)
        )
        cummax_M_minus_1_kw = monthly_peak_baseline_kw.cummax().shift(1)
        cummax_M_minus_1_kw.loc[months_2025.min()] = BASELINE_2024_PEAK_GRID_KW
        cummax_M_minus_1_kw = cummax_M_minus_1_kw.fillna(BASELINE_2024_PEAK_GRID_KW)
        access_power_conservative = cummax_M_minus_1_kw + MARGIN_KW
        access_power_hp_monthly = access_power_conservative + hp_additional_peak_kw
        df_billing_hp["access_power_hp"] = df_billing_hp["month"].map(
            access_power_hp_monthly.to_dict()
        ).astype(float)

        baseline_hp_bills = calculate_monthly_bills(
            df_billing_hp, billing_config, access_power_col="access_power_hp"
        )
        baseline_hp_injection = calculate_monthly_injection_bills(df_billing_hp, billing_config)
        baseline_hp_net_total = (
            baseline_hp_bills["total_cost_eur"].sum()
            - baseline_hp_injection["injection_net_revenue_eur"].sum()
        )
        hp_volume_uncontrolled_mwh = hp_load_df["hp_electrical_load"].sum() / 1000.0
        delta_uncontrolled = baseline_hp_net_total - baseline_net_total
        cost_of_heating_uncontrolled = (
            delta_uncontrolled / hp_volume_uncontrolled_mwh
            if hp_volume_uncontrolled_mwh > 0
            else np.nan
        )
        results["uncontrolled"] = {
            "net_total": baseline_hp_net_total,
            "hp_volume_mwh": hp_volume_uncontrolled_mwh,
            "cost_of_heating_eur_per_mwh": cost_of_heating_uncontrolled,
            "delta_cost": delta_uncontrolled,
        }
    finally:
        if os.path.exists(temp_data_path):
            os.remove(temp_data_path)
        if os.path.exists(temp_hp_output_path):
            os.remove(temp_hp_output_path)

    # --- optimised runs from disk ---
    missing = [bs for bs in buffer_sizes if not optimised_csv_path(output_path, bs).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing cached optimisation CSV(s): "
            + ", ".join(f"deterministic_hp_{bs}m3.csv" for bs in missing)
        )

    print(f"Loading {len(buffer_sizes)} cached optimised HP schedules from {output_path}...")
    for buffer_size in buffer_sizes:
        csv_path = optimised_csv_path(output_path, buffer_size)
        results_df = pd.read_csv(csv_path)
        optimized_bills = calculate_monthly_bills(
            results_df.rename(columns={"spot_price": "price"}),
            billing_config,
            access_power_col="access_power",
        )
        optimized_injection = calculate_monthly_injection_bills(
            results_df.rename(columns={"spot_price": "price"}),
            billing_config,
        )
        optimized_net_total = (
            optimized_bills["total_cost_eur"].sum()
            - optimized_injection["injection_net_revenue_eur"].sum()
        )
        hp_volume_mwh = results_df["hp_electrical_input"].sum() / 1000.0
        delta_cost = optimized_net_total - baseline_net_total
        cost_of_heating_eur_per_mwh = (
            delta_cost / hp_volume_mwh if hp_volume_mwh > 0 else np.nan
        )
        results[buffer_size] = {
            "results_df": results_df,
            "bills": optimized_bills,
            "injection": optimized_injection,
            "net_total": optimized_net_total,
            "hp_volume_mwh": hp_volume_mwh,
            "cost_of_heating_eur_per_mwh": cost_of_heating_eur_per_mwh,
            "delta_cost": delta_cost,
            "output_file": str(csv_path),
        }

    return results, baseline_net_total
