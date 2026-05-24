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
from heat_pump_load import interpolate_cop, load_hp_config
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
    from heat_pump_load import calculate_heat_pump_load
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
