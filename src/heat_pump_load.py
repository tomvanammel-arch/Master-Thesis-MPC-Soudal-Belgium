"""
Heat Pump Load Calculation

This module calculates the electrical power consumption of a heat pump based on:
- Thermal load demand (kWh per 15-minute interval)
- Outdoor temperature (°C)
- COP (Coefficient of Performance) as a function of outdoor temperature

**Formulas:**
- COP = f(T_outdoor) [interpolated from datasheet values]
- Thermal power (kW) = Thermal load (kWh) / 0.25 h
- Electrical power (kW) = Thermal power (kW) / COP
- Electrical load (kWh) = Electrical power (kW) × 0.25 h

**Units:**
- Thermal load: kWh per 15-minute interval
- Outdoor temperature: °C
- COP: dimensionless (thermal power / electrical power)
- Electrical power: kW
- Electrical load: kWh per 15-minute interval
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import yaml


def load_hp_config(config_path: str) -> Dict:
    """
    Load heat pump configuration from YAML file.
    
    Returns dictionary with:
    - COP_data: Dictionary mapping temperature (°C) to COP values
    - capacity: Dictionary with thermal_max_kw (maximum thermal capacity)
    - buffer: Dictionary with buffer parameters (size_m3, water_density_kg_per_m3, etc.)
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Convert COP_data keys from strings to floats
        cop_data = config.get('COP_data', {})
        cop_data_float = {float(k): float(v) for k, v in cop_data.items()}
        
        return {
            'COP_data': cop_data_float,
            'capacity': config.get('capacity', {}),
            'buffer': config.get('buffer', {})
        }
    except Exception as e:
        raise ValueError(f"Error loading HP config from {config_path}: {e}")


def interpolate_cop(outdoor_temp: float, cop_data: Dict[float, float]) -> float:
    """
    Interpolate COP value for given outdoor temperature.
    
    Args:
        outdoor_temp: Outdoor temperature (°C)
        cop_data: Dictionary mapping temperature (°C) to COP values
    
    Returns:
        Interpolated COP value
    
    Formula:
        Linear interpolation between nearest temperature points
        For temperatures outside range: extrapolate using nearest two points
    """
    if not cop_data:
        raise ValueError("COP data is empty")
    
    temps = sorted(cop_data.keys())
    cop_values = [cop_data[t] for t in temps]
    
    # If temperature is below minimum, extrapolate using first two points
    if outdoor_temp < temps[0]:
        if len(temps) < 2:
            return cop_values[0]
        # Linear extrapolation: COP = COP_min + slope * (T - T_min)
        slope = (cop_values[1] - cop_values[0]) / (temps[1] - temps[0])
        return cop_values[0] + slope * (outdoor_temp - temps[0])
    
    # If temperature is above maximum, extrapolate using last two points
    if outdoor_temp > temps[-1]:
        if len(temps) < 2:
            return cop_values[-1]
        # Linear extrapolation: COP = COP_max + slope * (T - T_max)
        slope = (cop_values[-1] - cop_values[-2]) / (temps[-1] - temps[-2])
        return cop_values[-1] + slope * (outdoor_temp - temps[-1])
    
    # Interpolate between two nearest points
    for i in range(len(temps) - 1):
        if temps[i] <= outdoor_temp <= temps[i + 1]:
            # Linear interpolation
            t1, t2 = temps[i], temps[i + 1]
            cop1, cop2 = cop_values[i], cop_values[i + 1]
            if t2 == t1:
                return cop1
            cop = cop1 + (cop2 - cop1) * (outdoor_temp - t1) / (t2 - t1)
            return cop
    
    # Fallback (should not reach here)
    return cop_values[-1]


def calculate_heat_pump_load(
    data_path: str,
    config_path: str,
    output_path: str
) -> pd.DataFrame:
    """
    Calculate heat pump electrical load from thermal load and outdoor temperature.
    
    Args:
        data_path: Path to plant1.csv
        config_path: Path to hp.yaml
        output_path: Path to output CSV file (e.g., output/uncontrolled_hp.csv)
    
    Returns:
        DataFrame with columns: timestamp, hp_electrical_load (kWh)
    
    Process:
    1. Load HP configuration (COP data, capacity)
    2. Load thermal load and outdoor temperature from plant1.csv
    3. Handle DST issues (strip timezone, parse as naive datetime)
    4. Interpolate COP for each timestep based on outdoor temperature
    5. Calculate electrical power = thermal_power / COP
    6. Convert to electrical load (kWh per 15-min interval)
    7. Save to output CSV
    """
    # Load HP configuration
    hp_config = load_hp_config(config_path)
    cop_data = hp_config['COP_data']
    thermal_max_kw = hp_config['capacity'].get('thermal_max_kw', 1000.0)
    
    # Load plant data
    data = pd.read_csv(data_path)
    
    # Normalize column names
    data.columns = data.columns.str.strip()
    
    # Handle DST issues: strip timezone offset and parse as naive datetime
    # This avoids timezone conversion issues that cause NaT values
    timestamp_col = 'timestamp'
    if timestamp_col in data.columns:
        if data[timestamp_col].dtype == 'object' or isinstance(data[timestamp_col].iloc[0] if len(data) > 0 else None, str):
            # If string, strip timezone offset before parsing
            data[timestamp_col] = data[timestamp_col].astype(str).str.replace(r'[+-]\d{2}:\d{2}$', '', regex=True)
            data[timestamp_col] = pd.to_datetime(data[timestamp_col], errors='coerce')
        else:
            # Already datetime, but may have timezone - convert to naive
            data[timestamp_col] = pd.to_datetime(data[timestamp_col], errors='coerce')
            if data[timestamp_col].dt.tz is not None:
                data[timestamp_col] = data[timestamp_col].dt.tz_localize(None)
    
    # Extract thermal load and outdoor temperature
    thermal_load = pd.to_numeric(data['thermal_load'], errors='coerce')
    outdoor_temp = pd.to_numeric(data['outdoor_temperature'], errors='coerce')
    
    # Calculate COP for each timestep
    cop_values = []
    for temp in outdoor_temp:
        if pd.isna(temp):
            cop_values.append(np.nan)
        else:
            cop = interpolate_cop(float(temp), cop_data)
            cop_values.append(cop)
    
    cop_series = pd.Series(cop_values, index=data.index)
    
    # Calculate thermal power (kW) from thermal load (kWh per 15-min)
    # Thermal power = Thermal load / 0.25 h
    thermal_power_kw = thermal_load / 0.25
    
    # Apply capacity limit (thermal_max_kw)
    thermal_power_kw = thermal_power_kw.clip(upper=thermal_max_kw)
    
    # Recalculate thermal load after capacity limit
    thermal_load_limited = thermal_power_kw * 0.25
    
    # Calculate electrical power (kW) = Thermal power (kW) / COP
    # Handle division by zero and NaN values
    electrical_power_kw = thermal_power_kw / cop_series
    electrical_power_kw = electrical_power_kw.replace([np.inf, -np.inf], np.nan)
    
    # Convert electrical power to electrical load (kWh per 15-min interval)
    # Electrical load = Electrical power × 0.25 h
    hp_electrical_load = electrical_power_kw * 0.25
    
    # Create output DataFrame
    # Use original timestamps from input (with timezone info preserved)
    original_data = pd.read_csv(data_path)
    original_data.columns = original_data.columns.str.strip()
    original_timestamps = original_data['timestamp']
    
    result_df = pd.DataFrame({
        'timestamp': original_timestamps,
        'hp_electrical_load': hp_electrical_load.values
    })
    
    # Save to output CSV
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    
    print(f"Heat pump load calculated and saved to {output_path}")
    print(f"Total electrical load: {hp_electrical_load.sum():.2f} kWh ({hp_electrical_load.sum()/1000:.2f} MWh)")
    print(f"Average COP: {cop_series.mean():.2f}")
    print(f"Min COP: {cop_series.min():.2f}, Max COP: {cop_series.max():.2f}")
    
    return result_df
