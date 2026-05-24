"""
Electricity billing consistent with thesis §3.3 (shadow reconstruction).

Symbols (thesis): spot offtake cost C_spot, volumetric adder c_energy, grid-loss
share r_loss on the spot price, access C_access, monthly peak C_peak, rolling
over-usage C_over, injection revenue R_inj. YAML under `config/billing.yaml`
stores **numerical instances** of those tariff coefficients (€/MWh, €/kW/month).

The legacy YAML section key `acces_power` maps to contracted access power P_access
(monthly defaults); the spelling is kept for backward compatibility with existing
YAML files.
"""

import datetime as dt
from typing import Dict, List, Optional

import pandas as pd


def _parse_float(value: str) -> float:
    """Parse float from string, handling comments and comma decimals."""
    cleaned = value.strip().split("#", 1)[0].strip()
    if cleaned == "":
        return 0.0
    return float(cleaned.replace(",", "."))


def _parse_billing_yaml(path: str) -> Dict[str, Dict[str, float]]:
    """Fallback YAML parser (handles duplicate keys: last value wins)."""
    sections = {
        "energy_based_costs": {},
        "injection_costs": {},
        "peak_based_costs": {},
        "acces_power": {},
    }
    current = None
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line.strip() or line.strip().startswith("#"):
                continue
            if not line.startswith(" "):
                key = line.split(":", 1)[0].strip().lower()
                if key in sections:
                    current = key
                else:
                    current = None
                continue
            if current is None:
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            sections[current][key.strip()] = _parse_float(value)
    return sections


def load_billing_config(path: str) -> Dict[str, Dict[str, float]]:
    """
    Load billing configuration from YAML file.
    
    Returns normalized dictionary with:
    - energy_based_costs: Fixed rates (€/MWh) and grid_losses_percentage
    - injection_costs: Imbalance cost (21.148 €/MWh)
    - peak_based_costs: Access power, monthly peak, over-usage prices (€/kW/month)
    - acces_power: Default access power per month (kW)
    """
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return {
            "energy_based_costs": data.get("Energy_based_costs", {}) or {},
            "injection_costs": data.get("Injection_costs", {}) or {},
            "peak_based_costs": data.get("Peak_based_costs", {}) or {},
            "acces_power": data.get("Acces_power", {}) or {},
        }
    except Exception:
        return _parse_billing_yaml(path)


def calculate_monthly_injection_bills(
    df: pd.DataFrame,
    config: Dict[str, Dict[str, float]],
    injection_col: str = "grid_injection",
    price_col: str = "price",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Calculate monthly injection revenue (feed-in to grid).
    
    **Formula:**
    - Net injection price (€/MWh) = Spot price - Imbalance cost
    - Imbalance cost: 21.148 €/MWh (injection) vs 4.612 €/MWh (consumption)
    - Net revenue (€) = Σ(Injection_kWh[t] × (Spot_price[t] - 21.148) / 1000)
    
    **Units:**
    - Injection: kWh per 15-minute interval
    - Spot price: €/MWh
    - Net revenue: € per month
    
    **Returns:** Monthly injection bills with volumes, prices, and net revenue.
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Missing column: {timestamp_col}")
    if injection_col not in df.columns:
        raise ValueError(f"Missing column: {injection_col}")
    if price_col not in df.columns:
        raise ValueError(f"Missing column: {price_col}")

    # Work on a copy of the relevant columns.
    data = df[[timestamp_col, injection_col, price_col]].copy()
    
    # Parse timestamps: CSV has Belgium local time with fixed offsets (+01:00 or +02:00)
    # Strip timezone offset and parse as naive datetime (treating as Belgium local time)
    # This avoids timezone conversion issues that cause NaT values
    if data[timestamp_col].dtype == 'object' or isinstance(data[timestamp_col].iloc[0] if len(data) > 0 else None, str):
        # If string, strip timezone offset before parsing
        data[timestamp_col] = data[timestamp_col].astype(str).str.replace(r'[+-]\d{2}:\d{2}$', '', regex=True)
        data[timestamp_col] = pd.to_datetime(data[timestamp_col], errors="coerce")
    else:
        # Already datetime, but may have timezone - convert to naive
        data[timestamp_col] = pd.to_datetime(data[timestamp_col], errors="coerce")
        if data[timestamp_col].dt.tz is not None:
            # Convert to naive datetime preserving local time values
            data[timestamp_col] = data[timestamp_col].dt.tz_localize(None)

    # Restrict to the 2025 calendar year in Belgium local time
    # Filter using naive datetime comparison (treating as Belgium local time)
    start_2025 = pd.Timestamp("2025-01-01 00:00:00")
    end_2025 = pd.Timestamp("2026-01-01 00:00:00")
    data = data[(data[timestamp_col] >= start_2025) & (data[timestamp_col] < end_2025)]

    # Month index for grouping (using Belgium local time)
    # Convert to naive datetime preserving local time values, then extract year/month
    # This ensures month boundaries are based on Belgium local time, not UTC
    # Create a naive datetime series with local time values
    naive_timestamps = pd.to_datetime(
        data[timestamp_col].dt.strftime("%Y-%m-%d %H:%M:%S"),
        format="%Y-%m-%d %H:%M:%S"
    )
    data["month"] = naive_timestamps.dt.to_period("M")

    inj_costs = config.get("injection_costs", {}) or {}
    imbalance_cost = float(inj_costs.get("imbalance_cost_eur_per_mwh", 0.0))

    # Convert kWh to MWh: Injection_kWh / 1000 → Injection_MWh
    data["inj_mwh"] = data[injection_col].fillna(0.0) / 1000.0
    data["spot_eur_per_mwh"] = data[price_col].fillna(0.0)
    
    # Net injection price: Spot_price - Imbalance_cost (21.148 €/MWh)
    data["net_injection_price_eur_per_mwh"] = data["spot_eur_per_mwh"] - imbalance_cost
    
    # Interval revenue: Injection_MWh × Net_injection_price
    data["interval_net_revenue_eur"] = data["inj_mwh"] * data["net_injection_price_eur_per_mwh"]

    # Monthly aggregation
    grouped = data.groupby("month", dropna=True)
    rows = []
    for month, g in grouped:
        injected_volume_mwh = float(g["inj_mwh"].sum())
        # Your requested logic (per 15-min interval):
        #   net_price_t = spot_t - imbalance_cost
        #   net_revenue_t = inj_mwh_t * net_price_t
        # Monthly net revenue = sum_t(net_revenue_t)
        injection_net_revenue_eur = float(g["interval_net_revenue_eur"].sum())

        # (Optional breakdown columns for transparency)
        injection_spot_revenue_eur = float((g["inj_mwh"] * g["spot_eur_per_mwh"]).sum())
        injection_imbalance_cost_eur = float((g["inj_mwh"] * imbalance_cost).sum())

        # Weighted averages (only over intervals with injection > 0)
        if injected_volume_mwh > 0:
            avg_spot_price = float((g["inj_mwh"] * g["spot_eur_per_mwh"]).sum() / injected_volume_mwh)
            avg_net_injection_price = float((g["inj_mwh"] * g["net_injection_price_eur_per_mwh"]).sum() / injected_volume_mwh)
        else:
            avg_spot_price = 0.0
            avg_net_injection_price = 0.0

        rows.append(
            {
                "month": str(month),
                "injected_volume_mwh": injected_volume_mwh,
                "avg_spot_price_eur_per_mwh": avg_spot_price,
                "avg_net_injection_price_eur_per_mwh": avg_net_injection_price,
                "imbalance_cost_eur_per_mwh": imbalance_cost,
                "injection_spot_revenue_eur": injection_spot_revenue_eur,
                "injection_imbalance_cost_eur": injection_imbalance_cost_eur,
                "injection_net_revenue_eur": injection_net_revenue_eur,
            }
        )

    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def calculate_monthly_bills(
    df: pd.DataFrame,
    config: Dict[str, Dict[str, float]],
    volume_col: str = "grid_consumption",
    price_col: str = "price",
    timestamp_col: str = "timestamp",
    access_power_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate monthly electricity bills for grid consumption (offtake).
    
    **Energy-Based Costs:**
    - Fixed rate: ~30.4 €/MWh (imbalance: 4.612, certificates, taxes)
    - Grid losses: Monthly_avg_spot_price × 1.75%
    - Total energy rate = Fixed rate + Grid losses
    - Energy cost (€) = Volume_MWh × Total_energy_rate
    
    **Spot Costs:**
    - Spot cost (€) = Σ(Volume_kWh[t] × Spot_price[t] / 1000) for all intervals t
    
    **Peak-Based Costs (Capacity Tariff):**
    - Monthly peak (kW) = max(Grid_consumption_kWh[t] × 4) for all t in month
    - Exceedance (kW) = max(0, Monthly_peak - Access_power)
    - Rolling max exceedance (kW) = max(Exceedance) over 12-month window
    - Access power cost (€) = Access_power_kW × 2.9975 €/kW/month
    - Monthly peak cost (€) = Monthly_peak_kW × 4.227 €/kW/month
    - Over-usage cost (€) = Rolling_max_exceedance_kW × 4.496 €/kW/month
    
    **Units:**
    - Energy: kWh per 15-minute interval
    - Power: kW (kWh × 4 for 15-min intervals)
    - Prices: €/MWh or €/kW/month
    - Monthly peak based on grid consumption (offtake) only, not net power
    
    **Returns:** Monthly bills with all cost components and totals.

    **Access power:** ``access_power_col`` must name a column on ``df`` with the
    contracted access power (kW) per timestep (constant within each month).
    Values from ``config['acces_power']`` are not used.
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Missing column: {timestamp_col}")
    if volume_col not in df.columns:
        raise ValueError(f"Missing column: {volume_col}")
    if price_col not in df.columns:
        raise ValueError(f"Missing column: {price_col}")
    if access_power_col is None or (
        isinstance(access_power_col, str) and not access_power_col.strip()
    ):
        raise ValueError(
            "calculate_monthly_bills requires access_power_col to name a column on df "
            "with access power (kW) per row; billing.yaml no longer supplies a default AP series."
        )
    if access_power_col not in df.columns:
        raise ValueError(
            f"access_power_col={access_power_col!r} not found in df columns: {list(df.columns)}"
        )

    # Work directly with the timestamps from the meter data.
    cols_to_copy = [timestamp_col, volume_col, price_col, access_power_col]
    data = df[cols_to_copy].copy()
    
    # Parse timestamps: CSV has Belgium local time with fixed offsets (+01:00 or +02:00)
    # Strip timezone offset and parse as naive datetime (treating as Belgium local time)
    # This avoids timezone conversion issues that cause NaT values
    if data[timestamp_col].dtype == 'object' or isinstance(data[timestamp_col].iloc[0] if len(data) > 0 else None, str):
        # If string, strip timezone offset before parsing
        data[timestamp_col] = data[timestamp_col].astype(str).str.replace(r'[+-]\d{2}:\d{2}$', '', regex=True)
        data[timestamp_col] = pd.to_datetime(data[timestamp_col], errors="coerce")
    else:
        # Already datetime, but may have timezone - convert to naive
        data[timestamp_col] = pd.to_datetime(data[timestamp_col], errors="coerce")
        if data[timestamp_col].dt.tz is not None:
            # Convert to naive datetime preserving local time values
            data[timestamp_col] = data[timestamp_col].dt.tz_localize(None)

    # Restrict to the 2025 calendar year in Belgium local time
    # Filter using naive datetime comparison (treating as Belgium local time)
    start_2025 = pd.Timestamp("2025-01-01 00:00:00")
    end_2025 = pd.Timestamp("2026-01-01 00:00:00")
    data = data[(data[timestamp_col] >= start_2025) & (data[timestamp_col] < end_2025)]

    # Month column for grouping (using Belgium local time).
    # Convert to naive datetime preserving local time values, then extract year/month
    # This ensures month boundaries are based on Belgium local time, not UTC
    # Create a naive datetime series with local time values
    naive_timestamps = pd.to_datetime(
        data[timestamp_col].dt.strftime("%Y-%m-%d %H:%M:%S"),
        format="%Y-%m-%d %H:%M:%S"
    )
    data["month"] = naive_timestamps.dt.to_period("M")

    energy_costs = config.get("energy_based_costs", {})
    peak_costs = config.get("peak_based_costs", {})
    access_power_cfg = config.get("acces_power", {})

    # Grid losses: Monthly_avg_spot_price × grid_losses_percentage (1.75%)
    grid_losses_percentage = float(energy_costs.get("grid_losses_percentage", 0.0))
    
    # Fixed energy rate: Sum of all fixed costs (~30.4 €/MWh)
    # Includes: imbalance (4.612), certificates, taxes, etc.
    energy_rate_eur_per_mwh = sum(
        float(v) for k, v in energy_costs.items() 
        if k != "grid_losses_percentage"
    )
    
    access_power_price = float(peak_costs.get("access_power_price_eur_per_kw", 0.0))
    monthly_peak_price = float(peak_costs.get("monthly_peak_price_eur_per_kw", 0.0))
    over_usage_price = float(peak_costs.get("over_usage_price_eur_per_kw", 0.0))

    # Get unique months (filter out NaT)
    months: List[pd.Period] = sorted([m for m in data["month"].unique() if pd.notna(m)])
    rows = []

    for month in months:
        month_data = data[data["month"] == month]
        # Only use rows with valid data (not NaN)
        month_data = month_data[month_data[volume_col].notna()]
        
        if len(month_data) == 0:
            continue
            
        volume_kwh = month_data[volume_col].sum()
        volume_mwh = volume_kwh / 1000.0

        # Spot cost = Σ(Volume_kWh[t] × Spot_price[t] / 1000)
        spot_cost_eur = (month_data[volume_col] * month_data[price_col] / 1000.0).sum()
        
        # Grid losses: Volume-weighted average spot price × grid_losses_percentage
        # Volume-weighted avg = (Σ(volume[t] × spot_price[t])) / Σ(volume[t])
        # Which equals: (spot_cost_eur × 1000) / volume_kwh to get EUR/MWh
        if volume_kwh > 0:
            volume_weighted_avg_spot_price = (spot_cost_eur * 1000.0) / volume_kwh
        else:
            volume_weighted_avg_spot_price = 0.0
        grid_losses_eur_per_mwh = volume_weighted_avg_spot_price * grid_losses_percentage
        
        # Total energy rate = Fixed rate + Grid losses
        total_energy_rate_eur_per_mwh = energy_rate_eur_per_mwh + grid_losses_eur_per_mwh

        # Energy cost = Volume_MWh × Total_energy_rate
        energy_cost_eur = volume_mwh * total_energy_rate_eur_per_mwh

        # Monthly peak (kW) = max(Grid_consumption_kWh[t] × 4) for all t in month
        # Based on grid consumption (offtake) only, not net power
        monthly_peak_kw = (month_data[volume_col] * 4).max()
        
        # Access power: Contracted capacity (kW), can be optimized monthly
        access_power_kw = float(month_data[access_power_col].iloc[0])
        
        # Exceedance (kW) = max(0, Monthly_peak - Access_power)
        exceedance_kw = max(0.0, monthly_peak_kw - access_power_kw)

        access_cost_eur = access_power_kw * access_power_price
        monthly_peak_cost_eur = monthly_peak_kw * monthly_peak_price

        rows.append(
            {
                "month": str(month),
                "volume_kwh": volume_kwh,
                "volume_mwh": volume_mwh,
                "monthly_avg_spot_price_eur_per_mwh": volume_weighted_avg_spot_price,  # Now volume-weighted
                "grid_losses_eur_per_mwh": grid_losses_eur_per_mwh,
                "energy_cost_eur": energy_cost_eur,
                "spot_cost_eur": spot_cost_eur,
                "access_power_kw": access_power_kw,
                "access_cost_eur": access_cost_eur,
                "monthly_peak_kw": monthly_peak_kw,
                "monthly_peak_cost_eur": monthly_peak_cost_eur,
                "exceedance_kw": exceedance_kw,
            }
        )

    bills = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)

    # Rolling max exceedance (kW) = max(Exceedance) over 12-month window
    rolling_max_exceedance = []
    for i in range(len(bills)):
        window = bills.loc[max(0, i - 11) : i, "exceedance_kw"]
        rolling_max_exceedance.append(window.max() if not window.empty else 0.0)
    bills["rolling_max_exceedance_kw"] = rolling_max_exceedance
    
    # Over-usage cost (€) = Rolling_max_exceedance_kW × 4.496 €/kW/month
    bills["over_usage_cost_eur"] = bills["rolling_max_exceedance_kw"] * over_usage_price
    
    # Peak-based cost (€) = Access_cost + Monthly_peak_cost + Over_usage_cost
    bills["peak_based_cost_eur"] = (
        bills["access_cost_eur"] + bills["monthly_peak_cost_eur"] + bills["over_usage_cost_eur"]
    )
    
    # Total cost (€) = Energy_cost + Spot_cost + Peak_based_cost
    bills["total_cost_eur"] = (
        bills["energy_cost_eur"] + bills["spot_cost_eur"] + bills["peak_based_cost_eur"]
    )
    
    # Effective energy rate (€/MWh) = Total_cost / Volume_MWh
    bills["energy_rate_eur_per_mwh"] = bills["total_cost_eur"] / bills["volume_mwh"].replace(0, pd.NA)

    return bills
