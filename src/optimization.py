"""
Deterministic MPC Optimization for EV Charging with Capacity Tariff

**Objective Function:**
Minimize: Energy_cost + Spot_cost + Peak_costs - Injection_revenue

**Cost Components:**

1. **Energy Cost (€):**
   - Energy_cost = Σ(Grid_consumption_kWh[t] × (Fixed_rate/1000 + Grid_losses% × Spot_price[t]))
   - Fixed rate: ~30.4 €/MWh
   - Grid losses: 1.75% of spot price per interval

2. **Spot Cost (€):**
   - Spot_cost = Σ(Grid_consumption_kWh[t] × Spot_price[t])
   - Spot price in €/kWh (converted from €/MWh)

3. **Peak-Based Costs (€):**
   - Access_cost = Σ(Access_power_kW[m] × 2.9975 €/kW/month)
   - Monthly_peak_cost = Σ(Monthly_peak_kW[m] × 4.227 €/kW/month)
   - Over_usage_cost = Σ(Rolling_max_exceedance_kW[m] × 4.496 €/kW/month)

4. **Injection Revenue (€):**
   - Injection_revenue = -Σ(Grid_injection_kWh[t] × Net_injection_price[t])
   - Net_injection_price = Spot_price - Imbalance_cost (21.148 €/MWh)

**Constraints:**

- **EV Charging:**
  - Power: P_ev(t) = E_ev(t) × 4 (kWh → kW conversion for 15-min intervals)
  - Envelope: P_ev(t) ≤ P_ev,max(t) (dynamic power envelope)
  - Energy: Σ(E_ev(t) for t in day) = Daily_EV_demand (must meet daily demand)

- **Power Balance:**
  - Grid_consumption - Grid_injection = Inflex_load + EV_charge - PV_production

- **Peak Tracking:**
  - Monthly_peak[m] ≥ Grid_consumption_kW[t] for all t in month m
  - Exceedance[t] ≥ Monthly_peak[m] - Access_power[m]
  - Rolling_max_exceedance[m] ≥ Exceedance[k] for k in [m-11, m]

- **Access Power Rules:**
  - Can increase anytime: Access_power[m] ≥ Access_power[m-1] (if increased)
  - Lock-in period: After increase, cannot reduce for 12 months

**EV Power Envelope:**
- P_ev,max,cum(d,t) = max(τ≤t) P_ev,bench(τ) for t ≤ 15:30
- P_ev,max(d,t) = P_ev,max,cum(d,15:30) × (17 - t) / 1.5 for 15:30 < t < 17:00
- P_ev,max(d,t) = 0 for t ≥ 17:00

**Units:**
- Energy: kWh per 15-minute interval
- Power: kW (kWh × 4)
- Prices: €/MWh or €/kW/month
"""

import datetime as dt
from typing import Dict, List, Optional, Tuple, Sequence

import numpy as np
import pandas as pd
from pyomo.environ import (
    ConcreteModel,
    Var,
    Objective,
    Constraint,
    Set,
    Param,
    minimize,
    maximize,
    NonNegativeReals,
    Reals,
    Binary,
    SolverFactory,
    value,
)

from billing import load_billing_config
import yaml
from heat_pump_load import load_hp_config, interpolate_cop


def solve_highs_model(model, tee: bool = False):
    """Solve a Pyomo model with HiGHS.

    Retries on a known Pyomo issue where ``timing_info.wall_time`` can become
    negative if the OS clock steps backward between timestamps (seen on Windows).
    """
    import time

    solver = SolverFactory("highs")
    if not solver.available():
        raise RuntimeError("HiGHS not available. Install with: pip install highspy")

    last_exc: Optional[BaseException] = None
    for attempt in range(3):
        try:
            return solver.solve(model, tee=tee)
        except ValueError as exc:
            last_exc = exc
            if "timing_info.wall_time" in str(exc) and attempt < 2:
                time.sleep(0.05 * (attempt + 1))
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("solve_highs_model: solve failed without exception")


def deterministic_mpc_ev(
    df: pd.DataFrame,
    config_path: str,
    timestamp_col: str = "timestamp",
    pv_col: str = "pv_production",
    inflex_load_col: str = "inflex_load",
    price_col: str = "price",
    ev_col: str = "ev",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Solve deterministic MPC optimization for EV charging site over full year.
    
    Minimizes total cost (energy + spot + peak-based) and maximizes injection revenues.
    Implements full capacity tariff structure including:
    - Access power cost
    - Monthly peak cost
    - Rolling max exceedance cost (over-usage)
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data with columns: timestamp, pv_production, inflex_load, price, ev
        All energy values in kWh (15-min intervals)
        The 'ev' column contains actual EV charging demand (kWh per 15-min)
    config_path : str
        Path to billing configuration YAML file
    timestamp_col : str
        Name of timestamp column
    pv_col : str
        Name of PV production column
    inflex_load_col : str
        Name of inflexible load column
    price_col : str
        Name of spot price column (EUR/MWh)
    ev_col : str
        Name of EV demand column (actual charging data)
    
    Returns
    -------
    results_df : pd.DataFrame
        Results with optimized variables for each time step
    summary : dict
        Summary statistics and costs
    """
    print("=" * 80)
    print("Deterministic MPC Optimization for EV Charging Site")
    print("=" * 80)
    
    # Load billing configuration
    print(f"\n[1/9] Loading billing configuration from: {config_path}")
    config = load_billing_config(config_path)
    print("   ✓ Billing configuration loaded")
    
    # Prepare data
    print(f"\n[2/9] Preparing input data...")
    print(f"   Input DataFrame shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    # Store original timestamps with timezone info before any processing
    # This ensures we can save them correctly later
    original_timestamps = df[timestamp_col].copy()
    
    data = df.copy()
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
    
    # Filter to 2025 calendar year in Belgium local time
    # Filter using naive datetime comparison (treating as Belgium local time)
    start_2025 = pd.Timestamp("2025-01-01 00:00:00")
    end_2025 = pd.Timestamp("2026-01-01 00:00:00")
    data = data[(data[timestamp_col] >= start_2025) & (data[timestamp_col] < end_2025)].copy()
    data = data.sort_values(timestamp_col)
    
    # Store original indices before reset_index (needed to map back to original timestamps)
    original_indices = data.index.values
    
    data = data.reset_index(drop=True)
    
    n_periods = len(data)
    print(f"   ✓ Filtered to {n_periods} periods (15-min intervals)")
    print(f"   Date range: {data[timestamp_col].min()} to {data[timestamp_col].max()}")
    
    # Extract parameters
    print(f"\n[3/9] Extracting input parameters...")
    pv_production = data[pv_col].fillna(0.0).values  # kWh per 15-min
    inflex_load = data[inflex_load_col].fillna(0.0).values  # kWh per 15-min
    spot_price = data[price_col].fillna(0.0).values  # EUR/MWh
    ev_demand_actual = data[ev_col].fillna(0.0).values  # kWh per 15-min (actual charging from data)
    
    print(f"   PV production: {pv_production.sum():.2f} kWh total")
    print(f"   Inflexible load: {inflex_load.sum():.2f} kWh total")
    print(f"   EV demand: {ev_demand_actual.sum():.2f} kWh total")
    print(f"   Spot price range: {spot_price.min():.2f} - {spot_price.max():.2f} EUR/MWh")
    
    # Daily EV energy demand (must be satisfied per day)
    data["date"] = data[timestamp_col].dt.date
    
    # EV Power Envelope Calculation:
    # P_ev,bench(t) = EV_demand_kWh(t) × 4 (convert 15-min kWh to kW)
    # P_ev,max,cum(d,t) = max(τ≤t) P_ev,bench(τ) for t ≤ 15:30
    # P_ev,max(d,t) = P_ev,max,cum(d,15:30) × (17 - t) / 1.5 for 15:30 < t < 17:00
    # P_ev,max(d,t) = 0 for t ≥ 17:00
    print(f"\n   Calculating dynamic EV power envelope...")
    # Online MPC passes a precomputed `ev_power_envelope_fixed_kw` on each horizon window.
    # Deterministic notebook callers typically omit it: build the same envelope from `ev`
    # (same day-cumulative + 15:30–17:00 taper rule as deterministic_mpc_ev_hp).
    if "ev_power_envelope_fixed_kw" in data.columns:
        ev_power_envelope = (
            pd.to_numeric(data["ev_power_envelope_fixed_kw"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        print(
            "   Using EV power envelope from column 'ev_power_envelope_fixed_kw' "
            f"(range {ev_power_envelope.min():.2f} - {ev_power_envelope.max():.2f} kW)"
        )
    else:
        data["time_of_day"] = data[timestamp_col].dt.hour + data[timestamp_col].dt.minute / 60.0
        ev_power_benchmark = ev_demand_actual * 4.0  # kWh → kW (15-min intervals)
        ev_power_envelope = np.zeros(n_periods)
        unique_dates = sorted(data["date"].unique())
        for date in unique_dates:
            day_mask = data["date"] == date
            day_data = data[day_mask].copy()
            day_indices = day_data.index.values
            day_power_benchmark = ev_power_benchmark[day_indices]
            day_time_of_day = day_data["time_of_day"].values
            ev_power_cum_max = np.maximum.accumulate(day_power_benchmark)
            t_15_5_mask = day_time_of_day <= 15.5
            if np.any(t_15_5_mask):
                t_15_5_idx = np.where(t_15_5_mask)[0][-1]
                p_max_at_15_30 = ev_power_cum_max[t_15_5_idx]
            else:
                p_max_at_15_30 = ev_power_cum_max[0] if len(ev_power_cum_max) > 0 else 0.0
            for i, idx in enumerate(day_indices):
                time_of_day = day_time_of_day[i]
                if time_of_day <= 15.5:
                    ev_power_envelope[idx] = ev_power_cum_max[i]
                elif 15.5 < time_of_day < 17.0:
                    ev_power_envelope[idx] = p_max_at_15_30 * (17.0 - time_of_day) / 1.5
                else:
                    ev_power_envelope[idx] = 0.0
        data["ev_power_envelope_fixed_kw"] = ev_power_envelope
        print(f"   ✓ Dynamic EV power envelope calculated (from '{ev_col}')")
        print(
            f"   Envelope range: {ev_power_envelope.min():.2f} - {ev_power_envelope.max():.2f} kW"
        )
        print(
            f"   Envelope > 0 periods: {np.sum(ev_power_envelope > 0)} / {n_periods}"
        )
    daily_ev_energy_demand = data.groupby("date")[ev_col].sum().to_dict()  # Total kWh per day
    # Map each period to its date for daily constraint
    period_to_date = {}
    for t in range(n_periods):
        period_date = data.iloc[t]["date"]
        period_to_date[t] = period_date
    
    # Convert spot price to EUR/kWh
    spot_price_eur_per_kwh = spot_price / 1000.0
    
    # Extract billing parameters
    energy_costs = config.get("energy_based_costs", {})
    peak_costs = config.get("peak_based_costs", {})
    injection_costs = config.get("injection_costs", {})
    access_power_cfg = config.get("acces_power", {})
    
    # Energy-based cost parameters
    grid_losses_percentage = float(energy_costs.get("grid_losses_percentage", 0.0))
    energy_rate_eur_per_mwh = sum(
        float(v) for k, v in energy_costs.items() 
        if k != "grid_losses_percentage"
    )
    
    # Peak-based cost parameters
    access_power_price_eur_per_kw = float(peak_costs.get("access_power_price_eur_per_kw", 0.0))
    monthly_peak_price_eur_per_kw = float(peak_costs.get("monthly_peak_price_eur_per_kw", 0.0))
    over_usage_price_eur_per_kw = float(peak_costs.get("over_usage_price_eur_per_kw", 0.0))
    max_access_power = float(peak_costs.get("max_access_power_kw", 50000.0))
    
    # Injection parameters
    imbalance_cost_eur_per_mwh = float(injection_costs.get("imbalance_cost_eur_per_mwh", 0.0))
    net_injection_price_eur_per_mwh = spot_price - imbalance_cost_eur_per_mwh
    net_injection_price_eur_per_kwh = net_injection_price_eur_per_mwh / 1000.0
    
    # Get month information for access power optimization
    print(f"\n[4/9] Setting up optimization model...")
    # Create month using naive datetime (same approach as billing.py)
    naive_ts = pd.to_datetime(data[timestamp_col].dt.strftime("%Y-%m-%d %H:%M:%S"), format="%Y-%m-%d %H:%M:%S")
    data["month"] = naive_ts.dt.to_period("M")
    months = data["month"].unique()
    month_to_idx = {str(m): i for i, m in enumerate(sorted(months))}
    print(f"   Months in optimization: {len(months)} ({', '.join([str(m) for m in sorted(months)[:3]])}...)")
    
    # Map periods to months for access power
    period_to_month = {}
    for t in range(n_periods):
        month_str = str(data.iloc[t]["month"])
        period_to_month[t] = month_to_idx[month_str]
    
    # Create Pyomo model
    model = ConcreteModel()
    print(f"   Created Pyomo model with {n_periods} time periods")
    
    # Sets
    model.T = Set(initialize=range(n_periods), doc="Time periods")
    
    # Parameters
    model.pv = Param(model.T, initialize={i: pv_production[i] for i in range(n_periods)}, doc="PV production kWh")
    model.inflex_load = Param(model.T, initialize={i: inflex_load[i] for i in range(n_periods)}, doc="Inflexible load kWh")
    model.spot_price = Param(model.T, initialize={i: spot_price_eur_per_kwh[i] for i in range(n_periods)}, doc="Spot price EUR/kWh")
    model.net_injection_price = Param(model.T, initialize={i: net_injection_price_eur_per_kwh[i] for i in range(n_periods)}, doc="Net injection price EUR/kWh")
    model.ev_power_envelope = Param(model.T, initialize={i: ev_power_envelope[i] for i in range(n_periods)}, doc="EV power envelope kW (max charging power)")
    
    # Variables
    # EV charging
    model.ev_charge = Var(model.T, domain=NonNegativeReals, doc="EV charging energy kWh per 15-min")
    model.ev_charge_power = Var(model.T, domain=NonNegativeReals, doc="EV charging power kW")
    
    # Grid
    model.grid_consumption = Var(model.T, domain=NonNegativeReals, doc="Grid consumption kWh")
    model.grid_injection = Var(model.T, domain=NonNegativeReals, doc="Grid injection kWh")
    model.grid_power = Var(model.T, domain=Reals, doc="Grid power kW (positive = consumption, negative = injection)")
    
    # Peak tracking variables
    # Monthly peak: one variable per month (the maximum grid power in that month)
    model.M = Set(initialize=range(len(months)), doc="Month indices")
    model.monthly_peak = Var(model.M, domain=NonNegativeReals, doc="Monthly peak power kW")
    
    # Access power: optimization variable per month (optimal contracted capacity)
    model.access_power = Var(model.M, domain=NonNegativeReals, doc="Access power kW (optimized)")
    
    # Binary variables to track access power increases
    model.access_power_increase = Var(model.M, domain=Binary, doc="1 if access power increased in month m")
    
    # Variable to track the access power level when an increase occurs (for lock-in period)
    model.access_power_at_increase = Var(model.M, domain=NonNegativeReals, doc="Access power level when increase occurred")
    
    # Exceedance per period
    model.exceedance = Var(model.T, domain=NonNegativeReals, doc="Exceedance over access power kW")
    
    # Rolling max exceedance: one variable per month (max exceedance over rolling 12-month window)
    model.rolling_max_exceedance = Var(model.M, domain=NonNegativeReals, doc="Rolling max exceedance kW")
    
    # Objective: Minimize Total_cost = Energy_cost + Spot_cost + Peak_costs - Injection_revenue
    
    def objective_rule(model):
        """
        Objective: Minimize total electricity costs.
        
        Energy_cost = Σ(Grid_consumption_kWh[t] × (Fixed_rate/1000 + Grid_losses% × Spot_price[t]))
        Spot_cost = Σ(Grid_consumption_kWh[t] × Spot_price[t])
        Access_cost = Σ(Access_power_kW[m] × 2.9975 €/kW/month)
        Monthly_peak_cost = Σ(Monthly_peak_kW[m] × 4.227 €/kW/month)
        Over_usage_cost = Σ(Rolling_max_exceedance_kW[m] × 4.496 €/kW/month)
        Injection_revenue = -Σ(Grid_injection_kWh[t] × Net_injection_price[t])
        """
        # Energy_cost = Σ(Grid_consumption × (Fixed_rate/1000 + Grid_losses% × Spot_price))
        energy_cost = sum(
            model.grid_consumption[t] * (energy_rate_eur_per_mwh / 1000.0 + 
                                        grid_losses_percentage * model.spot_price[t])
            for t in model.T
        )
        
        # Spot_cost = Σ(Grid_consumption × Spot_price)
        spot_cost = sum(
            model.grid_consumption[t] * model.spot_price[t]
            for t in model.T
        )
        
        # Access_cost = Σ(Access_power × 2.9975 €/kW/month)
        access_cost = sum(
            model.access_power[m] * access_power_price_eur_per_kw
            for m in model.M
        )
        
        # Monthly_peak_cost = Σ(Monthly_peak × 4.227 €/kW/month)
        monthly_peak_cost = sum(
            model.monthly_peak[m] * monthly_peak_price_eur_per_kw
            for m in model.M
        )
        
        # Over_usage_cost = Σ(Rolling_max_exceedance × 4.496 €/kW/month)
        over_usage_cost = sum(
            model.rolling_max_exceedance[m] * over_usage_price_eur_per_kw
            for m in model.M
        )
        
        # Injection_revenue = -Σ(Grid_injection × Net_injection_price)
        # Net_injection_price = Spot_price - Imbalance_cost (21.148 €/MWh)
        injection_revenue = -sum(
            model.grid_injection[t] * model.net_injection_price[t]
            for t in model.T
        )
        
        return energy_cost + spot_cost + access_cost + monthly_peak_cost + over_usage_cost + injection_revenue
    
    model.objective = Objective(rule=objective_rule, sense=minimize)
    print(f"   ✓ Objective function defined")
    
    # Constraints
    print(f"\n[5/9] Adding constraints...")
    
    # EV Charging Constraints:
    # P_ev(t) = E_ev(t) × 4 (kWh → kW conversion)
    # P_ev(t) ≤ P_ev,max(t) (power envelope)
    # Σ(E_ev(t) for t in day) = Daily_EV_demand (daily energy)
    
    def ev_power_constraint(model, t):
        # P_ev(t) = E_ev(t) × 4
        return model.ev_charge_power[t] == model.ev_charge[t] * 4.0
    
    model.ev_power_constraint = Constraint(model.T, rule=ev_power_constraint)
    print(f"   ✓ EV power constraint added")
    
    def ev_power_envelope_constraint(model, t):
        # P_ev(t) ≤ P_ev,max(t)
        return model.ev_charge_power[t] <= model.ev_power_envelope[t]
    
    model.ev_power_envelope_constraint = Constraint(model.T, rule=ev_power_envelope_constraint)
    print(f"   ✓ EV power envelope constraint added")
    
    unique_dates = sorted(data["date"].unique())
    date_to_periods = {}
    for date in unique_dates:
        date_to_periods[date] = [t for t in range(n_periods) if period_to_date[t] == date]
    
    def ev_energy_constraint_daily(model, date):
        # Σ(E_ev(t) for t in day) = Daily_EV_demand
        periods_for_date = date_to_periods[date]
        daily_demand = daily_ev_energy_demand[date]
        return sum(model.ev_charge[t] for t in periods_for_date) == daily_demand
    
    model.D = Set(initialize=unique_dates, doc="Dates")
    model.ev_energy_constraint_daily = Constraint(model.D, rule=ev_energy_constraint_daily)
    print(f"   ✓ EV daily energy constraint added")
    
    # Power Balance: Grid_consumption - Grid_injection = Inflex_load + EV_charge - PV_production
    def power_balance(model, t):
        total_load = model.inflex_load[t] + model.ev_charge[t]
        net_load = total_load - model.pv[t]
        return model.grid_consumption[t] - model.grid_injection[t] == net_load
    
    model.power_balance = Constraint(model.T, rule=power_balance)
    print(f"   ✓ Power balance constraint added")
    
    # Grid Power: P_grid(t) = (Grid_consumption - Grid_injection) × 4 (kW)
    def grid_power_constraint(model, t):
        return model.grid_power[t] == (model.grid_consumption[t] - model.grid_injection[t]) * 4.0
    
    model.grid_power_constraint = Constraint(model.T, rule=grid_power_constraint)
    print(f"   ✓ Grid power constraint added")
    
    # Peak Tracking:
    # Monthly_peak[m] ≥ Grid_consumption_kW[t] for all t in month m
    # Based on grid consumption (offtake) only, not net power
    def monthly_peak_constraint(model, t):
        m = period_to_month[t]
        grid_consumption_kw = model.grid_consumption[t] * 4.0  # kWh → kW
        return model.monthly_peak[m] >= grid_consumption_kw
    
    model.monthly_peak_constraint = Constraint(model.T, rule=monthly_peak_constraint)
    print(f"   ✓ Monthly peak constraint added")
    
    # Exceedance: Exceedance[t] ≥ Monthly_peak[m] - Access_power[m]
    def exceedance_constraint(model, t):
        m = period_to_month[t]
        return model.exceedance[t] >= model.monthly_peak[m] - model.access_power[m]
    
    model.exceedance_constraint = Constraint(model.T, rule=exceedance_constraint)
    print(f"   ✓ Exceedance constraint added")
    
    # Access power adjustment rules:
    # 1. Access power can be increased at any time (monthly)
    # 2. Access power can only be reduced 12 months after increase
    # Big-M constant for access power (upper bound) - loaded from config
    # Detect increases: access_power[m] > access_power[m-1]
    # If access_power[m] > access_power[m-1], then increase[m] = 1
    def detect_increase_constraint_1(model, m):
        if m == 0:
            # First month: no previous month, so no increase possible
            return model.access_power_increase[m] == 0
        else:
            # If access_power[m] > access_power[m-1], then increase[m] must be 1
            # access_power[m] - access_power[m-1] <= max_access_power * increase[m]
            return model.access_power[m] - model.access_power[m-1] <= max_access_power * model.access_power_increase[m]
    
    def detect_increase_constraint_2(model, m):
        if m == 0:
            return Constraint.Skip
        else:
            # If increase[m] = 1, then access_power[m] > access_power[m-1]
            # access_power[m] - access_power[m-1] >= epsilon * increase[m]
            # Using a small epsilon to detect any increase
            epsilon = 0.01  # Small value to detect any increase
            return model.access_power[m] - model.access_power[m-1] >= epsilon * model.access_power_increase[m]
    
    model.detect_increase_constraint_1 = Constraint(model.M, rule=detect_increase_constraint_1)
    model.detect_increase_constraint_2 = Constraint(model.M, rule=detect_increase_constraint_2)
    
    # Track access power level when increase occurs
    def track_increase_level_constraint_1(model, m):
        # If increase[m] = 1, then access_power_at_increase[m] = access_power[m]
        # access_power_at_increase[m] <= access_power[m]
        return model.access_power_at_increase[m] <= model.access_power[m]
    
    def track_increase_level_constraint_2(model, m):
        # If increase[m] = 1, then access_power_at_increase[m] >= access_power[m]
        # access_power_at_increase[m] >= access_power[m] - max_access_power * (1 - increase[m])
        return model.access_power_at_increase[m] >= model.access_power[m] - max_access_power * (1 - model.access_power_increase[m])
    
    def track_increase_level_constraint_3(model, m):
        # If increase[m] = 0, then access_power_at_increase[m] = 0
        return model.access_power_at_increase[m] <= max_access_power * model.access_power_increase[m]
    
    model.track_increase_level_constraint_1 = Constraint(model.M, rule=track_increase_level_constraint_1)
    model.track_increase_level_constraint_2 = Constraint(model.M, rule=track_increase_level_constraint_2)
    model.track_increase_level_constraint_3 = Constraint(model.M, rule=track_increase_level_constraint_3)
    
    # Lock-in constraint: After an increase, access power cannot be reduced for 12 months
    def access_power_lock_in_constraint(model, m, k):
        # For each month m where increase occurred, ensure access_power[m+k] >= access_power_at_increase[m]
        # for k = 0, 1, 2, ..., 11 (12 months including the increase month)
        if m + k >= len(months):
            return Constraint.Skip  # Beyond optimization horizon
        if k > 11:
            return Constraint.Skip  # Only lock for 12 months
        # access_power[m+k] >= access_power_at_increase[m]
        # But only if increase[m] = 1
        # access_power[m+k] >= access_power_at_increase[m] - max_access_power * (1 - increase[m])
        return model.access_power[m+k] >= model.access_power_at_increase[m] - max_access_power * (1 - model.access_power_increase[m])
    
    # Create set for lock-in periods
    model.K = Set(initialize=range(12), doc="Lock-in period months (0-11)")
    model.access_power_lock_in_constraint = Constraint(model.M, model.K, rule=access_power_lock_in_constraint)
    print(f"   ✓ Access power rules constraint added")
    
    # Rolling Max Exceedance: Rolling_max_exceedance[m] ≥ Exceedance[k] for k in [m-11, m]
    def rolling_max_exceedance_constraint(model, m, t):
        period_month = period_to_month[t]
        if period_month <= m and period_month >= max(0, m - 11):
            return model.rolling_max_exceedance[m] >= model.exceedance[t]
        else:
            return Constraint.Skip
    
    model.rolling_max_exceedance_constraint = Constraint(model.M, model.T, rule=rolling_max_exceedance_constraint)
    print(f"   ✓ Rolling max exceedance constraint added")
    
    print(f"   ✓ All constraints added")
    
    # Solve
    print(f"\n[6/9] Solving optimization problem...")
    print(f"   Model statistics:")
    print(f"     - Variables: {len(list(model.component_objects(Var)))} sets")
    print(f"     - Constraints: {len(list(model.component_objects(Constraint)))} sets")
    print(f"     - Objective: {'Minimize' if model.objective.sense == minimize else 'Maximize'}")
    
    solver = SolverFactory('highs')  # Use HiGHS solver (open source, highspy package)
    if solver.available():
        print(f"   Using solver: HiGHS")
        results = solve_highs_model(model, tee=False)
    else:
        # Try other solvers as fallback
        print(f"   HiGHS not available, trying alternatives...")
        for solver_name in ['cbc', 'glpk', 'cplex', 'gurobi']:
            solver = SolverFactory(solver_name)
            if solver.available():
                print(f"   Using solver: {solver_name.upper()}")
                results = solver.solve(model, tee=False)
                break
        else:
            raise RuntimeError(
                "No suitable solver found. Please install HiGHS:\n"
                "  pip install highspy\n"
                "Or install CBC, GLPK, CPLEX, or Gurobi as alternatives."
            )
    
    # Check solution status
    if results.solver.termination_condition.value == 'optimal':
        print(f"   ✓ Optimization solved successfully!")
        print(f"   Objective value: {value(model.objective):,.2f} EUR")
    else:
        print(f"   ⚠ Warning: Solver termination condition: {results.solver.termination_condition.value}")
        print(f"   Objective value: {value(model.objective):,.2f} EUR")
    
    # Extract results
    print(f"\n[7/9] Extracting optimization results...")
    # Map monthly peak back to periods
    monthly_peak_by_period = []
    for t in range(n_periods):
        m = period_to_month[t]
        monthly_peak_by_period.append(value(model.monthly_peak[m]))
    
    # Map optimized access power back to periods
    access_power_by_period = []
    for t in range(n_periods):
        m = period_to_month[t]
        access_power_by_period.append(value(model.access_power[m]))
    
    results_dict = {
        timestamp_col: data[timestamp_col].values,
        "pv_production": pv_production,
        "inflex_load": inflex_load,
        "spot_price": spot_price,
        "ev_demand_actual": ev_demand_actual,
        "ev_power_envelope": ev_power_envelope,
        "ev_charge": [value(model.ev_charge[t]) for t in model.T],
        "ev_charge_power": [value(model.ev_charge_power[t]) for t in model.T],
        "grid_consumption": [value(model.grid_consumption[t]) for t in model.T],
        "grid_injection": [value(model.grid_injection[t]) for t in model.T],
        "grid_power": [value(model.grid_power[t]) for t in model.T],
        "monthly_peak": monthly_peak_by_period,
        "exceedance": [value(model.exceedance[t]) for t in model.T],
        "access_power": access_power_by_period,
    }
    
    results_df = pd.DataFrame(results_dict)
    print(f"   ✓ Results DataFrame created with {len(results_df)} rows")
    
    # Ensure timestamp column is datetime type
    results_df[timestamp_col] = pd.to_datetime(results_df[timestamp_col], errors="coerce")
    
    # Post-process rolling max exceedance correctly
    # Create month using naive datetime (same approach as billing.py)
    naive_ts = pd.to_datetime(results_df[timestamp_col].dt.strftime("%Y-%m-%d %H:%M:%S"), format="%Y-%m-%d %H:%M:%S")
    results_df["month"] = naive_ts.dt.to_period("M")
    rolling_max_exceedance = []
    for i in range(len(results_df)):
        # Find all periods in the same month or previous 11 months up to current period
        current_month = results_df.iloc[i]["month"]
        window_start = max(0, i - 12 * 30 * 4)  # Approximate 12 months
        window = results_df.iloc[window_start:i+1]
        rolling_max_exceedance.append(window["exceedance"].max() if len(window) > 0 else 0.0)
    results_df["rolling_max_exceedance"] = rolling_max_exceedance
    
    # Calculate costs
    # Note: cost calculation and billing is intentionally handled outside this function
    # (e.g. in notebooks), using `billing.py`. This keeps the optimizer focused purely
    # on finding the optimal EV charging and access power schedule.
    print(f"\n[8/9] Optimization finished. Objective value: {value(model.objective):,.2f} EUR")
    
    # Save optimized EV schedule to CSV file
    print(f"\n[9/9] Saving optimized EV schedule...")
    from pathlib import Path
    
    # Create output directory if it doesn't exist
    # Get the project root directory (parent of src directory)
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "output" / "optimised_ts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use original timestamps from input df (with timezone info preserved)
    # Use the stored original_indices to get the corresponding original timestamps
    # These indices correspond to the filtered and sorted data rows (same order as results_df)
    if isinstance(original_timestamps, pd.Series):
        filtered_original_timestamps = original_timestamps.iloc[original_indices].values
    else:
        filtered_original_timestamps = original_timestamps.iloc[original_indices].values if hasattr(original_timestamps, 'iloc') else original_timestamps[original_indices]
    
    # Create output DataFrame with timestamp and ev_deterministic
    output_df = pd.DataFrame({
        'timestamp': filtered_original_timestamps,
        'ev_deterministic': results_df['ev_charge'].values
    })
    
    # Save to CSV
    output_file = output_dir / "deterministic_ev.csv"
    output_df.to_csv(output_file, index=False)
    
    print(f"   ✓ Optimized EV schedule saved to: {output_file}")
    print(f"   Total rows: {len(output_df)}")
    print(f"   Total EV energy: {output_df['ev_deterministic'].sum():.2f} kWh")
    print(f"{'='*80}\n")

    # Minimal summary – external billing logic should be used for detailed costs
    summary = {
        "objective_value": value(model.objective),
    }

    return results_df, summary


def mpc_hp_24h(
    df_window: pd.DataFrame,
    config_path: str,
    hp_config_path: str,
    monthly_peak_so_far: Dict[str, float],
    rolling12_max_exceedance_so_far_by_month: Dict[str, float],
    soc_initial: float,
    soc_slack_penalty_eur_per_soc: Optional[float] = None,
    soc_min_slack_penalty_eur_per_soc: float = 1.0e6,
    monthly_peak_price_multiplier: float = 1.0,
    timestamp_col: str = "timestamp",
    pv_col: str = "pv_production",
    inflex_load_col: str = "inflex_load",
    price_col: str = "price",
    ev_col: str = "ev",
    thermal_load_col: str = "thermal_load",
    outdoor_temp_col: str = "outdoor_temperature",
    access_power_by_month: Dict[str, float] = None,
    buffer_soc_min: Optional[float] = None,
    buffer_soc_min_profile: Optional[Sequence[float]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    HP-only 24h myopic MPC (96 x 15-min steps) using forecasts.

    - Decision: HP electrical input (kWh/15min) and buffer SOC trajectory.
    - Objective: energy + spot + monthly peak cost - injection revenue
      (no access power decision, no rolling exceedance).
    - Constraints:
        - HP thermal output via COP(T_out)
        - HP thermal power limit (kW)
        - Buffer SOC dynamics with losses (uses SOC initial feedback for t=0)
        - Power balance and grid power
        - Monthly planned peak and effective peak (max(planned, peak_so_far))
        - Access power is not a hard constraint in the planner; exceedance is priced via the rolling-12 exceedance terms.

    buffer_soc_min : float, optional
        If set, buffer SOC lower bound is max(soc_min from config, buffer_soc_min). Used by the online
        wrapper for forecast-access stress (raised operational floor).

    buffer_soc_min_profile : sequence of float, optional
        If set, a time-varying SOC minimum (length n_periods) to be applied as
        max(soc_min from config, profile[t]) for each step. This is the preferred
        interface for forecast-stress SOC floors that vary within the 24h horizon.
    """
    if df_window.empty:
        raise ValueError("df_window is empty; expected a 24h (96-row) window.")
    if access_power_by_month is None:
        raise ValueError(
            "mpc_hp_24h requires 'access_power_by_month' mapping month key 'YYYY-MM' -> access power (kW)."
        )
    if rolling12_max_exceedance_so_far_by_month is None:
        raise ValueError(
            "mpc_hp_24h requires 'rolling12_max_exceedance_so_far_by_month' mapping month key 'YYYY-MM' -> "
            "rolling 12-month max exceedance so far (kW)."
        )

    data = df_window.copy()
    data[timestamp_col] = pd.to_datetime(data[timestamp_col], errors="coerce")
    data = data.sort_values(timestamp_col).reset_index(drop=True)
    n_periods = len(data)

    pv_production = data[pv_col].fillna(0.0).values
    inflex_load = data[inflex_load_col].fillna(0.0).values
    spot_price = data[price_col].fillna(0.0).values
    ev_baseline = data[ev_col].fillna(0.0).values
    thermal_load = data[thermal_load_col].fillna(0.0).values
    outdoor_temp = data[outdoor_temp_col].fillna(0.0).values

    config = load_billing_config(config_path)
    hp_config = load_hp_config(hp_config_path)
    cop_data = hp_config["COP_data"]
    thermal_max_kw = float(hp_config["capacity"]["thermal_max_kw"])

    buf = hp_config["buffer"]
    buffer_size_m3 = float(buf["size_m3"])
    water_density_kg_per_m3 = float(buf["water_density_kg_per_m3"])
    cp_kj_per_kg_k = float(buf["cp_kj_per_kg_k"])
    usable_delta_t_k = float(buf["usable_delta_t_k"])
    soc_min_phys = float(buf["soc_min"])
    soc_max = float(buf["soc_max"])
    soc_final = float(buf.get("soc_final", soc_initial))
    loss_coefficient_per_hour = float(buf["loss_coefficient_per_hour"])
    # Scalar floor is kept for backward-compatibility. If a profile is provided, it takes precedence.
    soc_min = (
        float(max(soc_min_phys, float(buffer_soc_min)))
        if buffer_soc_min is not None
        else soc_min_phys
    )
    soc_min_profile = None
    if buffer_soc_min_profile is not None:
        if len(buffer_soc_min_profile) != n_periods:
            raise ValueError(
                "mpc_hp_24h: buffer_soc_min_profile must have length n_periods. "
                f"Got {len(buffer_soc_min_profile)} vs n_periods={n_periods}."
            )
        soc_min_profile = [
            float(max(soc_min_phys, float(x))) for x in buffer_soc_min_profile
        ]

    buffer_capacity_kwh = (
        buffer_size_m3
        * water_density_kg_per_m3
        * cp_kj_per_kg_k
        * usable_delta_t_k
    ) / 3600.0

    cop_values: List[float] = []
    for temp in outdoor_temp:
        if pd.isna(temp):
            cop_values.append(2.5)
        else:
            cop_values.append(float(interpolate_cop(float(temp), cop_data)))
    cop_array = np.array(cop_values, dtype=float)

    energy_costs = config.get("energy_based_costs", {})
    peak_costs = config.get("peak_based_costs", {})
    injection_costs = config.get("injection_costs", {})

    grid_losses_percentage = float(energy_costs.get("grid_losses_percentage", 0.0))
    energy_rate_eur_per_mwh = sum(
        float(v) for k, v in energy_costs.items() if k != "grid_losses_percentage"
    )
    monthly_peak_price_eur_per_kw_cfg = float(
        peak_costs.get("monthly_peak_price_eur_per_kw", 0.0)
    )
    monthly_peak_price_multiplier = float(max(0.0, monthly_peak_price_multiplier))
    monthly_peak_price_eur_per_kw = (
        monthly_peak_price_multiplier * monthly_peak_price_eur_per_kw_cfg
    )
    over_usage_price_eur_per_kw = float(peak_costs.get("over_usage_price_eur_per_kw", 0.0))
    imbalance_cost_eur_per_mwh = float(
        injection_costs.get("imbalance_cost_eur_per_mwh", 0.0)
    )

    spot_price_eur_per_kwh = spot_price / 1000.0
    net_injection_price_eur_per_mwh = spot_price - imbalance_cost_eur_per_mwh
    net_injection_price_eur_per_kwh = net_injection_price_eur_per_mwh / 1000.0

    naive_ts = pd.to_datetime(
        data[timestamp_col].dt.strftime("%Y-%m-%d %H:%M:%S"),
        format="%Y-%m-%d %H:%M:%S",
    )
    data["month"] = naive_ts.dt.to_period("M")
    months = sorted(data["month"].unique())
    month_to_idx = {str(m): i for i, m in enumerate(months)}
    period_to_month = {
        t: month_to_idx[str(data.iloc[t]["month"])] for t in range(n_periods)
    }

    access_power_fixed_by_month_idx: Dict[int, float] = {}
    peak_so_far_by_month_idx: Dict[int, float] = {}
    rolling12_so_far_by_month_idx: Dict[int, float] = {}
    for m in months:
        key = str(m)
        if key not in access_power_by_month:
            raise KeyError(
                "mpc_hp_24h: access_power_by_month is missing entry for month "
                f"{key!r} (needed for this 24h window)."
            )
        if key not in rolling12_max_exceedance_so_far_by_month:
            raise KeyError(
                "mpc_hp_24h: rolling12_max_exceedance_so_far_by_month is missing entry for month "
                f"{key!r} (needed for this 24h window)."
            )
        idx = month_to_idx[key]
        access_power_fixed_by_month_idx[idx] = float(access_power_by_month[key])
        peak_so_far_by_month_idx[idx] = float(monthly_peak_so_far.get(key, 0.0))
        rolling12_so_far_by_month_idx[idx] = float(
            rolling12_max_exceedance_so_far_by_month.get(key, 0.0)
        )

    model = ConcreteModel()
    model.T = Set(initialize=range(n_periods))
    model.M = Set(initialize=range(len(months)))

    model.pv = Param(
        model.T, initialize={i: float(pv_production[i]) for i in range(n_periods)}
    )
    model.inflex_load = Param(
        model.T, initialize={i: float(inflex_load[i]) for i in range(n_periods)}
    )
    model.ev_baseline = Param(
        model.T, initialize={i: float(ev_baseline[i]) for i in range(n_periods)}
    )
    model.thermal_load = Param(
        model.T, initialize={i: float(thermal_load[i]) for i in range(n_periods)}
    )
    model.cop = Param(model.T, initialize={i: float(cop_array[i]) for i in range(n_periods)})
    model.spot_price = Param(
        model.T, initialize={i: float(spot_price_eur_per_kwh[i]) for i in range(n_periods)}
    )
    model.net_injection_price = Param(
        model.T,
        initialize={i: float(net_injection_price_eur_per_kwh[i]) for i in range(n_periods)},
    )

    model.access_power_fixed = Param(model.M, initialize=access_power_fixed_by_month_idx)
    model.peak_so_far = Param(model.M, initialize=peak_so_far_by_month_idx)
    model.rolling12_exceedance_so_far = Param(model.M, initialize=rolling12_so_far_by_month_idx)

    model.grid_consumption = Var(model.T, domain=NonNegativeReals)
    model.grid_injection = Var(model.T, domain=NonNegativeReals)
    model.grid_power = Var(model.T, domain=Reals)

    model.hp_electrical_input = Var(model.T, domain=NonNegativeReals)
    model.hp_thermal_output = Var(model.T, domain=NonNegativeReals)
    model.hp_thermal_power = Var(model.T, domain=NonNegativeReals)

    # Buffer SOC is kept within configured hard bounds.
    model.buffer_soc = Var(model.T, domain=NonNegativeReals)
    model.buffer_energy = Var(model.T, domain=NonNegativeReals)

    model.monthly_peak = Var(model.M, domain=NonNegativeReals)
    model.effective_peak = Var(model.M, domain=NonNegativeReals)
    # Over-usage billing (rolling 12-month max exceedance):
    # - rolling12_max_exceedance[m] is the rolling 12-month maximum exceedance for month m
    # - delta_rolling12[m] is the increase above the locked-in rolling max passed in from the wrapper
    model.rolling12_max_exceedance = Var(model.M, domain=NonNegativeReals)
    model.delta_rolling12 = Var(model.M, domain=NonNegativeReals)

    energy_cost = sum(
        model.grid_consumption[t]
        * (energy_rate_eur_per_mwh / 1000.0 + grid_losses_percentage * model.spot_price[t])
        for t in model.T
    )
    spot_cost = sum(model.grid_consumption[t] * model.spot_price[t] for t in model.T)
    monthly_peak_cost = sum(
        model.effective_peak[m] * monthly_peak_price_eur_per_kw for m in model.M
    )
    # Only penalize *increases* above the already locked-in rolling max.
    # In a 24h rolling-horizon MPC, increasing the rolling-12-month maximum has a multi-month impact.
    # Approximate that long-run effect by multiplying the monthly tariff by 12.
    over_usage_cost = sum(
        12.0 * model.delta_rolling12[m] * over_usage_price_eur_per_kw for m in model.M
    )
    injection_revenue = -sum(
        model.grid_injection[t] * model.net_injection_price[t] for t in model.T
    )

    # One penalty coefficient shared for both SOC-min and SOC-max slack.
    # Backwards compatible: if soc_slack_penalty_eur_per_soc is not set, fall back to old name.
    _pen = (
        soc_min_slack_penalty_eur_per_soc
        if soc_slack_penalty_eur_per_soc is None
        else soc_slack_penalty_eur_per_soc
    )
    soc_slack_penalty = float(max(0.0, float(_pen)))
    model.soc_min_slack = Var(model.T, domain=NonNegativeReals)
    model.soc_max_slack = Var(model.T, domain=NonNegativeReals)

    def _soc_floor_at(t: int) -> float:
        if soc_min_profile is not None:
            return float(soc_min_profile[t])
        return float(soc_min)

    # Soft SOC-min via slack: buffer_soc[t] + slack[t] >= floor[t]
    def soc_min_soft(model, t):
        return model.buffer_soc[t] + model.soc_min_slack[t] >= _soc_floor_at(int(t))

    # Soft SOC-max via slack: buffer_soc[t] <= soc_max + slack[t]
    def soc_max_soft(model, t):
        return model.buffer_soc[t] <= float(soc_max) + model.soc_max_slack[t]

    model.objective = Objective(
        expr=energy_cost
        + spot_cost
        + monthly_peak_cost
        + over_usage_cost
        + injection_revenue
        + soc_slack_penalty
        * (
            sum(model.soc_min_slack[t] for t in model.T)
            + sum(model.soc_max_slack[t] for t in model.T)
        ),
        sense=minimize,
    )

    def hp_thermal_output_constraint(model, t):
        return model.hp_thermal_output[t] == model.hp_electrical_input[t] * model.cop[t]

    model.hp_thermal_output_constraint = Constraint(model.T, rule=hp_thermal_output_constraint)

    def hp_thermal_power_constraint(model, t):
        return model.hp_thermal_power[t] == model.hp_thermal_output[t] * 4.0

    model.hp_thermal_power_constraint = Constraint(model.T, rule=hp_thermal_power_constraint)

    def hp_thermal_power_limit(model, t):
        return model.hp_thermal_power[t] <= thermal_max_kw

    model.hp_thermal_power_limit = Constraint(model.T, rule=hp_thermal_power_limit)

    def buffer_energy_constraint(model, t):
        return model.buffer_energy[t] == model.buffer_soc[t] * buffer_capacity_kwh

    model.buffer_energy_constraint = Constraint(model.T, rule=buffer_energy_constraint)

    model.soc_min_soft = Constraint(model.T, rule=soc_min_soft)
    model.soc_max_soft = Constraint(model.T, rule=soc_max_soft)

    loss_rate_per_interval = loss_coefficient_per_hour / 4.0

    def buffer_state_update(model, t):
        if t == 0:
            net = (
                model.hp_thermal_output[t]
                - model.thermal_load[t]
                - model.buffer_energy[t] * loss_rate_per_interval
            )
            return model.buffer_soc[t] == float(soc_initial) + net / buffer_capacity_kwh
        net = (
            model.hp_thermal_output[t]
            - model.thermal_load[t]
            - model.buffer_energy[t - 1] * loss_rate_per_interval
        )
        return model.buffer_soc[t] == model.buffer_soc[t - 1] + net / buffer_capacity_kwh

    model.buffer_state_update = Constraint(model.T, rule=buffer_state_update)

    # Terminal SOC target: enforce ONLY for year-end windows.
    # Rationale: in rolling-horizon MPC, a hard terminal SOC on every 24h solve would
    # implicitly force daily return-to-target behavior. We only want year-end closure.
    #
    # We enforce if either:
    # - the window is truncated (n_periods < 96), which only happens at the dataset tail, OR
    # - the window's last timestamp is the last 15-min slot of 2025 (2025-12-31 23:45).
    #
    # Note: buffer_soc[t] represents SOC after applying interval t decisions.
    last_ts = pd.to_datetime(data[timestamp_col].iloc[-1], errors="coerce")
    end_2025_last_slot = pd.Timestamp("2026-01-01 00:00:00") - pd.Timedelta(minutes=15)
    enforce_terminal_soc = (n_periods < 96) or (not pd.isna(last_ts) and last_ts >= end_2025_last_slot)
    if enforce_terminal_soc:
        model.terminal_soc = Constraint(expr=model.buffer_soc[n_periods - 1] == soc_final)

    def power_balance(model, t):
        total_load = model.inflex_load[t] + model.ev_baseline[t] + model.hp_electrical_input[t]
        net_load = total_load - model.pv[t]
        return model.grid_consumption[t] - model.grid_injection[t] == net_load

    model.power_balance = Constraint(model.T, rule=power_balance)

    def grid_power_constraint(model, t):
        return model.grid_power[t] == (model.grid_consumption[t] - model.grid_injection[t]) * 4.0

    model.grid_power_constraint = Constraint(model.T, rule=grid_power_constraint)

    def monthly_peak_constraint(model, t):
        m = period_to_month[t]
        return model.monthly_peak[m] >= model.grid_consumption[t] * 4.0

    model.monthly_peak_constraint = Constraint(model.T, rule=monthly_peak_constraint)

    model.effective_peak_ge_plan = Constraint(
        model.M, rule=lambda model, m: model.effective_peak[m] >= model.monthly_peak[m]
    )
    model.effective_peak_ge_sofar = Constraint(
        model.M, rule=lambda model, m: model.effective_peak[m] >= model.peak_so_far[m]
    )
    # Rolling 12-month max exceedance modeling:
    # rolling12_max_exceedance[m] must be at least:
    # - the already-locked-in rolling max from history, AND
    # - the planned exceedance in this month (effective_peak - access_power_fixed)
    # delta_rolling12[m] captures any increase beyond history.
    model.rolling12_ge_so_far = Constraint(
        model.M,
        rule=lambda model, m: model.rolling12_max_exceedance[m]
        >= model.rolling12_exceedance_so_far[m],
    )
    model.rolling12_ge_exceedance = Constraint(
        model.M,
        rule=lambda model, m: model.rolling12_max_exceedance[m]
        >= model.effective_peak[m] - model.access_power_fixed[m],
    )
    model.delta_rolling12_def = Constraint(
        model.M,
        rule=lambda model, m: model.delta_rolling12[m]
        >= model.rolling12_max_exceedance[m] - model.rolling12_exceedance_so_far[m],
    )

    solve_highs_model(model, tee=False)

    monthly_peak_plan_by_period = [
        float(value(model.monthly_peak[period_to_month[t]])) for t in range(n_periods)
    ]
    effective_peak_by_period = [
        float(value(model.effective_peak[period_to_month[t]])) for t in range(n_periods)
    ]
    access_power_by_period = [
        float(access_power_fixed_by_month_idx[period_to_month[t]]) for t in range(n_periods)
    ]
    rolling12_max_exceedance_by_period = [
        float(value(model.rolling12_max_exceedance[period_to_month[t]])) for t in range(n_periods)
    ]
    rolling12_increment_by_period = [
        float(value(model.delta_rolling12[period_to_month[t]])) for t in range(n_periods)
    ]

    results_df = pd.DataFrame(
        {
            timestamp_col: data[timestamp_col].values,
            "pv_production": pv_production,
            "inflex_load": inflex_load,
            "spot_price_eur_per_mwh": spot_price,
            "ev_baseline": ev_baseline,
            "thermal_load": thermal_load,
            "outdoor_temperature": outdoor_temp,
            "cop": cop_array,
            "hp_electrical_input": [float(value(model.hp_electrical_input[t])) for t in model.T],
            "hp_thermal_output": [float(value(model.hp_thermal_output[t])) for t in model.T],
            "buffer_soc": [float(value(model.buffer_soc[t])) for t in model.T],
            "grid_consumption": [float(value(model.grid_consumption[t])) for t in model.T],
            "grid_injection": [float(value(model.grid_injection[t])) for t in model.T],
            "grid_power": [float(value(model.grid_power[t])) for t in model.T],
            "monthly_peak_plan": monthly_peak_plan_by_period,
            "effective_peak": effective_peak_by_period,
            "access_power_fixed": access_power_by_period,
            # Backwards-compatible name; now represents rolling 12-month max exceedance.
            "access_overage_kw": rolling12_max_exceedance_by_period,
            "rolling12_max_exceedance_kw": rolling12_max_exceedance_by_period,
            "rolling12_increment_kw": rolling12_increment_by_period,
        }
    )
    results_df[timestamp_col] = pd.to_datetime(results_df[timestamp_col], errors="coerce")
    results_df["month"] = results_df[timestamp_col].dt.to_period("M")

    summary = {
        "objective_value": float(value(model.objective)),
        "months_in_window": [str(m) for m in months],
        "monthly_peak_plan": {
            str(months[i]): float(value(model.monthly_peak[i])) for i in range(len(months))
        },
        "effective_peak": {
            str(months[i]): float(value(model.effective_peak[i])) for i in range(len(months))
        },
        "rolling12_max_exceedance_kw": {
            str(months[i]): float(value(model.rolling12_max_exceedance[i]))
            for i in range(len(months))
        },
        "rolling12_increment_kw": {
            str(months[i]): float(value(model.delta_rolling12[i])) for i in range(len(months))
        },
    }
    return results_df, summary


def deterministic_mpc_hp(
    df: pd.DataFrame,
    config_path: str,
    hp_config_path: str,
    timestamp_col: str = "timestamp",
    pv_col: str = "pv_production",
    inflex_load_col: str = "inflex_load",
    price_col: str = "price",
    ev_col: str = "ev",
    thermal_load_col: str = "thermal_load",
    outdoor_temp_col: str = "outdoor_temperature",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Solve deterministic MPC optimization for heat pump system over full year.
    
    EV charging is treated as uncontrollable (baseline case from data).
    Heat pump constraints will be added later.
    
    Minimizes total cost (energy + spot + peak-based) and maximizes injection revenues.
    Implements full capacity tariff structure including:
    - Access power cost
    - Monthly peak cost
    - Rolling max exceedance cost (over-usage)
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data with columns: timestamp, pv_production, inflex_load, price, ev
        All energy values in kWh (15-min intervals)
        The 'ev' column contains baseline EV charging (uncontrollable)
    config_path : str
        Path to billing configuration YAML file
    timestamp_col : str
        Name of timestamp column
    pv_col : str
        Name of PV production column
    inflex_load_col : str
        Name of inflexible load column
    price_col : str
        Name of spot price column (EUR/MWh)
    ev_col : str
        Name of EV demand column (baseline, uncontrollable)
    
    Returns
    -------
    results_df : pd.DataFrame
        Results with optimized variables for each time step
    summary : dict
        Summary statistics and costs
    """
    print("=" * 80)
    print("Deterministic MPC Optimization for Heat Pump System")
    print("=" * 80)
    
    # Load billing configuration
    print(f"\n[1/10] Loading billing configuration from: {config_path}")
    config = load_billing_config(config_path)
    print("   ✓ Billing configuration loaded")
    
    # Load HP configuration
    print(f"\n[2/10] Loading HP configuration from: {hp_config_path}")
    hp_config = load_hp_config(hp_config_path)
    cop_data = hp_config['COP_data']
    thermal_max_kw = hp_config['capacity']['thermal_max_kw']
    
    # Buffer parameters
    buffer_size_m3 = hp_config['buffer']['size_m3']
    water_density_kg_per_m3 = hp_config['buffer']['water_density_kg_per_m3']
    cp_kj_per_kg_k = hp_config['buffer']['cp_kj_per_kg_k']
    usable_delta_t_k = hp_config['buffer']['usable_delta_t_k']
    soc_min = hp_config['buffer']['soc_min']
    soc_max = hp_config['buffer']['soc_max']
    soc_initial = hp_config['buffer']['soc_initial']
    soc_final = float(hp_config['buffer'].get('soc_final', soc_initial))
    loss_coefficient_per_hour = hp_config['buffer']['loss_coefficient_per_hour']
    
    # Calculate buffer capacity in kWh from physical parameters
    # Formula: capacity_kwh = volume_m3 × density_kg_per_m3 × cp_kj_per_kg_k × delta_t_k / 3600
    # 1 kWh = 3600 kJ
    buffer_capacity_kwh = (buffer_size_m3 * water_density_kg_per_m3 * cp_kj_per_kg_k * usable_delta_t_k) / 3600.0
    
    print("   ✓ HP configuration loaded")
    print(f"     - Thermal max capacity: {thermal_max_kw} kW")
    print(f"     - Buffer size: {buffer_size_m3} m³")
    print(f"     - Buffer capacity: {buffer_capacity_kwh:.2f} kWh (calculated)")
    print(f"     - SOC limits: {soc_min:.2f} - {soc_max:.2f}")
    print(f"     - Initial SOC: {soc_initial:.2f}")
    
    # Prepare data
    print(f"\n[3/10] Preparing input data...")
    print(f"   Input DataFrame shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    # Store original timestamps with timezone info before any processing
    # This ensures we can save them correctly later
    original_timestamps = df[timestamp_col].copy()
    
    data = df.copy()
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
    
    # Filter to 2025 calendar year in Belgium local time
    # Filter using naive datetime comparison (treating as Belgium local time)
    start_2025 = pd.Timestamp("2025-01-01 00:00:00")
    end_2025 = pd.Timestamp("2026-01-01 00:00:00")
    data = data[(data[timestamp_col] >= start_2025) & (data[timestamp_col] < end_2025)].copy()
    data = data.sort_values(timestamp_col)
    
    # Store original indices before reset_index (needed to map back to original timestamps)
    original_indices = data.index.values
    
    data = data.reset_index(drop=True)
    
    n_periods = len(data)
    print(f"   ✓ Filtered to {n_periods} periods (15-min intervals)")
    print(f"   Date range: {data[timestamp_col].min()} to {data[timestamp_col].max()}")
    
    # Extract parameters
    print(f"\n[4/10] Extracting input parameters...")
    pv_production = data[pv_col].fillna(0.0).values  # kWh per 15-min
    inflex_load = data[inflex_load_col].fillna(0.0).values  # kWh per 15-min
    spot_price = data[price_col].fillna(0.0).values  # EUR/MWh
    ev_baseline = data[ev_col].fillna(0.0).values  # kWh per 15-min (baseline, uncontrollable)
    thermal_load = data[thermal_load_col].fillna(0.0).values  # kWh per 15-min
    outdoor_temp = data[outdoor_temp_col].fillna(0.0).values  # °C
    
    # Calculate COP for each timestep
    cop_values = []
    for temp in outdoor_temp:
        if pd.isna(temp):
            cop_values.append(2.5)  # Default COP if temperature is missing
        else:
            cop = interpolate_cop(float(temp), cop_data)
            cop_values.append(cop)
    cop_array = np.array(cop_values)
    
    print(f"   PV production: {pv_production.sum():.2f} kWh total")
    print(f"   Inflexible load: {inflex_load.sum():.2f} kWh total")
    print(f"   EV baseline (uncontrollable): {ev_baseline.sum():.2f} kWh total")
    print(f"   Thermal load: {thermal_load.sum():.2f} kWh total")
    print(f"   Spot price range: {spot_price.min():.2f} - {spot_price.max():.2f} EUR/MWh")
    print(f"   Outdoor temp range: {outdoor_temp.min():.2f} - {outdoor_temp.max():.2f} °C")
    print(f"   COP range: {cop_array.min():.2f} - {cop_array.max():.2f}")
    
    # Convert spot price to EUR/kWh
    spot_price_eur_per_kwh = spot_price / 1000.0
    
    # Extract billing parameters
    energy_costs = config.get("energy_based_costs", {})
    peak_costs = config.get("peak_based_costs", {})
    injection_costs = config.get("injection_costs", {})
    access_power_cfg = config.get("acces_power", {})
    
    # Energy-based cost parameters
    grid_losses_percentage = float(energy_costs.get("grid_losses_percentage", 0.0))
    energy_rate_eur_per_mwh = sum(
        float(v) for k, v in energy_costs.items() 
        if k != "grid_losses_percentage"
    )
    
    # Peak-based cost parameters
    access_power_price_eur_per_kw = float(peak_costs.get("access_power_price_eur_per_kw", 0.0))
    monthly_peak_price_eur_per_kw = float(peak_costs.get("monthly_peak_price_eur_per_kw", 0.0))
    over_usage_price_eur_per_kw = float(peak_costs.get("over_usage_price_eur_per_kw", 0.0))
    max_access_power = float(peak_costs.get("max_access_power_kw", 50000.0))
    
    # Injection parameters
    imbalance_cost_eur_per_mwh = float(injection_costs.get("imbalance_cost_eur_per_mwh", 0.0))
    net_injection_price_eur_per_mwh = spot_price - imbalance_cost_eur_per_mwh
    net_injection_price_eur_per_kwh = net_injection_price_eur_per_mwh / 1000.0
    
    # Get month information for access power optimization
    print(f"\n[5/10] Setting up optimization model...")
    naive_ts = pd.to_datetime(data[timestamp_col].dt.strftime("%Y-%m-%d %H:%M:%S"), format="%Y-%m-%d %H:%M:%S")
    data["month"] = naive_ts.dt.to_period("M")
    months = data["month"].unique()
    month_to_idx = {str(m): i for i, m in enumerate(sorted(months))}
    print(f"   Months in optimization: {len(months)} ({', '.join([str(m) for m in sorted(months)[:3]])}...)")
    
    # Map periods to months for access power
    period_to_month = {}
    for t in range(n_periods):
        month_str = str(data.iloc[t]["month"])
        period_to_month[t] = month_to_idx[month_str]
    
    # Create Pyomo model
    model = ConcreteModel()
    print(f"   Created Pyomo model with {n_periods} time periods")
    
    # Sets
    model.T = Set(initialize=range(n_periods), doc="Time periods")
    
    # Parameters
    model.pv = Param(model.T, initialize={i: pv_production[i] for i in range(n_periods)}, doc="PV production kWh")
    model.inflex_load = Param(model.T, initialize={i: inflex_load[i] for i in range(n_periods)}, doc="Inflexible load kWh")
    model.ev_baseline = Param(model.T, initialize={i: ev_baseline[i] for i in range(n_periods)}, doc="EV baseline (uncontrollable) kWh")
    model.thermal_load = Param(model.T, initialize={i: thermal_load[i] for i in range(n_periods)}, doc="Thermal load demand kWh")
    model.cop = Param(model.T, initialize={i: cop_array[i] for i in range(n_periods)}, doc="COP (coefficient of performance)")
    model.spot_price = Param(model.T, initialize={i: spot_price_eur_per_kwh[i] for i in range(n_periods)}, doc="Spot price EUR/kWh")
    model.net_injection_price = Param(model.T, initialize={i: net_injection_price_eur_per_kwh[i] for i in range(n_periods)}, doc="Net injection price EUR/kWh")
    
    # Variables
    # Grid
    model.grid_consumption = Var(model.T, domain=NonNegativeReals, doc="Grid consumption kWh")
    model.grid_injection = Var(model.T, domain=NonNegativeReals, doc="Grid injection kWh")
    model.grid_power = Var(model.T, domain=Reals, doc="Grid power kW (positive = consumption, negative = injection)")
    
    # Heat pump variables
    model.hp_electrical_input = Var(model.T, domain=NonNegativeReals, doc="HP electrical input kWh per 15-min")
    model.hp_thermal_output = Var(model.T, domain=NonNegativeReals, doc="HP thermal output kWh per 15-min")
    model.hp_thermal_power = Var(model.T, domain=NonNegativeReals, doc="HP thermal power kW")
    
    # Buffer state of charge (SOC) - fraction of capacity (0-1)
    model.buffer_soc = Var(model.T, domain=NonNegativeReals, bounds=(soc_min, soc_max), doc="Buffer SOC (fraction)")
    
    # Buffer energy content (kWh)
    model.buffer_energy = Var(model.T, domain=NonNegativeReals, doc="Buffer energy content kWh")
    
    # Peak tracking variables
    model.M = Set(initialize=range(len(months)), doc="Month indices")
    model.monthly_peak = Var(model.M, domain=NonNegativeReals, doc="Monthly peak power kW")
    
    # Access power: optimization variable per month (optimal contracted capacity)
    model.access_power = Var(model.M, domain=NonNegativeReals, doc="Access power kW (optimized)")
    
    # Binary variables to track access power increases
    model.access_power_increase = Var(model.M, domain=Binary, doc="1 if access power increased in month m")
    model.access_power_at_increase = Var(model.M, domain=NonNegativeReals, doc="Access power level when increase occurred")
    
    # Exceedance per period
    model.exceedance = Var(model.T, domain=NonNegativeReals, doc="Exceedance over access power kW")
    
    # Rolling max exceedance: one variable per month (max exceedance over rolling 12-month window)
    model.rolling_max_exceedance = Var(model.M, domain=NonNegativeReals, doc="Rolling max exceedance kW")
    
    # Objective: Minimize Total_cost = Energy_cost + Spot_cost + Peak_costs - Injection_revenue
    def objective_rule(model):
        """
        Objective: Minimize total electricity costs.
        
        Energy_cost = Σ(Grid_consumption_kWh[t] × (Fixed_rate/1000 + Grid_losses% × Spot_price[t]))
        Spot_cost = Σ(Grid_consumption_kWh[t] × Spot_price[t])
        Access_cost = Σ(Access_power_kW[m] × 2.9975 €/kW/month)
        Monthly_peak_cost = Σ(Monthly_peak_kW[m] × 4.227 €/kW/month)
        Over_usage_cost = Σ(Rolling_max_exceedance_kW[m] × 4.496 €/kW/month)
        Injection_revenue = -Σ(Grid_injection_kWh[t] × Net_injection_price[t])
        """
        # Energy_cost = Σ(Grid_consumption × (Fixed_rate/1000 + Grid_losses% × Spot_price))
        energy_cost = sum(
            model.grid_consumption[t] * (energy_rate_eur_per_mwh / 1000.0 + 
                                        grid_losses_percentage * model.spot_price[t])
            for t in model.T
        )
        
        # Spot_cost = Σ(Grid_consumption × Spot_price)
        spot_cost = sum(
            model.grid_consumption[t] * model.spot_price[t]
            for t in model.T
        )
        
        # Access_cost = Σ(Access_power × 2.9975 €/kW/month)
        access_cost = sum(
            model.access_power[m] * access_power_price_eur_per_kw
            for m in model.M
        )
        
        # Monthly_peak_cost = Σ(Monthly_peak × 4.227 €/kW/month)
        monthly_peak_cost = sum(
            model.monthly_peak[m] * monthly_peak_price_eur_per_kw
            for m in model.M
        )
        
        # Over_usage_cost = Σ(Rolling_max_exceedance × 4.496 €/kW/month)
        over_usage_cost = sum(
            model.rolling_max_exceedance[m] * over_usage_price_eur_per_kw
            for m in model.M
        )
        
        # Injection_revenue = -Σ(Grid_injection × Net_injection_price)
        injection_revenue = -sum(
            model.grid_injection[t] * model.net_injection_price[t]
            for t in model.T
        )
        
        return energy_cost + spot_cost + access_cost + monthly_peak_cost + over_usage_cost + injection_revenue
    
    model.objective = Objective(rule=objective_rule, sense=minimize)
    print(f"   ✓ Objective function defined")
    
    # Constraints
    print(f"\n[6/10] Adding constraints...")
    
    # Heat Pump Constraints:
    # HP thermal output: Q_thermal(t) = P_elec(t) × COP(t) × 0.25 h
    # Q_thermal(t) = E_elec(t) × COP(t)
    def hp_thermal_output_constraint(model, t):
        return model.hp_thermal_output[t] == model.hp_electrical_input[t] * model.cop[t]
    
    model.hp_thermal_output_constraint = Constraint(model.T, rule=hp_thermal_output_constraint)
    print(f"   ✓ HP thermal output constraint added")
    
    # HP thermal power: Q_power(t) = Q_thermal(t) × 4 (kWh → kW)
    def hp_thermal_power_constraint(model, t):
        return model.hp_thermal_power[t] == model.hp_thermal_output[t] * 4.0
    
    model.hp_thermal_power_constraint = Constraint(model.T, rule=hp_thermal_power_constraint)
    print(f"   ✓ HP thermal power constraint added")
    
    # HP thermal power limit: Q_power(t) ≤ Q_thermal,max
    def hp_thermal_power_limit(model, t):
        return model.hp_thermal_power[t] <= thermal_max_kw
    
    model.hp_thermal_power_limit = Constraint(model.T, rule=hp_thermal_power_limit)
    print(f"   ✓ HP thermal power limit constraint added")
    
    # Buffer energy content: E_buffer(t) = SOC(t) × Capacity
    def buffer_energy_constraint(model, t):
        return model.buffer_energy[t] == model.buffer_soc[t] * buffer_capacity_kwh
    
    model.buffer_energy_constraint = Constraint(model.T, rule=buffer_energy_constraint)
    print(f"   ✓ Buffer energy constraint added")
    
    # Buffer state update: SOC(t+1) = SOC(t) + (HP_thermal_output - thermal_load - losses) / capacity
    # Losses: loss_rate = loss_coefficient_per_hour / 4 (per 15-min interval)
    loss_rate_per_interval = loss_coefficient_per_hour / 4.0
    
    def buffer_state_update(model, t):
        if t == 0:
            # Initial condition: SOC(0) = SOC_initial
            net_thermal_flow = model.hp_thermal_output[t] - model.thermal_load[t] - model.buffer_energy[t] * loss_rate_per_interval
            return model.buffer_soc[t] == soc_initial + net_thermal_flow / buffer_capacity_kwh
        else:
            # State update: SOC(t) = SOC(t-1) + (HP_output - thermal_load - losses) / capacity
            net_thermal_flow = model.hp_thermal_output[t] - model.thermal_load[t] - model.buffer_energy[t-1] * loss_rate_per_interval
            return model.buffer_soc[t] == model.buffer_soc[t-1] + net_thermal_flow / buffer_capacity_kwh
    
    model.buffer_state_update = Constraint(model.T, rule=buffer_state_update)
    print(f"   ✓ Buffer state update constraint added")
    print(f"     Note: SOC bounds (SOC_min ≤ SOC(t) ≤ SOC_max) ensure thermal load satisfaction")

    # Terminal SOC target (hard constraint): end of year SOC equals configured target.
    # Note: `buffer_soc[t]` represents the SOC *after* applying decisions at interval `t`.
    model.terminal_soc = Constraint(expr=model.buffer_soc[n_periods - 1] == soc_final)
    print(f"   ✓ Terminal SOC constraint added (SOC_end = SOC_final = {soc_final:.2f})")
    
    # Power Balance: Grid_consumption - Grid_injection = Inflex_load + EV_baseline + HP_electrical_input - PV_production
    def power_balance(model, t):
        total_load = model.inflex_load[t] + model.ev_baseline[t] + model.hp_electrical_input[t]
        net_load = total_load - model.pv[t]
        return model.grid_consumption[t] - model.grid_injection[t] == net_load
    
    model.power_balance = Constraint(model.T, rule=power_balance)
    print(f"   ✓ Power balance constraint added (includes HP)")
    
    # Grid Power: P_grid(t) = (Grid_consumption - Grid_injection) × 4 (kW)
    def grid_power_constraint(model, t):
        return model.grid_power[t] == (model.grid_consumption[t] - model.grid_injection[t]) * 4.0
    
    model.grid_power_constraint = Constraint(model.T, rule=grid_power_constraint)
    print(f"   ✓ Grid power constraint added")
    
    # Peak Tracking:
    # Monthly_peak[m] ≥ Grid_consumption_kW[t] for all t in month m
    def monthly_peak_constraint(model, t):
        m = period_to_month[t]
        grid_consumption_kw = model.grid_consumption[t] * 4.0  # kWh → kW
        return model.monthly_peak[m] >= grid_consumption_kw
    
    model.monthly_peak_constraint = Constraint(model.T, rule=monthly_peak_constraint)
    print(f"   ✓ Monthly peak constraint added")
    
    # Exceedance: Exceedance[t] ≥ Monthly_peak[m] - Access_power[m]
    def exceedance_constraint(model, t):
        m = period_to_month[t]
        return model.exceedance[t] >= model.monthly_peak[m] - model.access_power[m]
    
    model.exceedance_constraint = Constraint(model.T, rule=exceedance_constraint)
    print(f"   ✓ Exceedance constraint added")
    
    # Access Power Rules:
    # - Can increase anytime: Access_power[m] ≥ Access_power[m-1] (if increased)
    # - Lock-in: After increase, cannot reduce for 12 months
    # Big-M constant for access power (upper bound) - loaded from config
    
    def detect_increase_constraint_1(model, m):
        if m == 0:
            return model.access_power_increase[m] == 0
        else:
            return model.access_power[m] - model.access_power[m-1] <= max_access_power * model.access_power_increase[m]
    
    def detect_increase_constraint_2(model, m):
        if m == 0:
            return Constraint.Skip
        else:
            epsilon = 0.01
            return model.access_power[m] - model.access_power[m-1] >= epsilon * model.access_power_increase[m]
    
    model.detect_increase_constraint_1 = Constraint(model.M, rule=detect_increase_constraint_1)
    model.detect_increase_constraint_2 = Constraint(model.M, rule=detect_increase_constraint_2)
    
    def track_increase_level_constraint_1(model, m):
        return model.access_power_at_increase[m] <= model.access_power[m]
    
    def track_increase_level_constraint_2(model, m):
        return model.access_power_at_increase[m] >= model.access_power[m] - max_access_power * (1 - model.access_power_increase[m])
    
    def track_increase_level_constraint_3(model, m):
        return model.access_power_at_increase[m] <= max_access_power * model.access_power_increase[m]
    
    model.track_increase_level_constraint_1 = Constraint(model.M, rule=track_increase_level_constraint_1)
    model.track_increase_level_constraint_2 = Constraint(model.M, rule=track_increase_level_constraint_2)
    model.track_increase_level_constraint_3 = Constraint(model.M, rule=track_increase_level_constraint_3)
    
    def access_power_lock_in_constraint(model, m, k):
        if m + k >= len(months):
            return Constraint.Skip
        if k > 11:
            return Constraint.Skip
        return model.access_power[m+k] >= model.access_power_at_increase[m] - max_access_power * (1 - model.access_power_increase[m])
    
    model.K = Set(initialize=range(12), doc="Lock-in period months (0-11)")
    model.access_power_lock_in_constraint = Constraint(model.M, model.K, rule=access_power_lock_in_constraint)
    print(f"   ✓ Access power rules constraint added")
    
    # Rolling Max Exceedance: Rolling_max_exceedance[m] ≥ Exceedance[k] for k in [m-11, m]
    def rolling_max_exceedance_constraint(model, m, t):
        period_month = period_to_month[t]
        if period_month <= m and period_month >= max(0, m - 11):
            return model.rolling_max_exceedance[m] >= model.exceedance[t]
        else:
            return Constraint.Skip
    
    model.rolling_max_exceedance_constraint = Constraint(model.M, model.T, rule=rolling_max_exceedance_constraint)
    print(f"   ✓ Rolling max exceedance constraint added")
    
    print(f"   ✓ All constraints added")
    
    # Solve
    print(f"\n[7/10] Solving optimization problem...")
    print(f"   Model statistics:")
    print(f"     - Variables: {len(list(model.component_objects(Var)))} sets")
    print(f"     - Constraints: {len(list(model.component_objects(Constraint)))} sets")
    print(f"     - Objective: {'Minimize' if model.objective.sense == minimize else 'Maximize'}")
    
    solver = SolverFactory('highs')
    if solver.available():
        print(f"   Using solver: HiGHS")
        results = solve_highs_model(model, tee=False)
    else:
        print(f"   HiGHS not available, trying alternatives...")
        for solver_name in ['cbc', 'glpk', 'cplex', 'gurobi']:
            solver = SolverFactory(solver_name)
            if solver.available():
                print(f"   Using solver: {solver_name.upper()}")
                results = solver.solve(model, tee=False)
                break
        else:
            raise RuntimeError(
                "No suitable solver found. Please install HiGHS:\n"
                "  pip install highspy\n"
                "Or install CBC, GLPK, CPLEX, or Gurobi as alternatives."
            )
    
    # Check solution status
    if results.solver.termination_condition.value == 'optimal':
        print(f"   ✓ Optimization solved successfully!")
        print(f"   Objective value: {value(model.objective):,.2f} EUR")
    else:
        print(f"   ⚠ Warning: Solver termination condition: {results.solver.termination_condition.value}")
        print(f"   Objective value: {value(model.objective):,.2f} EUR")
    
    # Extract results
    print(f"\n[8/10] Extracting optimization results...")
    monthly_peak_by_period = []
    for t in range(n_periods):
        m = period_to_month[t]
        monthly_peak_by_period.append(value(model.monthly_peak[m]))
    
    access_power_by_period = []
    for t in range(n_periods):
        m = period_to_month[t]
        access_power_by_period.append(value(model.access_power[m]))
    
    results_dict = {
        timestamp_col: data[timestamp_col].values,
        "pv_production": pv_production,
        "inflex_load": inflex_load,
        "spot_price": spot_price,
        "ev_baseline": ev_baseline,
        "thermal_load": thermal_load,
        "outdoor_temperature": outdoor_temp,
        "cop": cop_array,
        "hp_electrical_input": [value(model.hp_electrical_input[t]) for t in model.T],
        "hp_thermal_output": [value(model.hp_thermal_output[t]) for t in model.T],
        "hp_thermal_power": [value(model.hp_thermal_power[t]) for t in model.T],
        "buffer_soc": [value(model.buffer_soc[t]) for t in model.T],
        "buffer_energy": [value(model.buffer_energy[t]) for t in model.T],
        "grid_consumption": [value(model.grid_consumption[t]) for t in model.T],
        "grid_injection": [value(model.grid_injection[t]) for t in model.T],
        "grid_power": [value(model.grid_power[t]) for t in model.T],
        "monthly_peak": monthly_peak_by_period,
        "exceedance": [value(model.exceedance[t]) for t in model.T],
        "access_power": access_power_by_period,
    }
    
    results_df = pd.DataFrame(results_dict)
    print(f"   ✓ Results DataFrame created with {len(results_df)} rows")

    # Convenience columns to match "online" reporting style:
    # - soc_before: SOC at start of interval (t), with t=0 equal to configured initial SOC
    # - soc_after: SOC at end of interval (t) (same as buffer_soc)
    results_df["soc_before"] = results_df["buffer_soc"].shift(1, fill_value=float(soc_initial))
    results_df["soc_after"] = results_df["buffer_soc"]
    
    # Ensure timestamp column is datetime type
    results_df[timestamp_col] = pd.to_datetime(results_df[timestamp_col], errors="coerce")
    
    # Post-process rolling max exceedance
    naive_ts = pd.to_datetime(results_df[timestamp_col].dt.strftime("%Y-%m-%d %H:%M:%S"), format="%Y-%m-%d %H:%M:%S")
    results_df["month"] = naive_ts.dt.to_period("M")
    rolling_max_exceedance = []
    for i in range(len(results_df)):
        current_month = results_df.iloc[i]["month"]
        window_start = max(0, i - 12 * 30 * 4)
        window = results_df.iloc[window_start:i+1]
        rolling_max_exceedance.append(window["exceedance"].max() if len(window) > 0 else 0.0)
    results_df["rolling_max_exceedance"] = rolling_max_exceedance
    
    print(f"\n[9/10] Optimization finished. Objective value: {value(model.objective):,.2f} EUR")
    
    # Save optimized HP schedule to CSV file
    print(f"\n[10/10] Saving optimized HP schedule...")
    from pathlib import Path
    
    # Create output directory if it doesn't exist
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "output" / "optimised_ts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use original timestamps from input df (with timezone info preserved)
    if isinstance(original_timestamps, pd.Series):
        filtered_original_timestamps = original_timestamps.iloc[original_indices].values
    else:
        filtered_original_timestamps = original_timestamps.iloc[original_indices].values if hasattr(original_timestamps, 'iloc') else original_timestamps[original_indices]
    
    # Create output DataFrame with timestamp and hp_deterministic
    output_df = pd.DataFrame({
        'timestamp': filtered_original_timestamps,
        'hp_deterministic': results_df['hp_electrical_input'].values
    })
    
    # Save to CSV
    output_file = output_dir / "deterministic_hp.csv"
    output_df.to_csv(output_file, index=False)
    
    print(f"   ✓ Optimized HP schedule saved to: {output_file}")
    print(f"   Total rows: {len(output_df)}")
    print(f"   Total HP energy: {output_df['hp_deterministic'].sum():.2f} kWh")
    print(f"{'='*80}\n")
    
    summary = {
        "objective_value": value(model.objective),
    }
    
    return results_df, summary


def deterministic_access_power_hp_min_access(
    df: pd.DataFrame,
    config_path: str,
    hp_config_path: str,
    timestamp_col: str = "timestamp",
    pv_col: str = "pv_production",
    inflex_load_col: str = "inflex_load",
    price_col: str = "price",
    ev_col: str = "ev",
    thermal_load_col: str = "thermal_load",
    outdoor_temp_col: str = "outdoor_temperature",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Access power planning scenario for HP + buffer (deterministic, full-year).

    Specialized *planning* optimization to choose contracted access power ahead of time:
    - Same physical model as `deterministic_mpc_hp` (HP + buffer SOC + losses).
    - Same access power rules (monthly decision + increase + 12-month lock-in).
    - Objective: minimize Σ access_power[m] over the year (monthly kW decision variables).

    This function intentionally does NOT modify `deterministic_mpc_hp`.
    """
    print("=" * 80)
    print("Access Power Planning (HP) – Minimize Access Power")
    print("=" * 80)

    # Load billing configuration
    print(f"\n[1/10] Loading billing configuration from: {config_path}")
    config = load_billing_config(config_path)
    print("   ✓ Billing configuration loaded")

    # Load HP configuration
    print(f"\n[2/10] Loading HP configuration from: {hp_config_path}")
    hp_config = load_hp_config(hp_config_path)
    cop_data = hp_config["COP_data"]
    thermal_max_kw = hp_config["capacity"]["thermal_max_kw"]

    # Buffer parameters
    buffer_size_m3 = hp_config["buffer"]["size_m3"]
    water_density_kg_per_m3 = hp_config["buffer"]["water_density_kg_per_m3"]
    cp_kj_per_kg_k = hp_config["buffer"]["cp_kj_per_kg_k"]
    usable_delta_t_k = hp_config["buffer"]["usable_delta_t_k"]
    soc_min = hp_config["buffer"]["soc_min"]
    soc_max = hp_config["buffer"]["soc_max"]
    soc_initial = hp_config["buffer"]["soc_initial"]
    loss_coefficient_per_hour = hp_config["buffer"]["loss_coefficient_per_hour"]

    buffer_capacity_kwh = (
        buffer_size_m3 * water_density_kg_per_m3 * cp_kj_per_kg_k * usable_delta_t_k
    ) / 3600.0

    print("   ✓ HP configuration loaded")
    print(f"     - Thermal max capacity: {thermal_max_kw} kW")
    print(f"     - Buffer size: {buffer_size_m3} m³")
    print(f"     - Buffer capacity: {buffer_capacity_kwh:.2f} kWh (calculated)")
    print(f"     - SOC limits: {soc_min:.2f} - {soc_max:.2f}")
    print(f"     - Initial SOC: {soc_initial:.2f}")

    # Prepare data (match deterministic_mpc_hp timestamp parsing convention)
    print(f"\n[3/10] Preparing input data...")
    print(f"   Input DataFrame shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")

    original_timestamps = df[timestamp_col].copy()

    data = df.copy()
    if (
        data[timestamp_col].dtype == "object"
        or isinstance(data[timestamp_col].iloc[0] if len(data) > 0 else None, str)
    ):
        data[timestamp_col] = (
            data[timestamp_col]
            .astype(str)
            .str.replace(r"[+-]\d{2}:\d{2}$", "", regex=True)
        )
        data[timestamp_col] = pd.to_datetime(data[timestamp_col], errors="coerce")
    else:
        data[timestamp_col] = pd.to_datetime(data[timestamp_col], errors="coerce")
        if data[timestamp_col].dt.tz is not None:
            data[timestamp_col] = data[timestamp_col].dt.tz_localize(None)

    start_2025 = pd.Timestamp("2025-01-01 00:00:00")
    end_2025 = pd.Timestamp("2026-01-01 00:00:00")
    data = data[(data[timestamp_col] >= start_2025) & (data[timestamp_col] < end_2025)].copy()
    data = data.sort_values(timestamp_col)

    original_indices = data.index.values
    data = data.reset_index(drop=True)

    n_periods = len(data)
    print(f"   ✓ Filtered to {n_periods} periods (15-min intervals)")
    print(f"   Date range: {data[timestamp_col].min()} to {data[timestamp_col].max()}")

    # Extract parameters
    print(f"\n[4/10] Extracting input parameters...")
    pv_production = data[pv_col].fillna(0.0).values
    inflex_load = data[inflex_load_col].fillna(0.0).values
    spot_price = data[price_col].fillna(0.0).values
    ev_baseline = data[ev_col].fillna(0.0).values
    thermal_load = data[thermal_load_col].fillna(0.0).values
    outdoor_temp = data[outdoor_temp_col].fillna(0.0).values

    cop_values = []
    for temp in outdoor_temp:
        if pd.isna(temp):
            cop_values.append(2.5)
        else:
            cop_values.append(interpolate_cop(float(temp), cop_data))
    cop_array = np.array(cop_values)

    print(f"   PV production: {pv_production.sum():.2f} kWh total")
    print(f"   Inflexible load: {inflex_load.sum():.2f} kWh total")
    print(f"   EV baseline (uncontrollable): {ev_baseline.sum():.2f} kWh total")
    print(f"   Thermal load: {thermal_load.sum():.2f} kWh total")
    print(f"   Outdoor temp range: {outdoor_temp.min():.2f} - {outdoor_temp.max():.2f} °C")
    print(f"   COP range: {cop_array.min():.2f} - {cop_array.max():.2f}")

    # Only used for bounds in lock-in constraints
    peak_costs = config.get("peak_based_costs", {})
    max_access_power = float(peak_costs.get("max_access_power_kw", 50000.0))

    # Month mapping
    print(f"\n[5/10] Setting up optimization model...")
    naive_ts = pd.to_datetime(
        data[timestamp_col].dt.strftime("%Y-%m-%d %H:%M:%S"),
        format="%Y-%m-%d %H:%M:%S",
    )
    data["month"] = naive_ts.dt.to_period("M")
    months = data["month"].unique()
    month_to_idx = {str(m): i for i, m in enumerate(sorted(months))}

    period_to_month: Dict[int, int] = {}
    for t in range(n_periods):
        month_str = str(data.iloc[t]["month"])
        period_to_month[t] = month_to_idx[month_str]

    model = ConcreteModel()
    print(f"   Created Pyomo model with {n_periods} time periods")

    # Sets
    model.T = Set(initialize=range(n_periods), doc="Time periods")
    model.M = Set(initialize=range(len(months)), doc="Month indices")

    # Parameters
    model.pv = Param(model.T, initialize={i: float(pv_production[i]) for i in range(n_periods)})
    model.inflex_load = Param(model.T, initialize={i: float(inflex_load[i]) for i in range(n_periods)})
    model.ev_baseline = Param(model.T, initialize={i: float(ev_baseline[i]) for i in range(n_periods)})
    model.thermal_load = Param(model.T, initialize={i: float(thermal_load[i]) for i in range(n_periods)})
    model.cop = Param(model.T, initialize={i: float(cop_array[i]) for i in range(n_periods)})

    # Variables
    model.grid_consumption = Var(model.T, domain=NonNegativeReals)
    model.grid_injection = Var(model.T, domain=NonNegativeReals)
    model.grid_power = Var(model.T, domain=Reals)

    model.hp_electrical_input = Var(model.T, domain=NonNegativeReals)
    model.hp_thermal_output = Var(model.T, domain=NonNegativeReals)
    model.hp_thermal_power = Var(model.T, domain=NonNegativeReals)

    model.buffer_soc = Var(model.T, domain=NonNegativeReals, bounds=(soc_min, soc_max))
    model.buffer_energy = Var(model.T, domain=NonNegativeReals)

    model.monthly_peak = Var(model.M, domain=NonNegativeReals)
    model.access_power = Var(model.M, domain=NonNegativeReals)
    model.access_power_increase = Var(model.M, domain=Binary)
    model.access_power_at_increase = Var(model.M, domain=NonNegativeReals)
    model.exceedance = Var(model.T, domain=NonNegativeReals)
    model.rolling_max_exceedance = Var(model.M, domain=NonNegativeReals)

    # Objective: minimize contracted access power (sum over months)
    model.objective = Objective(expr=sum(model.access_power[m] for m in model.M), sense=minimize)
    print("   ✓ Objective defined: minimize Σ access_power[m]")

    # Constraints
    print(f"\n[6/10] Adding constraints...")

    def hp_thermal_output_constraint(model, t):
        return model.hp_thermal_output[t] == model.hp_electrical_input[t] * model.cop[t]

    model.hp_thermal_output_constraint = Constraint(model.T, rule=hp_thermal_output_constraint)

    def hp_thermal_power_constraint(model, t):
        return model.hp_thermal_power[t] == model.hp_thermal_output[t] * 4.0

    model.hp_thermal_power_constraint = Constraint(model.T, rule=hp_thermal_power_constraint)

    def hp_thermal_power_limit(model, t):
        return model.hp_thermal_power[t] <= thermal_max_kw

    model.hp_thermal_power_limit = Constraint(model.T, rule=hp_thermal_power_limit)

    def buffer_energy_constraint(model, t):
        return model.buffer_energy[t] == model.buffer_soc[t] * buffer_capacity_kwh

    model.buffer_energy_constraint = Constraint(model.T, rule=buffer_energy_constraint)

    loss_rate_per_interval = loss_coefficient_per_hour / 4.0

    def buffer_state_update(model, t):
        if t == 0:
            net_thermal_flow = (
                model.hp_thermal_output[t]
                - model.thermal_load[t]
                - model.buffer_energy[t] * loss_rate_per_interval
            )
            return model.buffer_soc[t] == soc_initial + net_thermal_flow / buffer_capacity_kwh
        net_thermal_flow = (
            model.hp_thermal_output[t]
            - model.thermal_load[t]
            - model.buffer_energy[t - 1] * loss_rate_per_interval
        )
        return model.buffer_soc[t] == model.buffer_soc[t - 1] + net_thermal_flow / buffer_capacity_kwh

    model.buffer_state_update = Constraint(model.T, rule=buffer_state_update)

    def power_balance(model, t):
        total_load = model.inflex_load[t] + model.ev_baseline[t] + model.hp_electrical_input[t]
        net_load = total_load - model.pv[t]
        return model.grid_consumption[t] - model.grid_injection[t] == net_load

    model.power_balance = Constraint(model.T, rule=power_balance)

    def grid_power_constraint(model, t):
        return model.grid_power[t] == (model.grid_consumption[t] - model.grid_injection[t]) * 4.0

    model.grid_power_constraint = Constraint(model.T, rule=grid_power_constraint)

    def monthly_peak_constraint(model, t):
        m = period_to_month[t]
        grid_consumption_kw = model.grid_consumption[t] * 4.0
        return model.monthly_peak[m] >= grid_consumption_kw

    model.monthly_peak_constraint = Constraint(model.T, rule=monthly_peak_constraint)

    def exceedance_constraint(model, t):
        m = period_to_month[t]
        return model.exceedance[t] >= model.monthly_peak[m] - model.access_power[m]

    model.exceedance_constraint = Constraint(model.T, rule=exceedance_constraint)

    def detect_increase_constraint_1(model, m):
        if m == 0:
            return model.access_power_increase[m] == 0
        return model.access_power[m] - model.access_power[m - 1] <= max_access_power * model.access_power_increase[m]

    def detect_increase_constraint_2(model, m):
        if m == 0:
            return Constraint.Skip
        epsilon = 0.01
        return model.access_power[m] - model.access_power[m - 1] >= epsilon * model.access_power_increase[m]

    model.detect_increase_constraint_1 = Constraint(model.M, rule=detect_increase_constraint_1)
    model.detect_increase_constraint_2 = Constraint(model.M, rule=detect_increase_constraint_2)

    def track_increase_level_constraint_1(model, m):
        return model.access_power_at_increase[m] <= model.access_power[m]

    def track_increase_level_constraint_2(model, m):
        return model.access_power_at_increase[m] >= model.access_power[m] - max_access_power * (1 - model.access_power_increase[m])

    def track_increase_level_constraint_3(model, m):
        return model.access_power_at_increase[m] <= max_access_power * model.access_power_increase[m]

    model.track_increase_level_constraint_1 = Constraint(model.M, rule=track_increase_level_constraint_1)
    model.track_increase_level_constraint_2 = Constraint(model.M, rule=track_increase_level_constraint_2)
    model.track_increase_level_constraint_3 = Constraint(model.M, rule=track_increase_level_constraint_3)

    model.K = Set(initialize=range(12), doc="Lock-in period months (0-11)")

    def access_power_lock_in_constraint(model, m, k):
        if m + k >= len(months):
            return Constraint.Skip
        if k > 11:
            return Constraint.Skip
        return model.access_power[m + k] >= model.access_power_at_increase[m] - max_access_power * (1 - model.access_power_increase[m])

    model.access_power_lock_in_constraint = Constraint(model.M, model.K, rule=access_power_lock_in_constraint)

    def rolling_max_exceedance_constraint(model, m, t):
        period_month = period_to_month[t]
        if period_month <= m and period_month >= max(0, m - 11):
            return model.rolling_max_exceedance[m] >= model.exceedance[t]
        return Constraint.Skip

    model.rolling_max_exceedance_constraint = Constraint(model.M, model.T, rule=rolling_max_exceedance_constraint)
    print("   ✓ Constraints added")

    # Solve
    print(f"\n[7/10] Solving optimization problem...")
    results = solve_highs_model(model, tee=False)

    term_cond = str(results.solver.termination_condition)
    if term_cond.lower() not in {"optimal", "feasible"}:
        print(f"WARNING: solver termination condition: {term_cond}")
    print("   ✓ Optimization solved")

    # Build results dataframe
    print(f"\n[8/10] Building results DataFrame...")
    monthly_peak_by_period = []
    access_power_by_period = []
    for t in range(n_periods):
        m = period_to_month[t]
        monthly_peak_by_period.append(value(model.monthly_peak[m]))
        access_power_by_period.append(value(model.access_power[m]))

    results_df = pd.DataFrame(
        {
            timestamp_col: data[timestamp_col].values,
            "pv_production": pv_production,
            "inflex_load": inflex_load,
            "spot_price": spot_price,
            "ev_baseline": ev_baseline,
            "thermal_load": thermal_load,
            "outdoor_temperature": outdoor_temp,
            "cop": cop_array,
            "hp_electrical_input": [value(model.hp_electrical_input[t]) for t in model.T],
            "hp_thermal_output": [value(model.hp_thermal_output[t]) for t in model.T],
            "hp_thermal_power": [value(model.hp_thermal_power[t]) for t in model.T],
            "buffer_soc": [value(model.buffer_soc[t]) for t in model.T],
            "buffer_energy": [value(model.buffer_energy[t]) for t in model.T],
            "grid_consumption": [value(model.grid_consumption[t]) for t in model.T],
            "grid_injection": [value(model.grid_injection[t]) for t in model.T],
            "grid_power": [value(model.grid_power[t]) for t in model.T],
            "monthly_peak": monthly_peak_by_period,
            "exceedance": [value(model.exceedance[t]) for t in model.T],
            "access_power": access_power_by_period,
        }
    )

    results_df[timestamp_col] = pd.to_datetime(results_df[timestamp_col], errors="coerce")
    naive_ts2 = pd.to_datetime(
        results_df[timestamp_col].dt.strftime("%Y-%m-%d %H:%M:%S"),
        format="%Y-%m-%d %H:%M:%S",
    )
    results_df["month"] = naive_ts2.dt.to_period("M")

    rolling_max_exceedance = []
    for i in range(len(results_df)):
        window_start = max(0, i - 12 * 30 * 4)
        window = results_df.iloc[window_start : i + 1]
        rolling_max_exceedance.append(window["exceedance"].max() if len(window) > 0 else 0.0)
    results_df["rolling_max_exceedance"] = rolling_max_exceedance

    print(f"\n[9/10] Finished. Objective (Σ access power): {value(model.objective):,.2f} kW-month")

    # Save schedule (HP electrical input) to CSV
    print(f"\n[10/10] Saving optimized HP schedule...")
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "output" / "optimised_ts"
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(original_timestamps, pd.Series):
        filtered_original_timestamps = original_timestamps.iloc[original_indices].values
    else:
        filtered_original_timestamps = (
            original_timestamps.iloc[original_indices].values
            if hasattr(original_timestamps, "iloc")
            else original_timestamps[original_indices]
        )

    output_df = pd.DataFrame(
        {
            "timestamp": filtered_original_timestamps,
            "hp_access_power_minap": results_df["hp_electrical_input"].values,
        }
    )
    output_file = output_dir / "hp_access_power_minap.csv"
    output_df.to_csv(output_file, index=False)
    print(f"   ✓ Saved to: {output_file}")

    summary = {"objective_value": float(value(model.objective))}
    return results_df, summary


def deterministic_mpc_ev_hp(
    df: pd.DataFrame,
    config_path: str,
    hp_config_path: str,
    timestamp_col: str = "timestamp",
    pv_col: str = "pv_production",
    inflex_load_col: str = "inflex_load",
    price_col: str = "price",
    ev_col: str = "ev",
    thermal_load_col: str = "thermal_load",
    outdoor_temp_col: str = "outdoor_temperature",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Solve deterministic MPC optimization for combined EV charging and heat pump system over full year.
    
    This function optimizes both EV charging and heat pump operation simultaneously in a single
    optimization problem, allowing coordination between the two loads to minimize total electricity costs.
    
    Minimizes total cost (energy + spot + peak-based) and maximizes injection revenues.
    Implements full capacity tariff structure including:
    - Access power cost
    - Monthly peak cost
    - Rolling max exceedance cost (over-usage)
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data with columns: timestamp, pv_production, inflex_load, price, ev, thermal_load, outdoor_temperature
        All energy values in kWh (15-min intervals)
        The 'ev' column contains actual EV charging demand (kWh per 15-min) used for envelope calculation
    config_path : str
        Path to billing configuration YAML file
    hp_config_path : str
        Path to heat pump configuration YAML file
    timestamp_col : str
        Name of timestamp column
    pv_col : str
        Name of PV production column
    inflex_load_col : str
        Name of inflexible load column
    price_col : str
        Name of spot price column (EUR/MWh)
    ev_col : str
        Name of EV demand column (actual charging data, used for envelope)
    thermal_load_col : str
        Name of thermal load column
    outdoor_temp_col : str
        Name of outdoor temperature column
    
    Returns
    -------
    results_df : pd.DataFrame
        Results with optimized variables for each time step including:
        - EV charging schedule (ev_charge, ev_charge_power)
        - Heat pump schedule (hp_electrical_input, hp_thermal_output, buffer_soc)
        - Grid consumption/injection
        - Peak tracking variables
    summary : dict
        Summary statistics including objective value
    """
    print("=" * 80)
    print("Deterministic MPC Optimization for Combined EV + Heat Pump System")
    print("=" * 80)
    
    # Load billing configuration
    print(f"\n[1/11] Loading billing configuration from: {config_path}")
    config = load_billing_config(config_path)
    print("   ✓ Billing configuration loaded")
    
    # Load HP configuration
    print(f"\n[2/11] Loading HP configuration from: {hp_config_path}")
    hp_config = load_hp_config(hp_config_path)
    cop_data = hp_config['COP_data']
    thermal_max_kw = hp_config['capacity']['thermal_max_kw']
    
    # Buffer parameters
    buffer_size_m3 = hp_config['buffer']['size_m3']
    water_density_kg_per_m3 = hp_config['buffer']['water_density_kg_per_m3']
    cp_kj_per_kg_k = hp_config['buffer']['cp_kj_per_kg_k']
    usable_delta_t_k = hp_config['buffer']['usable_delta_t_k']
    soc_min = hp_config['buffer']['soc_min']
    soc_max = hp_config['buffer']['soc_max']
    soc_initial = hp_config['buffer']['soc_initial']
    loss_coefficient_per_hour = hp_config['buffer']['loss_coefficient_per_hour']
    
    # Calculate buffer capacity in kWh from physical parameters
    buffer_capacity_kwh = (buffer_size_m3 * water_density_kg_per_m3 * cp_kj_per_kg_k * usable_delta_t_k) / 3600.0
    
    print("   ✓ HP configuration loaded")
    print(f"     - Thermal max capacity: {thermal_max_kw} kW")
    print(f"     - Buffer size: {buffer_size_m3} m³")
    print(f"     - Buffer capacity: {buffer_capacity_kwh:.2f} kWh (calculated)")
    print(f"     - SOC limits: {soc_min:.2f} - {soc_max:.2f}")
    print(f"     - Initial SOC: {soc_initial:.2f}")
    
    # Prepare data
    print(f"\n[3/11] Preparing input data...")
    print(f"   Input DataFrame shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    # Store original timestamps with timezone info before any processing
    original_timestamps = df[timestamp_col].copy()
    
    data = df.copy()
    # Parse timestamps: CSV has Belgium local time with fixed offsets (+01:00 or +02:00)
    # Strip timezone offset and parse as naive datetime (treating as Belgium local time)
    if data[timestamp_col].dtype == 'object' or isinstance(data[timestamp_col].iloc[0] if len(data) > 0 else None, str):
        # If string, strip timezone offset before parsing
        data[timestamp_col] = data[timestamp_col].astype(str).str.replace(r'[+-]\d{2}:\d{2}$', '', regex=True)
        data[timestamp_col] = pd.to_datetime(data[timestamp_col], errors="coerce")
    else:
        # Already datetime, but may have timezone - convert to naive
        data[timestamp_col] = pd.to_datetime(data[timestamp_col], errors="coerce")
        if data[timestamp_col].dt.tz is not None:
            data[timestamp_col] = data[timestamp_col].dt.tz_localize(None)
    
    # Filter to 2025 calendar year in Belgium local time
    start_2025 = pd.Timestamp("2025-01-01 00:00:00")
    end_2025 = pd.Timestamp("2026-01-01 00:00:00")
    data = data[(data[timestamp_col] >= start_2025) & (data[timestamp_col] < end_2025)].copy()
    data = data.sort_values(timestamp_col)
    
    # Store original indices before reset_index
    original_indices = data.index.values
    data = data.reset_index(drop=True)
    
    n_periods = len(data)
    print(f"   ✓ Filtered to {n_periods} periods (15-min intervals)")
    print(f"   Date range: {data[timestamp_col].min()} to {data[timestamp_col].max()}")
    
    # Extract parameters
    print(f"\n[4/11] Extracting input parameters...")
    pv_production = data[pv_col].fillna(0.0).values  # kWh per 15-min
    inflex_load = data[inflex_load_col].fillna(0.0).values  # kWh per 15-min
    spot_price = data[price_col].fillna(0.0).values  # EUR/MWh
    ev_demand_actual = data[ev_col].fillna(0.0).values  # kWh per 15-min (actual charging from data)
    thermal_load = data[thermal_load_col].fillna(0.0).values  # kWh per 15-min
    outdoor_temp = data[outdoor_temp_col].fillna(0.0).values  # °C
    
    # Calculate COP for each timestep
    cop_values = []
    for temp in outdoor_temp:
        if pd.isna(temp):
            cop_values.append(2.5)  # Default COP if temperature is missing
        else:
            cop = interpolate_cop(float(temp), cop_data)
            cop_values.append(cop)
    cop_array = np.array(cop_values)
    
    print(f"   PV production: {pv_production.sum():.2f} kWh total")
    print(f"   Inflexible load: {inflex_load.sum():.2f} kWh total")
    print(f"   EV demand: {ev_demand_actual.sum():.2f} kWh total")
    print(f"   Thermal load: {thermal_load.sum():.2f} kWh total")
    print(f"   Spot price range: {spot_price.min():.2f} - {spot_price.max():.2f} EUR/MWh")
    print(f"   Outdoor temp range: {outdoor_temp.min():.2f} - {outdoor_temp.max():.2f} °C")
    print(f"   COP range: {cop_array.min():.2f} - {cop_array.max():.2f}")
    
    # Calculate EV power envelope (same as deterministic_mpc_ev)
    print(f"\n[5/11] Calculating dynamic EV power envelope...")
    data["date"] = data[timestamp_col].dt.date
    ev_power_benchmark = ev_demand_actual * 4.0  # kWh → kW (15-min intervals)
    data["time_of_day"] = data[timestamp_col].dt.hour + data[timestamp_col].dt.minute / 60.0
    ev_power_envelope = np.zeros(n_periods)
    
    unique_dates = sorted(data["date"].unique())
    for date in unique_dates:
        day_mask = data["date"] == date
        day_data = data[day_mask].copy()
        day_indices = day_data.index.values
        day_power_benchmark = ev_power_benchmark[day_indices]
        day_time_of_day = day_data["time_of_day"].values
        ev_power_cum_max = np.maximum.accumulate(day_power_benchmark)
        
        t_15_5_mask = day_time_of_day <= 15.5
        if np.any(t_15_5_mask):
            t_15_5_idx = np.where(t_15_5_mask)[0][-1]
            p_max_at_15_30 = ev_power_cum_max[t_15_5_idx]
        else:
            p_max_at_15_30 = ev_power_cum_max[0] if len(ev_power_cum_max) > 0 else 0.0
        
        for i, idx in enumerate(day_indices):
            time_of_day = day_time_of_day[i]
            if time_of_day <= 15.5:
                ev_power_envelope[idx] = ev_power_cum_max[i]
            elif 15.5 < time_of_day < 17.0:
                ev_power_envelope[idx] = p_max_at_15_30 * (17.0 - time_of_day) / 1.5
            else:
                ev_power_envelope[idx] = 0.0
    
    print(f"   ✓ Dynamic EV power envelope calculated")
    print(f"   Envelope range: {ev_power_envelope.min():.2f} - {ev_power_envelope.max():.2f} kW")
    print(f"   Envelope > 0 periods: {np.sum(ev_power_envelope > 0)} / {n_periods}")
    
    # Daily EV energy demand (must be satisfied per day)
    daily_ev_energy_demand = data.groupby("date")[ev_col].sum().to_dict()
    period_to_date = {}
    for t in range(n_periods):
        period_date = data.iloc[t]["date"]
        period_to_date[t] = period_date
    
    # Convert spot price to EUR/kWh
    spot_price_eur_per_kwh = spot_price / 1000.0
    
    # Extract billing parameters
    energy_costs = config.get("energy_based_costs", {})
    peak_costs = config.get("peak_based_costs", {})
    injection_costs = config.get("injection_costs", {})
    access_power_cfg = config.get("acces_power", {})
    
    # Energy-based cost parameters
    grid_losses_percentage = float(energy_costs.get("grid_losses_percentage", 0.0))
    energy_rate_eur_per_mwh = sum(
        float(v) for k, v in energy_costs.items() 
        if k != "grid_losses_percentage"
    )
    
    # Peak-based cost parameters
    access_power_price_eur_per_kw = float(peak_costs.get("access_power_price_eur_per_kw", 0.0))
    monthly_peak_price_eur_per_kw = float(peak_costs.get("monthly_peak_price_eur_per_kw", 0.0))
    over_usage_price_eur_per_kw = float(peak_costs.get("over_usage_price_eur_per_kw", 0.0))
    max_access_power = float(peak_costs.get("max_access_power_kw", 50000.0))
    
    # Injection parameters
    imbalance_cost_eur_per_mwh = float(injection_costs.get("imbalance_cost_eur_per_mwh", 0.0))
    net_injection_price_eur_per_mwh = spot_price - imbalance_cost_eur_per_mwh
    net_injection_price_eur_per_kwh = net_injection_price_eur_per_mwh / 1000.0
    
    # Get month information for access power optimization
    print(f"\n[6/11] Setting up optimization model...")
    naive_ts = pd.to_datetime(data[timestamp_col].dt.strftime("%Y-%m-%d %H:%M:%S"), format="%Y-%m-%d %H:%M:%S")
    data["month"] = naive_ts.dt.to_period("M")
    months = data["month"].unique()
    month_to_idx = {str(m): i for i, m in enumerate(sorted(months))}
    print(f"   Months in optimization: {len(months)} ({', '.join([str(m) for m in sorted(months)[:3]])}...)")
    
    # Map periods to months for access power
    period_to_month = {}
    for t in range(n_periods):
        month_str = str(data.iloc[t]["month"])
        period_to_month[t] = month_to_idx[month_str]
    
    # Create Pyomo model
    model = ConcreteModel()
    print(f"   Created Pyomo model with {n_periods} time periods")
    
    # Sets
    model.T = Set(initialize=range(n_periods), doc="Time periods")
    model.M = Set(initialize=range(len(months)), doc="Month indices")
    
    # Parameters
    model.pv = Param(model.T, initialize={i: pv_production[i] for i in range(n_periods)}, doc="PV production kWh")
    model.inflex_load = Param(model.T, initialize={i: inflex_load[i] for i in range(n_periods)}, doc="Inflexible load kWh")
    model.thermal_load = Param(model.T, initialize={i: thermal_load[i] for i in range(n_periods)}, doc="Thermal load demand kWh")
    model.cop = Param(model.T, initialize={i: cop_array[i] for i in range(n_periods)}, doc="COP (coefficient of performance)")
    model.spot_price = Param(model.T, initialize={i: spot_price_eur_per_kwh[i] for i in range(n_periods)}, doc="Spot price EUR/kWh")
    model.net_injection_price = Param(model.T, initialize={i: net_injection_price_eur_per_kwh[i] for i in range(n_periods)}, doc="Net injection price EUR/kWh")
    model.ev_power_envelope = Param(model.T, initialize={i: ev_power_envelope[i] for i in range(n_periods)}, doc="EV power envelope kW (max charging power)")
    
    # Variables
    # EV charging
    model.ev_charge = Var(model.T, domain=NonNegativeReals, doc="EV charging energy kWh per 15-min")
    model.ev_charge_power = Var(model.T, domain=NonNegativeReals, doc="EV charging power kW")
    
    # Heat pump variables
    model.hp_electrical_input = Var(model.T, domain=NonNegativeReals, doc="HP electrical input kWh per 15-min")
    model.hp_thermal_output = Var(model.T, domain=NonNegativeReals, doc="HP thermal output kWh per 15-min")
    model.hp_thermal_power = Var(model.T, domain=NonNegativeReals, doc="HP thermal power kW")
    
    # Buffer state of charge (SOC) - fraction of capacity (0-1)
    model.buffer_soc = Var(model.T, domain=NonNegativeReals, bounds=(soc_min, soc_max), doc="Buffer SOC (fraction)")
    
    # Buffer energy content (kWh)
    model.buffer_energy = Var(model.T, domain=NonNegativeReals, doc="Buffer energy content kWh")
    
    # Grid
    model.grid_consumption = Var(model.T, domain=NonNegativeReals, doc="Grid consumption kWh")
    model.grid_injection = Var(model.T, domain=NonNegativeReals, doc="Grid injection kWh")
    model.grid_power = Var(model.T, domain=Reals, doc="Grid power kW (positive = consumption, negative = injection)")
    
    # Peak tracking variables
    model.monthly_peak = Var(model.M, domain=NonNegativeReals, doc="Monthly peak power kW")
    
    # Access power: optimization variable per month (optimal contracted capacity)
    model.access_power = Var(model.M, domain=NonNegativeReals, doc="Access power kW (optimized)")
    
    # Binary variables to track access power increases
    model.access_power_increase = Var(model.M, domain=Binary, doc="1 if access power increased in month m")
    model.access_power_at_increase = Var(model.M, domain=NonNegativeReals, doc="Access power level when increase occurred")
    
    # Exceedance per period
    model.exceedance = Var(model.T, domain=NonNegativeReals, doc="Exceedance over access power kW")
    
    # Rolling max exceedance: one variable per month (max exceedance over rolling 12-month window)
    model.rolling_max_exceedance = Var(model.M, domain=NonNegativeReals, doc="Rolling max exceedance kW")
    
    # Objective: Minimize Total_cost = Energy_cost + Spot_cost + Peak_costs - Injection_revenue
    def objective_rule(model):
        """
        Objective: Minimize total electricity costs.
        
        Energy_cost = Σ(Grid_consumption_kWh[t] × (Fixed_rate/1000 + Grid_losses% × Spot_price[t]))
        Spot_cost = Σ(Grid_consumption_kWh[t] × Spot_price[t])
        Access_cost = Σ(Access_power_kW[m] × 2.9975 €/kW/month)
        Monthly_peak_cost = Σ(Monthly_peak_kW[m] × 4.227 €/kW/month)
        Over_usage_cost = Σ(Rolling_max_exceedance_kW[m] × 4.496 €/kW/month)
        Injection_revenue = -Σ(Grid_injection_kWh[t] × Net_injection_price[t])
        """
        energy_cost = sum(
            model.grid_consumption[t] * (energy_rate_eur_per_mwh / 1000.0 + 
                                        grid_losses_percentage * model.spot_price[t])
            for t in model.T
        )
        spot_cost = sum(
            model.grid_consumption[t] * model.spot_price[t]
            for t in model.T
        )
        access_cost = sum(
            model.access_power[m] * access_power_price_eur_per_kw
            for m in model.M
        )
        monthly_peak_cost = sum(
            model.monthly_peak[m] * monthly_peak_price_eur_per_kw
            for m in model.M
        )
        over_usage_cost = sum(
            model.rolling_max_exceedance[m] * over_usage_price_eur_per_kw
            for m in model.M
        )
        injection_revenue = -sum(
            model.grid_injection[t] * model.net_injection_price[t]
            for t in model.T
        )
        return energy_cost + spot_cost + access_cost + monthly_peak_cost + over_usage_cost + injection_revenue
    
    model.objective = Objective(rule=objective_rule, sense=minimize)
    print(f"   ✓ Objective function defined")
    
    # Constraints
    print(f"\n[7/11] Adding constraints...")
    
    # EV Constraints:
    # EV charging power: P_ev(t) = E_ev(t) × 4 (kWh → kW)
    def ev_power_constraint(model, t):
        return model.ev_charge_power[t] == model.ev_charge[t] * 4.0
    
    model.ev_power_constraint = Constraint(model.T, rule=ev_power_constraint)
    print(f"   ✓ EV power constraint added")
    
    # EV power envelope: P_ev(t) ≤ P_ev,max(t)
    def ev_envelope_constraint(model, t):
        return model.ev_charge_power[t] <= model.ev_power_envelope[t]
    
    model.ev_envelope_constraint = Constraint(model.T, rule=ev_envelope_constraint)
    print(f"   ✓ EV envelope constraint added")
    
    # EV daily energy demand: Σ(E_ev(t) for t in day) = Daily_EV_demand
    def ev_daily_demand_constraint(model, date):
        periods_for_date = [t for t in model.T if period_to_date[t] == date]
        if len(periods_for_date) == 0:
            return Constraint.Skip
        daily_demand = daily_ev_energy_demand.get(date, 0.0)
        return sum(model.ev_charge[t] for t in periods_for_date) == daily_demand
    
    model.D = Set(initialize=sorted(set(period_to_date.values())), doc="Dates")
    model.ev_daily_demand_constraint = Constraint(model.D, rule=ev_daily_demand_constraint)
    print(f"   ✓ EV daily demand constraint added")
    
    # Heat Pump Constraints:
    # HP thermal output: Q_thermal(t) = E_elec(t) × COP(t)
    def hp_thermal_output_constraint(model, t):
        return model.hp_thermal_output[t] == model.hp_electrical_input[t] * model.cop[t]
    
    model.hp_thermal_output_constraint = Constraint(model.T, rule=hp_thermal_output_constraint)
    print(f"   ✓ HP thermal output constraint added")
    
    # HP thermal power: Q_power(t) = Q_thermal(t) × 4 (kWh → kW)
    def hp_thermal_power_constraint(model, t):
        return model.hp_thermal_power[t] == model.hp_thermal_output[t] * 4.0
    
    model.hp_thermal_power_constraint = Constraint(model.T, rule=hp_thermal_power_constraint)
    print(f"   ✓ HP thermal power constraint added")
    
    # HP thermal power limit: Q_power(t) ≤ Q_thermal,max
    def hp_thermal_power_limit(model, t):
        return model.hp_thermal_power[t] <= thermal_max_kw
    
    model.hp_thermal_power_limit = Constraint(model.T, rule=hp_thermal_power_limit)
    print(f"   ✓ HP thermal power limit constraint added")
    
    # Buffer energy content: E_buffer(t) = SOC(t) × Capacity
    def buffer_energy_constraint(model, t):
        return model.buffer_energy[t] == model.buffer_soc[t] * buffer_capacity_kwh
    
    model.buffer_energy_constraint = Constraint(model.T, rule=buffer_energy_constraint)
    print(f"   ✓ Buffer energy constraint added")
    
    # Buffer state update: SOC(t+1) = SOC(t) + (HP_thermal_output - thermal_load - losses) / capacity
    loss_rate_per_interval = loss_coefficient_per_hour / 4.0
    
    def buffer_state_update(model, t):
        if t == 0:
            net_thermal_flow = model.hp_thermal_output[t] - model.thermal_load[t] - model.buffer_energy[t] * loss_rate_per_interval
            return model.buffer_soc[t] == soc_initial + net_thermal_flow / buffer_capacity_kwh
        else:
            net_thermal_flow = model.hp_thermal_output[t] - model.thermal_load[t] - model.buffer_energy[t-1] * loss_rate_per_interval
            return model.buffer_soc[t] == model.buffer_soc[t-1] + net_thermal_flow / buffer_capacity_kwh
    
    model.buffer_state_update = Constraint(model.T, rule=buffer_state_update)
    print(f"   ✓ Buffer state update constraint added")
    
    # Combined Power Balance: Grid_consumption - Grid_injection = Inflex_load + EV_charge + HP_electrical_input - PV_production
    def power_balance(model, t):
        total_load = model.inflex_load[t] + model.ev_charge[t] + model.hp_electrical_input[t]
        net_load = total_load - model.pv[t]
        return model.grid_consumption[t] - model.grid_injection[t] == net_load
    
    model.power_balance = Constraint(model.T, rule=power_balance)
    print(f"   ✓ Combined power balance constraint added (includes EV + HP)")
    
    # Grid Power: P_grid(t) = (Grid_consumption - Grid_injection) × 4 (kW)
    def grid_power_constraint(model, t):
        return model.grid_power[t] == (model.grid_consumption[t] - model.grid_injection[t]) * 4.0
    
    model.grid_power_constraint = Constraint(model.T, rule=grid_power_constraint)
    print(f"   ✓ Grid power constraint added")
    
    # Peak Tracking:
    # Monthly_peak[m] ≥ Grid_power_kW[t] for all t in month m
    def monthly_peak_constraint(model, t):
        m = period_to_month[t]
        return model.monthly_peak[m] >= model.grid_power[t]
    
    model.monthly_peak_constraint = Constraint(model.T, rule=monthly_peak_constraint)
    print(f"   ✓ Monthly peak constraint added")
    
    # Exceedance: Exceedance[t] ≥ Monthly_peak[m] - Access_power[m]
    def exceedance_constraint(model, t):
        m = period_to_month[t]
        return model.exceedance[t] >= model.monthly_peak[m] - model.access_power[m]
    
    model.exceedance_constraint = Constraint(model.T, rule=exceedance_constraint)
    print(f"   ✓ Exceedance constraint added")
    
    # Access Power Rules:
    def detect_increase_constraint_1(model, m):
        if m == 0:
            return model.access_power_increase[m] == 0
        else:
            return model.access_power[m] - model.access_power[m-1] <= max_access_power * model.access_power_increase[m]
    
    def detect_increase_constraint_2(model, m):
        if m == 0:
            return Constraint.Skip
        else:
            epsilon = 0.01
            return model.access_power[m] - model.access_power[m-1] >= epsilon * model.access_power_increase[m]
    
    model.detect_increase_constraint_1 = Constraint(model.M, rule=detect_increase_constraint_1)
    model.detect_increase_constraint_2 = Constraint(model.M, rule=detect_increase_constraint_2)
    
    def track_increase_level_constraint_1(model, m):
        return model.access_power_at_increase[m] <= model.access_power[m]
    
    def track_increase_level_constraint_2(model, m):
        return model.access_power_at_increase[m] >= model.access_power[m] - max_access_power * (1 - model.access_power_increase[m])
    
    def track_increase_level_constraint_3(model, m):
        return model.access_power_at_increase[m] <= max_access_power * model.access_power_increase[m]
    
    model.track_increase_level_constraint_1 = Constraint(model.M, rule=track_increase_level_constraint_1)
    model.track_increase_level_constraint_2 = Constraint(model.M, rule=track_increase_level_constraint_2)
    model.track_increase_level_constraint_3 = Constraint(model.M, rule=track_increase_level_constraint_3)
    
    def access_power_lock_in_constraint(model, m, k):
        if m + k >= len(months):
            return Constraint.Skip
        if k > 11:
            return Constraint.Skip
        return model.access_power[m+k] >= model.access_power_at_increase[m] - max_access_power * (1 - model.access_power_increase[m])
    
    model.K = Set(initialize=range(12), doc="Lock-in period months (0-11)")
    model.access_power_lock_in_constraint = Constraint(model.M, model.K, rule=access_power_lock_in_constraint)
    print(f"   ✓ Access power rules constraint added")
    
    # Rolling Max Exceedance: Rolling_max_exceedance[m] ≥ Exceedance[k] for k in [m-11, m]
    def rolling_max_exceedance_constraint(model, m, t):
        period_month = period_to_month[t]
        if period_month <= m and period_month >= max(0, m - 11):
            return model.rolling_max_exceedance[m] >= model.exceedance[t]
        else:
            return Constraint.Skip
    
    model.rolling_max_exceedance_constraint = Constraint(model.M, model.T, rule=rolling_max_exceedance_constraint)
    print(f"   ✓ Rolling max exceedance constraint added")
    print(f"   ✓ All constraints added")
    
    # Solve
    print(f"\n[8/11] Solving optimization problem...")
    print(f"   Model statistics:")
    print(f"     - Variables: {len(list(model.component_objects(Var)))} sets")
    print(f"     - Constraints: {len(list(model.component_objects(Constraint)))} sets")
    print(f"     - Objective: {'Minimize' if model.objective.sense == minimize else 'Maximize'}")
    
    solver = SolverFactory('highs')
    if solver.available():
        print(f"   Using solver: HiGHS")
        results = solve_highs_model(model, tee=False)
    else:
        print(f"   HiGHS not available, trying alternatives...")
        for solver_name in ['cbc', 'glpk', 'cplex', 'gurobi']:
            solver = SolverFactory(solver_name)
            if solver.available():
                print(f"   Using solver: {solver_name.upper()}")
                results = solver.solve(model, tee=False)
                break
        else:
            raise RuntimeError(
                "No suitable solver found. Please install HiGHS:\n"
                "  pip install highspy\n"
                "Or install CBC, GLPK, CPLEX, or Gurobi as alternatives."
            )
    
    # Check solution status
    if results.solver.termination_condition.value == 'optimal':
        print(f"   ✓ Optimization solved successfully!")
        print(f"   Objective value: {value(model.objective):,.2f} EUR")
    else:
        print(f"   ⚠ Warning: Solver termination condition: {results.solver.termination_condition.value}")
        print(f"   Objective value: {value(model.objective):,.2f} EUR")
    
    # Extract results
    print(f"\n[9/11] Extracting optimization results...")
    monthly_peak_by_period = []
    for t in range(n_periods):
        m = period_to_month[t]
        monthly_peak_by_period.append(value(model.monthly_peak[m]))
    
    access_power_by_period = []
    for t in range(n_periods):
        m = period_to_month[t]
        access_power_by_period.append(value(model.access_power[m]))
    
    results_dict = {
        timestamp_col: data[timestamp_col].values,
        "pv_production": pv_production,
        "inflex_load": inflex_load,
        "spot_price": spot_price,
        "ev_demand_actual": ev_demand_actual,
        "ev_power_envelope": ev_power_envelope,
        "ev_charge": [value(model.ev_charge[t]) for t in model.T],
        "ev_charge_power": [value(model.ev_charge_power[t]) for t in model.T],
        "thermal_load": thermal_load,
        "outdoor_temperature": outdoor_temp,
        "cop": cop_array,
        "hp_electrical_input": [value(model.hp_electrical_input[t]) for t in model.T],
        "hp_thermal_output": [value(model.hp_thermal_output[t]) for t in model.T],
        "hp_thermal_power": [value(model.hp_thermal_power[t]) for t in model.T],
        "buffer_soc": [value(model.buffer_soc[t]) for t in model.T],
        "buffer_energy": [value(model.buffer_energy[t]) for t in model.T],
        "grid_consumption": [value(model.grid_consumption[t]) for t in model.T],
        "grid_injection": [value(model.grid_injection[t]) for t in model.T],
        "grid_power": [value(model.grid_power[t]) for t in model.T],
        "monthly_peak": monthly_peak_by_period,
        "exceedance": [value(model.exceedance[t]) for t in model.T],
        "access_power": access_power_by_period,
    }
    
    results_df = pd.DataFrame(results_dict)
    print(f"   ✓ Results DataFrame created with {len(results_df)} rows")
    
    # Ensure timestamp column is datetime type
    results_df[timestamp_col] = pd.to_datetime(results_df[timestamp_col], errors="coerce")
    
    # Post-process rolling max exceedance
    naive_ts = pd.to_datetime(results_df[timestamp_col].dt.strftime("%Y-%m-%d %H:%M:%S"), format="%Y-%m-%d %H:%M:%S")
    results_df["month"] = naive_ts.dt.to_period("M")
    rolling_max_exceedance = []
    for i in range(len(results_df)):
        current_month = results_df.iloc[i]["month"]
        window_start = max(0, i - 12 * 30 * 4)
        window = results_df.iloc[window_start:i+1]
        rolling_max_exceedance.append(window["exceedance"].max() if len(window) > 0 else 0.0)
    results_df["rolling_max_exceedance"] = rolling_max_exceedance
    
    print(f"\n[10/11] Optimization finished. Objective value: {value(model.objective):,.2f} EUR")
    
    # Save optimized schedules to CSV file
    print(f"\n[11/11] Saving optimized schedules...")
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "output" / "optimised_ts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use original timestamps from input df
    if isinstance(original_timestamps, pd.Series):
        filtered_original_timestamps = original_timestamps.iloc[original_indices].values
    else:
        filtered_original_timestamps = original_timestamps.iloc[original_indices].values if hasattr(original_timestamps, 'iloc') else original_timestamps[original_indices]
    
    # Create output DataFrame with timestamp and optimized schedules
    output_df = pd.DataFrame({
        'timestamp': filtered_original_timestamps,
        'ev_deterministic': results_df['ev_charge'].values,
        'hp_deterministic': results_df['hp_electrical_input'].values
    })
    
    # Save to CSV
    output_file = output_dir / "deterministic_ev_hp.csv"
    output_df.to_csv(output_file, index=False)
    
    print(f"   ✓ Optimized EV + HP schedules saved to: {output_file}")
    print(f"   Total rows: {len(output_df)}")
    print(f"   Total EV energy: {output_df['ev_deterministic'].sum():.2f} kWh")
    print(f"   Total HP energy: {output_df['hp_deterministic'].sum():.2f} kWh")
    print(f"{'='*80}\n")
    
    summary = {
        "objective_value": value(model.objective),
    }
    
    return results_df, summary


def mpc_ev_24h(
    df_window: pd.DataFrame,
    config_path: str,
    monthly_peak_so_far: Dict[str, float],
    timestamp_col: str = "timestamp",
    pv_col: str = "pv_production",
    inflex_load_col: str = "inflex_load",
    price_col: str = "price",
    ev_col: str = "ev",
    ev_deadline_slack_minutes: int = 0,
    daily_ev_remaining: Optional[Dict[object, float]] = None,
    access_power_by_month: Dict[str, float] = None,
    rolling12_max_exceedance_so_far_by_month: Dict[str, float] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    EV-only 24h myopic MPC (96 x 15-min steps) using forecasts.

    - Objective: energy + spot + monthly peak cost + rolling-12 over-usage cost
      - injection revenue (no access power decision variable).
    - Constraints: EV envelope (with deadline slack), daily EV energy,
      power balance, monthly peak, and rolling-12 exceedance (vs fixed access).
    - Access power is **not** a hard cap on ``grid_power`` in the planner; exceeding
      access raises rolling exceedance and is penalised via over-usage cost (same
      structure as ``mpc_hp_24h``). The online wrapper still clips realised power
      to the contract.

    Parameters
    ----------
    df_window : pd.DataFrame
        24h window (96 rows) with at least: timestamp, pv_production,
        inflex_load, price, ev (EV energy demand, app+forecast).
    config_path : str
        Path to billing configuration YAML file.
    monthly_peak_so_far : Dict[str, float]
        Mapping from month string (e.g. "2025-01") to actual peak so far (kW).
    access_power_by_month : Dict[str, float]
        Required mapping from month key "YYYY-MM" (e.g. "2025-01") to access power (kW).
        This function does not fall back to `config_path` for access power.
    rolling12_max_exceedance_so_far_by_month : Dict[str, float]
        Mapping "YYYY-MM" -> rolling 12-month max exceedance locked in so far (kW),
        aligned with ``mpc_hp_24h`` / online HP wrapper semantics.
    ev_deadline_slack_minutes : int
        Slack before the real 17:00 deadline (0, 15, 30, ...). The MPC envelope
        remains unchanged, except that during the last `ev_deadline_slack_minutes`
        before 17:00 (on the 15-min grid) the MPC EV power envelope is forced to
        0 kW so that charging is pushed into the catch-up logic.

    Returns
    -------
    results_df : pd.DataFrame
        96-row DataFrame with optimized variables for the window.
    summary : dict
        Minimal summary with objective value and monthly peak decisions.
    """
    if df_window.empty:
        raise ValueError("df_window is empty; expected a 24h (96-row) window.")
    if access_power_by_month is None:
        raise ValueError(
            "mpc_ev_24h requires 'access_power_by_month' mapping month key 'YYYY-MM' -> access power (kW)."
        )
    if rolling12_max_exceedance_so_far_by_month is None:
        raise ValueError(
            "mpc_ev_24h requires 'rolling12_max_exceedance_so_far_by_month' mapping "
            "month key 'YYYY-MM' -> rolling 12-month max exceedance so far (kW)."
        )

    data = df_window.copy()

    # Parse timestamps (naive) and derive time features
    data[timestamp_col] = pd.to_datetime(data[timestamp_col], errors="coerce")
    data = data.sort_values(timestamp_col).reset_index(drop=True)
    n_periods = len(data)

    data["date"] = data[timestamp_col].dt.date
    data["time_of_day"] = (
        data[timestamp_col].dt.hour + data[timestamp_col].dt.minute / 60.0
    )

    # Extract basic series (kWh per 15-min, price EUR/MWh)
    pv_production = data[pv_col].fillna(0.0).values
    inflex_load = data[inflex_load_col].fillna(0.0).values
    spot_price = data[price_col].fillna(0.0).values
    ev_req = data[ev_col].fillna(0.0).values

    # Load billing configuration (for energy and peak prices)
    config = load_billing_config(config_path)
    energy_costs = config.get("energy_based_costs", {})
    peak_costs = config.get("peak_based_costs", {})
    injection_costs = config.get("injection_costs", {})
    # NOTE: access power is provided by `access_power_by_month` (YYYY-MM -> kW).
    # We intentionally do not use any access power values from billing.yaml here.

    grid_losses_percentage = float(energy_costs.get("grid_losses_percentage", 0.0))
    energy_rate_eur_per_mwh = sum(
        float(v) for k, v in energy_costs.items() if k != "grid_losses_percentage"
    )
    monthly_peak_price_eur_per_kw = float(
        peak_costs.get("monthly_peak_price_eur_per_kw", 0.0)
    )
    over_usage_price_eur_per_kw = float(
        peak_costs.get("over_usage_price_eur_per_kw", 0.0)
    )
    imbalance_cost_eur_per_mwh = float(
        injection_costs.get("imbalance_cost_eur_per_mwh", 0.0)
    )

    # Convert prices
    spot_price_eur_per_kwh = spot_price / 1000.0
    net_injection_price_eur_per_mwh = spot_price - imbalance_cost_eur_per_mwh
    net_injection_price_eur_per_kwh = net_injection_price_eur_per_mwh / 1000.0

    # ------------------------------------------------------------------
    # EV power envelope for this 24h window
    # ------------------------------------------------------------------
    # For online MPC we expect the caller (e.g. online_MPC_1_EV.py) to
    # provide:
    # - a blended physical EV power envelope per step via
    #   `ev_power_envelope_fixed_kw` (current-day blend of forecast/actual)
    # - a forecast-only envelope via `ev_power_envelope_forecast_kw`
    #
    # The MPC envelope uses the blend only for the *current* calendar day
    # in the window; for all *future* days it must use the forecast-only
    # envelope to avoid unrealistic foresight across days.
    if "ev_power_envelope_fixed_kw" not in data.columns:
        raise KeyError(
            "mpc_ev_24h requires 'ev_power_envelope_fixed_kw' column in df_window "
            "(kW envelope per 15-min step)."
        )
    if "ev_power_envelope_forecast_kw" not in data.columns:
        raise KeyError(
            "mpc_ev_24h requires 'ev_power_envelope_forecast_kw' column in df_window "
            "for all steps in the 24h window (forecast-only EV envelope)."
        )

    # Read blended and forecast-only envelopes
    env_blend_array = pd.to_numeric(
        data["ev_power_envelope_fixed_kw"], errors="coerce"
    ).to_numpy(dtype=float)
    env_forecast_array = pd.to_numeric(
        data["ev_power_envelope_forecast_kw"], errors="coerce"
    ).to_numpy(dtype=float)

    # If any entries are NaN after parsing, this indicates missing data and
    # should be treated as a hard error rather than silently assuming 0 kW.
    if np.isnan(env_blend_array).any():
        raise KeyError(
            "mpc_ev_24h received NaN values in 'ev_power_envelope_fixed_kw' "
            "for the 24h window; envelope data must be complete."
        )
    if np.isnan(env_forecast_array).any():
        raise KeyError(
            "mpc_ev_24h received NaN values in 'ev_power_envelope_forecast_kw' "
            "for the 24h window; forecast envelope data must be complete."
        )

    # Build a combined envelope: for the first calendar day in this window
    # use the blended envelope; for subsequent days use forecast-only.
    unique_dates = sorted(data["date"].unique())
    if len(unique_dates) == 0:
        raise KeyError(
            "mpc_ev_24h requires a non-empty 'date' column in df_window "
            "to construct day-specific EV envelopes."
        )
    current_date = unique_dates[0]

    ev_power_envelope_combined = np.zeros_like(env_blend_array)
    for idx in range(n_periods):
        date_val = data.iloc[idx]["date"]
        if date_val == current_date:
            ev_power_envelope_combined[idx] = env_blend_array[idx]
        else:
            ev_power_envelope_combined[idx] = env_forecast_array[idx]

    # Build MPC envelope by optionally forcing the *tail* of the *current day's*
    # envelope to 0 kW during the final slack period before the real 17:00 deadline
    # (on the 15-min grid). When slack is zero, we simply use the combined envelope
    # as the hard constraint.
    slack_hours = ev_deadline_slack_minutes / 60.0
    if slack_hours <= 0.0:
        # No slack: use the deterministic (combined) envelope as hard constraint.
        ev_power_envelope_mpc = ev_power_envelope_combined.copy()
    else:
        # Start from the deterministic envelope for the whole window.
        ev_power_envelope_mpc = ev_power_envelope_combined.copy()

        # Compute integer 15-min slack steps (e.g. 1 step for 15 min, 2 for 30 min, ...)
        n_steps_slack = int(round(slack_hours / 0.25))
        if n_steps_slack > 0:
            # Only apply deadline slack to the *current* calendar day in this window:
            # force the EV MPC envelope to 0 kW during [17:00 - slack, 17:00).
            #
            # IMPORTANT: do NOT "zero the last steps of the day slice", because the
            # 24h window may start mid-day (e.g. 15:00) and the last rows for the
            # current_date would be near midnight rather than near 17:00.
            slack_start_tod = 17.0 - slack_hours
            mask_slack_band = (
                (data["date"] == current_date)
                & (data["time_of_day"] >= slack_start_tod)
                & (data["time_of_day"] < 17.0)
            )
            slack_idx = data.index[mask_slack_band].to_numpy()

            if slack_idx.size > 0:
                # Snap to exactly n_steps_slack points if rounding created a mismatch.
                # Prefer the last points before 17:00.
                if slack_idx.size > n_steps_slack:
                    slack_idx = slack_idx[-n_steps_slack:]
                ev_power_envelope_mpc[slack_idx] = 0.0

    # Daily EV energy demand in this window (per calendar day), based on ev_col.
    # The online MPC caller is responsible for providing the remaining
    # daily energy requirement via `daily_ev_remaining`. We do not fall
    # back to a baseline sum(ev) requirement, to avoid double-counting.
    period_to_date = {t: data.iloc[t]["date"] for t in range(n_periods)}

    # Month information for this window
    naive_ts = pd.to_datetime(
        data[timestamp_col].dt.strftime("%Y-%m-%d %H:%M:%S"),
        format="%Y-%m-%d %H:%M:%S",
    )
    data["month"] = naive_ts.dt.to_period("M")
    months = sorted(data["month"].unique())
    month_to_idx = {str(m): i for i, m in enumerate(months)}
    period_to_month = {
        t: month_to_idx[str(data.iloc[t]["month"])] for t in range(n_periods)
    }

    # Access power per month (fixed) from required mapping keyed by "YYYY-MM".
    access_power_fixed_by_month_idx: Dict[int, float] = {}
    for m in months:
        month_key = str(m)  # e.g. "2025-01"
        if month_key not in access_power_by_month:
            raise KeyError(
                "mpc_ev_24h: access_power_by_month is missing entry for month "
                f"{month_key!r} (needed for this 24h window)."
            )
        access_kw = float(access_power_by_month[month_key])
        access_power_fixed_by_month_idx[month_to_idx[str(m)]] = access_kw

    # Peak so far (kW) parameter per month idx
    peak_so_far_by_month_idx: Dict[int, float] = {}
    for m in months:
        key = str(m)  # e.g. "2025-01"
        peak_so_far_by_month_idx[month_to_idx[str(m)]] = float(
            monthly_peak_so_far.get(key, 0.0)
        )

    rolling12_so_far_by_month_idx: Dict[int, float] = {}
    for m in months:
        key = str(m)
        if key not in rolling12_max_exceedance_so_far_by_month:
            raise KeyError(
                "mpc_ev_24h: rolling12_max_exceedance_so_far_by_month is missing entry for month "
                f"{key!r} (needed for this 24h window)."
            )
        rolling12_so_far_by_month_idx[month_to_idx[str(m)]] = float(
            rolling12_max_exceedance_so_far_by_month[key]
        )

    # Build Pyomo model
    model = ConcreteModel()
    model.T = Set(initialize=range(n_periods))
    model.M = Set(initialize=range(len(months)))

    model.pv = Param(
        model.T,
        initialize={i: float(pv_production[i]) for i in range(n_periods)},
        doc="PV production kWh",
    )
    model.inflex_load = Param(
        model.T,
        initialize={i: float(inflex_load[i]) for i in range(n_periods)},
        doc="Inflexible load kWh",
    )
    model.spot_price = Param(
        model.T,
        initialize={i: float(spot_price_eur_per_kwh[i]) for i in range(n_periods)},
        doc="Spot price EUR/kWh",
    )
    model.net_injection_price = Param(
        model.T,
        initialize={
            i: float(net_injection_price_eur_per_kwh[i]) for i in range(n_periods)
        },
        doc="Net injection price EUR/kWh",
    )
    model.ev_power_envelope = Param(
        model.T,
        initialize={i: float(ev_power_envelope_mpc[i]) for i in range(n_periods)},
        doc="EV MPC envelope kW",
    )

    # Fixed access power and peak-so-far per month
    model.access_power_fixed = Param(
        model.M,
        initialize=access_power_fixed_by_month_idx,
        doc="Fixed access power kW (from YAML)",
    )
    model.peak_so_far = Param(
        model.M,
        initialize=peak_so_far_by_month_idx,
        doc="Monthly peak so far kW",
    )

    # Decision variables
    model.ev_charge = Var(model.T, domain=NonNegativeReals, doc="EV energy kWh/15min")
    model.ev_charge_power = Var(model.T, domain=NonNegativeReals, doc="EV power kW")
    model.grid_consumption = Var(model.T, domain=NonNegativeReals, doc="Grid kWh")
    model.grid_injection = Var(model.T, domain=NonNegativeReals, doc="Injection kWh")
    model.grid_power = Var(model.T, domain=Reals, doc="Grid power kW")

    model.monthly_peak = Var(model.M, domain=NonNegativeReals, doc="Planned peak kW")
    model.effective_peak = Var(model.M, domain=NonNegativeReals, doc="Eff. peak kW")
    model.rolling12_max_exceedance = Var(
        model.M, domain=NonNegativeReals, doc="Rolling 12-month max exceedance kW"
    )
    model.delta_rolling12 = Var(
        model.M, domain=NonNegativeReals, doc="Increase in rolling max exceedance kW"
    )

    model.rolling12_exceedance_so_far = Param(
        model.M,
        initialize=rolling12_so_far_by_month_idx,
        doc="Rolling max exceedance so far from history (kW)",
    )

    # Daily energy sets and shortfall variables (must exist before objective)
    # Use the dates provided in daily_ev_remaining as the set of days that
    # have an explicit remaining-energy requirement.
    if daily_ev_remaining is None:
        raise KeyError(
            "mpc_ev_24h requires 'daily_ev_remaining' for all dates in the window; "
            "no baseline daily requirement is used."
        )
    dates_in_window = sorted(daily_ev_remaining.keys())
    model.D = Set(initialize=dates_in_window)

    date_to_periods: Dict[object, List[int]] = {}
    for date_val in dates_in_window:
        date_to_periods[date_val] = [
            t for t in range(n_periods) if period_to_date[t] == date_val
        ]

    model.ev_daily_shortfall = Var(
        model.D, domain=NonNegativeReals, doc="EV daily energy shortfall kWh"
    )

    # Penalty for undelivered EV energy (kWh) per day
    EV_SHORTFALL_PENALTY_EUR_PER_KWH = 10000.0

    # Objective
    def objective_24h(model):
        energy_cost = sum(
            model.grid_consumption[t]
            * (energy_rate_eur_per_mwh / 1000.0 + grid_losses_percentage * model.spot_price[t])
            for t in model.T
        )
        spot_cost = sum(
            model.grid_consumption[t] * model.spot_price[t] for t in model.T
        )
        peak_cost = sum(
            model.effective_peak[m] * monthly_peak_price_eur_per_kw for m in model.M
        )
        # Same rolling-12 increment structure as mpc_hp_24h (approximate multi-month impact).
        over_usage_cost = sum(
            12.0 * model.delta_rolling12[m] * over_usage_price_eur_per_kw for m in model.M
        )
        injection_revenue = -sum(
            model.grid_injection[t] * model.net_injection_price[t] for t in model.T
        )
        # Penalise any daily EV energy shortfall very heavily
        shortfall_penalty = sum(
            model.ev_daily_shortfall[d] * EV_SHORTFALL_PENALTY_EUR_PER_KWH
            for d in model.D
        )
        return (
            energy_cost
            + spot_cost
            + peak_cost
            + over_usage_cost
            + injection_revenue
            + shortfall_penalty
        )

    model.objective = Objective(rule=objective_24h, sense=minimize)

    # Constraints
    # EV power link and envelope
    def ev_power_link(model, t):
        return model.ev_charge_power[t] == model.ev_charge[t] * 4.0

    model.ev_power_link = Constraint(model.T, rule=ev_power_link)

    def ev_envelope(model, t):
        return model.ev_charge_power[t] <= model.ev_power_envelope[t]

    model.ev_envelope = Constraint(model.T, rule=ev_envelope)

    # Daily energy constraints in this window (with shortfall slack)
    def ev_daily_energy(model, date_val):
        periods = date_to_periods.get(date_val, [])
        if not periods:
            return Constraint.Skip
        daily_req = float(daily_ev_remaining[date_val])
        return (
            sum(model.ev_charge[t] for t in periods)
            + model.ev_daily_shortfall[date_val]
            == daily_req
        )

    model.ev_daily_energy = Constraint(model.D, rule=ev_daily_energy)

    # Power balance and grid power
    def power_balance(model, t):
        total_load = model.inflex_load[t] + model.ev_charge[t]
        net_load = total_load - model.pv[t]
        return model.grid_consumption[t] - model.grid_injection[t] == net_load

    model.power_balance = Constraint(model.T, rule=power_balance)

    def grid_power_rule(model, t):
        return model.grid_power[t] == (model.grid_consumption[t] - model.grid_injection[t]) * 4.0

    model.grid_power_rule = Constraint(model.T, rule=grid_power_rule)

    # Monthly peak and effective peak
    def monthly_peak_rule(model, t):
        m_idx = period_to_month[t]
        return model.monthly_peak[m_idx] >= model.grid_power[t]

    model.monthly_peak_rule = Constraint(model.T, rule=monthly_peak_rule)

    def effective_peak_ge_month(model, m_idx):
        return model.effective_peak[m_idx] >= model.monthly_peak[m_idx]

    model.effective_peak_ge_month = Constraint(model.M, rule=effective_peak_ge_month)

    def effective_peak_ge_sofar(model, m_idx):
        return model.effective_peak[m_idx] >= model.peak_so_far[m_idx]

    model.effective_peak_ge_sofar = Constraint(model.M, rule=effective_peak_ge_sofar)

    # Rolling 12-month max exceedance (same idea as mpc_hp_24h; access is not a hard grid cap).
    model.rolling12_ge_so_far = Constraint(
        model.M,
        rule=lambda model, m: model.rolling12_max_exceedance[m]
        >= model.rolling12_exceedance_so_far[m],
    )
    model.rolling12_ge_exceedance = Constraint(
        model.M,
        rule=lambda model, m: model.rolling12_max_exceedance[m]
        >= model.effective_peak[m] - model.access_power_fixed[m],
    )
    model.delta_rolling12_def = Constraint(
        model.M,
        rule=lambda model, m: model.delta_rolling12[m]
        >= model.rolling12_max_exceedance[m] - model.rolling12_exceedance_so_far[m],
    )

    # Solve
    results = solve_highs_model(model, tee=False)

    obj_val = float(value(model.objective))

    # Extract results for this window
    monthly_peak_plan_by_period = []
    effective_peak_by_period = []
    rolling12_max_by_period = []
    rolling12_increment_by_period = []
    for t in range(n_periods):
        m_idx = period_to_month[t]
        monthly_peak_plan_by_period.append(float(value(model.monthly_peak[m_idx])))
        effective_peak_by_period.append(float(value(model.effective_peak[m_idx])))
        rolling12_max_by_period.append(float(value(model.rolling12_max_exceedance[m_idx])))
        rolling12_increment_by_period.append(float(value(model.delta_rolling12[m_idx])))

    # For backward-compatibility in results, expose the combined base envelope
    # (current-day blend, future days forecast-only) as ev_power_envelope_base.
    ev_power_envelope_base = ev_power_envelope_combined

    results_dict = {
        timestamp_col: data[timestamp_col].values,
        "pv_production": pv_production,
        "inflex_load": inflex_load,
        "spot_price_eur_per_mwh": spot_price,
        "ev_req": ev_req,
        "ev_power_envelope_base": ev_power_envelope_base,
        "ev_power_envelope_mpc": ev_power_envelope_mpc,
        "ev_charge": [float(value(model.ev_charge[t])) for t in model.T],
        "ev_charge_power": [float(value(model.ev_charge_power[t])) for t in model.T],
        "grid_consumption": [float(value(model.grid_consumption[t])) for t in model.T],
        "grid_injection": [float(value(model.grid_injection[t])) for t in model.T],
        "grid_power": [float(value(model.grid_power[t])) for t in model.T],
        "monthly_peak_plan": monthly_peak_plan_by_period,
        "effective_peak": effective_peak_by_period,
        "access_power_fixed": [
            float(access_power_fixed_by_month_idx[period_to_month[t]]) for t in range(n_periods)
        ],
        "rolling12_max_exceedance_kw": rolling12_max_by_period,
        "rolling12_increment_kw": rolling12_increment_by_period,
    }

    results_df = pd.DataFrame(results_dict)
    results_df[timestamp_col] = pd.to_datetime(results_df[timestamp_col], errors="coerce")
    results_df["month"] = results_df[timestamp_col].dt.to_period("M")

    summary = {
        "objective_value": obj_val,
        "months_in_window": [str(m) for m in months],
        "monthly_peak_plan": {
            str(months[i]): float(value(model.monthly_peak[i])) for i in range(len(months))
        },
        "effective_peak": {
            str(months[i]): float(value(model.effective_peak[i])) for i in range(len(months))
        },
        "rolling12_max_exceedance_kw": {
            str(months[i]): float(value(model.rolling12_max_exceedance[i]))
            for i in range(len(months))
        },
        "rolling12_increment_kw": {
            str(months[i]): float(value(model.delta_rolling12[i])) for i in range(len(months))
        },
    }

    return results_df, summary


def mpc_ev_hp_24h(
    df_window: pd.DataFrame,
    config_path: str,
    hp_config_path: str,
    monthly_peak_so_far: Dict[str, float],
    rolling12_max_exceedance_so_far_by_month: Dict[str, float],
    soc_initial: float,
    daily_ev_remaining: Optional[Dict[object, float]] = None,
    ev_deadline_slack_minutes: int = 0,
    access_power_by_month: Dict[str, float] = None,
    soc_slack_penalty_eur_per_soc: Optional[float] = None,
    soc_min_slack_penalty_eur_per_soc: float = 1.0e6,
    monthly_peak_price_multiplier: float = 1.0,
    timestamp_col: str = "timestamp",
    pv_col: str = "pv_production",
    inflex_load_col: str = "inflex_load",
    price_col: str = "price",
    ev_col: str = "ev",
    thermal_load_col: str = "thermal_load",
    outdoor_temp_col: str = "outdoor_temperature",
    buffer_soc_min_profile: Optional[Sequence[float]] = None,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Joint EV + HP 24h myopic MPC (thesis §3.7.4 Step 2).

    Combines ``mpc_ev_24h`` EV envelope / daily energy constraints with
    ``mpc_hp_24h`` buffer dynamics and shared billing (monthly peak + rolling-12).
    """
    if df_window.empty:
        raise ValueError("df_window is empty; expected a 24h (96-row) window.")
    if access_power_by_month is None:
        raise ValueError(
            "mpc_ev_hp_24h requires 'access_power_by_month' mapping month key 'YYYY-MM' -> access power (kW)."
        )
    if rolling12_max_exceedance_so_far_by_month is None:
        raise ValueError(
            "mpc_ev_hp_24h requires 'rolling12_max_exceedance_so_far_by_month'."
        )
    if daily_ev_remaining is None:
        raise KeyError("mpc_ev_hp_24h requires 'daily_ev_remaining' for all dates in the window.")

    data = df_window.copy()
    data[timestamp_col] = pd.to_datetime(data[timestamp_col], errors="coerce")
    data = data.sort_values(timestamp_col).reset_index(drop=True)
    n_periods = len(data)

    data["date"] = data[timestamp_col].dt.date
    data["time_of_day"] = (
        data[timestamp_col].dt.hour + data[timestamp_col].dt.minute / 60.0
    )

    pv_production = data[pv_col].fillna(0.0).values
    inflex_load = data[inflex_load_col].fillna(0.0).values
    spot_price = data[price_col].fillna(0.0).values
    ev_req = data[ev_col].fillna(0.0).values
    thermal_load = data[thermal_load_col].fillna(0.0).values
    outdoor_temp = data[outdoor_temp_col].fillna(0.0).values

    if "ev_power_envelope_fixed_kw" not in data.columns:
        raise KeyError("mpc_ev_hp_24h requires 'ev_power_envelope_fixed_kw'.")
    if "ev_power_envelope_forecast_kw" not in data.columns:
        raise KeyError("mpc_ev_hp_24h requires 'ev_power_envelope_forecast_kw'.")

    env_blend_array = pd.to_numeric(
        data["ev_power_envelope_fixed_kw"], errors="coerce"
    ).to_numpy(dtype=float)
    env_forecast_array = pd.to_numeric(
        data["ev_power_envelope_forecast_kw"], errors="coerce"
    ).to_numpy(dtype=float)
    if np.isnan(env_blend_array).any() or np.isnan(env_forecast_array).any():
        raise KeyError("EV envelope columns must not contain NaN.")

    unique_dates = sorted(data["date"].unique())
    if not unique_dates:
        raise KeyError("mpc_ev_hp_24h requires non-empty 'date' in df_window.")
    current_date = unique_dates[0]

    ev_power_envelope_combined = np.zeros_like(env_blend_array)
    for idx in range(n_periods):
        if data.iloc[idx]["date"] == current_date:
            ev_power_envelope_combined[idx] = env_blend_array[idx]
        else:
            ev_power_envelope_combined[idx] = env_forecast_array[idx]

    slack_hours = ev_deadline_slack_minutes / 60.0
    ev_power_envelope_mpc = ev_power_envelope_combined.copy()
    if slack_hours > 0.0:
        n_steps_slack = int(round(slack_hours / 0.25))
        if n_steps_slack > 0:
            slack_start_tod = 17.0 - slack_hours
            mask_slack_band = (
                (data["date"] == current_date)
                & (data["time_of_day"] >= slack_start_tod)
                & (data["time_of_day"] < 17.0)
            )
            slack_idx = data.index[mask_slack_band].to_numpy()
            if slack_idx.size > n_steps_slack:
                slack_idx = slack_idx[-n_steps_slack:]
            if slack_idx.size > 0:
                ev_power_envelope_mpc[slack_idx] = 0.0

    config = load_billing_config(config_path)
    hp_config = load_hp_config(hp_config_path)
    cop_data = hp_config["COP_data"]
    thermal_max_kw = float(hp_config["capacity"]["thermal_max_kw"])

    buf = hp_config["buffer"]
    buffer_size_m3 = float(buf["size_m3"])
    water_density_kg_per_m3 = float(buf["water_density_kg_per_m3"])
    cp_kj_per_kg_k = float(buf["cp_kj_per_kg_k"])
    usable_delta_t_k = float(buf["usable_delta_t_k"])
    soc_min_phys = float(buf["soc_min"])
    soc_max = float(buf["soc_max"])
    soc_final = float(buf.get("soc_final", soc_initial))
    loss_coefficient_per_hour = float(buf["loss_coefficient_per_hour"])

    soc_min_profile = None
    if buffer_soc_min_profile is not None:
        if len(buffer_soc_min_profile) != n_periods:
            raise ValueError(
                "mpc_ev_hp_24h: buffer_soc_min_profile length must match window."
            )
        soc_min_profile = [
            float(max(soc_min_phys, float(x))) for x in buffer_soc_min_profile
        ]

    buffer_capacity_kwh = (
        buffer_size_m3
        * water_density_kg_per_m3
        * cp_kj_per_kg_k
        * usable_delta_t_k
    ) / 3600.0

    cop_values: List[float] = []
    for temp in outdoor_temp:
        if pd.isna(temp):
            cop_values.append(2.5)
        else:
            cop_values.append(float(interpolate_cop(float(temp), cop_data)))
    cop_array = np.array(cop_values, dtype=float)

    energy_costs = config.get("energy_based_costs", {})
    peak_costs = config.get("peak_based_costs", {})
    injection_costs = config.get("injection_costs", {})

    grid_losses_percentage = float(energy_costs.get("grid_losses_percentage", 0.0))
    energy_rate_eur_per_mwh = sum(
        float(v) for k, v in energy_costs.items() if k != "grid_losses_percentage"
    )
    monthly_peak_price_eur_per_kw_cfg = float(
        peak_costs.get("monthly_peak_price_eur_per_kw", 0.0)
    )
    monthly_peak_price_multiplier = float(max(0.0, monthly_peak_price_multiplier))
    monthly_peak_price_eur_per_kw = (
        monthly_peak_price_multiplier * monthly_peak_price_eur_per_kw_cfg
    )
    over_usage_price_eur_per_kw = float(peak_costs.get("over_usage_price_eur_per_kw", 0.0))
    imbalance_cost_eur_per_mwh = float(
        injection_costs.get("imbalance_cost_eur_per_mwh", 0.0)
    )

    spot_price_eur_per_kwh = spot_price / 1000.0
    net_injection_price_eur_per_kwh = (spot_price - imbalance_cost_eur_per_mwh) / 1000.0

    naive_ts = pd.to_datetime(
        data[timestamp_col].dt.strftime("%Y-%m-%d %H:%M:%S"),
        format="%Y-%m-%d %H:%M:%S",
    )
    data["month"] = naive_ts.dt.to_period("M")
    months = sorted(data["month"].unique())
    month_to_idx = {str(m): i for i, m in enumerate(months)}
    period_to_month = {t: month_to_idx[str(data.iloc[t]["month"])] for t in range(n_periods)}
    period_to_date = {t: data.iloc[t]["date"] for t in range(n_periods)}

    access_power_fixed_by_month_idx: Dict[int, float] = {}
    peak_so_far_by_month_idx: Dict[int, float] = {}
    rolling12_so_far_by_month_idx: Dict[int, float] = {}
    for m in months:
        key = str(m)
        if key not in access_power_by_month:
            raise KeyError(f"mpc_ev_hp_24h: missing access_power for {key!r}.")
        if key not in rolling12_max_exceedance_so_far_by_month:
            raise KeyError(f"mpc_ev_hp_24h: missing rolling12 for {key!r}.")
        idx = month_to_idx[key]
        access_power_fixed_by_month_idx[idx] = float(access_power_by_month[key])
        peak_so_far_by_month_idx[idx] = float(monthly_peak_so_far.get(key, 0.0))
        rolling12_so_far_by_month_idx[idx] = float(
            rolling12_max_exceedance_so_far_by_month[key]
        )

    dates_in_window = sorted(daily_ev_remaining.keys())
    date_to_periods: Dict[object, List[int]] = {
        d: [t for t in range(n_periods) if period_to_date[t] == d] for d in dates_in_window
    }

    model = ConcreteModel()
    model.T = Set(initialize=range(n_periods))
    model.M = Set(initialize=range(len(months)))
    model.D = Set(initialize=dates_in_window)

    model.pv = Param(model.T, initialize={i: float(pv_production[i]) for i in range(n_periods)})
    model.inflex_load = Param(
        model.T, initialize={i: float(inflex_load[i]) for i in range(n_periods)}
    )
    model.thermal_load = Param(
        model.T, initialize={i: float(thermal_load[i]) for i in range(n_periods)}
    )
    model.cop = Param(model.T, initialize={i: float(cop_array[i]) for i in range(n_periods)})
    model.spot_price = Param(
        model.T, initialize={i: float(spot_price_eur_per_kwh[i]) for i in range(n_periods)}
    )
    model.net_injection_price = Param(
        model.T,
        initialize={i: float(net_injection_price_eur_per_kwh[i]) for i in range(n_periods)},
    )
    model.ev_power_envelope = Param(
        model.T,
        initialize={i: float(ev_power_envelope_mpc[i]) for i in range(n_periods)},
    )
    model.access_power_fixed = Param(model.M, initialize=access_power_fixed_by_month_idx)
    model.peak_so_far = Param(model.M, initialize=peak_so_far_by_month_idx)
    model.rolling12_exceedance_so_far = Param(model.M, initialize=rolling12_so_far_by_month_idx)

    model.ev_charge = Var(model.T, domain=NonNegativeReals)
    model.ev_charge_power = Var(model.T, domain=NonNegativeReals)
    model.hp_electrical_input = Var(model.T, domain=NonNegativeReals)
    model.hp_thermal_output = Var(model.T, domain=NonNegativeReals)
    model.hp_thermal_power = Var(model.T, domain=NonNegativeReals)
    model.buffer_soc = Var(model.T, domain=NonNegativeReals)
    model.buffer_energy = Var(model.T, domain=NonNegativeReals)
    model.grid_consumption = Var(model.T, domain=NonNegativeReals)
    model.grid_injection = Var(model.T, domain=NonNegativeReals)
    model.grid_power = Var(model.T, domain=Reals)
    model.monthly_peak = Var(model.M, domain=NonNegativeReals)
    model.effective_peak = Var(model.M, domain=NonNegativeReals)
    model.rolling12_max_exceedance = Var(model.M, domain=NonNegativeReals)
    model.delta_rolling12 = Var(model.M, domain=NonNegativeReals)
    model.ev_daily_shortfall = Var(model.D, domain=NonNegativeReals)
    model.soc_min_slack = Var(model.T, domain=NonNegativeReals)
    model.soc_max_slack = Var(model.T, domain=NonNegativeReals)

    EV_SHORTFALL_PENALTY_EUR_PER_KWH = 10000.0
    _pen = (
        soc_min_slack_penalty_eur_per_soc
        if soc_slack_penalty_eur_per_soc is None
        else soc_slack_penalty_eur_per_soc
    )
    soc_slack_penalty = float(max(0.0, float(_pen)))

    def _soc_floor_at(t: int) -> float:
        if soc_min_profile is not None:
            return float(soc_min_profile[t])
        return float(soc_min_phys)

    model.objective = Objective(
        expr=(
            sum(
                model.grid_consumption[t]
                * (
                    energy_rate_eur_per_mwh / 1000.0
                    + grid_losses_percentage * model.spot_price[t]
                )
                for t in model.T
            )
            + sum(model.grid_consumption[t] * model.spot_price[t] for t in model.T)
            + sum(model.effective_peak[m] * monthly_peak_price_eur_per_kw for m in model.M)
            + sum(
                12.0 * model.delta_rolling12[m] * over_usage_price_eur_per_kw
                for m in model.M
            )
            - sum(model.grid_injection[t] * model.net_injection_price[t] for t in model.T)
            + sum(
                model.ev_daily_shortfall[d] * EV_SHORTFALL_PENALTY_EUR_PER_KWH
                for d in model.D
            )
            + soc_slack_penalty
            * (
                sum(model.soc_min_slack[t] for t in model.T)
                + sum(model.soc_max_slack[t] for t in model.T)
            )
        ),
        sense=minimize,
    )

    model.ev_power_link = Constraint(
        model.T, rule=lambda m, t: m.ev_charge_power[t] == m.ev_charge[t] * 4.0
    )
    model.ev_envelope = Constraint(
        model.T, rule=lambda m, t: m.ev_charge_power[t] <= m.ev_power_envelope[t]
    )

    def ev_daily_energy(model, date_val):
        periods = date_to_periods.get(date_val, [])
        if not periods:
            return Constraint.Skip
        return (
            sum(model.ev_charge[t] for t in periods)
            + model.ev_daily_shortfall[date_val]
            == float(daily_ev_remaining[date_val])
        )

    model.ev_daily_energy = Constraint(model.D, rule=ev_daily_energy)

    model.hp_thermal_output_constraint = Constraint(
        model.T,
        rule=lambda m, t: m.hp_thermal_output[t] == m.hp_electrical_input[t] * m.cop[t],
    )
    model.hp_thermal_power_constraint = Constraint(
        model.T, rule=lambda m, t: m.hp_thermal_power[t] == m.hp_thermal_output[t] * 4.0
    )
    model.hp_thermal_power_limit = Constraint(
        model.T, rule=lambda m, t: m.hp_thermal_power[t] <= thermal_max_kw
    )
    model.buffer_energy_constraint = Constraint(
        model.T, rule=lambda m, t: m.buffer_energy[t] == m.buffer_soc[t] * buffer_capacity_kwh
    )
    model.soc_min_soft = Constraint(
        model.T,
        rule=lambda m, t: m.buffer_soc[t] + m.soc_min_slack[t] >= _soc_floor_at(int(t)),
    )
    model.soc_max_soft = Constraint(
        model.T, rule=lambda m, t: m.buffer_soc[t] <= float(soc_max) + m.soc_max_slack[t]
    )

    loss_rate_per_interval = loss_coefficient_per_hour / 4.0

    def buffer_state_update(model, t):
        if t == 0:
            net = (
                model.hp_thermal_output[t]
                - model.thermal_load[t]
                - model.buffer_energy[t] * loss_rate_per_interval
            )
            return model.buffer_soc[t] == float(soc_initial) + net / buffer_capacity_kwh
        net = (
            model.hp_thermal_output[t]
            - model.thermal_load[t]
            - model.buffer_energy[t - 1] * loss_rate_per_interval
        )
        return model.buffer_soc[t] == model.buffer_soc[t - 1] + net / buffer_capacity_kwh

    model.buffer_state_update = Constraint(model.T, rule=buffer_state_update)

    last_ts = pd.to_datetime(data[timestamp_col].iloc[-1], errors="coerce")
    end_2025_last_slot = pd.Timestamp("2026-01-01 00:00:00") - pd.Timedelta(minutes=15)
    enforce_terminal_soc = (n_periods < 96) or (
        not pd.isna(last_ts) and last_ts >= end_2025_last_slot
    )
    if enforce_terminal_soc:
        model.terminal_soc = Constraint(expr=model.buffer_soc[n_periods - 1] == soc_final)

    model.power_balance = Constraint(
        model.T,
        rule=lambda m, t: m.grid_consumption[t]
        - m.grid_injection[t]
        == m.inflex_load[t] + m.ev_charge[t] + m.hp_electrical_input[t] - m.pv[t],
    )
    model.grid_power_constraint = Constraint(
        model.T,
        rule=lambda m, t: m.grid_power[t]
        == (m.grid_consumption[t] - m.grid_injection[t]) * 4.0,
    )
    model.monthly_peak_constraint = Constraint(
        model.T,
        rule=lambda m, t: m.monthly_peak[period_to_month[t]] >= m.grid_consumption[t] * 4.0,
    )
    model.effective_peak_ge_plan = Constraint(
        model.M, rule=lambda m, mo: m.effective_peak[mo] >= m.monthly_peak[mo]
    )
    model.effective_peak_ge_sofar = Constraint(
        model.M, rule=lambda m, mo: m.effective_peak[mo] >= m.peak_so_far[mo]
    )
    model.rolling12_ge_so_far = Constraint(
        model.M,
        rule=lambda m, mo: m.rolling12_max_exceedance[mo] >= m.rolling12_exceedance_so_far[mo],
    )
    model.rolling12_ge_exceedance = Constraint(
        model.M,
        rule=lambda m, mo: m.rolling12_max_exceedance[mo]
        >= m.effective_peak[mo] - m.access_power_fixed[mo],
    )
    model.delta_rolling12_def = Constraint(
        model.M,
        rule=lambda m, mo: m.delta_rolling12[mo]
        >= m.rolling12_max_exceedance[mo] - m.rolling12_exceedance_so_far[mo],
    )

    solve_highs_model(model, tee=False)

    monthly_peak_plan_by_period = [
        float(value(model.monthly_peak[period_to_month[t]])) for t in range(n_periods)
    ]
    effective_peak_by_period = [
        float(value(model.effective_peak[period_to_month[t]])) for t in range(n_periods)
    ]
    rolling12_max_by_period = [
        float(value(model.rolling12_max_exceedance[period_to_month[t]]))
        for t in range(n_periods)
    ]
    rolling12_increment_by_period = [
        float(value(model.delta_rolling12[period_to_month[t]])) for t in range(n_periods)
    ]

    results_df = pd.DataFrame(
        {
            timestamp_col: data[timestamp_col].values,
            "pv_production": pv_production,
            "inflex_load": inflex_load,
            "spot_price_eur_per_mwh": spot_price,
            "ev_req": ev_req,
            "ev_power_envelope_base": ev_power_envelope_combined,
            "ev_power_envelope_mpc": ev_power_envelope_mpc,
            "thermal_load": thermal_load,
            "outdoor_temperature": outdoor_temp,
            "cop": cop_array,
            "ev_charge": [float(value(model.ev_charge[t])) for t in model.T],
            "ev_charge_power": [float(value(model.ev_charge_power[t])) for t in model.T],
            "hp_electrical_input": [float(value(model.hp_electrical_input[t])) for t in model.T],
            "hp_thermal_output": [float(value(model.hp_thermal_output[t])) for t in model.T],
            "buffer_soc": [float(value(model.buffer_soc[t])) for t in model.T],
            "grid_consumption": [float(value(model.grid_consumption[t])) for t in model.T],
            "grid_injection": [float(value(model.grid_injection[t])) for t in model.T],
            "grid_power": [float(value(model.grid_power[t])) for t in model.T],
            "monthly_peak_plan": monthly_peak_plan_by_period,
            "effective_peak": effective_peak_by_period,
            "access_power_fixed": [
                float(access_power_fixed_by_month_idx[period_to_month[t]])
                for t in range(n_periods)
            ],
            "rolling12_max_exceedance_kw": rolling12_max_by_period,
            "rolling12_increment_kw": rolling12_increment_by_period,
        }
    )
    results_df[timestamp_col] = pd.to_datetime(results_df[timestamp_col], errors="coerce")
    results_df["month"] = results_df[timestamp_col].dt.to_period("M")

    summary = {
        "objective_value": float(value(model.objective)),
        "months_in_window": [str(m) for m in months],
        "monthly_peak_plan": {
            str(months[i]): float(value(model.monthly_peak[i])) for i in range(len(months))
        },
        "effective_peak": {
            str(months[i]): float(value(model.effective_peak[i])) for i in range(len(months))
        },
        "rolling12_max_exceedance_kw": {
            str(months[i]): float(value(model.rolling12_max_exceedance[i]))
            for i in range(len(months))
        },
        "rolling12_increment_kw": {
            str(months[i]): float(value(model.delta_rolling12[i])) for i in range(len(months))
        },
    }
    return results_df, summary


