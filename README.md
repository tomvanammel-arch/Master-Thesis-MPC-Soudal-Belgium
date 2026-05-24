# Master Thesis - MPC @ Soudal (EV + Heat Pump)

This repository contains the code and notebooks used to **simulate and optimize industrial electricity consumption** under the **Belgian capacity tariff**. The core use-case is a site with:

- **Inflexible electrical load** (plant baseline consumption)
- **PV production**
- **Uncontrolled EV charging demand** (treated as flexible in deterministic MPC, forecast-driven in online MPC)
- **Thermal load** served by a **heat pump + hot-water buffer** (flexible via buffer SOC)

The work is organised around two main experiment families:

- **Deterministic MPC (perfect foresight for 2025)**: optimise a full year in one optimisation problem (EV-only, HP-only, EV+HP).
- **Online MPC (rolling horizon with forecasts)**: simulate a year step-by-step using **24h rolling-horizon MPC** fed by forecasts, plus "real-time" post-processing (clipping / safeguards).

---

## Quick start

### Environment

The repo is Python-based (notebooks + `src/` modules).

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Run notebooks

Start JupyterLab and run notebooks in numerical order:

```bash
jupyter lab
```

The notebooks write intermediate and final artifacts to `output/` (see [Outputs](#outputs)).

---

## Repository structure

### `config/` (simulation + billing parameters)

- `config/billing.yaml`
  - **Belgian capacity tariff** parameters used throughout:
    - energy-based adder components (€/MWh) and grid loss factor
    - peak-based tariff components (€/kW/month): access power, monthly peak, over-usage
    - injection imbalance cost for feed-in revenue calculation
- `config/hp.yaml`
  - **Heat pump + buffer model** parameters:
    - COP curve vs outdoor temperature (datasheet points, interpolated)
    - thermal capacity limit
    - buffer size and thermodynamics (density, \(c_p\), usable \(\Delta T\))
    - SOC bounds and SOC targets for MPC
    - standing losses per hour
    - optional economics (buffer CAPEX / lifetime, if used in analysis)

### `data/` (raw inputs)

The project is built around **15-minute time series** (kWh per 15-min interval; prices in €/MWh).

- `data/plant1.csv`
  - primary dataset used by deterministic and online experiments
  - contains at least: `timestamp`, `price`, `pv_production`, `inflex_load`, `ev`, `thermal_load`, `outdoor_temperature`, plus derived grid columns in some runs
- `data/plant1_2024_training.csv`
  - training year used by forecasting notebooks (to predict 2025)
- `data/elia_forecast_pv_antwerp_DAH_11AM.csv`
  - day-ahead PV forecast from Elia (used as baseline PV forecast shape)
- `data/temperature_forecast_day_ahead_open_meteo_Turnhout_15min.csv`
  - day-ahead outdoor temperature forecast on the 15-min grid (used as exogenous input for thermal-load forecasting and HP planning COP)

### `src/` (reusable building blocks)

This folder contains the "library" used by the notebooks.

- `src/billing.py`
  - implements **monthly shadow billing** for:
    - offtake (grid consumption): energy + spot + peak-based components
    - injection (feed-in): spot-based revenue minus imbalance component
  - key outputs are monthly tables used for comparisons in notebooks
- `src/heat_pump_load.py`
  - HP physics helper:
    - loads `config/hp.yaml`
    - interpolates COP(\(T_{out}\))
    - converts thermal demand (kWh/15min) to electrical HP input (kWh/15min) for the **uncontrolled baseline**
  - writes `output/uncontrolled_hp.csv`
- `src/optimization.py`
  - the optimisation core (Pyomo-based):
    - `deterministic_mpc_ev`: full-year deterministic EV optimisation
    - `deterministic_mpc_hp`: full-year deterministic HP + buffer optimisation
    - `deterministic_mpc_ev_hp`: full-year combined EV + HP optimisation
    - `mpc_ev_24h`: **24h myopic EV MPC** used by online simulations (96×15min window; access priced via rolling-12 exceedance like `mpc_hp_24h`, not a hard per-step grid cap)
    - `mpc_hp_24h`: **24h myopic HP MPC** used by online simulations (96×15min window)
    - `mpc_ev_hp_24h`: **24h myopic joint EV+HP MPC** (notebook 11 / thesis §3.7.4)
    - `deterministic_access_power_hp_min_access`: access-power planning variant (HP) that minimises contracted access kW
- `src/online_MPC_1_EV.py`
  - full-year **rolling-horizon online simulation** for EV-only
  - uses `mpc_ev_24h` each step and applies:
    - a planner region (MPC)
    - a catch-up region near the deadline
    - real-time grid clipping policies (**contractual access enforced here**, not as a hard inequality inside `mpc_ev_24h`)
  - tracks rolling-12 exceedance state (deque of completed months + current-month exceedance) to parameterise each `mpc_ev_24h` solve
  - computes monthly bills from the simulated grid power
- `src/online_MPC_1_HP.py`
  - full-year **rolling-horizon online simulation** for HP-only
  - uses `mpc_hp_24h` each step and applies:
    - grid clipping (planner vs realised peak)
    - SOC feedback with losses using **actual** thermal load and actual COP
    - optional "forecast stress SOC floor" (raise SOC-min during forecasted access-power stress)
    - optional PLC-like safeguard energy to enforce physical SOC-min
- `src/online_MPC_1_EV_slack_sensitivity.py`
  - runs EV online MPC for multiple deadline slack settings and exports combined results + a summary table
- `src/online_MPC_1_HP_scenario_analysis.py`
  - runs HP online MPC over a set of scenarios (forecast choices, SOC rules, penalty weights, etc.)
  - exports per-scenario 15-min results + JSON summaries + a master summary CSV
- `src/online_MPC_1_EV_HP.py`
  - full-year **joint** EV+HP online simulation (thesis §3.7.4)
  - uses `mpc_ev_hp_24h` + joint shared proportional clipping (EV/HP flex power), EV catch-up, HP PLC
- `src/online_MPC_1_EV_HP_scenario_analysis.py`
  - batch driver for notebook 11 Part 4 (inflex stress × SOC floor grid, 17 scenarios)
- `src/buffer_size_sensitivity.py`
  - deterministic HP optimisation rerun for multiple buffer sizes and exports `output/optimised_ts/deterministic_hp_<Xm3>.csv`
- `src/notebook_visualisation/`
  - small helper(s) extracted from Notebook 10 so Part 4 scenario viewing can be reused as Python code

### `notebooks/` (main analysis workflow)

The notebooks are the "story" of the thesis work: validation -> deterministic MPC -> forecasting -> online MPC.

See [Notebook guide](#notebook-guide) for an extensive per-notebook description.

### `output/` (generated artifacts)

This repository writes results into structured subfolders:

- `output/forecast/`
  - **rolling-horizon forecasts** on the 2025 15-min grid
  - typical files:
    - `forecast_ev_rolling_horizon.csv`
    - `forecast_inflex_load_rolling_horizon.csv`
    - `forecast_pv_rolling_horizon.csv`
    - `forecast_thermal_load_rolling_horizon.csv`
- `output/optimised_ts/`
  - "clean" time series outputs (mainly deterministic schedules), e.g.
    - `deterministic_ev.csv` (timestamp, `ev_deterministic`)
    - `deterministic_hp.csv` (timestamp, `hp_deterministic`)
    - `deterministic_ev_hp.csv` (timestamp, `ev_deterministic`, `hp_deterministic`)
    - `deterministic_hp_<Xm3>.csv` (buffer sensitivity)
- `output/notebooks/`
  - notebook-specific exports (usually richer and/or debug-friendly), e.g.
    - deterministic monthly billing tables for plots and tables
    - online 15-min simulation results
    - online MPC debug logs (can be very large)
- `output/uncontrolled_hp.csv`
  - baseline electrical HP load derived from thermal demand + COP

---

## Core modelling conventions

### Time base and units

- **Time resolution**: 15 minutes (96 steps per day, 35040 steps per year for 2025)
- **Energy columns**: kWh per 15-min interval
- **Power columns**: kW (commonly computed as `kWh * 4`)
- **Spot price**: €/MWh
- **Billing aggregation**: monthly (Belgian local time)

### Meter vs planning time alignment (DST note)

Several parts of the project intentionally avoid timestamp joins when comparing series (especially online simulations vs plant series), because DST/timezone offsets can create subtle misalignment. In those cases, sequences are treated as **row-aligned** (same 15-min grid length) rather than merged on timestamps.

---

## Optimization problems (objective, constraints, solver)

This section documents the optimisation problems implemented in `src/optimization.py`. The code uses **Pyomo** and solves with **HiGHS** via `SolverFactory("highs")` (and, in some places, falls back to other MILP solvers if HiGHS is unavailable).

All problems use the same modelling conventions:

- Indices:
  - \(t \in \mathcal{T}\): 15-min steps
  - \(m \in \mathcal{M}\): months (2025-01 ... 2025-12)
- Conversions:
  - \(P[\mathrm{kW}] = 4 \cdot E[\mathrm{kWh/15min}]\)
- Exogenous time series:
  - \(E^{pv}_t\) PV production (kWh/15min)
  - \(E^{inflex}_t\) inflex load (kWh/15min)
  - \(E^{ev,base}_t\) uncontrolled EV demand / baseline (kWh/15min)
  - \(E^{th}_t\) thermal demand (kWh_th/15min)
  - \(\pi_t\) spot price (EUR/MWh)
- Tariff parameters from `config/billing.yaml` (see `src/billing.py` for the same billing decomposition used in reporting):
  - energy adders (EUR/MWh) and grid-loss factor
  - peak prices (EUR/kW/month): access power, monthly peak, over-usage
  - injection imbalance cost (EUR/MWh)

### A) Deterministic full-year EV MPC (`deterministic_mpc_ev`)

**Decision variables**

- EV charging energy \(E^{ev}_t \ge 0\) (kWh/15min)
- Grid exchange:
  - offtake \(E^{grid+}_t \ge 0\), injection \(E^{grid-}_t \ge 0\) (kWh/15min)
  - grid power \(P^{grid}_t = 4(E^{grid+}_t - E^{grid-}_t)\) (kW)
- Monthly peak / capacity-tariff state:
  - monthly peak \(P^{peak}_m\) (kW)
  - access power \(P^{access}_m\) (kW)
  - exceedance \(P^{exc}_m = \max(0, P^{peak}_m - P^{access}_m)\) (kW)
  - rolling 12-month max exceedance \(P^{roll12}_m\) (kW)
  - access-power increase detection/lock-in auxiliaries (binary and continuous) used to model the tariff rules

**Objective**

Minimise a full-year proxy of the Belgian capacity-tariff bill:

- **Spot cost**:
  - \(\sum_t E^{grid+}_t \cdot \pi_t / 1000\)
- **Energy-based adders** (fixed EUR/MWh + grid-loss factor):
  - \(\sum_m \big( V_m[\mathrm{MWh}] \cdot (\text{fixed\_adder} + \text{grid\_loss\_pct} \cdot \overline{\pi}_m) \big)\)
  - where \(V_m = \sum_{t \in m} E^{grid+}_t / 1000\) and \(\overline{\pi}_m\) is the volume-weighted average spot price in the month
- **Peak-based (capacity-tariff) costs**:
  - access cost: \(\sum_m P^{access}_m \cdot c^{access}\)
  - monthly peak cost: \(\sum_m P^{peak}_m \cdot c^{peak}\)
  - over-usage cost: \(\sum_m P^{roll12}_m \cdot c^{over}\)
- **Injection revenue**:
  - subtract \(\sum_t E^{grid-}_t \cdot (\pi_t - c^{inj\_imb})/1000\)

The implemented objective in code is the algebraic combination of these terms.

**Constraints**

- **Power/energy balance (net grid energy)**:
  - \(E^{grid+}_t - E^{grid-}_t = E^{inflex}_t + E^{ev}_t - E^{pv}_t\)
  - (split into nonnegative offtake/injection via variables)
- **EV feasibility**:
  - nonnegativity and an EV power envelope constraint (kW envelope encoded as kWh/15min)
  - a daily energy constraint ensuring total EV energy delivered per day matches the required daily amount (derived from the baseline EV series)
- **Monthly peak definition**:
  - \(P^{peak}_m \ge P^{grid}_t\) for all \(t\) in month \(m\)
- **Exceedance and rolling max exceedance**:
  - \(P^{exc}_m \ge P^{peak}_m - P^{access}_m\), \(P^{exc}_m \ge 0\)
  - \(P^{roll12}_m \ge P^{exc}_j\) for relevant months \(j\) in the 12-month lookback window
- **Access power tariff rules (lock-in / increases)**:
  - a set of constraints with binary auxiliaries to detect increases and enforce that access power cannot freely decrease once increased (the capacity-tariff "ratchet" logic)

**Solver loop**

- Primary solver: HiGHS (`SolverFactory("highs")`)
- If HiGHS fails, the code attempts other solvers (when available) to still produce a result.

Outputs are exported as:

- `output/optimised_ts/deterministic_ev.csv` (timestamp + optimised EV schedule)

---

### B) Deterministic full-year HP + buffer MPC (`deterministic_mpc_hp`)

**Decision variables**

- HP electrical input \(E^{hp}_t \ge 0\) (kWh/15min)
- Buffer state of charge \(SOC_t\) (fraction)
- Grid offtake/injection and grid power as in EV problem
- Monthly peak/access/exceedance/rolling-12 state as in EV problem

**Thermal model pieces**

- COP is computed from a datasheet curve (see `config/hp.yaml`) and interpolated:
  - \(COP_t = COP(T^{out}_t)\)
- HP thermal output:
  - \(E^{hp,th}_t = E^{hp}_t \cdot COP_t\) (kWh_th/15min)
- Buffer capacity (kWh_th) computed from physical parameters in `hp.yaml`:
  - \(C^{buf} = V \rho c_p \Delta T / 3600\)
- Buffer dynamics with standing losses:
  - \(SOC_{t+1} = SOC_t + (E^{hp,th}_t - E^{th}_t - Loss_t)/C^{buf}\)
  - \(Loss_t = SOC_t \cdot C^{buf} \cdot (k_{loss}/4)\) (loss coefficient given per hour)
- Terminal SOC constraint:
  - \(SOC_{T_{end}} = SOC^{final}\) (typically equal to the initial SOC for cyclic fairness)

**Objective**

Same billing objective structure as EV deterministic:

- spot + energy adders + peak-based capacity tariff
- minus injection revenue

with the difference that EV is treated as **uncontrollable** (baseline series) and HP is the flexible lever through \(E^{hp}_t\) and \(SOC_t\).

**Constraints**

- all grid/billing constraints as in EV deterministic
- HP thermal power limit:
  - \(4 E^{hp,th}_t \le P^{th,max}\) (kW_th), from `hp.yaml`
- SOC bounds:
  - \(SOC^{min} \le SOC_t \le SOC^{max}\)
- SOC dynamics with losses + terminal SOC

Outputs are exported as:

- `output/optimised_ts/deterministic_hp.csv` (timestamp + optimised HP schedule)

---

### C) Deterministic full-year combined EV + HP (`deterministic_mpc_ev_hp`)

This is the joint problem of A) and B) in a single optimisation:

**Decision variables**

- \(E^{ev}_t\), \(E^{hp}_t\), \(SOC_t\), grid offtake/injection
- monthly peak/access/exceedance/rolling-12 state and lock-in auxiliaries

**Objective**

- same full-year capacity-tariff proxy as A/B (spot + energy adders + peak-based) minus injection revenue

**Constraints**

- combines:
  - EV envelope + daily EV energy constraints
  - HP COP + thermal output + SOC dynamics + terminal SOC
  - shared grid balance and billing state constraints

Outputs are exported as:

- `output/optimised_ts/deterministic_ev_hp.csv` (timestamp + optimised EV and HP schedules)

---

### D) 24h EV MPC used online (`mpc_ev_24h`)

This is the myopic controller solved repeatedly in the online EV simulation (`src/online_MPC_1_EV.py`).

**Window**

- 96 timesteps (24 hours), each 15 minutes

**Decision variables**

- EV charging \(E^{ev}_t\) over the window
- grid offtake/injection \(E^{grid+}_t, E^{grid-}_t\)
- monthly peak plan variable(s) for any month(s) that appear in the 24h window
- rolling-12 exceedance increment variable(s) (same structure as `mpc_hp_24h`)

**Objective (window level)**

- spot + energy adders + monthly peak price
- plus a priced increase of rolling-12 max exceedance relative to the locked-in exceedance state passed in by the online simulator (same `12 × delta × over_usage_price` structure as `mpc_hp_24h`, read from `billing.yaml`)
- minus injection revenue
- (no full-year access *decision* here; contracted access enters as fixed parameters and through exceedance pricing)

**Constraints**

- grid balance (inflex + EV - PV)
- EV envelope, including "deadline slack":
  - during the last `ev_deadline_slack_minutes` before 17:00, the envelope is forced to 0 kW so charging is pushed into catch-up logic (see below)
- per-day EV energy constraints:
  - ensures the controller delivers the required EV energy per day within the window (for the current day this uses a dynamic "remaining energy" input provided by the online simulator)
- monthly peak and effective-peak constraints (planner peak vs `peak_so_far` from realised simulation)
- rolling-12 exceedance constraints vs `access_power_by_month` and `rolling12_max_exceedance_so_far_by_month` (passed in explicitly; not taken from `billing.yaml` for access kW)
- no hard per-step grid cap at access power inside the planner; contractual limits are enforced in `online_MPC_1_EV.py` clipping / catch-up

---

### E) 24h HP MPC used online (`mpc_hp_24h`)

This is the myopic controller solved repeatedly in the online HP simulation (`src/online_MPC_1_HP.py`).

**Window**

- 96 timesteps (24 hours), each 15 minutes

**Decision variables**

- HP electrical input \(E^{hp}_t\) over the window
- buffer SOC \(SOC_t\) over the window
- grid offtake/injection and grid power
- monthly peak plan variables for any month(s) in the 24h window
- rolling-12 exceedance increment variable(s) used to price exceedance against the already locked-in rolling-12 state

**Objective (window level)**

- spot + energy adders + monthly peak cost
- plus a priced increase of rolling-12 max exceedance relative to the current locked-in exceedance state
- minus injection revenue

**Constraints**

- HP COP and thermal output, thermal max
- SOC dynamics with losses, and terminal SOC target
- SOC lower bound can be optionally raised via:
  - `buffer_soc_min` (scalar) or
  - `buffer_soc_min_profile` (time-varying floor; preferred for forecast-stress SOC floor)
- grid balance, monthly peak and effective peak constraints
- access power is *not* a hard constraint inside the HP planner; exceedance is discouraged through the rolling-12 increment cost term

---

## Online MPC "execution logic" (scripts and non-obvious mechanics)

The online notebooks simulate a controller that solves a 24h MPC every 15 minutes, but then applies additional real-time logic. This logic is in `src/online_MPC_1_EV.py` and `src/online_MPC_1_HP.py` and is essential for interpreting results.

### EV online MPC 1 - catch-up logic (`src/online_MPC_1_EV.py`)

The 24h EV planner (`mpc_ev_24h`) treats **access power** like `mpc_hp_24h` at the optimisation level: exceedance is priced through rolling-12 terms, while **this script** still clips realised charging so the applied step respects access (and peak-protection rules). The EV online simulation splits each weekday into regions (Belgian local time):

- **MPC region**: from 07:00 until a ramp end time (`mpc_ramp_end`)
- **Catch-up region**: from `mpc_ramp_end` until 17:00
- Outside these windows (night/weekend): no EV charging by the controller

`mpc_ramp_end` is shifted earlier when using `ev_deadline_slack_minutes`. The key idea:

- The MPC problem is intentionally made more conservative near the end of the day (by forcing envelope to zero during the slack period), and the remaining energy is then handled by a dedicated catch-up routine.

At each step \(k\) in the catch-up region, the script:

1. Computes **remaining daily EV energy**:
   - `remaining = ev_to_deliver` (kWh) where `ev_to_deliver = blended_daily_demand - charged_so_far`
2. Computes an "ideal" constant power to finish charging immediately:
   - \(P^{ev,ideal} = remaining / 0.25\) (kW)
3. Caps by a **physical EV power envelope** derived from car presence patterns:
   - \(P^{ev,set} = \min(P^{ev,ideal}, P^{ev,env})\)
4. Computes planned grid power:
   - \(P^{grid,plan} = 4(E^{inflex}_k + P^{ev,set}/4 - E^{pv}_k)\)
5. Applies a catch-up clipping policy designed to avoid creating a new monthly peak:
   - \(P^{limit} = \min(P^{access}(m), P^{peak\_so\_far}(m))\)
   - if \(P^{grid,plan} > P^{limit}\), reduce EV power to fit:
     - \(P^{ev,new} = \max(P^{ev,set} - (P^{grid,plan} - P^{limit}), 0)\)

This is why in online EV results you can see:

- a clean planner behaviour earlier in the day (MPC),
- followed by late-day "finish what remains" behaviour,
- strongly shaped by access power and the already-observed monthly peak so far.

### HP online MPC 1 - forecast-stress SOC floor ("raising the buffer") (`src/online_MPC_1_HP.py`)

The HP online simulation can pre-empt access-power stress by raising the SOC minimum seen by the 24h planner.

**Step 1: detect forecast stress**

Before simulation, the script constructs a full-year forecasted grid power series:

- uses forecasts for inflex, EV (uncontrollable in HP-only mode), PV, thermal load
- converts thermal forecast to an estimated HP electrical load using COP computed from a temperature forecast
- forms:
  - \(P^{grid,fc}_t = 4(E^{inflex,fc}_t + E^{ev,fc}_t + E^{hp,est}_t - E^{pv,fc}_t)\)

Then flags "stress" where forecast grid exceeds monthly access power:

- `stress_active[t] = (P_grid_fc[t] > access_kw[t])`

**Step 2: build a time-varying SOC-min schedule**

If enabled, the planner SOC floor is raised during stress periods:

- physical floor: \(SOC^{min}_{phys}\)
- target raised floor:
  - \(SOC^{min}_{target} = SOC^{min}_{phys} + strength \cdot (SOC^{max}_{phys} - SOC^{min}_{phys})\)
- the schedule is:
  - \(SOC^{min}_{sched}[t] = SOC^{min}_{target}\) if stress_active[t] else \(SOC^{min}_{phys}\)

To make the transition less abrupt, the script also ramps the floor in over ~3 hours before a stress period (linear ramp on the 15-min grid).

**Step 3: pass the floor into each 24h MPC solve**

When solving `mpc_hp_24h` at step \(k\), the script passes:

- `buffer_soc_min_profile = soc_min_schedule_full[k:k_end]`

Inside the planner, the SOC lower bound becomes:

- \(SOC_t \ge \max(SOC^{min}_{phys}, SOC^{min}_{profile}[t])\)

This is the mechanism that "raises the buffer SOC" ahead of expected access-power stress.

### HP online MPC 1 - real-time safeguards (access-aware actuator + PLC energy)

After the planner produces a first-step HP plan, the online HP script applies real-time logic:

- **Grid clipping**: compares the planned grid power against a clip limit based on realised peaks and access power, and reduces HP if needed.
- **Access-aware actuator**: additionally caps HP to avoid exceeding access power when possible, but does not violate the physical SOC-min requirement.
- **PLC-like safeguard energy (optional)**:
  - if, after applying the command and updating SOC with actual thermal demand and actual COP, the SOC would fall below the physical minimum, the script injects extra HP energy to bring SOC back to \(SOC^{min}_{phys}\).
  - alternatively, if SOC-min enforcement is disabled, the script accounts this as **unmet thermal demand** instead of extra HP energy.

These safeguards are why online HP runs can remain feasible even when the 24h planner cannot perfectly anticipate realised disturbances.

---

## Notebook guide

Below is a detailed "what and why" description for each notebook, focusing on the **simulation setup**, the **main ideas**, and the **structure**.

### `notebooks/01_benchmark.ipynb` — Benchmark analysis / data validation

**Purpose**

- Establish trust in the raw input dataset (`data/plant1.csv`) before optimisation.
- Provide baseline plots and billing logic checks.

**Key ideas**

- Sanity-check time series consistency (missing values, magnitude ranges, seasonal patterns).
- Visualise and summarise:
  - thermal load vs outdoor temperature (physical plausibility)
  - PV production patterns
  - EV charging behaviour
  - baseline grid consumption/injection behaviour
- Explain the **billing formulas** that later notebooks optimise against.

**Structure**

- Thermal load and outdoor temperature analysis (quality checks + plots + monthly summaries)
- Billing formulas and parameters (energy/spot/peak components + injection revenue)
- EV data visualisation
- Baseline impact of uncontrolled EV charging

**Outputs**

- Mostly diagnostic figures; no critical CSV artifacts required by later notebooks.

---

### `notebooks/02_EV_deterministic_MPC.ipynb` — EV charging optimisation (deterministic MPC)

**Purpose**

- Demonstrate full-year **deterministic** MPC for EV charging: optimise EV charging schedule with perfect foresight of PV, inflex load, and prices.

**Key ideas**

- The EV is treated as *flexible* (within an envelope / feasibility constraints).
- Objective combines:
  - spot energy shifting (charge more when price is low / PV is high)
  - peak reduction to reduce monthly peak and (implicitly) access power pressure
  - injection revenue effects via net export
- Compare **baseline vs optimised** costs and peaks.

**Structure**

- Overview and data loading
- Baseline cost calculation (unoptimised)
- Optimisation model description (objective + constraints, pulling key parameters from `config/billing.yaml`)
- Comparison: costs, monthly peak power, monthly volumes
- Visualisations: weekly and daily comparisons
- Takeaways and cost summary

**Outputs**

- Deterministic EV schedule in `output/optimised_ts/deterministic_ev.csv`
- Notebook-level exports in `output/notebooks/` (e.g. monthly cost tables used by later analysis)

---

### `notebooks/03_HP_deterministic_MPC.ipynb` — Heat pump optimisation (deterministic MPC)

**Purpose**

- Optimise heat pump operation over a full year with perfect foresight, using a buffer SOC model.

**Key ideas**

- The heat pump converts electricity to thermal energy via COP(\(T_{out}\)).
- The buffer (hot-water storage) provides **time-shifting** of thermal production:
  - charge buffer during cheap/low-peak moments
  - discharge when expensive/high-peak moments
- Baselines:
  - plant baseline (no HP)
  - "baseline + uncontrolled HP" using `src/heat_pump_load.py` (thermal load served immediately)
- Deterministic MPC then coordinates HP and buffer to reduce cost while meeting thermal demand and SOC constraints.
- Includes a **buffer size sensitivity** section.

**Structure**

- Overview and data loading
- Baseline costs (no HP)
- Compute uncontrolled HP electrical load from thermal demand
- Baseline + HP costs comparison
- Optimisation model (objective + constraints; parameters from `config/hp.yaml` and `config/billing.yaml`)
- Optimised vs baseline HP comparison
- Visualisations: peaks, weekly/daily HP operation, SOC behaviour
- Buffer size sensitivity analysis

**Outputs**

- Deterministic HP schedule in `output/optimised_ts/deterministic_hp.csv`
- Uncontrolled HP baseline in `output/uncontrolled_hp.csv`
- Sensitivity schedules in `output/optimised_ts/deterministic_hp_<Xm3>.csv`
- Notebook-level exports (deterministic billing tables and 15-min outputs) in `output/notebooks/`

---

### `notebooks/04_EV+HP_deterministic_MPC.ipynb` — Combined EV + HP optimisation (deterministic MPC)

**Purpose**

- Solve a single full-year optimisation that **co-optimises EV charging and HP/buffer**.

**Key ideas**

- Joint optimisation can:
  - reduce conflicts for peak power
  - use PV surplus more efficiently
  - coordinate flexible electrical (EV) and flexible thermal (buffered HP) demands
- Provides comparisons against multiple baselines:
  - uncontrolled
  - EV-only optimised
  - HP-only optimised
  - combined optimised

**Structure**

- Overview and data loading
- Baseline costs (uncontrolled EV + uncontrolled HP)
- Optimisation model (objective + constraints, key config parameters)
- Comparison: costs, monthly peaks, monthly volumes
- Monthly savings tables and plots
- "Four-case" comparison section to synthesise results across scenarios

**Outputs**

- Combined deterministic schedules in `output/optimised_ts/deterministic_ev_hp.csv`
- Notebook-level exports in `output/notebooks/` for figures/tables

---

### `notebooks/05_forecasting_EV.ipynb` — EV forecasting + envelope construction

**Purpose**

- Build 15-min forecasts for uncontrolled EV demand and derive an **operational envelope** used by online MPC.

**Key ideas**

- Train/fit on 2024 and forecast 2025 on the same 15-min grid.
- Compare forecasting strategies:
  - **Strategy B**: average of last \(K\) weeks (same time-of-week)
  - **Strategy C**: Chronos2 foundation model forecast
- Evaluate error metrics and visualise forecast vs actual.
- Construct an envelope that turns the EV demand forecast into a feasible power cap over the workday (used later by `mpc_ev_24h` / online logic).

**Structure**

- Part 1: data preparation + EV energy overview
- Part 2: forecasting strategies (incl. Chronos)
- Part 3: evaluation, \(K\) selection, envelope visualisation, and forecast plots

**Outputs**

- Rolling-horizon EV forecast CSV in `output/forecast/forecast_ev_rolling_horizon.csv`
  - contains one or more strategy columns like `forecast_ev_<strategy>`

---

### `notebooks/06_forecasting_inflex_load.ipynb` — Inflex load forecasting

**Purpose**

- Produce 15-min inflexible load forecasts for 2025 used by online MPC.

**Key ideas**

- Similar workflow to EV forecasting:
  - data sanity checks (multi-year view, monthly totals)
  - evaluate forecasting strategies (Chronos2 and rolling-week averages)
  - export rolling-horizon forecasts for MPC

**Structure**

- Part 1: data preparation + sanity checks
- Part 2: forecasting strategies (Chronos2; average of last \(K\) weeks)
- Part 3: error metrics and visual comparison

**Outputs**

- Rolling-horizon inflex forecast CSV in `output/forecast/forecast_inflex_load_rolling_horizon.csv`
  - contains strategy columns like `forecast_inflex_<strategy>`

---

### `notebooks/07_forecasting_PV.ipynb` — PV forecasting (Elia + rolling bias scaling)

**Purpose**

- Produce 15-min PV forecasts for 2025 used by online MPC.

**Key ideas**

- Use Elia day-ahead PV forecast as a **shape** forecast.
- Apply rolling bias scaling (using only past error) to adjust magnitude without look-ahead.
- Optionally benchmark Chronos2-based PV forecasts.

**Structure**

- Part 1: data preparation
- Part 2: baseline Elia-based forecast (shape)
- Part 3: rolling \(K\)-week bias scaling
- Part 4: error metrics and sanity checks
- Part 5: export rolling PV forecast for MPC

**Outputs**

- Rolling-horizon PV forecast CSV in `output/forecast/forecast_pv_rolling_horizon.csv`
  - typically uses columns like `pv_forecast_kWh_15min_<strategy>`

---

### `notebooks/08_forecasting_thermal_load.ipynb` — Thermal load forecasting (temperature DAH + rolling Ridge)

**Purpose**

- Forecast 15-min thermal load for 2025 (used by HP online MPC planner).

**Key ideas**

- Thermal load is strongly driven by outdoor temperature; use day-ahead temperature forecasts as exogenous input.
- Compare:
  - rolling Ridge regression with exogenous temperature (no future actuals)
  - Chronos-based benchmarks (univariate and with covariate)
- Export rolling-horizon thermal load forecasts used by the HP online MPC planner.

**Structure**

- Part 1: data preparation
- Part 2: actual vs day-ahead forecast temperature sanity check
- Part 3: rolling Ridge thermal load forecast (DAH temperature)
- Part 4+: Chronos benchmark(s)
- Comparison + export

**Outputs**

- Rolling-horizon thermal forecast CSV in `output/forecast/forecast_thermal_load_rolling_horizon.csv`
  - contains strategy columns like `forecast_thermal_<strategy>`

---

### `notebooks/09_online_MPC_1_EV.ipynb` — Online EV-only MPC 1 (access power + simulation + comparison)

**Purpose**

- Run a realistic **full-year online MPC simulation** where the controller only sees forecasts (plus limited "current" information) and makes decisions every 15 minutes.

**Key ideas**

- Separate **planning** from **execution**:
  - planner: solves `mpc_ev_24h` every step on a 24h forecast window (access enters via **rolling-12 exceedance cost**, aligned with `mpc_hp_24h`; forecast trajectories may exceed access if cheaper after over-usage pricing)
  - executor: applies clipping logic to enforce **contractual** access power / peak targets on the realised step, and uses catch-up logic near deadline
- Explicitly plans or sets **access power** per month, then evaluates monthly bills under the capacity tariff.
- Includes a **slack sensitivity** analysis for EV deadline flexibility.

**Structure**

- 1) Access power optimisation (forecast-based)
  - build reference grid consumption excluding EV
  - optimise a monthly access power plan that is "safe enough" for online simulation
- 2) Rolling-horizon online simulation (EV-only)
  - execute full-year simulation with forecasts from `output/forecast/`
- 3) Visualisation and comparison
  - energy volumes, costs, and shadow billing tables
- 4) Slack sensitivity
  - repeat the online run for multiple slack values and compare unmet energy vs savings

**Outputs**

- Online 15-min results in `output/notebooks/online_ev_15min_notebook_09.csv` (and per-slack variants)
- Detailed MPC window debug CSVs in `output/notebooks/online_ev_mpc_debug_notebook_09*.csv` (optional/large)
- Slack sensitivity summary in `output/notebooks/online_ev_slack_sensitivity_summary_notebook_09.csv`

---

### `notebooks/11_online_MPC_EV+HP.ipynb` — Joint online MPC (EV + HP)

**Purpose**

- Full-year online simulation with **one** planner controlling both EV and HP (thesis §3.7.4).
- Access power from joint full-day headroom (EV + HP reference energy).
- Part 4: scenario grid like notebook 10 (forecast-stress SOC floor), with EV slack fixed at 105 min.

**Outputs**

- `output/notebooks/online_ev_hp_15min_notebook_11_part2.csv`
- `output/optimised_ts/online_ev_hp.csv`
- Part 4: `output/notebooks/online_ev_hp_scenario_analysis_summary_notebook_11.csv`

See [notebooks/readme/readme_notebook11_online_MPC_EV_HP.md](notebooks/readme/readme_notebook11_online_MPC_EV_HP.md).

---

### `notebooks/10_online_MPC_1_HP.ipynb` — Online HP-only MPC 1 (access power + simulation + comparison + scenarios)

**Purpose**

- Run a full-year **online HP-only MPC** with buffer SOC feedback and compare to:
  - baseline (uncontrolled HP)
  - deterministic HP MPC (perfect foresight)
  - multiple online scenarios (forecast choices, penalties, safeguards)

**Key ideas**

- HP planner uses forecasts for:
  - inflex load, PV, EV (uncontrollable in this HP-only setup), thermal load, and temperature (for COP)
- Controller structure:
  - 24h MPC planner `mpc_hp_24h`
  - "executor" step applies grid clipping and SOC update with **actual** thermal load and COP
  - optional forecast-stress SOC floor to pre-charge buffer before forecasted access-power stress
  - optional PLC-like safety energy to respect physical SOC-min (or allow unmet thermal if disabled)
- Scenario analysis:
  - run a batch of scenarios and export a master summary CSV for comparison

**Structure**

- 1) Access power planning (heuristic)
  - derive a conservative month-by-month access power plan used in the online run
- 2) Rolling-horizon online simulation (HP-only)
  - execute full-year simulation using forecast CSVs from `output/forecast/`
- 3) Visualisation and comparison
  - compare deterministic vs baseline vs online costs/peaks
- 4) Scenario analysis
  - run multiple scenarios via `src/online_MPC_1_HP_scenario_analysis.py` and visualise results

**Outputs**

- Online HP 15-min results (main run) in `output/notebooks/online_hp_15min_notebook_10_part2.csv`
- Per-scenario 15-min results in `output/notebooks/online_hp_15min_notebook_10_scenario_*.csv`
- Per-scenario JSON summaries in `output/notebooks/online_hp_summary_notebook_10_scenario_*.json`
- Master scenario comparison in `output/notebooks/online_hp_scenario_analysis_summary_notebook_10.csv`

---

## Typical end-to-end workflow

If you want to reproduce the full pipeline in a clean environment, a practical sequence is:

1. Run `01_benchmark.ipynb` (data sanity and baseline understanding).
2. Run deterministic notebooks:
   - `02_EV_deterministic_MPC.ipynb`
   - `03_HP_deterministic_MPC.ipynb`
   - `04_EV+HP_deterministic_MPC.ipynb`
3. Run forecasting notebooks to generate `output/forecast/*.csv`:
   - `05_forecasting_EV.ipynb`
   - `06_forecasting_inflex_load.ipynb`
   - `07_forecasting_PV.ipynb`
   - `08_forecasting_thermal_load.ipynb`
4. Run online notebooks:
   - `09_online_MPC_1_EV.ipynb`
   - `10_online_MPC_1_HP.ipynb`
   - `11_online_MPC_EV+HP.ipynb`

---

## Notes and gotchas

- **Solvers**: optimisation is implemented in Pyomo (`src/optimization.py`) and needs a solver available in your environment. The repo includes `highspy` in `requirements.txt` (HiGHS), but your Pyomo solver backend still needs to be properly discoverable on your machine.
- **Outputs are stateful**: many notebooks read CSVs produced by earlier notebooks. If you delete `output/`, re-run notebooks in the recommended order.
- **Forecast strategies**: online MPC scripts expect column naming conventions such as:
  - `forecast_ev_<strategy>`
  - `forecast_inflex_<strategy>`
  - `forecast_thermal_<strategy>`
  - `pv_forecast_kWh_15min_<strategy>`

