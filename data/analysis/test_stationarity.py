import numpy as np
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import gc
import re
import json

def check_stationarity_adf_numpy(data):
    results = []
    for i, row in enumerate(data):
        # Perform the ADF test on each row (time series)
        adf_result = adfuller(row)
        
        # Store the results in a list
        results.append({
            'Time Series': i,  # Index of the time series
            'ADF Statistic': adf_result[0],
            'p-value': adf_result[1],
            'Critical Values': adf_result[4],
            'Is Stationary': adf_result[1] < 0.05  # True if p-value < 0.05
        })
        
        # Print the results for each time series
        print(f"Results for Time Series {i}:")
        print(f"  ADF Statistic: {adf_result[0]}")
        print(f"  p-value: {adf_result[1]}")
        print(f"  Critical Values: {adf_result[4]}")
        if adf_result[1] < 0.05:
            print(f"  The time series {i} is likely stationary.")
        else:
            print(f"  The time series {i} is likely non-stationary.")
        print("\n")
    
    return pd.DataFrame(results)  # Return the results as a DataFrame for easy viewing


name = "path/to/base/directory/"
name_2017 = "path/to/2017/directory/"
subregion = "p1uhs0_1247"

# Load the data without headers
last_year_data = pd.read_csv(name_2017 + "demand_imag.csv", sep=" ", header=None)
demand_type = 'reactive'

with open(name + 'visibility=M.json', 'r') as f:
    ami_data = json.load(f)

# Parse DSS file to extract bus and load information
dss_bus_coords_file = name + 'Buscoords.dss'
dss_loads_file = name + 'Loads.dss'

# Lists to hold data
observable_loads = []
observable_loads_names = []
observable_buses_names = []
unobservable_loads = []
unobservable_loads_names = []
unobservable_buses_names = []

# Step 2: Extract all loads and buses from DSS files
# Parsing DSS bus coordinates file
with open(dss_bus_coords_file, 'r') as f:
    all_buses = [line.split()[0] for line in f if line.strip()]

# Step 1: Extract observable loads and buses from JSON file
for bus_load_key, loads in ami_data.items():
    subreg = bus_load_key.split('->')[0]
    if subreg == subregion:
        bus_name = bus_load_key.split('->')[1]  # Only take the part before the arrow
        if bus_name in all_buses:
            observable_buses_names.append(bus_name)  # Buses with AMI
            observable_loads_names.extend(loads)  # Loads associated with those buses

del ami_data
gc.collect()


# Parsing DSS loads file
with open(dss_loads_file, 'r') as f:
    load_pattern = re.compile(r'New Load\.(\S+)')
    all_loads = []
    load_index = 0

    for line in f:
        match = load_pattern.search(line)
        if match:
            load_name = match.group(1)
            all_loads.append((load_index, load_name))
            load_index += 1

# Step 4: Identify observable and unobservable loads
load_phases = {}
for idx, load_name in all_loads:
    if len(load_name.split('_')) > 2:
        load_phases[idx] = int(load_name.split('_')[2])
    else:
        load_phases[idx] = 3
    # Extract the load prefix to match with JSON file (remove suffix after the first underscore)
    load_prefix = load_name.split('_')[0] + "_" + load_name.split('_')[1]
    if load_prefix in observable_loads_names:
        observable_loads.append(idx)
    else:
        unobservable_loads.append(idx)
        if len(load_name.split('_')) > 2:
            load_name = load_name.split('_')[1] + "_" + load_name.split('_')[2]
        else:
            load_name = load_name.split('_')[1]
        unobservable_loads_names.append(load_name)


# Step 4: Identify unobservable buses (those not listed in the observable buses)
unobservable_buses_names = [bus for bus in all_buses if bus not in observable_buses_names]

# Organize observable loads by phase
observable_loads_by_phase = {1: [], 2: [], 3: []}
for idx in observable_loads:
    phase = load_phases[idx]
    observable_loads_by_phase[phase].append(idx)


del observable_loads_names
del observable_buses_names
del unobservable_loads_names
del unobservable_buses_names
gc.collect()


if __name__ == "__main__":
    for phase, loads in observable_loads_by_phase.items():
            
        print('Phase: ', phase)
        observable_data_phase_df = last_year_data.iloc[loads].copy(deep=True)

        # Identify duplicate rows
        duplicates = observable_data_phase_df.duplicated(keep='first')

        # Loop through duplicates and add small random noise to make each duplicate unique
        for idx in observable_data_phase_df[duplicates].index:
            observable_data_phase_df.loc[idx] += np.random.normal(0, 1e-5, observable_data_phase_df.shape[1])

        # Convert back to numpy array
        observable_data_phase = observable_data_phase_df.values

        # Run the stationarity check
        stationarity_results = check_stationarity_adf_numpy(observable_data_phase)

        stationarity_results.to_csv(name_2017 + f'stationarity_results_{demand_type}_demand_phase_{phase}.csv')

        # Print the summary results in a table
        print(stationarity_results)