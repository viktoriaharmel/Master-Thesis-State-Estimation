import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import re
import gc
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA


name = "path/to/feeder/dir/"
name_2017 = "path/to/feeder/dir/last_year/"
subregion = "p1uhs0_1247"

# Load the data without headers
last_year_data = pd.read_csv(name_2017 + "demand_unc.csv", sep=" ", header=None)
demand_type = 'active'

# Load the JSON file containing observable loads (with AMI)
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

# Function to fit ARIMA model and print AIC
def fit_arima(series, p, d, q):
    model = ARIMA(series, order=(p, d, q))
    model_fit = model.fit()
    print(f'ARIMA({p},{d},{q}) - AIC: {model_fit.aic}')
    return model_fit

# Plot ACF and PACF for each time series
def plot_acf_pacf(data):
    num_series = data.shape[0]
    for i in range(1):
        series = data.iloc[i]
        n = len(series)
        
        plt.figure(figsize=(12, 6))
        
        # Plot ACF
        plt.subplot(1, 2, 1)
        plot_acf(series, ax=plt.gca(), lags=20, alpha=None)  # alpha=None to remove the default CI shading
        significance_level = 1.96 / np.sqrt(n)
        plt.axhline(y=significance_level, linestyle='--', color='red', label='5% Significance Level')
        plt.axhline(y=-significance_level, linestyle='--', color='red')
        plt.title(f'Time Series {i+1} - ACF')
        plt.legend()

        # Plot PACF
        plt.subplot(1, 2, 2)
        plot_pacf(series, ax=plt.gca(), lags=20, alpha=None, method='ywm')  # alpha=None to remove the default CI shading
        plt.axhline(y=significance_level, linestyle='--', color='red', label='5% Significance Level')
        plt.axhline(y=-significance_level, linestyle='--', color='red')
        plt.title(f'Time Series {i+1} - PACF')
        plt.legend()

        plt.tight_layout()
        plt.show()
        plt.close()

        # Fit ARIMA models with different configurations and compare AIC
        best_aic = np.inf
        best_model = None
        best_order = None

        for p in range(0, 6):  # Testing AR lags from 0 to 5 (based on PACF)
            for q in range(0, 2):  # Testing MA lags 0 and 1 (based on ACF)
                try:
                    model_fit = fit_arima(series, p, 1, q)
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_model = model_fit
                        best_order = (p, 1, q)
                except Exception as e:
                    print(f"Error for ARIMA({p},1,{q}): {e}")

        # Print the best model
        print(f"Best ARIMA Model: ARIMA{best_order} with AIC: {best_aic}")

        # Plot the residuals of the best model to check for any patterns
        residuals = best_model.resid
        plt.figure(figsize=(10, 6))
        plt.subplot(211)
        plt.plot(residuals)
        plt.title('Residuals of Best ARIMA Model')
        plt.subplot(212)
        plt.hist(residuals, bins=20)
        plt.title('Histogram of Residuals')
        plt.tight_layout()
        plt.show()
        plt.close()

        # Diagnostic plots to check if residuals are white noise
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plot_acf(residuals, ax=plt.gca(), lags=20)
        plt.title('ACF of Residuals')

        plt.subplot(1, 2, 2)
        plot_pacf(residuals, ax=plt.gca(), lags=20)
        plt.title('PACF of Residuals')

        plt.tight_layout()
        plt.show()
        plt.close()

if __name__ == "__main__":
    # Call the function to plot ACF and PACF
    plot_acf_pacf(last_year_data.iloc[observable_loads])
    fit_arima(last_year_data.iloc[observable_loads[0]], 5, 0, 1)



