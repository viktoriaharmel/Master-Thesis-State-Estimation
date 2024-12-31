import pandas as pd
import numpy as np
import gc
import json
from datetime import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar

from utils import get_t_power_filename

name = 'path/to/feeder/dir/'  # Replace with your feeder name
data_name = 'path/to/data/dir/'
year = 2018

# Load from the JSON file
with open(data_name + "circuit_data.json", "r") as f:
    loaded_data = json.load(f)

# Access the lists
bus_names = loaded_data["bus_names"]
load_names = loaded_data["load_names"]
transformer_names = loaded_data["transformer_names"]

fname = get_t_power_filename(name)
loadfile = pd.read_csv(fname, sep=',')
ts = loadfile[loadfile['Element'].str.contains("Transformer")]
t_phases = ts[' Terminal'].reset_index(drop=True)

# Step 1: Create the DataFrame with the time index
start_time = datetime(year=year, month=1, day=1, hour=0, minute=0)
end_time = datetime(year=year, month=12, day=31, hour=23, minute=45)
time_index = pd.date_range(start=start_time, end=end_time, freq='15min')

df = pd.DataFrame(index=time_index)

time_steps = 3504 # Number of time steps in each partial file

# Step 2: Add Complex Voltage Measurements and Angles from npz files
for p, partial_file in enumerate([data_name + "local_metrics_complex_volts_angles_p1.npz", data_name + "local_metrics_complex_volts_angles_p2.npz", data_name + "local_metrics_complex_volts_angles_p3.npz", data_name + "local_metrics_complex_volts_angles_p4.npz", 
                    data_name + "local_metrics_complex_volts_angles_p5.npz", data_name + "local_metrics_complex_volts_angles_p6.npz", data_name + "local_metrics_complex_volts_angles_p7.npz", data_name + "local_metrics_complex_volts_angles_p8.npz", data_name + "local_metrics_complex_volts_angles_p9.npz", data_name + "local_metrics_complex_volts_angles_p10.npz"]):
    data = np.load(partial_file, allow_pickle=True) # Load the partial file
    voltage_data = data['v_complex'].item()
    angle_data = data['v_angles'].item()
    for bus, values in voltage_data.items():
        bus_name = bus_names[bus-1]
        for t in range(time_steps):
            for i, phase_value in enumerate(values[t]):
                phase = (i // 2) + 1
                if i % 2 == 0:
                    col_name = 'v_real_' + bus_name + '_phase' + str(phase)
                else:
                    col_name = 'v_reac_' + bus_name + '_phase' + str(phase)
                # Add random noise around 10^-6 to phase_value
                current_idx = (p * time_steps) + t
                noise = np.random.normal(loc=0, scale=1e-6)  # mean 0, standard deviation 10^-6
                df.at[df.index[current_idx], col_name] = phase_value + noise # Add the values to the appropriate time indices
                df.at[df.index[current_idx], col_name + '_target'] = phase_value
    for bus, values in angle_data.items():
        bus_name = bus_names[bus-1]
        for t in range(time_steps):
            for i, angle_value in enumerate(values[t]):
                col_name = 'v_angle_' + bus_name + '_phase' + str(i+1)
                current_idx = (p * time_steps) + t
                noise = np.random.normal(loc=0, scale=1e-6)  # mean 0, standard deviation 10^-6
                df.at[df.index[current_idx], col_name] = angle_value + noise
                df.at[df.index[current_idx], col_name + '_target'] = angle_value

del data, voltage_data, angle_data
gc.collect()

time_steps = 4380
# Step 3: Add Current Magnitudes from npz files
for p, partial_file in enumerate([data_name + "local_metrics_current_mags_p1.npz", data_name + "local_metrics_current_mags_p2.npz", data_name + "local_metrics_current_mags_p3.npz", data_name + "local_metrics_current_mags_p4.npz", 
                                data_name + "local_metrics_current_mags_p5.npz", data_name + "local_metrics_current_mags_p6.npz", data_name + "local_metrics_current_mags_p7.npz", data_name + "local_metrics_current_mags_p8.npz"]):
    data = np.load(partial_file, allow_pickle=True)
    current_data = data['c_mags_all']
    for t in range(time_steps):
        for i, mag_values in enumerate(current_data):
            line = mag_values[0]
            for phase, mag_value in enumerate(mag_values[t+1]):
                col_name = 'current_magnitude_' + line + '_phase' + str(phase+1)
                current_idx = (p * time_steps) + t
                noise = np.random.normal(loc=0, scale=1e-3)  # mean 0, standard deviation 10^-3
                df.at[df.index[current_idx], col_name] = mag_value + noise

del data, current_data
gc.collect()

# Step 4: Add Load Demands from CSV files
demand_real = np.loadtxt(data_name + 'demand_unc.csv')
demand_reac = np.loadtxt(data_name + 'demand_imag.csv')
for i, load_id in enumerate(load_names):
    noise = np.random.normal(loc=0, scale=1e-3)  # mean 0, standard deviation 10^-3
    df[load_id + '_real'] = demand_real[i] + noise  # Real load demand
    df[load_id + '_reac'] = demand_reac[i] + noise  # Imaginary load demand

del demand_real, demand_reac
gc.collect()

# Step 5: Add Transformer Power Data
transformer_data = np.load(data_name + "local_metrics_t_powers.npz", allow_pickle=True)
t_real_all = transformer_data['t_real_all']

# Add real power data
t_name_idx = 0
for i, t_real in enumerate(t_real_all):
    phase = t_phases[i]
    if phase == 1:
        transformer_id = transformer_names[t_name_idx]
        t_name_idx += 1
    noise = np.random.normal(loc=0, scale=1e-3)  # mean 0, standard deviation 10^-3
    df['t_real_' + transformer_id + '_phase' + str(phase)] = t_real + noise

del t_real_all
gc.collect()

t_reac_all = transformer_data['t_reac_all']
# Add reactive power data
t_name_idx = 0
for i, t_reac in enumerate(t_reac_all):
    phase = t_phases[i]
    if phase == 1:
        transformer_id = transformer_names[t_name_idx]
        t_name_idx += 1
    noise = np.random.normal(loc=0, scale=1e-3)  # mean 0, standard deviation 10^-3
    df['t_reac_' + transformer_id + '_phase' + str(phase)] = t_reac + noise

del t_reac_all
gc.collect()

t_powers_all = transformer_data['t_powers_all']
# Add apparent power data
t_name_idx = 0
for i, t_power in enumerate(t_powers_all):
    phase = t_phases[i]
    if phase == 1:
        transformer_id = transformer_names[t_name_idx]
        t_name_idx += 1
    noise = np.random.normal(loc=0, scale=1e-3)  # mean 0, standard deviation 10^-3
    df['t_apparent_' + transformer_id + '_phase' + str(phase)] = t_power + noise

del t_powers_all, transformer_data
gc.collect()

df['bias'] = 1

# Add temporal features
# 1. Hour of the Day
df['hour'] = df.index.hour.astype(int)

# 2. Day of the Week
df['day_of_week'] = df.index.dayofweek.astype(int)

# 3. Is Weekend
df['is_weekend'] = df.index.dayofweek >= 5
df['is_weekend'] = df['is_weekend'].astype(int)

# 4. Day of the Month
df['day_of_month'] = df.index.day.astype(int)

# 5. Month of the Year
df['month'] = df.index.month.astype(int)

# 6. Quarter of the Year
df['quarter'] = df.index.quarter.astype(int)

# 7. Holiday Indicator (using US Federal Holidays as an example)
cal = USFederalHolidayCalendar()
holidays = cal.holidays(start=df.index.min(), end=df.index.max())
df['is_holiday'] = df.index.isin(holidays)
df['is_holiday'] = df['is_holiday'].astype(int)

# 8. Season Indicator
def get_season(month):
    if month in [12, 1, 2]:     # winter
        return 0
    elif month in [3, 4, 5]:    # spring
        return 1
    elif month in [6, 7, 8]:    # summer
        return 2
    else:                       # fall
        return 3
df['season'] = df.index.month.map(get_season)

# Save the DataFrame to a CSV file
chunk_size = 73  # Number of rows per chunk
for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i:i + chunk_size]
    mode = 'w' if i == 0 else 'a'  # Write mode for the first chunk, append mode thereafter
    header = True if i == 0 else False  # Include header only in the first chunk
    chunk.to_csv(data_name + 'dataset_complete_' + str(year) + '_base.csv', mode=mode, header=header, index=True)
