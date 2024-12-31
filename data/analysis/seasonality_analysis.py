import matplotlib.pyplot as plt
from datetime import datetime
import dss
import os
import pandas as pd
import numpy as np

name = 'path/to/feeder/dir/'  # Replace with your feeder name
year = 2018
load_type = 'Commercial'
demand_type = 'Real'

dssObj = dss.DSS
dssCircuit = dssObj.ActiveCircuit

dssObj.Text.Command = "compile " + os.path.join(name, 'Master.dss')

load_names = dssCircuit.Loads.AllNames

loads_fname = name + "Loads.dss"

load_types = {'Residential': [], 'Commercial': []}

load_idx = 0

with open(loads_fname, 'r') as file:
    for line in file:
        # Remove any trailing whitespace or newline characters
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue

        if 'res' in line:
            load_types['Residential'].append(load_idx)
        else:
            load_types['Commercial'].append(load_idx)
        
        load_idx += 1



start_time = datetime(year=year, month=1, day=1, hour=0, minute=0)
end_time = datetime(year=year, month=12, day=31, hour=23, minute=45)
time_index = pd.date_range(start=start_time, end=end_time, freq='15min')
df = pd.DataFrame(index=time_index)


if demand_type == 'Real':
    demand_real = np.loadtxt(name + 'demand_unc.csv')
    demand_real = demand_real[load_types[load_type]]
    load_names = [load_names[i] for i in load_types[load_type]]
    for i, load_id in enumerate(load_names):
        df[f'{load_id}_real'] = demand_real[i]  # Real load demand
else:
    demand_reac = np.loadtxt(name + 'demand_imag.csv')
    demand_reac = demand_reac[load_types[load_type]]
    load_names = [load_names[i] for i in load_types[load_type]]
    for i, load_id in enumerate(load_names):
        df[f'{load_id}_reac'] = demand_reac[i]  # Imaginary load demand


def plot_yearly_data(df, ylabel, title):
    # Plot each load demand over time to see general trends and seasonality
    plt.figure(figsize=(12, 6))
    for column in df.columns:  # Loop through each load demand column
        plt.plot(df.index, df[column], label=column)
    plt.xlabel('Time')
    plt.ylabel(ylabel)
    plt.title(title)
    #plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    #plt.show()
    plt.savefig(name + 'seasonality_analysis/' + f'{title}')
    plt.close()


def plot_weekly_data(df, ylabel, title):
    # Weekly aggregation - mean demand per week
    df_weekly = df.resample('W').mean()

    # Plot the aggregated data
    plt.figure(figsize=(12, 6))
    for column in df_weekly.columns:
        plt.plot(df_weekly.index, df_weekly[column], label=column)

    plt.xlabel('Time')
    plt.ylabel(ylabel)
    plt.title(title)
    #plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    #plt.show()
    plt.savefig(name + 'seasonality_analysis/' + f'{title}')
    plt.close()


def plot_monthly_data(df, ylabel, title):
    # Monthly aggregation - mean demand per month
    df_monthly = df.resample('ME').mean()

    # Plot the aggregated data
    plt.figure(figsize=(12, 6))
    for column in df_monthly.columns:
        plt.plot(df_monthly.index, df_monthly[column], label=column)

    plt.xlabel('Time')
    plt.ylabel(ylabel)
    plt.title(title)
    #plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    #plt.show()
    plt.savefig(name + 'seasonality_analysis/' + f'{title}')
    plt.close()


def plot_quarterly_data(df, ylabel, title):
    # Quarterly Seasonal Plot
    # Extract quarter information from the timestamp index
    df['quarter'] = df.index.to_series().dt.quarter

    # Plot seasonal demand by quarter for each load
    fig, ax = plt.subplots(figsize=(12, 6))
    for column in df.columns[:-1]:  # Exclude 'quarter' column
        # Calculate the mean demand for each quarter for the current load
        quarterly_data = df.groupby('quarter')[column].mean()
        ax.plot(quarterly_data.index, quarterly_data.values, marker='o', label=column)

    ax.set_title(title)
    ax.set_xlabel("Quarter")
    ax.set_ylabel(ylabel)
    #plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    #plt.show()
    plt.savefig(name + 'seasonality_analysis/' + f'{title}')
    plt.close()


def fourier_transform(df, title):
    # Set the sampling interval to 60 minutes in terms of days
    d = 60 / 1440

    # Loop over each load demand column in df
    plt.figure(figsize=(12, 6))
    for column in df.columns:
        # Apply Fourier Transform to the current column
        fft_vals = np.fft.fft(df[column].values)
        fft_freqs = np.fft.fftfreq(len(fft_vals), d=d)
        
        # Plot Fourier spectrum for the current load demand
        plt.plot(fft_freqs, np.abs(fft_vals), label=column)

    # Plot settings
    plt.title(title)
    plt.xlabel("Frequency (cycles per day)")
    plt.ylabel("Magnitude")
    plt.xlim(0, 1)  # Only plot frequencies up to 1 cycle per day for better visibility
    #plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    #plt.show()
    plt.savefig(name + 'seasonality_analysis/' + f'{title}')
    plt.close()


if __name__ == "__main__":
    plot_yearly_data(df, 'Load Demand [kW]', f'{demand_type} {load_type} Electricity Demand in {year}')
    plot_weekly_data(df, 'Average Weekly Load Demand [kW]', f'Weekly Average {demand_type} {load_type} Electricity Demand in {year} - Each Line Representing the Weekly Load Demand of a Commercial Building')
    plot_monthly_data(df, 'Average Monthly Load Demand [kW]', f'Monthly Average {demand_type} {load_type} Electricity Demand in {year}')
    plot_quarterly_data(df, 'Average Quarterly Load Demand [kW]', f'Average Seasonal {demand_type} {load_type} Electricity Demand by Quarter in {year}')
    fourier_transform(df, f"Fourier Transform Spectrum for {load_type} {demand_type} Load Demands in {year}")
