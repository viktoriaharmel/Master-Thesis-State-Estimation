import pandas as pd

""" Combine metrics from different scenarios into a single dataframe. """

name = 'path/to/grid/directory/'
feeder_name = 'p1rhs0_1247--p1rdt5663'

# Load the metrics dataframes
base_df = pd.read_csv(name + 'scenarios/base_timeseries/metrics.csv')
base_df = base_df.loc[base_df['Feeder Name'] == feeder_name]

sh_bl_df = pd.read_csv(name + 'scenarios/solar_high_batteries_low_timeseries/metrics.csv')
sh_bl_df = sh_bl_df.loc[sh_bl_df['Feeder Name'] == feeder_name]

sh_bh_df = pd.read_csv(name + 'scenarios/solar_high_batteries_high_timeseries/metrics.csv')
sh_bh_df = sh_bh_df.loc[sh_bh_df['Feeder Name'] == feeder_name]

index = ['base data', 'high PV, low battery quota', 'high PV, high battery quota']
columns = ['Total Peak Planning Load (MW)', 'Total Reactive Peak Planning Load (MVar)', 'Average Peak Planning Load Imbalance by Phase', 
           'Total Number of Customers', 'Diameter (Maximum Eccentricity)', 'Number of PVs', 'Total PV Capacity (MW)', 'Number of Batteries', 
           'Total Capacity of Batteries (MW)', 'Percentage of Residential Customers', 'Percentage of Commercial Customers', 
           'Percentage of Industrial Customers', 'Line Configuration']

result_df = pd.DataFrame(index=index, columns=columns)

# Fill in the data
for col in columns:
    result_df.loc['base data', col] = base_df.loc[:, col].iloc[0]
    result_df.loc['high PV, low battery quota', col] = sh_bl_df.loc[:, col].iloc[0]
    result_df.loc['high PV, high battery quota', col] = sh_bh_df.loc[:, col].iloc[0]

result_df = result_df.transpose()

result_df.to_csv(name + 'data_descriptions/comp_metrics.csv')

