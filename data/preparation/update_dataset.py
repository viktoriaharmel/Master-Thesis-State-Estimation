import pandas as pd
import numpy as np

full_data_path = 'path/to/full/data/'
estimated_data_path = 'path/to/estimated/data/'
dataset = 'dataset_complete_2018_base_no_pmu_targets.csv'
estimated_data_names = ['average_samples_angles_1.csv', 'average_samples_angles_2.csv', 'average_samples_angles_3.csv',
                        'average_samples_angles_1.csv', 'average_samples_angles_2.csv', 'average_samples_angles_3.csv',
                        'average_samples_voltages_reac_1.csv', 'average_samples_voltages_reac_2.csv', 'average_samples_voltages_reac_3.csv',
                        'average_samples_voltages_real_1.csv', 'average_samples_voltages_real_2.csv', 'average_samples_voltages_real_3.csv']
method = 'average'

full_data = pd.read_csv(full_data_path + dataset)

estimates = []
# Load the estimated data
for name in estimated_data_names:
    df = pd.read_csv(estimated_data_path + name)
    if 'loads' in name:
        df.set_index('Unnamed: 0', inplace=True)
        df = df.transpose()
        df.index = df.index.astype(int)
    else:
        df.drop(columns=['Unnamed: 0'], inplace=True)
    estimates.append(df)

# Adjust column names
for i, estimated_data in enumerate(estimates):
    adjusted_columns = []
    for col in estimated_data.columns:
        if 'angle_real' in col:
            adjusted_columns.append(col.replace('real_', ''))
        elif 'angle_reac' in col:
            adjusted_columns.append(col.replace('reac_', ''))
        else:
            adjusted_columns.append(col)

    estimated_data.columns = adjusted_columns

    estimates[i] = estimated_data


# Update the full data with the estimated data

for data_idx, estimated_data in enumerate(estimates):

    # Number of empty columns at the start
    estimated_data_filled = pd.DataFrame()
    errors_df = pd.DataFrame()
    n_empty_cols = estimated_data.isna().all().sum()

    if n_empty_cols > 0:
        # Identify the last n columns with values
        n_value_cols = estimated_data.shape[1] - n_empty_cols

        # Move values from the last n_value_cols to the first n_value_cols
        estimated_data.iloc[:, :n_value_cols] = estimated_data.iloc[:, -n_value_cols:].values

        # Drop the last n_empty_cols (columns with all missing values)
        estimated_data = estimated_data.iloc[:, :-n_empty_cols]

    for col in estimated_data.columns:
        if col in full_data.columns:
            # Insert estimated data for each group of columns
            previous_values = full_data[col]
            full_data[col] = estimated_data[col]

            errors_df[col] = abs(previous_values - full_data[col])
        
    errors_df.to_csv(estimated_data_path + f'errors_{estimated_data_names[data_idx]}')
    full_data.to_csv(full_data_path + f'dataset_{method}_2018_base_no_pmu_targets.csv')

print("Updated Dataset")
