import pandas as pd
import ast
import csv

import json

def save_input_feature_map(input_feature_map, file_path):
    """
    Save the input_feature_map to a JSON file.

    Parameters:
    - input_feature_map: A dictionary mapping each measurement to its column index in the dataset.
    - file_path: Path to the file where the input_feature_map will be saved.
    """
    with open(file_path, 'w') as file:
        json.dump(input_feature_map, file, indent=4)
    print(f"Input feature map saved to {file_path}")


def load_input_feature_map(file_path):
    """
    Load the input_feature_map from a JSON file.

    Parameters:
    - file_path: Path to the file where the input_feature_map is saved.

    Returns:
    - input_feature_map: A dictionary mapping each measurement to its column index in the dataset.
    """
    with open(file_path, 'r') as file:
        input_feature_map = json.load(file)
    print(f"Input feature map loaded from {file_path}")
    return input_feature_map


def create_input_feature_map(bus_measurement_df, dataset_columns):
    """
    Create a mapping from measurements to their column indices in the dataset.
    
    Parameters:
    - bus_measurement_df: A DataFrame where each row represents a bus and its associated measurements.
      The DataFrame should have columns like ['bus', 'neighbor_buses', 'lines', 'loads', 'transformers'].
    - dataset_columns: List or Index object containing the column names of the dataset.

    Returns:
    - input_feature_map: A dictionary mapping each measurement to its column index in the dataset.
    """
    input_feature_map = {}

    # Flatten all measurements in the bus_measurement_df into a set
    all_measurements = set()
    for _, row in bus_measurement_df.iterrows():
        # Combine all associated measurements for the current bus
        measurements = row['neighbor_buses'] + row['lines'] + row['loads'] + row['transformers']
        measurements.append(row['bus'])
        all_measurements.update(measurements)

    # Map each measurement to its column index in the dataset
    for measurement in all_measurements:
        matching_columns = [col for col in dataset_columns if measurement in col and 'target' not in col]
        input_feature_map[measurement] = []
        for col in matching_columns:
            input_feature_map[measurement].append(dataset_columns.get_loc(col))

    return input_feature_map


pmu_locations = ['p1rdt4659.1.2.3', 'p1rdt4949.1.2.3', 'p1rdt833_b1_1.1', 'p1rdt1317_b1_1.1.2.3', 'p1rdt1175.3', 'p1rdm6643.1.2', 'p1rdt8002.1.2.3', 'p1rdt7077.1.2.3', 'p1rdt4568lv.1.2', 'p1rdt2829.1.2.3', 'p1rdt4396lv.1.2', 'p1rdt3614-p1rdt4656xx.1.2.3', 'p1rdt4879_b2_1.1', 'p1rdt5032-p1rdt6794x_b1_1.3', 'p1rdt5663.1.2.3', 'p1rdt6284lv.1.2', 'p1rdt5033_b1_1.1.2.3', 'p1rdt4730_b1_1.1', 'p1rdt1069_b2_1.1.2.3', 'p1rdt3810-p1rdt7137x_b1_1.1', 'p1rdt6285-p1rdt7899x.2', 'p1rdt6852.1', 'p1rdt4396-p1rdt831x_b1_1.1', 'p1rdt7136.3', 'p1rdt4401_b1_1.3', 'p1rdt6284.1.2.3', 'p1rdt320.2', 'p1rdm9214.1.2', 'p1rdt5351.3', 'p1rdt1318-p1rdt3520x_b1_1.2', 'p1rdt4728.2', 'p1rdm1612.1.2', 'p1rdt1762.1.2.3', 'p1rdm117.1.2', 'p1rdm759.1.2', 'p1rdt319lv.1.2', 'p1rdt7137.1.2.3', 'p1rdt7136lv.1.2', 'p1rdm7296.1.2', 'p1rdt3243-p1rdt5935x_b1_1.2', 'p1rdt3243.1', 'p1rdt3243-p1rdt8281x_b1_1.1', 'p1rdt1436.1.2.3', 'p1rdt7437_b1_1.1', 'p1rdt4949.2', 'p1rdt830.2', 'p1rdt5031-p1rdt7257x_b1_1.3', 'p1rdt2109-p1rdt833x_b1_1.1', 'p1rdt6853.1', 'p1rdt2110-p1rdt963x_b1_1.2', 'p1rdm12169.1.2', 'p1rdm6959.1.2', 'p1rdt2630-p1rdt831x.3', 'p1rdt2925-p1rdt3614x.3', 'p1rdt2109.1.2.3', 'p1rdt1436lv.1.2', 'p1rdt2830.1.2.3', 'p1rdt441.1', 'p1rdm7458.1.2', 'p1rdm123.1.2', 'p1rdt3616.1.2.3', 'p1rdt1869-p1rdt53x.2', 'p1rdt1761-p1rdt3616x.2', 'p1rdt1760-p1rdt7325x_b2_1.2', 'p1rdt3614-p1rdt7257xx.1.2.3', 'p1rdt834.1.2.3', 'p1rdt1070.1.2.3', 'p1rdm5898.1.2', 'p1rdm9429.1.2', 'p1rdt1318_b2_1.1.2.3', 'p1rdm4531.1.2', 'p1rdt963lv.1.2', 'p1rdt7265_b1_1.1', 'p1rdt5360-p1rdt565x.2']
pmu_locations = [pmu_loc.split('.')[0] for pmu_loc in pmu_locations]

data_path = 'path/to/data/dir/'

mapping = pd.read_csv(data_path + 'bus_measurement_mapping.csv')
dataset = pd.read_csv(data_path + 'dataset_complete_2018_base.csv')
with open(data_path + 'target_cols.csv', 'r') as file:
    reader = csv.reader(file)
    target_columns = next(reader)  # Read the first row

# Only for test dataset
mapping_cols = mapping['bus']
dataset_bus_cols = [col for col in dataset.columns if 'angle' in col and not any(f'_{loc}_' in col for loc in pmu_locations)]
cols_to_remove = [col for col in mapping_cols if not any(f'_{col}_' in data_col for data_col in dataset_bus_cols)]

mapping_reduced = mapping[~mapping['bus'].isin(cols_to_remove)]

dataset.set_index('Unnamed: 0', inplace=True)

# Save the reduced dataset excluding the PMU target columns
dataset.to_csv(data_path + 'dataset_complete_2018_reduced_no_pmu_targets.csv')


# Ensure that each field is a list
for col in ['neighbor_buses', 'lines', 'loads', 'transformers']:
    mapping_reduced[col] = mapping_reduced[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Apply a function to remove unwanted elements from the lists
mapping_reduced['neighbor_buses'] = mapping_reduced['neighbor_buses'].apply(
    lambda buses: [bus for bus in buses if bus not in cols_to_remove]
)

mapping_reduced.to_csv(data_path + 'bus_measurement_mapping.csv')

feature_dataset = dataset.drop(columns=target_columns)

input_feature_map = create_input_feature_map(mapping_reduced, feature_dataset.columns)

save_input_feature_map(input_feature_map, data_path + 'input_feature_map.json')



