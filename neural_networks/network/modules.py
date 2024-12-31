import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from dataset_class import CSVDataset 
from datetime import datetime

class PAWNNLayer(nn.Module):
    ''' Class for a PAWNN layer with a mask. '''
    def __init__(self, input_size, output_size, mask):
        super(PAWNNLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        # Register the mask as a buffer so it doesn't update during training
        self.register_buffer("mask", mask) 

    def forward(self, x):
        # Apply the mask to the weights before forwarding
        self.linear.weight.data *= self.mask
        return self.linear(x)


class PAWNN(nn.Module):
    ''' Class for a PAWNN model. '''
    def __init__(self, hparams, bus_measurements, input_feature_map, target_cols):
        super().__init__() 

        # Load hyperparameters
        self.hparams = {
            "input_size": hparams.get("input_size", 1000),
            "hidden_sizes_per_bus": hparams.get("hidden_sizes_per_bus", [100, 100, 100, 100, 100, 100, 100, 100, 100]),
            "output_size": hparams.get("output_size", 9920),
            "num_layers": hparams.get("num_layers", 10),
            "activation": hparams.get("activation", nn.LeakyReLU()),
            "learning_rate": hparams.get("learning_rate", 0.001),
            "weight_decay": hparams.get("weight_decay", 0),
            "batch_size": hparams.get("batch_size", 64),
        }

        self.bus_measurements = bus_measurements
        self.input_feature_map = input_feature_map
        self.target_cols = target_cols

        # Create mappings to identify the neurons corresponding to each bus and output
        self.bus_to_neuron_map = {bus: idx for idx, bus in enumerate(bus_measurements.keys())}
        self.bus_to_output_map = {
            bus: [index for index, target in enumerate(target_cols) if f"_{bus}_" in target]
            for bus in bus_measurements.keys()
        }

        self.num_buses = len(bus_measurements.keys())

        # Create PAWNN layers with masks
        layers = []
        input_size = self.hparams['input_size']
        for i in range(self.hparams["num_layers"]):
            output_size = self.hparams["hidden_sizes_per_bus"][i] * self.num_buses
            if i == 0:
                mask = self.create_input_to_hidden_mask()
            else:
                mask = self.create_layer_mask(i)
            layers.append(PAWNNLayer(input_size, output_size, mask))
            input_size = output_size
        mask = self.create_hidden_to_output_mask()
        layers.append(PAWNNLayer(input_size, self.hparams['output_size'], mask))

        # Register the layers as a ModuleList
        self.pawnn_layers = nn.ModuleList(layers)

    def create_layer_mask(self, layer_idx):
        """
        Creates a mask for each layer to enforce locality in the connections.
        Defines a mask where each bus's neurons are connected only to the neurons of neighboring buses.
        """
        # Initialize the mask with zeros
        mask = torch.zeros((self.hparams['hidden_sizes_per_bus'][layer_idx] * self.num_buses, self.hparams['hidden_sizes_per_bus'][layer_idx - 1] * self.num_buses))

        for bus in self.bus_measurements.keys():
            # Current bus and its neighbors in the partition graph
            neighbors = list(self.bus_measurements[bus]['neighbor_buses']) + [bus]
            # Get the neuron indices for this bus in the current and previous layers
            bus_start_idx = self.bus_to_neuron_map[bus] * self.hparams['hidden_sizes_per_bus'][layer_idx - 1]
            bus_end_idx = bus_start_idx + self.hparams['hidden_sizes_per_bus'][layer_idx - 1]
            for neighbor in neighbors:
                # Define the range of neurons corresponding to this bus's neighbor
                neighbor_start_idx = self.hparams['hidden_sizes_per_bus'][layer_idx] * self.bus_to_neuron_map[neighbor]
                neighbor_end_idx = neighbor_start_idx + self.hparams['hidden_sizes_per_bus'][layer_idx]
                # Connect the current bus's neurons to the neighbor's neurons
                mask[neighbor_start_idx:neighbor_end_idx, bus_start_idx:bus_end_idx] = 1

        return mask
    
    def create_input_to_hidden_mask(self):
        """
        Creates a mask for the connections from the input layer to the first hidden layer.
        Ensures each measurement is connected only to the neurons of the bus it belongs to.
        """
        # Initialize the mask with zeros
        mask = torch.zeros((self.hparams['hidden_sizes_per_bus'][0] * self.num_buses, self.hparams['input_size']))  # First hidden layer x input features

        for bus_idx, (bus, measurements) in enumerate(self.bus_measurements.items()):
            # Flatten the list of measurements for this bus
            measurements = [item for sublist in measurements.values() for item in sublist] + [bus]

            # Get neuron indices for this bus in the first hidden layer
            start_idx = bus_idx * self.hparams['hidden_sizes_per_bus'][0]
            end_idx = start_idx + self.hparams['hidden_sizes_per_bus'][0]

            # Get input feature indices for measurements belonging to this bus
            measurement_indices = [self.input_feature_map[m] for m in measurements if m in self.input_feature_map]
            measurement_indices = [m for sub in measurement_indices for m in sub]

            # Set the mask values for this bus's neurons and input features to 1
            for neuron_idx in range(start_idx, end_idx):
                mask[neuron_idx, measurement_indices] = 1
        
        # Connect global features to all neurons
        global_features_start_idx = self.hparams['input_size'] - 9  # The last 9 columns are global features
        for feature_idx in range(global_features_start_idx,self.hparams['input_size']):
            mask[:, feature_idx] = 1  # Connect each global feature to all neurons in the first hidden layer

        return mask
    
    def create_hidden_to_output_mask(self):
        """
        Creates a mask for the connections from the input layer to the first hidden layer.
        Ensures each measurement is connected only to the neurons of the bus it belongs to.
        """
        # Initialize the mask with zeros
        mask = torch.zeros((self.hparams['output_size'], self.hparams['hidden_sizes_per_bus'][-1] * self.num_buses))  # First hidden layer x input features

        for bus in self.bus_measurements.keys():
            # Current bus and its targets in the partition graph
            target_indices = self.bus_to_output_map[bus]
            bus_start_idx = self.bus_to_neuron_map[bus] * self.hparams['hidden_sizes_per_bus'][-1]
            bus_end_idx = bus_start_idx + self.hparams['hidden_sizes_per_bus'][-1]
            # Connect the current bus's neurons to the target neurons
            for target_index in target_indices:
                mask[target_index, bus_start_idx:bus_end_idx] = 1

        return mask


    def forward(self, x):
        """ Forward pass through the PAWNN model. """
        # Forward pass through each PAWNN layer
        for idx, layer in enumerate(self.pawnn_layers):
            x = layer(x)
            if idx < len(self.pawnn_layers) - 1:
                x = self.hparams['activation'](x)
        return x

    def mean_absolute_percentage_error(self, y_true, y_pred):
        """ Compute the mean absolute percentage error. """
        mask = y_true != 0  # Exclude zero values in y_true
        return torch.mean(torch.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def r2_score(self, y_true, y_pred):
        """ Compute the R^2 score. """
        y_true_mean = torch.mean(y_true)
        ss_total = torch.sum((y_true - y_true_mean) ** 2)
        ss_residual = torch.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)

    def configure_optimizers(self):
        """ Create optimizer based on hyperparameters """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"]
        )
        return optimizer



class PAWNN_DataModule(nn.Module):
    """ DataModule for the PAWNN model. """
    def __init__(self, data_path, target_columns, batch_size, num_workers=4, split_ratios=(0.7, 0.15, 0.15), root=None):
        super().__init__()
        self.data_path = data_path
        self.target_columns = target_columns
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratios = split_ratios
        self.root = root

        # Placeholder for column statistics, to be computed in setup()
        self.column_stats = None
        self.transform = None

    def calculate_column_stats(self, df, selected_columns):
        """
        Calculate min, max, and mean for normalization.
        """
        mn, mx, mean = df.min(skipna=True, numeric_only=True),\
                       df.max(skipna=True, numeric_only=True),\
                       df.mean(skipna=True, numeric_only=True)

        column_stats = {}
        for column in selected_columns:
            column_stats[column] = {
                'min': mn[column],
                'max': mx[column],
                'mean': mean[column]
            }
        return column_stats

    def stratified_split(self, df):
        """
        Perform stratified splitting of the data by day and quarter.
        """
        # Add day and column for grouping
        df['day'] = df.index.date  # Extract date as day identifier

        # Get unique days for each quarter
        unique_days = df.groupby(['quarter', 'day']).size().reset_index().drop(columns=0)

        # Prepare stratified splits
        train_days, val_days, test_days = [], [], []
        for quarter in unique_days['quarter'].unique():
            days_in_quarter = unique_days[unique_days['quarter'] == quarter]['day'].values
            np.random.shuffle(days_in_quarter)  # Shuffle to ensure random selection

            # Perform split by quarter
            train_split = int(self.split_ratios[0] * len(days_in_quarter))
            val_split = int(self.split_ratios[1] * len(days_in_quarter)) + train_split

            train_days.extend(days_in_quarter[:train_split])
            val_days.extend(days_in_quarter[train_split:val_split])
            test_days.extend(days_in_quarter[val_split:])

        # Filter the DataFrame based on the split days
        train_df = df[df['day'].isin(train_days)]
        val_df = df[df['day'].isin(val_days)]
        test_df = df[df['day'].isin(test_days)]

        # Drop temporary column
        train_df = train_df.drop(columns=['day'])
        val_df = val_df.drop(columns=['day'])
        test_df = test_df.drop(columns=['day'])

        return train_df, val_df, test_df

    def setup(self):
        """ Load and prepare the dataset. """

        # Load the full dataset
        data = pd.read_csv(self.data_path, parse_dates=True)
        
        # data.drop(columns=['Unnamed: 0.1', 'Unnamed: 0.2', 'Unnamed: 0'], inplace=True) # Uncomment this line if the dataset contains pseudo-measurements
        data.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'], inplace=True)
        
        start_time = datetime(year=2018, month=1, day=1, hour=0, minute=0)
        end_time = datetime(year=2018, month=12, day=31, hour=23, minute=45)
        time_index = pd.date_range(start=start_time, end=end_time, freq='15min')
        
        data.index = time_index
        
        data.index = pd.to_datetime(data.index)

        # Calculate column statistics for normalization
        feature_columns = [col for col in data.columns if col not in self.target_columns]
        feature_columns_no_indicators = [col for col in feature_columns if col not in ['bias', 'hour', 'day_of_week', 'is_weekend', 'day_of_month', 'month', 'quarter', 'is_holiday', 'season']]
        self.column_stats = self.calculate_column_stats(data, feature_columns_no_indicators + self.target_columns)

        # Define transformation using column statistics
        self.transform = FeatureSelectorAndNormalizationTransform(self.column_stats, self.target_columns)

        # Create datasets with transformations applied
        self.train_dataset = CSVDataset(root=self.root, target_columns=self.target_columns, input_data=data, transform=self.transform, mode='train')
        self.val_dataset = CSVDataset(root=self.root, target_columns=self.target_columns, input_data=data, transform=self.transform, mode='val')
        self.test_dataset = CSVDataset(root=self.root, target_columns=self.target_columns, input_data=data, transform=self.transform, mode='test')

    def return_dataloader_dict(self, dataset):
        """
        Helper function to return DataLoader arguments.
        """
        return {
            'dataset': dataset,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'drop_last': False,
            #'persistent_workers': True,
            'pin_memory': True
        }

    def train_dataloader(self):
        """
        Returns a DataLoader for training data.
        """
        # drop_last = True
        return DataLoader(**self.return_dataloader_dict(self.train_dataset), shuffle=True)

    def val_dataloader(self):
        """
        Returns a DataLoader for validation data.
        """
        return DataLoader(**self.return_dataloader_dict(self.val_dataset), shuffle=False)

    def test_dataloader(self):
        """
        Returns a DataLoader for test data.
        """
        return DataLoader(**self.return_dataloader_dict(self.test_dataset), shuffle=False)


class FC(nn.Module):
    """ Fully connected neural network model. """
    def __init__(self, hparams, bus_measurements, target_cols):
        super().__init__() 

        # Set hyperparameters
        self.hparams = {
            "input_size": hparams.get("input_size", 1000),  # example size, update accordingly
            "hidden_sizes_per_bus": hparams.get("hidden_sizes_per_bus", [100, 100, 100, 100, 100, 100, 100, 100, 100]),
            "output_size": hparams.get("output_size", 9920),  # update accordingly
            "num_layers": hparams.get("num_layers", 10),
            "activation": hparams.get("activation", nn.LeakyReLU()),
            "learning_rate": hparams.get("learning_rate", 0.001),
            "weight_decay": hparams.get("weight_decay", 0),
            "batch_size": hparams.get("batch_size", 64),
        }

        self.bus_measurements = bus_measurements
        self.target_cols = target_cols
        self.num_buses = len(bus_measurements.keys())
        self.input_size = sum(len(measurements) for measurements in bus_measurements.values())

        # Create FC layers based on hidden sizes
        layers = []
        input_size = self.hparams['input_size']
        for i in range(self.hparams["num_layers"]):
            output_size = self.hparams["hidden_sizes_per_bus"][i] * self.num_buses
            layers.append(nn.Linear(input_size, output_size))
            input_size = output_size
        layers.append(nn.Linear(input_size, self.hparams['output_size']))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx < len(self.layers) - 1:
                x = self.hparams['activation'](x)
        return x

    def mean_absolute_percentage_error(self, y_true, y_pred):
        mask = y_true != 0  # Exclude zero values in y_true
        return torch.mean(torch.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def r2_score(self, y_true, y_pred):
        y_true_mean = torch.mean(y_true)
        ss_total = torch.sum((y_true - y_true_mean) ** 2)
        ss_residual = torch.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)


    def configure_optimizers(self):
        # Create optimizer based on hyperparameters
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"]
        )
        return optimizer



class FC_DataModule(nn.Module):
    def __init__(self, data_path, target_columns, batch_size, num_workers=4, split_ratios=(0.7, 0.15, 0.15), root=None):
        super().__init__()
        self.data_path = data_path
        self.target_columns = target_columns
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_ratios = split_ratios
        self.root = root

        # Placeholder for column statistics, to be computed in setup()
        self.column_stats = None
        self.transform = None

    def calculate_column_stats(self, df, selected_columns):
        """
        Calculate min, max, and mean for normalization.
        """
        mn, mx, mean = df.min(skipna=True, numeric_only=True),\
                       df.max(skipna=True, numeric_only=True),\
                       df.mean(skipna=True, numeric_only=True)

        column_stats = {}
        for column in selected_columns:
            column_stats[column] = {
                'min': mn[column],
                'max': mx[column],
                'mean': mean[column]
            }
        return column_stats

    def stratified_split(self, df):
        """
        Perform stratified splitting of the data by day and quarter.
        """
        # Add day column for grouping
        df['day'] = df.index.date  # Extract date as day identifier

        # Get unique days for each quarter
        unique_days = df.groupby(['quarter', 'day']).size().reset_index().drop(columns=0)

        # Prepare stratified splits
        train_days, val_days, test_days = [], [], []
        for quarter in unique_days['quarter'].unique():
            days_in_quarter = unique_days[unique_days['quarter'] == quarter]['day'].values
            np.random.shuffle(days_in_quarter)  # Shuffle to ensure random selection

            # Perform split by quarter
            train_split = int(self.split_ratios[0] * len(days_in_quarter))
            val_split = int(self.split_ratios[1] * len(days_in_quarter)) + train_split

            train_days.extend(days_in_quarter[:train_split])
            val_days.extend(days_in_quarter[train_split:val_split])
            test_days.extend(days_in_quarter[val_split:])

        # Filter the DataFrame based on the split days
        train_df = df[df['day'].isin(train_days)]
        val_df = df[df['day'].isin(val_days)]
        test_df = df[df['day'].isin(test_days)]

        # Drop temporary columns
        train_df = train_df.drop(columns=['day'])
        val_df = val_df.drop(columns=['day'])
        test_df = test_df.drop(columns=['day'])

        return train_df, val_df, test_df

    def setup(self):
        # Load the full dataset
        
        data = pd.read_csv(self.data_path, parse_dates=True)

        print(data.head)
        
        data.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'], inplace=True)
        
        start_time = datetime(year=2018, month=1, day=1, hour=0, minute=0)
        end_time = datetime(year=2018, month=12, day=31, hour=23, minute=45)
        time_index = pd.date_range(start=start_time, end=end_time, freq='15min')
        
        data.index = time_index
        
        # Ensure the index is a DatetimeIndex
        data.index = pd.to_datetime(data.index)

        # Calculate column statistics for normalization
        feature_columns = [col for col in data.columns if col not in self.target_columns]
        feature_columns_no_indicators = [col for col in feature_columns if col not in ['bias', 'hour', 'day_of_week', 'is_weekend', 'day_of_month', 'month', 'quarter', 'is_holiday', 'season']]
        self.column_stats = self.calculate_column_stats(data, feature_columns_no_indicators + self.target_columns)

        # Define transformation using column statistics
        self.transform = FeatureSelectorAndNormalizationTransform(self.column_stats, self.target_columns)

        # Create datasets with transformations applied
        self.train_dataset = CSVDataset(root=self.root, target_columns=self.target_columns, input_data=data, transform=self.transform, mode='train')
        self.val_dataset = CSVDataset(root=self.root, target_columns=self.target_columns, input_data=data, transform=self.transform, mode='val')
        self.test_dataset = CSVDataset(root=self.root, target_columns=self.target_columns, input_data=data, transform=self.transform, mode='test')

    def return_dataloader_dict(self, dataset):
        """
        Helper function to return DataLoader arguments.
        """
        return {
            'dataset': dataset,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'drop_last': False,
            #'persistent_workers': True,
            'pin_memory': True
        }

    def train_dataloader(self):
        """
        Returns a DataLoader for training data.
        """
        # drop_last = True
        return DataLoader(**self.return_dataloader_dict(self.train_dataset), shuffle=True)

    def val_dataloader(self):
        """
        Returns a DataLoader for validation data.
        """
        return DataLoader(**self.return_dataloader_dict(self.val_dataset), shuffle=False)

    def test_dataloader(self):
        """
        Returns a DataLoader for test data.
        """
        return DataLoader(**self.return_dataloader_dict(self.test_dataset), shuffle=False)



class FeatureSelectorAndNormalizationTransform:
    """
    Select numerical features and normalize them between 0 and 1.
    """

    def __init__(self, column_stats, target_columns):
        """
        :param column_stats: A dictionary mapping each column to min, max, and mean statistics.
        :param target_columns: List of target column names.
        """
        self.column_stats = column_stats
        self.target_columns = target_columns

    def __call__(self, data_dict):
        def normalize_column(value, column_name):
            """ Normalize the column value between 0 and 1. """
            mn = self.column_stats[column_name]['min']
            mx = self.column_stats[column_name]['max']
            divisor = mx - mn
            if divisor == 0:
                divisor = 1e-6
            return (value - mn) / divisor
        
        def normalize_angle_column(value):
            """ Normalize the angle column value between -pi and pi. """
            value = value / torch.pi  # For radians
            return value

        # Normalize feature columns
        feature_values = []
        for column in data_dict['features'].index:
            if column in self.column_stats and column not in self.target_columns:

                if np.isnan(data_dict['features'][column]):
                    mean_col_val = self.column_stats[column]['mean']
                    data_dict['features'][column] = mean_col_val

                # Normalize the column value
                if 'angle' in column:
                    normalized_value = normalize_angle_column(data_dict['features'][column])
                else:
                    normalized_value = normalize_column(data_dict['features'][column], column)
                feature_values.append(normalized_value)
            if column in ['bias', 'hour', 'day_of_week', 'is_weekend', 'day_of_month', 'month', 'quarter', 'is_holiday', 'season']:
                feature_values.append(data_dict['features'][column])

        # Keep only selected feature columns and convert to float32
        data_dict['features'] = torch.tensor(feature_values, dtype=torch.float32)

        # Normalize each target column
        normalized_targets = []
        for target_col in self.target_columns:
            old_value = data_dict['targets'][target_col]
            normalized_targets.append(normalize_column(old_value, target_col))

        # Store normalized targets as a tensor
        data_dict['targets'] = torch.tensor(normalized_targets, dtype=torch.float32)

        return data_dict