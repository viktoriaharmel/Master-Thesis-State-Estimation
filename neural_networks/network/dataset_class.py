import torch
import numpy as np
import pandas as pd
import os.path

from abc import ABC, abstractmethod

from download_utils import download_dataset


class Dataset(ABC):
    """
    Abstract Dataset Base Class
    """
    def __init__(self, root, download_url=None, force_download=False):
        self.root_path = root
        if download_url is not None:
            dataset_zip_name = download_url[download_url.rfind('/')+1:]
            self.dataset_zip_name = dataset_zip_name
            download_dataset(
                url=download_url,
                data_dir=root,
                dataset_zip_name=dataset_zip_name,
                force_download=force_download,
            )

    @abstractmethod
    def __getitem__(self, index):
        """Return data sample at given index"""

    @abstractmethod
    def __len__(self):
        """Return size of the dataset"""



class CSVDataset(Dataset):
    """
    CSVDataset class.
    """

    def __init__(self, root, target_columns, transform=None, mode="train", input_data=None, root_path=None, dataset_csv_name=None, *args, **kwargs):
        super().__init__(root, *args, **kwargs)

         # Load dataset
        if input_data is not None:
            self.df = input_data
        else:
            data_path = os.path.join(root_path, dataset_csv_name)
            self.df = pd.read_csv(data_path)
            
        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise ValueError("The DataFrame index must be a DatetimeIndex.")

        self.target_columns = target_columns
        assert mode in ["train", "val", "test"], "Invalid mode for dataset given"

        # Add day column for grouping and stratified splitting
        self.df['day'] = self.df.index.date  # Extract date as day identifier

        # Get unique days for each quarter
        unique_days = self.df.groupby(['quarter', 'day']).size().reset_index().drop(columns=0)

        # Split days within each quarter into train, validation, and test
        train_days, val_days, test_days = [], [], []
        for quarter in unique_days['quarter'].unique():
            days_in_quarter = unique_days[unique_days['quarter'] == quarter]['day'].values
            np.random.shuffle(days_in_quarter)  # Shuffle to ensure random selection

            # split the days in the quarter
            train_split = int(0.8 * len(days_in_quarter))
            val_split = int(0.1 * len(days_in_quarter)) + train_split

            train_days.extend(days_in_quarter[:train_split])
            val_days.extend(days_in_quarter[train_split:val_split])
            test_days.extend(days_in_quarter[val_split:])

        # Filter the DataFrame based on the split days
        if mode == "train":
            self.df = self.df[self.df['day'].isin(train_days)]
        elif mode == "val":
            self.df = self.df[self.df['day'].isin(val_days)]
        elif mode == "test":
            self.df = self.df[self.df['day'].isin(test_days)]

        # Drop the temporary 'day' column
        self.df = self.df.drop(columns=['day'])

        # Separate features and targets
        self.data = self.df.drop(columns=self.target_columns)  # Features
        self.targets = self.df[self.target_columns]  # Multiple targets
        
        # Apply data transformation
        self.transforms = transform if transform is not None else lambda x: x


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns a dictionary containing features and all target variables for the given index.
        """
        # Retrieve features and targets for the given index
        data_dict = {
            'features': self.data.iloc[index],
            'targets': self.targets.iloc[index] 
        }

        return self.transforms(data_dict)


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

