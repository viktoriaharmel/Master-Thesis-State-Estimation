import numpy as np
import pandas as pd
import csv
import time
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error



def train_test_split(data, t_idx):
    ''' Split the data into training and testing sets. '''

    # Add day column for grouping and stratified splitting
    data['day'] = data.index.date  # Extract date as day identifier

    # Get unique days for each quarter
    unique_days = data.groupby(['quarter', 'day']).size().reset_index().drop(columns=0)

    # Split days within each quarter into train and test
    train_days, test_days = [], []
    for quarter in unique_days['quarter'].unique():
        days_in_quarter = unique_days[unique_days['quarter'] == quarter]['day'].values
        np.random.shuffle(days_in_quarter)  # Shuffle to ensure random selection

        # Split by quarter
        train_split = int(t_idx * len(days_in_quarter))
        
        train_days.extend(days_in_quarter[:train_split])
        test_days.extend(days_in_quarter[train_split:])
    
    train_data = data[data['day'].isin(train_days)]
    test_data = data[data['day'].isin(test_days)]

    # Drop the temporary 'day' column
    train_data = train_data.drop(columns=['day'])
    test_data = test_data.drop(columns=['day'])

    return train_data, test_data


def error_from_max(y, y_lin):
    ''' Calculate the maximum error as a percentage of the maximum value in the target. '''
    return np.max(np.abs(y - y_lin) / np.max(np.abs(y))) * 100


def normalize_data(data, target_cols):
    ''' Normalize the data between 0 and 1. '''
    def normalize_column(value, column_name):
            mn = data[column_name].min()
            mx = data[column_name].max()
            divisor = mx - mn
            if divisor == 0:
                divisor = 1e-6
            return (value - mn) / divisor
        
    def normalize_angle_column(value):
        value_norm = value / np.pi  # For radians
        return value_norm
    
    data_norm = pd.DataFrame()

    # Normalize feature columns
    for column in data.columns:
        # Skip temporal indicators and target columns
        if column not in target_cols and column not in ['bias', 'hour', 'day_of_week', 'is_weekend', 'day_of_month', 'month', 'quarter', 'is_holiday', 'season']:

            if any(np.isnan(data[column])):
                mean_col_val = data[column].mean(ignore_nan=True)
                data_norm[column] = mean_col_val

            # Normalize the column value
            if 'angle' in column:
                data_norm[column] = normalize_angle_column(data[column])
            else:
                data_norm[column] = normalize_column(data[column], column)
        if column in ['bias', 'hour', 'day_of_week', 'is_weekend', 'day_of_month', 'month', 'quarter', 'is_holiday', 'season']:
            data_norm[column] = data[column]

    # Normalize each target column
    for target_col in target_cols:
        data_norm[target_col] = normalize_column(data[target_col], target_col)

    return data_norm

def restore_data(value, column_name):
    ''' Restore the normalized data to its original scale. '''
    if 'angle' in column_name:
        return value * np.pi
    mn = data[column_name].min()
    mx = data[column_name].max()
    divisor = mx - mn
    if divisor == 0:
        divisor = 1e-6
    return (value + mn) * divisor


def train_lin_models(X_train, y_train):
    ''' Train linear models for all targets. '''
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)

    model_l = LinearRegression(fit_intercept=True)

    y_lin = model_l.fit(X_train, y_train).predict(X_train)
    
    mse = mean_squared_error(y_train, y_lin)
    mae = mean_absolute_error(y_train, y_lin)

    if np.mean(np.abs(y_train)) > 1e-5:
        error_max = error_from_max(y_train, y_lin)
    else:
        print('values are too close to 0')

    # Coefficients for all targets
    coefs_mat = model_l.coef_  # Shape: (n_targets, n_features)
    intercept_mat = model_l.intercept_  # Shape: (n_targets,)

    return y_lin, coefs_mat, intercept_mat, error_max, mse, mae



def test_models(x, y, coefs, intercept):
    ''' Test the linear models on the test set. '''
    y_pred = np.dot(x, coefs.T) + intercept

    if np.mean(np.abs(y)) > 1e-5:
        errors_all = error_from_max(y, y_pred)
    else:
        print('values are too close to 0')

    # Calculate MSE and MAE for this target
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    return errors_all, y_pred, mse, mae



if __name__ == '__main__':
    name = 'path/to/data/'
    method = 'average' # method for estimating missing data
    scenario = 'sh_bh' # DER scenario
    t_idx = 0.8 # train-test split index

    # load dataset
    data = pd.read_csv(name + f'dataset_{method}_2018_no_pmu_targets.csv', parse_dates=True)
    # load target columns
    with open(name + 'target_cols.csv', 'r') as file:
        reader = csv.reader(file)
        target_columns = next(reader)  # Read the first row
    
    data.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'], inplace=True)
    
    start_time = datetime(year=2018, month=1, day=1, hour=0, minute=0)
    end_time = datetime(year=2018, month=12, day=31, hour=23, minute=45)
    time_index = pd.date_range(start=start_time, end=end_time, freq='15min')

    data.index = time_index

    # Ensure the index is a DatetimeIndex
    data.index = pd.to_datetime(data.index)

    data_norm = normalize_data(data, target_columns)

    feature_columns = [col for col in data.columns if col not in target_columns]

    train_data, test_data = train_test_split(data_norm, t_idx)

    # Separating features and targets for training and testing data
    X_train = train_data.drop(columns=target_columns)
    y_train = train_data.drop(columns=feature_columns)
    X_test = test_data.drop(columns=target_columns)
    y_test = test_data.drop(columns=feature_columns)

    train_start = time.time()
    y_train_pred, coefs_mat, intercept_mat, error, mse, mae = train_lin_models(X_train, y_train)
    train_end = time.time()
    training_time = train_end - train_start

    print(f'Training Runtime: {training_time:.6f} s')

    training_performance = {
        "Error": [error],
        "MSE": [mse],
        "MAE": [mae]
    }

    # test grid model errors on test set
    test_start = time.time()
    error_test, y_test_pred, mse_test, mae_test = test_models(X_test, y_test, coefs_mat, intercept_mat)
    test_end = time.time()
    test_time = test_end - test_start

    print(f'Testing Runtime: {test_time:.6f} s')

    testing_performance = {
        "Error": [error_test],
        "MSE": [mse_test],
        "MAE": [mae_test]
    }

    training_df = pd.DataFrame(training_performance)
    testing_df = pd.DataFrame(testing_performance)

    performance_df = pd.concat([training_df, testing_df], keys=["Training", "Testing"])

    # Save to CSV
    performance_df.to_csv(name + f'linear_model_performance_{method}_{scenario}.csv', index=True)

    print('Saved model performance')

    # Save model coefficients and predictions
    np.savez(name + f'linear_model_coefs_{method}_{scenario}.npz', coefs_mat=coefs_mat, intercept_mat=intercept_mat)
    print('Saved model coefs')

    # optionally restore the data to its original scale
    y_train_pred_restored = []
    y_test_pred_restored = []
    for i, col in enumerate(target_columns):
        y_train_pred_restored.append(restore_data(y_train_pred[i], col))
        y_test_pred_restored.append(restore_data(y_test_pred[i], col))

    np.savez(name + f'linear_model_preds_{method}_{scenario}.npz', y_train=y_train, y_train_pred=y_train_pred, y_test=y_test, y_test_pred=y_test_pred)
    
