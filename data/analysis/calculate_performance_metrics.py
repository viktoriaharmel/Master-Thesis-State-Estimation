import pandas as pd
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred):
    mask = y_true != 0  # Exclude zero values in y_true
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def r2_score(y_true, y_pred):
    y_true_mean = np.mean(y_true)
    ss_total = np.sum((y_true - y_true_mean) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

# Load the results file - Uncomment the appropriate line based on the file format (npz for linear results and csv for neural network results)
#path = 'path/to/results.npz'
path = 'path/to/results.csv'

#data = np.load(path, allow_pickle=True)
#y_test = data['y_test']
#y_test_pred = data['y_test_pred']
results = pd.read_csv(path)

#print('MSE: ' + str(mean_squared_error(y_test, y_test_pred)))
#print('MAPE: ' + str(mean_absolute_percentage_error(y_test, y_test_pred)))
#print('R2: ' + str(r2_score(y_test, y_test_pred)))

print('MSE: ' + str(mean_squared_error(results['Target Value'], results['Prediction'])))
print('MAPE: ' + str(mean_absolute_percentage_error(results['Target Value'], results['Prediction'])))
print('R2: ' + str(r2_score(results['Target Value'], results['Prediction'])))

