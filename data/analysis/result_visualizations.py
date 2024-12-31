import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem, t

# Define paths
base_path = 'path/to/base/directory/'
sh_bh_path = 'path/to/sh_bh/directory/'
sh_bl_path = 'path/to/sh_bl/directory/'

# Mapping dictionary for descriptive names
mapping = {
    base_path: 'Base Scenario',
    sh_bl_path: 'High PV, Low Battery Scenario',
    sh_bh_path: 'High PV, High Battery Scenario',
    'bayesian': 'Bayesian',
    'average': 'Averaged',
    'reac': 'Reactive',
    'real': 'Real'
}

def load_and_filter_results(path, contains_str):
    """Load and filter results based on the target string."""
    df = pd.read_csv(path)
    return df[df['Target'].str.contains(contains_str)]

def calculate_absolute_errors(df):
    """Calculate absolute errors for predictions."""
    return np.abs(df['Target Value'] - df['Prediction'])

def plot_absolute_errors_boxplot(errors, approaches, colors, save_path, title, xlabel, ylabel):
    """Plot a boxplot of absolute errors."""
    boxplot_data = pd.DataFrame({
        'Absolute Errors': np.concatenate(errors),
        'DER Scenario': np.repeat(approaches, [len(data) for data in errors]),
    })

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=boxplot_data, x='DER Scenario', y='Absolute Errors', palette=colors)
    plt.yscale('log')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_path)
    plt.show()

def plot_absolute_errors_histogram(errors, labels, colors, save_path, title, xlabel, ylabel):
    """Plot histograms of absolute errors."""
    bins = np.linspace(0, max([err.max() for err in errors]), 50)
    plt.figure(figsize=(10, 6))

    for err, label, color in zip(errors, labels, colors):
        plt.hist(err, bins=bins, density=True, alpha=0.5, label=label, color=color)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_mean_errors_with_confidence_intervals(errors, save_path, title):
    """Plot mean errors with confidence intervals."""
    mean_errors = errors.mean()
    stderr = sem(errors)
    confidence = 0.95
    degrees_of_freedom = len(errors) - 1
    ci = t.ppf((1 + confidence) / 2, degrees_of_freedom) * stderr

    x = range(len(errors))
    plt.figure(figsize=(18, 6))
    plt.plot(x, mean_errors, marker="o", label="Mean Absolute Errors")
    plt.fill_between(x, errors - ci, errors + ci, color="b", alpha=0.2, label="95% Confidence Interval")
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Absolute Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_forecast_errors(posterior_predictions, observed_values, phase, save_path, title):
    """Plot forecast errors with credible intervals."""
    posterior_errors = posterior_predictions - observed_values
    mean_errors = posterior_errors.mean(axis=0)
    lower_ci = np.percentile(posterior_errors, 2.5, axis=0)
    upper_ci = np.percentile(posterior_errors, 97.5, axis=0)

    p3_cols = [col.split('_')[2] for col in posterior_predictions.columns]
    plt.figure(figsize=(10, 6))
    plt.plot(p3_cols, mean_errors, label="Mean Estimation Error", color="blue")
    plt.fill_between(p3_cols, lower_ci, upper_ci, color="blue", alpha=0.2, label="95% Credible Interval")
    plt.axhline(0, color="black", linestyle="--", linewidth=1, label="Zero Error")
    plt.xticks(rotation=75)
    plt.xlabel("Bus")
    plt.ylabel("Mean Error")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_targets_and_errors(results_base_pawnn, results_base_fc, save_path, title):
    """Plot phase angles and absolute errors for PAWNN and FCNN."""
    # Group data by target to calculate mean values
    grouped_pawnn = results_base_pawnn.groupby("Target").mean()
    indices_pawnn = [idx.split('_')[2] for idx in grouped_pawnn.index]
    true_values_pawnn = grouped_pawnn["Target Value"]
    pawnns_predictions = grouped_pawnn["Prediction"]
    errors_pawnn = np.abs(grouped_pawnn["Target Value"] - grouped_pawnn["Prediction"])

    grouped_fc = results_base_fc.groupby("Target").mean()
    indices_fc = [idx.split('_')[2] for idx in grouped_fc.index]
    true_values_fc = grouped_fc["Target Value"]
    fcs_predictions = grouped_fc["Prediction"]
    errors_fc = np.abs(grouped_fc["Target Value"] - grouped_fc["Prediction"])

    fig, ax1 = plt.subplots(figsize=(20, 6))

    # Plot phase angles
    ax1.plot(indices_pawnn, true_values_pawnn, 'k-', label='True Phase Angle PAWNN', marker='s', markersize=5)
    ax1.plot(indices_fc, true_values_fc, color='grey', linestyle='-', label='True Phase Angle FCNN', marker='s', markersize=5)
    ax1.plot(indices_pawnn, pawnns_predictions, 'b-', label='PAWNN estimate', marker='o', markersize=5)
    ax1.plot(indices_fc, fcs_predictions, 'r-', label='FCNN estimate', marker='*', markersize=5)

    # Configure primary y-axis
    ax1.set_xlabel("Bus")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_ylabel("Phase Angles (normalized)", fontsize=12)
    ax1.legend(loc="upper left")
    ax1.set_xticks(indices_pawnn)
    ax1.set_xticklabels(indices_pawnn, rotation=75)

    # Add secondary y-axis for absolute errors
    ax2 = ax1.twinx()
    ax2.bar(indices_pawnn, errors_pawnn, width=0.4, alpha=0.6, color="lightblue", label="Absolute error PAWNN")
    ax2.bar(indices_fc, errors_fc, width=0.4, alpha=0.6, color="sandybrown", label="Absolute error FCNN")

    ax2.set_ylim(0, 0.16)
    ax2.set_ylabel("Absolute Error", fontsize=12)
    ax2.legend(loc="upper right")

    # Add title
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    # Load data
    results_base_pawnn = load_and_filter_results(sh_bl_path + 'bayesian_test_results.csv', 'v_rea')
    results_base_fc = load_and_filter_results(sh_bh_path + 'bayesian_test_results.csv', 'v_rea')

    # Calculate absolute errors
    errors_base_pawnn = calculate_absolute_errors(results_base_pawnn)
    errors_base_fc = calculate_absolute_errors(results_base_fc)

    # Plot boxplot of absolute errors
    plot_absolute_errors_boxplot(
        [errors_base_pawnn, errors_base_fc],
        ['Solar High, Batteries Low', 'Solar High, Batteries High'],
        ['blue', 'green'],
        '/path/to/save/boxplot.png',
        'Log-Scaled Absolute Test Errors for Voltage Magnitude Predictions under Different DER Penetration Scenarios',
        'DER Scenario',
        'Log Absolute Errors'
    )

    # Plot histograms of absolute errors
    plot_absolute_errors_histogram(
        [errors_base_pawnn, errors_base_fc],
        ['Voltage Magnitude Errors - High PV, Low Battery', 'Voltage Magnitude Errors - High PV, High Battery'],
        ['skyblue', 'lightgreen'],
        '/path/to/save/histogram.png',
        'Comparison of Absolute Voltage Magnitude Prediction Errors Across DER Penetration Scenarios',
        'Absolute Error',
        'Relative Frequency'
    )

    # Plot mean errors with confidence intervals
    be_errors = pd.read_csv(base_path + 'errors_bayesian_samples_loads_real_v2_100.csv').iloc[::300, 0]
    plot_mean_errors_with_confidence_intervals(
        be_errors,
        '/path/to/save/mean_errors.png',
        'Mean Absolute Errors with Confidence Intervals'
    )

    # Plot phase angles and absolute errors
    plot_targets_and_errors(
        results_base_pawnn,
        results_base_fc,
        '/path/to/save/targets_and_abs_errors' + 
        'fname.png',
        'title'
    )
