import numpy as np
import json
import gc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from generation.simulate_data import assign_transformer_limits
from matplotlib import cm

# Define global parameters
name_base = 'path/to/base/directory/'
name_sh_bl = 'path/to/sh_bl/directory/'
name_sh_bh = '/path/to/sh_bh/directory/'

# Load from the JSON file
with open(name_base + "circuit_data.json", "r") as f:
    loaded_data = json.load(f)

# Access the lists
bus_names = loaded_data["bus_names"]
load_names = loaded_data["load_names"]
transformer_names = loaded_data["transformer_names"]

def load_data(file_path, filter_columns=None, step=100):
    """Load and filter data."""
    df = pd.read_csv(file_path)
    if filter_columns:
        df = df[[col for col in df.columns if filter_columns in col and 'target' not in col]]
    return df.iloc[::step, :]

def plot_transformer_power_levels(scenarios, name_base, name_sh_bl, name_sh_bh):
    """Plot transformer power levels normalized against limits."""
    colors = ['blue', 'orange', 'green']
    transformer_power_data = []

    for scenario in scenarios:
        if scenario == 'Base':
            df = load_data(name_base + 'dataset_complete_2018_base_no_pmu_targets.csv', 't_apparent_')
        elif scenario == 'High PV Low Battery':
            df = load_data(name_sh_bl + 'dataset_complete_2018_base_no_pmu_targets.csv', 't_apparent_')
        else:
            df = load_data(name_sh_bh + 'dataset_complete_2018_base_no_pmu_targets.csv', 't_apparent_')

        transformer_limits = assign_transformer_limits(np.abs(df))
        normalized_power = (np.abs(df) / transformer_limits)

        transformer_power_data.append({
            'Scenario': scenario,
            'Normalized Power Levels': np.concatenate(normalized_power.values),
        })

    boxplot_data = pd.DataFrame({
        'Normalized Power Levels': np.concatenate([data['Normalized Power Levels'] for data in transformer_power_data]),
        'Scenario': np.repeat(scenarios, [len(data['Normalized Power Levels']) for data in transformer_power_data]),
    })

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=boxplot_data, x='Scenario', y='Normalized Power Levels', palette=colors)
    plt.title("Transformer Power Levels (Normalized) Across Scenarios")
    plt.ylabel("Normalized Power Levels (Relative to Limit)")
    plt.xlabel("Scenario")
    plt.axhline(y=1.0, color='red', linestyle='--', label='100% Utilization (Limit)')
    plt.legend()
    plt.savefig(name_base + 'seasonality_analysis/' + 'Transformer Power Levels (Normalized) Across Scenarios.png')
    plt.show()

def analyze_critical_lines(scenarios, name_base, name_sh_bl, name_sh_bh, utilization_threshold=0.9, top_n_lines=10):
    """Identify and plot critical lines across scenarios."""
    critical_lines = {scenario: [] for scenario in scenarios}
    base_colors = {'Base': 'Blues', 'High PV Low Battery': 'Oranges', 'High PV High Battery': 'Greens'}

    for scenario in scenarios:
        if scenario == 'Base':
            df = load_data(name_base + 'dataset_complete_2018_base_no_pmu_targets.csv', 'current_magnitude_')
        elif scenario == 'High PV Low Battery':
            df = load_data(name_sh_bl + 'dataset_complete_2018_base_no_pmu_targets.csv', 'current_magnitude_')
        else:
            df = load_data(name_sh_bh + 'dataset_complete_2018_base_no_pmu_targets.csv', 'current_magnitude_')

        for current_col in df.columns:
            phase = int(current_col.split('_')[-1][-1])
            if phase == 1:
                max_current = df[current_col].max()
                utilization = df[current_col] / max_current
                critical_utilizations = (utilization > utilization_threshold).sum()
                var = np.var(df[current_col])
                critical_lines[scenario].append((current_col, critical_utilizations, var))

    for scenario in scenarios:
        critical_lines[scenario] = sorted(
            critical_lines[scenario],
            key=lambda x: (x[1], x[2]),
            reverse=True
        )[:top_n_lines]

    fig, axes = plt.subplots(1, len(scenarios) + 1, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 2, 2, 0.7]}, sharey=True)
    for ax, scenario in zip(axes[:-1], scenarios):
        df = load_data(
            name_base if scenario == 'Base' else name_sh_bl if scenario == 'High PV Low Battery' else name_sh_bh,
            'current_magnitude_'
        )
        colormap = cm.get_cmap(base_colors[scenario])
        restricted_colormap = colormap(np.linspace(0.5, 1.5, len(critical_lines[scenario])))
        
        for idx, (col, _, _) in enumerate(critical_lines[scenario]):
            if col in df.columns:
                ax.plot(df.index, df[col], label=col, color=restricted_colormap[idx], alpha=0.7)
        
        ax.set_title(f"{scenario} Scenario")
        ax.set_xlabel("Time Step")
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    
    fig.suptitle("Current Magnitudes Across Critical Lines for Different Scenarios", fontsize=16, fontweight='bold')
    fig.supylabel("Current Magnitude (A)", x=0)
    handles, labels = [], []
    for ax in axes[:-1]:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    legend_ax = axes[-1]
    legend_ax.axis('off')
    legend_ax.legend(handles, labels, loc='center', fontsize='small', ncol=1)

    plt.savefig(name_base + 'seasonality_analysis/' + 'Current Magnitudes by Scenario Subplots.png')
    plt.show()

def plot_voltage_differences(name_sh_bl, name_sh_bh, bus_names):
    """Plot differences in voltage values between scenarios."""
    sh_bl_df = load_data(name_sh_bl + 'dataset_complete_2018_base_no_pmu_targets.csv', 'v_real_')
    sh_bh_df = load_data(name_sh_bh + 'dataset_complete_2018_base.csv', 'v_real_')

    sh_bl_v_real = np.concatenate([sh_bl_df[col].values for col in sh_bl_df.columns])
    sh_bh_v_real = np.concatenate([sh_bh_df[col].values for col in sh_bh_df.columns])
    diff_bh_bl = sh_bh_v_real - sh_bl_v_real

    plt.figure(figsize=(10, 6))
    sns.histplot(diff_bh_bl, bins=100, color='green', kde=False, label="High PV High Battery - High PV Low Battery", stat="density")
    plt.title("Active Voltage Differences Between Scenarios")
    plt.xlabel("Active Voltage Difference (V)")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

def plot_voltage_histogram(scenarios, colors, name_base, name_sh_bl, name_sh_bh, bus_names):
    """Plot histogram of voltage magnitudes across scenarios."""
    plt.figure(figsize=(10, 6))
    
    for scenario, color in zip(scenarios, colors):
        if scenario == 'Base':
            df = load_data(name_base + 'dataset_complete_2018_base_no_pmu_targets.csv', 'v_')
        elif scenario == 'High PV Low Battery':
            df = load_data(name_sh_bl + 'dataset_complete_2018_base_no_pmu_targets.csv', 'v_')
        else:
            df = load_data(name_sh_bh + 'dataset_complete_2018_base.csv', 'v_')
        
        # Initialize a list to collect all voltage magnitudes for this scenario
        voltage_magnitudes = []
        
        # Iterate over all buses and phases
        for bus in bus_names:  # Unique bus names
            for phase in [1, 2, 3]:  # 3 phases per bus
                # Construct real and reactive voltage column names dynamically
                v_real_col = f'v_real_{bus}_phase{phase}'
                v_reac_col = f'v_reac_{bus}_phase{phase}'
                
                # Check if the columns exist in the dataset
                if v_real_col in df.columns and v_reac_col in df.columns:
                    # Calculate voltage magnitudes for this bus and phase
                    v_real = df[v_real_col].values
                    v_reac = df[v_reac_col].values
                    mag = np.sqrt(v_real**2 + v_reac**2)
                    # Filter to desired voltage range
                    voltage_magnitudes.extend(mag[(mag <= 300) & (mag >= 100)])
        
        # Plot histogram for this scenario
        sns.histplot(voltage_magnitudes, bins=50, color=color, kde=False, label=scenario, stat="density")
    
    # Finalize plot
    plt.title("High Voltage Magnitudes Across Scenarios")
    plt.xlabel("Voltage Magnitude (V)")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(name_base + 'seasonality_analysis/' + 'Voltage Magnitudes Across Scenarios.png')
    plt.show()


if __name__ == "__main__":
    scenarios = ['Base', 'High PV Low Battery', 'High PV High Battery']
    plot_transformer_power_levels(scenarios, name_base, name_sh_bl, name_sh_bh)
    analyze_critical_lines(scenarios, name_base, name_sh_bl, name_sh_bh)
    plot_voltage_histogram(['Base', 'High PV High Battery'], ['blue', 'green'], name_base, name_sh_bl, name_sh_bh, bus_names)
    plot_voltage_differences(name_sh_bl, name_sh_bh, bus_names)
