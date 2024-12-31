import numpy as np
import pandas as pd
import json
import gc
from tslearn.clustering import TimeSeriesKMeans
from sklearn.mixture import GaussianMixture
from tslearn.metrics import dtw
from scipy.spatial.distance import euclidean
from scipy.stats import multivariate_t
from scipy.spatial.distance import cdist

data_name = 'path/to/data/'
data_name_last_year = 'path/to/data/last_year/'

# Load complex voltage and phase angle data
voltage_data_p1 = np.load(data_name + "local_metrics_complex_volts_angles_p1.npz", allow_pickle=True)["v_complex"].item()
angle_data_p1 = np.load(data_name + "local_metrics_complex_volts_angles_p1.npz", allow_pickle=True)["v_angles"].item()
last_year_voltage_data_p1 = np.load(data_name_last_year + "local_metrics_complex_volts_angles_p1.npz", allow_pickle=True)["v_complex"].item()
last_year_angle_data_p1 = np.load(data_name_last_year + "local_metrics_complex_volts_angles_p1.npz", allow_pickle=True)["v_angles"].item()

voltage_df = pd.DataFrame.from_dict(voltage_data_p1, orient='index')
angle_df = pd.DataFrame.from_dict(angle_data_p1, orient='index')
last_year_voltage_df = pd.DataFrame.from_dict(last_year_voltage_data_p1, orient='index')
last_year_angle_df = pd.DataFrame.from_dict(last_year_angle_data_p1, orient='index')

partial_files = [
        "local_metrics_complex_volts_angles_p2.npz",
        "local_metrics_complex_volts_angles_p3.npz",
        "local_metrics_complex_volts_angles_p4.npz",
        "local_metrics_complex_volts_angles_p5.npz",
        "local_metrics_complex_volts_angles_p6.npz",
        "local_metrics_complex_volts_angles_p7.npz",
        "local_metrics_complex_volts_angles_p8.npz",
        "local_metrics_complex_volts_angles_p9.npz",
        "local_metrics_complex_volts_angles_p10.npz"
    ]

def combine_partial_files(partial_files, base_df, path, data):

    # Iterate through partial files and append their columns to voltage_df
    for file in partial_files:
        partial_data = np.load(path + file, allow_pickle=True)[data].item()
        partial_df = pd.DataFrame.from_dict(partial_data, orient='index')
        
        # Concatenate the partial DataFrame with the base DataFrame along columns
        base_df = pd.concat([base_df, partial_df], axis=1)
    
    return base_df

voltage_df = combine_partial_files(partial_files, voltage_df, data_name, 'v_complex')
last_year_voltage_df = combine_partial_files(partial_files, last_year_voltage_df, data_name_last_year, 'v_complex')
angle_df = combine_partial_files(partial_files, angle_df, data_name, 'v_angles')
last_year_angle_df = combine_partial_files(partial_files, last_year_angle_df, data_name_last_year, 'v_angles')

del voltage_data_p1
del angle_data_p1
del last_year_voltage_data_p1
del last_year_angle_data_p1
gc.collect()

def seperate_data(df):
    """ Split the data into real and reactive components for each phase """
    # Initialize lists to store the split data
    real_phase1, reactive_phase1 = {}, {}
    real_phase2, reactive_phase2 = {}, {}
    real_phase3, reactive_phase3 = {}, {}

    # Iterate over each cell in the DataFrame
    for bus, values in df.iterrows():
        real_phase1[bus] = []
        reactive_phase1[bus] = []
        real_phase2[bus] = []
        reactive_phase2[bus] = []
        real_phase3[bus] = []
        reactive_phase3[bus] = []
        for cell in values:
            cell = list(cell)
            # Extract real and reactive components for each phase
            if len(cell) < 6:
                cell = cell + [np.nan] * (6 - len(cell))
            real_phase1[bus].append(cell[0])
            reactive_phase1[bus].append(cell[1])
            real_phase2[bus].append(cell[2])
            reactive_phase2[bus].append(cell[3])
            real_phase3[bus].append(cell[4])
            reactive_phase3[bus].append(cell[5])

    # Convert lists to DataFrames
    real_phase1_df = pd.DataFrame.from_dict(real_phase1, orient='index')
    reactive_phase1_df = pd.DataFrame.from_dict(reactive_phase1, orient='index')
    real_phase2_df = pd.DataFrame.from_dict(real_phase2, orient='index')
    reactive_phase2_df = pd.DataFrame.from_dict(reactive_phase2, orient='index')
    real_phase3_df = pd.DataFrame.from_dict(real_phase3, orient='index')
    reactive_phase3_df = pd.DataFrame.from_dict(reactive_phase3, orient='index')

    return real_phase1_df, real_phase2_df, real_phase3_df, reactive_phase1_df, reactive_phase2_df, reactive_phase3_df

voltages_real_phase1_df, voltages_real_phase2_df, voltages_real_phase3_df, voltages_reac_phase1_df, voltages_reac_phase2_df, voltages_reac_phase3_df = seperate_data(voltage_df) 
angles_real_phase1_df, angles_real_phase2_df, angles_real_phase3_df, angles_reac_phase1_df, angles_reac_phase2_df, angles_reac_phase3_df = seperate_data(angle_df) 
last_year_voltages_real_phase1_df, last_year_voltages_real_phase2_df, last_year_voltages_real_phase3_df, last_year_voltages_reac_phase1_df, last_year_voltages_reac_phase2_df, last_year_voltages_reac_phase3_df = seperate_data(last_year_voltage_df) 
last_year_angles_real_phase1_df, last_year_angles_real_phase2_df, last_year_angles_real_phase3_df, last_year_angles_reac_phase1_df, last_year_angles_reac_phase2_df, last_year_angles_reac_phase3_df = seperate_data(last_year_angle_df) 

del voltage_df
del last_year_voltage_df
del angle_df
del last_year_angle_df
gc.collect()

pmu_locations = ['p1rdt4659.1.2.3', 'p1rdt4949.1.2.3', 'p1rdt833_b1_1.1', 'p1rdt1317_b1_1.1.2.3', 'p1rdt1175.3', 'p1rdm6643.1.2', 'p1rdt8002.1.2.3', 'p1rdt7077.1.2.3', 'p1rdt4568lv.1.2', 'p1rdt2829.1.2.3', 'p1rdt4396lv.1.2', 'p1rdt3614-p1rdt4656xx.1.2.3', 'p1rdt4879_b2_1.1', 'p1rdt5032-p1rdt6794x_b1_1.3', 'p1rdt5663.1.2.3', 'p1rdt6284lv.1.2', 'p1rdt5033_b1_1.1.2.3', 'p1rdt4730_b1_1.1', 'p1rdt1069_b2_1.1.2.3', 'p1rdt3810-p1rdt7137x_b1_1.1', 'p1rdt6285-p1rdt7899x.2', 'p1rdt6852.1', 'p1rdt4396-p1rdt831x_b1_1.1', 'p1rdt7136.3', 'p1rdt4401_b1_1.3', 'p1rdt6284.1.2.3', 'p1rdt320.2', 'p1rdm9214.1.2', 'p1rdt5351.3', 'p1rdt1318-p1rdt3520x_b1_1.2', 'p1rdt4728.2', 'p1rdm1612.1.2', 'p1rdt1762.1.2.3', 'p1rdm117.1.2', 'p1rdm759.1.2', 'p1rdt319lv.1.2', 'p1rdt7137.1.2.3', 'p1rdt7136lv.1.2', 'p1rdm7296.1.2', 'p1rdt3243-p1rdt5935x_b1_1.2', 'p1rdt3243.1', 'p1rdt3243-p1rdt8281x_b1_1.1', 'p1rdt1436.1.2.3', 'p1rdt7437_b1_1.1', 'p1rdt4949.2', 'p1rdt830.2', 'p1rdt5031-p1rdt7257x_b1_1.3', 'p1rdt2109-p1rdt833x_b1_1.1', 'p1rdt6853.1', 'p1rdt2110-p1rdt963x_b1_1.2', 'p1rdm12169.1.2', 'p1rdm6959.1.2', 'p1rdt2630-p1rdt831x.3', 'p1rdt2925-p1rdt3614x.3', 'p1rdt2109.1.2.3', 'p1rdt1436lv.1.2', 'p1rdt2830.1.2.3', 'p1rdt441.1', 'p1rdm7458.1.2', 'p1rdm123.1.2', 'p1rdt3616.1.2.3', 'p1rdt1869-p1rdt53x.2', 'p1rdt1761-p1rdt3616x.2', 'p1rdt1760-p1rdt7325x_b2_1.2', 'p1rdt3614-p1rdt7257xx.1.2.3', 'p1rdt834.1.2.3', 'p1rdt1070.1.2.3', 'p1rdm5898.1.2', 'p1rdm9429.1.2', 'p1rdt1318_b2_1.1.2.3', 'p1rdm4531.1.2', 'p1rdt963lv.1.2', 'p1rdt7265_b1_1.1', 'p1rdt5360-p1rdt565x.2']
pmu_locations = [pmu_loc.split('.')[0] for pmu_loc in pmu_locations]

# Load from the JSON file
with open(data_name + "circuit_data.json", "r") as f:
    loaded_data = json.load(f)

# Access the lists
bus_names = loaded_data["bus_names"]

# Extract indices of observable buses
observable_bus_indices = [i for i, bus in enumerate(bus_names) if bus in pmu_locations]

# Extract indices of unobservable buses
unobservable_bus_indices = [i for i, bus in enumerate(bus_names) if bus not in pmu_locations]

data_dict = {}

# Split the data into observable and unobservable buses
data_dict['last_year_obs_voltages_real_phase1'], data_dict['last_year_obs_voltages_real_phase2'], data_dict['last_year_obs_voltages_real_phase3'], data_dict['last_year_obs_voltages_reac_phase1'], data_dict['last_year_obs_voltages_reac_phase2'], data_dict['last_year_obs_voltages_reac_phase3'] = last_year_voltages_real_phase1_df.iloc[observable_bus_indices], last_year_voltages_real_phase2_df.iloc[observable_bus_indices], last_year_voltages_real_phase3_df.iloc[observable_bus_indices], last_year_voltages_reac_phase1_df.iloc[observable_bus_indices], last_year_voltages_reac_phase2_df.iloc[observable_bus_indices], last_year_voltages_reac_phase3_df.iloc[observable_bus_indices]
data_dict['last_year_unobs_voltages_real_phase1'], data_dict['last_year_unobs_voltages_real_phase2'], data_dict['last_year_unobs_voltages_real_phase3'], data_dict['last_year_unobs_voltages_reac_phase1'], data_dict['last_year_unobs_voltages_reac_phase2'], data_dict['last_year_unobs_voltages_reac_phase3'] = last_year_voltages_real_phase1_df.iloc[unobservable_bus_indices], last_year_voltages_real_phase2_df.iloc[unobservable_bus_indices], last_year_voltages_real_phase3_df.iloc[unobservable_bus_indices], last_year_voltages_reac_phase1_df.iloc[unobservable_bus_indices], last_year_voltages_reac_phase2_df.iloc[unobservable_bus_indices], last_year_voltages_reac_phase3_df.iloc[unobservable_bus_indices]
data_dict['obs_voltages_real_phase1'], data_dict['obs_voltages_real_phase2'], data_dict['obs_voltages_real_phase3'], data_dict['obs_voltages_reac_phase1'], data_dict['obs_voltages_reac_phase2'], data_dict['obs_voltages_reac_phase3'] = voltages_real_phase1_df.iloc[observable_bus_indices], voltages_real_phase2_df.iloc[observable_bus_indices], voltages_real_phase3_df.iloc[observable_bus_indices], voltages_reac_phase1_df.iloc[observable_bus_indices], voltages_reac_phase2_df.iloc[observable_bus_indices], voltages_reac_phase3_df.iloc[observable_bus_indices]
data_dict['unobs_voltages_real_phase1'], data_dict['unobs_voltages_real_phase2'], data_dict['unobs_voltages_real_phase3'], data_dict['unobs_voltages_reac_phase1'], data_dict['unobs_voltages_reac_phase2'], data_dict['unobs_voltages_reac_phase3'] = voltages_real_phase1_df.iloc[unobservable_bus_indices], voltages_real_phase2_df.iloc[unobservable_bus_indices], voltages_real_phase3_df.iloc[unobservable_bus_indices], voltages_reac_phase1_df.iloc[unobservable_bus_indices], voltages_reac_phase2_df.iloc[unobservable_bus_indices], voltages_reac_phase3_df.iloc[unobservable_bus_indices]
data_dict['last_year_obs_angles_real_phase1'], data_dict['last_year_obs_angles_real_phase2'], data_dict['last_year_obs_angles_real_phase3'], data_dict['last_year_obs_angles_reac_phase1'], data_dict['last_year_obs_angles_reac_phase2'], data_dict['last_year_obs_angles_reac_phase3'] = last_year_angles_real_phase1_df.iloc[observable_bus_indices], last_year_angles_real_phase2_df.iloc[observable_bus_indices], last_year_angles_real_phase3_df.iloc[observable_bus_indices], last_year_angles_reac_phase1_df.iloc[observable_bus_indices], last_year_angles_reac_phase2_df.iloc[observable_bus_indices], last_year_angles_reac_phase3_df.iloc[observable_bus_indices]
data_dict['last_year_unobs_angles_real_phase1'], data_dict['last_year_unobs_angles_real_phase2'], data_dict['last_year_unobs_angles_real_phase3'], data_dict['last_year_unobs_angles_reac_phase1'], data_dict['last_year_unobs_angles_reac_phase2'], data_dict['last_year_unobs_angles_reac_phase3'] = last_year_angles_real_phase1_df.iloc[unobservable_bus_indices], last_year_angles_real_phase2_df.iloc[unobservable_bus_indices], last_year_angles_real_phase3_df.iloc[unobservable_bus_indices], last_year_angles_reac_phase1_df.iloc[unobservable_bus_indices], last_year_angles_reac_phase2_df.iloc[unobservable_bus_indices], last_year_angles_reac_phase3_df.iloc[unobservable_bus_indices]
data_dict['obs_angles_real_phase1'], data_dict['obs_angles_real_phase2'], data_dict['obs_angles_real_phase3'], data_dict['obs_angles_reac_phase1'], data_dict['obs_angles_reac_phase2'], data_dict['obs_angles_reac_phase3'] = angles_real_phase1_df.iloc[observable_bus_indices], angles_real_phase2_df.iloc[observable_bus_indices], angles_real_phase3_df.iloc[observable_bus_indices], angles_reac_phase1_df.iloc[observable_bus_indices], angles_reac_phase2_df.iloc[observable_bus_indices], angles_reac_phase3_df.iloc[observable_bus_indices]
data_dict['unobs_angles_real_phase1'], data_dict['unobs_angles_real_phase2'], data_dict['unobs_angles_real_phase3'], data_dict['unobs_angles_reac_phase1'], data_dict['unobs_angles_reac_phase2'], data_dict['unobs_angles_reac_phase3'] = angles_real_phase1_df.iloc[unobservable_bus_indices], angles_real_phase2_df.iloc[unobservable_bus_indices], angles_real_phase3_df.iloc[unobservable_bus_indices], angles_reac_phase1_df.iloc[unobservable_bus_indices], angles_reac_phase2_df.iloc[unobservable_bus_indices], angles_reac_phase3_df.iloc[unobservable_bus_indices]

del voltages_real_phase1_df, voltages_real_phase2_df, voltages_real_phase3_df, voltages_reac_phase1_df, voltages_reac_phase2_df, voltages_reac_phase3_df
del last_year_voltages_real_phase1_df, last_year_voltages_real_phase2_df, last_year_voltages_real_phase3_df, last_year_voltages_reac_phase1_df, last_year_voltages_reac_phase2_df, last_year_voltages_reac_phase3_df
del angles_real_phase1_df, angles_real_phase2_df, angles_real_phase3_df, angles_reac_phase1_df, angles_reac_phase2_df, angles_reac_phase3_df
del last_year_angles_real_phase1_df, last_year_angles_real_phase2_df, last_year_angles_real_phase3_df, last_year_angles_reac_phase1_df, last_year_angles_reac_phase2_df, last_year_angles_reac_phase3_df
gc.collect()

# Define minimum variance threshold to avoid singular matrices
MIN_VARIANCE_THRESHOLD = 1e-34

# Define fixed parameters for the sampling function
fixed_params = {
    'num_clusters': 4,  # Adjust based on data size
    'distance_metric': 'euclidean',
    'distance_metric_clustering': 'euclidean',
    'multivariate_distribution': 't_distribution',
    'n_components': None,
    'degrees_of_freedom': 4,
}

# Function to calculate medoid of a cluster
def calculate_medoid(cluster_data):
    distances = cdist(cluster_data, cluster_data, metric='euclidean')
    return np.argmin(distances.sum(axis=1))

def merge_small_clusters(clusters, data, min_samples=2):
    """
    Merges clusters with fewer than `min_samples` into the nearest larger cluster, using medoids.

    Parameters:
    - clusters: Dictionary where keys are cluster labels and values are lists of data indices in each cluster.
    - data: The original data used for clustering (2D array or DataFrame).
    - min_samples: Minimum number of samples required in a cluster.

    Returns:
    - merged_clusters: Updated cluster dictionary after merging small clusters.
    """
    # Step 1: Calculate medoids for each cluster
    medoids = {}
    for cluster_label, indices in clusters.items():
        cluster_data = data.loc[indices].values
        medoid_index = calculate_medoid(cluster_data)
        medoids[cluster_label] = cluster_data[medoid_index]

    # Step 2: Identify small clusters
    small_clusters = [label for label, indices in clusters.items() if len(indices) < min_samples]

    # Step 3: Merge small clusters into the nearest larger clusters
    for small_cluster in small_clusters:
        # Calculate distances to all other cluster medoids
        other_clusters = {label: medoid for label, medoid in medoids.items() if
                          label != small_cluster and len(clusters[label]) >= min_samples}
        if not other_clusters:
            continue  # If no other large clusters exist, skip this small cluster

        # Find the nearest large cluster using the medoids
        other_medoids = np.array(list(other_clusters.values()))
        other_labels = list(other_clusters.keys())
        small_cluster_medoid = medoids[small_cluster].reshape(1, -1)
        distances = cdist(small_cluster_medoid, other_medoids, metric='euclidean')
        nearest_cluster = other_labels[np.argmin(distances)]

        # Merge small cluster into the nearest large cluster
        clusters[nearest_cluster].extend(clusters[small_cluster])  # Reassign samples
        del clusters[small_cluster]  # Remove the small cluster
        del medoids[small_cluster]
        print(f"Merged cluster {small_cluster} into cluster {nearest_cluster} due to insufficient samples.")

    # Step 4: Recompute medoids after merging
    merged_medoids = {}
    for label, indices in clusters.items():
        cluster_data = data.loc[indices].values
        medoid_index = calculate_medoid(cluster_data)
        merged_medoids[label] = cluster_data[medoid_index]

    return clusters, merged_medoids

# Clustering function
def cluster_observable_buses(observable_data, num_clusters, distance_metric):
    clustering_model = TimeSeriesKMeans(n_clusters=num_clusters, metric=distance_metric, random_state=0)
    cluster_labels = clustering_model.fit_predict(observable_data)
    clusters = {i: [] for i in range(num_clusters)}
    for idx, cluster_label in enumerate(cluster_labels):
        bus = observable_data.index[idx]
        clusters[cluster_label].append(bus)
    return clusters


def bayesian_update_multivariate(prior_mean, prior_cov, observed_mean, observed_cov):
    """ Perform Bayesian update for multivariate distributions. """
    
    kalman_gain = np.eye(len(prior_cov))
    try:
        kalman_gain = prior_cov @ np.linalg.pinv(prior_cov + observed_cov)
    except np.linalg.LinAlgError:
        kalman_gain = np.eye(len(prior_cov))  # Fallback if inversion fails
    posterior_mean = prior_mean + kalman_gain @ (observed_mean - prior_mean)
    posterior_cov = (np.eye(len(prior_cov)) - kalman_gain) @ prior_cov
    return posterior_mean, posterior_cov

# Sampling function
def run_sampling(params, unobservable_bus_indices, obs_values, last_year_unobs_values, last_year_obs_values, element_name, phase_name, type, method="bayesian"):
    """ Run sampling for unobservable buses based on observable buses. """
    
    num_clusters = params['num_clusters']
    distance_metric = params['distance_metric']
    multivariate_distribution = params['multivariate_distribution']
    n_components = params['n_components']
    degrees_of_freedom = params['degrees_of_freedom']

    # Initialize DataFrame to store sampled results
    sampled_results = pd.DataFrame(index=[f'{element_name}_{type}_{bus_names[i]}_phase{phase_name}' for i in unobservable_bus_indices], columns=range(last_year_unobs_values.shape[1]))

    obs_values.dropna(inplace=True)
    last_year_unobs_values.dropna(inplace=True)
    last_year_obs_values.dropna(inplace=True)

    if last_year_obs_values.empty:
        return sampled_results

    # Cluster observable buses
    clusters = cluster_observable_buses(last_year_obs_values, num_clusters, distance_metric)

    clusters, _ = merge_small_clusters(clusters, last_year_obs_values, min_samples=2)
    num_clusters = len(clusters)
    params['num_clusters'] = num_clusters

    for bus, values in last_year_unobs_values.iterrows():
        # Find best cluster
        min_distance = float('inf')
        best_cluster = None
        for cluster_label, buses in clusters.items():
            cluster_data = last_year_obs_values.loc[buses].values
            mean_vector = np.mean(cluster_data, axis=0)
            distance = euclidean(values, mean_vector) if distance_metric == 'euclidean' else dtw(values, mean_vector)
            if distance < min_distance:
                min_distance = distance
                best_cluster = cluster_label

        # Initialize posterior distribution with the best cluster
        cluster_buses = clusters[best_cluster]
        initial_mean = np.mean(last_year_obs_values.loc[cluster_buses].values, axis=0)
        initial_cov = np.cov(last_year_obs_values.loc[cluster_buses].values, rowvar=False) + np.eye(last_year_obs_values.loc[cluster_buses].values.shape[1]) * MIN_VARIANCE_THRESHOLD
        posterior_distribution = {'mean': initial_mean, 'cov': initial_cov}

        for t in range(values.shape[0]):
            if method == "bayesian":
                # Update posterior distribution with observed values
                cluster_values = obs_values.loc[cluster_buses, t].values
                observed_mean = np.mean(cluster_values)
                observed_cov = np.var(cluster_values) + MIN_VARIANCE_THRESHOLD

                prior_mean = posterior_distribution["mean"]
                prior_cov = posterior_distribution["cov"]

                posterior_mean, posterior_cov = bayesian_update_multivariate(prior_mean, prior_cov, observed_mean, observed_cov)
                posterior_distribution["mean"] = posterior_mean
                posterior_distribution["cov"] = posterior_cov

                # Sample from posterior distribution
                if multivariate_distribution == 'multivariate_normal':
                    sample = np.random.multivariate_normal(posterior_mean, posterior_cov)
                elif params['multivariate_distribution'] == 'gmm':
                    # Initialize GMM with the stored parameters
                    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
                    gmm.means_ = np.array([posterior_mean])  # GMM expects 2D array for means
                    gmm.covariances_ = np.array([posterior_cov])
                    gmm.weights_ = np.array([1.0])  # Only one component
                    try:
                        gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(gmm.covariances_[0])).T
                        sample = gmm.sample()[0].flatten()
                    except np.linalg.LinAlgError:
                        epsilon = 1e-5  # or another small value
                        cov_matrix_reg = gmm.covariances_[0] + epsilon * np.eye(gmm.covariances_[0].shape[0])

                        try:
                            gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.pinv(cov_matrix_reg)).T
                            sample = gmm.sample()[0].flatten()
                        except np.linalg.LinAlgError:
                            sample = np.random.multivariate_normal(posterior_mean, posterior_cov) if posterior_cov is not None else np.full_like(posterior_mean, np.nan)
                elif multivariate_distribution == 't_distribution':
                    sample = multivariate_t.rvs(posterior_mean, posterior_cov, df=degrees_of_freedom)

                sampled_results.loc[bus, t] = sample[t]          

            elif method == "average":
                # Average observed values from the best cluster
                cluster_values = obs_values.loc[cluster_buses, t].values
                sampled_results.loc[bus, t] = np.mean(cluster_values)

    return sampled_results


for element in ['voltages', 'angles']:
        for phase in range(3):
            for type in ['real', 'reac']:
                # Run sampling with Bayesian updates
                element_name = 'v' if element == 'voltages' else 'v_angle'
                bayesian_samples = run_sampling(
                    fixed_params,
                    unobservable_bus_indices,
                    data_dict[f'obs_{element}_{type}_phase{phase+1}'],  # Replace with appropriate data
                    data_dict[f'last_year_unobs_{element}_{type}_phase{phase+1}'],
                    data_dict[f'last_year_obs_{element}_{type}_phase{phase+1}'],
                    element_name,
                    phase+1,
                    type,
                    method="bayesian"
                )

                # Run sampling with average strategy
                average_samples = run_sampling(
                    fixed_params,
                    unobservable_bus_indices,
                    data_dict[f'obs_{element}_{type}_phase{phase+1}'],  # Replace with appropriate data
                    data_dict[f'last_year_unobs_{element}_{type}_phase{phase+1}'],
                    data_dict[f'last_year_obs_{element}_{type}_phase{phase+1}'],
                    element_name,
                    phase+1,
                    type,
                    method="average"
                )

                bayesian_samples = bayesian_samples.transpose()
                average_samples = average_samples.transpose()

                # Save results to CSV
                bayesian_samples.to_csv(data_name + f"bayesian_samples_{element}_{type}_{phase+1}.csv", index=True)
                average_samples.to_csv(data_name + f"average_samples_{element}_{type}_{phase+1}.csv", index=True)

print("Sampling completed and results saved to CSV.")