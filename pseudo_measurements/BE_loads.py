import numpy as np
import pandas as pd
import json
import re
import gc
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import dtw
from scipy.spatial.distance import euclidean
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_t
from scipy.spatial.distance import cdist

# Define a minimum variance threshold to avoid zero variance
MIN_VARIANCE_THRESHOLD = 1e-34  # Adjust as needed

# Fixed parameters for the experiment
fixed_params = {
    'num_clusters': {1: 3, 2: 5}, 
    'distance_metric_clustering': 'euclidean',
    'distance_metric': 'dtw',
    'multivariate_distribution': 'gmm',
    'n_components': 3,
    'degrees_of_freedom': None,  # Only used for t-distribution
}

# Directories and file names
name = 'path/to/feeder/dir/'
name_2017 = 'path/to/feeder/dir/last_year/'
subregion = "p1rhs0_1247"
network = 'aus_p1r'

# Load data
last_year_data_real = pd.read_csv(name_2017 + "demand_unc.csv", sep=" ", header=None)
this_year_observable_data_real = pd.read_csv(name + "demand_unc.csv", sep=" ", header=None)
last_year_data_reac = pd.read_csv(name_2017 + "demand_imag.csv", sep=" ", header=None)
this_year_observable_data_reac = pd.read_csv(name + "demand_imag.csv", sep=" ", header=None)

data_dict = {'real': {}, 'reac': {}}
data_dict['real']['last_year_data'] = last_year_data_real
data_dict['real']['this_year_observable_data'] = this_year_observable_data_real
data_dict['reac']['last_year_data'] = last_year_data_reac
data_dict['reac']['this_year_observable_data'] = this_year_observable_data_reac

# Load the JSON file containing observable loads (with AMI)
with open(name + 'visibility=M.json', 'r') as f:
    ami_data = json.load(f)

# Parse DSS file to extract bus and load information
dss_bus_coords_file = name + 'Buscoords.dss'
dss_loads_file = name + 'Loads.dss'

# Lists to hold data
observable_loads = []
observable_loads_names = []
observable_buses_names = []
unobservable_loads = []
unobservable_loads_names = []
unobservable_buses_names = []

# Step 2: Extract all loads and buses from DSS files
# Parsing DSS bus coordinates file
with open(dss_bus_coords_file, 'r') as f:
    all_buses = [line.split()[0] for line in f if line.strip()]

# Step 1: Extract observable loads and buses from JSON file
for bus_load_key, loads in ami_data.items():
    subreg = bus_load_key.split('->')[0]
    if subreg == subregion:
        bus_name = bus_load_key.split('->')[1]  # Only take the part before the arrow
        if bus_name in all_buses:
            observable_buses_names.append(bus_name)  # Buses with AMI
            observable_loads_names.extend(loads)  # Loads associated with those buses

del ami_data
gc.collect()

# Parsing DSS loads file
with open(dss_loads_file, 'r') as f:
    load_pattern = re.compile(r'New Load\.(\S+)')
    all_loads = []
    load_index = 0

    for line in f:
        match = load_pattern.search(line)
        if match:
            load_name = match.group(1)
            all_loads.append((load_index, load_name))
            load_index += 1

# Step 4: Identify observable and unobservable loads
load_phases = {}
for idx, load_name in all_loads:
    if len(load_name.split('_')) > 2:
        load_phases[idx] = int(load_name.split('_')[2])
    else:
        load_phases[idx] = 3
    # Extract the load prefix to match with JSON file (remove suffix after the first underscore)
    load_prefix = load_name.split('_')[0] + "_" + load_name.split('_')[1]
    if load_prefix in observable_loads_names:
        observable_loads.append(idx)
    else:
        unobservable_loads.append(idx)
        if len(load_name.split('_')) > 2:
            load_name = load_name.split('_')[1] + "_" + load_name.split('_')[2]
        else:
            load_name = load_name.split('_')[1]
        unobservable_loads_names.append(load_name)


# Step 4: Identify unobservable buses (those not listed in the observable buses)
unobservable_buses_names = [bus for bus in all_buses if bus not in observable_buses_names]

# Organize observable loads by phase
observable_loads_by_phase = {1: [], 2: []}
for idx in observable_loads:
    phase = load_phases[idx]
    observable_loads_by_phase[phase].append(idx)


del observable_loads_names
del observable_buses_names
del unobservable_loads_names
del unobservable_buses_names
gc.collect()

def calculate_medoid(cluster_data):
    """
    Calculates the medoid of a cluster. The medoid is the point that minimizes the sum
    of distances to all other points in the cluster.

    Parameters:
    - cluster_data: 2D array or DataFrame representing the data points in a cluster.

    Returns:
    - medoid: The medoid point of the cluster.
    """
    # Compute pairwise distances between all points in the cluster
    distances = cdist(cluster_data, cluster_data, metric='euclidean')

    # Find the index of the point with the minimum total distance to others
    medoid_index = np.argmin(distances.sum(axis=1))

    return medoid_index

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
        cluster_data = data.iloc[indices].values
        medoid_index = calculate_medoid(cluster_data)
        medoids[cluster_label] = cluster_data[medoid_index]

    # Step 2: Identify small clusters
    small_clusters = [label for label, indices in clusters.items() if len(indices) < min_samples]

    # Step 3: Merge small clusters into the nearest larger clusters
    for small_cluster in small_clusters:
        # small_cluster_data = data.iloc[clusters[small_cluster]].values

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
        cluster_data = data.iloc[indices].values
        medoid_index = calculate_medoid(cluster_data)
        merged_medoids[label] = cluster_data[medoid_index]

    return clusters, merged_medoids


# Bayesian update function (as defined in the original script)
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


# Clustering and Bayesian updates
def cluster_and_sample(data, params, observable_loads_by_phase, unobservable_loads, this_year_data, type, method):
    """ Perform clustering and sampling for unobservable loads. """

    clusters_by_phase = {}
    
    sampled_demands = pd.DataFrame(index=[f'{all_loads[i][1]}_{type}' for i in unobservable_loads], columns=range(this_year_data.shape[1]))

    # Cluster and process observable loads
    for phase, loads in observable_loads_by_phase.items():
        # Perform clustering only on observable loads for the current phase
        observable_data_phase_df = data.iloc[loads].copy(deep=True)

        # Identify duplicate rows
        unique_data = observable_data_phase_df.drop_duplicates(keep='first')

        # Convert back to numpy array if needed
        observable_data = unique_data.values
        if len(observable_data) == 0:
            return None
        num_clusters = params['num_clusters'][phase]
        clustering = TimeSeriesKMeans(n_clusters=num_clusters, metric=params['distance_metric_clustering'], random_state=0)
        cluster_labels = clustering.fit_predict(observable_data)

        # Store clusters for this phase
        clusters = {i: [] for i in range(num_clusters)}
        for load_idx, cluster_label in zip(loads, cluster_labels):
            if cluster_label != -1:  # Ignore noise points in DBSCAN
                clusters[cluster_label].append(load_idx)

        # Merge small clusters for this phase, if necessary
        clusters, _ = merge_small_clusters(clusters, data, min_samples=2)
        num_clusters = len(clusters)
        params['num_clusters'][phase] = num_clusters
        
        clusters_by_phase[phase] = clusters

    # Match unobservable loads to clusters
    best_cluster_matches = {}
    for unobs_index in unobservable_loads:
        phase = load_phases[unobs_index]
        unobs_data = data.iloc[unobs_index].values
        min_distance = float('inf')
        best_indices = None

        # Match only with clusters of the same phase
        if phase in clusters_by_phase.keys():
            possible_clusters = clusters_by_phase[phase].items()
        else:
             # Flatten all clusters across phases into a single list of (cluster_label, dist_params) pairs
            possible_clusters = [(cluster_label, indices) 
                  for phase, clusters in clusters_by_phase.items() 
                  for cluster_label, indices in clusters.items()]

        for cluster_label, indices in possible_clusters:
            cluster_data = data.loc[indices].values
            mean_vector = np.mean(cluster_data, axis=0)
            try:
                if params['distance_metric'] == 'euclidean':
                    distance = euclidean(unobs_data, mean_vector)
                elif params['distance_metric'] == 'dtw':
                    distance = dtw(unobs_data, mean_vector, global_constraint='sakoe_chiba', sakoe_chiba_radius=4)
            except np.linalg.LinAlgError:
                continue

            if distance < min_distance:
                min_distance = distance
                best_indices = indices

        # Store the best cluster match indices for this unobservable load
        best_cluster_matches[unobs_index] = best_indices


    # Sampling for unobservable loads
    for unobs_index in unobservable_loads:
        posterior_distribution = {}
        cluster_indices = best_cluster_matches[unobs_index]
        cluster_data_last_year = data.loc[cluster_indices].values
        posterior_distribution['prior mean'] = np.mean(cluster_data_last_year, axis=0)
        posterior_distribution['prior cov'] = np.cov(cluster_data_last_year, rowvar=False)
        for time_step in range(this_year_data.shape[1]):

            cluster_values = this_year_data.iloc[cluster_indices, time_step].values

            if method == "bayesian":
                # Update the prior distribution with the observed data
                observed_mean = np.mean(cluster_values)
                observed_cov = np.var(cluster_values) + MIN_VARIANCE_THRESHOLD
                posterior_mean, posterior_cov = bayesian_update_multivariate(
                    posterior_distribution['prior mean'], posterior_distribution['prior cov'], observed_mean, observed_cov
                )
                posterior_distribution["prior mean"] = posterior_mean
                posterior_distribution["prior cov"] = posterior_cov

                # Sample from the posterior distribution for each unobservable load
                if params['multivariate_distribution'] == 'multivariate_normal':
                    sample = np.random.multivariate_normal(posterior_mean, posterior_cov) if posterior_cov is not None else np.full_like(posterior_mean, np.nan)
                
                elif params['multivariate_distribution'] == 'gmm':
                    # Initialize GMM with the stored parameters
                    gmm = GaussianMixture(n_components=params['n_components'], covariance_type='full', random_state=0)
                    gmm.means_ = np.array([posterior_mean])  # GMM expects 2D array for means
                    gmm.covariances_ = np.array([posterior_cov])
                    gmm.weights_ = np.array([1.0])  # Only one component
                    try:
                        gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(gmm.covariances_[0])).T
                    except np.linalg.LinAlgError:
                        epsilon = 1e-5  # or another small value
                        cov_matrix_reg = gmm.covariances_[0] + epsilon * np.eye(gmm.covariances_[0].shape[0])

                        try:
                            gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.pinv(cov_matrix_reg)).T
                            sample = gmm.sample()[0].flatten()
                        except np.linalg.LinAlgError:
                            sample = np.random.multivariate_normal(posterior_mean, posterior_cov) if posterior_cov is not None else np.full_like(posterior_mean, np.nan)
                
                elif params['multivariate_distribution'] == 't_distribution':
                    # Sample from a multivariate t-distribution with specified degrees of freedom
                    df = params['degrees_of_freedom']
                    sample = multivariate_t.rvs(posterior_mean, posterior_cov, df=df)
                
                # Store the sampled value
                sampled_demands.loc[f'{all_loads[unobs_index][1]}_{type}', time_step] = sample[time_step]
            else:  # Average method
                # Use the average of the cluster values as the estimate
                sampled_demands.loc[f'{all_loads[unobs_index][1]}_{type}', time_step] = np.mean(cluster_values)

    return sampled_demands

# Generate samples with and without Bayesian updates
for type in ['real', 'reac']:
    bayesian_samples = cluster_and_sample(data_dict[type]['last_year_data'], fixed_params, observable_loads_by_phase, unobservable_loads, data_dict[type]['this_year_observable_data'], type, method="bayesian")
    average_samples = cluster_and_sample(data_dict[type]['last_year_data'], fixed_params, observable_loads_by_phase, unobservable_loads, data_dict[type]['this_year_observable_data'], type, method="average")

    # Save results to CSV
    bayesian_samples.to_csv(name + f"bayesian_samples_loads_{type}.csv", index=True)
    average_samples.to_csv(name + f"average_samples_loads_{type}.csv", index=True)

print("Sampling completed and results saved to CSV.")