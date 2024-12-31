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
from scipy.fftpack import fft
import random
from scipy.spatial.distance import cdist

# Define a minimum variance threshold to avoid zero variance
MIN_VARIANCE_THRESHOLD = 1e-34  # Adjust as needed

name = 'path/to/data/directory/'
name_2017 = 'path/to/data/directory/'
subregion = "p1rhs0_1247"
network = 'aus_p1r'

# Load the data without headers
last_year_data = pd.read_csv(name_2017 + "demand_unc.csv", sep=" ", header=None)
this_year_observable_data = pd.read_csv(name + "demand_unc.csv", sep=" ", header=None)

# Choose a subset of the data for faster processing
last_year_data = last_year_data.iloc[:, ::100]
this_year_observable_data = this_year_observable_data.iloc[:, ::100]
demand_type = 'active'

# Specify indices for observable and unobservable buses
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

print(observable_loads_by_phase)

param_space = {
    'num_clusters': {1: range(2, len(observable_loads_by_phase[1]) // 2), 2: range(2, len(observable_loads_by_phase[2]) // 2)},
    'distance_metric': ['euclidean', 'dtw'],
    'distance_metric_clustering': ['euclidean', 'dtw'],
    'multivariate_distribution': ['multivariate_normal', 'gmm', 't_distribution'],
    'n_components': range(1, 6),  # GMM's number of components from 1 to 5
    'degrees_of_freedom': range(3, 30)  # Degrees of freedom for the t-distribution
}

# Discrete Fourier Transform (DFT)
def apply_dft(data, num_components):
    transformed_data = fft(data, axis=1)
    magnitudes = np.abs(transformed_data)[:num_components]
    return magnitudes


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


# Function to randomly sample from the extended parameter space, including df for multivariate t-distribution
def sample_params(param_space):
    params = {}
    #params['representation'] = random.choice(param_space['representation'])
    params['distance_metric_clustering'] = random.choice(param_space['distance_metric_clustering'])

    params['num_clusters'] = {1: random.choice(param_space['num_clusters'][1]), 2: random.choice(param_space['num_clusters'][2])}#, 3: random.choice(param_space['num_clusters'][3])}

    params['distance_metric'] = random.choice(param_space['distance_metric'])
    params['multivariate_distribution'] = random.choice(param_space['multivariate_distribution'])

    if params['multivariate_distribution'] == 'gmm':
        params['n_components'] = random.choice(param_space['n_components'])
    # Degrees of freedom for t-distribution
    if params['multivariate_distribution'] == 't_distribution':
        params['degrees_of_freedom'] = random.choice(param_space['degrees_of_freedom'])
    else:
        params['degrees_of_freedom'] = None  # Not applicable for non-t-distribution

    return params


# Define function to run the experiment for a given parameter set
def run_experiment(params):
    #representation = params['representation']
    distance_metric_clustering = params['distance_metric_clustering']
    num_clusters = params['num_clusters']
    distance_metric = params['distance_metric']
    multivariate_distribution = params['multivariate_distribution']
    degrees_of_freedom = params['degrees_of_freedom']

    cluster_distributions_by_phase = {}  # Store cluster distributions by phase
    clusters_by_phase = {}

    for phase, loads in observable_loads_by_phase.items():
        # Perform clustering only on observable loads for the current phase
        observable_data_phase_df = last_year_data.iloc[loads].copy(deep=True)

        # Identify duplicate rows
        unique_data = observable_data_phase_df.drop_duplicates(keep='first')

        # Convert back to numpy array if needed
        observable_data_phase = unique_data.values

        #if representation:
        #    observable_data_phase = apply_dft(observable_data_phase, num_components_dft[network][demand_type][phase])
        
        # Choose the clustering algorithm based on parameters
        num_clusters = params['num_clusters'][phase]
        clustering = TimeSeriesKMeans(n_clusters=num_clusters, metric=distance_metric_clustering, random_state=0)

        # Fit clustering to data of the current phase
        cluster_labels = clustering.fit_predict(observable_data_phase)

        # Store clusters for this phase
        clusters = {i: [] for i in range(num_clusters)}
        for load_idx, cluster_label in zip(loads, cluster_labels):
            if cluster_label != -1: 
                clusters[cluster_label].append(load_idx)

        # Merge small clusters for this phase, if necessary
        clusters, _ = merge_small_clusters(clusters, last_year_data, min_samples=2)
        num_clusters = len(clusters)
        params['num_clusters'][phase] = num_clusters
        
        clusters_by_phase[phase] = clusters

        # Compute the distributions (mean and covariance) for each cluster in this phase
        cluster_distributions = {}
        for cluster_label, indices in clusters.items():
            cluster_data = observable_data_phase_df.loc[indices].values
            mean_vector = np.mean(cluster_data, axis=0)
            covariance_matrix = np.cov(cluster_data, rowvar=False)
            cluster_distributions[cluster_label] = {"mean": mean_vector, "cov": covariance_matrix}
        
        # Save cluster distributions by phase
        cluster_distributions_by_phase[phase] = cluster_distributions

    # Match each unobservable load to closest cluster using chosen distance metric
    best_cluster_matches = {}
    
    for unobs_index in unobservable_loads:
        phase = load_phases[unobs_index]
        unobs_data = last_year_data.iloc[unobs_index].values
        min_distance = float('inf')
        best_cluster = None

        # Match only with clusters of the same phase
        if phase in cluster_distributions_by_phase.keys():
            possible_clusters = cluster_distributions_by_phase[phase].items()
        else:
             # Flatten all clusters across phases into a single list of (cluster_label, dist_params) pairs
            possible_clusters = [
                (cluster_label, dist_params)
                for phase_clusters in cluster_distributions_by_phase.values()
                for cluster_label, dist_params in phase_clusters.items()
            ]
        for cluster_label, dist_params in possible_clusters:
            mean_vector = dist_params["mean"]
            try:
                if distance_metric == 'euclidean':
                    distance = euclidean(unobs_data, mean_vector)
                elif distance_metric == 'dtw':
                    distance = dtw(unobs_data, mean_vector, global_constraint='sakoe_chiba', sakoe_chiba_radius=4)
            except np.linalg.LinAlgError:
                continue

            if distance < min_distance:
                min_distance = distance
                best_cluster = cluster_label

        # Store the best cluster match for this unobservable load
        best_cluster_matches[unobs_index] = best_cluster

    # Bayesian update function with parameterized gain type
    def bayesian_update_multivariate(prior_mean, prior_cov, observed_mean, observed_cov, time_step):
        adaptive_factor = min(1.0, 0.1 * (time_step + 1))

        # Ensure prior_cov is an array
        if np.isscalar(prior_cov):
            prior_cov = np.array([[prior_cov]])  # Convert scalar to 1D array
        
        # Determine the Kalman gain based on prior_cov shape
        try:
            kalman_gain = prior_cov @ np.linalg.inv(prior_cov + observed_cov)
        except np.linalg.LinAlgError:
            epsilon = 1e-5  # Adjust as needed for numerical stability
            cov_matrix_reg = (prior_cov + observed_cov) + epsilon * np.eye((prior_cov + observed_cov).shape[0])
            kalman_gain = prior_cov @ np.linalg.inv(cov_matrix_reg)

        # Perform the Bayesian update
        if np.isscalar(kalman_gain):
            kalman_gain = np.array([[kalman_gain]])
        posterior_mean = prior_mean + kalman_gain @ (observed_mean - prior_mean)
        posterior_cov = (np.eye(len(prior_cov)) - kalman_gain) @ prior_cov if prior_cov.ndim > 1 else prior_cov * (1 - adaptive_factor)
        
        return posterior_mean, posterior_cov

    # Initialize the posterior distributions for unobservable loads
    posterior_distributions = {}
    matched_phases = {}

    for load in unobservable_loads:
        # Get the phase of the current load
        load_phase = load_phases[load]
        
        # Get the best cluster match for this load
        best_cluster = best_cluster_matches[load]

        # Search for the matched cluster in all phases
        matched_mean = None
        matched_cov = None
        for phase, phase_clusters in cluster_distributions_by_phase.items():
            if best_cluster in phase_clusters:
                matched_mean = phase_clusters[best_cluster]["mean"]
                matched_cov = phase_clusters[best_cluster]["cov"]
                matched_phases[load] = phase
                break
        
        if matched_mean is not None and matched_cov is not None:
            # Found the cluster; initialize the posterior distribution
            posterior_distributions[load] = {
                "mean": matched_mean,
                "cov": matched_cov,
            }
        else:
            # Handle unmatched cluster case
            print(f"Warning: No matching cluster found for load {load}. Using default distribution.")
            posterior_distributions[load] = {
                "mean": [0] * len(next(iter(cluster_distributions_by_phase[load_phase].values()))["mean"]),
                "cov": np.eye(len(next(iter(cluster_distributions_by_phase[load_phase].values()))["mean"])),
            }
    sampled_demands = pd.DataFrame(index=unobservable_loads, columns=range(this_year_observable_data.shape[1]))

    # Bayesian updates for each time step
    for time_step in range(this_year_observable_data.shape[1]):
        for unobs_index in unobservable_loads:
            # Get the matched cluster for this unobservable load
            matched_cluster = best_cluster_matches[unobs_index]
            cluster_indices = clusters_by_phase[matched_phases[unobs_index]][matched_cluster]
            cluster_values = this_year_observable_data.iloc[cluster_indices, time_step].values
            observed_mean = np.mean(cluster_values)
            observed_cov = np.cov(cluster_values) if len(cluster_values) > 1 else MIN_VARIANCE_THRESHOLD
            prior_mean = posterior_distributions[unobs_index]["mean"]
            prior_cov = posterior_distributions[unobs_index]["cov"]
            posterior_mean, posterior_cov = bayesian_update_multivariate(prior_mean, prior_cov, observed_mean,
                                                                         observed_cov, time_step)
            posterior_distributions[unobs_index]["mean"] = posterior_mean
            posterior_distributions[unobs_index]["cov"] = posterior_cov

        # Sample from the posterior distribution for each unobservable load
        for unobs_index in unobservable_loads:
            mean, cov = posterior_distributions[unobs_index]["mean"], posterior_distributions[unobs_index]["cov"]
            if params['multivariate_distribution'] == 'multivariate_normal':
                sample = np.random.multivariate_normal(mean, cov) if cov is not None else np.full_like(mean, np.nan)
            
            elif params['multivariate_distribution'] == 'gmm':
                # Initialize GMM with the stored parameters
                gmm = GaussianMixture(n_components=params['n_components'], covariance_type='full', random_state=0)
                gmm.means_ = np.array([mean])  # GMM expects 2D array for means
                gmm.covariances_ = np.array([cov])
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
                        sample = np.random.multivariate_normal(mean, cov) if cov is not None else np.full_like(mean, np.nan)
            
            elif params['multivariate_distribution'] == 't_distribution':
                # Sample from a multivariate t-distribution with specified degrees of freedom
                df = params['degrees_of_freedom']
                sample = multivariate_t.rvs(mean, cov, df=df)
            
            # Store the sampled value
            sampled_demands.loc[unobs_index, time_step] = sample[time_step]

    # Evaluate and return performance
    score = np.mean((sampled_demands.values - last_year_data.iloc[unobservable_loads].values) ** 2)
    return score


# Run random search and store top results
num_samples = 500  # Number of random samples to try
top_results = []

# Run the loop and append results to CSV
for _ in range(num_samples):
    params = sample_params(param_space)
    score = run_experiment(params)

    # Sort the top results and keep only top 5
    top_results.append((params, score))
    top_results.sort(key=lambda x: x[1])
    top_results = top_results[:5]  # Keep only top 5 results

    # Print the result
    print(f"Params: {params}, Score: {score}")

# Output the top results
print("Top Parameter Combinations:")
for i, (params, score) in enumerate(top_results, start=1):
    print(f"Rank {i}: Params={params}, Score={score}")
