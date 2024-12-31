# State Estimation Procedure
* 4 directories with the following order of execution:
  * data generation
    * To retrieve the SMART-DS network data and load profiles, first execute get_network_data.py. Both, the OpenDSS versions with and without loadshapes are required to generate the demand profiles
    * Use the version with loadshapes and compute_demand_profiles.py to generate the demand profile csv files. Save them in the OpenDSS directory without loadshapes. The directory with loadshapes is now not needed anymore.
    * Use save_circuit_data.py to safe relevant grid data for later processing
    * Finally simulate the data using simulate_data.py over the desired time frame
  
  * data preparation
    * First use create_dataset.py to build the complete dataset from the simulated data stored in npz files. The script uses partial files to meet resource limitations during execution. The npz files can either be split during simulation directly or afterwards.
    * Secondly, execute pmu_locations to place the PMUs in the grid, retrieve the network diameter (num. of hidden layers in PAWNN) and receive the PMU buses required for most subsequent processes
    * Then use get_target_columns to extract the relevant targets and remove PMU buses from the targets in the dataset
    * Create the bus_measurement_mapping
    * Use the PMU locations, target columns and the bus measurement mapping to create th input feature map
    * update_dataset.py is to be executed after pseudo-measurements have been generated to replace the corresponding features in the dataset

* After having generated the dataset, the order of the subsequent analyses is ambiguos:

  * linear_estimation.py provides linear approximation benchmarks
  * neural_networks/network contains the necessary modules and classes for NN training
  * neural_networks/training includes the training algorithms for the FCNN and PAWNN
  * neural_networks/tuning includes the hyperparameter tuning procedures for the FCNN and PAWNN
  
  * Pseudo-Measurements
    * Contains tuning and sampling scripts for the BE of bus and load data, respectively

* data/analysis contains various visualization and testing procedure for a detailed analysis of the dataset and results
