import os
import json
import pandas as pd
import re
import dss


dss_file_path = "path/to/your/dss/files"

pmu_locations = {'p1rdt4659.1.2.3', 'p1rdt4949.1.2.3', 'p1rdt833_b1_1.1', 'p1rdt1317_b1_1.1.2.3', 'p1rdt1175.3', 'p1rdm6643.1.2', 'p1rdt8002.1.2.3', 'p1rdt7077.1.2.3', 'p1rdt4568lv.1.2', 'p1rdt2829.1.2.3', 'p1rdt4396lv.1.2', 'p1rdt3614-p1rdt4656xx.1.2.3', 'p1rdt4879_b2_1.1', 'p1rdt5032-p1rdt6794x_b1_1.3', 'p1rdt5663.1.2.3', 'p1rdt6284lv.1.2', 'p1rdt5033_b1_1.1.2.3', 'p1rdt4730_b1_1.1', 'p1rdt1069_b2_1.1.2.3', 'p1rdt3810-p1rdt7137x_b1_1.1', 'p1rdt6285-p1rdt7899x.2', 'p1rdt6852.1', 'p1rdt4396-p1rdt831x_b1_1.1', 'p1rdt7136.3', 'p1rdt4401_b1_1.3', 'p1rdt6284.1.2.3', 'p1rdt320.2', 'p1rdm9214.1.2', 'p1rdt5351.3', 'p1rdt1318-p1rdt3520x_b1_1.2', 'p1rdt4728.2', 'p1rdm1612.1.2', 'p1rdt1762.1.2.3', 'p1rdm117.1.2', 'p1rdm759.1.2', 'p1rdt319lv.1.2', 'p1rdt7137.1.2.3', 'p1rdt7136lv.1.2', 'p1rdm7296.1.2', 'p1rdt3243-p1rdt5935x_b1_1.2', 'p1rdt3243.1', 'p1rdt3243-p1rdt8281x_b1_1.1', 'p1rdt1436.1.2.3', 'p1rdt7437_b1_1.1', 'p1rdt4949.2', 'p1rdt830.2', 'p1rdt5031-p1rdt7257x_b1_1.3', 'p1rdt2109-p1rdt833x_b1_1.1', 'p1rdt6853.1', 'p1rdt2110-p1rdt963x_b1_1.2', 'p1rdm12169.1.2', 'p1rdm6959.1.2', 'p1rdt2630-p1rdt831x.3', 'p1rdt2925-p1rdt3614x.3', 'p1rdt2109.1.2.3', 'p1rdt1436lv.1.2', 'p1rdt2830.1.2.3', 'p1rdt441.1', 'p1rdm7458.1.2', 'p1rdm123.1.2', 'p1rdt3616.1.2.3', 'p1rdt1869-p1rdt53x.2', 'p1rdt1761-p1rdt3616x.2', 'p1rdt1760-p1rdt7325x_b2_1.2', 'p1rdt3614-p1rdt7257xx.1.2.3', 'p1rdt834.1.2.3', 'p1rdt1070.1.2.3', 'p1rdm5898.1.2', 'p1rdm9429.1.2', 'p1rdt1318_b2_1.1.2.3', 'p1rdm4531.1.2', 'p1rdt963lv.1.2', 'p1rdt7265_b1_1.1', 'p1rdt5360-p1rdt565x.2'}

dssObj = dss.DSS
dssCircuit = dssObj.ActiveCircuit

dssObj.Text.Command = "compile " + os.path.join(dss_file_path, 'Master.dss')

bus_names = dssCircuit.AllBusNames


def parse_lines_file_for_bus(file_path, bus):
    """
    Parse an OpenDSS file and extract lines that belong to a specified partition of buses.

    Parameters:
    - file_path: Path to the DSS file (e.g., Lines.dss)
    - partition_buses: Set of bus names that define the partition

    Returns:
    - partition_edges: List of tuples representing edges (node1, node2, attributes) 
                       for lines within the specified partition
    """
    bus_edges = []
    neighbor_buses = set()
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Check if it's a line definition (you can expand this for other components)
            if line.startswith("New Line."):
                tokens = line.split()
                line_name = tokens[1].split(".")[1]
                bus1 = tokens[4].split("=")[1]  # e.g., bus1=bus1_name
                bus2 = tokens[5].split("=")[1]  # e.g., bus2=bus2_name
                
                # Extract other attributes, e.g., line length
                #length = float(tokens[3].split("=")[1])
                
                # Check if both buses are in the specified partition
                if bus1.split('.')[0] == bus:
                    # Add the line name to bus_edges
                    bus_edges.append(line_name)
                    # Add the other bus (bus2) to neighbor_buses
                    neighbor_buses.add(bus2.split('.')[0])
                elif bus2.split('.')[0] == bus:
                    # Add the line name to bus_edges
                    bus_edges.append(line_name)
                    # Add the other bus (bus1) to neighbor_buses
                    neighbor_buses.add(bus1.split('.')[0])
                    
    return bus_edges, list(neighbor_buses)


def parse_loads_for_bus(json_file_path, subregion, bus):
    """
    Extract loads from the JSON file that are connected to specified buses within a given subregion.

    Parameters:
    - json_file_path: Path to the JSON file containing load connections.
    - subregion: The specific subregion of the network to consider (e.g., 'p1uhs0_1247').
    - buses: Set of bus identifiers (e.g., 'p1udt12703') to filter loads connected to these buses.

    Returns:
    - A list of loads connected to the specified buses within the subregion.
    """
    # Load the JSON data
    with open(json_file_path, 'r') as f:
        load_data = json.load(f)
    
    # Initialize list to store loads connected to the partition's buses
    bus_loads = []

    # Pre-process partition_buses by stripping suffixes to match JSON bus format
    #base_bus = bus.split('.')[0]

    # Iterate over each connection entry in the JSON
    for connection, loads in load_data.items():
        # Split the connection key to get the subregion and bus ID
        connection_subregion, connection_bus = connection.split("->")

        # Check if the connection matches the specified subregion and bus
        if connection_subregion == subregion and connection_bus == bus:
            # If matches, extend the partition_loads list with these loads
            bus_loads.extend(loads)

    return bus_loads

def parse_transformers_for_bus(file_path, bus):
    """
    Parses a transformer DSS file to extract transformers that match buses in a specified partition.
    
    Parameters:
    - file_path: Path to the DSS transformer file.
    - partition_buses: Set of buses in the partition, including possible suffixes (e.g., 'p1udt323.1.2').
    
    Returns:
    - partition_transformers: List of transformers connected to buses in the partition.
    """
    bus_transformers = []

    # Process partition_buses to ignore suffixes
    #base_bus = bus.split('.')[0]

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Check if it is a transformer definition
            if line.startswith("New Transformer."):
                # Extract transformer name and other parameters
                tokens = line.split()
                transformer_name = tokens[1].split(".")[1]  # Extract transformer name after "New Transformer."

                # Extract bus connections using regex
                bus_connections = re.findall(r'bus=([\w\d_-]+)', line)
                
                # Check if any of the buses in the transformer matches the partition buses
                for connected_bus in bus_connections:
                    # Match against the stripped version of buses in partition
                    if connected_bus.split('.')[0] == bus:
                        bus_transformers.append(transformer_name)
                        break  # Move to next transformer if a match is found

    return bus_transformers

def retrieve_bus_measurements(bus, subregion, lines_dss_path, loads_json_path, transformers_dss_path):
    """
    Helper function to retrieve lines, loads, and transformers for the buses in a partition.
    """
    # Parse DSS files for lines, loads, and transformers based on the buses in the partition
    # These functions need to be implemented according to your data structure in DSS files
    bus_lines, neighbor_buses = parse_lines_file_for_bus(lines_dss_path, bus)
    bus_loads = parse_loads_for_bus(loads_json_path, subregion, bus)
    bus_transformers = parse_transformers_for_bus(transformers_dss_path, bus)
    
    return bus_lines, neighbor_buses, bus_loads, bus_transformers

def create_bus_measurement_mapping(bus_names, subregion, lines_dss_path, loads_json_path, transformers_dss_path):
    """
    Creates a DataFrame where each row represents a partition, with columns for buses, lines, loads, and transformers.
    
    Parameters:
    - partitions: Dictionary with partition identifiers as keys and sets of buses as values
    - lines_dss_path: Path to the lines DSS file
    - loads_dss_path: Path to the loads DSS file
    - transformers_dss_path: Path to the transformers DSS file

    Returns:
    - DataFrame with columns for buses, lines, loads, and transformers in each partition
    """
    # Initialize a list to store each partition's data
    bus_data = []

    # Loop over each partition
    for bus in bus_names:
        # Retrieve lines, loads, and transformers for the current partition
        lines, neighbor_buses, loads, transformers = retrieve_bus_measurements(bus, subregion, lines_dss_path, loads_json_path, transformers_dss_path)
        
        # Store data in dictionary format for DataFrame construction
        bus_data.append({
            "bus": bus,
            "neighbor_buses": neighbor_buses,
            "lines": lines,
            "loads": loads,
            "transformers": transformers
        })

    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(bus_data)
    return df


lines_dss_path = os.path.join(dss_file_path, 'Lines.dss')
loads_json_path = os.path.join(dss_file_path, 'visibility=E.json') 
transformers_dss_path = os.path.join(dss_file_path, 'Transformers.dss') 
subregion = 'p1rhs0_1247'

# Create DataFrame
bus_measurement_mapping = create_bus_measurement_mapping(bus_names, subregion, lines_dss_path, loads_json_path, transformers_dss_path)

bus_measurement_mapping.to_csv('bus_measurement_mapping.csv', index=False)

# Display the DataFrame
print(bus_measurement_mapping)