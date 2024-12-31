import networkx as nx

def parse_dss_file(file_path):
    """
    Parse an OpenDSS file to extract relevant components for the graph.
    This example focuses on lines between nodes.

    Parameters:
    - file_path: Path to the DSS file (e.g., Lines.dss)

    Returns:
    - edges: List of tuples representing graph edges (node1, node2, attributes)
    """
    edges = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Check if it's a line definition (you can expand this for other components)
            if line.startswith("New Line."):
                tokens = line.split()
                line_name = tokens[1].split(".")[1]
                bus1 = tokens[4].split("=")[1]  
                bus2 = tokens[5].split("=")[1]  
                
                length = float(tokens[3].split("=")[1])
                
                # Add an edge from bus1 to bus2 with length as an attribute
                edges.append((bus1, bus2, {"line_name": line_name, "length": length}))
    
    return edges


def build_electrical_graph(dss_folder):
    """
    Build an electrical network graph from OpenDSS files.
    
    Parameters:
    - dss_folder: Folder containing DSS files (e.g., Lines.dss, Transformers.dss)
    
    Returns:
    - G: A networkx graph object representing the electrical network
    """
    G = nx.Graph()
    
    # Parse the Lines.dss file (or other DSS files for additional components)
    edges = parse_dss_file(f"{dss_folder}/Lines.dss")
    
    # Add edges to the graph
    G.add_edges_from(edges)
    
    return G

def find_longest_shortest_path(graph):
    """
    Finds the longest shortest path (the path that has the maximum length among all shortest paths) 
    in a given graph.
    
    Parameters:
    - graph: A NetworkX graph object
    
    Returns:
    - longest_path: A list of nodes representing the longest shortest path
    """
    longest_path = []
    max_length = 0
    
    # Iterate over all pairs of nodes and calculate shortest paths
    for source in graph.nodes():
        for target in graph.nodes():
            if source != target:
                try:
                    path = nx.shortest_path(graph, source=source, target=target)
                    path_length = len(path) - 1  # Length of the path
                    if path_length > max_length:
                        max_length = path_length
                        longest_path = path
                except nx.NetworkXNoPath:
                    continue  # Skip if there's no path between the nodes

    return longest_path

def place_pmus(graph, K):
    """
    Places K PMUs in the graph using the described algorithm.
    
    Parameters:
    - graph: A NetworkX graph object
    - K: The number of PMUs to place
    
    Returns:
    - S: The set of nodes where PMUs are placed
    """
    S = set()  # Initialize the set of PMU locations
    
    for _ in range(K):
        # Determine the longest shortest path in the subgraph induced by nodes not in S
        subgraph = graph.subgraph(graph.nodes() - S)
        longest_shortest_path = find_longest_shortest_path(subgraph)
        
        if longest_shortest_path:
            # Place a PMU in the middle of the longest path
            middle_index = len(longest_shortest_path) // 2
            middle_node = longest_shortest_path[middle_index]
            S.add(middle_node)
    
    return S


def calculate_partition_diameters(graph, pmu_locations):
    """
    Calculates the diameter of each partition (connected component) including only the relevant PMU node.

    Parameters:
    - graph: A NetworkX graph object
    - pmu_locations: Set of nodes where PMUs are placed

    Returns:
    - partition_diameters: A list of diameters of the connected components
    """
    partition_diameters = []
    visited = set()  # Track visited nodes to avoid recounting components

    for pmu in pmu_locations:
        # Exclude all other PMUs for this specific iteration
        other_pmus = pmu_locations - {pmu}
        
        # Create a temporary subgraph that excludes other PMU nodes
        temp_graph = graph.subgraph(graph.nodes() - other_pmus)
        
        for neighbor in graph.neighbors(pmu):
            if neighbor not in visited and neighbor not in other_pmus:
                # Get all nodes connected to this neighbor in the temporary subgraph
                component = nx.node_connected_component(temp_graph, neighbor)
                relevant_nodes = component | {pmu}  # Include only the current PMU in this component
                
                # DEBUG: Print the component nodes and relevant PMU
                print(f"Component nodes (including PMU {pmu}): {relevant_nodes}")
                
                # Create subgraph for this component and calculate its diameter
                subgraph = graph.subgraph(relevant_nodes)
                if nx.is_connected(subgraph):
                    diameter = nx.diameter(subgraph)
                    partition_diameters.append(diameter)
                
                # Mark all nodes in this component as visited
                visited.update(component)

    return partition_diameters


def extract_partitions(graph, pmu_locations):
    """
    Extracts the partitions (connected components) and calculates their diameters based on PMU placement.

    Parameters:
    - graph: A NetworkX graph object
    - pmu_locations: Set of nodes where PMUs are placed

    Returns:
    - partitions: A dictionary where each key is a PMU and each value is a set of nodes in its partition
    - partition_diameters: A dictionary where each key is a PMU and each value is the diameter of its partition
    """
    partitions = {}
    visited = set()  # Track visited nodes to avoid recounting components

    for pmu in pmu_locations:
        # Exclude all other PMUs for this specific iteration
        other_pmus = pmu_locations - {pmu}
        
        # Create a temporary subgraph that excludes other PMU nodes
        temp_graph = graph.subgraph(graph.nodes() - other_pmus)
        
        for neighbor in graph.neighbors(pmu):
            if neighbor not in visited and neighbor not in other_pmus:
                # Get all nodes connected to this neighbor in the temporary subgraph
                component = nx.node_connected_component(temp_graph, neighbor)
                relevant_nodes = component | {pmu}  # Include only the current PMU in this component

                # Store the partition in the partitions dictionary
                partitions[pmu] = relevant_nodes
                
                # Mark all nodes in this component as visited
                visited.update(component)
    
    # Check for any unassigned buses and add them to the closest PMU partition
    unassigned_buses = set(graph.nodes()) - set().union(*partitions.values())
    for bus in unassigned_buses:
        # Find the closest PMU to assign the bus
        closest_pmu = min(pmu_locations, key=lambda p: nx.shortest_path_length(graph, p, bus))
        partitions[closest_pmu].add(bus)

    return partitions


dss_file_path = "path/to/your/dss/files"

G = build_electrical_graph(dss_file_path)
num_buses = G.number_of_nodes()

# Define the number of PMUs to place
K = num_buses // 10 # 10% of the total number of buses

# Place PMUs using the algorithm
pmu_locations = place_pmus(G, K)

print(f"PMUs placed at nodes: {pmu_locations}")

# Calculate partition diameters (number of layers for the PAWNN model)
partition_diameters = calculate_partition_diameters(G, pmu_locations)
print(f'partition diameters: + {partition_diameters}')

partitions = extract_partitions(G, pmu_locations)

print(f"Partitions: {partitions}")
