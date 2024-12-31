import opendssdirect as dss
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go

name = 'path/to/Master.dss'  # Replace with your feeder name

bus_coords = {}

# Path to the Buscoords file
buscoords_file = "path/to/Buscoords.dss"

# Read the Buscoords file
with open(buscoords_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 3:  # Ensure the line has BusName, X, Y
            bus_coords[parts[0].lower()] = (float(parts[1]), float(parts[2]))

# Load the OpenDSS model
dss.run_command("Redirect " + name)

# Extract buses
buses = dss.Circuit.AllBusNames()

# Extract lines
lines = dss.Lines.AllNames()

# Create a graph
G = nx.Graph()

# Add buses (nodes) to the graph
for bus in buses:
    bus_lower = bus.lower()
    coords = bus_coords.get(bus_lower, (None, None))  # Get coordinates or None
    G.add_node(bus, pos=coords)

# Add lines (edges) to the graph
for line in lines:
    dss.Lines.Name(line)  # Set the line as the active element
    bus1 = dss.Lines.Bus1().split('.')[0]  # Remove phase information
    bus2 = dss.Lines.Bus2().split('.')[0]
    length = dss.Lines.Length()  # Get the length of the line
    G.add_edge(bus1, bus2, length=length)

print("Graph created with", len(G.nodes), "nodes and", len(G.edges), "edges.")

# Get positions of nodes (buses)
pos = nx.get_node_attributes(G, 'pos')

# Plot the graph using matplotlib
plt.figure(figsize=(12, 8))
nx.draw(
    G, pos, with_labels=False, node_size=500, node_color="lightblue", edge_color="gray"
)
plt.title("Electrical Network from OpenDSS")
plt.show()

# Create edge traces
edge_x = []
edge_y = []
for edge in G.edges:
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(
    x=edge_x,
    y=edge_y,
    line=dict(width=1, color="#888"),
    hoverinfo="none",
    mode="lines",
)

# List of bus names with PMUs installed (for AUS P1R)
pmu_buses = ['p1rdt4659.1.2.3', 'p1rdt4949.1.2.3', 'p1rdt833_b1_1.1', 'p1rdt1317_b1_1.1.2.3', 'p1rdt1175.3', 'p1rdm6643.1.2', 'p1rdt8002.1.2.3', 'p1rdt7077.1.2.3', 'p1rdt4568lv.1.2', 'p1rdt2829.1.2.3', 'p1rdt4396lv.1.2', 'p1rdt3614-p1rdt4656xx.1.2.3', 'p1rdt4879_b2_1.1', 'p1rdt5032-p1rdt6794x_b1_1.3', 'p1rdt5663.1.2.3', 'p1rdt6284lv.1.2', 'p1rdt5033_b1_1.1.2.3', 'p1rdt4730_b1_1.1', 'p1rdt1069_b2_1.1.2.3', 'p1rdt3810-p1rdt7137x_b1_1.1', 'p1rdt6285-p1rdt7899x.2', 'p1rdt6852.1', 'p1rdt4396-p1rdt831x_b1_1.1', 'p1rdt7136.3', 'p1rdt4401_b1_1.3', 'p1rdt6284.1.2.3', 'p1rdt320.2', 'p1rdm9214.1.2', 'p1rdt5351.3', 'p1rdt1318-p1rdt3520x_b1_1.2', 'p1rdt4728.2', 'p1rdm1612.1.2', 'p1rdt1762.1.2.3', 'p1rdm117.1.2', 'p1rdm759.1.2', 'p1rdt319lv.1.2', 'p1rdt7137.1.2.3', 'p1rdt7136lv.1.2', 'p1rdm7296.1.2', 'p1rdt3243-p1rdt5935x_b1_1.2', 'p1rdt3243.1', 'p1rdt3243-p1rdt8281x_b1_1.1', 'p1rdt1436.1.2.3', 'p1rdt7437_b1_1.1', 'p1rdt4949.2', 'p1rdt830.2', 'p1rdt5031-p1rdt7257x_b1_1.3', 'p1rdt2109-p1rdt833x_b1_1.1', 'p1rdt6853.1', 'p1rdt2110-p1rdt963x_b1_1.2', 'p1rdm12169.1.2', 'p1rdm6959.1.2', 'p1rdt2630-p1rdt831x.3', 'p1rdt2925-p1rdt3614x.3', 'p1rdt2109.1.2.3', 'p1rdt1436lv.1.2', 'p1rdt2830.1.2.3', 'p1rdt441.1', 'p1rdm7458.1.2', 'p1rdm123.1.2', 'p1rdt3616.1.2.3', 'p1rdt1869-p1rdt53x.2', 'p1rdt1761-p1rdt3616x.2', 'p1rdt1760-p1rdt7325x_b2_1.2', 'p1rdt3614-p1rdt7257xx.1.2.3', 'p1rdt834.1.2.3', 'p1rdt1070.1.2.3', 'p1rdm5898.1.2', 'p1rdm9429.1.2', 'p1rdt1318_b2_1.1.2.3', 'p1rdm4531.1.2', 'p1rdt963lv.1.2', 'p1rdt7265_b1_1.1', 'p1rdt5360-p1rdt565x.2']
pmu_buses = [pmu_loc.split('.')[0] for pmu_loc in pmu_buses]

# Extract PV Systems and Battery Storages
pv_systems = dss.PVsystems.AllNames()
storages = dss.Storages.AllNames()

# Create dictionaries for PV and storage positions
pv_coords = {}
battery_coords = {}

# Get PV system positions
for pv in pv_systems:
    dss.PVsystems.Name(pv)
    bus = dss.Properties.Value("bus1").split('.')[0].lower()  # Get bus name
    coords = bus_coords.get(bus, (None, None))  # Get coordinates
    if coords:
        pv_coords[pv] = coords

# Get battery storage positions
for storage in storages:
    dss.Storages.Name(storage)
    bus = dss.Properties.Value("bus1").split('.')[0].lower()  # Get bus name
    coords = bus_coords.get(bus, (None, None))  # Get coordinates
    if coords:
        battery_coords[storage] = coords

# Create node traces for PV systems
pv_x = []
pv_y = []
pv_text = []

for pv, (x, y) in pv_coords.items():
    pv_x.append(x)
    pv_y.append(y)
    pv_text.append(f"PV System: {pv}")

pv_trace = go.Scatter(
    x=pv_x,
    y=pv_y,
    mode="markers",
    text=pv_text,
    hoverinfo="text",
    marker=dict(
        size=12,
        symbol="triangle-up",  # Triangle marker for PV systems
        color="yellow",
        line=dict(width=1, color="black"),
    ),
)

# Create node traces for battery storages
battery_x = []
battery_y = []
battery_text = []

for battery, (x, y) in battery_coords.items():
    battery_x.append(x)
    battery_y.append(y)
    battery_text.append(f"Battery Storage: {battery}")

battery_trace = go.Scatter(
    x=battery_x,
    y=battery_y,
    mode="markers",
    text=battery_text,
    hoverinfo="text",
    marker=dict(
        size=12,
        symbol="diamond",  # Diamond marker for batteries
        color="purple",
        line=dict(width=1, color="black"),
    ),
)

# Create node traces for buses (including PMU highlights)
node_x = []
node_y = []
node_text = []
node_colors = []

for node, (x, y) in pos.items():
    node_x.append(x)
    node_y.append(y)
    node_text.append(node)
    if node.lower() in pmu_buses:
        node_colors.append("red")  # PMU buses in red
    else:
        node_colors.append("blue")  # Non-PMU buses in blue

node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode="markers",
    text=node_text,
    hoverinfo="text",
    marker=dict(
        showscale=True,
        colorscale="YlGnBu",
        color=node_colors,
        size=10,
        colorbar=dict(thickness=15, title="Bus", xanchor="left", titleside="right"),
    ),
)

# Create edge traces with hover text for line names
edge_x = []
edge_y = []
edge_hovertext = []  # Store line names for hover text

for line in lines:
    dss.Lines.Name(line)  # Set the line as the active element
    bus1 = dss.Lines.Bus1().split('.')[0]  # Remove phase information
    bus2 = dss.Lines.Bus2().split('.')[0]
    coords1 = pos.get(bus1)
    coords2 = pos.get(bus2)
    
    if coords1 and coords2:
        x0, y0 = coords1
        x1, y1 = coords2
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_hovertext.append(line)  # Add the line name to the hover text

# Create edge trace
edge_trace = go.Scatter(
    x=edge_x,
    y=edge_y,
    line=dict(width=1, color="#888"),
    hoverinfo="text",  # Show hover text on hover
    hovertext=edge_hovertext,  # Add line names as hover text
    mode="lines",
)

# Create the figure with all traces
fig = go.Figure(data=[edge_trace, node_trace, pv_trace, battery_trace])
fig.update_layout(
    title="Electrical Network Visualization with Grid Elements",
    showlegend=False,
    hovermode="closest",
    margin=dict(b=0, l=0, r=0, t=40),
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=False, zeroline=False),
)
fig.show()

# Save the interactive visualization as an HTML file
fig.write_html("grid_visualization.html")