import dss
import os
import json

name = 'path/to/feeder/dir/'  # Replace with your feeder name
data_name = 'path/to/data/'  # Replace with your data name

dssObj = dss.DSS
dssCircuit = dssObj.ActiveCircuit

dssObj.Text.Command = "compile " + os.path.join(name, 'Master.dss')

# Assuming these are your lists
data = {
    "bus_names": dssCircuit.AllBusNames,
    "load_names": dssCircuit.Loads.AllNames,
    "transformer_names": dssCircuit.Transformers.AllNames,
    "line_names": dssCircuit.Lines.AllNames
}

# Save to a JSON file
with open(data_name + "circuit_data.json", "w") as f:
    json.dump(data, f)
