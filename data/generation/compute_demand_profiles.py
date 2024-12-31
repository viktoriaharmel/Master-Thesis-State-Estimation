import numpy as np
import os
import re
import dss

from utils import *

# Define the function to extract file paths from the loadshape definition
def extract_file_paths(loadshape_name, dss_file_content):
    # Regular expression pattern to match the specific loadshape definition
    pattern = rf"New Loadshape\.{re.escape(loadshape_name)}.*mult\s*=\s*\(file=([^\)]*)\).*qmult\s*=\s*\(file=([^\)]*)\)"
    
    # Search for the pattern in the file content
    match = re.search(pattern, dss_file_content, re.IGNORECASE)
    
    if match:
        mult_file_path = match.group(1).strip()
        qmult_file_path = match.group(2).strip()
        return mult_file_path, qmult_file_path
    else:
        raise ValueError(f"Loadshape {loadshape_name} not found in the file.")


if __name__ == '__main__':
    name = 'path/to/feeder/dir/'  # Replace with your feeder name
    master_fname = 'Master.dss'

    # Open and read the content of the .dss file
    with open( name + 'LoadShapes.dss', 'r') as file:
        loadshapes = file.read()
    
    # Initialize OpenDSS
    dssObj = dss.DSS
    dssCircuit = dssObj.ActiveCircuit

    dssObj.Text.Command = "compile " + os.path.join(name, master_fname)

    # Get the list of load names
    load_names = dssCircuit.Loads.AllNames

    # Preallocate list to store load profiles
    load_profiles_kW = {load_name: None for load_name in load_names}
    load_profiles_kvar = {load_name: None for load_name in load_names}
    
    for load_name in load_names:
        dssCircuit.SetActiveElement(f"Load.{load_name}")
        peak_kW = np.float64(dssCircuit.ActiveElement.Properties("kW").Val)
        peak_kvar = np.float64(dssCircuit.ActiveElement.Properties("kvar").Val)
        load_file = dssCircuit.ActiveElement.Properties("yearly").Val
        filepath_kW, filepath_kvar = extract_file_paths(load_file, loadshapes)
        new_kW = pd.read_csv(filepath_kW, header=None).values.flatten()
        new_kW *= peak_kW
        new_kvar = pd.read_csv(filepath_kvar, header=None).values.flatten()
        new_kvar *= peak_kvar
        load_profiles_kW[load_name] = new_kW
        load_profiles_kvar[load_name] = new_kvar
    
    # Convert load profiles to strings with space-separated values
    load_profiles_kW_str = [' '.join(map(str, profile)) for profile in load_profiles_kW.values()]

    # Save to CSV with space-separated load profiles without quotes or load names
    with open(name + 'demand_unc.csv', 'w') as f:
        for profile_str in load_profiles_kW_str:
            f.write(profile_str + '\n')

    
    # Convert load profiles to strings with space-separated values
    load_profiles_kvar_str = [' '.join(map(str, profile)) for profile in load_profiles_kvar.values()]

    # Save to CSV with space-separated load profiles without quotes or load names
    with open(name + 'demand_imag.csv', 'w') as f:
        for profile_str in load_profiles_kvar_str:
            f.write(profile_str + '\n')
            
    print('SAVED LOAD PROFILES TO CSV')