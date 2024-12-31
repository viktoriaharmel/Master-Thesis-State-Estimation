import numpy as np
import os
import dss
import math

from utils import *

''' adapted from "Coordinating distributed energy resources for reliability can significantly reduce future distribution grid upgrades and peak load"
Navidi, Thomas et al.
Joule, Volume 7, Issue 8, 1769 - 1792'''

if __name__ == '__main__':
    name = "path/to/feeder/dir/"  # Replace with your feeder name

    # Simulation parameters
    total_steps = 35040 # Number of 15-minute intervals in a year
    stepsize = 15/60  # 15-minute intervals

    cwd = os.getcwd()  # Get the current directory
    load_fname_orig = 'Loads_orig.dss'
    load_fname = 'Loads.dss'
    master_fname = 'Master.dss'
    lines_fname = 'Lines.dss'

    # Initialize OpenDSS
    dssObj = dss.DSS
    dssCircuit = dssObj.ActiveCircuit

    runPF(name, master_fname, dssObj)
    network_name = os.getcwd() + '/'

    # Export transformer tap and power information
    export_taps(network_name, master_fname, dssObj)
    t_fname = get_t_power_filename(network_name)

    # Read transformer powers
    t_real, t_reac, t_apparent_power = get_t_power(t_fname)

    # Get initial voltage magnitudes
    v_mags = get_VmagsPu(dssCircuit)

    # Print network characteristics
    print('N = ', v_mags.shape)
    print('N transformers = ', t_apparent_power.shape)

    line_names = dssCircuit.Lines.AllNames

    # Preallocate arrays for results
    t_real_all = np.zeros((t_apparent_power.size, total_steps), dtype=np.float32)
    t_reac_all = np.zeros((t_apparent_power.size, total_steps), dtype=np.float32)
    t_powers_all = np.zeros((t_apparent_power.size, total_steps), dtype=np.float32)
    v_mags_all = np.zeros((v_mags.size, total_steps), dtype=np.float32)
    c_mags_all = np.zeros((len(line_names), total_steps + 1), dtype=object)
    c_mags_all[:, 0] = line_names

    # Get the list of load names
    load_names = dssCircuit.Loads.AllNames
    num_loads = len(load_names)

    print(f'Number of loads: {num_loads}')

    bus_names = dssCircuit.AllBusNames
    num_busses = len(bus_names)

    print(f'Number of busses: {num_busses}')

    # Get the number of phases for each bus
    phases_per_bus = []
    for i in range(num_busses):
        bus = dssCircuit.Buses(i)
        phases_per_bus.append(bus.NumNodes)

    # Load the load profiles
    loads_real = np.loadtxt(name + 'demand_unc.csv')
    loads_imag = np.loadtxt(name + 'demand_imag.csv')


    bus_complex_voltages = {}
    bus_voltage_angles = {}

    for t in range(total_steps):
        print(f'Time step: {t+1}/{total_steps}')

        # Update the loads in the circuit
        for i in range(len(load_names)):
            load_name = load_names[i]
            dssCircuit.SetActiveElement(f"Load.{load_name}")
            dssCircuit.ActiveElement.Properties("kW").Val = str(loads_real[i][t])
            dssCircuit.ActiveElement.Properties("kvar").Val = str(loads_imag[i][t])

        # Run the power flow for this snapshot
        dssObj.Text.Command = "Solve"

        # Export transformer tap and power information
        export_taps(network_name, master_fname, dssObj)
        t_real, t_reac, t_apparent_power = get_t_power(t_fname)

        # Get the bus voltages
        v_complex = dssCircuit.AllBusVolts

        index = 0

        # Iterate over each bus and its number of phases
        for bus_id, phases in enumerate(phases_per_bus, start=1):
            # Each phase has a pair (real, imag), so for 'phases' phases, we need 2*phases values
            num_values = phases * 2
            if bus_id not in bus_complex_voltages.keys():
                bus_complex_voltages[bus_id] = []
            bus_complex_voltages[bus_id].append(v_complex[index:index + num_values])
            index += num_values

        # Get the bus voltage angles
        for bus_id, values in bus_complex_voltages.items():
            values = values[-1]
            phases = len(values) // 2  # Number of phases for this bus
            angles = []
            for i in range(phases):
                real = values[2 * i]
                imag = values[2 * i + 1]
                angle = math.atan2(imag, real)
                angles.append(angle)  # (angle in radians)
            if bus_id not in bus_voltage_angles.keys():
                bus_voltage_angles[bus_id] = []
            bus_voltage_angles[bus_id].append(angles)

        # Get the line currents
        for i, line_name in enumerate(line_names):
            # Access current magnitudes for the specified line
            dssCircuit.SetActiveElement(f"Line.{line_name}")
            currents = dssCircuit.ActiveElement.CurrentsMagAng
            current_magnitudes = currents[::2]  # Extract only magnitudes (odd indices)

            # Extract real parts (even indices)
            real_currents = tuple(current_magnitudes[::2])  # Real parts: values at even indices

            # Store the data with a timestamp
            c_mags_all[i, t + 1] = real_currents

        # Store results in the arrays
        t_real_all[:, t] = t_real  # Shape: (num_transformers, total_steps)
        t_reac_all[:, t] = t_reac  # Shape: (num_transformers, total_steps)
        t_powers_all[:, t] = t_apparent_power  # Shape: (num_transformers, total_steps)

        
    # Save the metrics
    np.savez(name + '_metrics_t_powers.npz', t_real_all=t_real_all, t_reac_all=t_reac_all,
                t_powers_all=t_powers_all)
    np.savez(name + f'_metrics_complex_volts_angles_v3.npz', v_complex=bus_complex_voltages,
                v_angles=bus_voltage_angles)
    np.savez(name + f'_metrics_current_mags_v3.npz', c_mags_all=c_mags_all)
    print('SAVED METRICS')
