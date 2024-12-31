import os
import dss
import numpy as np
import pandas as pd
import time

''' adapted from "Coordinating distributed energy resources for reliability can significantly reduce future distribution grid upgrades and peak load"
Navidi, Thomas et al.
Joule, Volume 7, Issue 8, 1769 - 1792'''

#pip install dss-python, numpy, pandas


def assign_transformer_limits(trans_base):
    transformer_limits = np.max(trans_base, axis=1) * 1.2
    return transformer_limits


def get_network_type(network_name):
    if network_name == 'iowa/':
        network_type = 'iowa'
    elif network_name == 'sacramento/':
        network_type = 'IEEE'
    elif network_name == 'arizona/':
        network_type = 'IEEE'
    elif network_name == 'vermont/':
        network_type = 'vermont'
    elif network_name == 'feeder37/':
        network_type = 'IEEE_short'
    else:
        network_type = 'SFO'

    return network_type


def copy_loadfile(file_orig, file_copy):
    from shutil import copyfile
    copyfile(file_orig, file_copy)
    return True


def get_load_bases(path_network, load_fname, network_type):
    if network_type == 'SFO':
        idx = 8
        sep = ' '
    elif network_type == 'IEEE':
        idx = 7
        sep = ' '
    elif network_type == 'iowa':
        idx = 6
        sep = ' '
    elif network_type == 'vermont':
        idx = 5
        sep = '\t'
    elif network_type == 'IEEE_short':
        idx = 6
        sep = ' '
    else:
        print('network type not recognized')
        idx = 8
        sep = ' '

    loadfile = pd.read_csv(path_network + load_fname, sep=sep, header=None, usecols=range(9), engine='python')
    loads_real_str = np.array(loadfile[idx], dtype=str)  # 6 for Iowa, 7 for IEEE, 8 for SFO, 5 for Vermont
    # loads_imag_str  = loadfile[8]  # 8 for Iowa and IEEE, 9 for SFO, 6 for Vermont

    loads = np.array([l[3:] for l in loads_real_str], dtype=float)

    return loads


def getSystemY(dssCircuit):
    systemY = dssCircuit.SystemY

    n_nodes = np.sqrt(systemY.size / 2)

    n_nodes = int(n_nodes)
    Y_real = np.zeros(n_nodes ** 2)
    Y_imag = np.zeros(n_nodes ** 2)
    for i in range(int(systemY.size / 2)):
        Y_real[i] = systemY[2 * i]
        Y_imag[i] = systemY[2 * i + 1]

    Y_real = np.reshape(Y_real, (n_nodes, n_nodes))
    Y_imag = np.reshape(Y_imag, (n_nodes, n_nodes))

    return Y_real, Y_imag, n_nodes


def runPF(path_network, master_fname, dssObj):
    # Run the PF to initialize values
    # dssObj.Text.Command = "compile " + path_network + master_fname
    start_compilation = time.time()
    dssObj.Text.Command = "compile " + os.path.join(path_network, master_fname)
    print('Compilation time: ', (time.time() - start_compilation))
    dssObj.Text.Command = 'calcv'
    start_solve = time.time()
    dssObj.ActiveCircuit.Solution.Solve()
    print('Solve time: ', (time.time() - start_solve))

    return True


def export_taps(path_network, master_fname, dssObj):
    dssObj.Text.Command = "Export Taps"
    dssObj.Text.Command = "Export powers"
    #dssObj.Text.Command = "Export voltages"

    return True


def get_tap_changes_filename(name):
    import fnmatch
    path = name
    for filename in os.listdir(path):
        if fnmatch.fnmatch(filename, '*EXP_Taps.csv'):
            #print('found file with name', filename)
            return name + filename
        elif fnmatch.fnmatch(filename, '*EXP_Taps.CSV'):
            # print('found file with name', filename)
            return name + filename
    print('failed to find file')
    return False

def get_export_files(name_network_path):
    import fnmatch
    exp_taps_path, exp_power_path = None, None
    tabs_string = '*EXP_Taps.csv'
    power_string = '*EXP_POWERS.csv'
    for fname in os.listdir(name_network_path):
        if fnmatch.fnmatch(fname.lower(), tabs_string.lower()):
            exp_taps_path = os.path.join(name_network_path, fname)
        elif fnmatch.fnmatch(fname.lower(), power_string.lower()):
            exp_power_path = os.path.join(name_network_path, fname)
    if exp_taps_path is None:
        raise Exception(f'failed to find EXP_TAPS.csv file in directory {name_network_path}')
    if exp_power_path is None:
        raise Exception(f'failed to find EXP_POWERS.csv file in directory {name_network_path}')
    return exp_taps_path, exp_power_path


def get_tap_changes(fname, taps_prev=np.nan):
    loadfile = pd.read_csv(fname, sep=',')
    taps = loadfile[' Position']
    if np.any(np.isnan(taps_prev)):
        tap_change = np.zeros(taps.shape)
    else:
        tap_change = np.abs(taps - taps_prev)
    taps_prev = taps

    return tap_change, taps_prev


def get_t_power(fname):
    loadfile = pd.read_csv(fname, sep=',')
    ts = loadfile[loadfile['Element'].str.contains("Transformer")]
    t_apparent_power = np.sqrt(ts[' P(kW)'] ** 2 + ts[' Q(kvar)'] ** 2)

    # make sign of apparent power equal to sign of real power component to show direction of real power flow
    t_apparent_power = t_apparent_power * np.sign(ts[' P(kW)'])

    return ts[' P(kW)'], ts[' Q(kvar)'], t_apparent_power


def t_input_powers(t_real, t_reac, t_apparent_power):
    return t_real[::2], t_reac[::2], t_apparent_power[::2]


def get_t_power_filename(name):
    import fnmatch
    path = name
    for filename in os.listdir(path):
        if fnmatch.fnmatch(filename, '*EXP_POWERS.csv'):
            #print('found file with name', filename)
            return name + filename
        elif fnmatch.fnmatch(filename, '*EXP_POWERS.CSV'):
            # print('found file with name', filename)
            return name + filename
    print('failed to find file')
    return False


def get_VmagsPu(dssCircuit):
    # Units are PU not Volts
    Vmag = dssCircuit.AllBusVmagPu

    return Vmag


def updateLoads_3ph(path_network, load_fname, loads_real_new, loads_imag_new, network_type):
    if network_type == 'SFO':
        idx_r = 8
        idx_i = 9
        sep = ' '
    elif network_type == 'IEEE':
        idx_r = 7
        idx_i = 8
        sep = ' '
    elif network_type == 'iowa':
        idx_r = 6
        idx_i = 8
        sep = ' '
    elif network_type == 'vermont':
        idx_r = 5
        idx_i = 6
        sep = '\t'
    elif network_type == 'IEEE_short':
        idx_r = 6
        idx_i = 7
        sep = ' '
    else:
        print('network type not recognized')
        idx_r = 8
        idx_i = 9
        sep = ' '

    # updates the kW and kvar value of existing loads
    # loadfile = pd.read_csv(path_network + load_fname, sep=sep, header=None, usecols=range(9), engine='python')
    # loadfile = pd.read_csv(os.path.join(path_network,load_fname), sep=sep, header=None, usecols=range(9), engine='python')
    loadfile = pd.read_csv(os.path.join(path_network,load_fname), sep=sep, header=None, engine='python')

    loads_real_str = ['kW=' + item for item in np.array(loads_real_new, dtype=str)]
    loads_imag_str = ['kvar=' + item for item in np.array(loads_imag_new, dtype=str)]

    name_str = ['"' + item + '"' for item in np.array(loadfile[1], dtype=str)]
    name_str = np.array(name_str, dtype=str)

    loadfile[1] = name_str  # 1 for Iowa and IEEE and SFO
    loadfile[idx_r] = loads_real_str  # 6 for Iowa, 7 for IEEE, 8 for SFO, 5 for Vermont
    loadfile[idx_i] = loads_imag_str  # 8 for Iowa and IEEE, 9 for SFO, 6 for Vermont

    # loadfile.to_csv(path_network + load_fname, sep=sep, header=None, index=False,
    #                 quoting=3, quotechar="", escapechar="\\")
    loadfile.to_csv(os.path.join(path_network,load_fname), sep=sep, header=None, index=False,
                quoting=3, quotechar="", escapechar="\\")

    return True


def clean_voltages(v_profile, v_max, v_min, align=False):
    if v_profile.ndim == 1:
        connected = v_profile > 0.1
        v_profile = v_profile[connected]
    else:
        connected = np.min(v_profile, axis=1) > 0.01
        v_profile = v_profile[connected, :]
    v_max = v_max[connected]
    v_min = v_min[connected]

    # align substation transformer tap
    if align:
        v_profile = v_profile - np.mean(v_profile[0, :]) + 1

    return v_profile, v_max, v_min


if __name__ == '__main__':
    #name = 'rural_san_benito/'
    name = 'tracy/'
    network_name  = name + 'network/'
    network_type = get_network_type(name)
    load_fname_orig = 'Loads_orig.dss'
    load_fname = 'Loads.dss'
    master_fname = 'Master.dss'
    #print(network_name)

    # get default network loads
    copy_loadfile(network_name + load_fname_orig, network_name + load_fname)
    demand = get_load_bases(network_name, load_fname, network_type)
    print('demand shape', demand.shape)

    # Initialize openDSS
    dssObj = dss.DSS
    dssCircuit = dssObj.ActiveCircuit
    # Run PF with initial values
    runPF(network_name, master_fname, dssObj)
    network_name = os.getcwd() + '/'

    # Get system Y matrix
    Y_real, Y_imag, n_nodes = getSystemY(dssCircuit)
    #print('Real part of Y matrix', Y_real)
    #print('number of nodes according to Y matrix', n_nodes)

    # export_taps writes the data files that contain info about transformer power and tap changes
    export_taps(network_name, master_fname, dssObj)
    tap_fname = get_tap_changes_filename(network_name)
    t_fname = get_t_power_filename(network_name)

    # read tap changes
    taps_prev = np.nan
    tap_changes, taps_prev = get_tap_changes(tap_fname, taps_prev=np.nan)

    # read transformer powers only the input power
    t_real, t_reac, t_apparent_power = get_t_power(t_fname)
    t_real, t_reac, t_apparent_power = t_input_powers(t_real, t_reac, t_apparent_power)
    #print('transformers shape', t_real.shape)

    # get voltages
    v_mags = get_VmagsPu(dssCircuit)

    # Print network characteristics
    print('N = ', v_mags.shape)
    print('Nc = ', demand.shape)
    print('N transformers = ', t_apparent_power.shape)

    ### make and test new loads
    loads_real_new = demand * 1.9
    loads_imag_new = demand * 0.3

    # update openDSS load file
    updateLoads_3ph(network_name, load_fname, loads_real_new, loads_imag_new, network_type)

    ### test that changes were successful
    runPF(network_name, master_fname, dssObj)
    export_taps(network_name, master_fname, dssObj)
    t_real, t_reac, t_apparent_power2 = get_t_power(t_fname)
    t_real, t_reac, t_apparent_power2 = t_input_powers(t_real, t_reac, t_apparent_power2)
    v_mags2 = get_VmagsPu(dssCircuit)
    v_max = 1.05 * np.ones(v_mags2.shape)
    v_min = 0.95 * np.ones(v_mags2.shape)
    v_mags2, v_max, v_min = clean_voltages(v_mags2, v_max, v_min)

    print('total change in v mag', np.sum(v_mags2 - v_mags))
    print('total change in transformer mag', np.sum(t_apparent_power2 - t_apparent_power))
