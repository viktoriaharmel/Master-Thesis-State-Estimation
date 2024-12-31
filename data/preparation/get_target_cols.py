import pandas as pd
import csv
import gc

data_path = 'path/to/data/'

data = pd.read_csv(data_path + 'dataset_complete_2018_base.csv')

pmu_locations = ['p1rdt4659.1.2.3', 'p1rdt4949.1.2.3', 'p1rdt833_b1_1.1', 'p1rdt1317_b1_1.1.2.3', 'p1rdt1175.3', 'p1rdm6643.1.2', 'p1rdt8002.1.2.3', 'p1rdt7077.1.2.3', 'p1rdt4568lv.1.2', 'p1rdt2829.1.2.3', 'p1rdt4396lv.1.2', 'p1rdt3614-p1rdt4656xx.1.2.3', 'p1rdt4879_b2_1.1', 'p1rdt5032-p1rdt6794x_b1_1.3', 'p1rdt5663.1.2.3', 'p1rdt6284lv.1.2', 'p1rdt5033_b1_1.1.2.3', 'p1rdt4730_b1_1.1', 'p1rdt1069_b2_1.1.2.3', 'p1rdt3810-p1rdt7137x_b1_1.1', 'p1rdt6285-p1rdt7899x.2', 'p1rdt6852.1', 'p1rdt4396-p1rdt831x_b1_1.1', 'p1rdt7136.3', 'p1rdt4401_b1_1.3', 'p1rdt6284.1.2.3', 'p1rdt320.2', 'p1rdm9214.1.2', 'p1rdt5351.3', 'p1rdt1318-p1rdt3520x_b1_1.2', 'p1rdt4728.2', 'p1rdm1612.1.2', 'p1rdt1762.1.2.3', 'p1rdm117.1.2', 'p1rdm759.1.2', 'p1rdt319lv.1.2', 'p1rdt7137.1.2.3', 'p1rdt7136lv.1.2', 'p1rdm7296.1.2', 'p1rdt3243-p1rdt5935x_b1_1.2', 'p1rdt3243.1', 'p1rdt3243-p1rdt8281x_b1_1.1', 'p1rdt1436.1.2.3', 'p1rdt7437_b1_1.1', 'p1rdt4949.2', 'p1rdt830.2', 'p1rdt5031-p1rdt7257x_b1_1.3', 'p1rdt2109-p1rdt833x_b1_1.1', 'p1rdt6853.1', 'p1rdt2110-p1rdt963x_b1_1.2', 'p1rdm12169.1.2', 'p1rdm6959.1.2', 'p1rdt2630-p1rdt831x.3', 'p1rdt2925-p1rdt3614x.3', 'p1rdt2109.1.2.3', 'p1rdt1436lv.1.2', 'p1rdt2830.1.2.3', 'p1rdt441.1', 'p1rdm7458.1.2', 'p1rdm123.1.2', 'p1rdt3616.1.2.3', 'p1rdt1869-p1rdt53x.2', 'p1rdt1761-p1rdt3616x.2', 'p1rdt1760-p1rdt7325x_b2_1.2', 'p1rdt3614-p1rdt7257xx.1.2.3', 'p1rdt834.1.2.3', 'p1rdt1070.1.2.3', 'p1rdm5898.1.2', 'p1rdm9429.1.2', 'p1rdt1318_b2_1.1.2.3', 'p1rdm4531.1.2', 'p1rdt963lv.1.2', 'p1rdt7265_b1_1.1', 'p1rdt5360-p1rdt565x.2']
pmu_locations = [pmu_loc.split('.')[0] for pmu_loc in pmu_locations]

target_cols = []
target_pmu_cols = [col for col in data.columns if any(f'_{loc}_' in col for loc in pmu_locations) and 'target' in col]

# Exclude PMU bus measurements from the target columns
for col in data.columns:
    if 'target' in col and col not in target_pmu_cols:
        target_cols.append(col)

data_reduced = data.drop(columns=target_pmu_cols)

del data
gc.collect()

data_reduced.to_csv(data_path + 'dataset_complete_2018_base_no_pmu_targets.csv')

# Write the list as a single row in a CSV file
with open(data_path + 'target_cols.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(target_cols)
