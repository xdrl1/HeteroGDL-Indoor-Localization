import argparse
import joblib
import pandas
import torch

from _config import *
from _tool import *


def loadFile(filePath, cols, report=True):
    zb = pandas.read_excel(filePath, sheet_name='Zigbee', usecols=cols)
    bt = pandas.read_excel(filePath, sheet_name='BLE', usecols=cols)
    wf = pandas.read_excel(filePath, sheet_name='WiFi', usecols=cols)
    assert(zb.shape == bt.shape == wf.shape)
    zb = torch.from_numpy(zb.values).float()
    bt = torch.from_numpy(bt.values).float()
    wf = torch.from_numpy(wf.values).float()
    assert(torch.equal(zb[:, :2], bt[:, :2]))
    assert(torch.equal(zb[:, :2], wf[:, :2]))
    assert(torch.equal(bt[:, :2], wf[:, :2]))
    if report:
        print('Loaded', zb[:, :2].size()[0], 'entries...')
    return zb[:, :2], zb[:, 2:], bt[:, 2:], wf[:, 2:]


parser = argparse.ArgumentParser(description="Choose the scenario")
parser.add_argument("scenario_idx", type=int, choices=[1, 3], help="Choose the scenario by index (1 or 3)")
args = parser.parse_args()

if __name__ == '__main__':
    reload_directory(TEMP_DIR)
    idx = args.scenario_idx
    print(f'Loading data from scenario {idx}...')
    joblib.dump(loadFile(DATA_DIR + f'{idx}/Database_Scenario{idx}.xlsx', 'C:G'), TEMP_DIR + 'rawTR')
    joblib.dump(loadFile(DATA_DIR + f'{idx}/Tests_Scenario{idx}.xlsx', 'B:F'), TEMP_DIR + 'rawTE')
    print('Data loaded successfully!')
