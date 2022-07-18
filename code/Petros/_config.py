import math


DATA_DIR = '../../data/Petros_dataset/Scenario'
TEMP_DIR = './temp/'

MAX_DIST = math.sqrt(1 + 1)

N_ZB = 3
N_BT = 3
N_WF = 3

E_WFWF = ('WF', 'bidirectWF', 'WF')
E_BTBT = ('BT', 'bidirectBT', 'BT')
E_ZBZB = ('ZB', 'bidirectZB', 'ZB')
E_WFTD = ('WF', 'direct', 'TD')
E_BTTD = ('BT', 'direct', 'TD')
E_ZBTD = ('ZB', 'direct', 'TD')


# Bluetooth only
# EDGE_TYPES = ['bidirectBT']
# N_TOTAL_AP = N_BT

# WiFi only
# EDGE_TYPES = ['bidirectWF']
# N_TOTAL_AP = N_WF

# all together
EDGE_TYPES = ['bidirectZB', 'bidirectBT', 'bidirectWF']
N_TOTAL_AP = N_ZB + N_BT + N_WF
