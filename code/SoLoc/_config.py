import math

DATA_DIR = '../../data/SoLoc_IPIN2017_dataset/'
TEMP_DIR = './temp/'

MAX_DIST = math.sqrt(1 + 1)

E_WFWF = ('WF', 'bidirectWF', 'WF')
E_BTBT = ('BT', 'bidirectBT', 'BT')
E_WFTD = ('WF', 'direct', 'TD')
E_BTTD = ('BT', 'direct', 'TD')

N_WIFI = 11
N_BLUE = 46

# WiFi only
# EDGE_TYPES = ['bidirectWF']
# N_AP = N_WIFI

# WiFi + Bluetooth
EDGE_TYPES = ['bidirectWF', 'bidirectBT']
N_AP = N_WIFI + N_BLUE

