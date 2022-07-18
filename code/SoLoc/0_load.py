import joblib
import numpy
import pandas

from _config import *
from _tool import *


def str2float(string):
    if ',' in string:
        int_part, dec_part = string.split(',')
        return int(int_part) + float(dec_part)/(pow(10, len(dec_part)))
    else:
        return float(string)

def format_convert(array):
    # convert 2D str data into float type.
    res = numpy.empty_like(array)
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i, j] = str2float(array[i, j])
    return res

def load_from_file(file_name, separator, report=False):
    data = pandas.read_csv(DATA_DIR + file_name, sep = separator, header = None).to_numpy()
    if isinstance(data[0][0], numpy.integer):
        res = data
    else:
        res = format_convert(data)
    if report:
        print('Loaded data shape:', res.shape, file_name)
    return res


if __name__ == '__main__':
    TAR_DIR = TEMP_DIR
    reload_directory(TAR_DIR)
    joblib.dump(load_from_file('APLocs.csv', ';', report=True),         TAR_DIR + 'locAP')
    joblib.dump(load_from_file('BeaconLocs.csv', '\t', report=True),    TAR_DIR + 'locBT')
    joblib.dump(load_from_file('P_Signatures.csv', ';', report=True),   TAR_DIR + 'tr_x')
    joblib.dump(load_from_file('P_Tests.csv', ';', report=True),        TAR_DIR + 'te_x')
    joblib.dump(load_from_file('SignatureLocs.csv', ';', report=True),  TAR_DIR + 'tr_y')
    joblib.dump(load_from_file('TestLocs.csv', ';', report=True),       TAR_DIR + 'te_y')
