# 2.83 4.37 5.98
# 3.04 4.55 6.34 (Wi-Fi only)

import argparse
import numpy
import joblib

from sklearn import svm
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor

from _config import *
from _dataset import *
from _model import *


def recover(est):
    res = numpy.zeros_like(est)
    res[:, 0] = est[:, 0] * 50
    res[:, 1] = est[:, 1] * 38
    return res


def getMAE(esY, teY):
    length = esY.shape[0]
    res = numpy.empty((length,))
    for i in range(length):
        res[i] = mean_absolute_error(esY[i], teY[i])
    return res


parser = argparse.ArgumentParser(description="Data choice")
parser.add_argument("type", type=str, choices=['w', 'a'], help="the type of data used by this script (w=wifi only, a=all, i.e., wifi+ble")
args = parser.parse_args()

trFileName = ''
teFileName = ''
if args.type == 'w':
    trFileName = 'DatasetITR_WF'
    teFileName = 'DatasetITE_WF'
else:
    trFileName = 'DatasetITR'
    teFileName = 'DatasetITE'
trX, trY = joblib.load(TEMP_DIR + trFileName)
teX, teY = joblib.load(TEMP_DIR + teFileName)

trX = torch.reshape(trX, (trX.size()[0], trX.size()[1]*trX.size()[2]))
teX = torch.reshape(teX, (teX.size()[0], teX.size()[1]*teX.size()[2]))
teY = teY.numpy()

model = MultiOutputRegressor(svm.SVR())
model.fit(trX, trY)
esY = model.predict(teX)

print(numpy.quantile(getMAE(recover(esY), recover(teY)), [0.25, 0.5, 0.75]))
