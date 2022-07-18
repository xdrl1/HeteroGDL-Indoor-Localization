# Scenario   MSE     Var
# S1         1.4224  0.6170
# S3         0.6961  0.2671

import argparse
import numpy
import joblib

from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor

from _config import *
from _dataset import *
from _model import *


def recover(est, mulX, mulY):
    est[:, 0] *= mulX
    est[:, 1] *= mulY
    return est


def getMSE(esY, teY):
    length = esY.shape[0]
    res = numpy.empty((length,))
    for i in range(length):
        res[i] = mean_squared_error(esY[i], teY[i])
    return res


parser = argparse.ArgumentParser(description="Choose the scenario")
parser.add_argument("scenario_idx", type=int, choices=[1, 3], help="Choose the scenario by index (1 or 3)")
parser.add_argument("type", type=str, choices=['b', 'w', 'a'], help="the type of data used by this script (b=bluetooth, w=wifi only, a=all, i.e., wifi+ble+zigbee")
args = parser.parse_args()
ORI_X = 0
ORI_Y = 0
if args.scenario_idx == 1:
    ORI_X = 4
    ORI_Y = 4
else:
    ORI_X = 9.625
    ORI_Y = 2.492
trFileName = f'DatasetITR_S{args.scenario_idx}'
teFileName = f'DatasetITE_S{args.scenario_idx}'
if args.type == 'b':
    trFileName = f'DatasetITR_S{args.scenario_idx}_BT'
    teFileName = f'DatasetITE_S{args.scenario_idx}_BT'
elif args.type == 'w':
    trFileName = f'DatasetITR_S{args.scenario_idx}_WF'
    teFileName = f'DatasetITE_S{args.scenario_idx}_WF'


if __name__ == '__main__':
    trX, trY = joblib.load(TEMP_DIR + trFileName)
    teX, teY = joblib.load(TEMP_DIR + teFileName)

    trX = torch.reshape(trX, (trX.size()[0], trX.size()[1]*trX.size()[2]))
    teX = torch.reshape(teX, (teX.size()[0], teX.size()[1]*teX.size()[2]))
    teY = teY.numpy()

    model = MultiOutputRegressor(KNeighborsRegressor(weights='distance'))
    model.fit(trX, trY)
    esY = model.predict(teX)

    mse = getMSE(recover(esY, ORI_X, ORI_Y), recover(teY, ORI_X, ORI_Y))
    print(numpy.mean(mse), numpy.var(mse))
