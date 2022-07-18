# Scenario     MSE     Var
# S1           1.1170  0.4164
# S1 (BT only) 1.2946  0.6558
# S3           0.6825, 0.3745
# S3 (WF only) 0.7731  0.3299

import argparse
import joblib

from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

from _config import *
from _dataset import *
from _model import *


def recover(est, mulX, mulY):
    est[:, 0] *= mulX
    est[:, 1] *= mulY
    return est


def calMSE(a, b):
    res = []
    for i in range(a.size(0)):
        res.append(mean_squared_error(a[i].tolist(), b[i].tolist()))
    return res


def train(device, net, dataloader, optimizer, criterion):
    net.train()
    totalNum = 0
    totalLoss = 0
    for data, label in dataloader:
        # batch graphs will be shipped to device in forward part of model
        data = data.to(device)
        label = label.to(device)
        outputs = net(data)
        loss = criterion(outputs, label)
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # record loss
        totalNum += len(label)
        totalLoss += loss.item() * len(label)
    loss = 1.0 * totalLoss / totalNum
    return loss


def eval_net(device, net, dataloader, criterion):
    net.eval()
    total = 0
    total_loss = 0
    err = []
    for data, label in dataloader:
        data = data.to(device)
        label = label.to(device)
        total += len(label)
        outputs = net(data)
        loss = criterion(outputs, label)
        err += calMSE(recover(outputs, ORI_X, ORI_Y), recover(label, ORI_X, ORI_Y))
        total_loss += loss.item() * len(label)
    loss = 1.0 * total_loss / total
    net.train()
    err = torch.tensor(err)
    return loss, torch.mean(err), torch.var(err)


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
model = CNNRegressor(128)
trFileName = f'DatasetITR_S{args.scenario_idx}'
teFileName = f'DatasetITE_S{args.scenario_idx}'
if args.type == 'b':
    trFileName = f'DatasetITR_S{args.scenario_idx}_BT'
    teFileName = f'DatasetITE_S{args.scenario_idx}_BT'
elif args.type == 'w':
    trFileName = f'DatasetITR_S{args.scenario_idx}_WF'
    teFileName = f'DatasetITE_S{args.scenario_idx}_WF'
elif args.type == 'a':
    model = CNNRegressor(128, 3)


if __name__ == '__main__':

    device = torch.device("cpu")

    trX, trY = joblib.load(TEMP_DIR + trFileName)
    teX, teY = joblib.load(TEMP_DIR + teFileName)
    trX = trX.unsqueeze(1).transpose(2, 3)
    teX = teX.unsqueeze(1).transpose(2, 3)

    trLoader = DataLoader(PetrosImageDataset(trX, trY), batch_size=32, drop_last=False, shuffle=True)
    teLoader = DataLoader(PetrosImageDataset(teX, teY), batch_size=32, drop_last=False, shuffle=True)

    epochs = 500
    criterion = torch.nn.HuberLoss()  # default reduce is true
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.75, patience=30, min_lr=1e-7, verbose=True)

    bestMSE = float('inf')
    bestVAR = float('inf')
    for epoch in range(epochs):
        trLoss = train(device, model, trLoader, optimizer, criterion)
        vaLoss, mse, var = eval_net(device, model, teLoader, criterion)
        scheduler.step(vaLoss)
        if mse < bestMSE:
            bestMSE = mse
            bestVAR = var
        print(f'# epoch {epoch}\tTRAIN_LOSS: {trLoss:.6f}\tVALID_LOSS: {vaLoss:.6f}\tBEST MODEL: MSE={bestMSE:.4f}, var={bestVAR:.4f}')
