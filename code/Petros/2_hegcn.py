# Scenario S1                  MSE     Var                                MSE     Var
# all RF signal                                     # BT only
# hetero + lstm + edge feat    0.9309  0.3913   *   # lstm + edge feat    1.2027  0.6722
# hetero + pool + edge feat    0.9510  0.4096       # pool + edge feat    1.1145  0.6017   *
# hetero + pool                0.9999  0.3275       # pool                1.2702  0.6616
# homo + pool                  1.1175  0.2873       # ----------------------------------
# CNN                          1.1170  0.4164       # CNN                 1.2946  0.6558
# MLP                          1.3833  0.4914       # MLP                 1.3695  0.4938
# SVM                          1.7463  0.9137       # SVM                 1.5159  2.2154
# w-knn                        1.4224  0.6170       # w-knn               1.6814  0.7059   & from the paper

# Scenario S3                  MSE     Var          # Scenario S3         MSE     Var
# all RF signal                                     # WF only
# hetero + lstm + edge feat    0.5808  0.1960       # lstm + edge feat    0.7240  0.3597   *
# hetero + pool + edge feat    0.6058  0.1731       # pool + edge feat    0.7417  0.3425
# hetero + pool                0.6551  0.1761       # pool                0.7602  0.3801
# homo + pool                  0.7841  0.3139       # ----------------------------------
# CNN                          0.6825  0.3745       # CNN                 0.7840  0.2693
# MLP                          0.7393  0.2196       # MLP                 0.9664  0.4975
# SVM                          0.7079  0.3425       # SVM                 1.3804  1.2058
# w-knn                        0.6961  0.2671       # w-knn               1.3856  0.5921   & from the paper

import argparse
import joblib
import warnings

from dgl.dataloading import GraphDataLoader
from sklearn.metrics import mean_squared_error

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
        outputs = net(data, args.edge)
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
        outputs = net(data, args.edge)
        loss = criterion(outputs, label)
        err += calMSE(recover(outputs, ORI_X, ORI_Y), recover(label, ORI_X, ORI_Y))
        total_loss += loss.item() * len(label)
    loss = 1.0 * total_loss / total
    err = torch.tensor(err)
    return loss, torch.mean(err), torch.var(err)


parser = argparse.ArgumentParser(description="Choose the scenario")
parser.add_argument("scenario_idx", type=int, choices=[1, 3], help="Choose the scenario by index (1 or 3)")
parser.add_argument("--data", type=str, choices=['b', 'w', 'a'], help="the type of data used by this script (b=bluetooth, w=wifi only, a=all, i.e., wifi+ble+zigbee")
parser.add_argument("--aggr", type=str, choices=['p', 'l'], help="the type of aggregator used by GraphSAGE (p=pooling, l=LSTM)")
parser.add_argument("--edge", default=False, action="store_true", help="add this if using edge features")

args = parser.parse_args()
ORI_X = 0
ORI_Y = 0
if args.scenario_idx == 1:
    ORI_X = 4
    ORI_Y = 4
else:
    ORI_X = 9.625
    ORI_Y = 2.492
trFileName = f'DatasetHeGTR_S{args.scenario_idx}'
teFileName = f'DatasetHeGTE_S{args.scenario_idx}'
model = HeteroGCN(3, 24)
if args.data == 'b':
    trFileName = f'DatasetHeGTR_S{args.scenario_idx}_BT'
    teFileName = f'DatasetHeGTE_S{args.scenario_idx}_BT'
    model = HeteroMonoGCN(3, 32, 'BT')
    if args.aggr == 'l':
        model = HeteroMonoGCN(3, 32, 'BT', 'lstm')
elif args.data == 'w':
    trFileName = f'DatasetHeGTR_S{args.scenario_idx}_WF'
    teFileName = f'DatasetHeGTE_S{args.scenario_idx}_WF'
    model = HeteroMonoGCN(3, 32, 'WF')
    if args.aggr == 'l':
        model = HeteroMonoGCN(3, 32, 'WF', 'lstm')
elif args.data == 'a':
    if args.aggr == 'l':
        model = HeteroGCN(3, 32, 'lstm')


if __name__ == '__main__':
    warnings.simplefilter("ignore")

    device = torch.device("cpu")

    trDataset = joblib.load(TEMP_DIR + trFileName)
    teDataset = joblib.load(TEMP_DIR + teFileName)

    trLoader = GraphDataLoader(trDataset, batch_size=32, drop_last=False, shuffle=True)
    teLoader = GraphDataLoader(teDataset, batch_size=32, drop_last=False, shuffle=True)

    epochs = 500
    criterion = torch.nn.HuberLoss()  # default reduce is true
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=20, min_lr=1e-7, verbose=True)

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
