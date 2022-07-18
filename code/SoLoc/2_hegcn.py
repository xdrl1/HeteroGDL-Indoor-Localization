# all results (WiFi+BLE)
# WiFi + Bluetooth           lq      median  uq         # WiFi only         lq      median  uq
# hetero + lstm + edge feat  1.91    2.98    4.34       # lstm + edge feat  2.27    3.51    5.12   *
# hetero + pool + edge feat  1.94    2.96    4.44   *   # pool + edge feat  2.22    3.54    5.22
# hetero + pool              2.18    3.23    4.60       # pool              2.36    3.55    5.16
# homo + pool                2.70    3.87    6.02       # -----------       ----    ----    ----   (mono-source RF induces homographs)
# CNN                        2.31    3.49    5.13       # CNN               2.54    3.99    5.69
# MLP                        2.52    3.90    5.66       # MLP               2.79    4.18    6.07
# SVM                        2.83    4.37    5.98       # SVM               3.04    4.55    6.34
# w-knn                      2.81    4.15    5.92       # w-knn             3.31    5.08    6.76

import argparse
import joblib
import warnings

from dgl.dataloading import GraphDataLoader
from sklearn.metrics import mean_absolute_error

from _config import *
from _dataset import *
from _model import *


def recover(est, mulX, mulY):
    est[:, 0] *= mulX
    est[:, 1] *= mulY
    return est


def calMAE(a, b):
    res = []
    for i in range(a.size(0)):
        res.append(mean_absolute_error(a[i].tolist(), b[i].tolist()))
    return res


def train(device, net, trainloader, optimizer, criterion):
    net.train()
    total = 0
    total_loss = 0
    for data, label in trainloader:
        # batch graphs will be shipped to device in forward part of model
        data = data.to(device)
        label = label.to(device)
        output = net(data, args.edge)
        loss = criterion(output, label)
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # record loss
        total += len(label)
        total_loss += loss.item() * len(label)
    return 1.0 * total_loss / total


def eval_net(device, net, dataloader, criterion):
    net.eval()
    total = 0
    total_loss = 0
    err = []
    for data, label in dataloader:
        data = data.to(device)
        label = label.to(device)
        output = net(data, args.edge)
        loss = criterion(output, label)
        err += calMAE(recover(output, 50, 38), recover(label, 50, 38))
        total += len(label)
        total_loss += loss.item() * len(label)
    return 1.0 * total_loss / total, torch.quantile(torch.tensor(err, dtype=torch.float32), torch.tensor([0.75, 0.5, 0.25]), dim=0)


parser = argparse.ArgumentParser(description="Data choice")
parser.add_argument("--data", type=str, choices=['w', 'a'], help="the type of data used by this script (w=wifi only, a=all, i.e., wifi+ble")
parser.add_argument("--aggr", type=str, choices=['p', 'l'], help="the type of aggregator used by GraphSAGE (p=pooling, l=LSTM)")
parser.add_argument("--edge", default=False, action="store_true", help="add this if using edge features")
args = parser.parse_args()

model = HeteroGCN(3, 256)
trFileName = ''
teFileName = ''
if args.data == 'w':
    trFileName = 'DatasetGTR_WF'
    teFileName = 'DatasetGTE_WF'
    model = HeteroMonoGCN(3, 256)
    if args.aggr == 'l':
        model = HeteroMonoGCN(3, 256, 'lstm')
else:
    trFileName = 'DatasetHeGTR'
    teFileName = 'DatasetHeGTE'
    if args.aggr == 'l':
        model = HeteroGCN(3, 256, 'lstm')


if __name__ == '__main__':
    warnings.simplefilter("ignore")

    device = torch.device("cpu")

    trDataset = joblib.load(TEMP_DIR + trFileName)
    teDataset = joblib.load(TEMP_DIR + teFileName)

    trLoader = GraphDataLoader(trDataset, batch_size=32, drop_last=False, shuffle=True)
    teLoader = GraphDataLoader(teDataset, batch_size=32, drop_last=False, shuffle=True)

    epochs = 500
    criterion = torch.nn.HuberLoss()  # default reduce is true
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.75, patience=30, min_lr=1e-7, verbose=True)

    loQuarMAE = 0
    medianMAE = float('inf')
    upQuarMAE = 0
    for epoch in range(epochs):
        trLoss = train(device, model, trLoader, optimizer, criterion)
        vaLoss, vaMAE = eval_net(device, model, teLoader, criterion)
        vaMAEPoor = vaMAE[0]
        vaMAEMean = vaMAE[1]
        vaMAEGood = vaMAE[2]
        scheduler.step(vaLoss)
        if vaMAEMean < medianMAE:
            loQuarMAE = vaMAEGood
            medianMAE = vaMAEMean
            upQuarMAE = vaMAEPoor
        print(f'# epoch {epoch}\tTRAIN_LOSS: {trLoss:.6f}\tVALID_LOSS: {vaLoss:.6f}\tBEST MODEL: median={medianMAE:.2f}, lo={loQuarMAE:.2f}, up={upQuarMAE:.2f}')
