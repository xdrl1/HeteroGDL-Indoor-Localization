# model             lq      median  uq
# homo + pool       2.70    3.87    6.02

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
        output = net(data)
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
        output = net(data)
        loss = criterion(output, label)
        err += calMAE(recover(output, 50, 38), recover(label, 50, 38))
        total += len(label)
        total_loss += loss.item() * len(label)
    return 1.0 * total_loss / total, torch.quantile(torch.tensor(err, dtype=torch.float32), torch.tensor([0.75, 0.5, 0.25]), dim=0)


if __name__ == '__main__':
    warnings.simplefilter("ignore")

    device = torch.device("cpu")

    trDataset = joblib.load(TEMP_DIR + 'DatasetHoGTR')
    teDataset = joblib.load(TEMP_DIR + 'DatasetHoGTE')

    trLoader = GraphDataLoader(trDataset, batch_size=32, drop_last=False, shuffle=True)
    teLoader = GraphDataLoader(teDataset, batch_size=32, drop_last=False, shuffle=True)

    model = HomoGCN(3, 256)

    epochs = 500
    criterion = torch.nn.HuberLoss()
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
