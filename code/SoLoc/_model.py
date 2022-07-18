import dgl
import torch
import torch.nn.functional as F

from _config import *
from _hetero import HeteroGraphConv


class HeteroGCN(torch.nn.Module):
    def __init__(self, iDim, hDim, agg='pool', oDim=2):
        super(HeteroGCN, self).__init__()
        self.gcn = HeteroGraphConv({
            eType: dgl.nn.pytorch.conv.SAGEConv(iDim, hDim, agg) for eType in EDGE_TYPES
        }, aggregate='min')
        self.mlpWF = torch.nn.Linear(hDim, hDim)
        self.mlpBT = torch.nn.Linear(hDim, hDim)
        self.mlp = torch.nn.Linear(hDim*len(EDGE_TYPES), hDim)
        self.regressor = torch.nn.Linear(hDim, oDim)

    def forward(self, g, useEdgeFeat=False):
        h = self.gcn(g, g.ndata['nodeFeature'])
        if useEdgeFeat:
            h = self.gcn(g, g.ndata['nodeFeature'], mod_kwargs={'edge_weight': g.edata['interferCoeff']})
        h = {k: F.leaky_relu(v) for k, v in h.items()}

        h['WF'] = F.leaky_relu(self.mlpWF(h['WF']))
        h['BT'] = F.leaky_relu(self.mlpBT(h['BT']))

        with g.local_scope():
            g.ndata['h'] = h
            hWF = dgl.max_nodes(g, 'h', ntype='WF')
            hBT = dgl.max_nodes(g, 'h', ntype='BT')
            h = F.relu(self.mlp(torch.cat((hWF, hBT), 1)))
            return self.regressor(h)


class HeteroMonoGCN(torch.nn.Module):
    # actually almost a homogeneous model, for wifi only
    def __init__(self, iDim, hDim, agg='pool', oDim=2):
        super(HeteroMonoGCN, self).__init__()
        self.gcn = HeteroGraphConv({
            eType: dgl.nn.pytorch.conv.SAGEConv(iDim, hDim, agg) for eType in EDGE_TYPES
        }, aggregate='min')
        self.mlpWF = torch.nn.Linear(hDim, hDim)
        self.mlp = torch.nn.Linear(hDim*len(EDGE_TYPES), hDim)
        self.regressor = torch.nn.Linear(hDim, oDim)

    def forward(self, g, useEdgeFeat=False):
        h = self.gcn(g, g.ndata['nodeFeature'])
        if useEdgeFeat:
            h = self.gcn(g, g.ndata['nodeFeature'], mod_kwargs={'edge_weight': g.edata['interferCoeff']})
        h = {k: F.leaky_relu(v) for k, v in h.items()}

        h['WF'] = F.leaky_relu(self.mlpWF(h['WF']))

        with g.local_scope():
            g.ndata['h'] = h
            hWF = dgl.max_nodes(g, 'h', ntype='WF')
            h = F.relu(self.mlp(hWF))
            return self.regressor(h)



class HomoGCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, n_output=2):
        super(HomoGCN, self).__init__()
        self.conv1 = dgl.nn.pytorch.conv.SAGEConv(in_dim, hidden_dim, 'pool')
        # self.conv2 = dgl.nn.pytorch.conv.SGConv(hidden_dim, hidden_dim)
        self.mlp1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.mlp2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.regressor = torch.nn.Linear(hidden_dim, n_output)

    def forward(self, g):
        # Apply graph convolution and activation.
        h = F.relu(self.conv1(g, g.ndata['x']))
        # h = F.relu(self.conv2(g, h))
        h = F.relu(self.mlp1(h))
        h = F.relu(self.mlp2(h))
        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation
            g.ndata['hg'] = dgl.softmax_nodes(g, 'h')
            final = dgl.max_nodes(g, 'hg')
            return self.regressor(final)


class CNNRegressor(torch.nn.Module):
    def __init__(self, hDim, iChannel=1, oDim=2):
        super(CNNRegressor, self).__init__()
        self.conv1 = torch.nn.Conv2d(iChannel, hDim, kernel_size=(3, 3))
        self.conv2 = torch.nn.Conv2d(hDim, hDim, kernel_size=(1, 3))
        self.mlp1 = torch.nn.Linear(hDim*(N_AP-4), hDim)
        self.mlp2 = torch.nn.Linear(hDim, hDim)
        self.regressor = torch.nn.Linear(hDim, oDim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # flatten all dimensions except batch
        x = torch.flatten(x, 1)
        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))
        return self.regressor(x)


class MLPRegressor(torch.nn.Module):
    def __init__(self, hDim, oDim=2):
        super(MLPRegressor, self).__init__()
        self.mlp1 = torch.nn.Linear(N_AP*3, hDim)
        self.mlp2 = torch.nn.Linear(hDim, hDim)
        self.regressor = torch.nn.Linear(hDim, oDim)

    def forward(self, x):
        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))
        return self.regressor(x)


if __name__ == '__main__':
    pass
    # device = torch.device("cpu")

    # trDataset = joblib.load(TEMP_DIR + 'DatasetGTR')
    # teDataset = joblib.load(TEMP_DIR + 'DatasetGTE')

    # trLoader = GraphDataLoader(trDataset, batch_size=16, drop_last=False, shuffle=True)
    # teLoader = GraphDataLoader(teDataset, batch_size=16, drop_last=False, shuffle=True)

    # model = HeteroGCN(3, 64)

    # for graphs, labels in trLoader:
    #     # batch graphs will be shipped to device in forward part of model
    #     labels = labels.to(device)
    #     graphs = graphs.to(device)
    #     # inputs = graphs.ndata['nodeFeature']
    #     # print(inputs.items())
    #     aaa = model(graphs)
    #     print(aaa.size(), labels.size)
    #     pass

    # device = torch.device("cpu")

    # trX, trY = joblib.load(TEMP_DIR + 'DatasetITR')
    # teX, teY = joblib.load(TEMP_DIR + 'DatasetITE')

    # trX = trX.unsqueeze(1).transpose(2, 3)
    # teX = teX.unsqueeze(1).transpose(2, 3)

    # trLoader = DataLoader(SoLocImageDataset(trX, trY), batch_size=32, shuffle=True)
    # teLoader = DataLoader(SoLocImageDataset(teX, teY), batch_size=32, shuffle=True)

    # model = CNNRegressor(256)

    # for images, labels in trLoader:
    #     images = images.to(device)
    #     labels = labels.to(device)
    #     aaa = model(images)
    #     print(aaa.size(), labels.size)


