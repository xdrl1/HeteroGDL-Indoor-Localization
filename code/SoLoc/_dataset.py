import torch

from dgl.data import DGLDataset


class SoLocGraphDataset(DGLDataset):
    def __init__(self, name, graphs, labels):
        self._graphs = graphs
        self._labels = labels
        super(SoLocGraphDataset, self).__init__(name=name)

    def __len__(self):
        return len(self._graphs)

    def __getitem__(self, idx):
        return self._graphs[idx], self._labels[idx]

    def has_cache(self):
        if len(self._graphs) == 0:
            return False
        else:
            return True

    def graphs(self):
        return self._graphs

    def labels(self):
        return self._labels

    def process():
        pass


class SoLocImageDataset(torch.utils.data.Dataset):
    def __init__(self, image, label):
        self._image = image
        self._label = label
        super(SoLocImageDataset, self).__init__()

    def __len__(self):
        return self._label.size()[0]

    def __getitem__(self, idx):
        return self._image[idx], self._label[idx]