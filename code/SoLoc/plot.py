import numpy

from matplotlib import pyplot as plt



res1 = {
    # algo #               # lq  median  uq #
    'GraphSAGE-LSTM-edge': [2.27, 3.51, 5.12],
    'GraphSAGE-pool-edge': [2.22, 3.54, 5.22],
    'GraphSAGE-pool':      [2.36, 3.55, 5.16],
    'CNN':                 [2.54, 3.99, 5.69],
    'SVM':                 [2.79, 4.18, 6.07],
    'MLP':                 [3.04, 4.55, 6.34],
    'weighted k-NN':       [3.31, 5.08, 6.76],
    'Error Bound':         [3.78, 5.81, 8.94],
    'SoLoc':               [4.03, 6.51, 11.01],
}

res2 = {
    # algo #                     # lq  median  uq #
    'HeteroGraphSAGE-LSTM-edge': [1.91, 2.98, 4.34],
    'HeteroGraphSAGE-pool-edge': [1.94, 2.96, 4.44],
    'HeteroGraphSAGE-pool':      [2.18, 3.23, 4.60],
    'HomoGraphSAGE-pool':        [2.70, 3.87, 6.02],
    'CNN':                       [2.31, 3.49, 5.13],
    'SVM':                       [2.83, 4.37, 5.98],
    'MLP':                       [2.52, 3.90, 5.66],
    'weighted k-NN':             [2.81, 4.15, 5.92],
    'Error Bound':               [2.39, 3.87, 5.58],
    'SoLoc':                     [2.65, 4.31, 5.99],
}

# plt.figure()
plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.subplot(121)
i = 0
normColor = 'blue'
goodColor = 'crimson'
for algo in res1.keys():
    if i < 3:
        plt.hlines(i, res1[algo][0], res1[algo][2], color=goodColor)
        plt.vlines(res1[algo][0], i-0.1, i+0.1, color=goodColor)
        plt.vlines(res1[algo][1], i-0.1, i+0.1, color=goodColor)
        plt.vlines(res1[algo][2], i-0.1, i+0.1, color=goodColor)
    else:
        plt.hlines(i, res1[algo][0], res1[algo][2], color=normColor)
        plt.vlines(res1[algo][0], i-0.1, i+0.1, color=normColor)
        plt.vlines(res1[algo][1], i-0.1, i+0.1, color=normColor)
        plt.vlines(res1[algo][2], i-0.1, i+0.1, color=normColor)
    i += 1
plt.xlim([0, 11.5])
plt.yticks(range(9), res1.keys(), fontsize=16)
plt.grid(axis='x')
plt.title('WiFi only', fontsize=16)
plt.xlabel('MAE (m) on the testing data', fontsize=16)

plt.subplot(122)
i = 0
for algo in res2.keys():
    if i < 3:
        plt.hlines(i, res2[algo][0], res2[algo][2], color=goodColor)
        plt.vlines(res2[algo][0], i-0.1, i+0.1, color=goodColor)
        plt.vlines(res2[algo][1], i-0.1, i+0.1, color=goodColor)
        plt.vlines(res2[algo][2], i-0.1, i+0.1, color=goodColor)
    else:
        plt.hlines(i, res2[algo][0], res2[algo][2], color=normColor)
        plt.vlines(res2[algo][0], i-0.1, i+0.1, color=normColor)
        plt.vlines(res2[algo][1], i-0.1, i+0.1, color=normColor)
        plt.vlines(res2[algo][2], i-0.1, i+0.1, color=normColor)
    i += 1
plt.xlim([0, 8])
plt.yticks(range(10), res2.keys(), fontsize=16)
plt.grid(axis='x')
plt.title('Hybrid RSSI (WiFi + Bluetooth)', fontsize=16)
plt.xlabel('MAE (m) on the testing data', fontsize=16)

plt.show()
