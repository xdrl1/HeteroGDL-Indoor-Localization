# Heterogeneous GDL Indoor Localization IPIN2022

This is the document for the code used in the following paper: 

Xuanshu Luo and Nirvana Meratnia. "A Geometric Deep Learning Framework for Accurate Indoor Localization" *2022 International Conference on Indoor Positioning and Indoor Navigation (IPIN)*. IEEE, 2022.

## 1. Environment Setup

The code is fully tested in **CPU-only** environment running Windows 11 Pro 22H2. The following steps should also be applicable in Linux environments.

1. Install `miniconda`. Please see [Miniconda &mdash; Conda documentation](https://docs.conda.io/en/latest/miniconda.html)

2. Setup the environment. 
   
   ```
   conda env create -f env.yaml
   ```
   
   Note that the environment name is `ipin2022` by default. You can edit the `name` field in the `env.yaml` file to specify the env name.

3. Activate the environment.
   
   ```
   conda activate ipin2022
   ```
   
   If you have changed the env name to `your_env_name`, then
   
   ```
   conda activate your_env_name
   ```

## 2. Obtaining Results

In this paper, two datasets are considered.

1. `SoLoc`: https://www.utwente.nl/en/eemcs/ps/dataset-folder/soloc-ipin2017-dataset.zip
   
   <u>Paper</u>: Le, Duc V., and Paul JM Havinga. "SoLoc: Self-organizing indoor localization for unstructured and dynamic environments." *2017 International Conference on Indoor Positioning and Indoor Navigation (IPIN)*. IEEE, 2017.

2. `Petros`: [GitHub - pspachos/RSSI-Dataset-for-Indoor-Localization-Fingerprinting: RSSI dataset for Fingerprinting with Zigbee, BLE and WiFi](https://github.com/pspachos/RSSI-Dataset-for-Indoor-Localization-Fingerprinting) 
   
   <u>Paper</u>: S. Sadowski, P. Spachos, K. Plataniotis, "Memoryless Techniques and Wireless Technologies for Indoor Localization with the Internet of Things", *IEEE Internet of Things Journal*.

### 2.1 SoLoc

First go to the code directory for the `SoLoc` dataset.

```
cd ./code/SoLoc
```

Then load the dataset

```
python 0_load.py
```

#### 2.1.1 WiFi only

First, please ensure that in `./SoLoc/_config.py`, we keep the setting of WiFi only senario and comment out the setting for the other case

```python
# WiFi only
EDGE_TYPES = ['bidirectWF']
N_AP = N_WIFI

# WiFi + Bluetooth
# EDGE_TYPES = ['bidirectWF', 'bidirectBT']
# N_AP = N_WIFI + N_BLUE
```

##### 2.1.1.1 Data generation

In the WiFi only senario, you can generate two kinds of data, i.e., images and graphs using different command line options.

```
python 1_preprocess_wf.py i  (for images)
python 1_preprocess_wf.py g  (for graphs)
```

##### 2.1.1.2 Apply different models

We can get seven results here. The `w` option means the scripts will consider the WiFi only data.

```
python 2_knn.py w                             (wkNN)
python 2_svm.py w                             (SVM)
python 2_mlp.py w                             (MLP)
python 2_cnn.py w                             (CNN)
python 2_hegcn.py --data w --aggr p           (GraphSAGE-pool)
python 2_hegcn.py --data w --aggr p --edge    (GraphSAGE-pool-edge)
python 2_hegcn.py --data w --aggr l --edge    (GraphSAGE-lstm-edge)
```

#### 2.1.2 All together

First, please ensure that in `./SoLoc/_config.py`, we keep the setting of WiFi+Bluetooth senario and comment out the setting for the WiFi only case.

```python
# WiFi only
# EDGE_TYPES = ['bidirectWF']
# N_AP = N_WIFI

# WiFi + Bluetooth
EDGE_TYPES = ['bidirectWF', 'bidirectBT']
N_AP = N_WIFI + N_BLUE
```

##### 2.1.2.1 Data generation

In this senario, you can generate three kinds of data, i.e., images and homogeneous graphs and heterogeneous graphs using different command line options.

```
python 1_preprocess_all.py i   (for images)
python 1_preprocess_all.py ho  (for homo. graphs)
python 1_preprocess_all.py he  (for hetero. graphs)
```

##### 2.1.2.2 Apply different models

We can get eight results here. The `a` option means the scripts will consider data using WiFi and Bluetooth together.

```
python 2_knn.py a                             (wkNN)
python 2_svm.py a                             (SVM)
python 2_mlp.py a                             (MLP)
python 2_cnn.py a                             (CNN)
python 2_hogcn.py                             (HomoGraphSAGE-pool)
python 2_hegcn.py --data a --aggr p           (HeteroGraphSAGE-pool)
python 2_hegcn.py --data a --aggr p --edge    (HeteroGraphSAGE-pool-edge)
python 2_hegcn.py --data a --aggr l --edge    (HeteroGraphSAGE-lstm-edge)
```

### 2.2 Petros

First go to the code directory for the `Petros` dataset.

```
cd ./code/Petros
```

Note that we only consider the Scenario 1 and 3 in this project.

#### 2.2.1 Scenario 1 (Bluetooth-only & All together)

To load the data in Scenario 1, for example, please run

```
python 0_load.py 1
```

##### 2.2.1.1 Bluetooth only (Scenario 1)

First, please ensure that in `./Petros/_config.py`, we keep the setting of Bluetooth only senario and comment out the setting for other cases.

```python
# Bluetooth only
EDGE_TYPES = ['bidirectBT']
N_TOTAL_AP = N_BT

# WiFi only
# EDGE_TYPES = ['bidirectWF']
# N_TOTAL_AP = N_WF

# all together
# EDGE_TYPES = ['bidirectZB', 'bidirectBT', 'bidirectWF']
# N_TOTAL_AP = N_ZB + N_BT + N_WF
```

###### 2.2.1.1.1 Data generation

In this senario, you can generate two kinds of data, i.e., images and graphs using different command line options.

```
python 1_preprocess_bt.py 1 i    (Scenario 1, images)
python 1_preprocess_bt.py 1 g    (Scenario 1, graphs)
```

###### 2.2.1.1.2 Apply different models

We can get seven results here. The `1` and `b` option means the scripts will consider the Bluetooth only data in Scenario 1.

```
python 2_knn.py 1 b                          (wkNN)
python 2_svm.py 1 b                          (SVM)
python 2_mlp.py 1 b                          (MLP)
python 2_cnn.py 1 b                          (CNN)
python 2_hegcn.py 1 --data b --aggr p        (GraphSAGE-pool)
python 2_hegcn.py 1 --data b --aggr p --edge (GraphSAGE-pool-edge)
python 2_hegcn.py 1 --data b --aggr l --edge (GraphSAGE-lstm-edge)
```

##### 2.2.1.2 All together (Scenario 1)

First, please ensure that in `./Petros/_config.py`, we keep the setting of all three RF signals and comment out the setting for other cases.

```python
# Bluetooth only
# EDGE_TYPES = ['bidirectBT']
# N_TOTAL_AP = N_BT

# WiFi only
# EDGE_TYPES = ['bidirectWF']
# N_TOTAL_AP = N_WF

# all together
EDGE_TYPES = ['bidirectZB', 'bidirectBT', 'bidirectWF']
N_TOTAL_AP = N_ZB + N_BT + N_WF
```

###### 2.2.1.2.1 Data generation

In this senario, you can generate three kinds of data, i.e., images and homogeneous graphs and heterogeneous graphs using different command line options.

```
python 1_preprocess_all.py 1 i     (Scenario 1, images)
python 1_preprocess_all.py 1 ho    (Scenario 1, homo. graphs)
python 1_preprocess_all.py 1 he    (Scenario 1, hetero. graphs)
```

###### 2.2.1.2.2 Apply different models

We can get eight results here. The `1` and `a` option means the scripts will consider the data using all kinds of RF in Scenario 1.

```
python 2_knn.py 1 a                          (wkNN)
python 2_svm.py 1 a                          (SVM)
python 2_mlp.py 1 a                          (MLP)
python 2_cnn.py 1 a                          (CNN)
python 2_hogcn.py 1                          (HomoGraphSAGE-pool)
python 2_hegcn.py 1 --data a --aggr p        (HeteroGraphSAGE-pool)
python 2_hegcn.py 1 --data a --aggr p --edge (HeteroGraphSAGE-pool-edge)
python 2_hegcn.py 1 --data a --aggr l --edge (HeteroGraphSAGE-lstm-edge)
```

#### 2.2.2 Scenario 3 (WiFi-only & All together)

Similarly, for the Scenario 3, please run

```
python 0_load.py 3
```

Note that when generating new data, the existing data will be overwritten.

##### 2.2.2.1 WiFi only (Scenario 3)

First, please ensure that in `./Petros/_config.py`, we keep the setting of WiFi only senario and comment out the setting for other cases.

```python
# Bluetooth only
# EDGE_TYPES = ['bidirectBT']
# N_TOTAL_AP = N_BT

# WiFi only
EDGE_TYPES = ['bidirectWF']
N_TOTAL_AP = N_WF

# all together
# EDGE_TYPES = ['bidirectZB', 'bidirectBT', 'bidirectWF']
# N_TOTAL_AP = N_ZB + N_BT + N_WF
```

###### 2.2.2.1.1 Data generation

In this only senario, you can generate two kinds of data, i.e., images and graphs using different command line options.

```
python 1_preprocess_wf.py 3 i    (Scenario 3, images)
python 1_preprocess_wf.py 3 g    (Scenario 3, graphs)
```

###### 2.2.2.1.2 Apply different models

We can get seven results here. The `3` and `w` option means the scripts will consider the WiFi only data in Scenario 3.

```
python 2_knn.py 3 w                          (wkNN)
python 2_svm.py 3 w                          (SVM)
python 2_mlp.py 3 w                          (MLP)
python 2_cnn.py 3 w                          (CNN)
python 2_hegcn.py 3 --data w --aggr p        (GraphSAGE-pool)
python 2_hegcn.py 3 --data w --aggr p --edge (GraphSAGE-pool-edge)
python 2_hegcn.py 3 --data w --aggr l --edge (GraphSAGE-lstm-edge)
```

##### 2.2.2.2 All together (Scenario 3)

First, please ensure that in `./Petros/_config.py`, we keep the setting of all three RF signals and comment out the setting for other cases.

```python
# Bluetooth only
# EDGE_TYPES = ['bidirectBT']
# N_TOTAL_AP = N_BT

# WiFi only
# EDGE_TYPES = ['bidirectWF']
# N_TOTAL_AP = N_WF

# all together
EDGE_TYPES = ['bidirectZB', 'bidirectBT', 'bidirectWF']
N_TOTAL_AP = N_ZB + N_BT + N_WF
```

###### 2.2.2.2.1 Data generation

In this senario, you can generate three kinds of data, i.e., images and homogeneous graphs and heterogeneous graphs using different command line options.

```
python 1_preprocess_all.py 3 i     (Scenario 3, images)
python 1_preprocess_all.py 3 ho    (Scenario 3, homo. graphs)
python 1_preprocess_all.py 3 he    (Scenario 3, hetero. graphs)
```

###### 2.2.2.2.2 Apply different models

We can get eight results here. The `3` and `a` option means the scripts will consider the data using all kinds of RF in Scenario 3.

```
python 2_knn.py 3 a                          (wkNN)
python 2_svm.py 3 a                          (SVM)
python 2_mlp.py 3 a                          (MLP)
python 2_cnn.py 3 a                          (CNN)
python 2_hogcn.py 3                          (HomoGraphSAGE-pool)
python 2_hegcn.py 3 --data a --aggr p        (HeteroGraphSAGE-pool)
python 2_hegcn.py 3 --data a --aggr p --edge (HeteroGraphSAGE-pool-edge)
python 2_hegcn.py 3 --data a --aggr l --edge (HeteroGraphSAGE-lstm-edge)
```
