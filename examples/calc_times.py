import multiprocessing
import time
from sklearn.metrics import average_precision_score
from pyod.test.testhbosold import HBOSOLD
import h5py
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from pyod.models.hbospyod import HBOSPYOD
from pyod.utils.data import generate_data
from pyod.models.hbos import HBOS
from matplotlib import pyplot as plt
import matplotlib.cm as cm

import pandas as pd
from scipy.io import loadmat, arff
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from pyod.utils import precision_n_scores

dataset13 = pd.read_csv(r"C:\Users\david\Desktop\datasets\creditcard.csv")
orig13 = dataset13.copy()
dataset13label = dataset13['Class']
del dataset13['Time']
del dataset13['Amount']
del dataset13['Class']

harvard9 = pd.read_csv(r'C:\Users\david\Desktop\datasets_hbos\Harvard Dataverse\shuttle-unsupervised-ad.csv',
                       header=None)
lastcol = harvard9.columns[-1]
harvard9.rename(columns={lastcol: 'Class'}, inplace=True)
harvard9label = harvard9['Class']
harvardorig9 = harvard9.copy()
del harvard9['Class']

mat_data10 = loadmat(r"C:\Users\david\Desktop\datasets_hbos\ODDS\Dataset\musk.mat")
dataset10 = pd.DataFrame(mat_data10['X'])
dataset10["Class"] = mat_data10['y']
dataset10label = mat_data10['y']
orig10 = dataset10.copy()
del dataset10['Class']

mat_data7 = loadmat(r"C:\Users\david\Desktop\datasets_hbos\ODDS\Dataset\lympho.mat")
dataset7 = pd.DataFrame(mat_data7['X'])
dataset7["Class"] = mat_data7['y']
dataset7label = mat_data7['y']
orig7 = dataset7.copy()
del dataset7['Class']

elki_semantic6, meta = arff.loadarff(
    r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\InternetAds\InternetAds_norm_02_v01.arff')
elki_semantic6 = pd.DataFrame(elki_semantic6)
elki_semanticorig6 = elki_semantic6.copy()
elki_semanticorig6['Class'] = elki_semanticorig6['Class'].astype(int)
del elki_semantic6['id']
del elki_semantic6['Class']
elki_semantic6label = elki_semanticorig6['Class']

dateset= dataset13
print(len(dateset))
print(dateset.shape[1])

time1=[]
for i in range(10):
    clf = HBOSPYOD(mode="static", save_explainability_scores=False)
    start_time = time.time()
    clf.fit(dateset)
    end_time = time.time()
    scores = clf.decision_scores_
    time1.append( end_time - start_time)
print(len(time1))
print("Time taken hbos new, static: ", np.average(time1))



time2=[]
for i in range(10):
    clf2 = HBOSPYOD(mode="dynamic", save_explainability_scores=False)
    # clf = HBOSOLD()
    start_time = time.time()
    clf2.fit(dateset)
    end_time = time.time()
    time2.append(end_time - start_time)
print("Time taken hbos new, dynamic: ", np.average(time2))




time4 = []
for i in range(10):
    list = 'static binwidth' * dateset.shape[1]
    clf4 = HBOSOLD(mode_array=list)
    start_time = time.time()
    scores2 = clf4.fit_predict(dateset)
    end_time = time.time()
    time4.append(end_time - start_time)
print("Time taken hbos old, static: ", np.average(time4))

time3=[]
for i in range(10):
    clf3 = HBOSOLD()
    start_time = time.time()
    clf3.fit_predict(dateset)
    end_time = time.time()
    time3.append( end_time - start_time)
print("Time taken hbos old, dynamic: ", np.average(time3))
