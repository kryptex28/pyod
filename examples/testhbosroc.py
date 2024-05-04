from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from pyod.models.hbospyod import HBOSPYOD
from pyod.utils.data import generate_data
from pyod.models.hbos import HBOS
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.cm as cm

import pandas as pd
from scipy.io import loadmat, arff
import numpy as np

from pyod.utils import precision_n_scores

def calc_auc_graph(data_,orig_,count,mode_,ranked_):
    aucs = []
    for i in range(count):
        clf2=HBOSPYOD()
        bins=i+1
        clf2.set_params(n_bins=bins,mode=mode_,ranked=ranked_)
        clf2.fit(data_)
        scores = clf2.decision_scores_
        hbos_orig = orig_
        hbos_orig['scores'] = scores
        hbos_orig_sorted = hbos_orig.sort_values(by=['scores'], ascending=False)
        fpr, tpr, thresholds = metrics.roc_curve(hbos_orig_sorted['Class'], hbos_orig_sorted['scores'])
        auc = metrics.auc(fpr, tpr)
        aucs.append(auc)


    xval= range(1,count+1)

    plt.figure(figsize=[8, 5])
    plt.plot(xval, aucs, color='b', lw=2, label='AUC vs. n_bins')
    plt.xlabel('Number of Bins (n_bins)')
    plt.ylabel('Area Under the Curve (AUC)')
    plt.title('AUC vs. Number of Bins, Dataset 4, static  HBOS')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

def plot_explainability(id):

    y_values = clf.get_explainability_scores(id)
    print(y_values)

    # Labels erstellen
    labels = ['Feature: {}'.format(i + 1) for i in range(clf.n_features_)]

    print(labels)
    print(y_values)

    colors = cm.RdYlGn_r(y_values / (
                np.max(y_values) ))  # Skalieren der Werte auf [0, 1] und Farbskala umkehren
    plt.figure(figsize=[10, 8])
    # plt.bar(labels, y_values, align='center', color=colors, label="test")
    plt.barh(np.arange(len(y_values)), y_values, color=colors, tick_label=labels)

    plt.xlabel('score')

    plt.title(
        'Outlier score for sample: {}'.format(id) + ' with outlierscore = {0:0.4f}'.format(clf.decision_scores_[id]))
    plt.legend(loc="lower right")
    plt.show()


def calc_roc_auc(orig_):

    clf_name = 'HBOS'

    scores = clf.decision_scores_
    hbos_orig = orig_
    hbos_orig['scores'] = scores
    hbos_orig_sorted = hbos_orig.sort_values(by=['scores'], ascending=False)
    print(hbos_orig_sorted)
    # auc = roc_auc_score(hbos_orig_sorted['Class'], hbos_orig_sorted['scores'])
    fpr, tpr, thresholds = metrics.roc_curve(hbos_orig_sorted['Class'], hbos_orig_sorted['scores'])
    plt.figure(figsize=[8, 5])
    plt.plot(fpr, tpr, color='r', lw=2, label=clf_name)
    plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--', label='guessing')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    auc = metrics.auc(fpr, tpr)
    print(auc)
    plt.title('Receiver operating characteristic: HBOS_AUC ={0:0.4f}'.format(auc))
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":

    np.set_printoptions(threshold=np.inf)

    X_train, y_train = generate_data(n_train=1000, n_features=2, contamination=0.1, random_state=42, train_only=True)
    dataset6 = pd.DataFrame(X_train)
    dataset6['Class'] = y_train


    dataset1 = pd.read_csv(r"C:\Users\david\Desktop\datasets\creditcard.csv")
    dataset3 = pd.read_csv(r"C:\Users\david\Desktop\datasets\pen-local-unsupervised-ad.csv")
    dataset4, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets\literature\Shuttle\Shuttle_withoutdupl_norm_v01.arff')
    dataset4 = pd.DataFrame(dataset4)
    dataset5, meta2 = arff.loadarff(
        r'C:\Users\david\Desktop\datasets\semantic\HeartDisease\HeartDisease_withoutdupl_02_v01.arff')
    dataset5 = pd.DataFrame(dataset5)
    dataset4['Class'] = dataset4['Class'].astype(int)
    dataset5['Class'] = dataset5['Class'].astype(int)
    dataset7, meta7 = arff.loadarff(r'C:\Users\david\Desktop\datasets\literature\ALOI\ALOI.arff')
    dataset7 = pd.DataFrame(dataset7)
    dataset8, meta8 = arff.loadarff(r'C:\Users\david\Desktop\datasets\literature\ALOI\ALOI_norm.arff')
    dataset8 = pd.DataFrame(dataset8)
    dataset7['Class'] = dataset7['Class'].astype(int)
    dataset8['Class'] = dataset8['Class'].astype(int)

    orig1 = dataset1.copy()
    orig3 = dataset3.copy()
    orig4 = dataset4.copy()
    orig5 = dataset5.copy()
    orig6 = dataset6.copy()
    orig7 = dataset7.copy()
    orig8 = dataset8.copy()


    del dataset1['Time']
    del dataset1['Amount']
    del dataset1['Class']

    del dataset3['Class']
    del dataset4['id']
    del dataset4['Class']
    del dataset5['Class']
    del dataset7['id']
    del dataset7['Class']
    del dataset8['id']
    del dataset8['Class']

    clf = HBOSPYOD(save_explainability_scores=True,ranked=False,mode="static")
    data_=dataset1
    orig_=orig1
    print(data_.shape)
    clf.fit(data_)
    #print(clf.decision_scores_)
    #print(np.unique)
    #plot_explainability(366)
    calc_roc_auc(orig_)
    #calc_auc_graph(data_, orig_,10,"static",False)
