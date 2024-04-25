from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from pyod.models.hbospyod import HBOSPYOD
from pyod.models.hbos import HBOS

import pandas as pd
from scipy.io import loadmat, arff
import numpy as np

from pyod.utils import precision_n_scores


def calc_score(dataset_, orig_, n_bins_):
    data_ = np.array(dataset_)

    # Now apply LOF on normalized data ...
    clf_name = 'HBOS'
    clf = HBOSPYOD()
    clf.set_n_bins(n_bins_)
    #clf.set_ranked(True)
    #clf.set_smooth(True)
   #clf.set_mode("dynamic")
    lab = clf.fit(data_)

    scores = clf.decision_scores_

    # Add scores to original data frame
    hbos_orig = orig_
    hbos_orig['scores'] = scores

    # sort data frame by score
    hbos_orig_sorted = hbos_orig.sort_values(by=['scores'], ascending=False)
    #print(hbos_orig_sorted)
    # print(hbos_orig_sorted['Class'])
    fpr, tpr, thresholds = metrics.roc_curve(hbos_orig_sorted['Class'], hbos_orig_sorted['scores'])
    auc = metrics.auc(fpr, tpr)
    # auc = roc_auc_score(hbos_orig_sorted['Class'], hbos_orig_sorted['scores'])

    plt.figure(figsize=[8, 5])
    plt.plot(fpr, tpr, color='r', lw=2, label=clf_name)
    plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--', label='guessing')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic: AUC_LOF ={0:0.4f}'.format(auc))
    plt.legend(loc="lower right")
    plt.show()
    tupel = (n_bins_, auc)
    aucs.append(tupel)


if __name__ == "__main__":
    aucs = []
    np.set_printoptions(threshold=np.inf)

    dataset2 = pd.read_csv(r"C:\Users\david\Desktop\datasets\breast-cancer-unsupervised-ad.csv")
    dataset = pd.read_csv(r"C:\Users\david\Desktop\datasets\creditcard.csv")
    dataset3 = pd.read_csv(r"C:\Users\david\Desktop\datasets\pen-local-unsupervised-ad.csv")
    dataset4, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets\literature\Shuttle\Shuttle_withoutdupl_norm_v01.arff')
    dataset4 = pd.DataFrame(dataset4)
    dataset5, meta2 = arff.loadarff(
        r'C:\Users\david\Desktop\datasets\semantic\HeartDisease\HeartDisease_withoutdupl_02_v01.arff')
    dataset5 = pd.DataFrame(dataset5)

    dataset4['Class'] = dataset4['Class'].astype(int)
    dataset5['Class'] = dataset5['Class'].astype(int)
    orig = dataset.copy()
    orig2 = dataset2.copy()
    orig3 = dataset3.copy()
    orig4 = dataset4.copy()
    orig5 = dataset5.copy()
    print(len(dataset4))

    del dataset['Time']
    del dataset['Amount']
    del dataset['Class']

    del dataset2['Class']
    del dataset3['Class']

    del dataset4['id']
    del dataset4['Class']
    del dataset5['Class']

    for i in range (1):
        #calc_score(dataset4, orig4,n_bins_=i+1)
        calc_score(dataset4, orig4, n_bins_=71)
    sorted_array = sorted(aucs, key=lambda x: x[1],reverse=True)
    print(sorted_array)
