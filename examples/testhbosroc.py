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

def plot_explainability(data,id):
    clf=HBOSPYOD(save_explainability_scores=True)
    clf.fit(data)

    y_values = clf.get_explainability_scores(id)
    print(y_values)
    x_values = range(1, len(y_values)+1)

    # Labels erstellen
    labels = ['Dimension {}'.format(i+1) for i in range(clf.n_features_)]

    print(labels)
    print(y_values)

    colors = cm.RdYlGn_r((y_values - np.min(y_values)) / (np.max(y_values) - np.min(y_values)))  # Skalieren der Werte auf [0, 1] und Farbskala umkehren

    plt.bar(x_values, y_values, align='center', color=colors, label="test")


    plt.xlabel('feature n')
    plt.ylabel('score')

    plt.title('Outlier score for sample: {}'.format(id)+' with outlierscore = {0:0.4f}'.format(clf.decision_scores_[id]))
    plt.legend(loc="lower right")
    plt.show()






def calc_score(dataset_, orig_, n_bins_):
    data_ = np.array(dataset_)

    # Now apply LOF on normalized data ...
    clf_name = 'HBOS'
    clf = HBOSPYOD()
    clf.set_params(n_bins=n_bins_,mode="dynamic",ranked=False,smoothen=False)
    lab = clf.fit(data_)
    n_bins.append(n_bins_)
    scores = clf.decision_scores_

    # Add scores to original data frame
    hbos_orig = orig_
    hbos_orig['scores'] = scores

    # sort data frame by score
    hbos_orig_sorted = hbos_orig.sort_values(by=['scores'], ascending=False)
    #print(hbos_orig_sorted)
    fpr, tpr, thresholds = metrics.roc_curve(hbos_orig_sorted['Class'], hbos_orig_sorted['scores'])
    auc = metrics.auc(fpr, tpr)
    # auc = roc_auc_score(hbos_orig_sorted['Class'], hbos_orig_sorted['scores'])

    '''plt.figure(figsize=[8, 5])
    plt.plot(fpr, tpr, color='r', lw=2, label=clf_name)
    plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--', label='guessing')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic: HBOS_AUC ={0:0.4f}'.format(auc))
    plt.legend(loc="lower right")
    plt.show()
    tupel = (n_bins_, auc)
    aucs.append(tupel)'''
    aucs.append(auc)


if __name__ == "__main__":
    aucs = []
    n_bins=[]
    np.set_printoptions(threshold=np.inf)

    X_train, y_train =generate_data(n_train=1000,n_features=2,contamination=0.1,random_state=42,train_only=True)
    dataset6 = pd.DataFrame(X_train)
    dataset6['Class']=y_train


    dataset2 = pd.read_csv(r"C:\Users\david\Desktop\datasets\breast-cancer-unsupervised-ad.csv")
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
    orig1 = dataset1.copy()
    orig2 = dataset2.copy()
    orig3 = dataset3.copy()
    orig4 = dataset4.copy()
    orig5 = dataset5.copy()
    orig6= dataset6.copy()

    del dataset1['Time']
    del dataset1['Amount']
    del dataset1['Class']

    del dataset2['Class']
    del dataset3['Class']

    del dataset4['id']
    del dataset4['Class']
    del dataset5['Class']
    plot_explainability(dataset1,0)

    '''for i in range (100):
        #calc_score(dataset3, orig3,n_bins_=i+1)
        calc_score(dataset6, orig6, n_bins_=i+1)
    plt.figure(figsize=[8, 5])
    plt.plot(n_bins, aucs, color='b', lw=2, label='AUC vs. n_bins')
    plt.xlabel('Number of Bins (n_bins)')
    plt.ylabel('Area Under the Curve (AUC)')
    plt.title('AUC vs. Number of Bins, Dataset 2, static  HBOS')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()'''

