import time
import h5py
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


def calc_auc_graph(data_, orig_, count,dataname):
    print(dataname,": current dataset")
    aucs_static = []
    maxatstatic = 1
    maxaucstatic = 0

    aucs_dynamic = []
    maxatdynamic = 1
    maxaucdynamic = 0

    aucs_rbos = []

    for i in range(count):
        bins = i + 1
        clfstatic = HBOSPYOD(n_bins=bins,ranked=hbosranked)
        clfdynamic = HBOSPYOD(mode="dynamic", n_bins=bins,ranked=hbosranked)
        clfRBOS = HBOSPYOD()

        clfstatic.fit(data_)
        clfdynamic.fit(data_)
        clfRBOS.fit2(data_)

        scoresstatic = clfstatic.decision_scores_
        scoresdynamic = clfdynamic.decision_scores_
        scoreRBOS = clfRBOS.decision_scores_

        hbos_static = orig_.copy()
        hbos_dynamic = orig_.copy()
        rbos = orig_.copy()

        hbos_static['scores'] = scoresstatic
        hbos_static_sorted = hbos_static.sort_values(by=['scores'], ascending=False)
        fpr1, tpr1, thresholds1 = metrics.roc_curve(hbos_static_sorted['Class'], hbos_static_sorted['scores'])
        aucstatic = metrics.auc(fpr1, tpr1)
        if aucstatic > maxaucstatic:
            maxaucstatic = aucstatic
            maxatstatic = i + 1
        aucs_static.append(aucstatic)

        hbos_dynamic['scores'] = scoresdynamic
        hbos_dynamic_sorted = hbos_dynamic.sort_values(by=['scores'], ascending=False)
        fpr2, tpr2, thresholds2 = metrics.roc_curve(hbos_dynamic_sorted['Class'], hbos_dynamic_sorted['scores'])
        aucdynamic = metrics.auc(fpr2, tpr2)
        if aucdynamic > maxaucdynamic:
            maxaucdynamic = aucdynamic
            maxatdynamic = i + 1
        aucs_dynamic.append(aucdynamic)

        rbos['scores'] = scoreRBOS
        rbos_sorted = rbos.sort_values(by=['scores'], ascending=False)
        fpr3, tpr3, thresholds3 = metrics.roc_curve(rbos_sorted['Class'], rbos_sorted['scores'])
        aucrbos = metrics.auc(fpr3, tpr3)
        aucs_rbos.append(aucrbos)

    auto_auc_static , auto_bins_static= calc_roc_auc2(data_,orig_,"static", hbosranked)
    auto_auc_dynamic, auto_bins_dynamic= calc_roc_auc2(data_,orig_,"dynamic",hbosranked)

    xval = range(1, count + 1)
    plt.figure(figsize=[8, 6])
    #plt.plot(xval, aucs_static, color='b', lw=2, label='mode: ' + hbosmode + ', ranked: {}'.format(hbosranked))
    plt.plot(xval, aucs_static, color='b', lw=2, label='mode: static' + ', ranked: {}'.format(hbosranked))
    plt.plot(xval, aucs_dynamic, color='r', lw=2, label='mode: dynamic' + ', ranked: {}'.format(hbosranked))
    plt.plot(xval, aucs_rbos, color='c', lw=2, label='mode: RBOS')
    plt.scatter(auto_bins_static, auto_auc_static, color='b',s=100,marker='s', label='n_bins= sqrt(samples) static', zorder=10)
    plt.scatter(auto_bins_dynamic, auto_auc_dynamic, color='r',s=100,marker='x', label='n_bins= sqrt(samples) dynamic', zorder=10)
    plt.xlabel('Number of Bins')
    plt.ylabel('Area Under the Curve (AUC)')
    plt.title('AUC vs. n_bins \n' + ' max AUC "static": {0:0.4f}'.format(maxaucstatic) + ' at {}'.format(
        maxatstatic) + ' bins \n' + ' max AUC "dynamic": {0:0.4f}'.format(maxaucdynamic) + ' at {}'.format(
        maxatdynamic))
    plt.legend(loc="lower right")
    plt.grid(True)
    # plt.text(0, -0.1, 't_static: {0:0.2f}'.format(duration)+ ' s', fontsize=12, color='black', ha='left',transform=plt.gca().transAxes)
    pfad=r'C:\Users\david\Desktop\datasets_hbos\ODDS\Dataset\static_and_dynamic\aucs'
    filename = f"{pfad}_plot_{dataname}.png"
    plt.savefig(filename)
    plt.show()

    return maxatstatic


def plot_explainability(id):
    y_values = clf.get_explainability_scores(id)

    # Labels erstellen
    labels = ['Feature: {}'.format(i + 1) for i in range(clf.n_features_)]

    colors = cm.RdYlGn_r(y_values / (np.max(y_values)))  # Skalieren der Werte auf [0, 1]
    plt.figure(figsize=[10, 8])
    plt.barh(np.arange(len(y_values)), y_values, color=colors, tick_label=labels)

    plt.xlabel('score')

    plt.title(
        'explainability score for sample: {}'.format(id) + ' with outlierscore = {0:0.4f}'.format(
            clf.decision_scores_[id]))
    plt.legend(loc="lower right")
    plt.show()


def calc_roc_auc2(data_,orig_,mode_,ranked_):
    clfauc = HBOSPYOD(ranked=ranked_, mode=mode_)
    clfauc.fit(data_)
    scores = clfauc.decision_scores_
    hbos_orig = orig_
    hbos_orig['scores'] = scores
    hbos_orig_sorted = hbos_orig.sort_values(by=['scores'], ascending=False)

    fpr, tpr, thresholds = metrics.roc_curve(hbos_orig_sorted['Class'], hbos_orig_sorted['scores'])
    auc = metrics.auc(fpr, tpr)
    # auc = roc_auc_score(hbos_orig_sorted['Class'], hbos_orig_sorted['scores'])
    return auc, clfauc.n_bins

def calc_roc_auc(orig_, plot):
    clf_name = 'HBOS'
    scores = clf.decision_scores_
    hbos_orig = orig_
    hbos_orig['scores'] = scores
    hbos_orig_sorted = hbos_orig.sort_values(by=['scores'], ascending=False)

    fpr, tpr, thresholds = metrics.roc_curve(hbos_orig_sorted['Class'], hbos_orig_sorted['scores'])
    auc = metrics.auc(fpr, tpr)
    # auc = roc_auc_score(hbos_orig_sorted['Class'], hbos_orig_sorted['scores'])

    if (plot):
        plt.figure(figsize=[8, 5])
        plt.plot(fpr, tpr, color='r', lw=2, label='mode: ' + hbosmode + ', ranked: {}'.format(hbosranked))
        plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--', label='guessing')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        print(auc, "auc")
        plt.title('Receiver operating characteristic: HBOS_AUC ={0:0.4f}'.format(auc))
        plt.text(0, -0.1, 'n_bins: {}'.format(clf.n_bins), fontsize=12, color='black', ha='left',
                 transform=plt.gca().transAxes)
        plt.legend(loc="lower right")
        plt.show()
    return auc


if __name__ == "__main__":
    mat_data1 = loadmat(r"C:\Users\david\Desktop\datasets_hbos\ODDS\Dataset\annthyroid.mat")
    dataset1 = pd.DataFrame(mat_data1['X'])
    dataset1["Class"] = mat_data1['y']
    orig1 = dataset1.copy()
    del dataset1['Class']

    mat_data2 = loadmat(r"C:\Users\david\Desktop\datasets_hbos\ODDS\Dataset\cardio.mat")
    dataset2 = pd.DataFrame(mat_data2['X'])
    dataset2["Class"] = mat_data2['y']
    orig2 = dataset2.copy()
    del dataset2['Class']

    mat_data3 = loadmat(r"C:\Users\david\Desktop\datasets_hbos\ODDS\Dataset\cover.mat")
    dataset3 = pd.DataFrame(mat_data3['X'])
    dataset3["Class"] = mat_data3['y']
    orig3 = dataset3.copy()
    del dataset3['Class']

    mat_data4 = loadmat(r"C:\Users\david\Desktop\datasets_hbos\ODDS\Dataset\letter.mat")
    dataset4 = pd.DataFrame(mat_data4['X'])
    dataset4["Class"] = mat_data4['y']
    orig4 = dataset4.copy()
    del dataset4['Class']

    mat_data5 = loadmat(r"C:\Users\david\Desktop\datasets_hbos\ODDS\Dataset\glass.mat")
    dataset5 = pd.DataFrame(mat_data5['X'])
    dataset5["Class"] = mat_data5['y']
    orig5 = dataset5.copy()
    del dataset5['Class']

    with h5py.File(r"C:\Users\david\Desktop\datasets_hbos\ODDS\Dataset\http.mat", 'r') as file:
        dataset6 = pd.DataFrame(file['X'][:])
        dataset6 = dataset6.transpose()
        labels = file['y'][:]
        labels = labels.transpose()
    dataset6["Class"] = labels
    orig6 = dataset6.copy()
    del dataset6['Class']

    mat_data7 = loadmat(r"C:\Users\david\Desktop\datasets_hbos\ODDS\Dataset\lympho.mat")
    dataset7 = pd.DataFrame(mat_data7['X'])
    dataset7["Class"] = mat_data7['y']
    orig7 = dataset7.copy()
    del dataset7['Class']

    mat_data8 = loadmat(r"C:\Users\david\Desktop\datasets_hbos\ODDS\Dataset\mammography.mat")
    dataset8 = pd.DataFrame(mat_data8['X'])
    dataset8["Class"] = mat_data8['y']
    orig8 = dataset8.copy()
    del dataset8['Class']

    mat_data9 = loadmat(r"C:\Users\david\Desktop\datasets_hbos\ODDS\Dataset\mnist.mat")
    dataset9 = pd.DataFrame(mat_data9['X'])
    dataset9["Class"] = mat_data9['y']
    orig9 = dataset9.copy()
    del dataset9['Class']

    mat_data10 = loadmat(r"C:\Users\david\Desktop\datasets_hbos\ODDS\Dataset\musk.mat")
    dataset10 = pd.DataFrame(mat_data10['X'])
    dataset10["Class"] = mat_data10['y']
    orig10 = dataset10.copy()
    del dataset10['Class']

    mat_data11 = loadmat(r"C:\Users\david\Desktop\datasets_hbos\ODDS\Dataset\optdigits.mat")
    dataset11 = pd.DataFrame(mat_data11['X'])
    dataset11["Class"] = mat_data11['y']
    orig11 = dataset11.copy()
    del dataset11['Class']

    mat_data12 = loadmat(r"C:\Users\david\Desktop\datasets_hbos\ODDS\Dataset\pendigits.mat")
    dataset12 = pd.DataFrame(mat_data12['X'])
    dataset12["Class"] = mat_data12['y']
    orig12 = dataset12.copy()
    del dataset12['Class']

    dataset13 = pd.read_csv(r"C:\Users\david\Desktop\datasets\creditcard.csv")
    orig13 = dataset13.copy()
    del dataset13['Time']
    del dataset13['Amount']
    del dataset13['Class']
    annthyroid = arff.loadarff(r'C:\Users\david\Desktop\datasets\semantic\Annthyroid\Annthyroid_02_v01.arff')
    annthyroid_df = pd.DataFrame(annthyroid[0])
    origannthyroid = annthyroid_df.copy()
    del annthyroid_df['Class']
    del annthyroid_df['id']
    origannthyroid['Class'] = origannthyroid['Class'].astype(int)

    dataset14 = pd.read_csv(r'C:\Users\david\Desktop\datasets\breast-cancer-unsupervised-ad.csv')
    orig14 = dataset14.copy()
    del dataset14['Class']

    annthyroid_norm = arff.loadarff(r'C:\Users\david\Desktop\datasets\semantic\Annthyroid\Annthyroid_norm_02_v01.arff')
    annthyroid_norm_df = pd.DataFrame(annthyroid_norm[0])
    origannthyroid_norm = annthyroid_norm_df.copy()
    del annthyroid_norm_df['Class']
    del annthyroid_norm_df['id']
    origannthyroid_norm['Class'] = origannthyroid_norm['Class'].astype(int)

    cnt = 0

    fit2 = False
    if (fit2):
        hbosmode = "RBOS"
        hbosranked = "RBOS"
    else:
        hbosmode = ("static")
        hbosranked = False

    # data_ = annthyroid_norm_df
    # orig_ = origannthyroid_norm
    data_ = dataset1
    orig_ = orig1

    #max = calc_auc_graph(data_, orig_, 500,"auto")

    start_time = time.time()
    calc_auc_graph(dataset1, orig1, 500,"annthyroid")  #Das kann man doch multi threaden!!!!!!
    calc_auc_graph(dataset2, orig2, 500,"cardio")
    calc_auc_graph(dataset3, orig3, 500,"cover")
    calc_auc_graph(dataset4, orig4, 500,"letter")
    calc_auc_graph(dataset5, orig5, 500,"glass")
    calc_auc_graph(dataset6, orig6, 500,"http")
    calc_auc_graph(dataset7, orig7, 500,"lympho")
    calc_auc_graph(dataset8, orig8, 500,"mammography")
    calc_auc_graph(dataset9, orig9, 500,"mnist")
    calc_auc_graph(dataset10, orig10, 500,"musk")
    calc_auc_graph(dataset11, orig11, 500,"optdigits")
    calc_auc_graph(dataset12, orig12, 500,"pendigits")
    calc_auc_graph(dataset13, orig13, 500,"creditcard")
    calc_auc_graph(dataset14, orig14, 500,"Annthyroid_02_v01")
    end_time = time.time()
    print("Time taken: ", end_time - start_time)

    print(max)
    clf = HBOSPYOD(save_explainability_scores=True, ranked=hbosranked, mode=hbosmode, n_bins=max)
    if fit2:
        clf.fit2(data_)
        clf.n_bins = "RBOS"
    else:
        clf.fit(data_)

    # calc_roc_auc(orig_,True)

    # plot_explainability(0)


