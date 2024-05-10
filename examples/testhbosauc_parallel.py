import multiprocessing
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


def calc_auc_graph_static_or_dynamic(data_, orig_, count, dataname):
    print(dataname, ": current dataset")
    aucs_static = []
    maxatstatic = 0
    maxaucstatic = 0

    aucs_ranked = []
    maxatranked = 0
    maxaucranked = 0

    aucs_smooth = []
    maxatsmooth = 0
    maxaucsmooth = 0
    xval = []
    xvalsmooth = []
    hbosranked = False
    mode_ = "dynamic"
    start_time = time.time()
    # for i in range(count):
    for i in range(count):
        bins = i + 2
        xval.append(bins)

        clfstatic = HBOSPYOD(mode=mode_, n_bins=bins, ranked=False)
        clfstatic.fit(data_)
        scoresstatic = clfstatic.decision_scores_
        hbos_static = orig_.copy()
        hbos_static['scores'] = scoresstatic
        hbos_static_sorted = hbos_static.sort_values(by=['scores'], ascending=False)
        fpr1, tpr1, thresholds1 = metrics.roc_curve(hbos_static_sorted['Class'], hbos_static_sorted['scores'])
        aucstatic = metrics.auc(fpr1, tpr1)
        if aucstatic > maxaucstatic:
            maxaucstatic = aucstatic
            maxatstatic = bins
        aucs_static.append(aucstatic)

        clfranked = HBOSPYOD(mode=mode_, n_bins=bins, ranked=True)
        clfranked.fit(data_)
        scoresranked = clfranked.decision_scores_
        hbos_ranked = orig_.copy()
        hbos_ranked['scores'] = scoresranked
        hbos_ranked_sorted = hbos_ranked.sort_values(by=['scores'], ascending=False)
        fpr2, tpr2, thresholds2 = metrics.roc_curve(hbos_ranked_sorted['Class'], hbos_ranked_sorted['scores'])
        aucranked = metrics.auc(fpr2, tpr2)
        if aucranked > maxaucranked:
            maxaucranked = aucranked
            maxatranked = bins
        aucs_ranked.append(aucranked)

        if mode_ == "static":
            if bins > 2:
                xvalsmooth.append(bins)
                clfsmooth = HBOSPYOD(mode=mode_, n_bins=bins, ranked=False, smoothen=True)
                clfsmooth.fit(data_)
                scoressmooth = clfsmooth.decision_scores_
                hbos_smooth = orig_.copy()
                hbos_smooth['scores'] = scoressmooth
                hbos_smooth_sorted = hbos_smooth.sort_values(by=['scores'], ascending=False)
                fpr2, tpr2, thresholds2 = metrics.roc_curve(hbos_smooth_sorted['Class'], hbos_smooth_sorted['scores'])
                aucsmooth = metrics.auc(fpr2, tpr2)
                if aucsmooth > maxaucsmooth:
                    maxaucsmooth = aucsmooth
                    maxatsmooth = bins
                aucs_smooth.append(aucsmooth)
    end_time = time.time()
    print("Time taken to run: ", dataname, end_time - start_time, "seconds.")

    auto_auc_static, auto_bins_static = calc_roc_auc2(data_, orig_, mode_, hbosranked, False, "auto")
    auto_auc_ranked, auto_bins_ranked = calc_roc_auc2(data_, orig_, mode_, True, False, "auto")
    if mode_ == "static":
        auto_auc_smooth, auto_bins_smooth = calc_roc_auc2(data_, orig_, mode_, hbosranked, True, "auto")
    # xval = range(2, count + 1)
    plt.figure(figsize=[10, 8])
    # plt.plot(xval, aucs_static, color='b', lw=2, label='mode: ' + hbosmode + ', ranked: {}'.format(hbosranked))
    plt.plot(xval, aucs_static, color='b', lw=1, label='mode: ' + mode_)
    plt.plot(xval, aucs_ranked, color='r', lw=1, label='mode: ' + mode_ + ' ranked')
    if mode_ == "static":
        plt.plot(xvalsmooth, aucs_smooth, color='g', lw=1, label='mode: ' + mode_ + ' smooth')

    label_ = 'n_bins= sqrt(samples) '
    label2_ = 'n_bins= sqrt(samples) ranked ' + mode_
    label3_ = 'n_bins= sqrt(samples) smooth ' + mode_

    plt.scatter(auto_bins_static, auto_auc_static, color='k', s=100, marker='X', label=label_, zorder=10)
    plt.scatter(auto_bins_ranked, auto_auc_ranked, color='k', s=100, marker='X', zorder=10)
    if mode_ == "static":
        plt.scatter(auto_bins_smooth, auto_auc_smooth, color='k', s=100, marker='X', zorder=10)
    plt.xlabel('Number of Bins')
    plt.ylabel('Area Under the Curve (AUC)')
    if mode_ == "static":
        plt.title(
            'AUC vs. n_bins: ' + dataname + '\n' + ' max AUC "static": {0:0.4f}'.format(maxaucstatic) + ' at {}'.format(
                maxatstatic) + ' bins \n' + ' max AUC "ranked": {0:0.4f}'.format(maxaucranked) + ' at {}'.format(
                maxatranked) + ' bins \n' + ' max AUC "smoothed": {0:0.4f}'.format(maxaucsmooth) + 'at {}'.format(
                maxatsmooth) + ' bins')
    if mode_ == "dynamic":
        plt.title(
            'AUC vs. n_bins ' + dataname + '\n' + ' max AUC "static": {0:0.4f}'.format(maxaucstatic) + ' at {}'.format(
                maxatstatic) + ' bins \n' + ' max AUC "ranked": {0:0.4f}'.format(maxaucranked) + ' at {}'.format(
                maxatranked) + ' bins \n')
    plt.legend(loc="lower right")
    plt.ylim(0, 1.1)
    plt.grid(True)
    # plt.text(0, -0.1, 't_static: {0:0.2f}'.format(duration)+ ' s', fontsize=12, color='black', ha='left',transform=plt.gca().transAxes)
    # pfad=r'C:\Users\david\Desktop\datasets_hbos\ODDS\Dataset\benchmarks'
    # pfad = r'C:\Users\david\Desktop\datasets_hbos\Harvard Dataverse\benchmarks'
    # pfad = r'C:\Users\david\Desktop\datasets_hbos\ELKI\benchmarks'
    pfad = r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\benchmarks'
    filename = f"{pfad}\{dataname}_{mode_}.png"
    plt.savefig(filename)
    plt.show()

    return maxatstatic


def calc_auc_graph(data_, orig_, count, dataname):
    print(dataname, ": current dataset")
    aucs_static = []
    maxatstatic = 0
    maxaucstatic = 0

    aucs_dynamic = []
    maxatdynamic = 0
    maxaucdynamic = 0
    xval = []
    hbosranked = False
    start_time = time.time()
    for i in range(count):
        bins = i + 2
        xval.append(bins)
        clfstatic = HBOSPYOD(n_bins=bins, ranked=hbosranked)
        clfstatic.fit(data_)
        scoresstatic = clfstatic.decision_scores_
        hbos_static = orig_.copy()
        hbos_static['scores'] = scoresstatic
        hbos_static_sorted = hbos_static.sort_values(by=['scores'], ascending=False)
        fpr1, tpr1, thresholds1 = metrics.roc_curve(hbos_static_sorted['Class'], hbos_static_sorted['scores'])
        aucstatic = metrics.auc(fpr1, tpr1)
        if aucstatic > maxaucstatic:
            maxaucstatic = aucstatic
            maxatstatic = bins
        aucs_static.append(aucstatic)

        clfdynamic = HBOSPYOD(mode="dynamic", n_bins=bins, ranked=hbosranked)
        clfdynamic.fit(data_)
        scoresdynamic = clfdynamic.decision_scores_
        hbos_dynamic = orig_.copy()
        hbos_dynamic['scores'] = scoresdynamic
        hbos_dynamic_sorted = hbos_dynamic.sort_values(by=['scores'], ascending=False)
        fpr2, tpr2, thresholds2 = metrics.roc_curve(hbos_dynamic_sorted['Class'], hbos_dynamic_sorted['scores'])
        aucdynamic = metrics.auc(fpr2, tpr2)
        if aucdynamic > maxaucdynamic:
            maxaucdynamic = aucdynamic
            maxatdynamic = bins
        aucs_dynamic.append(aucdynamic)
    end_time = time.time()
    print("Time taken to run: ", dataname, end_time - start_time, "seconds.")

    auto_auc_static, auto_bins_static = calc_roc_auc2(data_, orig_, "static", hbosranked, False, "auto")
    auto_auc_dynamic, auto_bins_dynamic = calc_roc_auc2(data_, orig_, "dynamic", hbosranked, False, "auto")
    calc_auc_static, calc_bins_static = calc_roc_auc2(data_, orig_, "static", hbosranked, False, "calc")
    calc_auc_dynamic, calc_bins_dynamic = calc_roc_auc2(data_, orig_, "dynamic", hbosranked, False, "calc")
    unique_auc_static, unique_bins_static = calc_roc_auc2(data_, orig_, "static", hbosranked, False, "unique")
    unique_auc_dynamic, unique_bins_dynamic = calc_roc_auc2(data_, orig_, "dynamic", hbosranked, False, "unique")
    calcvaldynamic = [calc_auc_dynamic] * len(xval)
    calcvalstatic = [calc_auc_static] * len(xval)
    uniquevaldynamic = [unique_auc_dynamic] * len(xval)
    uniquevalstatic = [unique_auc_static] * len(xval)

    # xval = range(1, count + 1)
    plt.figure(figsize=[8, 6])
    # plt.plot(xval, aucs_static, color='b', lw=2, label='mode: ' + hbosmode + ', ranked: {}'.format(hbosranked))

    # plt.plot(xval, uniquevalstatic, color='g', lw=1, label="n_bins= unique static)")
    # plt.plot(xval, uniquevaldynamic, color='c', lw=1, label="n_bins= unique dynamic)")
    plt.plot(xval, aucs_static, color='b', lw=1, label='mode: static')
    plt.plot(xval, calcvalstatic, color='c', lw=1, label="n_bins= calc static: "+ '{0:0.4f}'.format(calc_auc_static))

    plt.plot(xval, aucs_dynamic, color='r', lw=1, label='mode: dynamic')
    plt.plot(xval, calcvaldynamic, color="#EDB120", lw=1, label="n_bins= calc dynamic: "+ '{0:0.4f}'.format(calc_auc_dynamic))


    plt.scatter(auto_bins_static, auto_auc_static, color='k', s=100, marker='X', label='n_bins= sqrt(samples)',
                zorder=10)
    plt.scatter(auto_bins_dynamic, auto_auc_dynamic, color='k', s=100, marker='X', zorder=10)

    # plt.scatter(calc_bins_static, calc_auc_static, color='m', s=100, marker='x', label='n_bins= Birge Rozenblac method',
    #            zorder=10)
    # plt.scatter(calc_bins_dynamic, calc_auc_dynamic, color='m', s=100, marker='x', zorder=10)

    # plt.scatter(unique_bins_static, unique_auc_static, color='y', s=100, marker='+', label='n_bins= sqrt(np.unique(samples))',
    #            zorder=10)
    # plt.scatter(unique_bins_dynamic, unique_auc_dynamic, color='y', s=100, marker='+', zorder=10)

    plt.xlabel('Number of Bins')
    plt.ylabel('Area Under the Curve (AUC)')
    plt.title(
        'AUC vs. n_bins: ' + dataname + ' \n' + ' max AUC "static": {0:0.4f}'.format(maxaucstatic) + ' at {}'.format(
            maxatstatic) + ' bins \n' + ' max AUC "dynamic": {0:0.4f}'.format(maxaucdynamic) + ' at {}'.format(
            maxatdynamic) + ' bins')
    plt.legend(loc="lower right")
    plt.ylim(0, 1.1)
    plt.grid(True)
    # plt.text(0, -0.1, 't_static: {0:0.2f}'.format(duration)+ ' s', fontsize=12, color='black', ha='left',transform=plt.gca().transAxes)
    # pfad=r'C:\Users\david\Desktop\datasets_hbos\ODDS\Dataset\benchmarks\static_dynamic'
    # pfad = r'C:\Users\david\Desktop\datasets_hbos\Harvard Dataverse\benchmarks\static_dynamic'
    # pfad= r'C:\Users\david\Desktop\datasets_hbos\ELKI\benchmarks'
    pfad = r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\benchmarks'
    filename = f"{pfad}\static_dynamic_{dataname}.png"
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


def calc_roc_auc2(data_, orig_, mode_, ranked_, smooth_, n_bins_):
    clfauc = HBOSPYOD(ranked=ranked_, mode=mode_, smoothen=smooth_, n_bins=n_bins_)
    clfauc.fit(data_)
    scores = clfauc.decision_scores_
    hbos_orig = orig_
    hbos_orig['scores'] = scores
    hbos_orig_sorted = hbos_orig.sort_values(by=['scores'], ascending=False)

    fpr, tpr, thresholds = metrics.roc_curve(hbos_orig_sorted['Class'], hbos_orig_sorted['scores'])
    auc = metrics.auc(fpr, tpr)
    # auc = roc_auc_score(hbos_orig_sorted['Class'], hbos_orig_sorted['scores'])
    return auc, clfauc.n_bins


def calc_roc_auc(orig_):
    clf_name = 'HBOS'
    scores = clf.decision_scores_
    hbos_orig = orig_
    hbos_orig['scores'] = scores
    hbos_orig_sorted = hbos_orig.sort_values(by=['scores'], ascending=False)

    fpr, tpr, thresholds = metrics.roc_curve(hbos_orig_sorted['Class'], hbos_orig_sorted['scores'])
    auc = metrics.auc(fpr, tpr)
    # auc = roc_auc_score(hbos_orig_sorted['Class'], hbos_orig_sorted['scores'])

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

    X_train, y_train = generate_data(n_train=200,n_test=100, n_features=2, contamination=0.1, random_state=42, train_only=True)
    datasettest = pd.DataFrame(X_train)
    datasettest['Class'] = y_train
    datasettestorig= datasettest.copy()
    del datasettest['Class']

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

    harvard1 = pd.read_csv(r'C:\Users\david\Desktop\datasets_hbos\Harvard Dataverse\aloi-unsupervised-ad.csv',
                           header=None)
    lastcol = harvard1.columns[-1]
    harvard1.rename(columns={lastcol: 'Class'}, inplace=True)
    harvardorig1 = harvard1.copy()
    del harvard1['Class']

    harvard2 = pd.read_csv(r'C:\Users\david\Desktop\datasets_hbos\Harvard Dataverse\annthyroid-unsupervised-ad.csv',
                           header=None)
    lastcol = harvard2.columns[-1]
    harvard2.rename(columns={lastcol: 'Class'}, inplace=True)
    harvardorig2 = harvard2.copy()
    del harvard2['Class']

    harvard3 = pd.read_csv(r'C:\Users\david\Desktop\datasets_hbos\Harvard Dataverse\breast-cancer-unsupervised-ad.csv',
                           header=None)
    lastcol = harvard3.columns[-1]
    harvard3.rename(columns={lastcol: 'Class'}, inplace=True)
    harvardorig3 = harvard3.copy()
    del harvard3['Class']

    harvard4 = pd.read_csv(r'C:\Users\david\Desktop\datasets_hbos\Harvard Dataverse\kdd99-unsupervised-ad.csv',
                           header=None)
    lastcol = harvard4.columns[-1]
    harvard4.rename(columns={lastcol: 'Class'}, inplace=True)
    harvardorig4 = harvard4.copy()
    del harvard4['Class']

    harvard5 = pd.read_csv(r'C:\Users\david\Desktop\datasets_hbos\Harvard Dataverse\letter-unsupervised-ad.csv',
                           header=None)
    lastcol = harvard5.columns[-1]
    harvard5.rename(columns={lastcol: 'Class'}, inplace=True)
    harvardorig5 = harvard5.copy()
    del harvard5['Class']

    harvard6 = pd.read_csv(r'C:\Users\david\Desktop\datasets_hbos\Harvard Dataverse\pen-global-unsupervised-ad.csv',
                           header=None)
    lastcol = harvard6.columns[-1]
    harvard6.rename(columns={lastcol: 'Class'}, inplace=True)
    harvardorig6 = harvard6.copy()
    del harvard6['Class']

    harvard7 = pd.read_csv(r'C:\Users\david\Desktop\datasets_hbos\Harvard Dataverse\pen-local-unsupervised-ad.csv',
                           header=None)
    lastcol = harvard7.columns[-1]
    harvard7.rename(columns={lastcol: 'Class'}, inplace=True)
    harvardorig7 = harvard7.copy()
    del harvard7['Class']

    harvard8 = pd.read_csv(r'C:\Users\david\Desktop\datasets_hbos\Harvard Dataverse\satellite-unsupervised-ad.csv',
                           header=None)
    lastcol = harvard8.columns[-1]
    harvard8.rename(columns={lastcol: 'Class'}, inplace=True)
    harvardorig8 = harvard8.copy()
    del harvard8['Class']

    harvard9 = pd.read_csv(r'C:\Users\david\Desktop\datasets_hbos\Harvard Dataverse\shuttle-unsupervised-ad.csv',
                           header=None)
    lastcol = harvard9.columns[-1]
    harvard9.rename(columns={lastcol: 'Class'}, inplace=True)
    harvardorig9 = harvard9.copy()
    del harvard9['Class']

    harvard10 = pd.read_csv(r'C:\Users\david\Desktop\datasets_hbos\Harvard Dataverse\speech-unsupervised-ad.csv',
                            header=None)
    lastcol = harvard10.columns[-1]
    harvard10.rename(columns={lastcol: 'Class'}, inplace=True)
    harvardorig10 = harvard10.copy()
    del harvard10['Class']

    elki1, meta = arff.loadarff(r'C:\Users\david\Desktop\datasets_hbos\ELKI\literature\ALOI\ALOI.arff')
    elki1 = pd.DataFrame(elki1)
    elkiorig1 = elki1.copy()
    del elki1['id']
    del elki1['Class']
    elkiorig1['Class'] = elkiorig1['Class'].astype(int)
    print(elkiorig1.shape)

    elki2, meta = arff.loadarff(r'C:\Users\david\Desktop\datasets_hbos\ELKI\literature\ALOI\ALOI_withoutdupl.arff')
    elki2 = pd.DataFrame(elki2)
    elkiorig2 = elki2.copy()
    elkiorig2['Class'] = elkiorig2['Class'].astype(int)
    del elki2['id']
    del elki2['Class']
    print(elkiorig2.shape)

    elki3, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\literature\Glass\Glass_withoutdupl_norm.arff')
    elki3 = pd.DataFrame(elki3)
    elkiorig3 = elki3.copy()
    elkiorig3['Class'] = elkiorig3['Class'].astype(int)
    del elki3['id']
    del elki3['Class']
    print(elkiorig3.shape)

    elki4, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\literature\Ionosphere\Ionosphere_withoutdupl_norm.arff')
    elki4 = pd.DataFrame(elki4)
    elkiorig4 = elki4.copy()
    elkiorig4['Class'] = elkiorig4['Class'].astype(int)
    del elki4['id']
    del elki4['Class']
    print(elkiorig4.shape)

    elki5, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\literature\KDDCup99\KDDCup99_catremoved.arff')
    elki5 = pd.DataFrame(elki5)
    elkiorig5 = elki5.copy()
    elkiorig5['Class'] = elkiorig5['Class'].astype(int)
    del elki5['id']
    del elki5['Class']
    print(elkiorig5.shape)

    elki6, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\literature\Lymphography\Lymphography_withoutdupl_catremoved.arff')
    elki6 = pd.DataFrame(elki6)
    elkiorig6 = elki6.copy()
    elkiorig6['Class'] = elkiorig6['Class'].astype(int)
    del elki6['id']
    del elki6['Class']
    print(elkiorig6.shape)

    elki7, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\literature\Lymphography\Lymphography_withoutdupl_norm_1ofn.arff')
    elki7 = pd.DataFrame(elki7)
    elkiorig7 = elki7.copy()
    elkiorig7['Class'] = elkiorig7['Class'].astype(int)
    del elki7['id']
    del elki7['Class']
    print(elkiorig7.shape)

    elki8, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\literature\PenDigits\PenDigits_withoutdupl_norm_v01.arff')
    elki8 = pd.DataFrame(elki8)
    elkiorig8 = elki8.copy()
    elkiorig8['Class'] = elkiorig8['Class'].astype(int)
    del elki8['id']
    del elki8['Class']
    print(elkiorig8.shape)

    elki9, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\literature\Shuttle\Shuttle_withoutdupl_v01.arff')
    elki9 = pd.DataFrame(elki9)
    elkiorig9 = elki9.copy()
    elkiorig9['Class'] = elkiorig9['Class'].astype(int)
    del elki9['id']
    del elki9['Class']
    print(elkiorig9.shape)

    elki10, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\literature\Waveform\Waveform_withoutdupl_v01.arff')
    elki10 = pd.DataFrame(elki10)
    elkiorig10 = elki10.copy()
    elkiorig10['Class'] = elkiorig10['Class'].astype(int)
    del elki10['id']
    del elki10['Class']
    print(elkiorig10.shape)

    elki11, meta = arff.loadarff(r'C:\Users\david\Desktop\datasets_hbos\ELKI\literature\WBC\WBC_v01.arff')
    elki11 = pd.DataFrame(elki11)
    elkiorig11 = elki11.copy()
    elkiorig11['Class'] = elkiorig11['Class'].astype(int)
    del elki11['id']
    del elki11['Class']
    print(elkiorig11.shape)

    elki12, meta = arff.loadarff(r'C:\Users\david\Desktop\datasets_hbos\ELKI\literature\WDBC\WDBC_withoutdupl_v01.arff')
    elki12 = pd.DataFrame(elki12)
    elkiorig12 = elki12.copy()
    elkiorig12['Class'] = elkiorig12['Class'].astype(int)
    del elki12['id']
    del elki12['Class']
    print(elkiorig12.shape)

    elki13, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\literature\WPBC\WPBC_withoutdupl_norm.arff')
    elki13 = pd.DataFrame(elki13)
    elkiorig13 = elki13.copy()
    elkiorig13['Class'] = elkiorig13['Class'].astype(int)
    del elki13['id']
    del elki13['Class']
    print(elkiorig13.shape)

    elki_semantic1, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Annthyroid\Annthyroid_02_v01.arff')
    elki_semantic1 = pd.DataFrame(elki_semantic1)
    elki_semanticorig1 = elki_semantic1.copy()
    del elki_semantic1['id']
    del elki_semantic1['Class']
    elki_semanticorig1['Class'] = elki_semanticorig1['Class'].astype(int)
    print(elki_semanticorig1.shape)

    elki_semantic2, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Arrhythmia\Arrhythmia_withoutdupl_02_v01.arff')
    elki_semantic2 = pd.DataFrame(elki_semantic2)
    elki_semanticorig2 = elki_semantic2.copy()
    elki_semanticorig2['Class'] = elki_semanticorig2['Class'].astype(int)
    del elki_semantic2['id']
    del elki_semantic2['Class']
    print(elki_semanticorig2.shape)

    elki_semantic3, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Cardiotocography\Cardiotocography_02_v01.arff')
    elki_semantic3 = pd.DataFrame(elki_semantic3)
    elki_semanticorig3 = elki_semantic3.copy()
    elki_semanticorig3['Class'] = elki_semanticorig3['Class'].astype(int)
    del elki_semantic3['id']
    del elki_semantic3['Class']
    print(elki_semanticorig3.shape)

    elki_semantic4, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\HeartDisease\HeartDisease_withoutdupl_02_v01.arff')
    elki_semantic4 = pd.DataFrame(elki_semantic4)
    elki_semanticorig4 = elki_semantic4.copy()
    elki_semanticorig4['Class'] = elki_semanticorig4['Class'].astype(int)
    del elki_semantic4['id']
    del elki_semantic4['Class']
    print(elki_semanticorig4.shape)

    elki_semantic5, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Hepatitis\Hepatitis_withoutdupl_05_v01.arff')
    elki_semantic5 = pd.DataFrame(elki_semantic5)
    elki_semanticorig5 = elki_semantic5.copy()
    elki_semanticorig5['Class'] = elki_semanticorig5['Class'].astype(int)
    del elki_semantic5['id']
    del elki_semantic5['Class']
    print(elki_semanticorig5.shape)

    elki_semantic6, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\InternetAds\InternetAds_norm_02_v01.arff')
    elki_semantic6 = pd.DataFrame(elki_semantic6)
    elki_semanticorig6 = elki_semantic6.copy()
    elki_semanticorig6['Class'] = elki_semanticorig6['Class'].astype(int)
    del elki_semantic6['id']
    del elki_semantic6['Class']
    print(elki_semanticorig6.shape)

    elki_semantic7, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\PageBlocks\PageBlocks_02_v01.arff')
    elki_semantic7 = pd.DataFrame(elki_semantic7)
    elki_semanticorig7 = elki_semantic7.copy()
    elki_semanticorig7['Class'] = elki_semanticorig7['Class'].astype(int)
    del elki_semantic7['id']
    del elki_semantic7['Class']
    print(elki_semanticorig7.shape)

    elki_semantic8, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Parkinson\Parkinson_withoutdupl_05_v01.arff')
    elki_semantic8 = pd.DataFrame(elki_semantic8)
    elki_semanticorig8 = elki_semantic8.copy()
    elki_semanticorig8['Class'] = elki_semanticorig8['Class'].astype(int)
    del elki_semantic8['id']
    del elki_semantic8['Class']
    print(elki_semanticorig8.shape)

    elki_semantic9, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Pima\Pima_withoutdupl_02_v01.arff')
    elki_semantic9 = pd.DataFrame(elki_semantic9)
    elki_semanticorig9 = elki_semantic9.copy()
    elki_semanticorig9['Class'] = elki_semanticorig9['Class'].astype(int)
    del elki_semantic9['id']
    del elki_semantic9['Class']
    print(elki_semanticorig9.shape)

    elki_semantic10, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\SpamBase\SpamBase_02_v01.arff')
    elki_semantic10 = pd.DataFrame(elki_semantic10)
    elki_semanticorig10 = elki_semantic10.copy()
    elki_semanticorig10['Class'] = elki_semanticorig10['Class'].astype(int)
    del elki_semantic10['id']
    del elki_semantic10['Class']
    print(elki_semanticorig10.shape)

    elki_semantic11, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Stamps\Stamps_withoutdupl_02_v01.arff')
    elki_semantic11 = pd.DataFrame(elki_semantic11)
    elki_semanticorig11 = elki_semantic11.copy()
    elki_semanticorig11['Class'] = elki_semanticorig11['Class'].astype(int)
    del elki_semantic11['id']
    del elki_semantic11['Class']
    print(elki_semanticorig11.shape)

    elki_semantic12, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Wilt\Wilt_02_v01.arff')
    elki_semantic12 = pd.DataFrame(elki_semantic12)
    elki_semanticorig12 = elki_semantic12.copy()
    elki_semanticorig12['Class'] = elki_semanticorig12['Class'].astype(int)
    del elki_semantic12['id']
    del elki_semantic12['Class']
    print(elki_semanticorig12.shape)

    annthyroid1, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Annthyroid\Annthyroid_02_v01.arff')
    annthyroid1 = pd.DataFrame(annthyroid1)
    annthyroidorig1 = annthyroid1.copy()
    del annthyroid1['id']
    del annthyroid1['Class']
    annthyroidorig1['Class'] = annthyroidorig1['Class'].astype(int)
    print(annthyroidorig1.shape)

    annthyroid2, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Annthyroid\Annthyroid_05_v01.arff')
    annthyroid2 = pd.DataFrame(annthyroid2)
    annthyroidorig2 = annthyroid2.copy()
    annthyroidorig2['Class'] = annthyroidorig2['Class'].astype(int)
    del annthyroid2['id']
    del annthyroid2['Class']
    print(annthyroidorig2.shape)

    annthyroid3, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Annthyroid\Annthyroid_07.arff')
    annthyroid3 = pd.DataFrame(annthyroid3)
    annthyroidorig3 = annthyroid3.copy()
    annthyroidorig3['Class'] = annthyroidorig3['Class'].astype(int)
    del annthyroid3['id']
    del annthyroid3['Class']
    print(annthyroidorig3.shape)

    annthyroid4, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Annthyroid\Annthyroid_norm_02_v01.arff')
    annthyroid4 = pd.DataFrame(annthyroid4)
    annthyroidorig4 = annthyroid4.copy()
    annthyroidorig4['Class'] = annthyroidorig4['Class'].astype(int)
    del annthyroid4['id']
    del annthyroid4['Class']
    print(annthyroidorig4.shape)

    annthyroid5, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Annthyroid\Annthyroid_norm_05_v01.arff')
    annthyroid5 = pd.DataFrame(annthyroid5)
    annthyroidorig5 = annthyroid5.copy()
    annthyroidorig5['Class'] = annthyroidorig5['Class'].astype(int)
    del annthyroid5['id']
    del annthyroid5['Class']
    print(annthyroidorig5.shape)

    annthyroid6, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Annthyroid\Annthyroid_norm_07.arff')
    annthyroid6 = pd.DataFrame(annthyroid6)
    annthyroidorig6 = annthyroid6.copy()
    annthyroidorig6['Class'] = annthyroidorig6['Class'].astype(int)
    del annthyroid6['id']
    del annthyroid6['Class']
    print(annthyroidorig6.shape)

    annthyroid7, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Annthyroid\Annthyroid_withoutdupl_02_v01.arff')
    annthyroid7 = pd.DataFrame(annthyroid7)
    annthyroidorig7 = annthyroid7.copy()
    annthyroidorig7['Class'] = annthyroidorig7['Class'].astype(int)
    del annthyroid7['id']
    del annthyroid7['Class']
    print(annthyroidorig7.shape)

    annthyroid8, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Annthyroid\Annthyroid_withoutdupl_05_v01.arff')
    annthyroid8 = pd.DataFrame(annthyroid8)
    annthyroidorig8 = annthyroid8.copy()
    annthyroidorig8['Class'] = annthyroidorig8['Class'].astype(int)
    del annthyroid8['id']
    del annthyroid8['Class']
    print(annthyroidorig8.shape)

    annthyroid9, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Annthyroid\Annthyroid_withoutdupl_07.arff')
    annthyroid9 = pd.DataFrame(annthyroid9)
    annthyroidorig9 = annthyroid9.copy()
    annthyroidorig9['Class'] = annthyroidorig9['Class'].astype(int)
    del annthyroid9['id']
    del annthyroid9['Class']
    print(annthyroidorig9.shape)

    annthyroid10, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Annthyroid\Annthyroid_withoutdupl_norm_02_v01.arff')
    annthyroid10 = pd.DataFrame(annthyroid10)
    annthyroidorig10 = annthyroid10.copy()
    annthyroidorig10['Class'] = annthyroidorig10['Class'].astype(int)
    del annthyroid10['id']
    del annthyroid10['Class']
    print(annthyroidorig10.shape)

    annthyroid11, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Annthyroid\Annthyroid_withoutdupl_norm_05_v01.arff')
    annthyroid11 = pd.DataFrame(annthyroid11)
    annthyroidorig11 = annthyroid11.copy()
    annthyroidorig11['Class'] = annthyroidorig11['Class'].astype(int)
    del annthyroid11['id']
    del annthyroid11['Class']
    print(annthyroidorig11.shape)

    annthyroid12, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Annthyroid\Annthyroid_withoutdupl_norm_07.arff')
    annthyroid12 = pd.DataFrame(annthyroid12)
    annthyroidorig12 = annthyroid12.copy()
    annthyroidorig12['Class'] = annthyroidorig12['Class'].astype(int)
    del annthyroid12['id']
    del annthyroid12['Class']
    print(annthyroidorig12.shape)

    cnt = 0



    # data_ = annthyroid_norm_df
    # orig_ = origannthyroid_norm
    data_ = dataset1
    orig_ = orig1
    VTEST = [

        (dataset1, orig1, 500, "test"),
        (dataset2, orig2, 500, "test"),

    ]

    V1 = [

        (dataset3, orig3, 100, "cover"),
        (dataset4, orig4, 100, "letter"),
        (dataset5, orig5, 100, "glass"),
        (dataset6, orig6, 100, "http"),
        (dataset7, orig7, 100, "lympho"),
        (dataset8, orig8, 100, "mammography"),
        (dataset9, orig9, 100, "mnist"),
        (dataset10, orig10, 100, "musk"),
        (dataset11, orig11, 100, "optdigits"),
        (dataset12, orig12, 100, "pendigits"),
        (dataset14, orig14, 100, "breast-cancer-unsupervised-ad")
    ]
    V2 = [
        (harvard1, harvardorig1, 100, "aloi-unsupervised-ad"),
        (harvard2, harvardorig2, 100, "annthyroid-unsupervised-ad"),
        (harvard3, harvardorig3, 100, "breast-cancer-unsupervised-ad"),
        (harvard4, harvardorig4, 100, "kdd99-unsupervised-ad"),
        (harvard5, harvardorig5, 100, "letter-unsupervised-ad"),
        (harvard6, harvardorig6, 100, "pen-global-unsupervised-ad"),
        (harvard7, harvardorig7, 100, "pen-local-unsupervised-ad"),
        (harvard8, harvardorig8, 100, "satellite-unsupervised-ad"),
        (harvard9, harvardorig9, 100, "shuttle-unsupervised-ad"),
        (harvard10, harvardorig10, 100, "speech-unsupervised-ad"),
    ]

    V3 = [
        (elki1, elkiorig1, 100, "ALOI"),
        (elki2, elkiorig2, 100, "ALOI_withoutdupl"),
        (elki3, elkiorig3, 100, "Glass_withoutdupl_norm"),
        (elki4, elkiorig4, 100, "Ionosphere_withoutdupl_norm"),
        (elki5, elkiorig5, 100, "KDDCup99_catremoved"),
        (elki6, elkiorig6, 100, "Lymphography_withoutdupl_catremoved"),
        (elki7, elkiorig7, 100, "Lymphography_withoutdupl_norm_1ofn"),
        (elki8, elkiorig8, 100, "PenDigits_withoutdupl_norm_v01"),
        (elki9, elkiorig9, 100, "Shuttle_withoutdupl_v01"),
        (elki10, elkiorig10, 100, "Waveform_withoutdupl_v01"),
        (elki11, elkiorig11, 100, "WBC_v01"),
        (elki12, elkiorig12, 100, "WDBC_withoutdupl_v01"),
        (elki13, elkiorig13, 100, "WPBC_withoutdupl_norm")

    ]

    # (elki_semantic6, elki_semanticorig6, 500, "InternetAds_norm_02_v01"),
    V4 = [
        (elki_semantic1, elki_semanticorig1, 100, "Annthyroid_02_v01"),
        (elki_semantic2, elki_semanticorig2, 100, "Arrhythmia_withoutdupl_02_v01"),
        (elki_semantic3, elki_semanticorig3, 100, "Cardiotocography_02_v01"),
        (elki_semantic4, elki_semanticorig4, 100, "HeartDisease_withoutdupl_02_v01"),
        (elki_semantic5, elki_semanticorig5, 100, "Hepatitis_withoutdupl_05_v01"),

        (elki_semantic7, elki_semanticorig7, 100, "PageBlocks_02_v01"),
        (elki_semantic8, elki_semanticorig8, 100, "Parkinson_withoutdupl_05_v01"),
        (elki_semantic9, elki_semanticorig9, 100, "Pima_withoutdupl_02_v01"),
        (elki_semantic10, elki_semanticorig10, 100, "SpamBase_02_v01"),
        (elki_semantic11, elki_semanticorig11, 100, "Stamps_withoutdupl_02_v01"),
        (elki_semantic12, elki_semanticorig12, 100, "Wilt_02_v01"),

    ]

    VAnnthyroid = [
        (annthyroid1, annthyroidorig1, 100, "Annthyroid_02_v01"),
        (annthyroid2, annthyroidorig2, 1500, "Annthyroid_05_v01"),
        (annthyroid3, annthyroidorig3, 100, "Annthyroid_07"),
        (annthyroid4, annthyroidorig4, 100, "Annthyroid_norm_02_v01"),
        (annthyroid5, annthyroidorig5, 100, "Annthyroid_norm_05_v01"),
        (annthyroid6, annthyroidorig6, 100, "Annthyroid_norm_07"),
        (annthyroid7, annthyroidorig7, 100, "Annthyroid_withoutdupl_02_v01"),
        (annthyroid8, annthyroidorig8, 100, "Annthyroid_withoutdupl_05_v01"),
        (annthyroid9, annthyroidorig9, 100, "Annthyroid_withoutdupl_07"),
        (annthyroid10, annthyroidorig10, 100, "Annthyroid_withoutdupl_norm_02_v01"),
        (annthyroid11, annthyroidorig11, 100, "Annthyroid_withoutdupl_norm_05_v01"),
        (annthyroid12, annthyroidorig12, 100, "Annthyroid_withoutdupl_norm_07"),

    ]

    VTEST2 = [
        (datasettest, datasettestorig, 100, "syntetischer Datensatz")
    ]

    processes = []
    # max = calc_auc_graph(data_, orig_, 500,"auto")
    # (dataset13, orig13, 500, "creditcard"),

    '''start_time = time.time()
    for data in VTEST2:
        p = multiprocessing.Process(target=calc_auc_graph, args=data)
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    end_time = time.time()
    print("Time taken: ", end_time - start_time)'''


    hbosmode = ("dynamic")
    hbosranked = False

    clf = HBOSPYOD()
    clf.set_params(n_bins="calc", smoothen=False, mode="dynamic", ranked=False, save_explainability_scores=True)
    clf.fit(datasettest)
    print(clf.labels_)

    # calc_roc_auc(orig_)

    # plot_explainability(0)

