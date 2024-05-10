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


class Evaluator:
    def __init__(self):
        self.aucs_static = []
        self.aucs_dynamic = []
        self.aucs_RBOS = []
        self.all_auto_auc_dynamic = []
        self.all_auto_bins_dynamic = []
        self.all_auto_auc_static = []
        self.all_auto_bins_static = []
        self.all_max_auc_static = []
        self.all_max_at_static = []
        self.all_max_auc_dynamic = []
        self.all_max_at_dynamic = []
        self.all_datanames = []

    def calc_aucs(self, data_, orig_, count, dataname, funcmode,result_queue):
        aucs_ = []
        maxat = 1
        maxauc = 0
        print(dataname)
        hbosranked = False
        for i in range(count):
            bins = i + 1
            if funcmode == "static":
                clfstatic = HBOSPYOD(n_bins=bins, ranked=hbosranked)
                clfstatic.fit(data_)
                scoresstatic = clfstatic.decision_scores_
                hbos_static = orig_.copy()
                hbos_static['scores'] = scoresstatic
                hbos_static_sorted = hbos_static.sort_values(by=['scores'], ascending=False)
                fpr1, tpr1, thresholds1 = metrics.roc_curve(hbos_static_sorted['Class'], hbos_static_sorted['scores'])
                aucstatic = metrics.auc(fpr1, tpr1)
                if aucstatic > maxauc:
                    maxauc = aucstatic
                    maxat = i + 1
                aucs_.append(aucstatic)

            elif funcmode == "dynamic":
                clfdynamic = HBOSPYOD(mode="dynamic", n_bins=bins, ranked=hbosranked)
                clfdynamic.fit(data_)
                scoresdynamic = clfdynamic.decision_scores_
                hbos_dynamic = orig_.copy()
                hbos_dynamic['scores'] = scoresdynamic
                hbos_dynamic_sorted = hbos_dynamic.sort_values(by=['scores'], ascending=False)
                fpr2, tpr2, thresholds2 = metrics.roc_curve(hbos_dynamic_sorted['Class'], hbos_dynamic_sorted['scores'])
                aucdynamic = metrics.auc(fpr2, tpr2)
                if aucdynamic > maxauc:
                    maxauc = aucdynamic
                    maxat = i + 1
                aucs_.append(aucdynamic)
        print(aucs_ , dataname, " ", funcmode)
        if funcmode == "RBOS":
            clfRBOS = HBOSPYOD()
            clfRBOS.fit2(data_)
            scoreRBOS = clfRBOS.decision_scores_
            rbos = orig_.copy()
            rbos['scores'] = scoreRBOS
            rbos_sorted = rbos.sort_values(by=['scores'], ascending=False)
            fpr3, tpr3, thresholds3 = metrics.roc_curve(rbos_sorted['Class'], rbos_sorted['scores'])
            aucrbos = metrics.auc(fpr3, tpr3)
            self.aucs_RBOS.append([aucrbos for _ in range(len(aucs_))])
        elif funcmode == "static":
            auto_auc_static, auto_bins_static = self.calc_roc_auc2(data_, orig_, "static", hbosranked)
            self.aucs_static.append(aucs_)
            self.all_auto_auc_static.append(auto_auc_static)
            self.all_auto_bins_static.append(auto_bins_static)
            self.all_max_auc_static.append(maxauc)
            self.all_max_at_static.append(maxat)
        elif funcmode == "dynamic":
            auto_auc_dynamic, auto_bins_dynamic = self.calc_roc_auc2(data_, orig_, "dynamic", hbosranked)
            self.aucs_dynamic.append(aucs_)
            self.all_auto_auc_dynamic.append(auto_auc_dynamic)
            self.all_auto_bins_dynamic.append(auto_bins_dynamic)
            self.all_max_auc_dynamic.append(maxauc)
            self.all_max_at_dynamic.append(maxat)
        print(aucs_ , dataname, " ", funcmode)

        self.all_datanames.append(dataname)
        result_queue.put((self.aucs_static, self.aucs_dynamic, self.aucs_RBOS,
                          self.all_auto_auc_dynamic, self.all_auto_bins_dynamic,
                          self.all_auto_auc_static, self.all_auto_bins_static,
                          self.all_max_auc_static, self.all_max_at_static,
                          self.all_max_auc_dynamic, self.all_max_at_dynamic,
                          self.all_datanames))


    def print_graph(self, count):
        print("print")
        for i in range(len(self.aucs_static)):
            aucsstatic = self.aucs_static[i]
            aucsdynamic = self.aucs_dynamic[i]
            aucsRBOS = self.aucs_RBOS[i]
            auto_dynamic_auc = self.all_auto_auc_dynamic[i]
            auto_dynamic_bins = self.all_auto_bins_dynamic[i]
            auto_static_auc = self.all_auto_auc_static[i]
            auto_static_bins = self.all_auto_bins_static[i]

            max_auc_static = self.all_max_auc_static[i]
            max_at_static = self.all_max_at_static[i]
            max_auc_dynamic = self.all_max_auc_static[i]
            max_at_dynamic = self.all_max_at_static[i]
            dataname = self.all_datanames[i]

            xval = range(1, count + 1)
            plt.figure(figsize=[8, 6])
            # plt.plot(xval, aucs_static, color='b', lw=2, label='mode: ' + hbosmode + ', ranked: {}'.format(hbosranked))
            plt.plot(xval, aucsstatic, color='b', lw=2, label='mode: static' + ', ranked: {}'.format(hbosranked))
            plt.plot(xval, aucsdynamic, color='r', lw=2, label='mode: dynamic' + ', ranked: {}'.format(hbosranked))
            plt.plot(xval, aucsRBOS, color='c', lw=2, label='mode: RBOS')
            plt.scatter(auto_static_bins, auto_static_auc, color='b', s=100, marker='s',
                        label='n_bins= sqrt(samples) static', zorder=10)
            plt.scatter(auto_dynamic_bins, auto_dynamic_auc, color='r', s=100, marker='x',
                        label='n_bins= sqrt(samples) dynamic', zorder=10)
            plt.xlabel('Number of Bins')
            plt.ylabel('Area Under the Curve (AUC)')
            plt.title('AUC vs. n_bins \n' + ' max AUC "static": {0:0.4f}'.format(max_auc_static) + ' at {}'.format(
                max_at_static) + ' bins \n' + ' max AUC "dynamic": {0:0.4f}'.format(max_auc_dynamic) + ' at {}'.format(
                max_at_dynamic))
            plt.legend(loc="lower right")
            plt.grid(True)
            # plt.text(0, -0.1, 't_static: {0:0.2f}'.format(duration)+ ' s', fontsize=12, color='black', ha='left',transform=plt.gca().transAxes)
            pfad = r'C:\Users\david\Desktop\datasets_hbos\ODDS\Dataset\static_and_dynamic\aucs'
            filename = f"{pfad}_plot_{dataname}.png"
            plt.savefig(filename)
            plt.show()

    def plot_explainability(self, id):
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

    def calc_roc_auc2(self, data_, orig_, mode_, ranked_):
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

    def calc_roc_auc(self, orig_, plot):
        clf_name = 'HBOS'
        scores = self.clf.decision_scores_
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

    ev = Evaluator()

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

    alldatasets = [
        (dataset1, orig1, 500, "annthyroid", "static"),
        (dataset1, orig1, 500, "annthyroid", "dynamic"),
        (dataset1, orig1, 500, "annthyroid", "RBOS"),
        (dataset2, orig2, 500, "cardio", "static"),
        (dataset2, orig2, 500, "cardio", "dynamic"),
        (dataset2, orig2, 500, "cardio", "RBOS"),

    ]
    processes = []
    # max = calc_auc_graph(data_, orig_, 500,"auto")
    result_queue = multiprocessing.Queue()
    start_time = time.time()
    for data in alldatasets:
        p = multiprocessing.Process(target=ev.calc_aucs, args=(*data,result_queue))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    for _ in range(len(alldatasets)):
        aucs_static, aucs_dynamic, aucs_RBOS, all_auto_auc_dynamic, all_auto_bins_dynamic, all_auto_auc_static, all_auto_bins_static, all_max_auc_static, all_max_at_static, all_max_auc_dynamic, all_max_at_dynamic, all_datanames = result_queue.get()

    end_time = time.time()
    print("Time taken: ", end_time - start_time)
    ev.print_graph(500)
    print(max)
    clf = HBOSPYOD(save_explainability_scores=True, ranked=hbosranked, mode=hbosmode, n_bins="auto")
    if fit2:
        clf.fit2(data_)
        clf.n_bins = "RBOS"
    else:
        clf.fit(data_)

    # ev.calc_roc_auc(orig_,True)

    # ev.plot_explainability(0)
