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


def plot_distributions(df, title):
    """
    Plots histograms and boxplots for all numeric columns in the dataframe.

    Parameters:
    df (pd.DataFrame): The dataframe containing the data.
    title (str): The title for the plots.
    """
    # Suche nach allen numerischen Spalten
    numeric_columns = df.select_dtypes(include='number').columns
    if len(numeric_columns) == 0:
        raise ValueError('Der DataFrame enthält keine numerischen Spalten.')

    for column in numeric_columns:
        # Histogram plot
        plt.figure(figsize=(12, 6))
        sns.histplot(df[column], kde=True)
        plt.title(f'Histogram of {column} - {title}')
        plt.show()


def calc_average_combined(args_):
    scaler = MinMaxScaler()

    combined = []
    combined_avg = 0
    bins_combined = []

    hbosranked = False

    smooth_ = False
    mode_ = "static"
    n_data = len(args_)
    print(n_data, "lol")
    norm = False

    for datatmp, labels_, a, name, in args_:
        labels_ = pd.DataFrame(labels_)
        value_counts = labels_.value_counts()

        anzahl_einsen = value_counts.get(1, 0)
        anzahl_nullen = value_counts.get(0, 0)

        print()

        print("dataset: ", name, "samples: ", len(datatmp), "features: ", datatmp.shape[1], "outlier: ", anzahl_einsen)

        '''if norm:
            datanorm = scaler.fit_transform(datatmp)
            datanorm = pd.DataFrame(datanorm)
            data_ = datanorm
        else:
            data_ = datatmp

        combined_auc, _, bins ,_= calc_roc_auc2(data_, labels_, mode_, hbosranked, smooth_, "combined3")
        combined.append(combined_auc)
        combined_avg = combined_avg + combined_auc
        bins_combined.append(bins)

    fd_st2_avg = combined_avg / n_data

    combined_var = np.std(combined)

    allauc = [[fd_st2_avg, "combined", combined_var]]

    for i in range(n_data):
        print(args_[i][3], ", samples: ", len(args_[i][0]), " features: ", args_[i][0].shape[1])
        print("combined", "  (AUC): ", round(combined[i], 5), bins_combined[i])

        print("-------------------------------------------------")

    values = []
    methods = []
    var = []

    for item in allauc:
        values.append(item[0])
        methods.append(item[1])
        var.append(item[2])

    # Plot
    plt.figure(figsize=(10, 6))
    plt.xlim([0.5, 1.0])
    plt.barh(methods, values, color='skyblue')
    plt.xlabel('Durchschnittlicher AUC')
    plt.ylabel('Methode')
    plt.title('Durchschnittliche AUC für verschiedene Methoden')
    plt.gca().invert_yaxis()  # Um die Anzeigereihenfolge der Methoden umzukehren
    plt.show()

    sorted_allauc = sorted(allauc, key=lambda x: x[0], reverse=True)

    for auc in sorted_allauc:
        print(auc[1], ": ", round(auc[0], 4), " s:", round(auc[2], 5))'''


def calc_average_static(args_):
    scaler = MinMaxScaler()

    auto = []
    auto_avg = 0
    bins_auto = []
    autotime = 0
    auto_pns = []

    br = []
    br_avg = 0
    bins_br = []
    brtime = 0
    br_pns = []

    scott = []
    scott_avg = 0
    bins_scott = []
    scotttime = 0
    scott_pns = []

    doane = []
    doane_avg = 0
    bins_doane = []
    doanetime = 0
    doane_pns = []

    fd = []
    fd_avg = 0
    bins_fd = []
    fdtime = 0
    fd_pns = []

    fd2 = []
    fd2_avg = 0
    bins_fd2 = []
    fd2time = 0
    fd2_pns = []

    combined = []
    combined_avg = 0
    bins_combined = []
    combinedtime = 0
    combined_pns = []

    fd_st = []
    fd_st_avg = 0
    bins_fd_st = []
    fd_sttime = 0
    fd_st_pns = []

    sturges = []
    sturges_avg = 0
    bins_sturges = []
    sturgestime = 0
    sturges_pns = []

    ten = []
    ten_avg = 0
    bins_ten = []
    tentime = 0
    ten_pns = []

    rice = []
    rice_avg = 0
    bins_rice = []
    ricetime = 0
    rice_pns = []

    fd_doane = []
    fd_doane_avg = 0
    bins_fd_doane = []
    fd_doanetime = 0
    fd_doane_pns = []

    combined2 = []
    combined2_avg = 0
    bins_combined2 = []
    combined2time = 0
    combined2_pns = []

    combined3 = []
    combined3_avg = 0
    bins_combined3 = []
    combined3time = 0
    combined3_pns = []

    hbosranked = True
    smooth_ = False
    mode_ = "static"
    n_data = len(args_)
    print(n_data)
    norm = False
    datasetnames = []

    for datatmp, labels_, _, datasetname in args_:
        if norm:
            datanorm = scaler.fit_transform(datatmp)
            datanorm = pd.DataFrame(datanorm)
            data_ = datanorm
        else:
            data_ = datatmp
        datasetnames.append(datasetname)
        starttime = time.time()
        auto_auc, _, bins, auto_pn = calc_roc_auc2(data_, labels_, mode_, hbosranked, smooth_, "sqrt","auto")
        auto.append(auto_auc)
        auto_avg = auto_avg + auto_auc
        bins_auto.append(bins)
        autotime = autotime + (time.time() - starttime)
        auto_pns.append(auto_pn)

        starttime = time.time()
        br_auc, _, bins, br_pn = calc_roc_auc2(data_, labels_, mode_, hbosranked, smooth_, "sqrt","auto")
        br.append(br_auc)
        br_avg = br_avg + br_auc
        bins_br.append(bins)
        brtime = brtime + (time.time() - starttime)
        br_pns.append(br_pn)

        starttime = time.time()
        scott_auc, _, bins, scott_pn = calc_roc_auc2(data_, labels_, mode_, hbosranked, smooth_, "scott","auto")
        scott.append(scott_auc)
        scott_avg = scott_avg + scott_auc
        bins_scott.append(bins)
        scotttime = scotttime + (time.time() - starttime)
        scott_pns.append(scott_pn)

        starttime = time.time()
        doane_auc, _, bins, doane_pn = calc_roc_auc2(data_, labels_, mode_, hbosranked, smooth_, "doane","auto")
        doane.append(doane_auc)
        doane_avg = doane_avg + doane_auc
        bins_doane.append(bins)
        doanetime = doanetime + (time.time() - starttime)
        doane_pns.append(doane_pn)

        starttime = time.time()
        fd_auc, _, bins, fd_pn = calc_roc_auc2(data_, labels_, mode_, hbosranked, smooth_, "fd","auto")
        fd.append(fd_auc)
        fd_avg = fd_avg + fd_auc
        bins_fd.append(bins)
        fdtime = fdtime + (time.time() - starttime)
        fd_pns.append(fd_pn)

        starttime = time.time()
        fd2_auc, _, bins, fd2_pn = calc_roc_auc2(data_, labels_, mode_, hbosranked, smooth_, "fd","auto")
        fd2.append(fd2_auc)
        fd2_avg = fd2_avg + fd2_auc
        bins_fd2.append(bins)
        fd2time = fd2time + (time.time() - starttime)
        fd2_pns.append(fd2_pn)

        starttime = time.time()
        combined_auc, _, bins, combined_pn = calc_roc_auc2(data_, labels_, mode_, hbosranked, smooth_, "sqrt","auto")
        combined.append(combined_auc)
        combined_avg = combined_avg + combined_auc
        bins_combined.append(bins)
        combinedtime = combinedtime + (time.time() - starttime)
        combined_pns.append(combined_pn)

        starttime = time.time()
        fd_st_auc, _, bins, fd_st_pn = calc_roc_auc2(data_, labels_, mode_, hbosranked, smooth_, "fd_st","auto")
        fd_st.append(fd_st_auc)
        fd_st_avg = fd_st_avg + fd_st_auc
        bins_fd_st.append(bins)
        fd_sttime = fd_sttime + (time.time() - starttime)
        fd_st_pns.append(fd_st_pn)

        starttime = time.time()
        sturges_auc, _, bins, sturges_pn = calc_roc_auc2(data_, labels_, mode_, hbosranked, smooth_, "sturges","auto")
        sturges.append(sturges_auc)
        sturges_avg = sturges_avg + sturges_auc
        bins_sturges.append(bins)
        sturgestime = sturgestime + (time.time() - starttime)
        sturges_pns.append(sturges_pn)

        starttime = time.time()
        ten_auc, _, bins, ten_pn = calc_roc_auc2(data_, labels_, mode_, hbosranked, smooth_, "ten","auto")
        ten.append(ten_auc)
        ten_avg = ten_avg + ten_auc
        bins_ten.append(bins)
        tentime = tentime + (time.time() - starttime)
        ten_pns.append(ten_pn)

        starttime = time.time()
        rice_auc, _, bins, rice_pn = calc_roc_auc2(data_, labels_, mode_, hbosranked, smooth_, "rice","auto")
        rice.append(rice_auc)
        rice_avg = rice_avg + rice_auc
        bins_rice.append(bins)
        ricetime = ricetime + (time.time() - starttime)
        rice_pns.append(rice_pn)

        starttime = time.time()
        fd_doane_auc, _, bins, fd_doane_pn = calc_roc_auc2(data_, labels_, mode_, hbosranked, smooth_, "scott_doane","auto")
        fd_doane.append(fd_doane_auc)
        fd_doane_avg = fd_doane_avg + fd_doane_auc
        bins_fd_doane.append(bins)
        fd_doanetime = fd_doanetime + (time.time() - starttime)
        fd_doane_pns.append(fd_doane_pn)

        starttime = time.time()
        combined2_auc, _, bins, combined2_pn = calc_roc_auc2(data_, labels_, mode_, hbosranked, smooth_, "st_doane","auto")
        combined2.append(combined2_auc)
        combined2_avg = combined2_avg + combined2_auc
        bins_combined2.append(bins)
        combined2time = combined2time + (time.time() - starttime)
        combined2_pns.append(combined2_pn)

        starttime = time.time()
        combined3_auc, _, bins, combined3_pn = calc_roc_auc2(data_, labels_, mode_, hbosranked, smooth_, "sqrt","auto")
        combined3.append(combined3_auc)
        combined3_avg = combined3_avg + combined3_auc
        bins_combined3.append(bins)
        combined3time = combined3time + (time.time() - starttime)
        combined3_pns.append(combined3_pn)

    auto_avg = auto_avg / n_data
    auto_avg2 = np.mean(auto)

    br_avg = br_avg / n_data
    scott_avg = scott_avg / n_data
    doane_avg = doane_avg / n_data
    fd_avg = fd_avg / n_data
    fd2_avg = fd2_avg / n_data
    # combined_avg = combined_avg / n_data
    fd_st_avg = fd_st_avg / n_data
    sturges_avg = sturges_avg / n_data
    ten_avg = ten_avg / n_data
    rice_avg = rice_avg / n_data
    fd_doane_avg = fd_doane_avg / n_data
    combined2_avg = combined2_avg / n_data
    # combined3_avg = combined3_avg / n_data

    auto_avg_pn = np.mean(auto_pns)
    br_avg_pn = np.mean(br_pns)
    scott_avg_pn = np.mean(scott_pns)
    doane_avg_pn = np.mean(doane_pns)
    fd_avg_pn = np.mean(fd_pns)
    fd2_avg_pn = np.mean(fd2_pns)
    # combined_avg_pn = np.mean(combined_pns)
    fd_st_avg_pn = np.mean(fd_st_pns)
    sturges_avg_pn = np.mean(sturges_pns)
    ten_avg_pn = np.mean(ten_pns)
    rice_avg_pn = np.mean(rice_pns)
    fd_doane_avg_pn = np.mean(fd_doane_pns)
    combined2_avg_pn = np.mean(combined2_pns)
    # combined3_avg_pn = np.mean(combined3_pns)

    '''auto_var=np.var(auto)
    br_var = np.var(br)
    scott_var = np.var(scott)
    doane_var = np.var(doane)
    fd_var = np.var(fd)
    combined_var = np.var(fd_st1)
    fd_st2_var = np.var(fd_st2)
    sturges_var = np.var(sturges)
    ten_var = np.var(ten)
    rice_var = np.var(rice)'''

    auto_var = np.std(auto)
    br_var = np.std(br)
    scott_var = np.std(scott)
    doane_var = np.std(doane)
    fd_var = np.std(fd)
    fd2_var = np.std(fd2)
    # combined_var = np.std(combined)
    fd_st_var = np.std(fd_st)
    sturges_var = np.std(sturges)
    ten_var = np.std(ten)
    rice_var = np.std(rice)
    fd_doane_var = np.std(fd_doane)
    combined2_var = np.std(combined2)
    # combined3_var = np.std(combined3)

    allauc = []
    allauc.append([(fd_avg), "fd", fd_var, fdtime])
    allauc.append([(fd2_avg), "fd min 2", fd2_var, fd2time])
    allauc.append([(br_avg), "br", br_var, brtime])
    allauc.append([(auto_avg), "sqrt", auto_var, autotime])
    allauc.append([(fd_st_avg), "fd_st", fd_st_var, fd_sttime])
    allauc.append([(fd_doane_avg), "scott_doane", fd_doane_var, fd_doanetime])
    allauc.append([(scott_avg), "scott", scott_var, scotttime])
    # allauc.append([(combined_avg), "combined", combined_var, combinedtime])
    allauc.append([(rice_avg), "rice", rice_var, ricetime])
    allauc.append([(ten_avg), "ten", ten_var, tentime])
    allauc.append([(doane_avg), "doane", doane_var, doanetime])
    allauc.append([(sturges_avg), "sturges", sturges_var, sturgestime])
    allauc.append([(combined2_avg), "st_doane", combined2_var, combined2time])
    # allauc.append([(combined3_avg), "combined3", combined3_var, combined3time])

    allpn = []
    allpn.append([(fd_avg_pn), "fd"])
    allpn.append([(fd2_avg_pn), "fd min 2"])
    allpn.append([(br_avg_pn), "br"])
    allpn.append([(auto_avg_pn), "sqrt"])
    allpn.append([(fd_st_avg_pn), "fd_st"])
    allpn.append([(fd_doane_avg_pn), "scott_doane"])
    allpn.append([(scott_avg_pn), "scott"])
    # allpn.append([(combined_avg_pn), "combined"])
    allpn.append([(rice_avg_pn), "rice"])
    allpn.append([(ten_avg_pn), "ten"])
    allpn.append([(doane_avg_pn), "doane"])
    allpn.append([(sturges_avg_pn), "sturges"])
    allpn.append([(combined2_avg_pn), "st_doane"])
    # allpn.append([(combined3_avg_pn), "combined3"])

    for i in range(n_data):
        print(args_[i][3], ", samples: ", len(args_[i][0]), " features: ", args_[i][0].shape[1])

        print("br", "       (AUC): ", round(br[i], 5), bins_br[i])
        print("scott", "    (AUC): ", round(scott[i], 5), bins_scott[i])
        print("sturges", "  (AUC): ", round(sturges[i], 5), bins_sturges[i])
        print("doane", "    (AUC): ", round(doane[i], 5), bins_doane[i])
        print("fd", "       (AUC): ", round(fd[i], 5), bins_fd[i])
        #print("fd2", "      (AUC): ", round(fd2[i], 5), bins_fd2[i])
        # print("combined", " (AUC): ", round(combined[i], 5), bins_combined[i])
        # print("combined3", "(AUC): ", round(combined3[i], 5), bins_combined3[i])
        #print("st_doane ", "(AUC): ", round(combined2[i], 5), bins_combined2[i])
        print("scott_doane", " (AUC): ", round(fd_doane[i], 5), bins_fd_doane[i])
        print("fd_st", "    (AUC): ", round(fd_st[i], 5), bins_fd_st[i])
        print("ten", "      (AUC): ", round(ten[i], 5), bins_ten[i])
        print("sqrt", "     (AUC): ", round(auto[i], 5), bins_auto[i])
        print("rice", "     (AUC): ", round(rice[i], 5), bins_rice[i])
        print("-------------------------------------------------")

    values = []
    methods = []
    var = []

    for item in allauc:
        values.append(item[0])
        methods.append(item[1])
        var.append(item[2])

    # Plot
    plt.figure(figsize=(10, 6))
    plt.xlim([0.8, 0.875])
    plt.barh(methods, values, color='skyblue')
    plt.xlabel('Durchschnittlicher AUC')
    plt.ylabel('Methode')
    plt.title('Durchschnittliche AUC für verschiedene Methoden')
    plt.gca().invert_yaxis()  # Um die Anzeigereihenfolge der Methoden umzukehren
    plt.show()

    sorted_allauc = sorted(allauc, key=lambda x: x[0], reverse=True)
    sorted_allpn = sorted(allpn, key=lambda x: x[0], reverse=True)

    for auc in sorted_allauc:
        print(auc[1], ": ", round(auc[0], 4), " s:", round(auc[2], 5), " time: ", auc[3])

    print("-------------------------------------------------")

    for pn in sorted_allpn:
        print(pn[1], ": ", round(pn[0], 4))

    exceldata = [
        ['Dataset', 'AVG-AUC'],

    ]

    fd_list = [round(num, 4) for num in fd]
    fdmin2_list = [round(num, 4) for num in fd2]
    br_list = [round(num, 4) for num in br]
    sqrt_list = [round(num, 4) for num in auto]
    fd_st_list = [round(num, 4) for num in fd_st]
    fd_doane_list = [round(num, 4) for num in fd_doane]
    scott_list = [round(num, 4) for num in scott]
    # combined_list = [round(num, 4) for num in combined]
    rice_list = [round(num, 4) for num in rice]
    ten_list = [round(num, 4) for num in ten]
    doane_list = [round(num, 4) for num in doane]
    sturges_list = [round(num, 4) for num in sturges]
    st_doane_list = [round(num, 4) for num in combined2]
    # combined3_list = [round(num, 4) for num in combined3]

    exceldata = []
    for dname, fd, fdmin2, br, sqrt, fd_st, fd_doane, scott, rice, ten, doane, sturges, st_doane in zip(
            datasetnames, fd_list, fdmin2_list, br_list, sqrt_list, fd_st_list, fd_doane_list, scott_list
            , rice_list, ten_list, doane_list, sturges_list, st_doane_list):
        exceldata.append(
            [dname, fd, fdmin2, br, sqrt, fd_st, fd_doane, scott, combined, rice, ten, doane, sturges, st_doane,
             combined3])

    # Spaltenüberschriften definieren
    header = ["Dataset Name", "FD", "FDmin2", "BR", "Sqrt", "FD_ST", "FD_Doane", "Scott", "Combined", "Rice", "Ten",
              "Doane", "Sturges", "ST_Doane", "Combined3"]

    # DataFrame erstellen
    df = pd.DataFrame(exceldata, columns=header)

    # DataFrame in eine Excel-Datei schreiben
    df.to_excel("example_pandas.xlsx", index=False)

    '''for dname, avg_auc in zip(datasetnames, combined_list):
        exceldata.append([dname, avg_auc])

    df = pd.DataFrame(exceldata[1:], columns=exceldata[0])

    # DataFrame in eine Excel-Datei schreiben
    df.to_excel("example_pandas.xlsx", index=False)'''

    '''endtime= time.time()
    print(auto_avg,"auto")
    print(br_avg,"br")
    print(scott_avg,"scott")
    print(doane_avg,"doane")
    print(fd_avg,"fd")
    print(fd_st1_avg,"fd_st1")
    print(fd_st2_avg,"fd_st2")
    print(sturges_avg,"struges")
    print(ten_avg,"ten")
    print("time: ", endtime-start_time)'''


def plot_boxplot(args_):
    estimators_ = [1,2]

    mode_ = "static"
    hbosranked= False
    smooth_= False
    auc_values=[]
    test=1
    dataset_names=[]
    print(len(args_))

    for estimator_ in estimators_:
        estimator_aucs=[]

        for datatmp, labels_, _, datasetname in args_:

            print("name: ", datasetname, " N: ", len(datatmp), "features: ", datatmp.shape[1], "outlier: ",  np.sum(labels_ == 1))

            if estimator_==1:
                estimator_="fd_st"
                mode_ = "static"
            elif estimator_ ==2:
                estimator_ = 101
                mode_ = "dynamic"



            auc, _, bins, _ = calc_roc_auc2(datatmp, labels_, mode_, hbosranked, smooth_, estimator_,test)
            estimator_aucs.append(auc)
        auc_values.append(estimator_aucs)
    estimators = estimators_
    estimator_names = estimators
    for datatmp, labels_, _, datasetname in args_:
        dataset_names.append(datasetname)




    # DataFrame erstellen
    data = {
        "model": np.repeat(estimators, [len(auc) for auc in auc_values]),
        "AUC": [auc for sublist in auc_values for auc in sublist]
    }

    df = pd.DataFrame(data)

    # Boxplot und Stripplot erstellen
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(data=df, x="model", y="AUC", boxprops={'alpha': 0.4}, dodge=False)
    sns.stripplot(data=df, x="model", y="AUC", hue="model", dodge=False, palette="deep", ax=ax)



    # Plot anzeigen
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()

    medians = df.groupby('model')['AUC'].median()
    means = df.groupby('model')['AUC'].mean()
    std_devs = df.groupby('model')['AUC'].std(ddof=0)

    # In ein DataFrame umwandeln, wobei jede Spalte einen Estimator darstellt
    df2 = pd.DataFrame(np.round(auc_values,4)).transpose()
    df2.columns = estimators
    df2.insert(0, 'Datasets', dataset_names)




    avg_aucs2  = np.mean(auc_values, axis=1)
    avg_std2  = np.std(auc_values, axis=1)
    new_row = ['Average AUC'] + np.round(avg_aucs2,4).tolist()
    # Append the new row
    df2.loc[len(df2)] = new_row

    new_row = ['Standard Deviation'] + np.round(avg_std2,4).tolist()
    # Append the new row
    df2.loc[len(df2)] = new_row


    # Excel-Datei speichern
    df2.to_excel("auc_values.xlsx", index=False)



    # Ausgabe der Ergebnisse
    print("Median AUC Werte für jeden Schätzer:")
    print(medians.sort_values(ascending=False))

    print("\nDurchschnittliche AUC Werte für jeden Schätzer:")
    print(means.sort_values(ascending=False))

    print("\nStandardabweichung der AUC Werte für jeden Schätzer:")
    print(std_devs.sort_values(ascending=True))

    # Erstellen des Boxplots und der Punkte
    for i, dataset_name in enumerate(dataset_names):
        # Neue Figur für jeden Datensatz erstellen
        plt.figure(figsize=(10, 6))

        dataset_auc = [auc_values[j][i] for j in range(len(estimator_names))]

        # Boxplot für den aktuellen Datensatz
        sns.boxplot(data=dataset_auc, palette="Set3")

        # Punkte für die AUC-Werte der Estimatoren
        for j, estimator_name in enumerate(estimator_names):
            y = auc_values[j][i]
            x = np.random.normal(j + 1, 0.04, size=1)  # Zufällige X-Position für die Punkte
            plt.plot(x, y, marker='o', markersize=8, color='black', alpha=0.7)  # Alle Punkte in Schwarz


        # Achsenbeschriftungen und Titel
        plt.xlabel("model")
        plt.ylabel("AUC")
        # X-Achsenbeschriftungen setzen
        plt.xticks(range(1, len(estimator_names) + 1), estimator_names)
        plt.ylim(0.5, 1.1)
        # Bild speichern oder anzeigen
        plt.tight_layout()
        plt.show()  # Anzeigen (kann auch auskommentiert werden, wenn nur gespeichert werden soll)'''

def calc_average_dynamic2(args_):
    all_aucs = []
    all_aucsranked=[]
    all_tupel = []
    datasetnames = []
    hbosranked = False
    xval=[]
    yval=[]
    xvalranked =[]
    yvalranked =[]
    smooth_ = False
    mode_ = "dynamic"
    n_data = len(args_)
    print(n_data)

    for i in range(199):
        bins_ = i+2
        xval.append(bins_)
        print(bins_)
        aucs = []
        aucsranked=[]
        for datatmp, labels_, _, datasetname in args_:


            data_ = datatmp
            datasetnames.append(datasetname)
            auto_auc, _, bins, _ = calc_roc_auc2(data_, labels_, mode_, hbosranked, smooth_, bins_,"auto")
            aucs.append(auto_auc)

            auto_auc_ranked, _, bins, _ = calc_roc_auc2(data_, labels_, mode_, True, smooth_, bins_,"auto")
            aucsranked.append(auto_auc_ranked)

        value = round(np.mean(aucs), 4)
        valueranked = round(np.mean(aucsranked), 4)
        std = round(np.std(aucs), 4)
        tupel = (value, std, bins_)
        all_tupel.append(tupel)
        all_aucs.append(aucs)
        all_aucsranked.append(aucsranked)
        yval.append(value)
        yvalranked.append(valueranked)

    plt.figure(figsize=[8, 6])
    plt.plot(xval, yval, color='r', lw=1, label='HBOS (dynamic mode)')
    plt.plot(xval, yvalranked, color='darkred', lw=1, label='HBOS (dynamic mode,ranked)')


    plt.xlabel('number of bins')
    plt.ylabel('average AUC')
    plt.legend(loc="lower right")
    plt.ylim(0.4, 1.1)
    plt.grid(True)
    plt.show()
    sortierte_tupel = sorted(all_tupel, key=lambda x: x[0])
    print(sortierte_tupel)


def calc_average_dynamic(args_):
    start_time = time.time()
    auto = []
    auto_avg = 0
    bins_auto = []
    autotime = 0

    br = []
    br_avg = 0
    bins_br = []
    brtime = 0

    unique = []
    unique_avg = 0
    bins_unique = []
    uniquetime = 0

    ten = []
    ten_avg = 0
    bins_ten = []
    tentime = 0

    doane = []
    doane_avg = 0
    bins_doane = []
    doanetime = 0

    scaler = MinMaxScaler()

    fd_st1 = []
    fd_st1_avg = 0
    bins_fd_st1 = []
    fd_st1time = 0

    datasetnames = []
    hbosranked = False

    smooth_ = False
    mode_ = "dynamic"
    n_data = len(args_)
    print(n_data)

    for datatmp, labels_, _, datasetname in args_:
        data_ = datatmp

        datasetnames.append(datasetname)
        print(datasetname)
        starttime = time.time()
        auto_auc, _, bins, _ = calc_roc_auc2(data_, labels_, mode_, hbosranked, smooth_, "sqrt","auto")
        auto.append(auto_auc)
        auto_avg = auto_avg + auto_auc
        bins_auto.append(bins)
        autotime = autotime + (time.time() - starttime)

        starttime = time.time()
        unique_auc, _, bins, _ = calc_roc_auc2(data_, labels_, mode_, hbosranked, smooth_, 49,"auto")
        unique.append(unique_auc)
        unique_avg = unique_avg + unique_auc
        bins_unique.append(bins)
        uniquetime = uniquetime + (time.time() - starttime)

        number = 200
        starttime = time.time()
        br_auc, _, bins, _ = calc_roc_auc2(data_, labels_, mode_, hbosranked, smooth_, 101,"auto")
        br.append(br_auc)
        br_avg = br_avg + br_auc
        bins_br.append(bins)
        brtime = brtime + (time.time() - starttime)

        starttime = time.time()
        fd_st1_auc, _, bins, _ = calc_roc_auc2(data_, labels_, mode_, hbosranked, smooth_, "m1","auto")
        fd_st1.append(fd_st1_auc)
        fd_st1_avg = fd_st1_avg + fd_st1_auc
        bins_fd_st1.append(bins)
        fd_st1time = fd_st1time + (time.time() - starttime)

        starttime = time.time()
        ten_auc, _, bins, _ = calc_roc_auc2(data_, labels_, mode_, hbosranked, smooth_, "ten","auto")
        ten.append(ten_auc)
        ten_avg = ten_avg + ten_auc
        bins_ten.append(bins)
        tentime = tentime + (time.time() - starttime)

        starttime = time.time()
        doane_auc, _, bins, _ = calc_roc_auc2(data_, labels_, "static", hbosranked, smooth_, "st_doane","auto")
        doane.append(doane_auc)
        doane_avg = doane_avg + doane_auc
        bins_doane.append(bins)
        doanetime = doanetime + (time.time() - starttime)

    auto_avg = np.mean(auto)
    br_avg = np.mean(br)
    unique_avg = np.mean(unique)
    fd_st1_avg = np.mean(fd_st1)
    doane_avg = np.mean(doane)
    ten_avg = np.mean(ten)

    auto_var = np.std(auto)
    br_var = np.std(br)
    doane_var = np.std(doane)
    combined_var = np.std(fd_st1)
    ten_var = np.std(ten)
    unique_var = np.std(unique)

    allauc = []
    allauc.append([(unique_avg), "49", unique_var, uniquetime])
    allauc.append([(auto_avg), "sqrt", auto_var, autotime])
    allauc.append([(br_avg), "101", br_var, brtime])
    allauc.append([(doane_avg), "st_doane_static", doane_var, doanetime])
    allauc.append([(fd_st1_avg), "m1", combined_var, fd_st1time])
    allauc.append([(ten_avg), "ten", ten_var, tentime])

    for i in range(n_data):
        print(args_[i][3], ", samples: ", len(args_[i][0]), " features: ", args_[i][0].shape[1])

        print("st_doane_static", "  (AUC): ", round(doane[i], 5), bins_doane[i])

        print("m1", "  (AUC): ", round(fd_st1[i], 5), bins_fd_st1[i])

        print("sqrt", "    (AUC): ", round(auto[i], 5), bins_auto[i])

        print("49", "     (AUC): ", round(unique[i], 5), bins_unique[i])

        print("101", "     (AUC): ", round(br[i], 5), bins_br[i])

        print("-------------------------------------------------")

    values = []
    methods = []
    var = []

    for item in allauc:
        values.append(item[0])
        methods.append(item[1])
        var.append(item[2])

    exceldata = [
        ['Dataset', 'AVG-AUC'],

    ]

    rounded_data_list = [round(num, 4) for num in br]
    for dname, avg_auc in zip(datasetnames, rounded_data_list):
        exceldata.append([dname, avg_auc])

    df = pd.DataFrame(exceldata[1:], columns=exceldata[0])

    # DataFrame in eine Excel-Datei schreiben
    df.to_excel("example_pandas.xlsx", index=False)

    print(br)
    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(methods, values, color='skyblue')
    plt.xlabel('Durchschnittlicher AUC')
    plt.ylabel('Methode')
    plt.title('Durchschnittliche AUC für verschiedene Methoden')
    plt.gca().invert_yaxis()  # Um die Anzeigereihenfolge der Methoden umzukehren
    plt.show()

    sorted_allauc = sorted(allauc, key=lambda x: x[0], reverse=True)

    for auc in sorted_allauc:
        print(auc[1], ": ", round(auc[0], 4), " s:", round(auc[2], 5), " time: ", auc[3])


def calc_auc_graph_static_or_dynamic_2(datatmp, labels_, count, dataname):
    print(dataname, ": current dataset")
    scaler = MinMaxScaler()
    aucs_static = []
    maxatstatic = 0
    maxaucstatic = 0
    norm = True

    if norm:
        datanorm = scaler.fit_transform(datatmp)
        datanorm = pd.DataFrame(datanorm)
        data_ = datanorm
    else:
        data_ = datatmp

    features = data_.shape[1]
    samples = len(data_)

    xval = []
    hbosranked = False
    smooth_ = False
    mode_ = "static"
    start_time = time.time()

    for i in range(count):
        bins = i + 2
        xval.append(bins)

        clfstatic = HBOSPYOD(mode=mode_, n_bins=bins, ranked=hbosranked, adjust=smooth_)
        clfstatic.fit(data_)
        scoresstatic = clfstatic.decision_scores_
        hbos_static = data_.copy()
        hbos_static['Scores'] = scoresstatic
        hbos_static['Class'] = labels_
        hbos_static_sorted = hbos_static.sort_values(by=['Scores'], ascending=False)
        fpr1, tpr1, thresholds1 = metrics.roc_curve(hbos_static_sorted['Class'], hbos_static_sorted['Scores'])
        aucstatic = metrics.auc(fpr1, tpr1)
        if aucstatic > maxaucstatic:
            maxaucstatic = aucstatic
            maxatstatic = bins
        aucs_static.append(aucstatic)

    end_time = time.time()
    print("Time taken to run: ", dataname, end_time - start_time, "seconds.")

    auto_auc_static, auto_bins_static, _ = calc_roc_auc2(data_, labels_, mode_, hbosranked, smooth_, "sqrt")
    br_auc, _, _ = calc_roc_auc2(data_, labels_, mode_, hbosranked, smooth_, "br")
    scott_auc, _, _ = calc_roc_auc2(data_, labels_, mode_, hbosranked, smooth_, "scott")
    doane_auc, _, _ = calc_roc_auc2(data_, labels_, mode_, hbosranked, smooth_, "doane")
    fd_auc, _, _ = calc_roc_auc2(data_, labels_, mode_, hbosranked, smooth_, "fd")
    fd_st1_auc, _, _ = calc_roc_auc2(data_, labels_, mode_, hbosranked, smooth_, "fd_st1")
    fd_st2_auc, _, _ = calc_roc_auc2(data_, labels_, mode_, hbosranked, smooth_, "fd_st2")
    sturges_auc, _, _ = calc_roc_auc2(data_, labels_, mode_, hbosranked, smooth_, "sturges")
    rice_auc, rice_bins, _ = calc_roc_auc2(data_, labels_, mode_, hbosranked, smooth_, "rice")
    ten_auc, _, _ = calc_roc_auc2(data_, labels_, mode_, hbosranked, smooth_, 10)
    brval = [br_auc] * len(xval)
    scottval = [scott_auc] * len(xval)
    doaneval = [doane_auc] * len(xval)
    fdval = [fd_auc] * len(xval)
    tenval = [ten_auc] * len(xval)
    sturgesval = [sturges_auc] * len(xval)
    fd_st1_val = [fd_st1_auc] * len(xval)
    fd_st2_val = [fd_st2_auc] * len(xval)
    rice_val = [rice_auc] * len(xval)

    plt.figure(figsize=[10, 8])

    if mode_ == "static":
        plt.plot(xval, aucs_static, color='blue', lw=1,
                 label='mode: ' + mode_ + ', smoothen: ' + str(smooth_) + ', ranked: ' + str(hbosranked), zorder=10)
        plt.plot(xval, brval, color='orange', lw=2, label="Birge Rozenblac: {0:0.4f}".format(br_auc), zorder=9)
        plt.plot(xval, scottval, color='green', lw=2, label="scott : {0:0.4f}".format(scott_auc), zorder=9)
        plt.plot(xval, doaneval, color='indigo', lw=2, label="doane: {0:0.4f}".format(doane_auc), zorder=9)

        plt.plot(xval, fd_st2_val, color='dimgrey', lw=2, label="fd & sturges: {0:0.4f}".format(fd_st2_auc), zorder=9)
        plt.plot(xval, fdval, color='yellow', lw=2, label="fd : {0:0.4f}".format(fd_auc), zorder=9)
        plt.plot(xval, tenval, color='black', lw=2, label="10: {0:0.4f}".format(ten_auc), zorder=9)
        plt.plot(xval, sturgesval, color='cyan', lw=2, label="sturges: {0:0.4f}".format(sturges_auc), zorder=9)
        plt.plot(xval, rice_val, color='pink', lw=2,
                 label="rice: {0:0.4f}".format(rice_auc) + "bins: {}".format(rice_bins), zorder=9)
        plt.plot(xval, fd_st1_val, color='red', lw=2, label="combined: {0:0.4f}".format(fd_st1_auc), zorder=9)

    if mode_ == "dynamic":
        plt.plot(xval, aucs_static, color='red', lw=1, label='mode: ' + mode_ + ', ranked: ' + str(hbosranked),
                 zorder=10)
        plt.plot(xval, brval, color='orange', lw=2, label="Birge Rozenblac: {0:0.4f}".format(br_auc), zorder=9)
        # plt.plot(xval, , color='green', lw=2, label=": {0:0.4f}".format(), zorder=9)
        # plt.plot(xval, doaneval, color='indigo', lw=2, label="doane: {0:0.4f}".format(doane_auc), zorder=9)
        plt.plot(xval, fdval, color='dimgrey', lw=2, label="fd: {0:0.4f}".format(fd_auc), zorder=9)
        plt.plot(xval, tenval, color='green', lw=2, label="10: {0:0.4f}".format(ten_auc), zorder=9)

    label_ = "n_bins= sqrt: {}".format(auto_bins_static) + " bins" + ", AUC: {0:0.4f}".format(auto_auc_static)
    plt.scatter(auto_bins_static, auto_auc_static, color='k', s=100, marker='X', label=label_, zorder=11)

    plt.xlabel('Number of Bins')
    plt.ylabel('Area Under the Curve (AUC)')

    plt.title(
        'AUC vs. n_bins: ' + dataname + ', samples: {}'.format(samples) + ', features: {}'.format(
            features) + '\n' + ' max AUC "' + mode_ + '": {0:0.4f}'.format(maxaucstatic) + ' at {}'.format(
            maxatstatic) + ' bins \n')

    plt.legend(loc="lower right")
    plt.ylim(0.4, 1.1)
    plt.grid(True)
    # plt.text(0, -0.1, 't_static: {0:0.2f}'.format(duration)+ ' s', fontsize=12, color='black', ha='left',transform=plt.gca().transAxes)
    # pfad=r'C:\Users\david\Desktop\datasets_hbos\ODDS\Dataset\benchmarks'
    # pfad = r'C:\Users\david\Desktop\datasets_hbos\Harvard Dataverse\benchmarks'
    # pfad = r'C:\Users\david\Desktop\datasets_hbos\ELKI\benchmarks'
    # pfad = r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\benchmarks'
    pfad = r'C:\Users\david\Desktop\datasets_hbos\Benchmarks'
    filename = f"{pfad}\{dataname}_{mode_}.png"
    plt.savefig(filename)
    plt.show()

    return maxatstatic


def calc_auc_graph_static_or_dynamic(data_, labels_, count, dataname):
    features = data_.shape[1]
    samples = len(data_)
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
    xvaldyn = []
    xvalsmooth = []
    hbosranked = False
    norm = False
    mode_ = "dynamic"
    start_time = time.time()

    for i in range(200):
        bins = i + 2
        xval.append(bins)
        xvaldyn.append(bins)

        clfstatic = HBOSPYOD(mode=mode_, n_bins=bins, ranked=False)
        clfstatic.fit(data_)
        scoresstatic = clfstatic.decision_scores_
        hbos_static = data_.copy()
        hbos_static['scores'] = scoresstatic
        hbos_static['Class'] = labels_
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
        hbos_ranked = data_.copy()
        hbos_ranked['scores'] = scoresranked
        hbos_ranked['Class'] = labels_
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
                clfsmooth = HBOSPYOD(mode=mode_, n_bins=bins, ranked=False, adjust=True)
                clfsmooth.fit(data_)
                scoressmooth = clfsmooth.decision_scores_
                hbos_smooth = data_.copy()
                hbos_smooth['scores'] = scoressmooth
                hbos_smooth['Class'] = labels_
                hbos_smooth_sorted = hbos_smooth.sort_values(by=['scores'], ascending=False)
                fpr2, tpr2, thresholds2 = metrics.roc_curve(hbos_smooth_sorted['Class'], hbos_smooth_sorted['scores'])
                aucsmooth = metrics.auc(fpr2, tpr2)
                if aucsmooth > maxaucsmooth:
                    maxaucsmooth = aucsmooth
                    maxatsmooth = bins
                aucs_smooth.append(aucsmooth)
    end_time = time.time()
    print("Time taken to run: ", dataname, end_time - start_time, "seconds.")

    #auto_auc_static, auto_bins_static, _, _ = calc_roc_auc2(data_, labels_, mode_, hbosranked, False, "sqrt")
    #auto_auc_ranked, auto_bins_ranked, _ ,_= calc_roc_auc2(data_, labels_, mode_, True, False, "sqrt")
    # calc_auc, calc_bins = calc_roc_auc2(data_, orig_, mode_, hbosranked, False, "calc")
    # calc_auc_ranked, calc_bins_ranked = calc_roc_auc2(data_, orig_, mode_, True, False, "calc")
    # calc2_auc, calc2_bins = calc_roc_auc2(data_, orig_, mode_, hbosranked, False, "calc2")
    # calc2_auc_ranked, calc2_bins_static = calc_roc_auc2(data_, orig_, mode_, True, False, "calc2")

    if mode_ == "static":
        auto_auc_smooth, auto_bins_smooth, _ ,_= calc_roc_auc2(data_, labels_, mode_, hbosranked, True, "sqrt")
        plt.figure(figsize=[10, 8])
        plt.plot(xval, aucs_static, color='blue', lw=1, label='mode: ' + mode_)
        plt.plot(xval, aucs_ranked, color='midnightblue', lw=1, label='mode: ' + mode_ + ' ranked')
        plt.plot(xvalsmooth, aucs_smooth, color='cyan', lw=1, label='mode: ' + mode_ + ' smooth')

    if mode_ == "dynamic":
        plt.figure(figsize=[10, 8])
        plt.plot(xvaldyn, aucs_static, color='red', lw=1, label='mode: ' + mode_)
        plt.plot(xvaldyn, aucs_ranked, color='darkred', lw=1, label='mode: ' + mode_ + ' ranked')

    #label_ = 'n_bins= sqrt(samples) {}'.format(auto_bins_static)
    label2_ = 'n_bins= sqrt(samples) ranked ' + mode_
    label3_ = 'n_bins= sqrt(samples) smooth ' + mode_

    if mode_ == "static":
        # plt.scatter(auto_bins_smooth, auto_auc_smooth, color='k', s=100, marker='X', zorder=10)
        # plt.scatter(auto_bins_static, auto_auc_static, color='k', s=100, marker='X', label=label_, zorder=10)
        # plt.scatter(auto_bins_ranked, auto_auc_ranked, color='k', s=100, marker='X', zorder=10)
        plt.xlabel('Number of Bins')
    if mode_ == "dynamic":
        # plt.scatter(auto_bins_static, auto_auc_static, color='k', s=100, marker='X', label=label_, zorder=10)
        # plt.scatter(auto_bins_ranked, auto_auc_ranked, color='k', s=100, marker='X', zorder=10)
        plt.xlabel('% of all samples per Bin')

    plt.ylabel('Area Under the Curve (AUC)')
    if mode_ == "static":
        plt.title('AUC vs. n_bins: ' + dataname + ', samples: {}'.format(samples) + ', features: {}'.format(
            features) + '\n' + ' max AUC "static": {0:0.4f}'.format(maxaucstatic) + ' at {}'.format(
            maxatstatic) + ' bins \n' + ' max AUC "ranked": {0:0.4f}'.format(maxaucranked) + ' at {}'.format(
            maxatranked) + ' bins \n' + ' max AUC "smoothed": {0:0.4f}'.format(maxaucsmooth) + 'at {}'.format(
            maxatsmooth) + ' bins')
    if mode_ == "dynamic":
        plt.title(
            'AUC vs. values per bin ' + dataname + ', samples: {}'.format(samples) + ', features: {}'.format(
                features) + '\n' + ' max AUC "dynamic": {0:0.4f}'.format(maxaucstatic) + ' at {}'.format(
                maxatstatic) + ' bins \n' + ' max AUC "ranked": {0:0.4f}'.format(maxaucranked) + ' at {}'.format(
                maxatranked) + ' bins \n')

    plt.legend(loc="lower right")
    plt.ylim(0, 1.1)
    plt.grid(True)
    # plt.text(0, -0.1, 't_static: {0:0.2f}'.format(duration)+ ' s', fontsize=12, color='black', ha='left',transform=plt.gca().transAxes)
    # pfad=r'C:\Users\david\Desktop\datasets_hbos\ODDS\Dataset\benchmarks'
    # pfad = r'C:\Users\david\Desktop\datasets_hbos\Harvard Dataverse\benchmarks'
    # pfad = r'C:\Users\david\Desktop\datasets_hbos\ELKI\benchmarks'
    # pfad = r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\benchmarks'
    pfad = r'C:\Users\david\Desktop\datasets_hbos\Benchmarks'
    filename = f"{pfad}\{dataname}_{mode_}.png"
    plt.savefig(filename)
    plt.show()

    return maxatstatic


def calc_auc_graph(data_, label_, count, dataname):
    print(dataname, ": current dataset")
    aucs_static = []
    maxatstatic = 0
    maxaucstatic = 0
    n_bins = "sqrt"
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
        hbos_static = data_.copy()
        hbos_static['scores'] = scoresstatic
        hbos_static['Class'] = label_
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
        hbos_dynamic = data_.copy()
        hbos_dynamic['scores'] = scoresdynamic
        hbos_dynamic['Class'] = label_
        hbos_dynamic_sorted = hbos_dynamic.sort_values(by=['scores'], ascending=False)
        fpr2, tpr2, thresholds2 = metrics.roc_curve(hbos_dynamic_sorted['Class'], hbos_dynamic_sorted['scores'])
        aucdynamic = metrics.auc(fpr2, tpr2)
        if aucdynamic > maxaucdynamic:
            maxaucdynamic = aucdynamic
            maxatdynamic = bins
        aucs_dynamic.append(aucdynamic)
    end_time = time.time()
    print("Time taken to run: ", dataname, end_time - start_time, "seconds.")

    auto_auc_static, auto_bins_static = calc_roc_auc2(data_, label_, "static", hbosranked, False, "sqrt")
    auto_auc_dynamic, auto_bins_dynamic = calc_roc_auc2(data_, label_, "dynamic", hbosranked, False, "sqrt")

    # xval = range(1, count + 1)
    plt.figure(figsize=[8, 6])
    # plt.plot(xval, aucs_static, color='b', lw=2, label='mode: ' + hbosmode + ', ranked: {}'.format(hbosranked))

    # plt.plot(xval, uniquevaldynamic, color='c', lw=1, label="n_bins= unique dynamic)")
    plt.plot(xval, aucs_static, color='b', lw=1, label='mode: static')
    # plt.plot(xval, uniquevalstatic, color='c', lw=1, label="n_bins= unique static: "+ '{0:0.4f}'.format(unique_auc_static))
    # plt.plot(xval, calcvalstatic, color='c', lw=1,
    #        label="n_bins= " + n_bins + ": " + '{0:0.4f}'.format(calc_auc_static))

    plt.plot(xval, aucs_dynamic, color='r', lw=1, label='mode: dynamic')
    # plt.plot(xval, uniquevaldynamic, color='#EDB120', lw=1, label="n_bins= unique dynamic: "+ '{0:0.4f}'.format(unique_auc_dynamic))
    # plt.plot(xval, calcvaldynamic, color="#EDB120", lw=1,
    #         label="n_bins= " + n_bins + " dynamic: " + '{0:0.4f}'.format(calc_auc_dynamic))

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
    plt.ylim(0.4, 1.1)
    plt.grid(True)
    # plt.text(0, -0.1, 't_static: {0:0.2f}'.format(duration)+ ' s', fontsize=12, color='black', ha='left',transform=plt.gca().transAxes)
    # pfad=r'C:\Users\david\Desktop\datasets_hbos\ODDS\Dataset\benchmarks\static_dynamic'
    # pfad = r'C:\Users\david\Desktop\datasets_hbos\Harvard Dataverse\benchmarks'
    # pfad= r'C:\Users\david\Desktop\datasets_hbos\ELKI\literature\benchmarks'
    # pfad = r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\benchmarks'
    pfad = r'C:\Users\david\Desktop\datasets_hbos\Benchmarks'
    filename = f"{pfad}\static_dynamic_{dataname}.png"
    plt.savefig(filename)
    plt.show()

    return maxatstatic


def calc_auc_new_and_old(data_, label_, count, dataname):
    print(dataname, ": current dataset")
    aucs_static = []
    maxatstatic = 0
    maxaucstatic = 0
    n_bins = "sqrt"
    aucs_dynamic = []
    maxatdynamic = 0
    maxaucdynamic = 0
    xval = []
    hbosranked = False
    start_time = time.time()
    for i in range(count):
        bins = i + 3
        xval.append(bins)
        clfstatic = HBOSPYOD(n_bins=bins, ranked=hbosranked)
        clfstatic.fit(data_)
        scoresstatic = clfstatic.decision_scores_
        hbos_static = data_.copy()
        hbos_static['scores'] = scoresstatic
        hbos_static['Class'] = label_
        hbos_static_sorted = hbos_static.sort_values(by=['scores'], ascending=False)
        fpr1, tpr1, thresholds1 = metrics.roc_curve(hbos_static_sorted['Class'], hbos_static_sorted['scores'])
        aucstatic = metrics.auc(fpr1, tpr1)
        if aucstatic > maxaucstatic:
            maxaucstatic = aucstatic
            maxatstatic = bins
        aucs_static.append(aucstatic)

        clfdynamic = HBOS(n_bins=bins)
        clfdynamic.fit(data_)
        scoresdynamic = clfdynamic.decision_scores_
        hbos_dynamic = data_.copy()
        hbos_dynamic['scores'] = scoresdynamic
        hbos_dynamic['Class'] = label_
        hbos_dynamic_sorted = hbos_dynamic.sort_values(by=['scores'], ascending=False)
        fpr2, tpr2, thresholds2 = metrics.roc_curve(hbos_dynamic_sorted['Class'], hbos_dynamic_sorted['scores'])
        aucdynamic = metrics.auc(fpr2, tpr2)
        if aucdynamic > maxaucdynamic:
            maxaucdynamic = aucdynamic
            maxatdynamic = bins
        aucs_dynamic.append(aucdynamic)
    end_time = time.time()
    print("Time taken to run: ", dataname, end_time - start_time, "seconds.")

    # xval = range(1, count + 1)
    plt.figure(figsize=[8, 6])
    # plt.plot(xval, aucs_static, color='b', lw=2, label='mode: ' + hbosmode + ', ranked: {}'.format(hbosranked))

    # plt.plot(xval, uniquevaldynamic, color='c', lw=1, label="n_bins= unique dynamic)")
    plt.plot(xval, aucs_static, color='b', lw=1, label='mode: static')

    plt.plot(xval, aucs_dynamic, color='r', lw=1, label='mode: pyod')

    plt.xlabel('Number of Bins')
    plt.ylabel('Area Under the Curve (AUC)')
    plt.title(
        'AUC vs. n_bins: ' + dataname + ' \n' + ' max AUC "static": {0:0.4f}'.format(maxaucstatic) + ' at {}'.format(
            maxatstatic) + ' bins \n' + ' max AUC "dynamic": {0:0.4f}'.format(maxaucdynamic) + ' at {}'.format(
            maxatdynamic) + ' bins')
    plt.legend(loc="lower right")
    plt.ylim(0.4, 1.1)
    plt.grid(True)
    # plt.text(0, -0.1, 't_static: {0:0.2f}'.format(duration)+ ' s', fontsize=12, color='black', ha='left',transform=plt.gca().transAxes)
    # pfad=r'C:\Users\david\Desktop\datasets_hbos\ODDS\Dataset\benchmarks\static_dynamic'
    # pfad = r'C:\Users\david\Desktop\datasets_hbos\Harvard Dataverse\benchmarks'
    # pfad= r'C:\Users\david\Desktop\datasets_hbos\ELKI\literature\benchmarks'
    # pfad = r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\benchmarks'
    pfad = r'C:\Users\david\Desktop\datasets_hbos\Benchmarks\test'
    filename = f"{pfad}\static_dynamic_{dataname}.png"
    plt.savefig(filename)
    plt.show()

    return maxatstatic


def plot_explainability(id_):
    y_values, highest = clf.get_explainability_scores(id_)

    print(highest)
    # Labels erstellen
    labels = ['d {}'.format(i + 1) for i in range(clf.n_features_)]
    colors=[]
    for i in range (len(highest)):
        colors.append(cm.RdYlGn_r(y_values[i] / highest[i]))
    plt.figure(figsize=[10, 8])
    plt.barh(np.arange(len(y_values)), y_values, color=colors, tick_label=labels)

    plt.xlabel('score')

    plt.title(
        'dimension-specific scores for sample: {}'.format(id_) + ' with outlier score = {0:0.4f}'.format(
            clf.decision_scores_[id_]))
    plt.legend(loc="lower right")
    plt.show()


def calc_roc_auc2(data_, labels_, mode_, ranked_, smooth_, n_bins_, test):
    if test ==1:
        clfauc = HBOSPYOD(ranked=ranked_, mode=mode_, adjust=smooth_, n_bins=n_bins_)
    else:
        clfauc = HBOSPYOD(ranked=ranked_, mode=mode_, adjust=smooth_, n_bins=n_bins_)
    clfauc.fit(data_)

    scores = clfauc.decision_scores_
    hbos_orig = data_.copy()
    hbos_orig['Scores'] = scores
    hbos_orig['Class'] = labels_
    hbos_sorted = hbos_orig.sort_values(by=['Scores'], ascending=False)

    fpr, tpr, thresholds = metrics.roc_curve(hbos_sorted['Class'], hbos_sorted['Scores'])
    auc = metrics.auc(fpr, tpr)
    # auc = roc_auc_score(hbos_orig_sorted['Class'], hbos_orig_sorted['scores'])
    patn = average_precision_score(labels_, scores)
    #n_bins_array_ = clfauc.n_bins_array_
    n_bins_array_ = 10
    return auc, clfauc.n_bins, n_bins_array_, patn


def calc_roc_auc(orig_, scores_):
    scores = scores_
    hbos_orig = orig_
    hbos_orig['scores'] = scores
    hbos_orig_sorted = hbos_orig.sort_values(by=['scores'], ascending=False)

    fpr, tpr, thresholds = metrics.roc_curve(hbos_orig_sorted['Class'], hbos_orig_sorted['scores'])
    auc = metrics.auc(fpr, tpr)
    # auc = roc_auc_score(hbos_orig_sorted['Class'], hbos_orig_sorted['scores'])

    plt.figure(figsize=[8, 5])
    plt.plot(fpr, tpr, color='r', lw=2, label='HBOS'.format(testhbosranked))
    plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--', label='guessing')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    print(auc, "auc")
    plt.title('Receiver Operating Characteristic, AUC ={0:0.4f}'.format(auc))
    plt.legend(loc="lower right")
    plt.show()
    return auc


if __name__ == "__main__":
    X_train, y_train = generate_data(n_train=200, n_test=100, n_features=2, contamination=0.1, random_state=42,
                                     train_only=True)
    datasettest = pd.DataFrame(X_train)
    datasettest['Class'] = y_train
    datasettestlabel = y_train
    datasettestorig = datasettest.copy()
    del datasettest['Class']

    mat_data1 = loadmat(r"C:\Users\david\Desktop\datasets_hbos\ODDS\Dataset\annthyroid.mat")
    dataset1 = pd.DataFrame(mat_data1['X'])
    dataset1["Class"] = mat_data1['y']
    dataset1label = mat_data1['y']
    orig1 = dataset1.copy()
    del dataset1['Class']

    mat_data2 = loadmat(r"C:\Users\david\Desktop\datasets_hbos\ODDS\Dataset\cardio.mat")
    dataset2 = pd.DataFrame(mat_data2['X'])
    dataset2["Class"] = mat_data2['y']
    dataset2label = mat_data2['y']
    orig2 = dataset2.copy()
    del dataset2['Class']

    mat_data3 = loadmat(r"C:\Users\david\Desktop\datasets_hbos\ODDS\Dataset\cover.mat")
    dataset3 = pd.DataFrame(mat_data3['X'])
    dataset3["Class"] = mat_data3['y']
    dataset3label = mat_data3['y']
    orig3 = dataset3.copy()
    del dataset3['Class']

    mat_data4 = loadmat(r"C:\Users\david\Desktop\datasets_hbos\ODDS\Dataset\letter.mat")
    dataset4 = pd.DataFrame(mat_data4['X'])
    dataset4["Class"] = mat_data4['y']
    dataset4label = mat_data4['y']
    orig4 = dataset4.copy()
    del dataset4['Class']

    mat_data5 = loadmat(r"C:\Users\david\Desktop\datasets_hbos\ODDS\Dataset\glass.mat")
    dataset5 = pd.DataFrame(mat_data5['X'])
    dataset5["Class"] = mat_data5['y']
    dataset5label = mat_data5['y']
    orig5 = dataset5.copy()
    del dataset5['Class']

    with h5py.File(r"C:\Users\david\Desktop\datasets_hbos\ODDS\Dataset\http.mat", 'r') as file:
        dataset6 = pd.DataFrame(file['X'][:])
        dataset6 = dataset6.transpose()
        labels = file['y'][:]
        labels = labels.transpose()
    dataset6["Class"] = labels
    dataset6label = labels
    orig6 = dataset6.copy()
    del dataset6['Class']

    mat_data7 = loadmat(r"C:\Users\david\Desktop\datasets_hbos\ODDS\Dataset\lympho.mat")
    dataset7 = pd.DataFrame(mat_data7['X'])
    dataset7["Class"] = mat_data7['y']
    dataset7label = mat_data7['y']
    orig7 = dataset7.copy()
    del dataset7['Class']

    mat_data8 = loadmat(r"C:\Users\david\Desktop\datasets_hbos\ODDS\Dataset\mammography.mat")
    dataset8 = pd.DataFrame(mat_data8['X'])
    dataset8["Class"] = mat_data8['y']
    dataset8label = mat_data8['y']
    orig8 = dataset8.copy()
    del dataset8['Class']

    mat_data9 = loadmat(r"C:\Users\david\Desktop\datasets_hbos\ODDS\Dataset\mnist.mat")
    dataset9 = pd.DataFrame(mat_data9['X'])
    dataset9["Class"] = mat_data9['y']
    dataset9label = mat_data9['y']
    orig9 = dataset9.copy()
    del dataset9['Class']

    mat_data10 = loadmat(r"C:\Users\david\Desktop\datasets_hbos\ODDS\Dataset\musk.mat")
    dataset10 = pd.DataFrame(mat_data10['X'])
    dataset10["Class"] = mat_data10['y']
    dataset10label = mat_data10['y']
    orig10 = dataset10.copy()
    del dataset10['Class']

    mat_data11 = loadmat(r"C:\Users\david\Desktop\datasets_hbos\ODDS\Dataset\optdigits.mat")
    dataset11 = pd.DataFrame(mat_data11['X'])
    dataset11["Class"] = mat_data11['y']
    dataset11label = mat_data11['y']
    orig11 = dataset11.copy()
    del dataset11['Class']

    mat_data12 = loadmat(r"C:\Users\david\Desktop\datasets_hbos\ODDS\Dataset\pendigits.mat")
    dataset12 = pd.DataFrame(mat_data12['X'])
    dataset12["Class"] = mat_data12['y']
    dataset12label = mat_data12['y']
    orig12 = dataset12.copy()
    del dataset12['Class']

    dataset13 = pd.read_csv(r"C:\Users\david\Desktop\datasets\creditcard.csv")
    orig13 = dataset13.copy()
    dataset13label = dataset13['Class']
    del dataset13['Time']
    del dataset13['Amount']
    del dataset13['Class']

    mat_data16 = loadmat(r"C:\Users\david\Desktop\datasets_hbos\ODDS\Dataset\satimage-2.mat")
    dataset16 = pd.DataFrame(mat_data16['X'])
    dataset16["Class"] = mat_data16['y']
    dataset16label = mat_data16['y']
    orig16 = dataset16.copy()
    del dataset16['Class']

    mat_data15 = loadmat(r"C:\Users\david\Desktop\datasets_hbos\ODDS\Dataset\vowels.mat")
    dataset15 = pd.DataFrame(mat_data15['X'])
    dataset15["Class"] = mat_data15['y']
    dataset15label = mat_data15['y']
    orig15 = dataset15.copy()
    del dataset15['Class']

    mat_data17 = loadmat(r"C:\Users\david\Desktop\datasets_hbos\ODDS\Dataset\ionosphere.mat")
    dataset17 = pd.DataFrame(mat_data17['X'])
    dataset17["Class"] = mat_data17['y']

    dataset17.to_csv(r"csv.csv")
    dataset17label = mat_data17['y']
    orig17 = dataset17.copy()
    del dataset17['Class']

    annthyroid = arff.loadarff(r'C:\Users\david\Desktop\datasets\semantic\Annthyroid\Annthyroid_02_v01.arff')
    annthyroid_df = pd.DataFrame(annthyroid[0])
    origannthyroid = annthyroid_df.copy()
    del annthyroid_df['Class']
    del annthyroid_df['id']
    origannthyroid['Class'] = origannthyroid['Class'].astype(int)

    dataset14 = pd.read_csv(r'C:\Users\david\Desktop\datasets\breast-cancer-unsupervised-ad.csv')
    orig14 = dataset14.copy()
    dataset14label = dataset14['Class']
    del dataset14['Class']

    harvard1 = pd.read_csv(r'C:\Users\david\Desktop\datasets_hbos\Harvard Dataverse\aloi-unsupervised-ad.csv',
                           header=None)
    lastcol = harvard1.columns[-1]
    harvard1.rename(columns={lastcol: 'Class'}, inplace=True)
    harvard1label = harvard1['Class']
    harvardorig1 = harvard1.copy()
    del harvard1['Class']

    harvard2 = pd.read_csv(r'C:\Users\david\Desktop\datasets_hbos\Harvard Dataverse\annthyroid-unsupervised-ad.csv',
                           header=None)
    lastcol = harvard2.columns[-1]
    harvard2.rename(columns={lastcol: 'Class'}, inplace=True)
    harvard2label = harvard2['Class']
    harvardorig2 = harvard2.copy()
    del harvard2['Class']

    harvard3 = pd.read_csv(r'C:\Users\david\Desktop\datasets_hbos\Harvard Dataverse\breast-cancer-unsupervised-ad.csv',
                           header=None)
    lastcol = harvard3.columns[-1]
    harvard3.rename(columns={lastcol: 'Class'}, inplace=True)
    harvard3label = harvard3['Class']
    harvardorig3 = harvard3.copy()
    del harvard3['Class']

    harvard4 = pd.read_csv(r'C:\Users\david\Desktop\datasets_hbos\Harvard Dataverse\kdd99-unsupervised-ad.csv',
                           header=None)
    lastcol = harvard4.columns[-1]
    harvard4.rename(columns={lastcol: 'Class'}, inplace=True)
    harvard4label = harvard4['Class']
    harvardorig4 = harvard4.copy()
    del harvard4['Class']

    harvard5 = pd.read_csv(r'C:\Users\david\Desktop\datasets_hbos\Harvard Dataverse\letter-unsupervised-ad.csv',
                           header=None)
    lastcol = harvard5.columns[-1]
    harvard5.rename(columns={lastcol: 'Class'}, inplace=True)
    harvard5label = harvard5['Class']
    harvardorig5 = harvard5.copy()
    del harvard5['Class']

    harvard6 = pd.read_csv(r'C:\Users\david\Desktop\datasets_hbos\Harvard Dataverse\pen-global-unsupervised-ad.csv',
                           header=None)
    lastcol = harvard6.columns[-1]
    harvard6.rename(columns={lastcol: 'Class'}, inplace=True)
    harvard6label = harvard6['Class']
    harvardorig6 = harvard6.copy()
    del harvard6['Class']

    harvard7 = pd.read_csv(r'C:\Users\david\Desktop\datasets_hbos\Harvard Dataverse\pen-local-unsupervised-ad.csv',
                           header=None)
    lastcol = harvard7.columns[-1]
    harvard7.rename(columns={lastcol: 'Class'}, inplace=True)
    harvard7label = harvard7['Class']
    harvardorig7 = harvard7.copy()
    del harvard7['Class']

    harvard8 = pd.read_csv(r'C:\Users\david\Desktop\datasets_hbos\Harvard Dataverse\satellite-unsupervised-ad.csv',
                           header=None)
    lastcol = harvard8.columns[-1]
    harvard8.rename(columns={lastcol: 'Class'}, inplace=True)
    harvard8label = harvard8['Class']
    harvardorig8 = harvard8.copy()
    del harvard8['Class']

    harvard9 = pd.read_csv(r'C:\Users\david\Desktop\datasets_hbos\Harvard Dataverse\shuttle-unsupervised-ad.csv',
                           header=None)
    lastcol = harvard9.columns[-1]
    harvard9.rename(columns={lastcol: 'Class'}, inplace=True)
    harvard9label = harvard9['Class']
    harvardorig9 = harvard9.copy()
    del harvard9['Class']

    harvard10 = pd.read_csv(r'C:\Users\david\Desktop\datasets_hbos\Harvard Dataverse\speech-unsupervised-ad.csv',
                            header=None)
    lastcol = harvard10.columns[-1]
    harvard10.rename(columns={lastcol: 'Class'}, inplace=True)
    harvard10label = harvard10['Class']
    harvardorig10 = harvard10.copy()
    del harvard10['Class']

    elki1, meta = arff.loadarff(r'C:\Users\david\Desktop\datasets_hbos\ELKI\literature\ALOI\ALOI.arff')
    elki1 = pd.DataFrame(elki1)
    elkiorig1 = elki1.copy()
    del elki1['id']
    del elki1['Class']
    elkiorig1['Class'] = elkiorig1['Class'].astype(int)
    elki1label = elkiorig1['Class']

    elki2, meta = arff.loadarff(r'C:\Users\david\Desktop\datasets_hbos\ELKI\literature\ALOI\ALOI_withoutdupl.arff')
    elki2 = pd.DataFrame(elki2)
    elkiorig2 = elki2.copy()
    elkiorig2['Class'] = elkiorig2['Class'].astype(int)
    del elki2['id']
    del elki2['Class']
    elki2label = elkiorig2['Class']

    elki3, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\literature\Glass\Glass_withoutdupl_norm.arff')
    elki3 = pd.DataFrame(elki3)
    elkiorig3 = elki3.copy()
    elkiorig3['Class'] = elkiorig3['Class'].astype(int)
    del elki3['id']
    del elki3['Class']
    elki3label = elkiorig3['Class']

    elki4, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\literature\Ionosphere\Ionosphere_withoutdupl_norm.arff')
    elki4 = pd.DataFrame(elki4)
    elkiorig4 = elki4.copy()
    elkiorig4['Class'] = elkiorig4['Class'].astype(int)
    del elki4['id']
    del elki4['Class']
    elki4label = elkiorig4['Class']

    elki5, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\literature\KDDCup99\KDDCup99_catremoved.arff')
    elki5 = pd.DataFrame(elki5)
    elkiorig5 = elki5.copy()
    elkiorig5['Class'] = elkiorig5['Class'].astype(int)
    del elki5['id']
    del elki5['Class']
    elki5label = elkiorig5['Class']

    elki6, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\literature\Lymphography\Lymphography_withoutdupl_catremoved.arff')
    elki6 = pd.DataFrame(elki6)
    elkiorig6 = elki6.copy()
    elkiorig6['Class'] = elkiorig6['Class'].astype(int)
    del elki6['id']
    del elki6['Class']
    elki6label = elkiorig6['Class']

    elki7, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\literature\Lymphography\Lymphography_withoutdupl_norm_1ofn.arff')
    elki7 = pd.DataFrame(elki7)
    elkiorig7 = elki7.copy()
    elkiorig7['Class'] = elkiorig7['Class'].astype(int)
    del elki7['id']
    del elki7['Class']
    elki7label = elkiorig7['Class']

    elki8, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\literature\PenDigits\PenDigits_withoutdupl_norm_v01.arff')
    elki8 = pd.DataFrame(elki8)
    elkiorig8 = elki8.copy()
    elkiorig8['Class'] = elkiorig8['Class'].astype(int)
    del elki8['id']
    del elki8['Class']
    elki8label = elkiorig8['Class']

    elki9, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\literature\Shuttle\Shuttle_withoutdupl_v01.arff')
    elki9 = pd.DataFrame(elki9)
    elkiorig9 = elki9.copy()
    elkiorig9['Class'] = elkiorig9['Class'].astype(int)
    del elki9['id']
    del elki9['Class']
    elki9label = elkiorig9['Class']

    elki10, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\literature\Waveform\Waveform_withoutdupl_v01.arff')
    elki10 = pd.DataFrame(elki10)
    elkiorig10 = elki10.copy()
    elkiorig10['Class'] = elkiorig10['Class'].astype(int)
    del elki10['id']
    del elki10['Class']
    elki10label = elkiorig10['Class']

    elki11, meta = arff.loadarff(r'C:\Users\david\Desktop\datasets_hbos\ELKI\literature\WBC\WBC_v01.arff')
    elki11 = pd.DataFrame(elki11)
    elkiorig11 = elki11.copy()
    elkiorig11['Class'] = elkiorig11['Class'].astype(int)
    del elki11['id']
    del elki11['Class']
    elki11label = elkiorig11['Class']

    elki12, meta = arff.loadarff(r'C:\Users\david\Desktop\datasets_hbos\ELKI\literature\WDBC\WDBC_withoutdupl_v01.arff')
    elki12 = pd.DataFrame(elki12)
    elkiorig12 = elki12.copy()
    elkiorig12['Class'] = elkiorig12['Class'].astype(int)
    del elki12['id']
    del elki12['Class']
    elki12label = elkiorig12['Class']

    elki13, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\literature\WPBC\WPBC_withoutdupl_norm.arff')
    elki13 = pd.DataFrame(elki13)
    elkiorig13 = elki13.copy()
    elkiorig13['Class'] = elkiorig13['Class'].astype(int)
    del elki13['id']
    del elki13['Class']
    elki13label = elkiorig13['Class']

    elki14, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\literature\Shuttle\Shuttle_withoutdupl_norm_v01.arff')
    elki14 = pd.DataFrame(elki14)
    elkiorig14 = elki14.copy()
    elkiorig14['Class'] = elkiorig14['Class'].astype(int)
    del elki14['id']
    del elki14['Class']
    elki14label = elkiorig14['Class']

    elki16, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\literature\WDBC\WDBC_withoutdupl_norm_v01.arff')
    elki16 = pd.DataFrame(elki16)
    elkiorig16 = elki16.copy()
    elkiorig16['Class'] = elkiorig16['Class'].astype(int)
    del elki16['id']
    del elki16['Class']
    elki16label = elkiorig16['Class']

    elki15, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\literature\WBC\WBC_norm_v01.arff')
    elki15 = pd.DataFrame(elki15)
    elkiorig15 = elki15.copy()
    elkiorig15['Class'] = elkiorig15['Class'].astype(int)
    del elki15['id']
    del elki15['Class']
    elki15label = elkiorig15['Class']

    elki_semantic1, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Annthyroid\Annthyroid_02_v01.arff')
    elki_semantic1 = pd.DataFrame(elki_semantic1)
    elki_semanticorig1 = elki_semantic1.copy()
    del elki_semantic1['id']
    del elki_semantic1['Class']
    elki_semanticorig1['Class'] = elki_semanticorig1['Class'].astype(int)
    elki_semantic1label = elki_semanticorig1['Class']

    elki_semantic2, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Arrhythmia\Arrhythmia_withoutdupl_02_v01.arff')
    elki_semantic2 = pd.DataFrame(elki_semantic2)
    elki_semanticorig2 = elki_semantic2.copy()
    elki_semanticorig2['Class'] = elki_semanticorig2['Class'].astype(int)
    del elki_semantic2['id']
    del elki_semantic2['Class']
    elki_semantic2label = elki_semanticorig2['Class']

    elki_semantic3, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Cardiotocography\Cardiotocography_02_v01.arff')
    elki_semantic3 = pd.DataFrame(elki_semantic3)
    elki_semanticorig3 = elki_semantic3.copy()
    elki_semanticorig3['Class'] = elki_semanticorig3['Class'].astype(int)
    del elki_semantic3['id']
    del elki_semantic3['Class']
    elki_semantic3label = elki_semanticorig3['Class']

    elki_semantic4, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\HeartDisease\HeartDisease_withoutdupl_02_v01.arff')
    elki_semantic4 = pd.DataFrame(elki_semantic4)
    elki_semanticorig4 = elki_semantic4.copy()
    elki_semanticorig4['Class'] = elki_semanticorig4['Class'].astype(int)
    del elki_semantic4['id']
    del elki_semantic4['Class']
    elki_semantic4label = elki_semanticorig4['Class']

    elki_semantic5, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Hepatitis\Hepatitis_withoutdupl_05_v01.arff')
    elki_semantic5 = pd.DataFrame(elki_semantic5)
    elki_semanticorig5 = elki_semantic5.copy()
    elki_semanticorig5['Class'] = elki_semanticorig5['Class'].astype(int)
    del elki_semantic5['id']
    del elki_semantic5['Class']
    elki_semantic5label = elki_semanticorig5['Class']

    elki_semantic6, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\InternetAds\InternetAds_norm_02_v01.arff')
    elki_semantic6 = pd.DataFrame(elki_semantic6)
    elki_semanticorig6 = elki_semantic6.copy()
    elki_semanticorig6['Class'] = elki_semanticorig6['Class'].astype(int)
    del elki_semantic6['id']
    del elki_semantic6['Class']
    elki_semantic6label = elki_semanticorig6['Class']

    elki_semantic7, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\PageBlocks\PageBlocks_02_v01.arff')
    elki_semantic7 = pd.DataFrame(elki_semantic7)
    elki_semanticorig7 = elki_semantic7.copy()
    elki_semanticorig7['Class'] = elki_semanticorig7['Class'].astype(int)
    del elki_semantic7['id']
    del elki_semantic7['Class']
    elki_semantic7label = elki_semanticorig7['Class']

    elki_semantic8, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Parkinson\Parkinson_withoutdupl_05_v01.arff')
    elki_semantic8 = pd.DataFrame(elki_semantic8)
    elki_semanticorig8 = elki_semantic8.copy()
    elki_semanticorig8['Class'] = elki_semanticorig8['Class'].astype(int)
    del elki_semantic8['id']
    del elki_semantic8['Class']
    elki_semantic8label = elki_semanticorig8['Class']

    elki_semantic9, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Pima\Pima_withoutdupl_02_v01.arff')
    elki_semantic9 = pd.DataFrame(elki_semantic9)
    elki_semanticorig9 = elki_semantic9.copy()
    elki_semanticorig9['Class'] = elki_semanticorig9['Class'].astype(int)
    del elki_semantic9['id']
    del elki_semantic9['Class']
    elki_semantic9label = elki_semanticorig9['Class']

    elki_semantic10, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\SpamBase\SpamBase_02_v01.arff')
    elki_semantic10 = pd.DataFrame(elki_semantic10)
    elki_semanticorig10 = elki_semantic10.copy()
    elki_semanticorig10['Class'] = elki_semanticorig10['Class'].astype(int)
    del elki_semantic10['id']
    del elki_semantic10['Class']
    elki_semantic10label = elki_semanticorig10['Class']

    elki_semantic11, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Stamps\Stamps_withoutdupl_02_v01.arff')
    elki_semantic11 = pd.DataFrame(elki_semantic11)
    elki_semanticorig11 = elki_semantic11.copy()
    elki_semanticorig11['Class'] = elki_semanticorig11['Class'].astype(int)
    del elki_semantic11['id']
    del elki_semantic11['Class']
    elki_semantic11label = elki_semanticorig11['Class']

    elki_semantic12, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Wilt\Wilt_02_v01.arff')
    elki_semantic12 = pd.DataFrame(elki_semantic12)
    elki_semanticorig12 = elki_semantic12.copy()
    elki_semanticorig12['Class'] = elki_semanticorig12['Class'].astype(int)
    del elki_semantic12['id']
    del elki_semantic12['Class']
    elki_semantic12label = elki_semanticorig12['Class']

    annthyroid1, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Annthyroid\Annthyroid_02_v01.arff')
    annthyroid1 = pd.DataFrame(annthyroid1)
    annthyroidorig1 = annthyroid1.copy()
    del annthyroid1['id']
    del annthyroid1['Class']
    annthyroidorig1['Class'] = annthyroidorig1['Class'].astype(int)
    annthyroid1label = annthyroidorig1['Class']

    annthyroid2, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Annthyroid\Annthyroid_05_v01.arff')
    annthyroid2 = pd.DataFrame(annthyroid2)
    annthyroidorig2 = annthyroid2.copy()
    annthyroidorig2['Class'] = annthyroidorig2['Class'].astype(int)
    del annthyroid2['id']
    del annthyroid2['Class']
    annthyroid2label = annthyroidorig2['Class']

    annthyroid3, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Annthyroid\Annthyroid_07.arff')
    annthyroid3 = pd.DataFrame(annthyroid3)
    annthyroidorig3 = annthyroid3.copy()
    annthyroidorig3['Class'] = annthyroidorig3['Class'].astype(int)
    del annthyroid3['id']
    del annthyroid3['Class']
    annthyroid3label = annthyroidorig3['Class']

    annthyroid4, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Annthyroid\Annthyroid_norm_02_v01.arff')
    annthyroid4 = pd.DataFrame(annthyroid4)
    annthyroidorig4 = annthyroid4.copy()
    annthyroidorig4['Class'] = annthyroidorig4['Class'].astype(int)
    del annthyroid4['id']
    del annthyroid4['Class']
    annthyroid4label = annthyroidorig4['Class']

    annthyroid5, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Annthyroid\Annthyroid_norm_05_v01.arff')
    annthyroid5 = pd.DataFrame(annthyroid5)
    annthyroidorig5 = annthyroid5.copy()
    annthyroidorig5['Class'] = annthyroidorig5['Class'].astype(int)
    del annthyroid5['id']
    del annthyroid5['Class']
    annthyroid5label = annthyroidorig5['Class']

    annthyroid6, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Annthyroid\Annthyroid_norm_07.arff')
    annthyroid6 = pd.DataFrame(annthyroid6)
    annthyroidorig6 = annthyroid6.copy()
    annthyroidorig6['Class'] = annthyroidorig6['Class'].astype(int)
    del annthyroid6['id']
    del annthyroid6['Class']
    annthyroid6label = annthyroidorig6['Class']

    annthyroid7, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Annthyroid\Annthyroid_withoutdupl_02_v01.arff')
    annthyroid7 = pd.DataFrame(annthyroid7)
    annthyroidorig7 = annthyroid7.copy()
    annthyroidorig7['Class'] = annthyroidorig7['Class'].astype(int)
    del annthyroid7['id']
    del annthyroid7['Class']
    annthyroid7label = annthyroidorig7['Class']

    annthyroid8, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Annthyroid\Annthyroid_withoutdupl_05_v01.arff')
    annthyroid8 = pd.DataFrame(annthyroid8)
    annthyroidorig8 = annthyroid8.copy()
    annthyroidorig8['Class'] = annthyroidorig8['Class'].astype(int)
    del annthyroid8['id']
    del annthyroid8['Class']
    annthyroid8label = annthyroidorig8['Class']

    annthyroid9, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Annthyroid\Annthyroid_withoutdupl_07.arff')
    annthyroid9 = pd.DataFrame(annthyroid9)
    annthyroidorig9 = annthyroid9.copy()
    annthyroidorig9['Class'] = annthyroidorig9['Class'].astype(int)
    del annthyroid9['id']
    del annthyroid9['Class']
    annthyroid9label = annthyroidorig9['Class']

    annthyroid10, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Annthyroid\Annthyroid_withoutdupl_norm_02_v01.arff')
    annthyroid10 = pd.DataFrame(annthyroid10)
    annthyroidorig10 = annthyroid10.copy()
    annthyroidorig10['Class'] = annthyroidorig10['Class'].astype(int)
    del annthyroid10['id']
    del annthyroid10['Class']
    annthyroid10label = annthyroidorig10['Class']

    annthyroid11, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Annthyroid\Annthyroid_withoutdupl_norm_05_v01.arff')
    annthyroid11 = pd.DataFrame(annthyroid11)
    annthyroidorig11 = annthyroid11.copy()
    annthyroidorig11['Class'] = annthyroidorig11['Class'].astype(int)
    del annthyroid11['id']
    del annthyroid11['Class']
    annthyroid11label = annthyroidorig11['Class']

    annthyroid12, meta = arff.loadarff(
        r'C:\Users\david\Desktop\datasets_hbos\ELKI\semantic\semantic\Annthyroid\Annthyroid_withoutdupl_norm_07.arff')
    annthyroid12 = pd.DataFrame(annthyroid12)
    annthyroidorig12 = annthyroid12.copy()
    annthyroidorig12['Class'] = annthyroidorig12['Class'].astype(int)
    del annthyroid12['id']
    del annthyroid12['Class']
    annthyroid12label = annthyroidorig12['Class']

    cnt = 0

    VTEST = [

        (dataset2, dataset2label, 500, "test"),

    ]

    V1 = [

        (dataset3, dataset3label, 600, "cover"),
        (dataset6, dataset6label, 800, "http"),
        (dataset4, dataset4label, 100, "letter"),
        (dataset5, dataset5label, 100, "glass"),

        (dataset7, dataset7label, 50, "lympho"),
        (dataset8, dataset8label, 800, "mammography"),
        (dataset9, dataset9label, 250, "mnist"),
        (dataset10, dataset10label, 400, "musk"),
        (dataset11, dataset11label, 100, "optdigits"),
        (dataset12, dataset12label, 250, "pendigits"),
        (dataset14, dataset14label, 100, "breast-cancer-unsupervised-ad")
    ]

    # (harvard1, harvardorig1, 100, "aloi-unsupervised-ad"), 0.5375 max
    #

    V2 = [
        (harvard2, harvard2label, 300, "annthyroid-unsupervised-ad"),
        (harvard3, harvard3label, 100, "breast-cancer-unsupervised-ad2"),
        (harvard4, harvard4label, 100, "kdd99-unsupervised-ad"),
        (harvard5, harvard5label, 100, "letter-unsupervised-ad"),
        (harvard6, harvard6label, 250, "pen-global-unsupervised-ad"),
        (harvard7, harvard7label, 100, "pen-local-unsupervised-ad"),
        (harvard8, harvard8label, 100, "satellite-unsupervised-ad"),
        (harvard9, harvard9label, 100, "shuttle-unsupervised-ad"),
        (harvard10, harvardorig10, 100, "speech-unsupervised-ad")
    ]

    #        (elki6, elki6label, 100, "Lymphography_withoutdupl_catremoved"),
    #        (elki7, elki7label, 100, "Lymphography_withoutdupl_norm_1ofn"),
    #   # (elki2, elkiorig2, 100, "ALOI_withoutdupl"),
    V3 = [
        (elki1, elki1label, 100, "ALOI"),

        (elki3, elki3label, 100, "Glass_withoutdupl_norm"),
        (elki4, elki4label, 100, "Ionosphere_withoutdupl_norm"),
        (elki5, elki5label, 100, "KDDCup99_catremoved"),

        (elki8, elki8label, 100, "PenDigits_withoutdupl_norm_v01"),
        (elki9, elki9label, 100, "Shuttle_withoutdupl_v01"),
        (elki10, elki10label, 100, "Waveform_withoutdupl_v01"),
        (elki11, elki11label, 100, "WBC_v01"),
        (elki12, elki12label, 100, "WDBC_withoutdupl_v01"),
        (elki13, elki13label, 100, "WPBC_withoutdupl_norm"),

        (elki_semantic1, elki_semantic1label, 350, "Annthyroid_02_v01"),
        (elki_semantic2, elki_semantic2label, 250, "Arrhythmia_withoutdupl_02_v01"),
        (elki_semantic3, elki_semantic3label, 500, "Cardiotocography_02_v01"),
        (elki_semantic4, elki_semantic4label, 100, "HeartDisease_withoutdupl_02_v01"),
        (elki_semantic5, elki_semantic5label, 100, "Hepatitis_withoutdupl_05_v01"),
        (elki_semantic6, elki_semanticorig6, 500, "InternetAds_norm_02_v01"),
        (elki_semantic7, elki_semantic7label, 100, "PageBlocks_02_v01"),
        (elki_semantic8, elki_semantic8label, 100, "Parkinson_withoutdupl_05_v01"),
        (elki_semantic9, elki_semantic9label, 100, "Pima_withoutdupl_02_v01"),
        (elki_semantic10, elki_semantic10label, 100, "SpamBase_02_v01"),
        (elki_semantic11, elki_semantic11label, 100, "Stamps_withoutdupl_02_v01")
    ]

    #

    VAnnthyroid = [
        (annthyroid1, annthyroid1label, 100, "Annthyroid_02_v01"),
        (annthyroid2, annthyroid2label, 1500, "Annthyroid_05_v01"),
        (annthyroid3, annthyroid3label, 100, "Annthyroid_07"),
        (annthyroid4, annthyroid4label, 100, "Annthyroid_norm_02_v01"),
        (annthyroid5, annthyroid5label, 100, "Annthyroid_norm_05_v01"),
        (annthyroid6, annthyroid6label, 100, "Annthyroid_norm_07"),
        (annthyroid7, annthyroid7label, 100, "Annthyroid_withoutdupl_02_v01"),
        (annthyroid8, annthyroid8label, 100, "Annthyroid_withoutdupl_05_v01"),
        (annthyroid9, annthyroid9label, 100, "Annthyroid_withoutdupl_07"),
        (annthyroid10, annthyroid10label, 100, "Annthyroid_withoutdupl_norm_02_v01"),
        (annthyroid11, annthyroid11label, 100, "Annthyroid_withoutdupl_norm_05_v01"),
        (annthyroid12, annthyroid12label, 100, "Annthyroid_withoutdupl_norm_07"),
        (harvard2, harvard2label, 300, "annthyroid-unsupervised-ad"),
    ]

    VTEST2 = [
        (datasettest, datasettestlabel, 100, "syntetischer Datensatz")
    ]
    VTESTBINARY = [
        (elki_semantic6, elki_semantic6label, 10, "InternetAds_norm_02_v01"),
    ]

    # (elki5, elki5label, 100, "KDDCup99_catremoved"),
    # (elki_semantic6, elki_semantic6label, 10, "InternetAds_norm_02_v01"),
    # (elki_semantic3, elki_semantic3label, 500, "Cardiotocography_02_v01"),
    # (harvard6, harvard6label, 250, "pen-global-unsupervised-ad"),
    # (elki_semantic7, elki_semantic7label, 100, "PageBlocks_02_v01"),
    # (elki7, elki7label, 100, "Lymphography_withoutdupl_norm_1ofn"),
    # (elki_semantic1, elki_semantic1label, 350, "Annthyroid_02_v01"),
    # (elki_semantic2, elki_semantic2label, 250, "Arrhythmia_withoutdupl_02_v01"),
    # (dataset11, dataset11label, 100, "optdigits"),
    # optdigits
    # (dataset11, dataset11label, 100, "optdigits"),
    #  (harvard5, harvard5label, 100, "letter-unsupervised-ad"),
    # (elki12, elki12label, 100, "WDBC_withoutdupl_v01"),
    # (harvard3, harvard3label, 100, "breast-cancer-unsupervised-ad2"),
    VALL = [
        (dataset3, dataset3label, 600, "cover"),
        (dataset6, dataset6label, 800, "http"),
        (dataset7, dataset7label, 50, "lympho"),
        (dataset8, dataset8label, 800, "mammography"),
        (elki_semantic1, elki_semantic1label, 350, "Annthyroid_02_v01"),
        (dataset10, dataset10label, 400, "musk"),
        (dataset15, dataset15label, 100, "vowel"),

        (harvard4, harvard4label, 100, "kdd99-unsupervised-ad"),
        (harvard8, harvard8label, 100, "satellite-unsupervised-ad"),
        (elki3, elki3label, 100, "Glass_withoutdupl_norm"),
        (elki8, elki8label, 100, "PenDigits_withoutdupl_norm_v01"),
        (elki9, elki9label, 100, "Shuttle_withoutdupl_v01"),
        (elki10, elki10label, 100, "Waveform_withoutdupl_v01"),
        (elki11, elki11label, 100, "WBC_v01"),

        (elki_semantic2, elki_semantic2label, 250, "Arrhythmia_withoutdupl_02_v01"),
        (elki_semantic3, elki_semantic3label, 500, "Cardiotocography_02_v01"),
        (elki_semantic5, elki_semantic5label, 100, "Hepatitis_withoutdupl_05_v01"),
        (elki_semantic6, elki_semantic6label, 500, "InternetAds_norm_02_v01"),
        (elki_semantic7, elki_semantic7label, 100, "PageBlocks_02_v01"),
        (elki_semantic8, elki_semantic8label, 100, "Parkinson_withoutdupl_05_v01"),
        (elki_semantic9, elki_semantic9label, 100, "Pima_withoutdupl_02_v01"),
        (elki_semantic10, elki_semantic10label, 100, "SpamBase_02_v01"),
        (elki_semantic11, elki_semantic11label, 100, "Stamps_withoutdupl_02_v01"),
        (dataset13, dataset13label, 100, "creditcart"),

    ]
    VGesamt = [
        (dataset3, dataset3label, 600, "cover"),
        (dataset6, dataset6label, 800, "http"),
        (dataset7, dataset7label, 50, "lympho"),
        (dataset8, dataset8label, 800, "mammography"),
        (dataset9, dataset9label, 250, "mnist"),
        (dataset10, dataset10label, 400, "musk"),
        (dataset11, dataset11label, 100, "optdigits"),
        (harvard3, harvard3label, 100, "breast-cancer-unsupervised-ad2"),
        (harvard4, harvard4label, 100, "kdd99-unsupervised-ad"),
        (harvard5, harvard5label, 100, "letter-unsupervised-ad"),
        (harvard6, harvard6label, 250, "pen-global-unsupervised-ad"),
        (harvard8, harvard8label, 100, "satellite-unsupervised-ad"),
        (elki3, elki3label, 100, "Glass_withoutdupl_norm"),
        (elki8, elki8label, 100, "PenDigits_withoutdupl_norm_v01"),
        (elki9, elki9label, 100, "Shuttle_withoutdupl_v01"),
        (elki10, elki10label, 100, "Waveform_withoutdupl_v01"),
        (elki11, elki11label, 100, "WBC_v01"),
        (elki12, elki12label, 100, "WDBC_withoutdupl_v01"),
        (elki_semantic2, elki_semantic2label, 250, "Arrhythmia_withoutdupl_02_v01"),
        (elki_semantic3, elki_semantic3label, 500, "Cardiotocography_02_v01"),
        (elki_semantic5, elki_semantic5label, 100, "Hepatitis_withoutdupl_05_v01"),
        (elki_semantic6, elki_semantic6label, 500, "InternetAds_norm_02_v01"),
        (elki_semantic7, elki_semantic7label, 100, "PageBlocks_02_v01"),
        (elki_semantic8, elki_semantic8label, 100, "Parkinson_withoutdupl_05_v01"),
        (elki_semantic9, elki_semantic9label, 100, "Pima_withoutdupl_02_v01"),
        (elki_semantic10, elki_semantic10label, 100, "SpamBase_02_v01"),
        (elki_semantic11, elki_semantic11label, 100, "Stamps_withoutdupl_02_v01"),
        (dataset13, dataset13label, 100, "creditcart"),

    ]

    # (dataset11, dataset11label, 100, "optdigits"),
    # (harvard9, harvard9label, 100, "shuttle-unsupervised-ad"),
    # (elki7, elki7label, 100, "Lymphography_withoutdupl_norm_1ofn"),

    VParameter = [
        (dataset3, dataset3label, 100, "cover"),
        (dataset6, dataset6label, 100, "http"),
        (dataset7, dataset7label, 100, "lympho"),
        (dataset8, dataset8label, 100, "mammography"),
        (dataset10, dataset10label, 100, "musk"),
        (dataset13, dataset13label, 100, "creditcart"),
        (dataset12, dataset12label, 100, "pendigits"),
        (dataset15, dataset15label, 100, "vowel"),
        (harvard3, harvard3label, 100, "breast-cancer-unsupervised-ad"),
        (harvard6, harvard6label, 250, "pen-global-unsupervised-ad"),
        (harvard8, harvard8label, 100, "satellite-unsupervised-ad"),
        (elki3, elki3label, 20, "Glass_withoutdupl_norm"),
        (elki9, elki9label, 100, "Shuttle_withoutdupl_v01"),
        (elki11, elki11label, 100, "WBC_v01"),
        (elki12, elki12label, 100, "WDBC_withoutdupl_v01"),
        (elki_semantic2, elki_semantic2label, 100, "Arrhythmia_withoutdupl_02_v01"),
        (elki_semantic3, elki_semantic3label, 100, "Cardiotocography_02_v01"),
        (elki_semantic5, elki_semantic5label, 100, "Hepatitis_withoutdupl_05_v01"),
        (elki_semantic8, elki_semantic8label, 100, "Parkinson_withoutdupl_05_v01"),
        (elki_semantic9, elki_semantic9label, 100, "Pima_withoutdupl_02_v01"),
        (elki_semantic11, elki_semantic11label, 100, "Stamps_withoutdupl_02_v01"),
        (elki_semantic10, elki_semantic10label, 100, "SpamBase_02_v01"),
        (harvard4, harvard4label, 100, "kdd99-unsupervised-ad")
    ]

    big = [
        (dataset6, dataset6label, 20, "http"),

    ]
    bad = [
        (dataset3, dataset3label, 100, "cover"),
        (dataset6, dataset6label, 80, "http"),
        (elki_semantic7, elki_semantic7label, 100, "PageBlocks_02_v01"),
        (harvard6, harvard6label, 250, "pen-global-unsupervised-ad"),
    ]
    VQ = [
        (dataset3, dataset3label, 600, "cover"),
        (dataset6, dataset6label, 800, "http"),
        (dataset4, dataset4label, 100, "letter"),
        (dataset5, dataset5label, 100, "glass"),
        (dataset7, dataset7label, 50, "lympho"),
        (dataset8, dataset8label, 800, "mammography"),
        (dataset9, dataset9label, 250, "mnist"),
        (dataset10, dataset10label, 400, "musk"),
        (dataset12, dataset12label, 250, "pendigits"),
        (dataset14, dataset14label, 100, "breast-cancer-unsupervised-ad"),
        (harvard4, harvard4label, 100, "kdd99-unsupervised-ad"),
        (harvard5, harvard5label, 100, "letter-unsupervised-ad"),
        (harvard6, harvard6label, 250, "pen-global-unsupervised-ad"),
        (harvard7, harvard7label, 100, "pen-local-unsupervised-ad"),
        (harvard8, harvard8label, 100, "satellite-unsupervised-ad"),
        (elki3, elki3label, 100, "Glass_withoutdupl_norm"),
        (elki4, elki4label, 100, "Ionosphere_withoutdupl_norm"),
        (elki5, elki5label, 100, "KDDCup99_catremoved"),
        (elki8, elki8label, 100, "PenDigits_withoutdupl_norm_v01"),
        (elki9, elki9label, 100, "Shuttle_withoutdupl_v01"),
        (elki10, elki10label, 100, "Waveform_withoutdupl_v01"),
        (elki11, elki11label, 100, "WBC_v01"),
        (elki12, elki12label, 100, "WDBC_withoutdupl_v01"),
        (elki13, elki13label, 100, "WPBC_withoutdupl_norm"),

        (elki_semantic2, elki_semantic2label, 250, "Arrhythmia_withoutdupl_02_v01"),
        (elki_semantic3, elki_semantic3label, 500, "Cardiotocography_02_v01"),
        (elki_semantic4, elki_semantic4label, 100, "HeartDisease_withoutdupl_02_v01"),
        (elki_semantic5, elki_semantic5label, 100, "Hepatitis_withoutdupl_05_v01"),
        (elki_semantic6, elki_semantic6label, 500, "InternetAds_norm_02_v01"),
        (elki_semantic7, elki_semantic7label, 100, "PageBlocks_02_v01"),
        (elki_semantic8, elki_semantic8label, 100, "Parkinson_withoutdupl_05_v01"),
        (elki_semantic9, elki_semantic9label, 100, "Pima_withoutdupl_02_v01"),
        (elki_semantic10, elki_semantic10label, 100, "SpamBase_02_v01"),
        (elki_semantic11, elki_semantic11label, 100, "Stamps_withoutdupl_02_v01"),
        (harvard1, harvard1label, 100, "aloi-unsupervised-ad"),
        (harvard10, harvard10label, 100, "speech-unsupervised-ad")
    ]

    kd99 = [

        (elki_semantic4, elki_semantic4label, 100, "test"),
    ]
    if ( False):
        processes = []

        start_time = time.time()
        for data in VALL:
            p = multiprocessing.Process(target=calc_auc_graph_static_or_dynamic, args=data)
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
        end_time = time.time()
        print("Time taken: ", end_time - start_time)

    #calc_average_dynamic(VALL)
#
    # 620098
    testhbosmode = "dynamic"
    testhbosranked = False


    clf = HBOSPYOD(mode="dynamic",n_bins=101, save_explainability_scores=True, ranked=True)
    #clf.fit(dataset13)
    '''start_time = time.time()
    clf.fit(dataset13)
    end_time = time.time()
    scores = clf.decision_scores_
    print(calc_roc_auc(orig13, scores))
    print("Time taken, static: ", end_time - start_time)


    clf2 = HBOSPYOD(mode="static", save_explainability_scores=False)
    #clf = HBOSOLD()
    start_time = time.time()
    clf2.fit(dataset13)
    end_time = time.time()
    print("Time taken, dynamic: ", end_time - start_time)

    clf3 = HBOSOLD()
    start_time = time.time()
    #clf3.fit_predict(dataset13)
    end_time = time.time()
    print("Time taken, hbos old dynamic: ", end_time - start_time)



    list= 'dynamic binwidth'* dataset13.shape[1]
    clf = HBOSOLD()
    start_time = time.time()
    scores2= clf.fit_predict(dataset13)
    end_time = time.time()
    print(calc_roc_auc(orig13, scores2))
    print("Time taken hbos old, static: ", end_time - start_time)'''




    #calc_roc_auc(orig13)

    # print(clf.labels_)
    # print(calc_roc_auc2(datasettest,datasettestorig,"dynamic",False,False,4))
    # calc_roc_auc(orig_)
    clf = HBOSPYOD(mode="static",adjust=False, n_bins="fd_st", save_explainability_scores=True, ranked=False)
    clf.fit(elki_semantic3)
    plot_explainability(3)

    alldatasets = []
    alltitles = []

    for sets in kd99:
        alldatasets.append(sets[0])
        alltitles.append(sets[3])

    #plot_distributions(alldatasets[0], alltitles[0])
    #plot_boxplot(VALL)
