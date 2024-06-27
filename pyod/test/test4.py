import time
from collections import Counter

from numpy import genfromtxt
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from pyod.models.hbospyod import HBOSPYOD
from pyod.models.hbos import HBOS

import pandas as pd
from scipy.io import loadmat
import numpy as np

if __name__ == "__main__":
    clf_name = 'HBOS2'
    np.set_printoptions(threshold=np.inf)
    clf = HBOSPYOD()
    clf2 = HBOSPYOD()

    mean = 200  # Mittelwert
    std_dev = 10  # Standardabweichung
    num_samples = 97  # Anzahl der zu generierenden Datenpunkte

    # Generieren der Daten
    data = np.random.normal(mean, std_dev, num_samples)

    # Generating 99 data points from a standard uniform distribution [0,1]
    uniform_data = np.random.uniform(250, 300, 97)
    uniform_data2 = np.random.uniform(0, 10, 99)

    # Adding an extreme point valued

    extreme_point = np.array([0,1,2])
    extreme_point2 = np.array([100, 101])


    # Combining the uniform data with the extreme point
    mixed_data = np.append(uniform_data, extreme_point)

    #print(mixed_data)

    dataset = pd.read_csv(r"C:\Users\david\Desktop\datasets\creditcard.csv")
    # data = arff.loadarff(r'C:\Users\david\Desktop\datasets\literature\ALOI\ALOI_withoutdupl_norm.arff')
    data = arff.loadarff(r'C:\Users\david\Desktop\datasets\literature\ALOI\ALOI_withoutdupl.arff')
    data2 = arff.loadarff(r'C:\Users\david\Desktop\datasets\literature\Shuttle\Shuttle_withoutdupl_v10.arff')
    # data2 = arff.loadarff(r'C:\Users\david\Desktop\datasets\literature\Shuttle\Shuttle_withoutdupl_norm_v10.arff')
    df2 = pd.DataFrame(data2[0])
    selected_columns2 = df2.iloc[:, :9]
    df3 = pd.DataFrame(data2[0])
    onecolumn = df3.iloc[:, 0]
    orig = dataset.copy()
    data2 = selected_columns2.to_numpy()
    data3 = np.array(onecolumn)
    del dataset['Time']
    del dataset['Amount']
    del dataset['Class']
    data = np.array(dataset)
    hbos_orig = orig.copy()

    harvard3 = pd.read_csv(r'C:\Users\david\Desktop\datasets_hbos\Harvard Dataverse\breast-cancer-unsupervised-ad.csv',
                           header=None)
    lastcol = harvard3.columns[-1]
    harvard3.rename(columns={lastcol: 'Class'}, inplace=True)
    harvard3label = harvard3['Class']
    harvardorig3 = harvard3.copy()
    del harvard3['Class']

    '''c=20
    time2=0
    for i in range(c):
        start_time = time.time()
        clf.fit(data)
        end_time = time.time()
        time2 += end_time - start_time
        print(i)
    time2=time2
    print(time2,"time")'''




    # clf.set_adjust(True)
    # clf.set_save_scores(True)
    # firstrow = hbos_orig["V1"]
    # firstrow = np.array(firstrow)
    #clf.set_params(smoothen=False,mode="static", n_bins="combined")
    clf.set_params(mode="dynamic",ranked=False,n_bins=10)
    mixed_data = mixed_data.reshape(-1, 1)
    time1start=time.time()
    clf.fit(mixed_data)
    print(len(mixed_data))
    time1end=time.time()
    print("time 1",time1end-time1start)
    #print(clf.score_array_)

    print(mixed_data)
    bin_edges= np.array(clf.bin_edges_array_[0])
    bin_breite = clf.bin_width_array_[0]
    heights= np.array(clf.hist_[0])
    scores = clf.score_array_[0]
    print(heights)
    print(clf.decision_scores_)
    #bin_positionen = np.cumsum([0] + bin_breite[:-1])
    density = heights / bin_breite
    print(density)
    # Erzeuge das Histogramm
    #plt.bar(bin_positionen, heights, width=bin_breite, align='edge', edgecolor='black', linewidth=1.5)
    plt.bar(bin_edges[:-1], density, width=np.diff(bin_edges), align='edge', edgecolor='black')

    plt.title('Histogram ')
    plt.xlabel('Data')
    plt.ylabel('Frequency')

    plt.show()
    '''hbos_scores = clf.decision_scores_

    hbos_orig['hbos'] = hbos_scores
    hbos_top1000_data = hbos_orig.sort_values(by=['hbos'], ascending=False)[:1000]
    hbos_top1000_data.to_csv('hbos_top1000_data.txt')
    print(hbos_top1000_data)
    print(len(hbos_top1000_data[lambda x: x['Class'] == 1]), " gefunden")'''

    '''print(hbos_top1000_data['Class'].cumsum().sum())
    plt.scatter(range(1000), hbos_top1000_data['Class'].cumsum(), marker='1')
    plt.xlabel('Top N data')
    plt.ylim(0,500)
    plt.ylabel('Anomalies found')
    plt.show()'''

    clf2.set_params(mode="dynamic")
    # clf2.set_save_scores(True)
    time1start=time.time()
    clf2.fit(data)
    time1end=time.time()
    print("time 2",time1end-time1start)
    #rere=clf2.get_explainability_scores(0)
    res = clf2.predict_proba(data)
    # print("TEST TEST TEST,", res)
    #ans = clf.predict(data)
    # print("TEST",ans[:20])
    hbos_scores2 = clf2.decision_scores_
    hbos_orig2 = orig.copy()
    hbos_orig2['hbos'] = hbos_scores2
    hbos_top1000_data2 = hbos_orig2.sort_values(by=['hbos'], ascending=False)[:1000]
    hbos_top1000_data2[:50]
    print(hbos_top1000_data2)
    firstrow = hbos_orig["V1"]
    firstrow = np.array(firstrow)
    #    firstrow.to_csv('firstrow.txt')

    # test.to_csv("testforjava.txt")
    print(len(hbos_top1000_data2[lambda x: x['Class'] == 1]), " gefunden")
    '''resultscores=clf2.all_scores_per_sample

    erste_werte = [subliste[0][0] for subliste in resultscores]
    zweite_werte = [subliste[1][0] for subliste in resultscores]
    dritte_werte = [subliste[2][0] for subliste in resultscores]
    wert_zähler = Counter(erste_werte)
    wert_zähler2 = Counter(zweite_werte)
    wert_zähler3 = Counter(dritte_werte)
    values= np.array(resultscores)

    print("wert zähler1",wert_zähler)
    print("wert zähler2",wert_zähler2)
    print("wert zähler3",wert_zähler3)'''

    '''clf3 = HBOSPYOD()
    clf.set_mode("dynamic")

    mu = 500  # Mittelwert
    sigma = 200  # Standardabweichung
    normal_values = np.random.normal(mu, sigma, 1000)
    scaled_values = normal_values.clip(1, 1000)
    scaled_values = np.array(scaled_values)
    scaled_values = scaled_values.reshape(-1, 1)
    clf3.fit(scaled_values)
    # print(clf3.histogram_array)
    # print(clf3.predict_proba(scaled_values))
    # my_dict= clf2.all_scores_per_sample_dict#
    # print(my_dict[0])'''

