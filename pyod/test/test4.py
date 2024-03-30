import time
from collections import Counter

from numpy import genfromtxt
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from pyod.models.hbos2 import HBOS2

import pandas as pd
from scipy.io import loadmat
import numpy as np

if __name__ == "__main__":
    clf_name = 'HBOS2'
    np.set_printoptions(threshold=np.inf)
    clf = HBOS2()
    clf2=HBOS2()

    dataset = pd.read_csv(r"C:\Users\david\Desktop\datasets\creditcard.csv")
    #data = arff.loadarff(r'C:\Users\david\Desktop\datasets\literature\ALOI\ALOI_withoutdupl_norm.arff')
    data = arff.loadarff(r'C:\Users\david\Desktop\datasets\literature\ALOI\ALOI_withoutdupl.arff')
    data2 = arff.loadarff(r'C:\Users\david\Desktop\datasets\literature\Shuttle\Shuttle_withoutdupl_v10.arff')
    #data2 = arff.loadarff(r'C:\Users\david\Desktop\datasets\literature\Shuttle\Shuttle_withoutdupl_norm_v10.arff')
    df2 = pd.DataFrame(data2[0])
    selected_columns2 = df2.iloc[:, :9]
    df3=pd.DataFrame(data2[0])
    onecolumn = df3.iloc[:, 0]
    orig = dataset.copy()
    data2 = selected_columns2.to_numpy()
    data3= np.array(onecolumn)
    del dataset['Time']
    del dataset['Amount']
    del dataset['Class']

    #df = pd.DataFrame(data[0])
    #data = df.iloc[:, :27]
    data= np.array(dataset)
    #clf.set_adjust(True)
    #clf.set_adjust(True)                                   #Works good with static and low n_bins (10)
    #print(clf.hbos_scores,"clf scores")
    #print(clf2.hbos_scores, "clf2 scores")

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


    clf.fit(data)
    hbos_scores=clf.hbos_scores
    hbos_orig = orig.copy()
    hbos_orig['hbos'] = hbos_scores
    hbos_top1000_data = hbos_orig.sort_values(by=['hbos'], ascending=False)[:1000]

    print(hbos_top1000_data)
    print(len(hbos_top1000_data[lambda x: x['Class'] == 1])," gefunden")



    '''print(hbos_top1000_data['Class'].cumsum().sum())
    plt.scatter(range(1000), hbos_top1000_data['Class'].cumsum(), marker='1')
    plt.xlabel('Top N data')
    plt.ylim(0,500)
    plt.ylabel('Anomalies found')
    plt.show()'''

    clf2.set_mode("dynamic")
    clf2.fit(data)
    hbos_scores2 = clf2.hbos_scores
    hbos_orig2= orig.copy()
    hbos_orig2['hbos'] = hbos_scores2
    hbos_top1000_data2 = hbos_orig2.sort_values(by=['hbos'], ascending=False)[:1000]
    hbos_top1000_data2[:50]
    print(hbos_top1000_data2)
    print(len(hbos_top1000_data2[lambda x: x['Class'] == 1])," gefunden")
    resultscores=clf2.all_scores_per_sample
    print(len(resultscores))
    numer1=0
    for i in range(len(resultscores)):
        numer1= numer1 + resultscores[i][0][0]

    numer1=numer1/len(resultscores)

    erste_werte = [subliste[0][0] for subliste in resultscores]
    zweite_werte = [subliste[1][0] for subliste in resultscores]
    dritte_werte = [subliste[2][0] for subliste in resultscores]
    wert_zähler = Counter(erste_werte)
    wert_zähler2 = Counter(zweite_werte)
    wert_zähler3 = Counter(dritte_werte)
    values= np.array(resultscores)

    print(wert_zähler,"wert zöhler")
    print(wert_zähler2, "wert zöhler2")
    print(wert_zähler3, "wert zöhler3")


    resultscores=clf2.all_scores_per_sample[:50]
    print(resultscores)





