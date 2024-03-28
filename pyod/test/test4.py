from numpy import genfromtxt
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder

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

    df = pd.DataFrame(data[0])
    data = df.iloc[:, :27]
    data= np.array(dataset)
    print(len(data), " samples")
    print(data.shape)

    #clf.set_adjust(True)                                   #Works good with static and low n_bins (10)
    #print(clf.hbos_scores,"clf scores")
    #print(clf2.hbos_scores, "clf2 scores")
    clf.fit(data)
    hbos_scores=clf.hbos_scores
    hbos_orig = orig.copy()
    hbos_orig['hbos'] = hbos_scores
    hbos_top1000_data = hbos_orig.sort_values(by=['hbos'], ascending=False)[:1000]
    hbos_top1000_data[:50]
    print(hbos_top1000_data)
    print(len(hbos_top1000_data[lambda x: x['Class'] == 1])," gefunden")

    clf2.set_mode("dynamic")
    clf2.fit(data)
    hbos_scores2 = clf2.hbos_scores
    hbos_orig2= orig.copy()
    hbos_orig2['hbos'] = hbos_scores2
    hbos_top1000_data2 = hbos_orig2.sort_values(by=['hbos'], ascending=False)[:1000]
    hbos_top1000_data2[:50]
    print(hbos_top1000_data2)
    print(len(hbos_top1000_data2[lambda x: x['Class'] == 1])," gefunden")





