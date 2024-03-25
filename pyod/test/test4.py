from numpy import genfromtxt
from scipy.io import arff

from pyod.models.hbos2 import HBOS2

import pandas as pd

import numpy as np

if __name__ == "__main__":
    clf_name = 'HBOS2'
    clf = HBOS2()
    dataset = pd.read_csv(r"C:\Users\david\Desktop\datasets\creditcard.csv")
    #data = arff.loadarff(r'C:\Users\david\Desktop\datasets\literature\ALOI\ALOI_withoutdupl_norm.arff')
    data = arff.loadarff(r'C:\Users\david\Desktop\datasets\literature\ALOI\ALOI_withoutdupl.arff')
    #data2 = arff.loadarff(r'C:\Users\david\Desktop\datasets\literature\Shuttle\Shuttle_withoutdupl_v10.arff')
    data2 = arff.loadarff(r'C:\Users\david\Desktop\datasets\literature\Shuttle\Shuttle_withoutdupl_norm_v10.arff')
    df2 = pd.DataFrame(data2[0])
    selected_columns2 = df2.iloc[:, :9]
    orig = dataset.copy()
    data2 = selected_columns2.to_numpy()
    del dataset['Time']
    del dataset['Amount']
    del dataset['Class']

    df = pd.DataFrame(data[0])
    data = df.iloc[:, :27]
    data= np.array(dataset)
    clf.fit(data)




    hbos_scores=clf.hbos_scores
    hbos_orig = orig.copy()
    hbos_orig['hbos'] = hbos_scores
    hbos_top1000_data = hbos_orig.sort_values(by=['hbos'], ascending=False)[:1000]
    #hbos_top1000_data.to_csv('out2.csv')
    hbos_top1000_data[:50]
    print(hbos_top1000_data)
    print(len(hbos_top1000_data[lambda x: x['Class'] == 1]))