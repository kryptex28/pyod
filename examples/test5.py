import pandas as pd
from scipy.io import loadmat
import numpy as np
import h5py
from pyod.models.hbos2 import HBOS2


mat_data = loadmat(r"C:\Users\david\Desktop\datasets\satimage-2.mat")
mat_dataset=mat_data['X']
labels=mat_data['y']

mat_data2=h5py.File(r"C:\Users\david\Desktop\datasets\http.mat")
mat_dataset2=mat_data2['X']
labels2=mat_data2['y']
mat_dataset2 = np.transpose(mat_dataset2)
print(mat_dataset.shape)
print(mat_dataset2.shape)
#print(mat_dataset)
mat_data2.close()
clf = HBOS2()
clf.fit(mat_dataset2)


hbos_scores = clf.hbos_scores
hbos_orig=pd.DataFrame(mat_dataset2)
hbos_orig["label"]=labels
hbos_orig['hbos'] = hbos_scores
hbos_top1000_data = hbos_orig.sort_values(by=['hbos'], ascending=False)[:2000]
hbos_top1000_data[:50]
print(hbos_top1000_data)
print(len(hbos_top1000_data[lambda x: x['label'] == 1]))

clf2=HBOS2()
clf2.set_mode("dynamic")
clf2.fit(mat_dataset2)

hbos_scores2 = clf2.hbos_scores
hbos_orig2=pd.DataFrame(mat_dataset2)
hbos_orig2["label"]=labels
hbos_orig2['hbos'] = hbos_scores2
hbos_top1000_data2 = hbos_orig2.sort_values(by=['hbos'], ascending=False)[:2000]
hbos_top1000_data2[:50]
print(hbos_top1000_data2)
print(len(hbos_top1000_data2[lambda x: x['label'] == 1]))