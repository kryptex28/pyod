import pandas as pd
from matplotlib import pyplot as plt
from scipy.io import loadmat, arff
import numpy as np
import h5py
from pyod.models.hbos2 import HBOS2

data = arff.loadarff(r'C:\Users\david\Desktop\datasets\literature\ALOI\ALOI_withoutdupl.arff')
df2 = pd.DataFrame(data[0])
data = df2.iloc[:, :27]
dataset=np.array(data)
csv_dateipfad = '111111111.csv'

mat_data = loadmat(r"C:\Users\david\Desktop\datasets\satimage-2.mat")
#mat_data = loadmat(r"C:\Users\david\Desktop\datasets\speech.mat")
mat_dataset=mat_data['X']
labels=mat_data['y']
np.savetxt(csv_dateipfad,mat_dataset, delimiter=',')
#mat_data2=h5py.File(r"C:\Users\david\Desktop\datasets\http.mat")
mat_data2=h5py.File(r"C:\Users\david\Desktop\datasets\smtp.mat")


mat_dataset2=mat_data2['X']
print(mat_dataset2.shape,"dataset shape")
labels2=mat_data2['y']
print(labels2.shape, "labels shape")
mat_dataset2 = np.transpose(mat_dataset2)

mat_dataset3=np.copy(mat_dataset2)
test = mat_dataset3[:, 0]

# Pfad zur CSV-Datei
#csv_dateipfad = 'array_data.csv'

# NumPy-Array in CSV speichern
#np.savetxt(csv_dateipfad, test, delimiter=',')

labels2 = np.transpose(labels2)
print(mat_dataset.shape)
print(mat_dataset2.shape)
#print(mat_dataset)
mat_data2.close()
clf = HBOS2()
clf.set_adjust(True)
clf.fit(dataset)


hbos_scores = clf.hbos_scores
hbos_orig=pd.DataFrame(mat_dataset)
hbos_orig["label"]=labels
hbos_orig['hbos'] = hbos_scores
hbos_top1000_data = hbos_orig.sort_values(by=['hbos'], ascending=False)[:400]
hbos_top1000_data[:50]
print(hbos_top1000_data)
print(len(hbos_top1000_data[lambda x: x['label'] == 1])," gefunden", "\n")

print(hbos_top1000_data['label'].cumsum().sum())
plt.scatter(range(400), hbos_top1000_data['label'].cumsum(), marker='1')
plt.xlabel('Top N data')
plt.ylabel('Anomalies found')
plt.show()

clf2=HBOS2()
clf2.set_mode("dynamic")
clf2.fit(mat_dataset)

#liste=np.sort(clf2.hbos_scores)[::-1]
#np.set_printoptions(threshold=np.inf)
#first100=liste[0:1000]
#print(first100,"scores")
hbos_scores2 = clf2.hbos_scores
hbos_orig2=pd.DataFrame(mat_dataset)
hbos_orig2["label"]=labels
hbos_orig2['hbos'] = hbos_scores2
hbos_top1000_data2 = hbos_orig2.sort_values(by=['hbos'], ascending=False)[:400]
hbos_top1000_data2[:50]
print(hbos_top1000_data2)
print(len(hbos_top1000_data2[lambda x: x['label'] == 1])," gefunden")

