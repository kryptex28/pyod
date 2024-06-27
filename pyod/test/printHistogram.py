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

import numpy as np

# Die gegebene Liste
data = [
    [297.09866881], [278.60942996], [254.38540753], [263.60204812], [297.91158503],
    [258.27250972], [257.70891994], [261.3693017], [287.54531988], [271.47021517],
    [293.61669245], [272.50729582], [263.04595233], [284.18585675], [294.6040401],
    [292.23531717], [257.63695283], [260.8123757], [259.58861137], [278.4339106],
    [275.54256278], [261.65559101], [254.99745965], [261.67023106], [261.25043874],
    [272.62344237], [271.72040622], [263.8548642], [297.19208336], [280.85949339],
    [288.02066851], [266.29623447], [291.82094492], [279.23949318], [296.08500649],
    [276.58647375], [294.44458158], [282.04895154], [286.23074773], [257.78198003],
    [271.95591819], [260.38710238], [289.09699088], [260.63353132], [262.7134089],
    [284.26382891], [253.44514574], [292.4615555], [271.57367242], [253.25003864],
    [274.054087], [292.22467046], [257.92040171], [299.72132851], [297.67072673],
    [289.98198041], [279.65089376], [256.11280713], [265.83472144], [264.90969291],
    [274.26477745], [289.82288138], [294.04769021], [259.53207644], [293.52636033],
    [279.741746], [276.00101826], [263.27242337], [289.34522562], [276.6501815],
    [268.27579104], [297.19500583], [266.66949672], [287.91948483], [280.19820557],
    [253.73285374], [281.36532657], [269.15495584], [291.87298834], [298.94613502],
    [276.08281759], [295.39392414], [270.36371223], [272.92322007], [274.14750727],
    [258.24280076], [266.00512535], [268.51092456], [265.88737741], [286.70729041],
    [276.52614979], [288.09528173], [277.86723745], [267.14320888], [253.76309115],
    [275.28000246], [272.08123512], [0.], [1.], [2.]
]

uniform_data = np.random.uniform(250, 300, 97)
uniform_data2 = np.random.uniform(0, 10, 99)

# Adding an extreme point valued

extreme_point = np.array([0, 1, 2])
extreme_point2 = np.array([100])

# Combining the uniform data with the extreme point
np_array = np.append(uniform_data2, extreme_point2)
# In ein NumPy-Array umwandeln
#np_array = np.array(data)
np.set_printoptions(threshold=np.inf)
clf = HBOSPYOD()

clf.set_params(mode="dynamic", ranked=False, n_bins=10)
mixed_data = np_array.reshape(-1, 1)
time1start = time.time()
clf.fit(mixed_data)

bin_edges = np.array(clf.bin_edges_array_[0])
bin_breite = clf.bin_width_array_[0]
heights = np.array(clf.hist_[0])
scores = clf.score_array_[0]
print(heights)
print(clf.decision_scores_)
# bin_positionen = np.cumsum([0] + bin_breite[:-1])
density = heights / bin_breite


plt.bar(bin_edges[:-1], density, width=np.diff(bin_edges), align='edge', edgecolor='black')

plt.title('Histogram with static bin width ')
plt.xlabel('Data')
plt.ylabel('Density')

plt.show()

harvard2 = pd.read_csv(r'C:\Users\david\Desktop\datasets_hbos\Harvard Dataverse\annthyroid-unsupervised-ad.csv',
                       header=None)
lastcol = harvard2.columns[-1]
harvard2.rename(columns={lastcol: 'Class'}, inplace=True)
harvard2label = harvard2['Class']
harvardorig2 = harvard2.copy()
del harvard2['Class']

clf2=HBOSPYOD(mode="dynamic")
clf2.fit(harvard2)
print(sorted(clf.decision_scores_, reverse=True))

