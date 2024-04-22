import time

import numpy as np
import pandas as pd

from pyod.models.hbospyod import HBOSPYOD

clf = HBOSPYOD()
data= np.array([10,10,10,10,10,10,11,12,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,21,22,23,24,25,26,27,28,29,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30])
data=data.reshape(-1,1)
clf.set_mode("dynamic")
clf.fit(data)

er, count = np.unique(data, return_counts=True)
print(er," uniques")
print(count, " count")
print(clf.hist_, " hist")
print(clf.bin_edges_array_, " edges")


