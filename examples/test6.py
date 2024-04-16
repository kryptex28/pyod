import time

import numpy as np
import pandas as pd

from pyod.models.hbospyod import HBOSPYOD
dataset = pd.read_csv(r"C:\Users\david\Desktop\datasets\creditcard.csv")
orig = dataset.copy()
del dataset['Time']
del dataset['Amount']
del dataset['Class']
data = np.array(dataset)
hbos_orig = orig.copy()
hbos_orig2 = orig.copy()
hbos_orig3 = orig.copy()
hbos_orig4 = orig.copy()
hbos_orig5 = orig.copy()
hbos_orig6 = orig.copy()
i=5
valuesdyn=[]
valuesstat=[]

adjust_stat=[]
adjsut_stat_ranked=[]

ranked_stat=[]
ranked_dyn=[]
start_time_fit = time.time()
while i < 1006:
 clf = HBOSPYOD()
 clf2 = HBOSPYOD()
 clf2.set_mode("dynamic")
 clf.set_n_bins(i)
 clf2.set_n_bins(i)
 clf.fit(data)
 clf2.fit(data)
 clf3=HBOSPYOD()
 clf4=HBOSPYOD()
 clf3.set_adjust(True)
 clf4.set_adjust(True)
 clf4.set_ranked(True)
 clf4.set_mode("static")
 clf5=HBOSPYOD()
 clf6=HBOSPYOD()
 clf5.set_ranked(True)
 clf6.set_ranked(True)
 clf6.set_mode("dynamic")
 clf6.set_n_bins(i)
 clf3.set_n_bins(i)
 clf4.set_n_bins(i)
 clf5.set_n_bins(i)
 clf3.fit(data)
 clf4.fit(data)
 clf5.fit(data)
 clf6.fit(data)

 hbos_scores = clf.hbos_scores
 hbos_scores2 = clf2.hbos_scores
 hbos_scores3 = clf3.hbos_scores
 hbos_scores4 = clf4.hbos_scores
 hbos_scores5 = clf5.hbos_scores
 hbos_scores6 = clf6.hbos_scores

 hbos_orig['hbos'] = hbos_scores
 hbos_top1000_data = hbos_orig.sort_values(by=['hbos'], ascending=False)[:1000]
 tupel= (i,len(hbos_top1000_data[lambda x: x['Class'] == 1]))

 valuesstat.append(tupel)

 hbos_orig2['hbos'] = hbos_scores2
 hbos_top1000_data2 = hbos_orig2.sort_values(by=['hbos'], ascending=False)[:1000]
 tupel2= (i,len(hbos_top1000_data2[lambda x: x['Class'] == 1]))

 valuesdyn.append(tupel2)

 hbos_orig3['hbos'] = hbos_scores3
 hbos_top1000_data3 = hbos_orig3.sort_values(by=['hbos'], ascending=False)[:1000]
 tupel3= (i,len(hbos_top1000_data3[lambda x: x['Class'] == 1]))

 adjust_stat.append(tupel3)

 hbos_orig4['hbos'] = hbos_scores4
 hbos_top1000_data4 = hbos_orig4.sort_values(by=['hbos'], ascending=False)[:1000]
 tupel4= (i,len(hbos_top1000_data4[lambda x: x['Class'] == 1]))

 adjsut_stat_ranked.append(tupel4)

 hbos_orig5['hbos'] = hbos_scores5
 hbos_top1000_data5 = hbos_orig5.sort_values(by=['hbos'], ascending=False)[:1000]
 tupel5= (i,len(hbos_top1000_data5[lambda x: x['Class'] == 1]))

 ranked_stat.append(tupel5)

 hbos_orig6['hbos'] = hbos_scores6
 hbos_top1000_data6 = hbos_orig6.sort_values(by=['hbos'], ascending=False)[:1000]
 tupel6= (i,len(hbos_top1000_data6[lambda x: x['Class'] == 1]))

 ranked_dyn.append(tupel6)
 i=i+5


sortiertes_arraystat = sorted(valuesstat, key=lambda x: x[1], reverse=True)
sortiertes_dyn = sorted(valuesdyn, key=lambda x: x[1], reverse=True)
print(sortiertes_arraystat)
print(sortiertes_dyn)

sortiertes_adjust_stat = sorted(adjust_stat, key=lambda x: x[1], reverse=True)
sortiertes_adjsut_stat_ranked = sorted(adjsut_stat_ranked, key=lambda x: x[1], reverse=True)
print(sortiertes_adjust_stat)
print(sortiertes_adjsut_stat_ranked)

sortiertes_ranked_stat = sorted(ranked_stat, key=lambda x: x[1], reverse=True)
sortiertes_ranked_dyn = sorted(ranked_dyn, key=lambda x: x[1], reverse=True)
print(sortiertes_ranked_stat)
print(sortiertes_ranked_dyn)

end_time_fit = time.time()
print( end_time_fit - start_time_fit,": TIME" )

matrix = np.vstack((sortiertes_arraystat, sortiertes_dyn, sortiertes_adjust_stat, sortiertes_adjsut_stat_ranked, sortiertes_ranked_stat, sortiertes_ranked_dyn))

# Speichern der Matrix als CSV-Datei
np.savetxt('matrix.csv', matrix, delimiter=',', fmt='%d')



