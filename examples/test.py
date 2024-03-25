import math
import time

from pandas import DataFrame
from scipy.io import arff
import pandas as pd
import numpy as np
import timeit

# Laden Sie die .arff-Datei
data = arff.loadarff(r'C:\Users\david\Desktop\datasets\literature\ALOI\ALOI_withoutdupl_norm.arff')
data2 = arff.loadarff(r'C:\Users\david\Desktop\datasets\literature\Shuttle\Shuttle_withoutdupl_v10.arff')
# Konvertieren Sie die Daten in ein Pandas DataFrame
df = pd.DataFrame(data[0])
df2= pd.DataFrame(data2[0])
features = 27
features2 =9
selected_columns = df.iloc[:, :features]
selected_columns2 = df2.iloc[:, :features2]
dataset = pd.read_csv(r"C:\Users\david\Desktop\datasets\creditcard.csv")
orig = dataset.copy()
del dataset['Time']
del dataset['Amount']
del dataset['Class']
print(features)
first_20_rows_27_cols = df.iloc[:50, :2]
data = selected_columns.to_numpy()
data2 = selected_columns2.to_numpy()
data = np.array(dataset)

# data=first_20_rows_27_cols.to_numpy()
#data = np.array([[1,2],[2,4],[3,5],[2,5],[1,2],[4,6],[200,160],[1000,1001]])
# data= np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 5, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 10])
data0 = [1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 5, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 10]
data1 = [1, 2, 2, 5, 5, 5, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10]
#data = DataFrame(data={'attr1': data0, 'attr2': data1})
#data = np.array(data)
samples = len(dataset)
n_bins = 10
features=data.shape[1]
print(features)
np.set_printoptions(threshold=np.inf)
histogram_list = []
histogram_list_scores = []
bin_edges_list = []
hbos_scores = []
bin_inDlist = []
hist_ = []
max_values_per_bins = []
max_values_per_feature= np.max(data, axis=0)
bin_with_list = []
highest_bin = []
highest_score = []
score_list = []
n_bins_list = []


def fit():
    for i in range(features):

        if (features > 1):
            hist, bin_edges = np.histogram(data[:, i], bins=n_bins, density=False)
            bin_with = bin_edges[1] - bin_edges[0]
            n_bins_list.append(len(hist))
        else:
            hist, bin_edges = np.histogram(data, bins=n_bins, density=False)
            bin_with = bin_edges[1] - bin_edges[0]
            n_bins_list.append(len(hist))

        histogram_list.append(hist)
        bin_with_list.append(bin_with)
        bin_edges_list.append(bin_edges)


def fit2():
    for i in range(features):

        if (features > 1):
            hist, bin_edges = np.histogram(data[:, i], bins=n_bins, density=True)
            bin_with = bin_edges[1] - bin_edges[0]
            n_bins_list.append(len(hist))
        else:
            hist, bin_edges = np.histogram(data, bins=n_bins, density=True)
            bin_with = bin_edges[1] - bin_edges[0]
            n_bins_list.append(len(hist))

        #    max_bin_height = np.max(hist)
        #   inverted_hist = hist / max_bin_height
        #   histogram_list.append(inverted_hist)
        histogram_list.append(hist)
        bin_with_list.append(bin_with)
        bin_edges_list.append(bin_edges)


def digit():
    for i in range(features):
        if (features > 1):
            binIds = np.digitize(data[:, i], bin_edges_list[i], right=False)
            for j in range(len(binIds)):
                if (binIds[j] > n_bins_list[i]):
                    binIds[j] = binIds[j] - 1
            bin_inDlist.append(binIds)

        else:
            binIds = np.digitize(data, bin_edges_list[i], right=False)
            for j in range(len(binIds)):
                if (binIds[j] > n_bins_list[i]):
                    binIds[j] = binIds[j] - 1
            bin_inDlist.append(binIds)


def get_highest_value_per_bin():  # needed to normalize bin with?

    for i in range(features):
        iDs = bin_inDlist[i]
        max_values_per_iD = {}
        if (features > 1):
            dim = data[:, i]
        else:
            dim = data

        for zahlen, id_ in zip(dim, iDs):
            if id_ not in max_values_per_iD:
                max_values_per_iD[id_] = zahlen
            else:
                if zahlen > max_values_per_iD[id_]:
                    max_values_per_iD[id_] = zahlen

        max_values_per_bins.append(max_values_per_iD)


def get_highest_bin():
    for i in range(features):  # get highest bin
        max = np.amax(histogram_list[i])
        highest_bin.append(max)


def get_scores():
    get_highest_bin()
    for i in range(features):  # get also the highest score
        hist_ = histogram_list[i]
        scores_ = []
        max_score = (hist_[0]) / (bin_with_list[i] / (highest_bin[i]))
        scores_.append(max_score)
        for j in range(n_bins_list[i] - 1):
            score = (hist_[j + 1]) / (bin_with_list[i] / (highest_bin[i]))           # Bin Höhe / (Bin Breite / höchste win im Histogramm)
            scores_.append(score)
            if score > max_score:
                max_score = score
        highest_score.append(max_score)
        score_list.append(scores_)


def calcscore():
    get_scores()
    for i in range(len(data)):
        score = 0
        for b in range(features):                                             # Besser np.sum(allscores[:,i] Siehe pyod 
            iDList = bin_inDlist[b]                                          #Score =  Bin Höhe / (Bin Breite / höchste bin im Histogram)
            iD = iDList[i]
            allscores = score_list[b]                                        #TmpScore = Log (Score / höchsten Score im Histogram) / 1
            maxscore = highest_score[b]
            tmpscore = allscores[iD - 1]
            tmpscore = tmpscore / maxscore
            tmpscore = 1.0 / tmpscore
            score = score + math.log(tmpscore)
        hbos_scores.append(score)


def calcscore2():
    for i in range(len(data)):
        score = 0
        for b in range(features):
            iDList = bin_inDlist[b]
            iD = iDList[i]
            featurehist = histogram_list[b]                                 #Score = the result is the value of the probability density function at the bin,
            tmpscore = featurehist[iD - 1]                                  #normalized such that the integral over the range is 1
            tmpscore = 1.0 / tmpscore                                       #   sum(1/log(score)
            score = score + math.log(tmpscore)
        hbos_scores.append(score)


# fit()
# digit()
# calcscore()

start_time = time.time()
fit()
end_time = time.time()
execution_timefit = end_time - start_time
start_time = time.time()
digit()
end_time = time.time()
execution_timedigit = end_time - start_time
start_time = time.time()
calcscore()
end_time = time.time()

execution_timecalcscore = end_time - start_time
hbos_scores_sorted=np.sort(hbos_scores,axis=None)
#print(hbos_scores)

'''
print(histogram_list, "histogram")
print(bin_edges_list, "edges")
print(highest_bin, "highest_bin")
print(highest_score, "highest_score")
#print(bin_inDlist, "iDList")
print(score_list, "score_list")
'''
# for i in range(2000):
# print(hbos_scores[i], "score", [i], "\n")


#print(n_bins_list)
#print(hbos_scores)
print(len(data))
hbos_orig = orig.copy()
hbos_orig['hbos'] = hbos_scores

hbos_top1000_data = hbos_orig.sort_values(by=['hbos'], ascending=False)[:1000]

# hbos_top1000_data.to_csv('outauto.csv')
hbos_top1000_data[:50]
print(hbos_top1000_data)

print(len(hbos_top1000_data[lambda x: x['Class'] == 1]), "von ", len(hbos_orig[lambda x: x['Class'] == 1]))
print(execution_timefit, "execution time fit", "\n")
print(execution_timedigit, "execution time digit", "\n")
print(execution_timecalcscore, "execution time calcscore", "\n")
print(n_bins_list, "n_bins_list")

get_highest_value_per_bin()
#print( max_values_per_feature, "np.max")
#print(np.array(max_values_per_bins), "mv f")