import math

import numpy as np
import pandas as pd

np.set_printoptions(threshold=np.inf)
histogram_list=[]
bin_with_list=[]
bin_edges_list=[]
ids_list=[]
n_bins_list=[]

dataset = pd.read_csv(r"C:\Users\david\Desktop\datasets\creditcard.csv")
orig = dataset.copy()

del dataset['Time']
del dataset['Amount']
del dataset['Class']


dataset= np.array(dataset)
features=dataset.shape[1]

dataset= np.array([1,2,3,3,4,5,6,6,6,6,7,8,9,9,10]) #15

n_bins=5
samples=len(dataset)
features=1
print(features, "features")
#print(hist)
print(samples,"samples")
samples_per_bin = math.ceil(samples / n_bins)
print(samples_per_bin,"samples per bin")
edges=[]
binfirst=[]
binlast=[]
counters=[]
histogram_list2=[]



for i in range(features):
    edges = []
    binfirst = []
    binlast = []
    counters = []
    bin_edges = []
    bin_withs = []
    #idataset = dataset[:, i]
    data, anzahl = np.unique(dataset, return_counts=True)
    counter = 0
    for num, anzahl_ in zip(data, anzahl):
        if counter == 0:
            edges.append(num)
            binfirst.append(num)
            counter = counter + anzahl_
        elif anzahl_ <= samples_per_bin:
            if counter <= samples_per_bin:
                counter = counter + anzahl_
        else:
            if counter == 0:
                binfirst.append(num)
                binlast.append(num)
                edges.append(num)
                edges.append(num)
                counters.append(anzahl_)
            else:
                binlast.append(last)
                edges.append(last)
                counters.append(counter)

                binfirst.append(num)
                binlast.append(num)
                edges.append(num)
                edges.append(num)
                counters.append(anzahl_)

                counter = 0
        if counter >= samples_per_bin:
            binlast.append(num)
            edges.append(num)
            counters.append(counter)
            counter = 0
        elif num == data[-1] and counter != 0:
            binlast.append(num)
            edges.append(num)
            counters.append(counter)
        last = num
    n_bins_list.append(len(binfirst))
    for edge in binfirst:
        bin_edges.append(edge)
    bin_edges.append(binlast[-1])
    histogram_list.append(counters)
    bin_edges_list.append(bin_edges)
    for i in range(len(binfirst)):
        bin_with = binlast[i] - binfirst[i]
        if bin_with ==0:
            bin_with=1
        bin_withs.append(bin_with)
    bin_with_list.append(bin_withs)




print(bin_edges, " edges \n",binfirst," bin first \n",binlast, " bin last \n", counters, " counters \n")
print(n_bins_list,"n_bins_list")
print(histogram_list," histo 1")
for i in range(features):

    #idataset = dataset[:, i]
    iedges= bin_edges_list[i]
    ids=np.digitize(dataset,bins=iedges)
    for j in range(len(ids)):
        if ids[j] > len(counters):
            ids[j] = ids[j] - 1
    ids_list.append(ids)

print(ids_list,"ids")


for j in range(features):
    histo = []
    for k in range(n_bins):
        bin_ = []
        histo.append(bin_)
        print(k)

    for i in range(samples):
        tmpidlist=ids_list[j]
        id=tmpidlist[i]
        histo[id-1].append(dataset[i])
    histogram_list2.append(histo)

print(histogram_list2,"histogram_list2")

#hist, edgess = np.histogram(data, bins=bin_edges)
#ids = np.digitize(data,bins=bin_edges)
#print(ids)

'''alter entwurf

self.sorted_data = np.sort(X)
        for i in range(self.features):
            bin_edges=[]
            count=0
            while j < (len(self.sorted_data)):
                if(len(bin_edges)==0):
                    bin_edges.append(self.sorted_data[0,i])
                    count=count+1
                if count<samples_per_bin:
                    count=count + 1
                elif count==50:
                    tmp=self.sorted_data[j,i]
                    while tmp == self.sorted_data[j+1,i]:
                        j=j+1
                    bin_edges.append(tmp)
                    j=j+1
                    count=0
'''







'''for i in range(1):
    bin_edges = []
    count = 1
    j=1
    bin_edges.append(sorted_data[0])
    while j < (len(sorted_data)):

        if count < samples_per_bin:
            count = count + 1
            j=j+1
            if j == len(sorted_data):
                bin_edges.append(sorted_data[j-1])
        elif count == samples_per_bin:
            tmp = sorted_data[j]
            while j < (len(sorted_data)-1) and tmp == sorted_data[j]:
                j = j + 1
                print(tmp, "tmp")
            bin_edges.append(tmp)
            j=j+1
            count = 0
print (bin_edges)'''

'''samples_per_bin= self.samples/self.n_bins
        for i in range(self.features):
            self.n_bins_list.append(self.n_bins)
            edges = []
            binfirst = []
            binlast = []
            counters = []
            bin_edges = []
            bin_withs=[]
            dataset= X[:, i]
            data, anzahl = np.unique(dataset, return_counts=True)
            counter = 0
            for num, anzahl in zip(data, anzahl):
                if counter == 0:
                    edges.append(num)
                    binfirst.append(num)
                    counter = counter + anzahl
                elif anzahl <= samples_per_bin:
                    if counter <= samples_per_bin:
                        counter = counter + anzahl
                else:
                    if counter == 0:
                        binfirst.append(num)
                        binlast.append(num)
                        edges.append(num)
                        edges.append(num)
                        counters.append(anzahl)
                    else:
                        binlast.append(last)
                        edges.append(last)
                        counters.append(counter)

                        binfirst.append(num)
                        binlast.append(num)
                        edges.append(num)
                        edges.append(num)
                        counters.append(anzahl)

                        counter = 0
                if counter >= samples_per_bin:
                    binlast.append(num)
                    edges.append(num)
                    counters.append(counter)
                    counter = 0
                elif num == data[-1]:
                    binlast.append(num)
                    edges.append(num)
                    counters.append(counter)
                last = num

            self.n_bins_list.append(len(binfirst))
            for edge in binfirst:
                bin_edges.append(edge)
            bin_edges.append(binlast[-1])
            self.histogram_list.append(counters)
            self.bin_edges_list.append(bin_edges)
            for i in range(len(binfirst)):
                bin_with= binlast[i] - binfirst[i]
                bin_withs.append(bin_with)
            self.bin_with_list.append(bin_with)'''