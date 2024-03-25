import math

import numpy as np

#from .base import BaseDetector


class HBOS2():
    def __init__(self):
        #super(HBOS2, self).__init__(contamination=contamination)

        self.samples = None
        self.bin_inDlist = []
        self.features = None
        self.max_values_per_feature = []
        self.n_bins = None
        self.n_bins_list = []
        self.highest_bin = []
        self.histogram_list = []
        self.bin_with_list = []
        self.bin_edges_list = []
        self.highest_score = []
        self.score_list = []
        self.hbos_scores = []
    # X : numpy array of shape (n_samples, n_features)
    def fit(self, X, y=None):
        self.max_values_per_feature = np.max(X, axis=0)
        self.samples=len(X)
        self.n_bins= round(math.sqrt(self.samples))
        if len(X.shape) > 1:
            self.features = features = X.shape[1]
        else:
            self.features = 1
        # Create histograms
        for i in range(self.features):
            if self.features > 1:
                hist, bin_edges = np.histogram(X[:, i], bins=self.n_bins, density=False)
                bin_with = bin_edges[1] - bin_edges[0]
                self.n_bins_list.append(len(hist))
            else:
                hist, bin_edges = np.histogram(X, bins=self.n_bins, density=False)
                bin_with = bin_edges[1] - bin_edges[0]
                self.n_bins_list.append(len(hist))

            self.histogram_list.append(hist)
            self.bin_with_list.append(bin_with)
            self.bin_edges_list.append(bin_edges)

        # Digitize()
        for i in range(self.features):
            if self.features > 1:
                binIds = np.digitize(X[:, i], self.bin_edges_list[i], right=False)
                for j in range(len(binIds)):
                    if binIds[j] > self.n_bins_list[i]:
                        binIds[j] = binIds[j] - 1
                self.bin_inDlist.append(binIds)

            else:
                binIds = np.digitize(X, self.bin_edges_list[i], right=False)
                for j in range(len(binIds)):
                    if (binIds[j] > self.n_bins_list[i]):
                        binIds[j] = binIds[j] - 1
                self.bin_inDlist.append(binIds)
        #get raw scores
        self.get_scores()
        #calc hbos scores for every i in range(len(data))
        self.calc_score(X)
    def get_scores(self):
        for i in range(self.features):  # get highest bin
            max = np.amax(self.histogram_list[i])
            self.highest_bin.append(max)

        for i in range(self.features):  # get also the highest score
            if self.features == 1:
                max_ = self.max_values_per_feature
            else:
                max_ = self.max_values_per_feature[i]
            if max_ == 0:
                max_ = 1.0
            hist_ = self.histogram_list[i]
            scores_ = []
            max_score = (hist_[0]) / (self.bin_with_list[i] * 1 / (abs(max_)))
            scores_.append(max_score)
            for j in range(self.n_bins_list[i] - 1):  # Wie in Java
                score = (hist_[j + 1]) / (
                        self.bin_with_list[i] * 1 / (abs(max_)))  # Bin Höhe / (Bin Breite / höchste win im Histogramm)
                scores_.append(score)
                if score > max_score:
                    max_score = score
            self.highest_score.append(max_score)
            self.score_list.append(scores_)

    def calc_score(self,X):
        for i in range(len(X)):

            score = 0
            for b in range(self.features):
                maxscore = self.highest_score[b]
                scores_b = self.score_list[b]
                iDList = self.bin_inDlist[b]
                iD = iDList[i]
                tmpscore = scores_b[iD - 1]
                tmpscore = tmpscore * 1 / maxscore
                tmpscore = 1 / tmpscore
                score = score + math.log10(tmpscore)
            self.hbos_scores.append(score)