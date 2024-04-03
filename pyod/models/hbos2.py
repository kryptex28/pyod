import math
import time
from sklearn.preprocessing import LabelEncoder

import numpy as np


class HBOS2:
    def __init__(self, mode="static", adjust=False, save_scores=False):
        # super(HBOS2, self).__init__(contamination=contamination)
        self.save_scores = save_scores
        self.all_scores_per_sample = []
        self.samples_per_bin = "floor"  # ceil / floor
        self.last_bin_merge = True
        self.right_edge = True
        self.sorted_data = None
        self.mode = mode
        self.samples = None
        self.bin_id_list = []
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
        self.is_nominal = None
        self.adjust = adjust

    # X : numpy array of shape (n_samples, n_features)
    def fit(self, X, y=None):
        print("start")
        start_time_total = time.time()
        if len(X.shape) > 1:
            self.features = X.shape[1]
        else:
            self.features = 1

        # Check if any features is nominal if yes encode all nominal features using scikit-learn LabelEncoder
        if np.any(self.is_nominal):
            le = LabelEncoder()
            for i in range(len(self.is_nominal)):
                if self.is_nominal[i]:
                    X[:, i] = le.fit_transform(X[:, i])
            X = X.astype(int)

        self.samples = len(X)
        print(self.features, "features")
        print(self.samples, "samples")
        self.max_values_per_feature = np.max(X, axis=0)
        self.n_bins = round(math.sqrt(self.samples))
        #self.n_bins=10
        self.is_nominal = np.zeros(self.features, dtype=bool)

        # Create histograms for every dimension
        if self.mode == "static":
            print("static mode")
            start_time_fit = time.time()
            self.create_static_histogram(X)
            end_time_fit = time.time()
            start_time_digit = time.time()
            self.digitize(X)
            end_time_digit = time.time()
            start_time_getscores = time.time()

            # calculate raw scores for every bin
            if self.adjust:
                self.get_bin_scores_adjusted()  # (score[-i] + score[i] + score[i+1]) / 3
            else:
                self.get_bin_scores()
            end_time_getscores = time.time()

            # calculate the hbos scores for every i in range(len(data))
            start_time_calc = time.time()
            self.calc_hbos_score()
            end_time_calc = time.time()
            end_time_total = time.time()
            execution_timefit = end_time_fit - start_time_fit
            execution_digit = end_time_digit - start_time_digit
            execution_get_scores = end_time_getscores - start_time_getscores
            execution_calc_scores = end_time_calc - start_time_calc
            print(execution_timefit, "execution time fit", "\n")
            print(execution_digit, "execution time digit", "\n")
            print(execution_get_scores, "execution time get_scores", "\n")
            print(execution_calc_scores, "execution time calc_scores", "\n")
            print(end_time_total - start_time_total, "total", "\n")

        elif self.mode == "dynamic":
            print("dynamic mode")
            start_time_fit = time.time()
            self.create_dynamic_histogram(X)
            end_time_fit = time.time()
            start_time_digit = time.time()
            self.digitize(X)
            end_time_digit = time.time()
            start_time_getscores = time.time()
            self.get_bin_scores()
            end_time_getscores = time.time()
            start_time_calc = time.time()
            self.calc_hbos_score()
            end_time_calc = time.time()
            end_time_total = time.time()
            execution_timefit = end_time_fit - start_time_fit
            execution_digit = end_time_digit - start_time_digit
            execution_get_scores = end_time_getscores - start_time_getscores
            execution_calc_scores = end_time_calc - start_time_calc
            print(execution_timefit, "execution time fit", "\n")
            print(execution_digit, "execution time digit", "\n")
            print(execution_get_scores, "execution time get_scores", "\n")
            print(execution_calc_scores, "execution time calc_scores", "\n")
            print(end_time_total - start_time_total, "total", "\n")

    def digitize(self, X):
        for i in range(self.features):
            if self.features > 1:
                binids = np.digitize(X[:, i], self.bin_edges_list[i], right=False)
                for j in range(len(binids)):
                    if binids[j] > self.n_bins_list[i]:
                        binids[j] = binids[j] - 1
                self.bin_id_list.append(binids)

            else:
                binids = np.digitize(X, self.bin_edges_list[i], right=False)
                for j in range(len(binids)):
                    if binids[j] > self.n_bins_list[i]:
                        binids[j] = binids[j] - 1
                self.bin_id_list.append(binids)

    def create_static_histogram(self, X):
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

    def create_dynamic_histogram(self, X):
        if self.n_bins == "auto":
            self.n_bins = round(math.sqrt(self.samples))
        if self.samples_per_bin == "ceil":
            samples_per_bin = math.ceil(self.samples / self.n_bins)
        elif self.samples_per_bin == "floor":
            samples_per_bin = math.floor(self.samples / self.n_bins)
        else:
            samples_per_bin = self.samples_per_bin

        for i in range(self.features):
            edges = []
            binfirst = []
            binlast = []
            counters = []
            bin_edges = []
            bin_withs = []
            if self.features > 1:
                idataset = X[:, i]
            else:
                idataset = X
            data, anzahl = np.unique(idataset, return_counts=True)
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

            for edge in binfirst:
                bin_edges.append(edge)
            bin_edges.append(binlast[-1])
            if self.last_bin_merge:  # falls letztes bin länge 0, verschmelzen mit bin[-1]
                if binlast[-1] - binfirst[-1] == 0:
                    counters[-2] = counters[-2] + counters[-1]
                    counters = np.delete(counters, -1)
                    bin_edges = np.delete(bin_edges, -2)

            self.n_bins_list.append(len(counters))
            self.histogram_list.append(counters)
            self.bin_edges_list.append(bin_edges)
            if self.right_edge:
                for k in range(len(counters) - 1):
                    bin_with = binfirst[k + 1] - binfirst[k]  # bin start bis neue bin start,
                    if bin_with == 0:  # die rechte bin ist der nächste value außerhalb des bins
                        bin_with = 1
                    bin_withs.append(bin_with)
                binwith = binlast[-1] - binfirst[-1]
                if binwith == 0:
                    binwith = 1
                bin_withs.append(binwith)
            else:
                for m in range(len(binfirst)):
                    bin_with = binlast[m] - binfirst[m]  # genaue bin edges
                    if bin_with == 0:
                        bin_with = 1
                    bin_withs.append(bin_with)
            self.bin_with_list.append(bin_withs)

    def predict(self):
        return self.hbos_scores

    def set_adjust(self, adjust):
        self.adjust = adjust

    def set_save_scores(self, save_scores):
        self.save_scores = save_scores

    def set_is_nominal(self, is_nominal):
        self.is_nominal = is_nominal

    def get_bin_scores(self):
        for i in range(self.features):  # get highest bin
            maxbin = np.amax(self.histogram_list[i])
            self.highest_bin.append(maxbin)

        for i in range(self.features):  # get the highest score
            if self.features == 1:
                max_ = self.max_values_per_feature
            else:
                max_ = self.max_values_per_feature[i]
            if max_ == 0:
                max_ = 1.0

            hist_ = self.histogram_list[i]
            if self.mode == "dynamic":
                dynlist = self.bin_with_list[i]
                binwith = dynlist[0]
            else:
                binwith = self.bin_with_list[i]
            scores_ = []
            max_score = (hist_[0]) / (binwith * self.samples / (abs(max_)))
            scores_.append(max_score)
            for j in range(self.n_bins_list[i] - 1):
                if self.mode == "dynamic":
                    binwith = dynlist[j + 1]
                score = (hist_[j + 1]) / (
                        binwith * self.samples / (abs(max_)))  # Bin Höhe / (Bin Breite / höchste win im Histogramm)
                scores_.append(score)
                if score > max_score:
                    max_score = score
            self.highest_score.append(max_score)
            self.score_list.append(scores_)

    def get_bin_scores_adjusted(self):  # wie in https://dergipark.org.tr/en/download/article-file/2959698
        if self.mode == "dynamic":
            self.get_bin_scores()
        else:
            histogram_list_adjsuted = []
            for i in range(self.features):
                hist = []
                tmphist = self.histogram_list[i]
                tmphist_pad = np.pad(tmphist, (1, 1), 'constant')
                for j in range(len(tmphist_pad) - 2):
                    tmpvalue = (tmphist_pad[j] + tmphist_pad[j + 1] + tmphist_pad[j + 2]) / 3
                    hist.append(tmpvalue)
                histogram_list_adjsuted.append(hist)

            for i in range(self.features):  # get highest bin
                maxbin = np.amax(histogram_list_adjsuted[i])
                self.highest_bin.append(maxbin)

            for i in range(self.features):  # get the highest score
                if self.features == 1:
                    max_ = self.max_values_per_feature
                else:
                    max_ = self.max_values_per_feature[i]
                if max_ == 0:
                    max_ = 1.0
                hist_ = histogram_list_adjsuted[i]
                if self.mode == "dynamic":
                    dynlist = self.bin_with_list[i]
                    binwith = dynlist[0]
                else:
                    binwith = self.bin_with_list[i]
                scores_ = []
                max_score = (hist_[0]) / (binwith * 1 / (abs(max_)))
                scores_.append(max_score)
                for j in range(self.n_bins_list[i] - 1):
                    if self.mode == "dynamic":
                        binwith = dynlist[j]
                    score = (hist_[j + 1]) / (
                            binwith * 1 / (abs(max_)))  # Bin Höhe / (Bin Breite / höchste win im Histogramm)
                    scores_.append(score)
                    if score > max_score:
                        max_score = score
                self.highest_score.append(max_score)
                self.score_list.append(scores_)

    def set_mode(self, mode):
        self.mode = mode

    def calc_hbos_score(self):
        for i in range(self.samples):
            score = 0
            all_scores = []
            for b in range(self.features):
                maxscore = self.highest_score[b]
                scores_b = self.score_list[b]
                idlist = self.bin_id_list[b]
                idtmp = idlist[i]
                tmpscore = scores_b[idtmp - 1]
                tmpscore = tmpscore * 1 / maxscore
                tmpscore = 1 / tmpscore
                score = score + math.log10(tmpscore)
                if self.save_scores:
                    all_scores.append((b + 1, tmpscore))
            if self.save_scores:
                all_scores.sort(key=lambda x: x[1], reverse=True)
                self.all_scores_per_sample.append(all_scores)

            self.hbos_scores.append(score)
