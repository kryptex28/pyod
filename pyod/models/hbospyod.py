import math
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array
import numpy as np
from sklearn.utils.validation import check_is_fitted

from .base import BaseDetector


class HBOSPYOD(BaseDetector):
    def __init__(self, mode="static", n_bins="auto", adjust=True, save_scores=False, log_scale=True, ranked=True,
                  alpha=0.1, tol=0.5, contamination=0.1):
        super(HBOSPYOD, self).__init__(contamination=contamination)

        self.same_score_same_rank = False
        self.ranked = ranked
        self.log_scale = log_scale
        self.adjust = adjust
        self.save_scores = save_scores
        self.mode = mode
        self.n_bins = n_bins
        self.alpha = alpha
        self.tol = tol
        self.samples_per_bin = "floor"  # ceil / floor

        self.histogram_array = []
        self.bin_width_array = []
        self.bin_edges_array = []
        self.bin_id_array = []
        self.max_values_per_feature = []
        self.n_bins_array = []
        self.highest_score_id = []
        self.highest_score = []
        self.score_array = []
        self.hbos_scores = []
        self.all_scores_per_sample = []
        self.all_scores_per_sample_dict = {}
        self.is_nominal = None
        self.features = None
        self.samples = None

    # X : numpy array of shape (n_samples, n_features)
    def fit(self, X, y=None):
        print("start")
        start_time_total = time.time()
        if self.ranked:
            self.log_scale = False

        self._set_n_classes(y)
        self.features = X.shape[1]
        X = check_array(X)

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
        if self.n_bins == "auto":
            self.n_bins = round(math.sqrt(self.samples))

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
            self.normalize_scores()
            self.hbos_scores = self.calc_hbos_scores(self.samples, self.features, self.bin_id_array)
            self.decision_scores_ = self.hbos_scores
            self._process_decision_scores()
            self.is_fitted_ = True


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
            self.normalize_scores()
            self.hbos_scores = self.calc_hbos_scores(self.samples, self.features, self.bin_id_array)
            self.decision_scores_ = self.hbos_scores
            self._process_decision_scores()
            self.is_fitted_ = True

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
            binids = np.digitize(X[:, i], self.bin_edges_array[i], right=False)
            for j in range(len(binids)):
                if binids[j] > self.n_bins_array[i]:
                    binids[j] = binids[j] - 1
            self.bin_id_array.append(binids)

    def create_static_histogram(self, X):
        for i in range(self.features):
            hist, bin_edges = np.histogram(X[:, i], bins=self.n_bins, density=False)
            bin_width = bin_edges[1] - bin_edges[0]
            self.n_bins_array.append(len(hist))

            self.histogram_array.append(hist)
            self.bin_width_array.append(bin_width)
            self.bin_edges_array.append(bin_edges)

    def create_dynamic_histogram(self, X):
        # if self.n_bins == "auto":
        #    self.n_bins = round(math.sqrt(self.samples))
        if self.samples_per_bin == "ceil":
            samples_per_bin = math.ceil(self.samples / self.n_bins)
        elif self.samples_per_bin == "floor":
            samples_per_bin = math.floor(self.samples / self.n_bins)
        else:
            samples_per_bin = self.samples_per_bin
        print(samples_per_bin, "samples per bin")
        for i in range(self.features):

            binfirst = []
            binlast = []
            counters = []
            bin_edges = []
            bin_widths = []
            idataset = X[:, i]
            data, anzahl = np.unique(idataset, return_counts=True)
            counter = 0
            for num, quantity_of_num in zip(data, anzahl):
                if counter == 0:
                    binfirst.append(num)
                    counter = counter + quantity_of_num
                elif quantity_of_num <= samples_per_bin:
                    if counter < samples_per_bin:
                        counter = counter + quantity_of_num
                else:
                    binlast.append(last)
                    counters.append(counter)
                    binfirst.append(num)
                    binlast.append(num)
                    counter = quantity_of_num
                    counters.append(counter)
                    counter = 0

                if counter >= samples_per_bin:
                    binlast.append(num)
                    counters.append(counter)
                    counter = 0
                elif num == data[-1] and counter != 0:
                    binlast.append(num)
                    counters.append(counter)
                    counter = 0
                last = num

            for edge in binfirst:
                bin_edges.append(edge)
            bin_edges.append(binlast[-1])

            # falls letztes bin länge 0, verschmelzen mit bin[-2], Problem unendlich dichte da breite = 0
            if binlast[-1] - binfirst[-1] == 0:
                counters[-2] = counters[-2] + counters[-1]
                counters = np.delete(counters, -1)
                bin_edges = np.delete(bin_edges, -2)
                binlast = np.delete(binlast, -2)
                binfirst = np.delete(binfirst, -1)

            self.n_bins_array.append(len(counters))
            self.histogram_array.append(counters)
            self.bin_edges_array.append(bin_edges)
            for k in range(len(counters) - 1):
                # # bin start bis neue bin start, Problem da sonst löcher im histogram
                bin_width = binfirst[k + 1] - binfirst[k]
                bin_widths.append(bin_width)
            binwidth = binlast[-1] - binfirst[-1]
            bin_widths.append(binwidth)
            self.bin_width_array.append(bin_widths)

    def set_adjust(self, adjust):
        self.adjust = adjust

    def set_version(self, version):
        self.version = version

    def set_ranked(self,ranked):
        self.ranked = ranked

    def set_save_scores(self, save_scores):
        self.save_scores = save_scores

    def set_is_nominal(self, is_nominal):
        self.is_nominal = is_nominal

    def set_n_bins(self, n_bins):
        self.n_bins = n_bins

    def get_bin_scores(self):

        for i in range(self.features):  # get the highest data value
            max_ = self.max_values_per_feature[i]
            if max_ == 0:
                max_ = 1.0

            hist_ = self.histogram_array[i]
            if self.mode == "dynamic":
                dynlist = self.bin_width_array[i]
                binwidth = dynlist[0]
            else:
                binwidth = self.bin_width_array[i]
            scores_bins = []
            max_score = (hist_[0]) / (binwidth * self.samples / (abs(max_)))
            scores_bins.append(max_score)
            for j in range(self.n_bins_array[i] - 1):
                if self.mode == "dynamic":
                    binwidth = dynlist[j + 1]
                score = (hist_[j + 1]) / (
                        binwidth * self.samples / (abs(max_)))  # Bin Höhe / (Bin Breite / höchste bin im Histogramm)
                scores_bins.append(score)
                if score > max_score:
                    max_score = score

            self.highest_score.append(max_score)
            self.score_array.append(scores_bins)

    def get_bin_scores_adjusted(self):  # wie in https://dergipark.org.tr/en/download/article-file/2959698
        if self.mode == "dynamic":
            self.get_bin_scores()
        else:
            histogram_list_adjsuted = []
            for i in range(self.features):
                hist = []
                tmphist = self.histogram_array[i]
                tmphist_pad = np.pad(tmphist, (1, 1), 'constant')
                for j in range(len(tmphist_pad) - 2):
                    tmpvalue = (tmphist_pad[j] + tmphist_pad[j + 1] + tmphist_pad[j + 2]) / 3
                    hist.append(tmpvalue)
                histogram_list_adjsuted.append(hist)

            for i in range(self.features):  # get the highest score

                max_ = self.max_values_per_feature[i]
                if max_ == 0:
                    max_ = 1.0
                hist_ = histogram_list_adjsuted[i]
                if self.mode == "dynamic":
                    dynlist = self.bin_width_array[i]
                    binwidth = dynlist[0]
                else:
                    binwidth = self.bin_width_array[i]
                scores_ = []
                max_score = (hist_[0]) / (binwidth * 1 / (abs(max_)))
                scores_.append(max_score)
                for j in range(self.n_bins_array[i] - 1):
                    if self.mode == "dynamic":
                        binwidth = dynlist[j]
                    score = (hist_[j + 1]) / (
                            binwidth * 1 / (abs(max_)))  # Bin Höhe / (Bin Breite / höchste win im Histogramm)
                    scores_.append(score)
                    if score > max_score:
                        max_score = score

                self.highest_score.append(max_score)
                self.score_array.append(scores_)

    def normalize_scores(self):
        normalized_scores = []
        for i in range(self.features):
            maxscore = self.highest_score[i]
            scores_i = self.score_array[i]
            for j in range(len(scores_i)):
                if scores_i[j] > 0:
                    tmp = scores_i[j]
                    tmp = tmp * 1 / maxscore
                    tmp = 1 / tmp
                    scores_i[j] = tmp
            # print(len(np.unique(scores_i)), " unique scores")
            normalized_scores.append(scores_i)
        self.score_array = normalized_scores

    def set_mode(self, mode):
        self.mode = mode

    '''def rank_scores(self):  # rank scores same scores different ranks, if bin is empty (score 0) still included
        ranked_scores = []  # im many bins are empty ( static) many early ranks are 0
        for i in range(self.features):
            scores_i = self.score_array[i]
            sorted_indices = np.argsort(
                scores_i)  # gives back the indices in scores_i from small to big  x[first] = smallest
            ranks = np.zeros(len(scores_i), dtype=int)
            current_rank = 1
            for ids in sorted_indices:
                ranks[ids] = current_rank
                current_rank += 1

            ranked_scores.append(ranks)
        self.score_array = ranked_scores'''

    '''def rank_scores2(self):
        ranked_scores = []
        for i in range(self.features):
            scores_i = self.score_array[i]
            sorted_indices = sorted(range(len(scores_i)), key=lambda i: scores_i[i])
            ranks = np.zeros(len(scores_i), dtype=int)
            current_rank = 1
            for idx in sorted_indices:
                ranks[idx] = current_rank
                current_rank += 1
            ranked_scores.append(ranks)
        self.score_array = ranked_scores'''

    def rank_scores_same_score_same_rank(
            self):  # rank scores the same scores get the same rank all 0 are rank 1 ... etc
        ranked_scores = []
        for i in range(self.features):
            scores_i = self.score_array[i]
            sorted_scores = sorted(scores_i)  # scores werden sortiert
            score_rank_dict = {}
            current_rank = 1
            for score in sorted_scores:
                if score not in score_rank_dict:
                    score_rank_dict[score] = current_rank  # wenn score nicht in dict wird dict hinzugefügt
                    current_rank += 1
            new_ranks = np.array([score_rank_dict[score] for score in
                                  scores_i])  # für jeden score in score_i wird im dict geschaut welche value
            ranked_scores.append(new_ranks)
        self.score_array = ranked_scores

    def rank_scores_no_empty_bins(
            self):  # rank scores same scores different ranks, if bin is empty (score 0) we do not include it
        ranked_scores = []
        for i in range(self.features):
            scores_i = self.score_array[i]
            sorted_indices = np.argsort(scores_i)
            # sorted_indices = sorted(range(len(scores_i)), key=lambda i: scores_i[i])
            ranks = np.zeros(len(scores_i), dtype=int)
            counter = 1
            for j in range(len(scores_i)):
                tmpid = sorted_indices[j]
                if scores_i[tmpid] > 0:
                    ranks[tmpid] = counter
                    counter = counter + 1
            ranked_scores.append(ranks)
        self.score_array = ranked_scores

    def set_ranked(self, ranked):
        self.ranked = ranked

    def calc_hbos_scores(self, samples, features, bin_id_array):
        for a in range(features):
            self.highest_score_id.append(np.argmax(self.score_array[a]) + 1)
        hbos_scores = []
        if self.ranked:
            if self.same_score_same_rank:
                self.rank_scores_same_score_same_rank()
            else:
                self.rank_scores_no_empty_bins()
        for i in range(samples):
            if self.log_scale:
                score = 0
            elif self.ranked:
                score = 0
            else:
                score = 1
            scores_per_sample = []
            for b in range(features):
                scores_b = self.score_array[b]
                idlist = bin_id_array[b]
                idtmp = idlist[i]
                tmpscore = scores_b[idtmp - 1]

                # if tmpscore is 0, the bin is empty, maximally abnormal, the value gets the highest score of the
                # most abnormal bin in the hist for this feature
                if tmpscore == 0:
                    biggestid = self.highest_score_id[b]
                    tmpscore = scores_b[biggestid - 1]

                if self.ranked:
                    score = score + tmpscore
                elif self.log_scale:
                    tmpscore = math.log10(tmpscore)
                    score = score + tmpscore
                else:
                    score = score * tmpscore
                if self.save_scores:
                    scores_per_sample.append(tmpscore)
            if self.save_scores:
                self.all_scores_per_sample_dict[i] = scores_per_sample
            hbos_scores.append(score)
        return np.array(hbos_scores)

    '''def calc_hbos_score_with_normalize(self):
        if self.ranked:
            self.rank_scores_no_empty_bins()
        for i in range(self.samples):
            if self.log_scale:
                score = 0
            elif self.ranked:
                score = 0
            else:
                score = 1
            all_scores = []
            for b in range(self.features):
                maxscore = self.highest_score[b]
                scores_b = self.score_array[b]
                idlist = self.bin_id_array[b]
                idtmp = idlist[i]
                tmpscore = scores_b[idtmp - 1]
                tmpscore = tmpscore * 1 / maxscore
                tmpscore = 1 / tmpscore
                if self.log_scale:
                    score = score + math.log10(tmpscore)
                elif self.ranked:
                    score = score + tmpscore
                else:
                    score = score * tmpscore
                if self.save_scores:
                    all_scores.append((b + 1, tmpscore))
            if self.save_scores:
                all_scores.sort(key=lambda x: x[1], reverse=True)
                self.all_scores_per_sample.append(all_scores)

            self.hbos_scores.append(score)'''

    def decision_function(self, X):
        check_is_fitted(self, ['is_fitted_'])
        X = check_array(X)
        samples = len(X)
        features = X.shape[1]

        bin_id_array = []
        for i in range(self.features):
            if self.mode == "dynamic":
                dynlist = self.bin_width_array[i]
                bin_width_left = dynlist[0]
                bin_width_right = dynlist[-1]
            else:
                bin_width_left = self.bin_width_array[i]
                bin_width_right = self.bin_width_array[i]
            edges = self.bin_edges_array[i]
            leftedge = edges[0]
            rightedge = edges[-1]
            binids = np.digitize(X[:, i], self.bin_edges_array[i], right=False)
            for j in range(len(binids)):

                # if the sample does not belong to any bin
                # binids[j] == 0: it is too small for every bin
                if binids[j] == 0:
                    distance = leftedge - X[j, i]

                    # If it is only slightly lower than the smallest bin
                    # assign it the id of bin 1
                    if distance <= bin_width_left * self.tol:
                        binids[j] = 1
                    # else assign it the ID of the bin with the highest score
                    else:
                        binids[j] = self.highest_score_id[i]
                # if the sample does not belong to any bin
                # binids[j] is bigger than number of bins in that histogram: it is too big for every bin
                elif binids[j] > self.n_bins_array[i]:
                    distance = X[j, i] - rightedge

                    # if it is only slightly bigger than the most right bin
                    # assign it to the most right bin
                    if distance <= bin_width_right * self.tol:
                        binids[j] = binids[j] - 1

                    # else assign it the ID of the bin with the highest score
                    else:
                        binids[j] = self.highest_score_id[i]
            bin_id_array.append(binids)
        hbos_scores = self.calc_hbos_scores(samples, features, bin_id_array)

        return hbos_scores
