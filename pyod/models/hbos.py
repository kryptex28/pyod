import math
import sys


from sklearn.utils import check_array
import numpy as np
from sklearn.utils.validation import check_is_fitted
from .base import BaseDetector
from ..utils import check_parameter


class HBOS(BaseDetector):
    """The HBOS algorithm calculates the outlier score based on
    univariate histograms and the normalized density of their bins.
    For every feature of the dataset a separate histogram is calculated.

    There are two modes:
    In the static mode all bins of a histogram have the same bin width and
    the value of the bin is the number of samples which fall into the range
    of the bin.
    In the dynamic mode, the number of samples per bin is the same,
    so there are always n samples in each bin with the values being the same
    (with a few minor exceptions) here the important thing is the bin width.
    See :cite:`goldstein2012histogram` for details.

    There are different ways to calculate the number of bins for the static mode:
    - Static number of bins: uses a static number of bins for all features.
    - Square root: n_bins is set to the square root of the number of samples
    - different estimators can be used to estiamte the number of bins for each
    dimension, a list of the different estimators is available at:
    https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html

    For the dynamic mode, the square root option is available and the number of bins
    can be set to a fixed number. "auto" sets the number of bins to 101.

    A ranked option and an adjust option are also available.
    In the ranked option, the histogram bins get sorted by density.
    Then the rank of the bin is its new score, the anomalie score is calculated with
    the new score.
    In the adjust option, the height of the histogram bins is averaged with the height
    of the neighbouring bins. see: https://dergipark.org.tr/en/pub/muglajsci/issue/78485/1252876

    It's also possible to see how much each feature contributed to the anomalie score,
    or which feature is responsible for a very high score. The get_explainability_scores method
    can be used to access the dimension specific scores.

    Parameters
    ----------
    mode: string, optional (default=”static”)
        Decides which mode is used. "static" uses the static mode,
        "dynamic" uses the dynamic mode

    n_bins : int or string, optional (default="auto)
        The number of bins.

    adjust : bool, optional (default=False)
        If adjust is True, instead of the bin score (number of samples which would fall into
        a bin), the average of the bin score and the bin score of
        the neighbour bins is taken to calculate the density.
        See get_bin_density_adjusted()

    save_explainability_scores : bool, optional (default=False)
        Decides for each sample, whether to save the explainability
        scores or not. "True" saves an array with the scores of the bins
        in which the sample belongs in every histogram. This can be used
        to see what impact each features has on the outlier score for each sample.

    log_scale : bool, optional (default=True)
        Use "True" for a logarithmic scaled score. (better precision)
        Only available if ranked mode is not used.

    ranked : bool, optional (default=False)
        Decides if the ranked mode is used.
        The bins get ranked according to their density score and the rank of
        the bin is used as its new score.

    same_score_same_rank: bool, optional (default=False)
        "True" uses a different ranking method for the ranked mode.
        It will be ensured, that every bin with the same score will
        have the same rank.

    tol : float in (0, 1), optional (default=0.5)
        The parameter to decide the flexibility while dealing
        the samples falling outside the bins.

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set,
        i.e. the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.

    Attributes
    ----------
    bin_edges_array_ : numpy array of shape (n_bins + 1, n_features )
        The edges of the bins.

    hist_ : numpy array of shape (n_bins, n_features)
        The number of samples in each bin

    bin_width_array_: array of shape (n_bins, n_features)
        The bin widths of the bins, needed to calculate the density.

    bin_id_array_ : array of shape (n_features, n_samples)
        Contains ids for all histograms in which bin every sample belongs to.

    max_values_per_feature_ : array of shape (n_features)
        The highest value of all samples for each feature. (needed to normalize bin width)

    n_bins_array_ : array of shape (n_features)
        The number of bins in every feature.

    highest_score_id_ : array of shape (n_features)
        The ids of the bin with the highest normalized density for each feature.

    highest_score_ : array of shape (n_features)
        The highest density for each feature. (needed to normalize densities)

    score_array_ : array of shape (n_bins, n_features)
        The scores of all bins.

    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is fitted.

    explainability_scores_ : array of shape (n_samples, n_features)
        When save_explainability_scores is "True" an array with the scores of the bins
        in which the sample belongs in every histogram is saved.

    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.

    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    def __init__(self, mode="dynamic", n_bins="auto", adjust=False,
                 save_explainability_scores=False, log_scale=True, ranked=False,
                 tol=0.5, contamination=0.1):
        super(HBOS, self).__init__(contamination=contamination)

        self.ranked = ranked
        self.log_scale = log_scale
        self.adjust = adjust
        self.save_explainability_scores = save_explainability_scores
        self.mode = mode
        self.n_bins = n_bins
        self.tol = tol

        check_parameter(tol, 0, 1, param_name='tol')

    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        self.hist_ = []
        self.bin_edges_array_ = []
        self.decision_scores_ = []
        self.bin_width_array_ = []
        self.bin_id_array_ = []
        self.max_values_per_feature_ = []
        self.n_bins_array_ = []
        self.highest_score_id_ = []
        self.highest_score_ = []
        self.score_array_ = []
        self.explainability_scores_ = []

        if self.ranked:
            self.log_scale = False

        # validate inputs X and y (optional)
        X = check_array(X)
        self._set_n_classes(y)

        self.n_features_ = X.shape[1]
        self.n_samples_ = len(X)
        self.max_values_per_feature_ = np.max(X, axis=0)

        if self.mode == "static":
            modes = ["stone","fd", "doane", "scott", "sturges", "auto", "rice","sqrt"]
            if isinstance(self.n_bins, str):
                if self.n_bins == "sqrt":
                    self.n_bins = round(math.sqrt(self.n_samples_))

                elif self.n_bins not in modes:
                    warning = 'Warning: "' + self.n_bins + '" is not a valid option for n_bins. "auto" will be used instead'
                    print(warning, file=sys.stderr)
                    self.n_bins = self.n_bins = "auto"
            # Create histograms for every dimension
            self.create_static_histograms(X)

            # calculate raw density scores for every bin
            if self.adjust:
                self.get_bin_density_adjsuted()
            else:
                self.get_bin_density()

            # normalize density scores and calculate the anomalie scores
            self.normalize_density()

            # get bin ids
            self.get_ids(X)

            self.decision_scores_ = self.calc_hbos_scores(self.n_samples_, self.n_features_, self.bin_id_array_)
            self._process_decision_scores()
            return self

        elif self.mode == "dynamic":
            modes = ["auto","sqrt"]
            if isinstance(self.n_bins, str):
                if self.n_bins == "sqrt":
                    self.n_bins = round(math.sqrt(self.n_samples_))
                elif self.n_bins not in modes:
                    warning = 'Warning: "' + self.n_bins + '" is not a valid option for n_bins. "auto" will be used instead'
                    print(warning, file=sys.stderr)
                    self.n_bins = "auto"

            # Create histograms for every dimension
            self.create_dynamic_histograms(X)

            if self.adjust:
                print("Warning: The adjust option is only supported in static mode and has no impact in dynamic mode.",
                      file=sys.stderr)

            # calculate raw density scores for every bin
            self.get_bin_density()

            # normalize density scores and calculate the anomalie scores
            self.normalize_density()

            # get bin ids
            self.get_ids(X)

            self.decision_scores_ = self.calc_hbos_scores(self.n_samples_, self.n_features_, self.bin_id_array_)
            self._process_decision_scores()

            return self

    def get_ids(self, X):
        """The internal function to calculate the ids for every histogram, in which bin
        each sample belong to.

            Parameters
            ----------
            X : numpy array of shape (n_samples, n_features)
                The input samples.
            """
        for i in range(self.n_features_):
            binids = np.digitize(X[:, i], self.bin_edges_array_[i], right=False)
            for j in range(len(binids)):
                if binids[j] > self.n_bins_array_[i]:
                    binids[j] = binids[j] - 1
            self.bin_id_array_.append(binids)

    def create_static_histograms(self, X):
        """The internal function to create the static histograms.
        Used when mode="static".

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        """
        for i in range(self.n_features_):
            data_ = X[:, i]
            bin_width = []

            n_bins = self.n_bins
            hist, bin_edges = np.histogram(data_, bins=n_bins, density=False)
            for i in range(len(hist)):
                bin_width.append(bin_edges[1] - bin_edges[0])

            self.n_bins_array_.append(len(hist))

            self.hist_.append(hist)
            self.bin_width_array_.append(bin_width)
            self.bin_edges_array_.append(bin_edges)

    def create_dynamic_histograms(self, X):
        """The internal function to create the dynamic histograms.
        Used when mode="dynamic".

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.
        """

        if self.n_bins == "auto":
            n_bins = 101
        else:
            n_bins = self.n_bins

        for i in range(self.n_features_):
            last = None
            binfirst = []
            binlast = []
            counters = []
            bin_edges = []
            bin_widths = []
            idataset = X[:, i]
            iscategorical = False
            unique_values = np.unique(idataset)  # get unique sorted values

            # if there is more than one unique value, check if all the unique values
            # have the same distances to their neighbors
            # -> may be a categorical feature which was label encoded
            if len(unique_values) > 1:
                dist = np.diff(unique_values)  # calculate distances
                iscategorical = np.all(dist == dist[0])  # check if all distances are the same

            samples_per_bin = np.floor(self.n_samples_ / n_bins)

            unique_instances, unique_count = np.unique(idataset, return_counts=True)
            counter = 0
            for num, quantity_of_num in zip(unique_instances, unique_count):
                if counter == 0:
                    binfirst.append(num)
                    counter = counter + quantity_of_num
                elif quantity_of_num <= samples_per_bin:
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
                elif num == unique_instances[-1] and counter != 0:
                    binlast.append(num)
                    counters.append(counter)
                    counter = 0
                last = num

            # if last bin got bin width zero (only one unique value is in that bin),
            # merge it with second last bin otherwise the last bin would have an infinite density
            if binlast[-1] - binfirst[-1] == 0:
                if len(binfirst) > 2:

                    # only features which are not nominal get merged
                    if iscategorical == False:
                        counters[-2] = counters[-2] + counters[-1]
                        counters = np.delete(counters, -1)
                        binlast = np.delete(binlast, -2)
                        binfirst = np.delete(binfirst, -1)

                # if there are only two bins, do not merge but increase the right edge of the
                # second (last) bin so that both bins have the same width
                elif len(binfirst) == 2:
                    binlast[-1] = binlast[-1] + (binfirst[1] - binfirst[0])

                # if there is only one bin (only one unique value in this feature for all samples)
                # do not merge but increase the right edge by one so that the bin with is one
                # otherwise the bin would have bin width zero and therefore infinite density aswell
                elif len(binfirst) == 1:
                    binlast[-1] = binlast[-1] + 1

            # Get bin edges, use the next bins left edge as right edge, otherwise
            # there may be holes in the histogram
            for edge in binfirst:
                bin_edges.append(edge)

            # last bin is an exception, here the biggest value in the bin is the right edge
            bin_edges.append(binlast[-1])

            self.n_bins_array_.append(len(counters))
            self.hist_.append(counters)
            self.bin_edges_array_.append(bin_edges)
            for k in range(len(counters) - 1):
                # calculate bin widths
                bin_width = binfirst[k + 1] - binfirst[k]
                bin_widths.append(bin_width)

            # calculate bin width of last bin
            binwidthlast = binlast[-1] - binfirst[-1]

            # if the last bin width is 0 (is only possible for a categorical feature
            # since the bin would have been merged otherwise)
            # The last bin is given the smallest occurring bin width in the feature.
            if iscategorical:
                if binwidthlast == 0:
                    smallest_width = np.min(bin_widths)
                    binwidthlast = smallest_width

            bin_widths.append(binwidthlast)
            self.bin_width_array_.append(bin_widths)

    def get_bin_density(self):
        """The internal function to calculate the raw density estimation scores
        of each bin in each histogram. (bin height/width)
        A normalized bin width is used.
        Save maximum density score for every feature (needed to normalize densities scores)
        """
        for i in range(self.n_features_):  # get the highest data value
            max_ = self.max_values_per_feature_[i]
            if max_ == 0:
                max_ = 1.0

            hist_ = self.hist_[i]

            dynlist = self.bin_width_array_[i]
            binwidth = dynlist[0]

            scores_bins = []
            max_score = (hist_[0]) / (binwidth * self.n_samples_ / (abs(max_)))
            scores_bins.append(max_score)
            for j in range(self.n_bins_array_[i] - 1):

                binwidth = dynlist[j + 1]
                score = (hist_[j + 1]) / (binwidth * self.n_samples_ / (abs(max_)))
                scores_bins.append(score)
                if score > max_score:
                    max_score = score

            self.highest_score_.append(max_score)
            self.score_array_.append(scores_bins)

    def get_bin_density_adjsuted(self):
        """The internal function to calculate the raw density scores of each bin in each histogram.
        of each bin in each histogram. (bin height/width) A normalized bin width is used.
        Here the average of the bin height and the height of the neighbour bins is taken.
        This version is used when adjust == True.
        Save maximum density score for every feature (needed to normalize density scores)
        see: https://dergipark.org.tr/en/pub/muglajsci/issue/78485/1252876
        """

        histogram_list_adjsuted = []
        for i in range(self.n_features_):
            hist = []
            tmphist = self.hist_[i]
            tmphist_pad = np.pad(tmphist, (1, 1), 'constant')
            if len(self.hist_[i]) > 5:
                for j in range(len(tmphist_pad) - 2):
                    tmpvalue = (tmphist_pad[j] + tmphist_pad[j + 1] + tmphist_pad[j + 2]) / 3
                    hist.append(tmpvalue)
            else:
                hist = tmphist
            histogram_list_adjsuted.append(hist)

        for i in range(self.n_features_):  # get the highest score

            max_ = self.max_values_per_feature_[i]
            if max_ == 0:
                max_ = 1.0
            hist_ = histogram_list_adjsuted[i]
            dynlist = self.bin_width_array_[i]
            binwidth = dynlist[0]

            scores_ = []
            max_score = (hist_[0]) / (binwidth * 1 / (abs(max_)))
            scores_.append(max_score)
            for j in range(self.n_bins_array_[i] - 1):

                binwidth = dynlist[j]

                # bin height / (bin width / biggest sample in the feature)
                score = (hist_[j + 1]) / (binwidth * 1 / (abs(max_)))
                scores_.append(score)
                if score > max_score:
                    max_score = score

            self.highest_score_.append(max_score)
            self.score_array_.append(scores_)

    def normalize_density(self):
        """The internal function to normalize the raw density scores
        of each bin in each histogram. The "most normal bin" is set to 1.
        Every other bin is set relativ to that bin. Additionally, all scores
        get inverted.
        """
        normalized_scores = []
        for i in range(self.n_features_):
            maxscore = self.highest_score_[i]
            scores_i = self.score_array_[i]
            for j in range(len(scores_i)):
                if scores_i[j] > 0:
                    tmp = scores_i[j]
                    tmp = tmp * 1 / maxscore
                    tmp = 1 / tmp
                    scores_i[j] = tmp
            normalized_scores.append(scores_i)
        self.score_array_ = normalized_scores

    def rank_scores(self):
        """The internal function for the ranked mode.
        The bins in each histogram are sorted then ranked,
        Every bin with the same density score gets the same rank.
        """
        ranked_scores = []
        for i in range(self.n_features_):
            scores_i = self.score_array_[i]
            sorted_scores = sorted(scores_i)
            score_rank_dict = {}
            current_rank = 1
            for score in sorted_scores:
                if score not in score_rank_dict:
                    score_rank_dict[score] = current_rank
                    current_rank += 1
            new_ranks = np.array([score_rank_dict[score] for score in scores_i])
            max_score = np.max(new_ranks)

            new_ranks_norm = new_ranks / max_score

            ranked_scores.append(new_ranks_norm)

        self.score_array_ = ranked_scores


    def calc_hbos_scores(self, samples, features, bin_id_array):
        """The internal function to calculate the outlier scores based on
        the bins and histograms constructed with the training data. The program

        Parameters
        ----------
        samples: int
            The number of input samples.

        features: int
            The number of features in the input samples.

        bin_id_array: array of shape (n_features, n_samples)
            Contains ids for all histograms in which bin every sample belongs to.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """

        hbos_scores = np.zeros(samples)
        if self.ranked:
            self.rank_scores()
        for a in range(features):
            self.highest_score_id_.append(np.argmax(self.score_array_[a]) + 1)

        for i in range(samples):
            if self.log_scale:
                score = 0
            elif self.ranked:
                score = 0
            else:
                score = 1
            scores_per_sample = []
            for b in range(features):
                scores_b = self.score_array_[b]
                idlist = bin_id_array[b]
                idtmp = idlist[i]
                tmpscore = scores_b[idtmp - 1]

                # if tmpscore is 0, the bin is empty, maximally abnormal, the value gets the highest score of the
                # most abnormal bin in the hist for this feature
                # there is a small possibility the scores are not reliable when there is a very small number of bins
                # and every bin, which is not empty, has a similar score -> most abnormal bin is still normal
                # not expected to happen in real life scenario and only when decicion_function is used
                # This is never a Problem wenn hbos is used for unsupervised anomaly detection.

                if tmpscore == 0:
                    biggestid = self.highest_score_id_[b]
                    tmpscore = scores_b[biggestid - 1]

                if self.ranked:
                    score = score + tmpscore
                elif self.log_scale:
                    tmpscore = math.log10(tmpscore)
                    score = score + tmpscore
                else:
                    score = score * tmpscore
                if self.save_explainability_scores:
                    scores_per_sample.append(tmpscore)
            if self.save_explainability_scores:
                self.explainability_scores_.append(scores_per_sample)
            hbos_scores[i] = score
        return hbos_scores

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.

        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """

        check_is_fitted(self, ['hist_', 'bin_edges_array_'])
        X = check_array(X)
        samples = len(X)
        features = X.shape[1]

        bin_id_array = []
        for i in range(self.n_features_):

            dynlist = self.bin_width_array_[i]
            bin_width_left = dynlist[0]
            bin_width_right = dynlist[-1]

            edges = self.bin_edges_array_[i]
            leftedge = edges[0]
            rightedge = edges[-1]
            binids = np.digitize(X[:, i], self.bin_edges_array_[i], right=False)
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
                        binids[j] = self.highest_score_id_[i]

                # if the sample does not belong to any bin
                # binids[j] is bigger than number of bins in that histogram: it is too big for every bin
                elif binids[j] > self.n_bins_array_[i]:
                    distance = X[j, i] - rightedge

                    # if it is only slightly bigger than the most right bin
                    # assign it to the most right bin
                    if distance <= bin_width_right * self.tol:
                        binids[j] = binids[j] - 1

                    # else assign it the ID of the bin with the highest score
                    # same case here, not reliable if every bin has a similar score,
                    # most abnormal bin is still normal ->
                    # sample does not belong to any bin -> still gets normal score
                    else:
                        binids[j] = self.highest_score_id_[i]
            bin_id_array.append(binids)
        hbos_scores = self.calc_hbos_scores(samples, features, bin_id_array)
        return hbos_scores

    def get_explainability_scores(self, sampleid):
        """Function to return explainability scores of the fitted estimator.
        Only works if save_explainability_scores was True while fitting the estimator.
        If this was not the case an empty array will be returned.
        Set save_explainability_scores to True.

            Parameters
            ----------
            sampleid : int, String
                The index of the data point one wishes get the explainability scores of.
                "all" returns the scores of all samples.

            Returns
            -------
            explainability_scores : array of shape (n_features) or array of shape (n_samples, n_features)
                The explainability scores.
            max_scores: array of shape (n_features) containing the maximum score of every histogram.
            """
        check_is_fitted(self, ['hist_', 'bin_edges_array_'])
        max_scores = []
        if len(self.explainability_scores_) == 0:
            print('Warning: Explainability scores have not been saved. Set save_explainability_scores to True '
                  'before fitting the estimator.', file=sys.stderr)
            return self.explainability_scores_, max_scores
        elif sampleid == "all":
            max_scores = np.max(self.explainability_scores_, axis=0)
            return self.explainability_scores_, max_scores
        else:
            max_scores = np.max(self.explainability_scores_, axis=0)
            return self.explainability_scores_[sampleid], max_scores
