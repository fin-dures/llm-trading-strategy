import numpy as np
import pandas as pd

class Strategy:
    def __init__(self,
                 cluster_window=60,
                 frac_increase_threshold=0.20,
                 separation_threshold=1.2,
                 vol_median_window=30,
                 long_sma=120,
                 kmeans_iters=10,
                 eps=1e-9):
        """
        Parameters (kept modest and interpretable):
         - cluster_window: number of closes to consider for the bimodality test
         - frac_increase_threshold: required increase in fraction assigned to upper cluster (second half vs first half)
         - separation_threshold: required separation (in z-units) between the two cluster centers
         - vol_median_window: window for median volume used as participation filter
         - long_sma: long SMA period used as trend/exit filter
         - kmeans_iters: iterations for the simple 1D k-means
         - eps: small value to avoid division by zero
        """
        self.cluster_window = cluster_window
        self.frac_increase_threshold = frac_increase_threshold
        self.separation_threshold = separation_threshold
        self.vol_median_window = vol_median_window
        self.long_sma = long_sma
        self.kmeans_iters = kmeans_iters
        self.eps = eps

    def _kmeans_1d_two_clusters(self, x, max_iters=10):
        """
        Simple 1D k-means for k=2.
        Returns: centers (array length 2), labels (0/1)
        If one cluster becomes empty, keep previous center.
        """
        # x is 1d numpy
        if len(x) == 0:
            return np.array([0.0, 0.0]), np.zeros(0, dtype=int)
        # initialize centers at min and max
        c0, c1 = float(np.min(x)), float(np.max(x))
        centers = np.array([c0, c1], dtype=float)
        labels = np.zeros_like(x, dtype=int)
        for _ in range(max_iters):
            # assign
            d0 = np.abs(x - centers[0])
            d1 = np.abs(x - centers[1])
            new_labels = (d1 < d0).astype(int)  # 1 if closer to centers[1]
            # update centers
            updated = centers.copy()
            for k in (0, 1):
                assigned = x[new_labels == k]
                if assigned.size > 0:
                    updated[k] = assigned.mean()
                # else keep previous center
            # check convergence
            if np.allclose(updated, centers, rtol=1e-6, atol=1e-8):
                labels = new_labels
                centers = updated
                break
            centers = updated
            labels = new_labels
        return centers, labels

    def generate_signals(self, df):
        """
        Input: df with columns open, high, low, close, volume.
        Output: entries, exits — boolean pandas Series aligned with df.index.
        """
        close = df['close'].astype(float)
        volume = df['volume'].astype(float)

        n = len(df)
        entries = pd.Series(False, index=df.index)
        exits = pd.Series(False, index=df.index)

        w = int(self.cluster_window)
        half = w // 2

        # precompute rolling vol median and long sma
        vol_median = volume.rolling(self.vol_median_window, min_periods=1).median()
        long_sma = close.rolling(self.long_sma, min_periods=1).mean()

        # We'll store cluster info for each window end to make exit decisions consistent
        cluster_centers_list = [None] * n
        cluster_labels_list = [None] * n
        separation_list = np.zeros(n)

        # iterate over bars where we can compute a full window
        for i in range(n):
            if i < w - 1:
                # not enough data yet
                continue
            window_vals = close.values[i - w + 1: i + 1]
            # normalize to z to make separation comparable across regimes
            mean_w = window_vals.mean()
            std_w = window_vals.std(ddof=0) + self.eps
            z = (window_vals - mean_w) / std_w

            centers, labels = self._kmeans_1d_two_clusters(z, max_iters=self.kmeans_iters)
            # ensure centers[0] is the lower in value, centers[1] is higher
            if centers[0] > centers[1]:
                centers = centers[::-1]
                labels = 1 - labels  # flip labels accordingly

            # compute separation in z-units (difference divided by 1 since z has std ~1)
            sep = (centers[1] - centers[0]) / (1.0 + self.eps)
            cluster_centers_list[i] = centers
            cluster_labels_list[i] = labels
            separation_list[i] = sep

            # Entry logic
            # Conditions:
            # 1) clear separation between clusters
            if sep < self.separation_threshold:
                continue
            # 2) current bar belongs to upper cluster
            current_label = labels[-1]
            upper_label = int(np.argmax(centers))  # should be 1 but robust
            if current_label != upper_label:
                continue
            # 3) fraction assigned to upper cluster rises in second half vs first half
            first_half_frac = np.mean(labels[:half] == upper_label)
            second_half_frac = np.mean(labels[half:] == upper_label)
            if (second_half_frac - first_half_frac) < self.frac_increase_threshold:
                continue
            # 4) today's volume above recent median (participation filter)
            if volume.iat[i] <= vol_median.iat[i]:
                continue
            # passed all filters -> entry
            entries.iat[i] = True

        # Exit logic:
        # Exit when current window indicates the observation belongs to the lower cluster
        # (i.e., clustering has flipped back) OR price falls below long_sma
        for i in range(n):
            # exit on trend failure
            if close.iat[i] < long_sma.iat[i]:
                exits.iat[i] = True
                continue
            # if cluster info exists at this bar, use it to detect flip
            labels = cluster_labels_list[i]
            centers = cluster_centers_list[i]
            sep = separation_list[i]
            if labels is None or centers is None:
                continue
            # require separation to be meaningful; if clustering weak, don't exit via cluster flip
            if sep < self.separation_threshold:
                continue
            upper_label = int(np.argmax(centers))
            current_label = labels[-1]
            if current_label != upper_label:
                exits.iat[i] = True

        # Return boolean Series
        entries = entries.astype(bool)
        exits = exits.astype(bool)
        return entries, exits