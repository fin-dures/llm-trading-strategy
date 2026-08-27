import pandas as pd
import numpy as np

class Strategy:
    def __init__(self,
                 ac_window=14,        # window to compute lag-1 autocorrelation of returns
                 ac_threshold=0.05,   # autocorr threshold that signals emergence of momentum
                 sma_long=100,        # long-term trend definition
                 vol_window=50        # window to compute median volume for participation filter
                 ):
        self.ac_window = ac_window
        self.ac_threshold = ac_threshold
        self.sma_long = sma_long
        self.vol_window = vol_window

    def _rolling_lag1_autocorr(self, returns, window):
        # Returns lag-1 autocorrelation computed on a rolling window.
        # Uses pandas' rolling.apply with raw=False so we can use Series.corr with shift(1).
        # Requires at least 'window' observations to produce a value.
        if window < 2:
            # lag-1 autocorr undefined for window < 2
            return pd.Series(np.nan, index=returns.index)
        return returns.rolling(window=window, min_periods=window).apply(
            lambda arr: pd.Series(arr).corr(pd.Series(arr).shift(1)),
            raw=False
        )

    def generate_signals(self, df):
        """
        Input:
            df - pandas DataFrame containing only the columns: open, high, low, close, volume
        Output:
            entries, exits - two boolean pandas Series aligned with df.index
        """
        close = df['close'].astype(float)
        volume = df['volume'].astype(float)

        # short returns
        returns = close.pct_change()

        # rolling lag-1 autocorrelation of returns
        ac = self._rolling_lag1_autocorr(returns, self.ac_window)

        # long-term trend: price above long SMA
        sma_long = close.rolling(window=self.sma_long, min_periods=1).mean()
        trend_up = close > sma_long

        # volume participation: current volume above recent median
        vol_median = volume.rolling(window=self.vol_window, min_periods=1).median()
        vol_confirm = volume > vol_median

        # autocorr crossing event: previous bar below threshold, current >= threshold
        ac_prev = ac.shift(1)
        ac_cross_up = (ac_prev < self.ac_threshold) & (ac >= self.ac_threshold)

        # require a positive immediate return to confirm initial momentum
        positive_move = returns > 0

        # Compose entry signal: momentum regime emergence + long-term uptrend + volume confirmation + positive move
        entries = ac_cross_up & trend_up & vol_confirm & positive_move

        # Exits: momentum collapse or momentum flip negative or trend broken
        ac_negative = ac < 0
        momentum_flip = returns < 0
        trend_break = close < sma_long

        exits = ac_negative | momentum_flip | trend_break

        # Ensure boolean Series (no NaNs)
        entries = entries.fillna(False).astype(bool)
        exits = exits.fillna(False).astype(bool)

        return entries, exits