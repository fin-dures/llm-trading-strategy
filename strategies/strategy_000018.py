import numpy as np
import pandas as pd

class Strategy:
    def __init__(self,
                 atr_window=14,
                 mom_lookback=3,
                 vol_norm_window=20,
                 thresh_window=100,
                 thresh_quantile=0.90,
                 sma_long=50,
                 sma_slope_lookback=5):
        """
        Parameters:
        - atr_window: window for ATR (volatility scaling)
        - mom_lookback: lookback for short-term momentum (close - close.shift(mom_lookback))
        - vol_norm_window: window to normalize volume (median volume)
        - thresh_window: window to compute the recent high-percentile threshold of the metric
        - thresh_quantile: percentile to treat as "unusually high" volume-weighted momentum
        - sma_long: window for the long SMA used to confirm trend
        - sma_slope_lookback: lookback to check SMA slope (trend must be rising)
        """
        self.atr_window = atr_window
        self.mom_lookback = mom_lookback
        self.vol_norm_window = vol_norm_window
        self.thresh_window = thresh_window
        self.thresh_quantile = thresh_quantile
        self.sma_long = sma_long
        self.sma_slope_lookback = sma_slope_lookback

    def _true_range(self, high, low, close):
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr

    def generate_signals(self, df):
        """
        Input:
            df: pandas DataFrame with columns: open, high, low, close, volume
        Returns:
            entries, exits: boolean pandas Series aligned with df.index
        """
        # Ensure required columns exist
        required = {'open', 'high', 'low', 'close', 'volume'}
        if not required.issubset(df.columns):
            raise ValueError(f"Input df must contain columns: {required}")

        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        volume = df['volume'].astype(float)

        # 1) ATR for volatility scaling
        tr = self._true_range(high, low, close)
        atr = tr.rolling(self.atr_window, min_periods=1).mean()

        # 2) Short-term momentum (price change over mom_lookback)
        price_diff = close - close.shift(self.mom_lookback)

        # 3) Normalize volume by recent typical volume (median reduces outlier sensitivity)
        vol_med = volume.rolling(self.vol_norm_window, min_periods=1).median()
        # avoid division by zero
        vol_norm = volume / (vol_med.replace(0, np.nan))
        vol_norm = vol_norm.fillna(1.0)

        # 4) Volume-weighted, volatility-adjusted momentum metric
        # Higher when the short-term price rise is large relative to recent volatility AND supported by above-typical volume.
        # Use small epsilon to avoid dividing by zero ATR.
        eps = 1e-9
        vol_adj_mom = (price_diff / (atr + eps)) * vol_norm

        # 5) Dynamic threshold: recent high-percentile of vol_adj_mom (rolling, inclusive of current bar)
        mom_threshold = vol_adj_mom.rolling(self.thresh_window, min_periods=1).quantile(self.thresh_quantile)

        # 6) Long-term trend confirmation: price above SMA and SMA slope positive
        sma = close.rolling(self.sma_long, min_periods=1).mean()
        sma_slope = sma - sma.shift(self.sma_slope_lookback)

        in_uptrend = (close > sma) & (sma_slope > 0)

        # 7) Entry rule: vol_adj_mom exceeds dynamic threshold AND uptrend holds
        entries = (vol_adj_mom > mom_threshold) & in_uptrend

        # 8) Exit rule: momentum flips negative (loss of participation/strength) OR price below long SMA (trend failure)
        exits = (vol_adj_mom < 0) | (close < sma)

        # Ensure boolean Series and aligned index, no NaNs (convert NaN to False)
        entries = entries.fillna(False).astype(bool)
        exits = exits.fillna(False).astype(bool)

        # Return boolean Series
        return entries.reindex(df.index, fill_value=False), exits.reindex(df.index, fill_value=False)