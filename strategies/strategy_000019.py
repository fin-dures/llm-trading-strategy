import pandas as pd
import numpy as np

class Strategy:
    def __init__(self,
                 sma_long=50,
                 vpt_mom_period=5,
                 vpt_accel_period=1,
                 normalize_window=20,
                 accel_threshold=1.5,
                 vol_median_window=20):
        """
        Parameters:
        - sma_long: period for long-term trend (price must be above this SMA to consider entries)
        - vpt_mom_period: lookback for VPT momentum (difference over this many bars)
        - vpt_accel_period: lookback for VPT acceleration (difference of momentum)
        - normalize_window: window to compute rolling std for normalization
        - accel_threshold: normalized-acceleration threshold to trigger entry
        - vol_median_window: window to compute median volume for participation filter
        """
        self.sma_long = sma_long
        self.vpt_mom_period = vpt_mom_period
        self.vpt_accel_period = vpt_accel_period
        self.normalize_window = normalize_window
        self.accel_threshold = accel_threshold
        self.vol_median_window = vol_median_window

    def generate_signals(self, df):
        """
        Input: df with columns ['open','high','low','close','volume']
        Returns: entries, exits boolean Series aligned with df.index
        Entry rule:
          - VPT acceleration (normalized by recent std) > accel_threshold
          - VPT momentum > 0 (net upward price-flow)
          - close > sma_long (longer-term uptrend)
          - volume > rolling median volume (participation)
        Exit rule:
          - normalized VPT acceleration < 0 (acceleration turned negative)
            OR VPT momentum <= 0 (buyer flow dissipated)
            OR close < sma_long (trend failure)
        Notes: No lookahead; all signals are computed from past/current bars only.
        """
        # Validate columns
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required}")

        close = df['close'].astype(float)
        volume = df['volume'].astype(float)

        # Compute Volume Price Trend (VPT): cumulative sum of volume * pct_change(close)
        # Use pct_change to scale by price level (standard VPT definition)
        close_pct_change = close.pct_change().fillna(0.0)
        vpt_flow = volume * close_pct_change
        vpt = vpt_flow.cumsum()  # starts at 0 and accumulates

        # VPT momentum: difference over vpt_mom_period bars
        vpt_mom = vpt.diff(self.vpt_mom_period)

        # VPT acceleration: difference of momentum over vpt_accel_period
        vpt_accel = vpt_mom.diff(self.vpt_accel_period)

        # Normalize acceleration by recent variability of vpt_mom to make threshold adaptive
        mom_std = vpt_mom.rolling(self.normalize_window, min_periods=1).std().replace(0, np.nan)
        norm_accel = vpt_accel / mom_std

        # Long-term trend filter: simple moving average of close
        sma_long = close.rolling(self.sma_long, min_periods=1).mean()

        # Volume participation filter: require volume above its recent median
        vol_median = volume.rolling(self.vol_median_window, min_periods=1).median()

        # Entry condition (computed at close of the bar)
        entry_cond = (
            (norm_accel > self.accel_threshold) &
            (vpt_mom > 0) &
            (close > sma_long) &
            (volume > vol_median)
        )

        # Exit condition: any of these conditions indicates weakening buyer flow or trend failure
        exit_cond = (
            (norm_accel < 0) |
            (vpt_mom <= 0) |
            (close < sma_long)
        )

        # Ensure boolean Series, no NaNs (treat insufficient-data rows as False)
        entries = entry_cond.fillna(False).astype(bool)
        exits = exit_cond.fillna(False).astype(bool)

        # Align index with input df
        entries = pd.Series(entries, index=df.index)
        exits = pd.Series(exits, index=df.index)

        return entries, exits