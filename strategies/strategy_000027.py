import pandas as pd
import numpy as np

class Strategy:
    def __init__(self,
                 wick_window=24,        # window to measure fraction of buyer-wick bars
                 wick_trend_lag=4,      # lookback lag to measure acceleration of buyer fraction
                 breakout_lookback=5,   # prior bars to define breakout high (excludes current bar)
                 vol_median_window=24,  # window for median volume (participation filter)
                 sma_long=50,           # long SMA for trend confirmation / exit
                 min_frac_increase=0.12,# required increase in buyer-fraction to consider acceleration
                 close_upper_frac=0.66, # close must be in upper X fraction of range
                 frac_drop_exit= -0.08  # exit when fraction falls by at least this amount relative to wick_trend_lag
                ):
        self.wick_window = wick_window
        self.wick_trend_lag = wick_trend_lag
        self.breakout_lookback = breakout_lookback
        self.vol_median_window = vol_median_window
        self.sma_long = sma_long
        self.min_frac_increase = min_frac_increase
        self.close_upper_frac = close_upper_frac
        self.frac_drop_exit = frac_drop_exit

    def generate_signals(self, df):
        """
        Input:
            df: pandas DataFrame with columns ['open','high','low','close','volume']
        Returns:
            entries, exits: boolean pandas Series aligned with df.index
        """
        # Basic checks
        required = {'open','high','low','close','volume'}
        if not required.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required}")

        o = df['open'].astype(float)
        h = df['high'].astype(float)
        l = df['low'].astype(float)
        c = df['close'].astype(float)
        v = df['volume'].astype(float)

        # Range and normalized positions (avoid division by zero)
        rng = (h - l).replace(0, np.nan)
        close_pos = (c - l) / rng  # 0 at low, 1 at high; NaN where range==0

        # Define a "buyer-defense" bar:
        # - closes above open (bullish)
        # - closes in upper portion of the range (close_pos > threshold)
        buyer_defense = (c > o) & (close_pos > self.close_upper_frac)
        buyer_defense = buyer_defense.fillna(False)

        # Rolling fraction of buyer_defense bars (includes current bar; allowed since we signal on close)
        frac_buyer = buyer_defense.rolling(self.wick_window, min_periods=1).mean()

        # Measure acceleration: difference between current fraction and its value wick_trend_lag bars ago
        frac_past = frac_buyer.shift(self.wick_trend_lag)
        frac_accel = frac_buyer - frac_past  # positive means rising share of buyer bars

        # Breakout check: current close strictly above the prior breakout_lookback high (exclude current bar)
        prior_high = h.shift(1).rolling(self.breakout_lookback, min_periods=1).max()
        breakout = c > prior_high

        # Volume participation: current volume above its recent median
        vol_median = v.rolling(self.vol_median_window, min_periods=1).median()
        vol_ok = v > vol_median

        # Long-term trend confirmation: close above long SMA
        sma_long = c.rolling(self.sma_long, min_periods=1).mean()
        trend_ok = c > sma_long

        # ENTRY rule:
        # - buyer-fraction acceleration exceeds threshold
        # - breakout above recent high
        # - close sits in upper fraction of the range (already implied by buyer_defense but keep explicit)
        # - volume above recent median
        # - optional trend confirmation (close > sma_long)
        entry_cond = (
            (frac_accel > self.min_frac_increase) &
            breakout &
            (close_pos > self.close_upper_frac) &
            vol_ok &
            trend_ok
        )

        # EXITS:
        # - buyer-fraction falls sufficiently relative to wick_trend_lag ago (weakening accumulation)
        # - OR price falls below long SMA (trend failure)
        frac_accel_drop = frac_buyer - frac_past  # same as frac_accel; negative means drop
        exit_cond = (
            (frac_accel_drop < self.frac_drop_exit) |
            (c < sma_long)
        )

        # Ensure boolean Series aligned to df.index and fill NaNs with False
        entries = entry_cond.reindex(df.index).fillna(False).astype(bool)
        exits = exit_cond.reindex(df.index).fillna(False).astype(bool)

        return entries, exits