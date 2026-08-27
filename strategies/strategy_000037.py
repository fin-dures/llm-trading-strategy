import numpy as np
import pandas as pd

class Strategy:
    def __init__(self,
                 k1=5,                 # length of the "recent leg" (second pullback length)
                 k2=10,                # length of the prior up-leg window
                 min_up=0.08,          # minimum return for the first leg (r1)
                 pullback_min=-0.06,   # deepest acceptable second-leg retracement (r2)
                 pullback_max=-0.01,   # shallowest acceptable second-leg retracement (r2)
                 vol_ratio=0.9,        # second-leg avg volume must be < vol_ratio * first-leg avg volume
                 sma_long=50,          # medium-term trend SMA period
                 sma_short=10,         # short-term breakout-failure SMA period (exit)
                 max_hold=30):         # maximum holding bars (time-based exit)
        self.k1 = k1
        self.k2 = k2
        self.min_up = min_up
        self.pullback_min = pullback_min
        self.pullback_max = pullback_max
        self.vol_ratio = vol_ratio
        self.sma_long = sma_long
        self.sma_short = sma_short
        self.max_hold = max_hold

    def generate_signals(self, df):
        """
        Input df must contain columns: open, high, low, close, volume
        Returns two boolean pandas Series (entries, exits) aligned with df.index
        """
        close = df['close'].astype(float)
        volume = df['volume'].astype(float)

        # Basic moving averages for trend/exit rules
        sma_long = close.rolling(self.sma_long, min_periods=1).mean()
        sma_short = close.rolling(self.sma_short, min_periods=1).mean()

        # Prices defining the pattern:
        # mid_price = price at the pivot between the first leg and second pullback (k1 bars ago)
        # earlier_price = price before the first leg (k1 + k2 bars ago)
        mid_price = close.shift(self.k1)
        earlier_price = close.shift(self.k1 + self.k2)

        # Returns describing the two-leg structure
        # r1: return of the initial up-leg (earlier -> mid)
        # r2: recent retracement (mid -> current)
        r1 = (mid_price / earlier_price) - 1.0
        r2 = (close / mid_price) - 1.0

        # Volume averages:
        # vol_recent: average volume over the recent leg (k1 bars, includes current bar)
        # vol_prev: average volume over the prior up-leg window (k2 bars immediately before the recent leg)
        vol_recent = volume.rolling(self.k1, min_periods=1).mean()
        vol_prev = volume.shift(self.k1).rolling(self.k2, min_periods=1).mean()

        # Conditions for the pattern
        cond_up_leg = r1 >= self.min_up
        cond_second_pullback = (r2 >= self.pullback_min) & (r2 <= self.pullback_max)
        cond_volume_drop = vol_recent < (vol_prev * self.vol_ratio)

        # Trend confirmation: market in medium-term uptrend
        cond_trend = close > sma_long

        # Confirmation: price has recovered above the mid pivot (small positive tolerance allowed)
        confirmation = close > mid_price

        # Combine into an entry signal (all conditions evaluated using only historical/lagged info)
        entries = cond_up_leg & cond_second_pullback & cond_volume_drop & cond_trend & confirmation
        entries = entries.fillna(False).astype(bool)

        # Exits:
        # 1) Break of short-term SMA (trend-failure stop)
        exit_sma_break = close < sma_short

        # 2) Time-based exit: exit after max_hold bars from entry (we shift the entry forward)
        #    shifting the boolean entry series forward schedules an exit max_hold bars after the entry bar
        exit_time = entries.shift(self.max_hold).fillna(False).astype(bool)

        exits = (exit_sma_break | exit_time).fillna(False).astype(bool)

        # Ensure series are aligned and boolean
        entries = pd.Series(entries, index=df.index, name='entries', dtype=bool)
        exits = pd.Series(exits, index=df.index, name='exits', dtype=bool)

        return entries, exits