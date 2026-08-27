import numpy as np
import pandas as pd

class Strategy:
    def __init__(self,
                 sma_long=200,
                 sma_short=20,
                 atr_period=14,
                 roc_period=3,
                 oversold_atr_mult=1.5,
                 vol_median_window=20,
                 vol_min_mult=0.7,
                 max_hold=10):
        """
        Parameters are purposely few and interpretable:
        - sma_long: trend filter period (e.g., 200)
        - sma_short: short mean-reversion target (e.g., 20)
        - atr_period: period for ATR
        - roc_period: lookback for short-term momentum flip
        - oversold_atr_mult: how many ATRs below sma_short counts as 'oversold'
        - vol_median_window: window to compute typical volume
        - vol_min_mult: minimum multiplier of median volume to consider the signal valid
        - max_hold: maximum bars to hold a trade before force-exit
        """
        self.sma_long = sma_long
        self.sma_short = sma_short
        self.atr_period = atr_period
        self.roc_period = roc_period
        self.oversold_atr_mult = oversold_atr_mult
        self.vol_median_window = vol_median_window
        self.vol_min_mult = vol_min_mult
        self.max_hold = max_hold

    def _atr(self, high, low, close, period):
        # True Range
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period, min_periods=1).mean()
        return atr

    def generate_signals(self, df):
        """
        Input:
            df: pandas DataFrame with columns ['open','high','low','close','volume']
        Returns:
            entries, exits: boolean pandas Series indexed like df
        """
        # Copy to avoid modifying original
        o = df['open']
        h = df['high']
        l = df['low']
        c = df['close']
        v = df['volume']

        # Indicators
        sma_long = c.rolling(self.sma_long, min_periods=1).mean()
        sma_short = c.rolling(self.sma_short, min_periods=1).mean()
        atr = self._atr(h, l, c, self.atr_period)
        roc = (c - c.shift(self.roc_period)) / c.shift(self.roc_period)  # can be NaN early

        vol_med = v.rolling(self.vol_median_window, min_periods=1).median()

        # Entry condition components evaluated at close of the bar
        in_long_term_uptrend = c > sma_long
        is_oversold = c < (sma_short - self.oversold_atr_mult * atr)
        momentum_flip = roc > 0  # price higher than roc_period bars ago
        bullish_close = c > o
        sufficient_volume = v > (vol_med * self.vol_min_mult)

        raw_entry_cond = in_long_term_uptrend & is_oversold & momentum_flip & bullish_close & sufficient_volume

        # Prepare output series
        entries = pd.Series(False, index=df.index)
        exits = pd.Series(False, index=df.index)

        # Simulate a simple position lifecycle so that entries don't stack and we can enforce max_hold:
        in_position = False
        hold_count = 0

        # Iterate through bars sequentially (no lookahead)
        for i in range(len(df)):
            if not in_position:
                # Can only enter if raw_entry_cond true at this bar
                if bool(raw_entry_cond.iloc[i]):
                    entries.iloc[i] = True
                    in_position = True
                    hold_count = 0
                # else remain flat
            else:
                # Check exit conditions at current bar (based only on information up to this bar)
                # 1) Price has reverted above the short SMA (profit target)
                # 2) Long-term trend broken (defensive exit)
                # 3) Max holding time reached
                exit_profit = c.iloc[i] > sma_short.iloc[i]
                exit_trend_break = c.iloc[i] < sma_long.iloc[i]
                exit_time = hold_count >= (self.max_hold - 1)  # if we entered at bar 0, after max_hold bars we exit
                if exit_profit or exit_trend_break or exit_time:
                    exits.iloc[i] = True
                    in_position = False
                    hold_count = 0
                else:
                    hold_count += 1

        # Ensure boolean dtype
        entries = entries.fillna(False).astype(bool)
        exits = exits.fillna(False).astype(bool)

        return entries, exits