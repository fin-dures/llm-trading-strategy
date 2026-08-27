import numpy as np
import pandas as pd

class Strategy:
    def __init__(self,
                 break_window=10,        # lookback for breakout (prior N bars)
                 abs_window=20,          # window for absorption calculation
                 vol_window=20,          # window for median volume
                 mom_period=5,           # short-term momentum period (bars)
                 sma_trend=100,          # long-term trend SMA period
                 absorption_quantile=0.7 # threshold quantile for absorption advantage
                 ):
        self.break_window = break_window
        self.abs_window = abs_window
        self.vol_window = vol_window
        self.mom_period = mom_period
        self.sma_trend = sma_trend
        self.absorption_quantile = absorption_quantile

    def _safe_div(self, num, den, eps=1e-9):
        return num / (den + eps)

    def generate_signals(self, df):
        """
        df: pandas DataFrame with columns: open, high, low, close, volume
        returns: entries, exits -> boolean pandas Series aligned with df.index
        """
        # basic series
        o = df['open']
        h = df['high']
        l = df['low']
        c = df['close']
        v = df['volume']

        # Range (avoid zero)
        rng = (h - l).replace(0, np.nan)
        # Lower wick ratio: portion of range taken by lower wick
        lower_wick = (np.minimum(o, c) - l)
        lower_wick_ratio = self._safe_div(lower_wick, rng).fillna(0).clip(0, 1)

        # Upper wick ratio
        upper_wick = (h - np.maximum(o, c))
        upper_wick_ratio = self._safe_div(upper_wick, rng).fillna(0).clip(0, 1)

        # Volume-weighted wick contributions
        vol_lower = v * lower_wick_ratio
        vol_upper = v * upper_wick_ratio

        # Rolling sums to measure absorption (buyers defending lows) vs selling (sellers at highs)
        vol_sum = v.rolling(self.abs_window, min_periods=1).sum()
        lower_sum = vol_lower.rolling(self.abs_window, min_periods=1).sum()
        upper_sum = vol_upper.rolling(self.abs_window, min_periods=1).sum()

        absorption = self._safe_div(lower_sum, vol_sum)  # fraction of volume near lows
        selling_pressure = self._safe_div(upper_sum, vol_sum)  # fraction near highs

        # Absorption advantage: positive means more volume concentrated in lower-wicks vs upper-wicks
        absorption_adv = (absorption - selling_pressure).fillna(0)

        # Normalized short-term momentum (simple pct change over mom_period)
        roc = c.pct_change(self.mom_period)

        # Long-term trend: simple moving average
        sma_long = c.rolling(self.sma_trend, min_periods=1).mean()

        # Prior N-bar high (exclude current bar): we compute rolling max on shifted highs
        prior_high = h.shift(1).rolling(self.break_window, min_periods=1).max()

        # Volume threshold: compare current volume to median volume over vol_window
        vol_med = v.rolling(self.vol_window, min_periods=1).median()

        # Absorption threshold: require current absorption_adv to exceed its recent quantile
        abs_threshold = absorption_adv.rolling(self.abs_window, min_periods=1).quantile(self.absorption_quantile)

        # Build entry conditions (all evaluated with information available at close of this bar)
        cond_trend = c > sma_long  # price in long-term uptrend
        cond_breakout = c > prior_high  # close above prior N-bar high (no look-ahead)
        cond_volume = v > vol_med  # above-normal participation
        cond_mom = roc > 0  # short-term positive momentum
        cond_abs = absorption_adv > abs_threshold  # elevated buy-side absorption advantage

        entries = (cond_trend & cond_breakout & cond_volume & cond_mom & cond_abs)
        entries = entries.fillna(False).astype(bool)

        # Exit conditions: any of these trigger an exit at the close of the bar
        exit_trend_break = c < sma_long  # long-term trend broken
        exit_mom_flip = roc < 0  # momentum flipped negative
        exit_abs_fade = absorption_adv < absorption_adv.rolling(self.abs_window, min_periods=1).median()  # absorption advantage faded

        exits = (exit_trend_break | exit_mom_flip | exit_abs_fade)
        exits = exits.fillna(False).astype(bool)

        # Ensure entries/exits are pandas Series aligned with df
        entries = pd.Series(entries, index=df.index, name='entries')
        exits = pd.Series(exits, index=df.index, name='exits')

        return entries, exits