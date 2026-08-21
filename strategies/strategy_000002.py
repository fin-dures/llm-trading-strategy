import pandas as pd
import numpy as np

class Strategy:
    def __init__(self,
                 atr_window=14,
                 range_window=20,
                 vol_window=20,
                 momentum_window=5,
                 atr_multiplier=1.6,
                 vol_z_thresh=1.0,
                 momentum_thresh=0.006,
                 exit_atr_multiplier=1.0):
        self.atr_window = atr_window
        self.range_window = range_window
        self.vol_window = vol_window
        self.momentum_window = momentum_window
        self.atr_multiplier = atr_multiplier
        self.vol_z_thresh = vol_z_thresh
        self.momentum_thresh = momentum_thresh
        self.exit_atr_multiplier = exit_atr_multiplier

    def generate_signals(self, df):
        """
        Input: df with columns ['open','high','low','close','volume']
        Output: entries, exits -> two boolean pandas Series aligned with df.index
        """
        o = df['open']
        h = df['high']
        l = df['low']
        c = df['close']
        v = df['volume']

        # Previous close for TR calculations (shifted so no future info)
        prev_close = c.shift(1)

        # True Range (TR) and ATR (use rolling mean)
        tr1 = (h - l).abs()
        tr2 = (h - prev_close).abs()
        tr3 = (l - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_window, min_periods=1).mean()
        atr_prev = atr.shift(1)  # use previous ATR for thresholds (no lookahead)

        # Recent highs and lows (exclude current bar by using shift(1))
        recent_high_max = h.shift(1).rolling(self.range_window, min_periods=1).max()
        recent_low_min = l.shift(1).rolling(self.range_window, min_periods=1).min()

        # Volume z-score based on history up to previous bar (no lookahead)
        vol_mean_prev = v.shift(1).rolling(self.vol_window, min_periods=1).mean()
        vol_std_prev = v.shift(1).rolling(self.vol_window, min_periods=1).std().replace(0, np.nan)
        vol_z_prev = (v.shift(1) - vol_mean_prev) / vol_std_prev

        # Momentum: percent change over momentum_window (uses current close and past close)
        momentum = c / c.shift(self.momentum_window) - 1

        # Entry rule: breakout above recent highs + ATR buffer, volume surge and positive momentum
        entry_condition = (
            (c > (recent_high_max + self.atr_multiplier * atr_prev)) &
            (vol_z_prev > self.vol_z_thresh) &
            (momentum > self.momentum_thresh)
        )

        # Additional filter: prefer bars that closed above open (bullish bar)
        entry_condition &= (c > o)

        # Exit rule: either breakdown below recent lows minus ATR buffer OR momentum strongly negative
        exit_condition = (
            (c < (recent_low_min - self.exit_atr_multiplier * atr_prev)) |
            (momentum < -self.momentum_thresh)
        )

        # Clean NaNs -> False
        entries = entry_condition.fillna(False).astype(bool)
        exits = exit_condition.fillna(False).astype(bool)

        # Return two boolean Series
        return entries, exits


# Quick usage example (not executed here):
# strat = Strategy()
# entries, exits = strat.generate_signals(df)