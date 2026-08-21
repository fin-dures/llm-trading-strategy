import pandas as pd
import numpy as np

class Strategy:
    """
    Volume-Adjusted Bollinger-ATR Breakout Strategy
    generate_signals returns two pandas Series (entries, exits) aligned with df.index
    """
    def __init__(self, bb_window=20, bb_k=2.0, atr_window=14, vol_window=20,
                 vol_surge_mult=1.6, min_atr_multiplier=0.4):
        self.bb_window = int(bb_window)
        self.bb_k = float(bb_k)
        self.atr_window = int(atr_window)
        self.vol_window = int(vol_window)
        self.vol_surge_mult = float(vol_surge_mult)
        self.min_atr_multiplier = float(min_atr_multiplier)

    def _true_range(self, high, low, close):
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr

    def generate_signals(self, df):
        # Expect df to contain columns: open, high, low, close, volume
        o = df['open']
        h = df['high']
        l = df['low']
        c = df['close']
        v = df['volume']

        # Typical price
        tp = (h + l + c) / 3.0

        # Bollinger-like bands on typical price
        tp_ma = tp.rolling(self.bb_window, min_periods=1).mean()
        tp_std = tp.rolling(self.bb_window, min_periods=1).std(ddof=0).fillna(0.0)

        # Volume normalization factor (recent average)
        vol_mean = v.rolling(self.vol_window, min_periods=1).mean()

        # Volume-adjusted band width: when volume is high, bands expand mildly
        vol_factor = (v / (vol_mean + 1e-9)).clip(lower=0.2, upper=5.0)
        upper_band = tp_ma + self.bb_k * tp_std * np.sqrt(vol_factor)
        lower_band = tp_ma - self.bb_k * tp_std * np.sqrt(vol_factor)

        # ATR for volatility thresholding
        tr = self._true_range(h, l, c)
        atr = tr.rolling(self.atr_window, min_periods=1).mean()

        # Momentum: short momentum on close (rate of change)
        mom = c / c.shift(1) - 1.0

        # Volume surge boolean
        vol_surge = v > (vol_mean * self.vol_surge_mult)

        # Price distance from center normalized by ATR
        dist_from_ma = (c - tp_ma) / (atr + 1e-9)

        # Entry conditions for long:
        #  - Close above upper_band (breakout)
        #  - Positive short-term momentum
        #  - Volume surge
        #  - Price has moved at least min_atr_multiplier * ATR above the MA (to avoid whipsaws)
        long_entry = (c > upper_band) & (mom > 0) & vol_surge & (dist_from_ma > self.min_atr_multiplier)

        # Exit conditions for long:
        #  - Close falls back below moving average (mean reversion)
        #  - OR close drops below upper_band by more than 1.0 ATR (failed breakout)
        long_exit = (c < tp_ma) | (c < (upper_band - 1.0 * atr))

        # Short idea (contrarian breakout fade) — optional: open short when strong negative breakout
        short_entry = (c < lower_band) & (mom < 0) & vol_surge & (dist_from_ma < -self.min_atr_multiplier)
        short_exit = (c > tp_ma) | (c > (lower_band + 1.0 * atr))

        # Combine into entries/exits with integer signals: +1 for long entry, -1 for short entry
        # The caller might expect boolean arrays; to be explicit, return DataFrame-like Series
        entries = pd.Series(0, index=df.index)
        exits = pd.Series(0, index=df.index)

        entries = entries.astype(int)
        exits = exits.astype(int)

        entries[long_entry] = 1
        entries[short_entry] = -1

        exits[long_exit] = 1   # exit long
        exits[short_exit] = -1 # exit short

        # Ensure no NaNs: replace with 0
        entries = entries.fillna(0).astype(int)
        exits = exits.fillna(0).astype(int)

        return entries, exits


# Example usage:
# df is a pandas DataFrame with columns: open, high, low, close, volume
# strat = Strategy()
# entries, exits = strat.generate_signals(df)