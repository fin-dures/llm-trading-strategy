import pandas as pd
import numpy as np

class Strategy:
    def __init__(self,
                 atr_period=14,
                 vol_range_short=10,
                 vol_range_long=50,
                 pct_up_short=10,
                 pct_up_long=50,
                 breakout_lookback=10,
                 short_sma=20,
                 mid_sma=50,
                 vol_spike_mult=1.2,
                 range_expand_mult=1.2,
                 vol_range_ratio_thresh=1.4,
                 pct_up_delta_thresh=0.05):
        # Parameters controlling the detection windows and thresholds
        self.atr_period = atr_period
        self.vol_range_short = vol_range_short
        self.vol_range_long = vol_range_long
        self.pct_up_short = pct_up_short
        self.pct_up_long = pct_up_long
        self.breakout_lookback = breakout_lookback
        self.short_sma = short_sma
        self.mid_sma = mid_sma
        self.vol_spike_mult = vol_spike_mult
        self.range_expand_mult = range_expand_mult
        self.vol_range_ratio_thresh = vol_range_ratio_thresh
        self.pct_up_delta_thresh = pct_up_delta_thresh

    def _atr(self, high, low, close, n):
        # Classic ATR (rolling mean of True Range)
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        # Use rolling mean to smooth ATR
        atr = tr.rolling(n, min_periods=1).mean()
        return atr

    def generate_signals(self, df):
        # Expect df to contain columns: open, high, low, close, volume
        o = df['open']
        h = df['high']
        l = df['low']
        c = df['close']
        v = df['volume']

        eps = 1e-9
        # 1) ATR and short/mid SMAs for trend and range-normalization
        atr = self._atr(h, l, c, self.atr_period)
        sma_short = c.rolling(self.short_sma, min_periods=1).mean()
        sma_mid = c.rolling(self.mid_sma, min_periods=1).mean()

        # 2) Volume-per-range: proxy for liquidity concentration (many contracts in a tight range)
        bar_range = (h - l).replace(0, eps).abs()
        vol_per_range = v / (bar_range + eps)

        # Rolling statistics for vol_per_range
        vpr_short_mean = vol_per_range.rolling(self.vol_range_short, min_periods=1).mean()
        vpr_long_mean = vol_per_range.rolling(self.vol_range_long, min_periods=1).mean()

        # 3) Up-bar participation: percent of bars with close > open
        is_up = (c > o).astype(float)
        pct_up_short = is_up.rolling(self.pct_up_short, min_periods=1).mean()
        pct_up_long = is_up.rolling(self.pct_up_long, min_periods=1).mean()

        # 4) Recent highs for breakout detection
        recent_high = h.rolling(self.breakout_lookback, min_periods=1).max().shift(1)  # prior N-bar high (no look-ahead)
        # We require breaking above prior high; comparing to prior-high ensures the breakout is observable at close

        # 5) Volume baseline for spike detection
        vol_median = v.rolling(self.vol_range_long, min_periods=1).median()

        # ENTRY CONDITIONS (evaluated at the close of the bar)
        cond_accumulation = (
            (vpr_short_mean > vpr_long_mean * self.vol_range_ratio_thresh)  # elevated volume-per-range (absorption/accumulation)
            & ((pct_up_short - pct_up_long) > self.pct_up_delta_thresh)     # rising share of up-bars (buyer participation rising)
        )

        cond_breakout = (
            (c > recent_high)                                               # close breaks prior short-window high
            & (c > sma_mid)                                                  # price above medium-term trend
            & (v > vol_median * self.vol_spike_mult)                         # breakout accompanied by above-normal volume
            & ((h - l) > (atr * self.range_expand_mult))                     # breakout bar has larger than normal range
        )

        entries = cond_accumulation & cond_breakout

        # EXIT CONDITIONS (evaluated at the close of the bar)
        # Exit on short-term trend breakdown, medium-term trend failure, or short-term momentum flip.
        momentum_flip = c < c.shift(3)  # price losing short-term momentum vs 3 bars ago
        exit_short_trend = c < sma_short
        exit_mid_trend = c < sma_mid

        exits = exit_short_trend | exit_mid_trend | momentum_flip

        # Ensure boolean Series aligned with df.index
        entries = entries.reindex(df.index, fill_value=False).astype(bool)
        exits = exits.reindex(df.index, fill_value=False).astype(bool)

        return entries, exits