import numpy as np
import pandas as pd

class Strategy:
    def __init__(self,
                 sma_short=20,
                 sma_long=100,
                 bp_window=60,        # window to measure recent buy-pressure
                 bp_baseline=240,     # long window to form baseline (median/std)
                 breakout_lookback=20,# lookback for prior high to define breakout
                 vol_window=20,
                 vol_factor=1.25,
                 bp_sigma=0.5):
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.bp_window = bp_window
        self.bp_baseline = bp_baseline
        self.breakout_lookback = breakout_lookback
        self.vol_window = vol_window
        self.vol_factor = vol_factor
        self.bp_sigma = bp_sigma

    def generate_signals(self, df):
        """
        Input:
            df: pandas DataFrame with columns ['open','high','low','close','volume']
        Output:
            entries, exits: boolean pandas Series aligned with df.index
        """
        # Basic safety checks
        required = {'open', 'high', 'low', 'close', 'volume'}
        if not required.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required}")

        high = df['high'].astype(float)
        low = df['low'].astype(float)
        close = df['close'].astype(float)
        vol = df['volume'].astype(float)

        # Position of close within the bar range: proxy for buying pressure within the bar
        rng = (high - low).replace(0, np.nan)
        pos = (close - low) / rng
        pos = pos.clip(0, 1).fillna(0.0)

        # buy-pressure per bar = volume weighted by how high the close sits in the range
        bp_bar = pos * vol

        # Rolling recent buy-pressure ratio: sum(bp_bar) / sum(volume)
        bp_num = bp_bar.rolling(self.bp_window, min_periods=1).sum()
        vol_den = vol.rolling(self.bp_window, min_periods=1).sum().replace(0, np.nan)
        bp_ratio = (bp_num / vol_den).fillna(0.0)  # fraction in [0,1] typically

        # Long-term baseline: median and std of bp_ratio
        bp_median = bp_ratio.rolling(self.bp_baseline, min_periods=1).median()
        bp_std = bp_ratio.rolling(self.bp_baseline, min_periods=1).std().fillna(0.0)

        # Signal that buy-pressure is elevated relative to long-term baseline
        bp_threshold = bp_median + self.bp_sigma * bp_std
        bp_signal = bp_ratio > bp_threshold

        # Short-term trend: price above short SMA
        sma_short = close.rolling(self.sma_short, min_periods=1).mean()
        sma_short_signal = close > sma_short

        # Breakout: close above the prior N-bar high (exclude current bar by shifting the rolling max)
        prior_high = high.rolling(self.breakout_lookback, min_periods=1).max().shift(1)
        breakout_signal = prior_high.notna() & (close > prior_high)

        # Volume must be above recent average (confirmation)
        vol_avg = vol.rolling(self.vol_window, min_periods=1).mean()
        vol_signal = vol > (vol_avg * self.vol_factor)

        # Entry when all conditions align
        entries = (bp_signal & sma_short_signal & breakout_signal & vol_signal)
        entries = entries.fillna(False).astype(bool)

        # Exits:
        # 1) Buying pressure fades: bp_ratio falls below its long-term median
        # 2) Price closes below the longer short-term trend (sma_long)
        sma_long = close.rolling(self.sma_long, min_periods=1).mean()
        exit_bp_fade = bp_ratio < bp_median
        exit_trend_break = close < sma_long

        exits = (exit_bp_fade | exit_trend_break)
        exits = exits.fillna(False).astype(bool)

        # Align index and return
        entries = pd.Series(entries, index=df.index)
        exits = pd.Series(exits, index=df.index)
        return entries, exits