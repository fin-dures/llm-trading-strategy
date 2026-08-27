import pandas as pd
import numpy as np

class Strategy:
    def __init__(self,
                 adx_len=14,
                 adx_entry_threshold=20.0,
                 adx_exit_threshold=18.0,
                 sma_short_len=10,
                 sma_long_len=50,
                 atr_len=14,
                 pullback_atr_mult=1.0,
                 vol_med_len=20):
        # Parameters (kept modest and interpretable)
        self.adx_len = adx_len
        self.adx_entry_threshold = adx_entry_threshold
        self.adx_exit_threshold = adx_exit_threshold
        self.sma_short_len = sma_short_len
        self.sma_long_len = sma_long_len
        self.atr_len = atr_len
        self.pullback_atr_mult = pullback_atr_mult
        self.vol_med_len = vol_med_len

    def _compute_true_range(self, high, low, close):
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr

    def _compute_atr(self, high, low, close, length):
        tr = self._compute_true_range(high, low, close)
        # use rolling mean as a simple ATR estimator (no look-ahead)
        atr = tr.rolling(length, min_periods=1).mean()
        return atr

    def _compute_directional_indicators(self, high, low, close, length):
        # Raw directional movement
        up_move = high.diff()
        down_move = low.shift(1) - low

        dm_plus = up_move.where((up_move > 0) & (up_move > down_move), 0.0)
        dm_minus = down_move.where((down_move > 0) & (down_move > up_move), 0.0)

        # Smooth using rolling sum (Wilder-style smoothing approximate)
        dm_plus_smooth = dm_plus.rolling(length, min_periods=1).sum()
        dm_minus_smooth = dm_minus.rolling(length, min_periods=1).sum()

        atr = self._compute_atr(high, low, close, length)
        # avoid division by zero
        atr_safe = atr.replace(0, np.nan)

        plus_di = 100.0 * dm_plus_smooth / atr_safe
        minus_di = 100.0 * dm_minus_smooth / atr_safe

        # DX and ADX
        denom = (plus_di + minus_di).replace(0, np.nan)
        dx = 100.0 * (plus_di - minus_di).abs() / denom
        adx = dx.rolling(length, min_periods=1).mean()

        # fill NaNs conservatively with zeros so boolean logic works
        plus_di = plus_di.fillna(0.0)
        minus_di = minus_di.fillna(0.0)
        adx = adx.fillna(0.0)

        return plus_di, minus_di, adx, atr

    def generate_signals(self, df):
        """
        Input:
            df: pandas DataFrame with columns: open, high, low, close, volume
        Output:
            entries, exits: boolean pandas Series indexed like df
        """

        # Ensure required columns exist
        required_cols = {"open", "high", "low", "close", "volume"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        volume = df["volume"].astype(float)

        # Directional indicators and ATR
        plus_di, minus_di, adx, atr = self._compute_directional_indicators(
            high, low, close, self.adx_len
        )

        # Short and long SMAs to define pullback window and trend filter
        sma_short = close.rolling(self.sma_short_len, min_periods=1).mean()
        sma_long = close.rolling(self.sma_long_len, min_periods=1).mean()

        # Volume confirmation: above median volume over recent window
        vol_med = volume.rolling(self.vol_med_len, min_periods=1).median()

        # Entry conditions (information available at close of the bar)
        # 1) Trend: ADX strong and +DI dominant
        trend_up = (adx > self.adx_entry_threshold) & (plus_di > minus_di)

        # 2) Shallow pullback: price sits slightly below short SMA but within N*ATR
        # Use atr clipped to avoid NaN issues
        atr_safe = atr.replace(0, np.nan).fillna(method="ffill").fillna(0.0)
        pullback_threshold = sma_short - (self.pullback_atr_mult * atr_safe)
        pullback = (close < sma_short) & (close >= pullback_threshold)

        # 3) Short-term recovery bar: current close higher than prior close
        recovery = close > close.shift(1)

        # 4) Volume participation: above recent median
        vol_ok = volume > vol_med

        entries = (trend_up & pullback & recovery & vol_ok).fillna(False).astype(bool)

        # Exit conditions:
        # a) Trend flips (plus_di <= minus_di)
        trend_flip = plus_di <= minus_di

        # b) ADX weakens below exit threshold
        adx_weak = adx < self.adx_exit_threshold

        # c) Longer-term trend breaks (close below sma_long)
        long_term_break = close < sma_long

        exits = (trend_flip | adx_weak | long_term_break).fillna(False).astype(bool)

        # Ensure alignment and boolean Series
        entries = pd.Series(entries, index=df.index, name="entries").astype(bool)
        exits = pd.Series(exits, index=df.index, name="exits").astype(bool)

        return entries, exits