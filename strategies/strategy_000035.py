import numpy as np
import pandas as pd

class Strategy:
    def __init__(self, 
                 sma_long=50,
                 sma_short=20,
                 atr_len=14,
                 atr_med_len=100,
                 ent_win=20,
                 ent_bins=8,
                 ent_thresh=0.35,
                 close_pos_mean_thresh=0.72,
                 prior_high_n=10,
                 vol_med_len=50,
                 vol_multiplier=1.3,
                 consecutive_down_bars=3):
        # Parameters (kept modest in number for interpretability)
        self.sma_long = sma_long
        self.sma_short = sma_short
        self.atr_len = atr_len
        self.atr_med_len = atr_med_len
        self.ent_win = ent_win
        self.ent_bins = ent_bins
        self.ent_thresh = ent_thresh
        self.close_pos_mean_thresh = close_pos_mean_thresh
        self.prior_high_n = prior_high_n
        self.vol_med_len = vol_med_len
        self.vol_multiplier = vol_multiplier
        self.consecutive_down_bars = consecutive_down_bars

    @staticmethod
    def _shannon_entropy_norm(arr, bins=8):
        # arr expected to be in [0,1]; ignore NaNs
        a = arr[~np.isnan(arr)]
        if a.size == 0:
            return np.nan
        hist, _ = np.histogram(a, bins=bins, range=(0.0, 1.0))
        s = hist.sum()
        if s == 0:
            return np.nan
        probs = hist.astype(float) / s
        probs = probs[probs > 0]
        ent = -np.sum(probs * np.log2(probs))
        max_ent = np.log2(bins) if bins > 1 else 1.0
        return float(ent / max_ent) if max_ent > 0 else 0.0

    def generate_signals(self, df):
        """
        Input df must contain: open, high, low, close, volume
        Returns two boolean pandas Series: entries, exits
        """
        # Work on a copy to avoid modifying original df
        data = df.copy()
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']

        # Simple moving averages for trend filters
        sma_long = close.rolling(self.sma_long, min_periods=1).mean()
        sma_short = close.rolling(self.sma_short, min_periods=1).mean()

        # ATR (classic)
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_len, min_periods=1).mean()
        atr_med = atr.rolling(self.atr_med_len, min_periods=1).median()

        # Close position within the bar [0..1] (avoid division by zero)
        rng = (high - low).replace(0, np.nan)
        close_pos = ((close - low) / rng).clip(0, 1).fillna(0.5)

        # Rolling normalized Shannon entropy of close positions
        ent_series = close_pos.rolling(self.ent_win, min_periods=2).apply(
            lambda x: self._shannon_entropy_norm(x, bins=self.ent_bins), raw=True
        )

        # Mean close_pos over the same window (to check whether closes cluster near highs)
        close_pos_mean = close_pos.rolling(self.ent_win, min_periods=1).mean()

        # Prior N-bar high (exclude current bar)
        prior_high = high.shift(1).rolling(self.prior_high_n, min_periods=1).max()

        # Volume baseline
        vol_med = volume.rolling(self.vol_med_len, min_periods=1).median()

        # Entry conditions:
        # 1) Price above long-term trend
        cond_trend = close > sma_long

        # 2) Low volatility regime (ATR below its recent median)
        cond_low_vol = atr < atr_med

        # 3) Low entropy (concentration) and cluster near highs
        cond_entropy = ent_series < self.ent_thresh
        cond_closepos = close_pos_mean > self.close_pos_mean_thresh

        # 4) Current close is a breakout above prior N-bar high (prior highs exclude current)
        cond_breakout = close > prior_high

        # 5) Volume participation
        cond_vol = volume > (vol_med * self.vol_multiplier)

        # 6) Momentum confirmation (close higher than prior close)
        cond_mom = close > close.shift(1)

        # Combine for entry
        entry_mask = cond_trend & cond_low_vol & cond_entropy & cond_closepos & cond_breakout & cond_vol & cond_mom

        # Exit conditions:
        # A) Price falls below short SMA (short-term trend failure)
        exit_a = close < sma_short
        # B) Price falls below long SMA (strong trend failure)
        exit_b = close < sma_long
        # C) A short run of consecutive down closes (loss of momentum)
        down = (close < close.shift(1)).astype(int)
        down_run = down.rolling(self.consecutive_down_bars, min_periods=1).sum() >= self.consecutive_down_bars

        exit_mask = exit_a | exit_b | down_run

        # Ensure boolean Series aligned with df index, NaNs -> False
        entries = entry_mask.reindex(df.index).fillna(False).astype(bool)
        exits = exit_mask.reindex(df.index).fillna(False).astype(bool)

        return entries, exits