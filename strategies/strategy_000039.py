import pandas as pd
import numpy as np

class Strategy:
    def __init__(self,
                 atr_len=14,
                 atr_median_len=100,
                 atr_spike_mult=1.6,
                 vol_med_len=50,
                 vol_low_mult=0.7,
                 vol_confirm_mult=1.2,
                 spike_lookback=6,
                 short_sma_len=10,
                 long_sma_len=50,
                 trail_high_lookback=20,
                 stop_atr_mult=2.5):
        """
        Parameters (kept transparent and modest in number):
        - atr_len: window for ATR calculation
        - atr_median_len: window to compute the "typical" ATR (median)
        - atr_spike_mult: ATR must exceed atr_median * this to count as spike
        - vol_med_len: window for median volume baseline
        - vol_low_mult: spike is considered low-volume if volume < vol_med * vol_low_mult
        - vol_confirm_mult: entry requires current volume >= vol_med * vol_confirm_mult
        - spike_lookback: how many prior bars may contain the initial spike (excluding current)
        - short_sma_len: short SMA used for entry confirmation and exit (trend-failure)
        - long_sma_len: medium/long SMA used to ensure overall uptrend
        - trail_high_lookback: lookback window to define recent high for ATR-scaled trailing exit
        - stop_atr_mult: multiplier for ATR to compute the trailing stop distance
        """
        self.atr_len = atr_len
        self.atr_median_len = atr_median_len
        self.atr_spike_mult = atr_spike_mult
        self.vol_med_len = vol_med_len
        self.vol_low_mult = vol_low_mult
        self.vol_confirm_mult = vol_confirm_mult
        self.spike_lookback = spike_lookback
        self.short_sma_len = short_sma_len
        self.long_sma_len = long_sma_len
        self.trail_high_lookback = trail_high_lookback
        self.stop_atr_mult = stop_atr_mult

    def _atr(self, df):
        # True range
        prev_close = df['close'].shift(1)
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - prev_close).abs()
        tr3 = (df['low'] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_len, min_periods=1).mean()
        return atr

    def generate_signals(self, df):
        """
        Input:
            df: pandas DataFrame with columns ['open','high','low','close','volume']
        Output:
            entries, exits: tuple of boolean pandas Series aligned to df.index
        Signals are based only on information available at each bar's close (no look-ahead).
        """
        # Basic checks
        required = {'open', 'high', 'low', 'close', 'volume'}
        if not required.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required}")

        close = df['close']
        high = df['high']
        low = df['low']
        vol = df['volume']

        # Indicators
        atr = self._atr(df)
        atr_median = atr.rolling(self.atr_median_len, min_periods=1).median()

        vol_med = vol.rolling(self.vol_med_len, min_periods=1).median()

        short_sma = close.rolling(self.short_sma_len, min_periods=1).mean()
        long_sma = close.rolling(self.long_sma_len, min_periods=1).mean()

        # Identify past volatility spikes that occurred on low volume (lack of participation)
        spike = (atr > (atr_median * self.atr_spike_mult)) & (vol < (vol_med * self.vol_low_mult))

        # We only care about spikes that occurred before the current bar (no look-ahead),
        # so create a lagged spike series
        spike_lag = spike.shift(1).fillna(False)

        # For entries we require that at least one such spike exists within the prior `spike_lookback` bars
        # (this uses only historical information at each bar)
        spike_recent = spike_lag.rolling(self.spike_lookback, min_periods=1).max().astype(bool)

        # Entry conditions at current close:
        # 1) There was a recent low-volume volatility spike (spike_recent)
        # 2) The market is in a medium-term uptrend (close > long_sma)
        # 3) Short-term recovery: today's close is higher than yesterday's close
        # 4) Volume confirmation: today's volume is elevated vs recent median
        # 5) Price is above short_sma (additional trend confirmation)
        cond_trend = close > long_sma
        cond_recovery = close > close.shift(1)
        cond_vol_confirm = vol >= (vol_med * self.vol_confirm_mult)
        cond_price_above_short = close > short_sma

        entries = spike_recent & cond_trend & cond_recovery & cond_vol_confirm & cond_price_above_short

        # Exits:
        # 1) Short-term trend failure: close closes below short_sma
        # 2) Volatility-scaled trailing stop: close falls more than stop_atr_mult * ATR from the recent N-bar high
        recent_high = close.rolling(self.trail_high_lookback, min_periods=1).max()
        trail_stop_level = recent_high - (atr * self.stop_atr_mult)
        cond_short_sma_break = close < short_sma
        cond_trail_breach = close < trail_stop_level

        exits = cond_short_sma_break | cond_trail_breach

        # Ensure boolean Series aligned to df.index and no NaNs (NaNs -> False)
        entries = entries.reindex(df.index).fillna(False).astype(bool)
        exits = exits.reindex(df.index).fillna(False).astype(bool)

        return entries, exits