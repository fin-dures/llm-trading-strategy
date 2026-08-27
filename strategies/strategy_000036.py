import numpy as np
import pandas as pd

class Strategy:
    def __init__(self,
                 n_high=20,
                 sma_trend=50,
                 sma_short=10,
                 atr_period=14,
                 atr_contraction_len=8,
                 vol_decline_len=6,
                 vol_median_period=50,
                 atr_stop_mul=2.0,
                 max_hold=20):
        """
        Parameters are chosen to keep the strategy interpretable and simple:
        - n_high: lookback to define the prior swing-high that breakout must exceed
        - sma_trend: medium-term trend filter (price must be above this SMA)
        - sma_short: short-term trend used as an exit trigger
        - atr_period: period for ATR calculation
        - atr_contraction_len: window to test ATR slope (negative => contraction)
        - vol_decline_len: window to test volume slope (negative => declining volume)
        - vol_median_period: period to compute baseline volume median (breakout volume must be <= this)
        - atr_stop_mul: multiply ATR_at_entry to set a fixed stop-loss distance below entry
        - max_hold: maximum bars to hold before force exit
        """
        self.n_high = n_high
        self.sma_trend = sma_trend
        self.sma_short = sma_short
        self.atr_period = atr_period
        self.atr_contraction_len = atr_contraction_len
        self.vol_decline_len = vol_decline_len
        self.vol_median_period = vol_median_period
        self.atr_stop_mul = atr_stop_mul
        self.max_hold = max_hold

    def _atr(self, high, low, close, period):
        # True range
        prev_close = close.shift(1)
        tr1 = (high - low).abs()
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period, min_periods=period).mean()
        return atr

    def _rolling_slope(self, series, window):
        # Compute slope (linear regression) over each rolling window.
        # Returns slope aligned to the window's last element (no shift).
        if window < 2:
            return pd.Series(np.nan, index=series.index)
        x = np.arange(window)
        x_mean = x.mean()
        x_demean = x - x_mean
        denom = (x_demean * x_demean).sum()
        def slope_func(arr):
            y = np.asarray(arr, dtype=float)
            if np.any(np.isnan(y)):
                return np.nan
            y_mean = y.mean()
            num = ((x_demean) * (y - y_mean)).sum()
            return num / denom
        return series.rolling(window, min_periods=window).apply(slope_func, raw=True)

    def generate_signals(self, df):
        """
        Input: df with columns ['open','high','low','close','volume']
        Returns: entries, exits  -> boolean pandas Series
        """
        # Ensure we work on a copy to avoid modifying user df
        data = df[['open', 'high', 'low', 'close', 'volume']].copy()
        close = data['close']
        high = data['high']
        low = data['low']
        vol = data['volume']

        # Indicators
        atr = self._atr(high, low, close, self.atr_period)

        sma_trend = close.rolling(self.sma_trend, min_periods=self.sma_trend).mean()
        sma_short = close.rolling(self.sma_short, min_periods=self.sma_short).mean()

        # Prior N-bar high (exclude current bar)
        prior_high = high.rolling(self.n_high, min_periods=self.n_high).max().shift(1)

        # ATR slope over contraction window (negative means ATR is falling)
        atr_slope = self._rolling_slope(atr, self.atr_contraction_len)

        # Volume slope: negative means volume declining over vol_decline_len
        vol_slope = self._rolling_slope(vol, self.vol_decline_len)

        # Baseline median volume (exclude current bar to avoid peeking)
        vol_median = vol.rolling(self.vol_median_period, min_periods=self.vol_median_period).median().shift(1)

        # Entry boolean preliminary conditions (vectorized)
        cond_breakout = close > prior_high  # close strictly above prior N-bar high
        cond_low_vol_breakout = vol <= vol_median  # breakout occurs on below-baseline volume
        cond_atr_contract = atr_slope < 0
        cond_vol_decline = vol_slope < 0
        cond_trend = close > sma_trend

        entry_cond = cond_breakout & cond_low_vol_breakout & cond_atr_contract & cond_vol_decline & cond_trend

        # Prepare empty boolean series for final entries and exits
        entries = pd.Series(False, index=df.index)
        exits = pd.Series(False, index=df.index)

        # We'll iterate to implement entry-dependent exits (ATR-stop referenced to entry ATR and entry price + max_hold)
        in_position = False
        entry_price = np.nan
        entry_atr = np.nan
        entry_idx = None
        hold_count = 0

        # To avoid using future info, when we decide entry at bar i we use close[i] and atr[i] etc.
        for i, idx in enumerate(df.index):
            # current values
            c_close = close.iloc[i]
            c_sma_short = sma_short.iloc[i]
            c_atr = atr.iloc[i]
            c_entry_possible = bool(entry_cond.iloc[i])

            if not in_position:
                if c_entry_possible and not np.isnan(c_atr) and not np.isnan(c_sma_short):
                    # Enter at close of this bar
                    entries.iloc[i] = True
                    in_position = True
                    entry_price = c_close
                    entry_atr = c_atr
                    entry_idx = i
                    hold_count = 0
                # else remain flat
            else:
                # We are in position: evaluate exits using only information up to current bar
                hold_count += 1

                # Exit conditions:
                # 1) short SMA breakdown: close < sma_short
                exit_short_sma = False
                if not np.isnan(c_sma_short):
                    exit_short_sma = c_close < c_sma_short

                # 2) ATR-based stop referencing ATR at entry
                exit_atr_stop = False
                if not np.isnan(entry_atr):
                    exit_atr_stop = c_close <= (entry_price - self.atr_stop_mul * entry_atr)

                # 3) maximum holding period reached
                exit_max_hold = hold_count >= self.max_hold

                if exit_short_sma or exit_atr_stop or exit_max_hold:
                    exits.iloc[i] = True
                    in_position = False
                    entry_price = np.nan
                    entry_atr = np.nan
                    entry_idx = None
                    hold_count = 0
                else:
                    # remain in position
                    pass

        # Ensure dtype boolean
        entries = entries.astype(bool)
        exits = exits.astype(bool)
        return entries, exits