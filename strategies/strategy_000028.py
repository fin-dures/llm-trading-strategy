import numpy as np
import pandas as pd

class Strategy:
    """
    Strategy implements a breakout-after-quiet-accumulation hypothesis using only OHLCV data.

    Signals:
      - Entry: On the close of a bar when:
          * Close > prior N-bar high (breakout above recent resistance)
          * The "close-position" ( (close-low)/(high-low) ) has shown range contraction
            (short-term std << long-term std) indicating compressed behavior
          * Recent mean close-position is above its consolidation median (buyers bias)
          * Market is in a longer-term uptrend (close > sma_long and sma_long slope > 0)
          * Current volume exceeds the recent median volume (participation confirms breakout)
      - Exit: On the close of a bar when any of:
          * Close falls below the medium SMA (trend breakdown)
          * Close falls below the long SMA (strong trend failure)
          * Recent mean close-position drops below its consolidation median (buyers faded)
    """
    def __init__(self,
                 sma_long=100,
                 sma_med=50,
                 consolidation_window=30,
                 breakout_lookback=10,
                 inc_cp_window=6,
                 vol_med_window=30,
                 contraction_thresh=0.6,
                 sma_slope_lookback=5,
                 eps=1e-9):
        # parameters (kept simple and interpretable)
        self.sma_long = sma_long
        self.sma_med = sma_med
        self.consolidation_window = consolidation_window
        self.breakout_lookback = breakout_lookback
        self.inc_cp_window = inc_cp_window
        self.vol_med_window = vol_med_window
        self.contraction_thresh = contraction_thresh
        self.sma_slope_lookback = sma_slope_lookback
        self.eps = eps

    def generate_signals(self, df):
        """
        df: pandas DataFrame with columns ['open','high','low','close','volume']
        Returns:
          entries, exits -- boolean pandas Series aligned with df.index
        """
        # Basic validations
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        volume = df['volume'].astype(float)

        # Long-term and medium SMAs
        sma_long = close.rolling(self.sma_long, min_periods=1).mean()
        sma_med = close.rolling(self.sma_med, min_periods=1).mean()

        # Slope of long SMA (to ensure trend is rising)
        sma_long_slope = sma_long.diff(self.sma_slope_lookback)

        # Close-position inside each bar: 0 at low, 1 at high (handle zero-range)
        range_ = (high - low).replace(0, np.nan)
        close_pos = ((close - low) / (range_ + self.eps)).fillna(0.5)  # if flat bar, neutral 0.5

        # Measure contraction: short vs long std of close-position
        short_std = close_pos.rolling(self.consolidation_window, min_periods=1).std()
        long_std = close_pos.rolling(2 * self.consolidation_window, min_periods=1).std().replace(0, np.nan)
        contraction = (short_std / long_std).fillna(1.0)  # default no contraction when undefined

        # Recent mean close-position and consolidation baseline median (use past data only)
        recent_mean_cp = close_pos.rolling(self.inc_cp_window, min_periods=1).mean()
        # median of the consolidation window, shifted by 1 to ensure it's based on prior bars only
        consolidation_median_cp = close_pos.rolling(self.consolidation_window, min_periods=1).median().shift(1)

        # Prior N-bar high (exclude current bar to avoid look-ahead)
        prior_high = high.rolling(self.breakout_lookback, min_periods=1).max().shift(1)

        # Recent median volume (exclude current bar for baseline)
        vol_median = volume.rolling(self.vol_med_window, min_periods=1).median().shift(1)

        # Entry conditions (all evaluated at close of current bar)
        cond_breakout = close > prior_high
        cond_contraction = contraction < self.contraction_thresh
        cond_buyer_bias = recent_mean_cp > consolidation_median_cp
        cond_trend = (close > sma_long) & (sma_long_slope > 0)
        cond_volume = volume > vol_median

        entries = cond_breakout & cond_contraction & cond_buyer_bias & cond_trend & cond_volume

        # Exit conditions:
        # - trend failure: close below medium or long SMA
        # - buyer bias evaporates: recent_mean_cp falls below consolidation median (use current values)
        exits = (close < sma_med) | (close < sma_long) | (recent_mean_cp < consolidation_median_cp)

        # Ensure boolean Series aligned with df.index and no NaNs (initial periods)
        entries = entries.fillna(False).astype(bool)
        exits = exits.fillna(False).astype(bool)

        # Return two boolean Series: entries and exits
        return entries, exits