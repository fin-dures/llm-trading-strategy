import numpy as np
import pandas as pd

class Strategy:
    def __init__(
        self,
        slope_lookback=8,         # window for regression slope (bars)
        atr_period=14,            # ATR period for volatility normalization
        sma_short=20,             # short SMA used as a trend filter and exit trigger
        vol_baseline_period=100,  # period to compute baseline median volume
        vol_multiplier=1.1,       # required multiple of baseline volume for confirmation
        consecutive_bars=3,       # required consecutive up-bars for persistence
        slope_threshold=0.05      # normalized slope threshold (slope / ATR > this)
    ):
        self.slope_lookback = int(slope_lookback)
        self.atr_period = int(atr_period)
        self.sma_short = int(sma_short)
        self.vol_baseline_period = int(vol_baseline_period)
        self.vol_multiplier = float(vol_multiplier)
        self.consecutive_bars = int(consecutive_bars)
        self.slope_threshold = float(slope_threshold)

    def _validate_df(self, df):
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

    def _compute_atr(self, df):
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period, min_periods=1).mean()
        return atr

    def _regression_slope(self, series, window):
        # compute slope (price change per bar) via linear regression on rolling window
        # returns series aligned to right edge (uses only past & current bars)
        x = np.arange(window)
        # Precompute elements for slope formula to speed up if desired; here we use polyfit via rolling.apply
        def slope_func(y):
            # y is a 1-D ndarray of length window
            # If any NaN, return NaN
            if np.isnan(y).any():
                return np.nan
            # polyfit for degree 1
            m = np.polyfit(x, y, 1)[0]
            return m
        return series.rolling(window, min_periods=window).apply(slope_func, raw=True)

    def generate_signals(self, df):
        """
        Input:
            df : pandas.DataFrame with columns 'open','high','low','close','volume'
        Output:
            entries, exits : two boolean pandas.Series aligned with df.index
        """
        self._validate_df(df)
        df = df.copy()

        # Compute ATR for volatility normalization
        df["atr"] = self._compute_atr(df)

        # Short SMA used as trend confirmation and exit trigger
        df["sma_short"] = df["close"].rolling(self.sma_short, min_periods=1).mean()

        # Regression slope (price change per bar) over the lookback window
        df["slope"] = self._regression_slope(df["close"], self.slope_lookback)

        # Normalize slope by current ATR to get volatility-adjusted momentum
        # Avoid division by zero: if atr==0, result is NaN
        df["slope_norm"] = df["slope"] / df["atr"]

        # Volume confirmation: average volume over slope window vs baseline median
        df["avg_vol_window"] = df["volume"].rolling(self.slope_lookback, min_periods=1).mean()
        df["vol_baseline"] = df["volume"].rolling(self.vol_baseline_period, min_periods=1).median()
        df["vol_confirm"] = df["avg_vol_window"] > (df["vol_baseline"] * self.vol_multiplier)

        # Persistence: count consecutive up-bars ending at current bar
        up = (df["close"] > df["close"].shift(1)).astype(int)
        df["consec_up_count"] = up.rolling(self.consecutive_bars, min_periods=1).sum()
        df["consec_up_ok"] = df["consec_up_count"] >= self.consecutive_bars

        # Entry condition:
        #  - volatility-adjusted slope exceeds threshold
        #  - price is above short SMA (trend filter)
        #  - volume confirms (average recent volume > baseline * multiplier)
        #  - there are at least `consecutive_bars` consecutive up-bars
        entry_cond = (
            (df["slope_norm"] > self.slope_threshold)
            & (df["close"] > df["sma_short"])
            & (df["vol_confirm"])
            & (df["consec_up_ok"])
        )

        # Exit condition:
        #  - momentum fades (slope_norm <= 0) OR price falls below the short SMA
        exit_cond = (
            (df["slope_norm"] <= 0)
            | (df["close"] < df["sma_short"])
        )

        # Ensure boolean Series and align index
        entries = entry_cond.fillna(False).astype(bool)
        exits = exit_cond.fillna(False).astype(bool)

        return entries, exits