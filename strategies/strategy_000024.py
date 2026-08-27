import numpy as np
import pandas as pd

class Strategy:
    def __init__(self,
                 sv_ema_short=5,
                 sv_ema_long=20,
                 vol_med_window=20,
                 trend_sma=50,
                 close_top_frac=0.6,
                 eps=1e-9):
        """
        Parameters:
        - sv_ema_short, sv_ema_long: EMAs for range-weighted signed-volume (acceleration signal)
        - vol_med_window: window for median-volume confirmation
        - trend_sma: window for medium-term trend confirmation (price above its SMA)
        - close_top_frac: required fraction of the bar range that close must be above (close near high)
        - eps: small number to avoid division by zero
        """
        self.sv_ema_short = sv_ema_short
        self.sv_ema_long = sv_ema_long
        self.vol_med_window = vol_med_window
        self.trend_sma = trend_sma
        self.close_top_frac = close_top_frac
        self.eps = eps

    def generate_signals(self, df):
        """
        Input:
        - df: pandas DataFrame with columns ['open','high','low','close','volume']

        Output:
        - entries: boolean pandas Series indicating entry signals at close of bar
        - exits: boolean pandas Series indicating exit signals at close of bar
        """
        # Ensure input has required columns
        required = {'open', 'high', 'low', 'close', 'volume'}
        if not required.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required}")

        o = df['open'].astype(float)
        h = df['high'].astype(float)
        l = df['low'].astype(float)
        c = df['close'].astype(float)
        v = df['volume'].astype(float)

        # Range-safe denominator
        rng = (h - l).replace(0, self.eps) + self.eps

        # Range-weighted signed volume: (close - open) normalized by range, scaled by volume.
        # Interpretation: if close is near high and price moved up during the bar with substantial volume,
        # this metric will be strongly positive (aggressive buying within the bar).
        signed_vol = ((c - o) / rng) * v

        # Short and long EMAs of signed volume to detect acceleration/momentum in buyer aggression.
        sv_ema_short = signed_vol.ewm(span=self.sv_ema_short, adjust=False).mean()
        sv_ema_long = signed_vol.ewm(span=self.sv_ema_long, adjust=False).mean()

        # Volume participation: require volume above recent median
        vol_med = v.rolling(window=self.vol_med_window, min_periods=1).median()

        # Medium-term trend: price above SMA
        sma_trend = c.rolling(window=self.trend_sma, min_periods=1).mean()

        # Close position inside the bar (0 = at low, 1 = at high). Require close to be on upper part of range.
        close_pos = (c - l) / rng

        # Entry conditions:
        # 1) Short EMA of range-weighted signed volume > long EMA (positive acceleration)
        # 2) Short EMA > 0 (net positive signed volume momentum)
        # 3) Volume participation: current volume > recent median
        # 4) Price in medium-term uptrend: close > SMA(trend)
        # 5) Close is in upper fraction of the bar's range (buyer-controlled close)
        entry_cond = (
            (sv_ema_short > sv_ema_long) &
            (sv_ema_short > 0) &
            (v > vol_med) &
            (c > sma_trend) &
            (close_pos >= self.close_top_frac)
        )

        # Exit conditions:
        # Exit when signed-volume momentum flips (short EMA below long EMA) OR trend breaks (close below SMA)
        exit_cond = (
            (sv_ema_short < sv_ema_long) |
            (c < sma_trend)
        )

        # Ensure boolean Series aligned with df index and no NaNs (NaNs -> False)
        entries = entry_cond.reindex(df.index).fillna(False).astype(bool)
        exits = exit_cond.reindex(df.index).fillna(False).astype(bool)

        return entries, exits