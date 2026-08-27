import pandas as pd
import numpy as np

class Strategy:
    def __init__(self,
                 short=6,
                 medium=24,
                 sma_short=20,
                 sma_medium=50,
                 atr_len=14,
                 accel_threshold=0.15,
                 eps=1e-9):
        """
        Parameters are chosen to be simple and interpretable:
        - short / medium: windows for detecting acceleration
        - sma_short / sma_medium: trend filters for entries/exits
        - atr_len: window for ATR-like normalization
        - accel_threshold: minimum normalized-body acceleration to trigger entry
        """
        self.short = short
        self.medium = medium
        self.sma_short = sma_short
        self.sma_medium = sma_medium
        self.atr_len = atr_len
        self.accel_threshold = accel_threshold
        self.eps = eps

    def generate_signals(self, df):
        """
        Input:
          df: pandas DataFrame with columns ['open','high','low','close','volume']
        Returns:
          entries, exits: boolean pandas Series aligned with df.index
        """
        o = df['open']
        h = df['high']
        l = df['low']
        c = df['close']
        v = df['volume']

        # Basic sanity: ensure floats
        o = o.astype(float)
        h = h.astype(float)
        l = l.astype(float)
        c = c.astype(float)
        v = v.astype(float)

        # Simple ATR-like normalization (uses only current and past bars)
        tr = (h - l).abs()
        atr = tr.rolling(self.atr_len, min_periods=1).mean()

        # Candle body (absolute) normalized by ATR to compare across regimes
        body = (c - o).abs()
        norm_body = body / (atr + self.eps)

        # Short and medium average normalized body (measures intensity & its change)
        nb_short = norm_body.rolling(self.short, min_periods=1).mean()
        nb_med = norm_body.rolling(self.medium, min_periods=1).mean()

        # Acceleration: short-term average body strength minus medium-term baseline
        accel = nb_short - nb_med

        # Participation: fraction of up-bars (close > open) in short and medium windows
        up = (c > o).astype(float)
        up_frac_short = up.rolling(self.short, min_periods=1).mean()
        up_frac_med = up.rolling(self.medium, min_periods=1).mean()

        # Volume confirmation: short vs medium volume averages
        vol_short = v.rolling(self.short, min_periods=1).mean()
        vol_med = v.rolling(self.medium, min_periods=1).mean()

        # Trend filters: short and medium simple moving averages of price
        sma_short = c.rolling(self.sma_short, min_periods=1).mean()
        sma_med = c.rolling(self.sma_medium, min_periods=1).mean()

        # Entry rule:
        # 1) Acceleration exceeds threshold (strengthening normalized bodies)
        # 2) Short-term up-bar fraction is higher than its medium baseline (more up-bars)
        # 3) Short-term volume exceeds medium-term average (participation)
        # 4) Price is above the short SMA (aligned with short-term trend)
        entry_cond = (
            (accel > self.accel_threshold) &
            (up_frac_short > up_frac_med) &
            (vol_short > vol_med) &
            (c > sma_short)
        )

        # Exit rule:
        # Exit when acceleration collapses (<= 0) or medium-term trend fails (close < medium SMA).
        # This keeps exits simple and interpretable: loss of accelerating participation or trend break.
        exit_cond = (
            (accel <= 0) |
            (c < sma_med)
        )

        # Ensure boolean Series aligned with input index
        entries = entry_cond.reindex(df.index, fill_value=False).astype(bool)
        exits = exit_cond.reindex(df.index, fill_value=False).astype(bool)

        # Clean initial warm-up area (optional): avoid signaling before sufficient medium window data
        warmup = max(self.medium, self.sma_medium, self.atr_len)
        if warmup is None:
            warmup = 0
        entries.iloc[:warmup] = False
        exits.iloc[:warmup] = False

        return entries, exits