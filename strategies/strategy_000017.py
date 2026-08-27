import pandas as pd
import numpy as np

class Strategy:
    def generate_signals(self, df):
        """
        Parameters (internal defaults):
         - zone_window: lookback for the value-zone quantiles (uses prior window for entry thresholds)
         - vol_window: lookback to measure recent volume concentration and average volume
         - width_thresh: maximum normalized (by recent std) zone width to consider the market "tight"
         - accumulation_ratio_thresh: fraction of recent volume occurring inside the prior value zone required to signal accumulation
         - breakout_volume_mult: multiplier of recent average volume required on the breakout bar
         - sma_long_period: longer SMA used as a trend / exit filter

        Returns:
         entries, exits: boolean pandas Series aligned with df.index
        """
        close = df['close']
        volume = df['volume']

        # Parameters (kept simple and interpretable)
        zone_window = 20
        vol_window = 20
        width_thresh = 0.35             # zone width must be <= 35% of recent std
        accumulation_ratio_thresh = 0.5 # at least 50% of recent volume concentrated in zone
        breakout_volume_mult = 1.0      # breakout bar volume must be >= recent average
        sma_long_period = 50

        # --- Compute prior value-zone bounds (quantiles) using only past data (shifted) ---
        # Use the previous zone_window bars (exclude current bar) as the reference zone for entry tests
        prev_lower = close.rolling(zone_window, min_periods=zone_window).quantile(0.2).shift(1)
        prev_upper = close.rolling(zone_window, min_periods=zone_window).quantile(0.8).shift(1)
        prev_std = close.rolling(zone_window, min_periods=zone_window).std().shift(1)

        # Normalized width to detect a "tight" zone (avoid periods of high volatility)
        normalized_width_prev = (prev_upper - prev_lower) / (prev_std + 1e-12)

        # --- Measure recent volume concentrated inside the prior value zone (accumulation) ---
        # For each bar, consider it "in-zone" if its close lies within the prior zone bounds.
        # This uses only information that would have been available at that bar's close.
        in_zone = (close >= prev_lower) & (close <= prev_upper)
        in_zone = in_zone.fillna(False)

        accumulation = in_zone.astype(float) * volume

        # Rolling sums over vol_window: how much of the recent total volume occurred inside the zone?
        accumulation_roll = accumulation.rolling(vol_window, min_periods=1).sum()
        total_vol_roll = volume.rolling(vol_window, min_periods=1).sum()

        # Fraction of recent volume that was inside the zone
        accumulation_ratio = accumulation_roll / (total_vol_roll + 1e-12)

        # Recent average volume to require reasonable participation on breakout
        recent_vol_avg_for_break = volume.rolling(vol_window, min_periods=1).mean().shift(1)

        # --- Entry condition (all must hold at close of bar to generate entry) ---
        # 1) Zone was tight relative to recent volatility
        c_tight_zone = normalized_width_prev < width_thresh

        # 2) Significant concentration of recent volume inside the prior zone (accumulation)
        c_accumulation = accumulation_ratio > accumulation_ratio_thresh

        # 3) Price breaks above the prior zone's upper bound (breakout) — prev_upper excludes current bar
        c_breakout = close > prev_upper

        # 4) Breakout has at-least-average volume (suggesting participation)
        c_breakout_volume = volume >= (recent_vol_avg_for_break * breakout_volume_mult)

        entries = (c_tight_zone & c_accumulation & c_breakout & c_breakout_volume).fillna(False).astype(bool)

        # --- Exit conditions ---
        # Use current-window lower quantile and current long SMA as exit triggers (decided at close)
        current_lower = close.rolling(zone_window, min_periods=1).quantile(0.2)
        sma_long = close.rolling(sma_long_period, min_periods=1).mean()

        # Exit if price falls back below the current zone lower bound (failed breakout / break of support)
        # OR if price breaks the longer-term SMA (trend failure)
        exits = ((close < current_lower) | (close < sma_long)).fillna(False).astype(bool)

        # Avoid generating an immediate exit on the same bar as an entry (prevents zero-length trades)
        exits = exits & (~entries)

        # Return boolean Series aligned with original index
        entries = pd.Series(entries, index=df.index)
        exits = pd.Series(exits, index=df.index)

        return entries, exits