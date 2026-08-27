import pandas as pd
import numpy as np

class Strategy:
    def __init__(self,
                 wick_window=8,            # how many prior bars we look for lower-wick accumulation
                 wick_ratio_th=0.55,       # lower-wick must be at least this fraction of the bar range
                 wick_count_th=3,          # minimum number of wick-accumulation bars in the window
                 vol_med_window=50,        # window for baseline volume median
                 vol_spike_mult=1.3,       # current volume must exceed this * baseline median
                 breakout_lookback=5,      # prior N bars for breakout high (exclude current bar)
                 sma_short=20,             # short SMA for confirmation (price should be above)
                 sma_long=50,              # long SMA for trend-break exit
                 atr_period=14,            # ATR period for volatility and stops
                 atr_stop_mult=3.0,        # multiple of ATR for fixed stop below entry
                 max_hold=20):             # maximum holding bars before forced exit
        self.wick_window = wick_window
        self.wick_ratio_th = wick_ratio_th
        self.wick_count_th = wick_count_th
        self.vol_med_window = vol_med_window
        self.vol_spike_mult = vol_spike_mult
        self.breakout_lookback = breakout_lookback
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.max_hold = max_hold

    def _compute_atr(self, high, low, close, period):
        # True range
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period, min_periods=1).mean()
        return atr

    def generate_signals(self, df):
        """
        Input:
            df: pandas DataFrame with columns: open, high, low, close, volume
        Returns:
            entries, exits: boolean pandas Series aligned with df.index
        """
        # Ensure required columns exist
        required = {'open', 'high', 'low', 'close', 'volume'}
        if not required.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required}")

        o = df['open']
        h = df['high']
        l = df['low']
        c = df['close']
        v = df['volume']

        n = len(df)
        entries = pd.Series(False, index=df.index)
        exits = pd.Series(False, index=df.index)

        # Basic indicators
        atr = self._compute_atr(h, l, c, self.atr_period)

        sma_short = c.rolling(self.sma_short, min_periods=1).mean()
        sma_long = c.rolling(self.sma_long, min_periods=1).mean()

        # Volume baseline (use past values, exclude current for baseline)
        vol_med = v.rolling(self.vol_med_window, min_periods=1).median().shift(1)

        # Prior N-bar high for breakout (exclude current bar)
        prior_high = h.shift(1).rolling(self.breakout_lookback, min_periods=1).max()

        # Lower-wick ratio for prior bars (exclude current bar to detect accumulation)
        range_ = (h - l).replace(0, np.nan)  # avoid division by zero
        lower_wick = (c - l)
        lower_wick_ratio = (lower_wick / range_).fillna(0)

        # Candidate wick accumulation count over the prior window
        wick_flag = (lower_wick_ratio.shift(1) >= self.wick_ratio_th) & (v.shift(1) > vol_med)
        wick_count = wick_flag.rolling(self.wick_window, min_periods=1).sum()

        # Entry boolean conditions (vectorized possible signals)
        cond_wick_accum = wick_count >= self.wick_count_th                # accumulation detected in prior bars
        cond_vol_spike = v > (vol_med * self.vol_spike_mult)             # current volume spike vs baseline
        cond_breakout = c > prior_high                                    # current close breaks prior N-bar high
        cond_above_short = c > sma_short                                  # price above short SMA (trend confirmation)
        cond_low_vol_regime = atr < atr.rolling(200, min_periods=1).median()  # volatility contracted vs long median

        # Combine entry signal candidates (these use only current and past data)
        raw_entry = cond_wick_accum & cond_vol_spike & cond_breakout & cond_above_short & cond_low_vol_regime

        # We'll simulate positions to produce exits that depend on entry price and holding time
        in_position = False
        entry_idx = None
        entry_price = None
        entry_atr = None
        highest_close_since_entry = None

        # Convert index to integer positions for iteration
        for i in range(n):
            idx = df.index[i]
            # Entry decision: must be true on this bar and we must not already be in a position
            if (not in_position) and bool(raw_entry.iloc[i]):
                entries.iloc[i] = True
                in_position = True
                entry_idx = i
                entry_price = c.iloc[i]
                entry_atr = atr.iloc[i] if not np.isnan(atr.iloc[i]) else 0.0
                highest_close_since_entry = c.iloc[i]
                # continue to next bar (we don't exit on same bar as entry)
                continue

            # If in position, update highest close and evaluate exit conditions
            if in_position:
                # update highest-close tracking
                if c.iloc[i] > highest_close_since_entry:
                    highest_close_since_entry = c.iloc[i]

                # Exit conditions:
                # 1) long-term trend break: close below long SMA
                exit_trend = c.iloc[i] < sma_long.iloc[i]

                # 2) ATR-based stop from entry (fixed one-way stop)
                if entry_atr is None:
                    entry_atr = atr.iloc[i] if not np.isnan(atr.iloc[i]) else 0.0
                exit_atr_stop = False
                if entry_atr > 0:
                    stop_price = entry_price - (self.atr_stop_mult * entry_atr)
                    exit_atr_stop = c.iloc[i] <= stop_price

                # 3) Maximum holding time
                held_bars = i - entry_idx
                exit_max_hold = held_bars >= self.max_hold

                # If any exit triggers, mark exit on this bar and clear position
                if exit_trend or exit_atr_stop or exit_max_hold:
                    exits.iloc[i] = True
                    in_position = False
                    entry_idx = None
                    entry_price = None
                    entry_atr = None
                    highest_close_since_entry = None
                    continue

        return entries, exits