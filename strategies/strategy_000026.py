import numpy as np
import pandas as pd

class Strategy:
    def __init__(self,
                 atr_len=14,
                 vol_med_lookback=20,
                 volume_multiplier=1.2,
                 gap_atr_mult=0.25,
                 recovery_fraction=0.6,
                 sma_long_len=50,
                 atr_stop_mult=1.5,
                 atr_take_mult=3.0,
                 max_hold=10):
        """
        Parameters chosen to be simple defaults; generate_signals does not rely on any external data.
        """
        self.atr_len = atr_len
        self.vol_med_lookback = vol_med_lookback
        self.volume_multiplier = volume_multiplier
        self.gap_atr_mult = gap_atr_mult
        self.recovery_fraction = recovery_fraction
        self.sma_long_len = sma_long_len
        self.atr_stop_mult = atr_stop_mult
        self.atr_take_mult = atr_take_mult
        self.max_hold = max_hold

    def _compute_atr(self, high, low, close):
        # True range
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_len, min_periods=1).mean()
        return atr

    def generate_signals(self, df):
        """
        Accepts dataframe with columns: open, high, low, close, volume
        Returns two boolean pd.Series: entries, exits (aligned with df.index)
        Entry rule (evaluated at close of bar t):
          - bar opens below prior close by at least gap_atr_mult * ATR (to avoid micro-gaps)
          - the bar is bullish (close > open) and closes at least recovery_fraction of the way from open toward prior close
            (i.e., a strong gap-fill recovery)
          - volume on bar t > rolling median(volume, vol_med_lookback) * volume_multiplier
        Exit rules (checked after entry on subsequent bars):
          - close <= entry_price - atr_entry * atr_stop_mult  (ATR stop)
          - close >= entry_price + atr_entry * atr_take_mult  (ATR profit target)
          - close < SMA_long (trend failure)
          - or exit after max_hold bars
        Implementation uses a forward loop to mark exits tied to each entry (so max_hold and ATR stops are applied per-entry).
        """
        # Ensure required columns
        required = {'open', 'high', 'low', 'close', 'volume'}
        if not required.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required}")

        o = df['open'].astype(float)
        h = df['high'].astype(float)
        l = df['low'].astype(float)
        c = df['close'].astype(float)
        v = df['volume'].astype(float)

        n = len(df)
        entries = np.zeros(n, dtype=bool)
        exits = np.zeros(n, dtype=bool)

        # Indicators available at time-of-bar
        atr = self._compute_atr(h, l, c)
        vol_med = v.rolling(self.vol_med_lookback, min_periods=1).median()
        sma_long = c.rolling(self.sma_long_len, min_periods=1).mean()

        i = 1  # start from 1 because we reference previous close
        while i < n:
            # Check entry possibility at bar i using information up to and including bar i
            prev_close = c.iat[i-1]
            open_i = o.iat[i]
            close_i = c.iat[i]
            vol_i = v.iat[i]
            atr_prev = atr.iat[i-1] if i-1 >= 0 else np.nan
            vol_med_prev = vol_med.iat[i-1] if i-1 >= 0 else np.nan

            # Basic sanity checks: no NaNs for required metrics
            if not (np.isfinite(prev_close) and np.isfinite(open_i) and np.isfinite(close_i)):
                i += 1
                continue

            gap_amount = prev_close - open_i  # positive for gap-down
            gap_condition = (gap_amount > max(self.gap_atr_mult * max(atr_prev, 1e-9), 0.0))
            bullish_recovery = (close_i > open_i) and (close_i >= open_i + self.recovery_fraction * gap_amount)
            vol_condition = np.isfinite(vol_med_prev) and (vol_i > vol_med_prev * self.volume_multiplier)

            if gap_condition and bullish_recovery and vol_condition:
                # Mark entry at this bar
                entries[i] = True

                entry_price = close_i
                atr_entry = atr.iat[i] if np.isfinite(atr.iat[i]) else max(1e-9, atr_prev)
                stop_price = entry_price - self.atr_stop_mult * atr_entry
                take_price = entry_price + self.atr_take_mult * atr_entry

                # Search forward for exit
                exit_idx = None
                last_j = min(n - 1, i + self.max_hold)
                for j in range(i + 1, last_j + 1):
                    close_j = c.iat[j]
                    sma_long_j = sma_long.iat[j] if np.isfinite(sma_long.iat[j]) else np.nan

                    # ATR stop
                    if np.isfinite(close_j) and close_j <= stop_price:
                        exit_idx = j
                        break
                    # ATR take
                    if np.isfinite(close_j) and close_j >= take_price:
                        exit_idx = j
                        break
                    # Trend break (close below long SMA)
                    if np.isfinite(sma_long_j) and close_j < sma_long_j:
                        exit_idx = j
                        break

                # If no exit triggered within max_hold window, exit at last_j
                if exit_idx is None:
                    # If last_j == i (max_hold 0) then exit immediately on next bar if exists, else mark exit at i
                    exit_target = last_j if last_j > i else i
                    # If exit target equals i (no future bars), exit on that same bar (close)
                    exits[exit_target] = True
                    # Continue scanning after exit_target
                    i = exit_target + 1
                else:
                    exits[exit_idx] = True
                    i = exit_idx + 1
            else:
                i += 1

        entries_series = pd.Series(entries, index=df.index)
        exits_series = pd.Series(exits, index=df.index)
        return entries_series, exits_series