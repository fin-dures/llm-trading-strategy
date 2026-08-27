import pandas as pd
import numpy as np

class Strategy:
    def __init__(self,
                 body_window=20,
                 vol_window=20,
                 atr_window=14,
                 body_spike=1.5,
                 vol_spike=1.5,
                 stop_atr_mult=1.5,
                 profit_atr_mult=2.5,
                 max_hold=10):
        self.body_window = body_window
        self.vol_window = vol_window
        self.atr_window = atr_window
        self.body_spike = body_spike
        self.vol_spike = vol_spike
        self.stop_atr_mult = stop_atr_mult
        self.profit_atr_mult = profit_atr_mult
        self.max_hold = max_hold

    def _prepare_indicators(self, df):
        # Ensure columns are present
        required = ['open', 'high', 'low', 'close', 'volume']
        for c in required:
            if c not in df.columns:
                raise ValueError(f"DataFrame must contain '{c}' column")

        o = df['open']
        h = df['high']
        l = df['low']
        c = df['close']
        v = df['volume']

        # Candle body size
        body = (c - o).abs()
        # Rolling average body and volume (shifted so current bar stats don't include current bar if needed)
        avg_body = body.rolling(self.body_window, min_periods=1).mean()
        avg_vol = v.rolling(self.vol_window, min_periods=1).mean()

        # True range and ATR
        tr = pd.concat([
            (h - l).abs(),
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_window, min_periods=1).mean()

        return body, avg_body, avg_vol, atr

    def generate_signals(self, df):
        """
        df: pandas DataFrame with columns ['open','high','low','close','volume']
        Returns:
            entries: pd.Series[bool] (True at bar where position is opened, execution assumed at that bar's open)
            exits:   pd.Series[bool] (True at bar where position is closed)
        Notes:
            - Entry signals are generated from information up to the previous bars and executed at next bar open.
            - Exits are determined intrabar after entry using subsequent bars (stop, profit, or max_hold).
        """
        df = df.copy().reset_index(drop=True)
        n = len(df)
        entries = pd.Series(False, index=df.index)
        exits = pd.Series(False, index=df.index)

        body, avg_body, avg_vol, atr = self._prepare_indicators(df)

        in_position = False
        pos_side = None  # 1 for long, -1 for short
        i = 1  # start at 1 because we inspect previous bar (i-1)
        while i < n - 1:  # need at least one bar ahead to execute entry (i+1)
            if not in_position:
                prev = i - 1
                # Skip if rolling stats not available for prev bar (conservative)
                if np.isnan(avg_body.iloc[prev]) or np.isnan(avg_vol.iloc[prev]) or np.isnan(atr.iloc[prev]):
                    i += 1
                    continue

                body_prev = body.iloc[prev]
                vol_prev = df['volume'].iloc[prev]
                open_prev = df['open'].iloc[prev]
                close_prev = df['close'].iloc[prev]

                # Detect bearish exhaustion candle (big down body + volume spike)
                bearish_exhaust = (close_prev < open_prev) and \
                                  (body_prev > avg_body.iloc[prev] * self.body_spike) and \
                                  (vol_prev > avg_vol.iloc[prev] * self.vol_spike)

                # Detect bullish exhaustion candle (big up body + volume spike)
                bullish_exhaust = (close_prev > open_prev) and \
                                  (body_prev > avg_body.iloc[prev] * self.body_spike) and \
                                  (vol_prev > avg_vol.iloc[prev] * self.vol_spike)

                # Use current bar (i) as confirmation (must be opposite-direction close that recovers beyond prev open)
                open_cur = df['open'].iloc[i]
                close_cur = df['close'].iloc[i]

                # Confirmation rules:
                # - For bullish reversal (after bearish exhaustion): current close > previous open
                # - For bearish reversal (after bullish exhaustion): current close < previous open
                if bearish_exhaust and (close_cur > open_prev):
                    # Signal long entry on next bar open (i+1) if available
                    exec_idx = i + 1
                    if exec_idx < n:
                        entries.iloc[exec_idx] = True
                        in_position = True
                        pos_side = 1
                        # Prepare trade parameters using ATR at exec_idx-1 (last known ATR)
                        entry_atr = atr.iloc[exec_idx - 1] if not np.isnan(atr.iloc[exec_idx - 1]) else atr.iloc[prev]
                        entry_price = df['open'].iloc[exec_idx]
                        stop = entry_price - self.stop_atr_mult * entry_atr
                        profit = entry_price + self.profit_atr_mult * entry_atr
                        # Walk forward to find exit
                        j = exec_idx
                        while j < n:
                            # If low breaches stop -> exit at j
                            if df['low'].iloc[j] <= stop:
                                exits.iloc[j] = True
                                in_position = False
                                pos_side = None
                                i = j  # continue scanning after exit
                                break
                            # If high reaches profit -> exit at j
                            if df['high'].iloc[j] >= profit:
                                exits.iloc[j] = True
                                in_position = False
                                pos_side = None
                                i = j
                                break
                            # Max hold
                            if j - exec_idx + 1 >= self.max_hold:
                                exits.iloc[j] = True
                                in_position = False
                                pos_side = None
                                i = j
                                break
                            j += 1
                        else:
                            # ran to end without exit - close at last bar
                            exits.iloc[n - 1] = True
                            in_position = False
                            pos_side = None
                            i = n - 1
                elif bullish_exhaust and (close_cur < open_prev):
                    # Signal short entry on next bar open (i+1)
                    exec_idx = i + 1
                    if exec_idx < n:
                        entries.iloc[exec_idx] = True
                        in_position = True
                        pos_side = -1
                        entry_atr = atr.iloc[exec_idx - 1] if not np.isnan(atr.iloc[exec_idx - 1]) else atr.iloc[prev]
                        entry_price = df['open'].iloc[exec_idx]
                        stop = entry_price + self.stop_atr_mult * entry_atr
                        profit = entry_price - self.profit_atr_mult * entry_atr
                        j = exec_idx
                        while j < n:
                            # For short: high breaches stop, low reaches profit
                            if df['high'].iloc[j] >= stop:
                                exits.iloc[j] = True
                                in_position = False
                                pos_side = None
                                i = j
                                break
                            if df['low'].iloc[j] <= profit:
                                exits.iloc[j] = True
                                in_position = False
                                pos_side = None
                                i = j
                                break
                            if j - exec_idx + 1 >= self.max_hold:
                                exits.iloc[j] = True
                                in_position = False
                                pos_side = None
                                i = j
                                break
                            j += 1
                        else:
                            exits.iloc[n - 1] = True
                            in_position = False
                            pos_side = None
                            i = n - 1
                # Move forward one bar if no entry (or after detection)
                i += 1
            else:
                # We should never get here because we process exits immediately after an entry loop.
                i += 1

        # Align boolean Series index back to original df index if it had one
        # Here we reset index earlier, so just return Series with integer index which aligns with df.reset_index.
        # Caller can align using original index if needed.
        entries.index = df.index
        exits.index = df.index
        return entries, exits