import numpy as np
import pandas as pd

class Strategy:
    def __init__(self,
                 sma_short=20,
                 sma_long=50,
                 atr_period=14,
                 vol_sma=20,
                 wick_ratio_thresh=2.0,
                 close_upper_frac=0.66,
                 vol_mult=1.1,
                 atr_stop_mult=1.5,
                 atr_profit_mult=2.5,
                 max_hold=10):
        """
        Parameters are chosen to be simple and interpretable; they can be adjusted externally.
        """
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.atr_period = atr_period
        self.vol_sma = vol_sma
        self.wick_ratio_thresh = wick_ratio_thresh
        self.close_upper_frac = close_upper_frac
        self.vol_mult = vol_mult
        self.atr_stop_mult = atr_stop_mult
        self.atr_profit_mult = atr_profit_mult
        self.max_hold = max_hold

    def _atr(self, high, low, close, period):
        # Classic ATR (rolling mean of True Range)
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
            df: pandas DataFrame with columns ['open','high','low','close','volume']
        Output:
            entries, exits: boolean pandas Series indexed like df.index
        """
        # Validate columns
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

        # Indicators
        sma_s = c.rolling(self.sma_short, min_periods=1).mean()
        sma_l = c.rolling(self.sma_long, min_periods=1).mean()
        atr = self._atr(h, l, c, self.atr_period)
        vol_avg = v.rolling(self.vol_sma, min_periods=1).mean()

        # Bar geometry
        body = (c - o).abs()
        # lower wick size irrespective of bull/bear body
        lower_wick = (pd.concat([o, c], axis=1).min(axis=1) - l).clip(lower=0.0)
        high_low_range = (h - l).replace(0, np.nan)  # avoid division by zero
        close_pos_frac = (c - l) / high_low_range  # 0 at low, 1 at high

        # Candidate rejection bars:
        #  - lower wick large relative to body
        #  - close in upper fraction of range (buyers closed near highs)
        #  - occurs while short-term trend is below long-term trend (downtrend context)
        #  - above-average volume
        # Avoid division by zero by requiring body > small_epsilon OR treat large wick w.r.t small bodies
        eps = 1e-9
        wick_ratio = lower_wick / (body + eps)
        cond_wick = wick_ratio >= self.wick_ratio_thresh
        cond_close_pos = close_pos_frac >= self.close_upper_frac
        cond_trend = sma_s < sma_l  # short SMA below long SMA => downtrend
        cond_vol = v >= (vol_avg * self.vol_mult)
        cond_valid_atr = ~atr.isna()

        candidate = cond_wick & cond_close_pos & cond_trend & cond_vol & cond_valid_atr

        # Iterate through bars to schedule non-overlapping trades (entries/exits)
        in_position = False
        i = 0
        index_list = list(df.index)
        while i < n:
            if not in_position and candidate.iloc[i]:
                # Open trade at close of bar i
                entries.iloc[i] = True
                in_position = True
                entry_price = c.iloc[i]
                entry_atr = atr.iloc[i]
                # Defensive: if ATR is zero or NaN, skip entering
                if np.isnan(entry_atr) or entry_atr <= 0:
                    # Cancel this entry
                    entries.iloc[i] = False
                    in_position = False
                    i += 1
                    continue

                stop_price = entry_price - self.atr_stop_mult * entry_atr
                profit_price = entry_price + self.atr_profit_mult * entry_atr

                # scan forward up to max_hold bars to find exit condition
                exit_found = False
                last_j = min(n - 1, i + self.max_hold)  # inclusive target for max hold
                # start scanning from next bar (cannot exit at same bar as entry)
                for j in range(i + 1, last_j + 1):
                    # All exit checks use only information available at bar j's close
                    close_j = c.iloc[j]
                    sma_l_j = sma_l.iloc[j]
                    # 1) stop loss hit
                    if close_j <= stop_price:
                        exits.iloc[j] = True
                        exit_found = True
                        in_position = False
                        i = j + 1
                        break
                    # 2) profit target hit
                    if close_j >= profit_price:
                        exits.iloc[j] = True
                        exit_found = True
                        in_position = False
                        i = j + 1
                        break
                    # 3) long-term trend break: close below SMA_long at bar j
                    if close_j < sma_l_j:
                        exits.iloc[j] = True
                        exit_found = True
                        in_position = False
                        i = j + 1
                        break
                if not exit_found:
                    # Exit at end of maximum holding period (last_j)
                    exits.iloc[last_j] = True
                    in_position = False
                    i = last_j + 1
                # continue loop from updated i
            else:
                # Not a candidate or currently in position: advance
                i += 1

        # Ensure dtype bool
        entries = entries.fillna(False).astype(bool)
        exits = exits.fillna(False).astype(bool)

        return entries, exits