import pandas as pd

class Strategy:
    def __init__(
        self,
        vwma_window=60,         # long window for VWMA (bars)
        vol_window=60,          # window to judge "normal" volume
        vol_mult=1.5,           # required volume multiple to consider spike
        mom_lookback=3,         # short momentum lookback (bars) for ROC flip
        dev_std_window=60,      # window to compute price std for deviation threshold
        dev_mult=1.0,           # how many std deviations from VWMA to trigger
        atr_window=14,          # ATR window for stops
        stop_atr_mult=1.5,      # stop loss in multiples of ATR
        max_hold=20             # maximum bars to hold a trade
    ):
        self.vwma_window = vwma_window
        self.vol_window = vol_window
        self.vol_mult = vol_mult
        self.mom_lookback = mom_lookback
        self.dev_std_window = dev_std_window
        self.dev_mult = dev_mult
        self.atr_window = atr_window
        self.stop_atr_mult = stop_atr_mult
        self.max_hold = max_hold

    def generate_signals(self, df):
        """
        df: pandas DataFrame with columns ['open','high','low','close','volume']
        Returns:
            entries, exits: two pandas.Series of booleans aligned with df.index.
                             True on the bar where an entry (or exit) is executed.
        """
        data = df.copy()
        close = data['close']
        high = data['high']
        low = data['low']
        vol = data['volume']

        # VWMA: volume-weighted moving average
        pv = close * vol
        sum_pv = pv.rolling(window=self.vwma_window, min_periods=1).sum()
        sum_vol = vol.rolling(window=self.vwma_window, min_periods=1).sum()
        vwma = sum_pv / sum_vol

        # Short-term momentum: Rate-of-change (ROC)
        roc = close.pct_change(self.mom_lookback)

        # Rolling previous ROC to detect a flip (sign change)
        roc_prev = roc.shift(1)

        # Average volume to gauge spikes
        avg_vol = vol.rolling(window=self.vol_window, min_periods=1).mean()
        vol_spike = vol > (avg_vol * self.vol_mult)

        # Price volatility used to set deviation threshold
        price_std = close.rolling(window=self.dev_std_window, min_periods=1).std()
        dev_threshold = self.dev_mult * price_std

        # ATR for stop calculations
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_window, min_periods=1).mean()

        # Prepare output series
        entries = pd.Series(False, index=data.index)
        exits = pd.Series(False, index=data.index)

        position = None       # None, 'long', 'short'
        entry_price = None
        entry_idx = None
        entry_atr = None
        exit_deadline_idx = None  # max hold index

        # Iterate forward through bars (no look-ahead)
        for i in range(len(data)):
            # current values (use iloc so indices don't matter)
            c = close.iloc[i]
            v = vol.iloc[i]
            vw = vwma.iloc[i]
            r = roc.iloc[i]
            rp = roc_prev.iloc[i]
            vs = vol_spike.iloc[i]
            thresh = dev_threshold.iloc[i]
            current_atr = atr.iloc[i]

            # skip if essential indicators are NaN
            if pd.isna(c) or pd.isna(vw) or pd.isna(thresh) or pd.isna(r) or pd.isna(rp) or pd.isna(current_atr):
                # if in a position and indicators unavailable, still allow time-based exit
                if position is not None and i >= exit_deadline_idx:
                    exits.iloc[i] = True
                    position = None
                    entry_price = None
                    entry_idx = None
                continue

            deviation = c - vw  # positive = above VWMA, negative = below VWMA

            if position is None:
                # Long entry conditions:
                # - price sufficiently below VWMA (mean reversion opportunity)
                # - short-term momentum flipped from negative to positive (rp < 0 and r > 0)
                # - volume spike confirms interest
                long_cond = (deviation < -thresh) and (rp < 0) and (r > 0) and (vs)

                # Short entry conditions: mirror of long
                short_cond = (deviation > thresh) and (rp > 0) and (r < 0) and (vs)

                if long_cond:
                    entries.iloc[i] = True
                    position = 'long'
                    entry_price = c
                    entry_idx = i
                    entry_atr = current_atr
                    exit_deadline_idx = min(len(data) - 1, i + self.max_hold)
                    continue  # move to next bar after opening
                elif short_cond:
                    entries.iloc[i] = True
                    position = 'short'
                    entry_price = c
                    entry_idx = i
                    entry_atr = current_atr
                    exit_deadline_idx = min(len(data) - 1, i + self.max_hold)
                    continue
            else:
                # Manage an existing trade
                if position == 'long':
                    # Stop-loss based on ATR from entry ATR (safer to use entry ATR)
                    stop_level = entry_price - (self.stop_atr_mult * entry_atr)
                    # Exit on:
                    # 1) price crosses back up to VWMA (mean reached)
                    # 2) price falls to or below stop_level
                    # 3) maximum holding time reached
                    if (c >= vw) or (c <= stop_level) or (i >= exit_deadline_idx):
                        exits.iloc[i] = True
                        position = None
                        entry_price = None
                        entry_idx = None
                        entry_atr = None
                        exit_deadline_idx = None
                        continue

                elif position == 'short':
                    stop_level = entry_price + (self.stop_atr_mult * entry_atr)
                    # Exit on:
                    # 1) price crosses back down to VWMA (mean reached)
                    # 2) price rises to or above stop_level
                    # 3) maximum holding time reached
                    if (c <= vw) or (c >= stop_level) or (i >= exit_deadline_idx):
                        exits.iloc[i] = True
                        position = None
                        entry_price = None
                        entry_idx = None
                        entry_atr = None
                        exit_deadline_idx = None
                        continue

        return entries, exits