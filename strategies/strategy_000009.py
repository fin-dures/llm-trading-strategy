DESCRIPTION: A divergence-based OBV (on-balance volume) contrarian breakout/reversal: detect price making a new short-term high (or low) while OBV fails to confirm (bearish or bullish divergence) with a concurrent volume anomaly, enter on the next bar open (short on bearish divergence, long on bullish), and exit on ATR-based stop/profit, divergence resolution, intrabar hit of stop/take, or maximum holding time.

CODE:
import pandas as pd
import numpy as np

class Strategy:
    def __init__(
        self,
        lookback=20,            # lookback to define new highs/lows
        obv_corr_window=30,     # window to compute correlation (for optional weakening confirmation)
        vol_med_window=20,      # volume median window for spike detection
        vol_mult=1.5,           # required multiplier above median volume
        atr_period=14,
        stop_atr=2.0,           # stop loss in multiples of ATR
        take_atr=3.0,           # take profit in multiples of ATR
        max_holding=12          # maximum bars to hold a trade
    ):
        self.lookback = lookback
        self.obv_corr_window = obv_corr_window
        self.vol_med_window = vol_med_window
        self.vol_mult = vol_mult
        self.atr_period = atr_period
        self.stop_atr = stop_atr
        self.take_atr = take_atr
        self.max_holding = max_holding

    def _compute_obv(self, df):
        # On-Balance Volume
        obv = np.zeros(len(df))
        closes = df['close'].values
        vols = df['volume'].values
        for i in range(1, len(df)):
            if closes[i] > closes[i-1]:
                obv[i] = obv[i-1] + vols[i]
            elif closes[i] < closes[i-1]:
                obv[i] = obv[i-1] - vols[i]
            else:
                obv[i] = obv[i-1]
        return pd.Series(obv, index=df.index)

    def _compute_atr(self, df):
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        # simple moving average ATR (no look-ahead; using past data)
        atr = tr.rolling(self.atr_period, min_periods=1).mean()
        return atr

    def generate_signals(self, df):
        """
        df: pandas DataFrame with columns ['open','high','low','close','volume']
        Returns:
            entries: pd.Series of ints (1 for long entry at that bar open, -1 for short entry, 0 none)
            exits:   pd.Series of ints (1 to exit long at that bar, -1 to exit short at that bar, 0 none)
        Note: Entries indicate the bar where the order is executed (we assume fills at bar open).
        """
        df = df.copy().reset_index(drop=True)
        n = len(df)
        entries = pd.Series(0, index=df.index, dtype=int)
        exits = pd.Series(0, index=df.index, dtype=int)

        # Indicators
        obv = self._compute_obv(df)
        atr = self._compute_atr(df)
        vol_med = df['volume'].rolling(self.vol_med_window, min_periods=1).median()
        # Rolling max/min for price and obv using prior bars (exclude current bar to avoid look-ahead)
        rolling_max_close = df['close'].shift(1).rolling(self.lookback, min_periods=1).max()
        rolling_min_close = df['close'].shift(1).rolling(self.lookback, min_periods=1).min()
        rolling_max_obv = obv.shift(1).rolling(self.lookback, min_periods=1).max()
        rolling_min_obv = obv.shift(1).rolling(self.lookback, min_periods=1).min()

        # Optional weakening confirmation: rolling correlation between close returns and OBV
        close_ret = df['close'].pct_change().fillna(0)
        obv_ret = obv.pct_change().fillna(0)
        corr = (
            pd.concat([close_ret, obv_ret], axis=1)
              .rolling(self.obv_corr_window, min_periods=1)
              .corr()
        )
        # Extract correlation of close_ret vs obv_ret; corr is a MultiIndex; handle safely
        corr_series = pd.Series(0.0, index=df.index)
        try:
            corr_series = corr.xs('close', level=0, axis=1).xs('obv_ret', level=1, axis=1)
        except Exception:
            # Build correlation manually if above fails
            corr_series = close_ret.rolling(self.obv_corr_window, min_periods=1).corr(obv_ret)

        # We'll use a simple weakening check: correlation < 0.3 indicates weakening confirmation
        corr_threshold = 0.3

        # Position simulation variables
        position = 0          # 0 no position, 1 long, -1 short
        scheduled_entry = 0   # entry scheduled to execute at next bar open
        entry_idx = None
        entry_price = None
        stop_price = None
        take_price = None
        holding = 0

        for i in range(n):
            # Execute scheduled entry at this bar open
            if scheduled_entry != 0 and position == 0:
                # Enter at open of bar i
                position = scheduled_entry
                entries.iat[i] = int(position)
                entry_idx = i
                entry_price = df['open'].iat[i]
                # Use ATR from previous bar if available to avoid look-ahead
                atr_for_entry = atr.iat[i-1] if i-1 >= 0 else atr.iat[i]
                if np.isnan(atr_for_entry) or atr_for_entry == 0:
                    atr_for_entry = df['high'].iat[max(0, i-1)] - df['low'].iat[max(0, i-1)]
                    if atr_for_entry == 0:
                        atr_for_entry = 1.0
                # For long: stop = entry - stop_atr*atr ; take = entry + take_atr*atr
                stop_price = entry_price - position * self.stop_atr * atr_for_entry
                take_price = entry_price + position * self.take_atr * atr_for_entry
                holding = 0
                scheduled_entry = 0

            # If in position, check intrabar stops/takes and other exits
            if position != 0:
                holding += 1
                low = df['low'].iat[i]
                high = df['high'].iat[i]
                close = df['close'].iat[i]

                exited = False
                # Intrabar hit: for long, low <= stop -> stop hit; high >= take -> take hit
                if position == 1:
                    if low <= stop_price:
                        exits.iat[i] = 1
                        exited = True
                    elif high >= take_price:
                        exits.iat[i] = 1
                        exited = True
                else:  # short
                    if high >= stop_price:
                        exits.iat[i] = -1
                        exited = True
                    elif low <= take_price:
                        exits.iat[i] = -1
                        exited = True

                if exited:
                    # clear position
                    position = 0
                    entry_idx = None
                    entry_price = None
                    stop_price = None
                    take_price = None
                    holding = 0
                    continue  # move to next bar

                # Max holding time exit (exit at close of this bar)
                if holding >= self.max_holding:
                    exits.iat[i] = int(position) * 1  # exit sign matching position
                    position = 0
                    entry_idx = None
                    entry_price = None
                    stop_price = None
                    take_price = None
                    holding = 0
                    continue

                # Divergence resolution exit: if OBV starts confirming price direction again,
                # e.g., for a bearish divergence short, OBV rises to make a new high relative to lookback
                # or correlation recovers above threshold, then exit on close
                if position == -1:
                    obv_confirms = obv.iat[i] >= rolling_max_obv.iat[i]  # OBV made new high (confirms buying)
                    corr_recovered = corr_series.iat[i] > corr_threshold
                    if obv_confirms or corr_recovered:
                        exits.iat[i] = -1
                        position = 0
                        entry_idx = None
                        entry_price = None
                        stop_price = None
                        take_price = None
                        holding = 0
                        continue
                elif position == 1:
                    obv_confirms = obv.iat[i] <= rolling_min_obv.iat[i]  # OBV made new low (confirms selling)
                    corr_recovered = corr_series.iat[i] > corr_threshold
                    if obv_confirms or corr_recovered:
                        exits.iat[i] = 1
                        position = 0
                        entry_idx = None
                        entry_price = None
                        stop_price = None
                        take_price = None
                        holding = 0
                        continue

            # If not in position and nothing scheduled, detect divergence signals at bar i and schedule entry for next bar
            if position == 0 and scheduled_entry == 0:
                # Need enough prior data: use rolling_max/min computed from shifted data
                price = df['close'].iat[i]
                obv_i = obv.iat[i]
                vol_i = df['volume'].iat[i]
                recent_vol_med = vol_med.iat[i]
                # Check for bullish divergence: price makes a new short-term low while OBV does NOT make new low (OBV higher)
                is_new_low = price < rolling_min_close.iat[i] if not np.isnan(rolling_min_close.iat[i]) else False
                obv_not_confirm_low = obv_i > rolling_min_obv.iat[i] if not np.isnan(rolling_min_obv.iat[i]) else False
                vol_spike = vol_i > (recent_vol_med * self.vol_mult)

                if is_new_low and obv_not_confirm_low and vol_spike:
                    # Also require weakening correlation (low correlation) to reduce false positives
                    if corr_series.iat[i] < corr_threshold:
                        # schedule long at next bar open
                        if i + 1 < n:
                            scheduled_entry = 1

                # Check for bearish divergence: price makes a new short-term high while OBV does NOT make new high (OBV lower)
                is_new_high = price > rolling_max_close.iat[i] if not np.isnan(rolling_max_close.iat[i]) else False
                obv_not_confirm_high = obv_i < rolling_max_obv.iat[i] if not np.isnan(rolling_max_obv.iat[i]) else False

                if is_new_high and obv_not_confirm_high and vol_spike:
                    if corr_series.iat[i] < corr_threshold:
                        # schedule short at next bar open
                        if i + 1 < n:
                            scheduled_entry = -1

        # Map indices back to original DataFrame index if needed: here we used reset_index so return with same integer index
        # Convert to Series aligned with original index type by copying index from original df (if original had datetime index user should adapt)
        # But for this function we'll return Series with same RangeIndex as input
        return entries, exits