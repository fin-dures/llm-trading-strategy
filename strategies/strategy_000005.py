import pandas as pd
import numpy as np

class Strategy:
    def __init__(
        self,
        n_break=20,
        atr_len=14,
        vol_len=20,
        vol_mult=1.5,
        range_mult=1.3,
        wick_thresh=0.4,
        exit_atr_mult=2.0,
        profit_atr_mult=3.0,
        max_holding=10,
    ):
        self.n_break = n_break
        self.atr_len = atr_len
        self.vol_len = vol_len
        self.vol_mult = vol_mult
        self.range_mult = range_mult
        self.wick_thresh = wick_thresh
        self.exit_atr_mult = exit_atr_mult
        self.profit_atr_mult = profit_atr_mult
        self.max_holding = max_holding

    def _compute_atr(self, df):
        prev_close = df['close'].shift(1)
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - prev_close).abs()
        tr3 = (df['low'] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_len, min_periods=1).mean()
        return atr

    def generate_signals(self, df):
        # df must contain only: open, high, low, close, volume
        df = df.copy()
        # compute ATR (no look-ahead)
        atr = self._compute_atr(df)
        atr_shift = atr.shift(1)  # use prior ATR when making today's decisions

        # previous n high (to detect breakout) - shift(1) so we don't include current bar
        prev_high_n = df['high'].rolling(self.n_break, min_periods=1).max().shift(1)

        # volume average (shifted)
        vol_avg = df['volume'].rolling(self.vol_len, min_periods=1).mean().shift(1)

        # Today's range and range expansion relative to prior ATR
        today_range = df['high'] - df['low']
        range_expand = today_range > (atr_shift * self.range_mult)

        # wick ratio: (upper_wick + lower_wick) / total_range
        body = (df['close'] - df['open']).abs()
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        # avoid division by zero
        total_range = (df['high'] - df['low']).replace(0, np.nan)
        wick_ratio = ((upper_wick + lower_wick) / total_range).fillna(1.0)

        # volume surge condition
        vol_surge = df['volume'] > (vol_avg * self.vol_mult)

        # breakout condition: close greater than prior n-high, with range expansion, volume surge, and small wick ratio
        breakout = (df['close'] > prev_high_n) & range_expand & vol_surge & (wick_ratio < self.wick_thresh)

        # Prepare entry/exits series
        entries = pd.Series(False, index=df.index)
        exits = pd.Series(False, index=df.index)

        # We'll walk forward in time to manage exits without look-ahead
        in_position = False
        entry_price = np.nan
        entry_atr = np.nan
        holding = 0

        # Convert to positional iteration for clarity
        for i in range(len(df)):
            if not in_position:
                if breakout.iat[i] and not pd.isna(prev_high_n.iat[i]) and not pd.isna(atr_shift.iat[i]):
                    # open a long position at close price of this bar
                    in_position = True
                    entries.iat[i] = True
                    entry_price = df['close'].iat[i]
                    entry_atr = atr_shift.iat[i] if not pd.isna(atr_shift.iat[i]) else atr.iat[i]
                    holding = 0
                # else remain flat
            else:
                holding += 1
                price = df['close'].iat[i]

                # compute stop and take-profit based on ATR measured at entry (no future info)
                stop = entry_price - entry_atr * self.exit_atr_mult
                take_profit = entry_price + entry_atr * self.profit_atr_mult

                # Exit conditions evaluated on current bar price
                if price <= stop:
                    exits.iat[i] = True
                    in_position = False
                    entry_price = np.nan
                    entry_atr = np.nan
                    holding = 0
                elif price >= take_profit:
                    exits.iat[i] = True
                    in_position = False
                    entry_price = np.nan
                    entry_atr = np.nan
                    holding = 0
                elif holding >= self.max_holding:
                    exits.iat[i] = True
                    in_position = False
                    entry_price = np.nan
                    entry_atr = np.nan
                    holding = 0
                # else remain in position

        # Ensure type boolean
        entries = entries.astype(bool)
        exits = exits.astype(bool)
        return entries, exits


# Example usage:
# strat = Strategy()
# entries, exits = strat.generate_signals(df)