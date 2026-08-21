import pandas as pd
import numpy as np

class Strategy:
    def __init__(self,
                 breakout_window=20,
                 exit_window=10,
                 atr_window=14,
                 squeeze_window=20,
                 vol_window=20,
                 vol_mult=1.5,
                 squeeze_thresh=0.9,
                 stop_atr=3.0,
                 profit_atr=6.0,
                 max_hold=50):
        self.breakout_window = breakout_window
        self.exit_window = exit_window
        self.atr_window = atr_window
        self.squeeze_window = squeeze_window
        self.vol_window = vol_window
        self.vol_mult = vol_mult
        self.squeeze_thresh = squeeze_thresh
        self.stop_atr = stop_atr
        self.profit_atr = profit_atr
        self.max_hold = max_hold

    def _compute_atr(self, df):
        # True range
        prev_close = df['close'].shift(1)
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - prev_close).abs()
        tr3 = (df['low'] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_window, min_periods=1).mean()
        return atr

    def generate_signals(self, df):
        """
        df: pandas DataFrame with columns ['open','high','low','close','volume']
        returns: entries, exits - pandas Series of booleans indexed like df
        """
        n = len(df)
        entries = pd.Series(False, index=df.index)
        exits = pd.Series(False, index=df.index)

        # Basic safety checks
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError("DataFrame must contain open, high, low, close, volume")

        # Indicators computed using only past data (shifted to avoid look-ahead)
        breakout_high = df['high'].rolling(window=self.breakout_window, min_periods=self.breakout_window).max().shift(1)
        breakout_low = df['low'].rolling(window=self.exit_window, min_periods=self.exit_window).min().shift(1)
        vol_mean = df['volume'].rolling(window=self.vol_window, min_periods=1).mean().shift(1)

        atr = self._compute_atr(df).shift(1)  # ATR up to previous bar to avoid look-ahead
        atr_mean = atr.rolling(window=self.squeeze_window, min_periods=1).mean()  # already shifted

        # Squeeze condition: ATR is below recent average (compressing volatility)
        squeeze = atr < (atr_mean * self.squeeze_thresh)

        in_position = False
        entry_price = np.nan
        entry_atr = np.nan
        hold_count = 0

        # Iterate bars sequentially to produce entries/exits without look-ahead
        for i in range(n):
            # current bar values
            idx = df.index[i]
            close = df['close'].iat[i]
            vol = df['volume'].iat[i]
            bh = breakout_high.iat[i]
            bl = breakout_low.iat[i]
            a = atr.iat[i] if not pd.isna(atr.iat[i]) else np.nan
            sq = squeeze.iat[i] if not pd.isna(squeeze.iat[i]) else False

            # If not enough data for indicators, skip
            if pd.isna(close):
                continue

            if not in_position:
                # Entry: price breakout of prior range, small-volatility squeeze beforehand, and volume spike
                cond_breakout = (not pd.isna(bh)) and (close > bh)
                cond_vol = (not pd.isna(vol_mean.iat[i])) and (vol > vol_mean.iat[i] * self.vol_mult)
                cond_squeeze = bool(sq)
                if cond_breakout and cond_vol and cond_squeeze and (not pd.isna(a)):
                    entries.iat[i] = True
                    in_position = True
                    entry_price = close
                    entry_atr = a
                    hold_count = 0
            else:
                hold_count += 1
                # Exit conditions evaluated on current bar:
                stop_price = entry_price - self.stop_atr * entry_atr if not pd.isna(entry_atr) else -np.inf
                profit_price = entry_price + self.profit_atr * entry_atr if not pd.isna(entry_atr) else np.inf

                cond_stop = (close <= stop_price)
                cond_profit = (close >= profit_price)
                cond_breakdown = (not pd.isna(bl)) and (close < bl)
                cond_timeout = (hold_count >= self.max_hold)

                if cond_stop or cond_profit or cond_breakdown or cond_timeout:
                    exits.iat[i] = True
                    in_position = False
                    entry_price = np.nan
                    entry_atr = np.nan
                    hold_count = 0

        return entries, exits

# Example usage:
# strategy = Strategy()
# entries, exits = strategy.generate_signals(df)