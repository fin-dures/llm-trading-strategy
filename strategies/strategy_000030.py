import pandas as pd
import numpy as np

class Strategy:
    def __init__(
        self,
        sma_short=10,
        sma_long=50,
        atr_window=14,
        atr_median_window=200,
        skew_window=21,
        skew_threshold=0.5,
        volume_median_window=50,
        max_hold=10,
    ):
        """
        Parameters are chosen to be reasonable defaults for typical BTC bar data (e.g., hourly/daily),
        but the strategy is primarily a hypothesis explorer, not an optimiser.
        """
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.atr_window = atr_window
        self.atr_median_window = atr_median_window
        self.skew_window = skew_window
        self.skew_threshold = skew_threshold
        self.volume_median_window = volume_median_window
        self.max_hold = max_hold

    def _atr(self, df):
        # True range
        prev_close = df["close"].shift(1)
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - prev_close).abs()
        tr3 = (df["low"] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_window, min_periods=1).mean()
        return atr

    def generate_signals(self, df):
        """
        Input:
            df: pandas DataFrame with columns ['open','high','low','close','volume']
        Output:
            entries, exits: boolean pandas Series aligned with df.index
        """
        # Work on a copy to avoid modifying user's df
        data = df.copy()

        # Basic moving averages
        sma_short = data["close"].rolling(self.sma_short, min_periods=1).mean()
        sma_long = data["close"].rolling(self.sma_long, min_periods=1).mean()

        # ATR and its long-term median to detect volatility compression
        atr = self._atr(data)
        atr_median = atr.rolling(self.atr_median_window, min_periods=1).median()

        # Short-term return skewness (rolling skew of pct returns)
        returns = data["close"].pct_change().fillna(0)
        # pandas supports rolling().skew()
        skew = returns.rolling(self.skew_window, min_periods=3).skew()

        # Volume baseline
        vol_median = data["volume"].rolling(self.volume_median_window, min_periods=1).median()

        # Entry conditions evaluated on close of current bar
        cond_trend = data["close"] > sma_long  # medium-term uptrend
        cond_vol_compress = atr < atr_median  # volatility compressed relative to recent
        cond_skew = skew > self.skew_threshold  # unusually positive skew
        cond_mom = data["close"] > data["close"].shift(1)  # short momentum (today's close > yesterday)
        cond_vol = data["volume"] > vol_median  # volume participation above recent median
        cond_short_above = data["close"] > sma_short  # also above short SMA

        entry_condition = cond_trend & cond_vol_compress & cond_skew & cond_mom & cond_vol & cond_short_above
        entry_condition = entry_condition.fillna(False).astype(bool)

        # Prepare output Series
        entries = pd.Series(False, index=data.index)
        exits = pd.Series(False, index=data.index)

        # Simulate a simple position state machine to generate exits (including max holding time)
        in_pos = False
        entry_idx = None  # integer index in positional iteration

        # To index by integer position, get arrays/views for speed
        closes = data["close"].values
        sma_short_arr = sma_short.values
        skew_arr = skew.values
        entry_bool = entry_condition.values
        n = len(data)

        for i in range(n):
            if not in_pos:
                if entry_bool[i]:
                    entries.iloc[i] = True
                    in_pos = True
                    entry_idx = i
            else:
                # Only allow exits starting from the bar after the entry (no immediate same-bar exit)
                held_bars = i - entry_idx  # 0 on same bar as entry
                exit_due_to_hold = held_bars >= self.max_hold and held_bars > 0
                # short-term trend break: close falls below short SMA
                exit_due_to_trend = (closes[i] < sma_short_arr[i]) if not np.isnan(sma_short_arr[i]) else False
                # skew flips negative
                exit_due_to_skew = (skew_arr[i] < 0) if not np.isnan(skew_arr[i]) else False

                if (exit_due_to_hold or exit_due_to_trend or exit_due_to_skew):
                    exits.iloc[i] = True
                    in_pos = False
                    entry_idx = None

        # Ensure boolean dtype and align lengths
        entries = entries.fillna(False).astype(bool)
        exits = exits.fillna(False).astype(bool)

        return entries, exits