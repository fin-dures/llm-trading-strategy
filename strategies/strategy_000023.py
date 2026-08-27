import pandas as pd
import numpy as np

class Strategy:
    def __init__(self,
                 skew_window=50,        # window to compute short-term skewness of returns
                 skew_ref_window=500,   # historical window to judge when skew is unusually negative
                 break_n=10,            # lookback for the prior high to define breakout
                 vol_window=50,         # window to compute typical volume
                 sma_mid=100,           # medium-term SMA: price must be above this to bias entries
                 sma_long=200,          # long-term SMA: trend-break exit
                 momentum_n=3):         # short-term momentum horizon for exit
        self.skew_window = skew_window
        self.skew_ref_window = skew_ref_window
        self.break_n = break_n
        self.vol_window = vol_window
        self.sma_mid = sma_mid
        self.sma_long = sma_long
        self.momentum_n = momentum_n

    def generate_signals(self, df):
        """
        Input:
            df: pandas DataFrame with columns ['open','high','low','close','volume'] and a datetime index
        Returns:
            entries, exits: boolean pandas Series aligned with df.index
        """
        # Ensure required columns exist
        required = {'open', 'high', 'low', 'close', 'volume'}
        if not required.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required}")

        close = df['close']
        open_p = df['open']
        high = df['high']
        volume = df['volume']

        # Short-term returns
        ret = close.pct_change()

        # Rolling skewness of returns (short window)
        skew_short = ret.rolling(self.skew_window, min_periods=int(self.skew_window/2)).skew()

        # Reference (historical) quantile of skewness to decide "unusually negative" skew
        # We compute the rolling lower quartile of skew_short over a longer window (only past values)
        skew_ref_q = skew_short.rolling(self.skew_ref_window, min_periods=int(self.skew_ref_window/4)).quantile(0.25)

        # Condition: skewness is unusually negative (short skew below its historical lower quartile)
        skew_condition = (skew_short < skew_ref_q)

        # Prior N-bar high (exclude current bar)
        prior_high = high.rolling(self.break_n, min_periods=1).max().shift(1)

        # Volume condition: today's volume above its recent median
        vol_med = volume.rolling(self.vol_window, min_periods=1).median()
        vol_condition = volume > vol_med

        # Bullish breakout: close above prior N-bar high and bullish candle (close > open)
        breakout_condition = (close > prior_high) & (close > open_p)

        # Medium-term trend bias: price above medium SMA
        sma_mid_series = close.rolling(self.sma_mid, min_periods=1).mean()
        trend_bias = close > sma_mid_series

        # Final entry: all conditions must hold on the close of the bar
        entries = (skew_condition & breakout_condition & vol_condition & trend_bias).fillna(False).astype(bool)

        # Exits:
        # 1) Long-term trend break: close below long SMA
        sma_long_series = close.rolling(self.sma_long, min_periods=1).mean()
        exit_trend_break = close < sma_long_series

        # 2) Short-term momentum flip: close falls below close n bars ago
        exit_momentum = close < close.shift(self.momentum_n)

        exits = (exit_trend_break | exit_momentum).fillna(False).astype(bool)

        # Ensure outputs are pandas Series aligned with df
        entries = pd.Series(entries, index=df.index, name='entries')
        exits = pd.Series(exits, index=df.index, name='exits')

        return entries, exits