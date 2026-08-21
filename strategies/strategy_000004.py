import pandas as pd
import numpy as np

class Strategy:
    def __init__(self,
                 range_window=20,
                 atr_window=14,
                 vol_window=20,
                 vol_quantile=0.75,
                 atr_break_mult=0.5,
                 stop_atr_mult=2.0,
                 exit_fade_window=5):
        self.range_window = range_window
        self.atr_window = atr_window
        self.vol_window = vol_window
        self.vol_quantile = vol_quantile
        self.atr_break_mult = atr_break_mult
        self.stop_atr_mult = stop_atr_mult
        self.exit_fade_window = exit_fade_window

    def generate_signals(self, df):
        # expect df with columns: open, high, low, close, volume
        data = df.copy().loc[:, ['open', 'high', 'low', 'close', 'volume']]

        # True Range and ATR
        prev_close = data['close'].shift(1)
        tr1 = data['high'] - data['low']
        tr2 = (data['high'] - prev_close).abs()
        tr3 = (data['low'] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_window, min_periods=1).mean()

        # Previous range (no look-ahead): range over past range_window bars, shifted so current bar compares to prior range
        prev_range_high = data['high'].rolling(self.range_window, min_periods=1).max().shift(1)
        prev_range_low  = data['low'].rolling(self.range_window, min_periods=1).min().shift(1)

        # Volume imbalance: up-volume minus down-volume over vol_window (includes current bar)
        bar_dir = np.sign(data['close'] - data['open']).fillna(0)
        vol_signed = data['volume'] * bar_dir
        vol_mom = vol_signed.rolling(self.vol_window, min_periods=1).sum()

        # Dynamic threshold for vol_mom based on its own past distribution (no future info)
        # Compute a rolling quantile of past vol_mom values (shifted so threshold uses only prior history)
        vol_mom_threshold = vol_mom.shift(1).rolling(self.vol_window, min_periods=1).quantile(self.vol_quantile)

        # Entry condition:
        # 1) Price breaks above the previous range high by a small ATR fraction
        # 2) Strong positive volume imbalance relative to recent history
        # 3) Current bar is bullish (close > open)
        breakout = (data['close'] > (prev_range_high + self.atr_break_mult * atr))
        vol_surge = vol_mom > vol_mom_threshold
        bullish_bar = data['close'] > data['open']

        entry_condition = breakout & vol_surge & bullish_bar & prev_range_high.notna()

        # Exit condition(s):
        # 1) Price falls below a fading moving-average of recent closes (based only on past closes)
        # 2) Price drops under the previous range low minus a small ATR buffer
        fade_ma = data['close'].rolling(self.exit_fade_window, min_periods=1).mean().shift(1)
        exit_fade = data['close'] < (fade_ma - self.stop_atr_mult * atr)
        exit_range_break = data['close'] < (prev_range_low - 0.5 * atr)

        exit_condition = (exit_fade | exit_range_break) & prev_range_low.notna()

        # Build boolean Series aligned with df
        entries = pd.Series(False, index=data.index)
        exits = pd.Series(False, index=data.index)

        entries.loc[entry_condition] = True
        exits.loc[exit_condition] = True

        return entries, exits


# Example usage:
# strat = Strategy()
# entries, exits = strat.generate_signals(df)