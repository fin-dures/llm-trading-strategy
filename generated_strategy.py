import pandas as pd


class Strategy:
    def generate_signals(self, df):
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        prior_high = high.rolling(20, min_periods=20).max().shift(1)
        prior_low = low.rolling(10, min_periods=10).min().shift(1)

        fast_ema = close.ewm(span=50, adjust=False, min_periods=50).mean()
        slow_ema = close.ewm(span=200, adjust=False, min_periods=200).mean()

        prior_median_volume = (
            volume.rolling(20, min_periods=20).median().shift(1)
        )

        breakout = (close > prior_high) & (
            close.shift(1) <= prior_high.shift(1)
        )

        entries = (
            breakout
            & (fast_ema > slow_ema)
            & (volume > prior_median_volume)
        )

        exits = (
            (close < prior_low)
            | ((fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1)))
        )

        entries = entries.fillna(False).astype(bool)
        exits = (exits & ~entries).fillna(False).astype(bool)

        return entries, exits