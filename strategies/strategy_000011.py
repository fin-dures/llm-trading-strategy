import pandas as pd
import numpy as np

class Strategy:
    def __init__(
        self,
        atr_len=14,
        atr_med_len=100,
        short_sma=10,
        long_sma=50,
        breakout_lookback=20,
        vol_ma_len=20,
        vol_mult=1.5,
        oversold_k=1.5,
        trailing_atr_mult=3.0,
        max_hold=20,
    ):
        self.atr_len = atr_len
        self.atr_med_len = atr_med_len
        self.short_sma = short_sma
        self.long_sma = long_sma
        self.breakout_lookback = breakout_lookback
        self.vol_ma_len = vol_ma_len
        self.vol_mult = vol_mult
        self.oversold_k = oversold_k
        self.trailing_atr_mult = trailing_atr_mult
        self.max_hold = max_hold

    def _compute_indicators(self, df):
        # Ensure required columns exist
        o = df["open"]
        h = df["high"]
        l = df["low"]
        c = df["close"]
        v = df["volume"]

        prev_close = c.shift(1)

        # True range
        tr1 = h - l
        tr2 = (h - prev_close).abs()
        tr3 = (l - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(self.atr_len, min_periods=1).mean()

        # ATR median to classify regime
        atr_median = atr.rolling(self.atr_med_len, min_periods=1).median()

        short_sma = c.rolling(self.short_sma, min_periods=1).mean()
        long_sma = c.rolling(self.long_sma, min_periods=1).mean()

        # previous N-bar high (exclude current bar)
        prev_high = c.rolling(self.breakout_lookback, min_periods=1).max().shift(1)

        vol_ma = v.rolling(self.vol_ma_len, min_periods=1).mean()

        indicators = pd.DataFrame(
            {
                "close": c,
                "open": o,
                "high": h,
                "low": l,
                "volume": v,
                "prev_close": prev_close,
                "atr": atr,
                "atr_median": atr_median,
                "short_sma": short_sma,
                "long_sma": long_sma,
                "prev_high": prev_high,
                "vol_ma": vol_ma,
            },
            index=df.index,
        )

        return indicators

    def generate_signals(self, df):
        """
        Returns:
            entries, exits : tuple of pandas Series (boolean) indexed like df
        """
        ind = self._compute_indicators(df)

        n = len(ind)
        entries = pd.Series(False, index=ind.index)
        exits = pd.Series(False, index=ind.index)

        # Build raw entry signals based on regime
        # Low-vol regime: ATR <= ATR median -> mean-reversion dip + small momentum flip (close > prev_close)
        low_vol = ind["atr"] <= ind["atr_median"]
        low_vol_oversold = ind["close"] < (ind["short_sma"] - self.oversold_k * ind["atr"])
        momentum_flip = ind["close"] > ind["prev_close"]  # simple intraday flip
        low_vol_entry = low_vol & low_vol_oversold & momentum_flip

        # High-vol regime: ATR > ATR median -> breakout above previous N-bar high with volume confirmation
        high_vol = ind["atr"] > ind["atr_median"]
        breakout = ind["close"] > ind["prev_high"]
        vol_confirm = ind["volume"] > (self.vol_mult * ind["vol_ma"])
        high_vol_entry = high_vol & breakout & vol_confirm

        candidate_entry = low_vol_entry | high_vol_entry

        # Simulate forward to ensure entries/exits do not overlap and to apply exits that depend on entry state
        in_position = False
        highest_since_entry = np.nan
        hold_count = 0

        closes = ind["close"].values
        atrs = ind["atr"].values
        long_smas = ind["long_sma"].values
        cand = candidate_entry.values
        idx = ind.index

        for i in range(n):
            if not in_position:
                if cand[i] and not np.isnan(closes[i]) and not np.isnan(atrs[i]):
                    # Enter at close of this bar
                    entries.iloc[i] = True
                    in_position = True
                    highest_since_entry = closes[i]
                    hold_count = 1  # count current bar as first held bar
                else:
                    continue
            else:
                # Update highest close since entry with current close
                if not np.isnan(closes[i]):
                    if np.isnan(highest_since_entry):
                        highest_since_entry = closes[i]
                    else:
                        highest_since_entry = max(highest_since_entry, closes[i])
                # Check exits only from the next bar after entry (avoid immediate exit)
                # Condition 1: trend break (close below long SMA)
                exit_trend = False
                if not np.isnan(closes[i]) and not np.isnan(long_smas[i]):
                    exit_trend = closes[i] < long_smas[i]
                # Condition 2: ATR-based trailing stop (drop from highest_since_entry)
                exit_trail = False
                if (
                    (not np.isnan(closes[i]))
                    and (not np.isnan(highest_since_entry))
                    and (not np.isnan(atrs[i]))
                ):
                    exit_trail = closes[i] < (highest_since_entry - self.trailing_atr_mult * atrs[i])
                # Condition 3: maximum holding time
                hold_count += 1
                exit_time = hold_count >= self.max_hold

                if exit_trend or exit_trail or exit_time:
                    exits.iloc[i] = True
                    in_position = False
                    highest_since_entry = np.nan
                    hold_count = 0
                    # Continue; next candidate entry can be taken on subsequent bars
                # else remain in position

        # Ensure dtype bool
        entries = entries.astype(bool)
        exits = exits.astype(bool)

        return entries, exits