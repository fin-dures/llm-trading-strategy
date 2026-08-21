import pandas as pd
import numpy as np

class Strategy:
    def __init__(self,
                 intraday_win=50,
                 vol_win=50,
                 atr_win=14,
                 sigma_mult=1.5,
                 vol_threshold=1.8,
                 max_hold=10,
                 stop_atr_mult=2.0,
                 take_profit_atr_mult=1.0,
                 min_periods=10):
        self.intraday_win = intraday_win
        self.vol_win = vol_win
        self.atr_win = atr_win
        self.sigma_mult = sigma_mult
        self.vol_threshold = vol_threshold
        self.max_hold = max_hold
        self.stop_atr_mult = stop_atr_mult
        self.take_profit_atr_mult = take_profit_atr_mult
        self.min_periods = min_periods

    def generate_signals(self, df):
        # df must contain columns: open, high, low, close, volume
        o = df['open'].astype(float)
        h = df['high'].astype(float)
        l = df['low'].astype(float)
        c = df['close'].astype(float)
        v = df['volume'].astype(float)

        # Intraday return (current bar)
        intraday_ret = (c - o) / (o.replace(0, np.nan))

        # Rolling mean and std of intraday returns (uses past including current)
        mu = intraday_ret.rolling(window=self.intraday_win, min_periods=self.min_periods).mean()
        sigma = intraday_ret.rolling(window=self.intraday_win, min_periods=self.min_periods).std()

        # Volume anomaly relative to recent median volume
        vol_med = v.rolling(window=self.vol_win, min_periods=self.min_periods).median()
        vol_anom = v / (vol_med.replace(0, np.nan))

        # ATR calculation (True Range then simple moving average)
        prev_close = c.shift(1)
        tr1 = h - l
        tr2 = (h - prev_close).abs()
        tr3 = (l - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_win, min_periods=1).mean()

        n = len(df)
        entries = pd.Series(0, index=df.index, dtype=int)
        exits = pd.Series(0, index=df.index, dtype=int)

        in_pos = None  # None, 'long', 'short'
        entry_index = None
        entry_price = None
        hold_count = 0

        for i in range(n):
            # current values (only using data available at or before i)
            cur_intr = intraday_ret.iat[i]
            cur_mu = mu.iat[i] if i < len(mu) else np.nan
            cur_sigma = sigma.iat[i] if i < len(sigma) else np.nan
            cur_vol_anom = vol_anom.iat[i]
            cur_atr = atr.iat[i] if i < len(atr) else np.nan
            cur_close = c.iat[i]
            cur_open = o.iat[i]

            # Entry logic (only when not in a position)
            if in_pos is None:
                long_cond = False
                short_cond = False

                if (not np.isnan(cur_intr)) and (not np.isnan(cur_mu)) and (not np.isnan(cur_sigma)):
                    # Large negative intraday move on high volume -> mean-reversion long
                    if (cur_intr < (cur_mu - self.sigma_mult * cur_sigma)) and (cur_vol_anom >= self.vol_threshold):
                        long_cond = True

                    # Large positive intraday move on high volume -> mean-reversion short
                    if (cur_intr > (cur_mu + self.sigma_mult * cur_sigma)) and (cur_vol_anom >= self.vol_threshold):
                        short_cond = True

                if long_cond:
                    entries.iat[i] = 1
                    in_pos = 'long'
                    entry_index = i
                    entry_price = cur_close
                    hold_count = 0
                    continue  # do not evaluate exit at same bar
                if short_cond:
                    entries.iat[i] = -1
                    in_pos = 'short'
                    entry_index = i
                    entry_price = cur_close
                    hold_count = 0
                    continue

            else:
                # We are in a position; evaluate exit conditions using data up to current bar
                hold_count += 1

                exited = False

                # protective stop and take-profit based on ATR (relative to entry price)
                if not np.isnan(cur_atr) and entry_price is not None:
                    if in_pos == 'long':
                        if cur_close <= entry_price - self.stop_atr_mult * cur_atr:
                            exits.iat[i] = 1
                            exited = True
                        elif cur_close >= entry_price + self.take_profit_atr_mult * cur_atr:
                            exits.iat[i] = 1
                            exited = True
                    else:  # short
                        if cur_close >= entry_price + self.stop_atr_mult * cur_atr:
                            exits.iat[i] = -1
                            exited = True
                        elif cur_close <= entry_price - self.take_profit_atr_mult * cur_atr:
                            exits.iat[i] = -1
                            exited = True

                if exited:
                    in_pos = None
                    entry_index = None
                    entry_price = None
                    hold_count = 0
                    continue

                # Reversion condition: sign flip of intraday return relative to rolling mean
                # Exit long when intraday return becomes above its rolling mean (reversion)
                if (not np.isnan(cur_intr)) and (not np.isnan(cur_mu)):
                    if in_pos == 'long' and (cur_intr >= cur_mu):
                        exits.iat[i] = 1
                        in_pos = None
                        entry_index = None
                        entry_price = None
                        hold_count = 0
                        continue
                    if in_pos == 'short' and (cur_intr <= cur_mu):
                        exits.iat[i] = -1
                        in_pos = None
                        entry_index = None
                        entry_price = None
                        hold_count = 0
                        continue

                # Max holding time enforced
                if hold_count >= self.max_hold:
                    exits.iat[i] = 1 if in_pos == 'long' else -1
                    in_pos = None
                    entry_index = None
                    entry_price = None
                    hold_count = 0
                    continue

        return entries, exits


# Example usage (not part of the required return, shown for completeness):
# strat = Strategy()
# entries, exits = strat.generate_signals(df)