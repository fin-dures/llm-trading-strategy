import numpy as np
import pandas as pd

class Strategy:
    def __init__(self,
                 mom_period=5,
                 atr_period=14,
                 atr_median_period=50,
                 vol_med_period=20,
                 sma_exit_period=20,
                 mom_z_thresh=0.8,
                 max_hold=20):
        """
        Parameters are exposed to keep the implementation interpretable.
        They are modest defaults; the backtester will evaluate performance.
        """
        self.mom_period = mom_period
        self.atr_period = atr_period
        self.atr_median_period = atr_median_period
        self.vol_med_period = vol_med_period
        self.sma_exit_period = sma_exit_period
        self.mom_z_thresh = mom_z_thresh
        self.max_hold = int(max_hold)

    def _true_range(self, df):
        prev_close = df['close'].shift(1)
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - prev_close).abs()
        tr3 = (df['low'] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr

    def generate_signals(self, df):
        """
        Input:
            df: pandas DataFrame with columns ['open','high','low','close','volume']
        Returns:
            entries, exits: tuple of boolean pandas Series aligned with df.index
              entries[i] True means "enter long at close of this bar"
              exits[i]   True means "exit long at close of this bar"
        """
        data = df.copy().loc[:, ['open', 'high', 'low', 'close', 'volume']]

        # Basic sanity: ensure numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # ATR
        tr = self._true_range(data)
        atr = tr.rolling(self.atr_period, min_periods=1).mean()

        # Short-term momentum (absolute return over mom_period)
        mom = data['close'] - data['close'].shift(self.mom_period)

        # Volatility-adjusted momentum
        vol_adj_mom = mom / (atr.replace(0, np.nan))
        vol_adj_mom = vol_adj_mom.fillna(0)

        # Regime filters
        atr_median = atr.rolling(self.atr_median_period, min_periods=1).median()
        vol_median = data['volume'].rolling(self.vol_med_period, min_periods=1).median()
        sma_exit = data['close'].rolling(self.sma_exit_period, min_periods=1).mean()

        # Entry: first crossing above threshold (no lookahead: use shift)
        mom_cross_up = (vol_adj_mom > self.mom_z_thresh) & (vol_adj_mom.shift(1) <= self.mom_z_thresh)
        low_vol_regime = atr < atr_median  # quieter-than-recent volatility
        volume_confirm = data['volume'] > vol_median
        bullish_bar = data['close'] > data['open']  # simple confirmation that bar closed up

        raw_entry = mom_cross_up & low_vol_regime & volume_confirm & bullish_bar

        # Raw exit conditions (stateless)
        mom_cross_down = (vol_adj_mom < 0) & (vol_adj_mom.shift(1) >= 0)
        trend_break = data['close'] < sma_exit
        raw_exit = mom_cross_down | trend_break

        # Now enforce max holding time and ensure we don't generate overlapping entries while in a position.
        entries = pd.Series(False, index=data.index)
        exits = pd.Series(False, index=data.index)

        in_position = False
        hold_count = 0  # number of bars held so far (0 means not in position)

        # Iterate sequentially to avoid lookahead and to implement max_hold
        for idx in range(len(data)):
            if not in_position:
                # If a raw entry is present at this bar, open position
                if raw_entry.iloc[idx]:
                    entries.iloc[idx] = True
                    in_position = True
                    hold_count = 0  # counting from entry bar
                    # Do not check exits on the same bar as entry (entry and exit on same close are pointless)
                else:
                    # nothing to do
                    pass
            else:
                # We are in a position: check exits at this bar (exit based on current bar information)
                # First check stateless exit conditions
                if raw_exit.iloc[idx]:
                    exits.iloc[idx] = True
                    in_position = False
                    hold_count = 0
                else:
                    # increment holding time (we count the entry bar as 0, so exit when hold_count >= max_hold-1)
                    hold_count += 1
                    if hold_count >= (self.max_hold - 1):
                        # force exit due to max hold
                        exits.iloc[idx] = True
                        in_position = False
                        hold_count = 0
                    else:
                        # remain in position
                        pass
                # ignore any raw_entry while in position (no pyramiding)
                # loop continues

        # Ensure boolean dtype
        entries = entries.fillna(False).astype(bool)
        exits = exits.fillna(False).astype(bool)

        return entries, exits