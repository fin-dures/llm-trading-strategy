import numpy as np
import pandas as pd

class Strategy:
    def __init__(self,
                 sv_window=20,          # window for signed-volume aggregation
                 sv_threshold=1.5,      # multiple of recent average magnitude required
                 sma_short=50,          # short-term trend SMA
                 vol_ma_window=20,      # volume average window for volume filter
                 max_holding_bars=30,   # maximum bars to hold a trade
                 exit_rel_threshold=0.5 # fraction of the entry threshold to force exit
                ):
        self.sv_window = sv_window
        self.sv_threshold = sv_threshold
        self.sma_short = sma_short
        self.vol_ma_window = vol_ma_window
        self.max_holding_bars = max_holding_bars
        self.exit_rel_threshold = exit_rel_threshold

    def generate_signals(self, df):
        """
        Input: df with columns ['open','high','low','close','volume']
        Output: entries, exits - boolean pandas Series aligned with df.index
        """
        # Validate required columns
        required = {'open', 'high', 'low', 'close', 'volume'}
        if not required.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required}")

        o = df['open'].astype(float)
        c = df['close'].astype(float)
        v = df['volume'].astype(float)

        # Signed volume: positive when close > open (buy pressure), negative when close < open
        signed_vol = (c - o) * v

        # Rolling aggregated metrics (include current bar because we decide at close)
        sv_sum = signed_vol.rolling(window=self.sv_window, min_periods=self.sv_window).sum()
        sv_avg_abs = signed_vol.abs().rolling(window=self.sv_window, min_periods=self.sv_window).mean()

        # A simple normalization: require aggregated net buying > k * (average absolute signed vol * window)
        # Note avg_abs is per-bar; multiply by window to compare to sum
        sv_threshold_series = sv_avg_abs * (self.sv_window * self.sv_threshold)

        # Short-term trend (simple moving average of close)
        sma_short = c.rolling(window=self.sma_short, min_periods=1).mean()

        # Volume filter to avoid very thin-volume signals
        vol_ma = v.rolling(window=self.vol_ma_window, min_periods=1).mean()

        # Preliminary entry condition (boolean Series) evaluated at the close of the bar
        prelim_entry = (
            (sv_sum > sv_threshold_series) &                 # unusually strong net buying flow
            (c > sma_short) &                                # price above short-term SMA (trend bias)
            (c > c.shift(1)) &                               # short-term momentum (close higher than previous)
            (v >= (0.5 * vol_ma))                            # at least some relative volume (>= 50% recent avg)
        )

        # Exit conditions evaluated at the close of the bar (when in position)
        # 1) Net buying pressure fades: sv_sum <= exit_threshold (we use a fraction of entry threshold)
        exit_threshold = sv_avg_abs * (self.sv_window * self.exit_rel_threshold)
        cond_pressure_fades = sv_sum <= exit_threshold

        # 2) Trend break: close below short-term SMA
        cond_trend_break = c < sma_short

        # We'll produce entries and exits by scanning forward so we can enforce max holding duration and avoid re-entry while long
        entries = pd.Series(False, index=df.index)
        exits = pd.Series(False, index=df.index)

        in_position = False
        entry_idx = None
        entry_threshold_at_entry = None  # store threshold at entry to allow relative exit

        idx_list = list(df.index)
        for i, idx in enumerate(idx_list):
            # At bar idx we only use information up to and including this bar (no future)
            if not in_position:
                # Only enter when prelim_entry is true at this bar
                if prelim_entry.iloc[i]:
                    entries.iloc[i] = True
                    in_position = True
                    entry_idx = i
                    # capture the threshold at entry time to allow an exit when pressure falls significantly from that level
                    # If sv_avg_abs is NaN (rare early bars), set a fallback small number to avoid NaNs
                    thresh_val = sv_threshold_series.iloc[i]
                    if np.isnan(thresh_val) or np.isinf(thresh_val):
                        thresh_val = 0.0
                    entry_threshold_at_entry = thresh_val
            else:
                # When in a position evaluate exit conditions using current-bar information
                time_in_trade = i - entry_idx + 1
                # pressure fades relative to either absolute exit_threshold OR relative drop from entry threshold
                fades_by_abs = cond_pressure_fades.iloc[i]
                fades_by_relative_drop = True
                if entry_threshold_at_entry is None or entry_threshold_at_entry == 0:
                    # if we didn't have a good threshold at entry, use absolute condition only
                    fades_by_relative_drop = False
                else:
                    # exit if current sv_sum falls below a fraction of the threshold that triggered entry
                    fades_by_relative_drop = sv_sum.iloc[i] <= (0.5 * entry_threshold_at_entry)
                if fades_by_abs or fades_by_relative_drop or cond_trend_break.iloc[i] or (time_in_trade >= self.max_holding_bars):
                    exits.iloc[i] = True
                    in_position = False
                    entry_idx = None
                    entry_threshold_at_entry = None
                    # note: we do not create an immediate re-entry on the same bar even if prelim_entry true;
                    # that would be indistinguishable from staying in the trade

        # Ensure boolean dtype
        entries = entries.fillna(False).astype(bool)
        exits = exits.fillna(False).astype(bool)

        return entries, exits