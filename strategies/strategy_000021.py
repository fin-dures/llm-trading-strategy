import numpy as np
import pandas as pd

class Strategy:
    def __init__(self,
                 formation_window=30,
                 long_window_multiplier=4,
                 low_slope_thresh=0.0008,
                 high_slope_thresh=0.0005,
                 range_contraction_factor=0.7,
                 volume_spike_factor=1.2,
                 sma_long=50,
                 max_hold=20):
        """
        Parameters (defaults chosen as reasonable starting values):
        - formation_window: bars used to define the triangle formation (ends at the previous bar)
        - long_window_multiplier: how many formation windows to use for range baseline (for contraction test)
        - low_slope_thresh: normalized (slope / mean_price) threshold for rising lows
        - high_slope_thresh: normalized threshold for flat/declining highs (upper bound)
        - range_contraction_factor: current formation range must be less than this fraction of its longer-term median
        - volume_spike_factor: breakout volume must exceed recent median by this factor
        - sma_long: length for a longer-term SMA filter (trend)
        - max_hold: max bars to hold a trade after entry
        """
        self.formation_window = int(formation_window)
        self.long_window = int(formation_window * long_window_multiplier)
        self.low_slope_thresh = float(low_slope_thresh)
        self.high_slope_thresh = float(high_slope_thresh)
        self.range_contraction_factor = float(range_contraction_factor)
        self.volume_spike_factor = float(volume_spike_factor)
        self.sma_long = int(sma_long)
        self.max_hold = int(max_hold)

    def _linreg_slope_intercept(self, x, y):
        # Return slope and intercept of linear fit y = slope * x + intercept
        # If degenerate (constant y), return slope=0 and intercept=mean(y)
        if len(y) == 0:
            return 0.0, 0.0
        if np.allclose(y, y[0]):
            return 0.0, float(y[0])
        # Use np.polyfit
        slope, intercept = np.polyfit(x, y, 1)
        return float(slope), float(intercept)

    def generate_signals(self, df):
        """
        df: pandas DataFrame with columns: open, high, low, close, volume
        Returns: entries, exits as boolean pandas Series aligned with df.index
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        vol = df['volume'].values
        n = len(df)
        idx = np.arange(n).astype(float)

        # Precompute SMA long (trend filter) - this uses data up to current bar
        sma_long_series = pd.Series(close).rolling(window=self.sma_long, min_periods=1).mean().values

        # Prepare arrays for formation metrics (based on data up to previous bar)
        slope_low = np.full(n, np.nan)
        intercept_low = np.full(n, np.nan)
        slope_high = np.full(n, np.nan)
        intercept_high = np.full(n, np.nan)
        formation_range = np.full(n, np.nan)
        prior_max_high = np.full(n, np.nan)
        vol_median = np.full(n, np.nan)

        # Compute formation metrics for windows that end at previous bar (i.e., window indices [t-formation_window, t-1])
        for t in range(self.formation_window, n):
            start = t - self.formation_window
            end = t  # exclusive, so window is start..t-1
            x_window = idx[start:end]
            lows_window = low[start:end]
            highs_window = high[start:end]

            s_low, i_low = self._linreg_slope_intercept(x_window, lows_window)
            s_high, i_high = self._linreg_slope_intercept(x_window, highs_window)
            slope_low[t] = s_low
            intercept_low[t] = i_low
            slope_high[t] = s_high
            intercept_high[t] = i_high

            formation_range[t] = highs_window.max() - lows_window.min()
            prior_max_high[t] = highs_window.max()
            vol_median[t] = np.median(vol[start:end]) if len(vol[start:end]) > 0 else np.nan

        # Compute long-term baseline of formation ranges (rolling median of formation_range up to previous bar)
        formation_range_series = pd.Series(formation_range)
        long_range_median = formation_range_series.shift(1).rolling(window=self.long_window, min_periods=1).median().values

        # Compute normalized slopes (slope divided by mean price in window) - for formation windows ending at prev bar
        # To compute mean price per window, loop similarly
        mean_price = np.full(n, np.nan)
        for t in range(self.formation_window, n):
            start = t - self.formation_window
            end = t
            mean_price[t] = np.mean(close[start:end])

        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            norm_slope_low = np.where(np.isfinite(slope_low) & (mean_price != 0), slope_low / mean_price, np.nan)
            norm_slope_high = np.where(np.isfinite(slope_high) & (mean_price != 0), slope_high / mean_price, np.nan)

        # Formation conditions (evaluated using data up to previous bar)
        formation_ok = (
            (norm_slope_low >= self.low_slope_thresh) &                       # rising lows
            (norm_slope_high <= self.high_slope_thresh) &                     # flat or declining highs
            (~np.isnan(formation_range)) & (~np.isnan(long_range_median)) &
            (formation_range < (long_range_median * self.range_contraction_factor))  # range contraction
        )

        # Entry condition at current bar t:
        # - formation_ok up to previous bar
        # - current close > prior_max_high (break above formation high)
        # - current volume > recent formation median * factor
        # - price above long SMA
        entries = np.zeros(n, dtype=bool)
        for t in range(self.formation_window, n):
            if not formation_ok[t]:
                continue
            if np.isnan(prior_max_high[t]) or np.isnan(vol_median[t]):
                continue
            if close[t] > prior_max_high[t] and vol[t] > vol_median[t] * self.volume_spike_factor and close[t] > sma_long_series[t]:
                entries[t] = True

        # Now generate exits by simulating position lifecycle:
        exits = np.zeros(n, dtype=bool)
        in_position = False
        entry_idx = None
        entry_lower_slope = None
        entry_lower_intercept = None

        for t in range(n):
            # If not in position, check entry
            if not in_position:
                if entries[t]:
                    in_position = True
                    entry_idx = t
                    # Use the formation slope/intercept computed at entry bar (those were for formation ending at t-1)
                    entry_lower_slope = slope_low[t]
                    entry_lower_intercept = intercept_low[t]
                # else continue
            else:
                # Evaluate exit conditions while in position
                # 1) price falls below extrapolated lower trendline from the formation at the current absolute index
                below_trendline = False
                if entry_lower_slope is not None and not np.isnan(entry_lower_intercept):
                    trend_value = entry_lower_slope * idx[t] + entry_lower_intercept
                    below_trendline = close[t] < trend_value

                # 2) price falls below long SMA
                below_sma = close[t] < sma_long_series[t]

                # 3) max holding time exceeded (exit on the bar when holding_time >= max_hold)
                holding_time = t - entry_idx + 1
                exceed_hold = holding_time >= self.max_hold

                if below_trendline or below_sma or exceed_hold:
                    exits[t] = True
                    in_position = False
                    entry_idx = None
                    entry_lower_slope = None
                    entry_lower_intercept = None
                # else remain in position

        # Return as pandas Series aligned to df.index
        entries_series = pd.Series(entries, index=df.index).astype(bool)
        exits_series = pd.Series(exits, index=df.index).astype(bool)
        return entries_series, exits_series