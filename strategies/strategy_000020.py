import numpy as np
import pandas as pd

class Strategy:
    def __init__(
        self,
        swing_lookback=20,      # N: lookback for the prior swing high
        fail_window=5,          # M: how many recent bars to look for a failed breakout attempt
        sma_short=10,           # short-term trend SMA
        vol_med_window=20,      # window to compute median volume
        min_break_pct=1e-6,     # minimal fractional exceedance to count as a breakout
        stop_pct=0.06,          # fixed stop-loss percent from entry price (6%)
        max_hold=20             # maximum holding bars
    ):
        self.swing_lookback = int(swing_lookback)
        self.fail_window = int(fail_window)
        self.sma_short = int(sma_short)
        self.vol_med_window = int(vol_med_window)
        self.min_break_pct = float(min_break_pct)
        self.stop_pct = float(stop_pct)
        self.max_hold = int(max_hold)

    def generate_signals(self, df):
        """
        Input df must contain columns: open, high, low, close, volume
        Returns two boolean pandas Series: entries, exits (same index as df)
        """
        # Basic checks
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                raise ValueError(f"DataFrame missing required column: {col}")

        n = len(df)
        highs = df['high'].values
        closes = df['close'].values
        vols = df['volume'].values

        # 1) Prior swing high H at each bar i: maximum high over previous swing_lookback bars (exclude current bar)
        # Use shift(1) then rolling max to avoid including current bar's high in H
        H_series = df['high'].shift(1).rolling(self.swing_lookback, min_periods=1).max()
        H = H_series.values  # H[i] corresponds to bar i's prior swing high (based on data up to i-1)

        # 2) Short SMA for trend (we allow it to include current close because the signal is decided at close)
        sma_short_series = df['close'].rolling(self.sma_short, min_periods=1).mean()
        sma_short = sma_short_series.values

        # 3) Rolling median volume (we compare current volume to recent median volume)
        vol_med_series = df['volume'].rolling(self.vol_med_window, min_periods=1).median()
        vol_med = vol_med_series.values

        # 4) Failed breakout detection:
        # For a bar i, there is a failed attempt if any bar in (i - fail_window) .. (i-1) had high > H[i] but closed <= H[i]
        tol = 1e-12
        failed_attempt = np.zeros(n, dtype=bool)
        # Vectorized approach: for each k=1..fail_window check shifted bars against H
        for k in range(1, self.fail_window + 1):
            high_shift_k = df['high'].shift(k).values  # high at i-k
            close_shift_k = df['close'].shift(k).values
            cond_k = (high_shift_k > H + tol) & (close_shift_k <= H + tol)
            # cond_k contains True at index i when the bar at i-k satisfies the failed attempt relative to H[i]
            failed_attempt |= cond_k

        # 5) Candidate breakout condition: close > H * (1 + min_break_pct)
        breakout = np.zeros(n, dtype=bool)
        # Only meaningful when H is not NaN
        valid_H = ~np.isnan(H)
        breakout[valid_H] = closes[valid_H] > H[valid_H] * (1.0 + self.min_break_pct)

        # 6) Volume and trend confirmation
        vol_ok = vols > vol_med  # compares current volume to recent median (includes current bar)
        trend_ok = closes > sma_short

        candidate_entry = breakout & failed_attempt & vol_ok & trend_ok

        # Build final entries and exits series by simulating a simple position manager (only long)
        entries = np.zeros(n, dtype=bool)
        exits = np.zeros(n, dtype=bool)

        in_position = False
        entry_price = np.nan
        entry_idx = -1
        hold_count = 0

        for i in range(n):
            if not in_position:
                if candidate_entry[i]:
                    # Enter at this bar's close
                    entries[i] = True
                    in_position = True
                    entry_price = closes[i]
                    entry_idx = i
                    hold_count = 1
                # else remain flat
            else:
                # already in position: check exits (do not exit on the same bar as entry)
                # 1) short-term trend break: close < sma_short
                trend_break = closes[i] < sma_short[i]
                # 2) fixed percentage stop-loss relative to entry price
                stop_hit = closes[i] <= entry_price * (1.0 - self.stop_pct)
                # 3) max holding time reached (if we've just held max_hold bars including entry, exit now)
                hold_count += 1
                max_hold_hit = hold_count >= self.max_hold

                if trend_break or stop_hit or max_hold_hit:
                    exits[i] = True
                    in_position = False
                    entry_price = np.nan
                    entry_idx = -1
                    hold_count = 0
                else:
                    # Remain in position
                    pass

        # Return as pandas Series aligned with original df index
        entries_series = pd.Series(entries, index=df.index, name='entries').astype(bool)
        exits_series = pd.Series(exits, index=df.index, name='exits').astype(bool)
        return entries_series, exits_series