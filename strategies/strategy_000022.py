import pandas as pd
import numpy as np

class Strategy:
    def generate_signals(self, df):
        """
        Inputs:
            df: pandas DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        Returns:
            entries, exits: two boolean pandas Series indexed like df
        """
        # Parameters (kept simple and interpretable)
        sma_long = 50                  # defines the longer-term uptrend
        body_window = 50               # window for defining a "strong" body quantile
        strong_quantile = 0.85         # threshold quantile for a strong bullish bar
        vol_med_window = 20            # window for volume median used in comparisons
        lookback_after_strong = 5      # how many bars after a strong bar we allow a pullback
        atr_window = 14                # ATR window for stop-sizing
        atr_stop_multiplier = 1.5      # stop distance in ATR units from entry
        max_holding = 20               # maximum bars to hold a trade
        
        df = df.copy()
        # ensure required columns exist
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        close = df['close']
        open_ = df['open']
        high = df['high']
        low = df['low']
        vol = df['volume']
        
        # Basic derived series
        body = close - open_
        # body quantile threshold computed using only prior information (shifted)
        body_q = body.rolling(body_window, min_periods=5).quantile(strong_quantile).shift(1)
        # rolling median volume (shifted so comparisons use past baseline)
        vol_med = vol.rolling(vol_med_window, min_periods=5).median().shift(1)
        # long SMA to define trend (using values available at the bar close)
        sma = close.rolling(sma_long, min_periods=1).mean()
        
        # Identify "strong bullish bars": large-bodied up-bars with above-normal volume
        strong_bull = (body > 0) & (body > body_q) & (vol > vol_med)
        strong_bull = strong_bull.fillna(False)
        
        # ATR (simple moving average of True Range) - no shift: ATR at bar's close is known then
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(atr_window, min_periods=1).mean()
        
        # Prepare output series
        entries = pd.Series(False, index=df.index)
        exits = pd.Series(False, index=df.index)
        
        in_position = False
        entry_i = None
        entry_close = None
        entry_atr = None
        
        # Iteratively scan bars to ensure we do not open multiple simultaneous positions (simple single-pos logic)
        # and to construct exits that are evaluated bar-by-bar using only information available at each bar.
        n = len(df)
        for i in range(n):
            # convenience accessors
            idx = df.index[i]
            c = close.iat[i]
            o = open_.iat[i]
            v = vol.iat[i]
            l = low.iat[i]
            current_sma = sma.iat[i]
            current_vol_med = vol_med.iat[i] if i < n else np.nan  # vol_med already shifted
            
            if in_position:
                # Evaluate exit conditions at this bar (information available at bar close)
                # 1) ATR-based stop (price has fallen enough relative to entry)
                stop_price = entry_close - atr_stop_multiplier * entry_atr if entry_atr is not None else -np.inf
                exit_by_stop = (c <= stop_price)
                # 2) Trend break: close below the long SMA
                exit_by_trend = (c < current_sma)
                # 3) Maximum holding time reached
                held = i - entry_i + 1
                exit_by_time = (held >= max_holding)
                
                if exit_by_stop or exit_by_trend or exit_by_time:
                    exits.iat[i] = True
                    in_position = False
                    entry_i = None
                    entry_close = None
                    entry_atr = None
                # otherwise keep holding
            else:
                # Potential entry evaluation at this bar (done using information up to this close)
                # Basic filters for a "low-volume pullback" bar:
                # - market in long-term uptrend
                # - current bar is bullish (close > open) and showing short-term positive momentum (close > prev close)
                # - current volume is below recent median (i.e., a "dry" pullback)
                # - there exists a recent strong bullish bar within the allowed lookback whose low has not been breached by this bar
                if i == 0:
                    prev_close = np.nan
                else:
                    prev_close = close.iat[i-1]
                
                cond_trend = (c > current_sma)
                cond_bull_bar = (c > o) and (not np.isnan(prev_close)) and (c > prev_close)
                cond_low_volume = False
                if not np.isnan(current_vol_med):
                    cond_low_volume = (v < current_vol_med)
                
                if cond_trend and cond_bull_bar and cond_low_volume:
                    # search for the most recent strong bullish bar within lookback window
                    found_strong = False
                    start_j = max(0, i - lookback_after_strong)
                    for j in range(i-1, start_j-1, -1):  # search backwards from immediate prior
                        if strong_bull.iat[j]:
                            # require that the current bar's low did not go below that strong bar's low
                            strong_low = low.iat[j]
                            if l >= strong_low:
                                found_strong = True
                                break
                    if found_strong:
                        # require ATR available at this bar to size stop
                        if not np.isnan(atr.iat[i]) and atr.iat[i] > 0:
                            entries.iat[i] = True
                            in_position = True
                            entry_i = i
                            entry_close = c
                            entry_atr = atr.iat[i]
                        # else skip entry if ATR not meaningful yet
        
        # Ensure boolean dtype
        entries = entries.astype(bool)
        exits = exits.astype(bool)
        return entries, exits