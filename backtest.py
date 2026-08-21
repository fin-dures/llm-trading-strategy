import pandas as pd
import vectorbt as vbt

df = pd.read_csv("btc_1h.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")

price = df["close"]

fast = vbt.MA.run(price, window=20)
slow = vbt.MA.run(price, window=100)

entries = fast.ma_crossed_above(slow)
exits = fast.ma_crossed_below(slow)

portfolio = vbt.Portfolio.from_signals(
    price,
    entries,
    exits,
    fees=0.001
)

print(portfolio.stats())