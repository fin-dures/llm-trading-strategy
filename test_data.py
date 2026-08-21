import ccxt
import pandas as pd
import time

exchange = ccxt.binance()
symbol = "BTC/USDT"
timeframe = "1h"

since = exchange.parse8601("2020-01-01T00:00:00Z")
all_data = []

while True:
    data = exchange.fetch_ohlcv(
        symbol,
        timeframe=timeframe,
        since=since,
        limit=1000
    )

    if not data:
        break

    all_data.extend(data)
    since = data[-1][0] + 1

    print(f"Downloaded {len(all_data)} candles")

    if len(data) < 1000:
        break

    time.sleep(exchange.rateLimit / 1000)

df = pd.DataFrame(
    all_data,
    columns=["timestamp", "open", "high", "low", "close", "volume"]
)

df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

df = df.drop_duplicates("timestamp")
df.to_csv("btc_1h.csv", index=False)

print(f"\nSaved {len(df)} candles to btc_1h.csv")