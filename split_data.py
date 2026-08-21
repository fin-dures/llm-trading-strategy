import pandas as pd

df = pd.read_csv("btc_1h.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

development = df[
    (df["timestamp"] >= "2023-01-01") &
    (df["timestamp"] < "2026-01-01")
]

test = df[
    df["timestamp"] >= "2026-01-01"
]

development.to_csv("btc_development.csv", index=False)
test.to_csv("btc_test.csv", index=False)

print("Development:", len(development))
print("Test:", len(test))