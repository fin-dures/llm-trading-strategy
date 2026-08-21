import pandas as pd
import vectorbt as vbt

def load_data():
    df = pd.read_csv("btc_development.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    return df


def backtest(strategy):
    df = load_data()

    entries, exits = strategy.generate_signals(df)

    portfolio = vbt.Portfolio.from_signals(
    df["close"],
    entries,
    exits,
    fees=0.001,
    slippage=0.0005,
    freq="1h"
    )
    stats = portfolio.stats()



    return {
        "return": stats["Total Return [%]"],
        "sharpe": stats["Sharpe Ratio"],
        "sortino": stats["Sortino Ratio"],
        "calmar": stats["Calmar Ratio"],
        "max_drawdown": stats["Max Drawdown [%]"],
        "trades": stats["Total Trades"],
        "win_rate": stats["Win Rate [%]"],
        "profit_factor": stats["Profit Factor"],
        "fees": stats["Total Fees Paid"],
    }
    