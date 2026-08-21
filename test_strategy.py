from strategy_template import Strategy
from backtester import backtest



class TestStrategy(Strategy):

    def generate_signals(self, df):

        fast = df["close"].rolling(20).mean()
        slow = df["close"].rolling(100).mean()

        entries = (
            (fast > slow) &
            (fast.shift(1) <= slow.shift(1))
        )

        exits = (
            (fast < slow) &
            (fast.shift(1) >= slow.shift(1))
        )

        return entries, exits
    


strategy = TestStrategy()

results = backtest(strategy)

print(results)