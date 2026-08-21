from generated_strategy import Strategy
from backtester import backtest

strategy = Strategy()

results = backtest(strategy)

print(results)