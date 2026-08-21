import importlib.util
import sys
from backtester import backtest


strategy_file = sys.argv[1]

spec = importlib.util.spec_from_file_location("strategy", strategy_file)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

strategy = module.Strategy()

results = backtest(strategy)

print(results)