from generated_strategy import Strategy
from backtester import backtest
import json
import subprocess


# 1. Backtest
strategy = Strategy()
results = backtest(strategy)

print("\nBACKTEST RESULTS")
print(results)


# 2. Cheap filter
if (
    results["sharpe"] < 0.5
    or results["max_drawdown"] > 50
    or results["trades"] < 20
):
    print("\nStrategy rejected by basic backtest filter.")
    exit()


# 3. Run LLM evaluator
print("\nSending strategy to evaluator...")

subprocess.run(
    ["python", "evaluate_strategy.py"],
    check=True
)