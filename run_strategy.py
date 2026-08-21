import importlib.util
import subprocess
import sys
import json

from backtester import backtest


strategy_file = sys.argv[1]

# Load strategy
spec = importlib.util.spec_from_file_location("strategy", strategy_file)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

strategy = module.Strategy()

# Backtest
results = backtest(strategy)

print("\nBACKTEST RESULTS")
for key, value in results.items():
    print(f"{key}: {value}")


# Cheap filter
if (
    results["sharpe"] < 0.5
    or results["max_drawdown"] > 50
    or results["trades"] < 20
):
    print("\nREJECTED: Failed basic backtest filter.")
    sys.exit(0)


# Run evaluator
print("\nSending strategy to evaluator...")


description_file = "strategies/strategy_descriptions.txt"
strategy_id = strategy_file.split("/")[-1].replace(".py", "")

description = "No description available."

with open(description_file, "r") as f:
    for line in f:
        if line.startswith(strategy_id + ":"):
            description = line.split(":", 1)[1].strip()
            break


result = subprocess.run(
    ["python", "evaluate_strategy.py", strategy_file, description],
    capture_output=True,
    text=True,
    check=True
)

print(result.stdout)

# Extract JSON from evaluator output
evaluation = json.loads(result.stdout)

if evaluation["decision"] == "PASS":
    print("\n*** STRATEGY PASSED ***")
else:
    print("\n*** STRATEGY FAILED ***")