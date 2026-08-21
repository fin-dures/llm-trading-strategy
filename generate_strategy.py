import os
import subprocess

from openai import OpenAI


client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# Create strategies folder
os.makedirs("strategies", exist_ok=True)


# Find next strategy number
existing = [
    f for f in os.listdir("strategies")
    if f.startswith("strategy_") and f.endswith(".py")
]

strategy_number = len(existing) + 1
strategy_id = f"{strategy_number:06d}"

filename = f"strategies/strategy_{strategy_id}.py"


prompt = """
Generate ONE novel Bitcoin trading strategy in Python.

Use only these columns:
- open
- high
- low
- close
- volume

Requirements:
- No external data
- No future information
- No look-ahead bias
- Use pandas
- The strategy must contain:

class Strategy:
    def generate_signals(self, df):
        ...
        return entries, exits

The strategy should be meaningfully different from a simple
moving-average crossover.

Do not explain the strategy.

Return ONLY valid Python code.
"""


response = client.responses.create(
    model="gpt-5-mini",
    input=prompt
)

code = response.output_text


# Save strategy
with open(filename, "w") as f:
    f.write(code)

print(f"\nGenerated: {filename}")


# Run backtest + evaluator
subprocess.run(
    ["python", "run_strategy.py", filename],
    check=True
)