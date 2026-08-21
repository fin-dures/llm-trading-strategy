import os
import subprocess

from openai import OpenAI


client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


description_file = "strategies/strategy_descriptions.txt"

if os.path.exists(description_file):
    with open(description_file, "r") as f:
        previous_strategies = f.read()
else:
    previous_strategies = "No previous strategies."


# Find next strategy number
existing = [
    f for f in os.listdir("strategies")
    if f.startswith("strategy_") and f.endswith(".py")
]

strategy_number = len(existing) + 1
strategy_id = f"{strategy_number:06d}"

filename = f"strategies/strategy_{strategy_id}.py"


prompt = f"""
Generate ONE novel Bitcoin trading strategy in Python.

Use only:
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
- Return entries and exits through:

class Strategy:
    def generate_signals(self, df):
        ...
        return entries, exits

The strategy should be meaningfully different from the
previous strategies listed below.

PREVIOUS STRATEGIES:
{previous_strategies}

Return your response in exactly this format:

DESCRIPTION:
A concise 1-2 sentence description of the trading idea.

CODE:
<complete Python code>

Do not include anything else.
"""

response = client.responses.create(
    model="gpt-5-mini",
    input=prompt
)

full_output = response.output_text
description = full_output.split("CODE:", 1)[0]
code = full_output.split("CODE:", 1)[1].strip()


# Save strategy
with open(filename, "w") as f:
    f.write(code)

with open(description_file, "a") as f:
    f.write(f"strategy_{strategy_id}: {description}\n")

print(f"\nGenerated: {filename}")


# Run backtest + evaluator
subprocess.run(
    ["python", "run_strategy.py", filename],
    check=True
)

