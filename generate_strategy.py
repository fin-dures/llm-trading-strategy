import os
import re
import subprocess

from openai import OpenAI


client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# --------------------------------------------------
# Configuration
# --------------------------------------------------

description_file = "strategies/strategy_descriptions.txt"


# --------------------------------------------------
# Load previous strategy descriptions
# --------------------------------------------------

if os.path.exists(description_file):
    with open(description_file, "r") as f:
        previous_strategies = f.read()
else:
    previous_strategies = "No previous strategies."


# --------------------------------------------------
# Find next strategy number
# --------------------------------------------------

existing_numbers = []

for filename in os.listdir("strategies"):
    match = re.fullmatch(r"strategy_(\d{6})\.py", filename)

    if match:
        existing_numbers.append(int(match.group(1)))


if existing_numbers:
    strategy_number = max(existing_numbers) + 1
else:
    strategy_number = 1


strategy_id = f"{strategy_number:06d}"

filename = f"strategies/strategy_{strategy_id}.py"


# --------------------------------------------------
# Prompt
# --------------------------------------------------

prompt = f"""
Generate ONE novel Bitcoin trading strategy in Python.

The purpose is to explore potentially profitable trading hypotheses,
not to optimise for historical backtest performance.

AVAILABLE DATA

The strategy may use ONLY these columns from the input dataframe:

- open
- high
- low
- close
- volume

Do not use external data, APIs, news, fundamentals, other assets,
future prices, future volume, or any information unavailable at the
time of the signal.

BACKTESTER CONSTRAINTS

The strategy will be tested using a long-only signal-based backtester.

Therefore:

- Generate only long entries and exits.
- Do not generate short signals.
- Do not assume custom order execution.
- Do not assume intrabar execution.
- Do not assume access to the next bar's data when generating a signal.
- Signals should be boolean pandas Series.
- The backtester handles fees and slippage.
- Do not implement a separate backtesting engine.
- Do not rely on information outside the dataframe.

SIGNAL INTERPRETATION

An entry signal means:

"Enter a long position based on information available at the
close of this bar."

An exit signal means:

"Exit the long position based on information available at the
close of this bar."

Avoid look-ahead bias.

If a feature uses historical data, ensure rolling calculations,
shifts and comparisons do not accidentally use future observations.

STRATEGY QUALITY

The strategy should:

- Have a clear economic or statistical rationale.
- Be reasonably simple and interpretable.
- Avoid excessive numbers of arbitrary parameters.
- Avoid simply combining many indicators without justification.
- Be meaningfully different from previous strategies.
- Explore a genuinely different trading hypothesis where possible.

Do NOT deliberately make the strategy complicated.

PREVIOUS STRATEGIES

The following are descriptions of strategies already generated.

You should avoid generating strategies that are substantially
similar to these.

{previous_strategies}


OUTPUT FORMAT

Return EXACTLY:

DESCRIPTION:
<1-2 sentence description of the trading hypothesis>

CODE:
<complete Python code>

The DESCRIPTION must describe exactly what the CODE implements.

The CODE must contain a class called Strategy with:

class Strategy:
    def generate_signals(self, df):
        ...
        return entries, exits

Do not include markdown code fences.

Do not include any text before DESCRIPTION or after the Python code.
"""


# --------------------------------------------------
# Generate strategy
# --------------------------------------------------

response = client.responses.create(
    model="gpt-5-mini",
    input=prompt
)

full_output = response.output_text.strip()


# --------------------------------------------------
# Parse LLM response safely
# --------------------------------------------------

if "CODE:" not in full_output:
    raise ValueError(
        "LLM response did not contain the required CODE: section."
    )


description_part, code = full_output.split("CODE:", 1)

description = description_part.replace("DESCRIPTION:", "").strip()

code = code.strip()


# --------------------------------------------------
# Remove accidental markdown fences
# --------------------------------------------------

if code.startswith("```python"):
    code = code[len("```python"):].strip()

elif code.startswith("```"):
    code = code[len("```"):].strip()


if code.endswith("```"):
    code = code[:-3].strip()


# --------------------------------------------------
# Basic validation
# --------------------------------------------------

if not description:
    raise ValueError("Generated strategy has no description.")


if not code:
    raise ValueError("Generated strategy has no code.")


if "class Strategy" not in code:
    raise ValueError(
        "Generated code does not contain a Strategy class."
    )


if "generate_signals" not in code:
    raise ValueError(
        "Generated code does not contain generate_signals()."
    )


# --------------------------------------------------
# Save strategy
# --------------------------------------------------

with open(filename, "w") as f:
    f.write(code)


with open(description_file, "a") as f:
    f.write(
        f"strategy_{strategy_id}: {description}\n"
    )


print(f"\nGenerated: {filename}")


# --------------------------------------------------
# Run backtest + evaluator
# --------------------------------------------------

result = subprocess.run(
    ["python", "run_strategy.py", filename]
)


# --------------------------------------------------
# Don't delete the strategy if the pipeline fails
# --------------------------------------------------

if result.returncode != 0:
    print(
        f"\nWARNING: {filename} could not be fully evaluated."
    )