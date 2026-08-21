import os
import sys
import json
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

strategy_file = sys.argv[1]
description = sys.argv[2]

with open(strategy_file, "r") as f:
    strategy_code = f.read()




instructions = """
You are an independent reviewer of Python Bitcoin trading strategy code.

Analyse the ACTUAL CODE and its accompanying DESCRIPTION.

Evaluate:

1. Look-ahead bias
2. Data leakage
3. Execution assumptions
4. Signal logic
5. Overfitting risk
6. Complexity
7. Logical coherence
8. Implementation correctness

For implementation correctness, specifically check whether the Python
code actually implements the strategy described in the DESCRIPTION.

Look for discrepancies between the description and implementation.

Do not use backtest results when making your decision.

First identify the main pros and cons.

Then assess the risks.

Finally decide PASS or FAIL.

Return ONLY valid JSON:

{
    "pros": ["...", "..."],
    "cons": ["...", "..."],
    "lookahead_risk": "LOW",
    "data_leakage_risk": "LOW",
    "execution_risk": "LOW",
    "overfitting_risk": "LOW",
    "implementation_risk": "LOW",
    "decision": "PASS"
}
"""


input_text = f"""
DESCRIPTION:

{description}


CODE:

{strategy_code}
"""


response = client.responses.create(
    model="gpt-5-mini",
    instructions=instructions,
    input=input_text
)

result = json.loads(response.output_text)

print(json.dumps(result))