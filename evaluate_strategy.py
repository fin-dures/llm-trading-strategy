import os
import sys
import json
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

strategy_file = sys.argv[1]

with open(strategy_file, "r") as f:
    strategy_code = f.read()




instructions = """
You are an independent reviewer of Python trading strategy code.

You MUST analyse the actual Python code provided in the next message.

Evaluate:

1. Look-ahead bias
2. Data leakage
3. Execution assumptions
4. Signal logic
5. Overfitting risk
6. Complexity
7. Logical coherence

First identify the main pros and cons of the ACTUAL CODE.

Then assign the four risk levels.

Finally decide PASS or FAIL.

Do not assume that the code is missing.

Return ONLY valid JSON in exactly this structure:

{
    "pros": ["...", "..."],
    "cons": ["...", "..."],
    "lookahead_risk": "LOW",
    "data_leakage_risk": "LOW",
    "execution_risk": "LOW",
    "overfitting_risk": "LOW",
    "decision": "PASS"
}
"""

response = client.responses.create(
    model="gpt-5-mini",
    instructions=instructions,
    input=strategy_code
)

result = json.loads(response.output_text)

print(json.dumps(result))