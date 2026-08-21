import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

with open("generated_strategy.py", "r") as f:
    strategy_code = f.read()

prompt = f"""
You are an independent evaluator of an automatically generated Bitcoin
trading strategy.

Analyse the strategy objectively before making your final decision.

Evaluate:

1. Look-ahead bias
2. Data leakage
3. Execution assumptions
4. Signal logic
5. Overfitting risk
6. Complexity
7. Whether the strategy is logically coherent

First identify the main PROS and CONS.

Then assess the risks.

Finally decide whether the strategy should PASS or FAIL.

Do not rewrite or modify the strategy.

Return ONLY valid JSON in exactly this structure:

{{
    "pros": ["...", "..."],
    "cons": ["...", "..."],
    "lookahead_risk": "LOW",
    "data_leakage_risk": "LOW",
    "execution_risk": "LOW",
    "overfitting_risk": "LOW",
    "decision": "PASS",
}}

The decision should be based on the analysis above.

STRATEGY CODE:

{strategy_code}
"""

response = client.responses.create(
    model="gpt-5-mini",
    input=prompt
)

result = json.loads(response.output_text)

print(json.dumps(result, indent=2))