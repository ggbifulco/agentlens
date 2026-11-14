# LLM judges

`llm_rubric` (score 1-5 against `criteria`, normalised to `(score - 1) / 4`, passes at
`min_score`, default 4) and `llm_pass_fail` (pass/fail with reasoning) ask a judge model at
temperature 0 for a JSON verdict, parse it robustly (code fences, prose, `"score": "4"`) and cache
it under `sha256(judge model, prompt, output)` in `.agentlens/judge_cache/`, so re-runs are free
and reproducible. The judge model comes from `model=` or `$AGENTLENS_JUDGE_MODEL` /
`$AGENTLENS_MODEL`; tests inject a `MockLLM`. Read [docs/judges.md](../judges.md) on judge bias
and calibration before trusting the numbers:

```python
import agentlens
from agentlens.evals import LLMPassFail, LLMRubric, evaluate_case
from agentlens.llm import MockLLM
from agentlens.schema import Case

with agentlens.run("support") as trace:
    trace.output = "Your refund of $42 for order 1234 is on its way."

judge = MockLLM(['{"reasoning": "Polite and states $42.", "score": 4}',
                 '```json\n{"reasoning": "Mentions order 1234.", "verdict": "pass"}\n```'])
case = Case(id="refund", input="refund 1234", expected="Confirms a $42 refund")
for result in evaluate_case(trace, case, [
    LLMRubric("Polite and confirms the refund amount", client=judge, cache="memory"),
    LLMPassFail("Mentions the order number", client=judge, cache="memory"),
    # in a suite: {llm_rubric: "Polite, confirms the refund amount"} with AGENTLENS_JUDGE_MODEL set
]):
    print(result.evaluator, result.score, result.reason)
# llm_rubric 0.75 judge score 4/5 (>= 4): Polite and states $42.
# llm_pass_fail 1.0 judge verdict: pass: Mentions order 1234.
```

`PairwiseJudge` compares two runs of the same case (baseline A vs candidate B). It asks the judge
in both orders — A shown first, then B shown first — and averages, so a judge that just prefers
whichever answer comes first ends up with a tie instead of a fake winner:

```python
from agentlens.evals import PairwiseJudge
from agentlens.llm import MockLLM
from agentlens.schema import Trace

baseline = Trace(name="support", output="Refund done.")
candidate = Trace(name="support", output="Your refund of $42 for order 1234 is on its way.")
judge = MockLLM(['{"reasoning": "Output 2 states $42.", "winner": "2"}',   # A shown first
                 '{"reasoning": "Output 1 states $42.", "winner": "1"}'])  # B shown first
result = PairwiseJudge("Confirms the refund amount", client=judge, cache=False).compare(
    baseline, candidate)
print(result.winner, result.preference, result.consistent)
print(result.to_eval_result(candidate="b").passed)
# b 0.0 True
# True
```


API: [`agentlens.evals`](../api/evals.md).
