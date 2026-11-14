# Evaluators

An evaluator scores one trace of one case. Write one as a function with `@agentlens.evaluator`;
it may return a `bool`, a score in `[0, 1]` (passes at `>= threshold`, default 0.5),
`(score, reason)` or a full `EvalResult`. Extra parameters come from the suite spec, and an
evaluator that raises gives a failed result instead of crashing the run:

```python
import agentlens
from agentlens.evals import evaluate_case, get_evaluator
from agentlens.schema import Case

@agentlens.evaluator(threshold=0.8)
def short_answer(trace, case, max_words: int = 50):
    words = len(str(trace.output).split())
    return min(1.0, max_words / max(words, 1)), f"{words} words"

with agentlens.run("support") as trace:
    trace.output = "Your refund for order 1234 has been issued."

case = Case(id="refund", input="refund 1234", evaluators=["short_answer"])
print(evaluate_case(trace, case)[0].passed)  # True: 8 words <= 50
check = get_evaluator({"short_answer": 4})  # YAML shorthand: scalar -> first parameter
print(check.evaluate(trace, case).reason)  # "8 words" (score 0.5 < 0.8: failed)
```

Suites refer to evaluators by name (`no_error`, `{max_steps: 8}`, `{tool_sequence: {mode:
ordered}}`); `agentlens.evals.register(name, factory)` adds class-based ones and
`available_evaluators()` lists them.

## Built-in output checks
`exact_match`, `contains`, `regex`, `json_schema` and `no_error` are registered out of the box.
Each returns a score in `[0, 1]`, a readable `reason` and `details`; bad parameters (an invalid
regex or schema) fail when the suite is loaded. Text checks see a dict/list output as canonical
JSON, and `path: answer` (or `items.0.text`) picks one field:

```python
import agentlens
from agentlens.evals import evaluate_case
from agentlens.schema import Case

with agentlens.run("support") as trace:
    trace.output = {"answer": "Refund issued for order 1234", "status": "refunded"}

case = Case(id="refund", input="refund 1234", expected="Refund issued for order 1234")
for result in evaluate_case(trace, case, [
    "no_error",                                  # no exception in the run or its spans
    {"exact_match": {"path": "answer"}},         # value defaults to case.expected
    {"contains": {"text": ["refund", "1234"], "case_sensitive": False}},
    {"regex": {"pattern": r"order \d{4}", "path": "answer"}},
    {"json_schema": {"schema": {"type": "object", "required": ["answer", "status"]}}},
]):
    print(result.evaluator, result.score, result.reason)
```

## Built-in trajectory checks
`tool_sequence`, `max_steps`, `cost_budget` and `latency_budget` check *how* the agent worked.
`tool_sequence` compares the tools called with `tools` (or the case's `expected.tools`) in
`strict` (exact list), `ordered` (default: in this order, extra calls allowed) or `subset`
(all called, any order) mode; repeats count, `forbidden: [...]` fails on any call to those tools.
Budgets score 1 within the limit and `limit / actual` above it; an unknown cost fails
`cost_budget` unless `on_unknown: pass`:

```python
import agentlens
from agentlens.evals import evaluate_case
from agentlens.schema import Case

@agentlens.wrap_tool
def lookup_order(order_id: str) -> dict:
    return {"id": order_id, "amount": 42}

@agentlens.wrap_tool
def issue_refund(order_id: str) -> str:
    return "refunded"

with agentlens.run("support") as trace:
    lookup_order("1234")
    issue_refund("1234")
    trace.output = "Refund issued"

case = Case(id="refund", input="refund 1234",
            expected={"tools": ["lookup_order", "issue_refund"]})
for result in evaluate_case(trace, case, [
    {"tool_sequence": {"mode": "ordered"}},      # tools from case.expected
    {"tool_sequence": {"mode": "strict", "forbidden": ["delete_account"]}},
    {"max_steps": 8},                            # agent_step spans, else LLM calls
    {"latency_budget": 5000},                    # milliseconds
    {"cost_budget": {"max_usd": 0.05, "on_unknown": "pass"}},  # no LLM calls here: unknown
]):
    print(result.evaluator, result.passed, result.reason)
```


For judgement calls (tone, helpfulness, faithfulness) use the [LLM judges](judges.md).
See also: [design §5](../design.md#5-evaluators), API: [`agentlens.evals`](../api/evals.md).
