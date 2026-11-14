# Suites

A suite names the agent under test (`module:callable`), the cases and the evaluators. Suite
`defaults` apply to every case; a case evaluator with the same name replaces the default one:

```yaml
# suites/support.yaml
name: support-agent
agent: support_agent:run_agent   # imported from the cwd or the suite's folder
repetitions: 3
defaults:
  evaluators: [no_error, {max_steps: 8}]
cases:
  - id: refund-simple
    input: "I want a refund for order 1234"
    expected: {tools: [lookup_order, issue_refund]}
    evaluators:
      - {tool_sequence: {mode: ordered}}
      - {max_steps: 4}               # overrides the default max_steps: 8
```

```python
from agentlens.suite import load_suite

suite = load_suite("suites/support.yaml")
agent = suite.load_agent()
for case in suite.resolved_cases():
    print(case.id, [spec.name for spec in case.evaluators], agent(case.input))
# refund-simple ['no_error', 'tool_sequence', 'max_steps'] ...
```

Every problem is reported at once, with its position: `suites/support.yaml:13:9:
cases[0].evaluators[1]: unknown evaluator 'max_step' (did you mean 'max_steps'?)`. Cases can live
in their own file (`cases: data/support.jsonl`), and `Suite(name=..., agent=run_agent,
cases=[...])` builds a suite in Python. `suite.content_hash()` fingerprints what the suite runs
(agent, model, repetitions, resolved cases), so experiments can tell whether the suite changed.

## Running a suite
`run_suite` runs every case `repetitions` times (at most `concurrency` at once, sync or `async`
agents), records each run as a trace, scores it and saves everything under a new experiment. An
agent that raises or exceeds `timeout_s` fails its run, not the suite. If the agent has a `model`
parameter it receives the model under test (`model=`, else the suite's `model:`, else
`$AGENTLENS_MODEL`):

```python
import agentlens
from agentlens.llm import LLMRequest, Message, get_client
from agentlens.store import SQLiteStore
from agentlens.suite import Suite, get_baseline, run_suite, set_baseline

def echo_agent(question: str, model: str | None = None) -> str:
    llm = agentlens.wrap_llm(get_client(model or "mock"))   # "mock" echoes the question
    return llm.complete(LLMRequest(messages=[Message.user(question)])).content

suite = Suite(name="echo", agent=echo_agent, model="mock", repetitions=3, concurrency=4,
              defaults={"evaluators": ["no_error", "exact_match"]},
              cases=[{"id": "hi", "input": "hi", "expected": "hi"}])

with SQLiteStore("runs.db") as store:
    result = run_suite(suite, store=store, name="main", config={"prompt_version": "v1"})
    print(f"{result.passed}/{result.total} passed ({result.pass_rate:.0%})")   # 3/3 passed (100%)
    print(result.experiment.config["git_sha"], result.experiment.config["suite_hash"][:12])
    set_baseline(store, "main")            # the one baseline of suite "echo"
    print(get_baseline(store, "echo").name)  # main
```

Each trace carries `case_id`, `repetition`, `suite` and `experiment` metadata; `arun_suite` is the
`async` version, and `progress=lambda result, done, total: ...` reports each finished run.


See also: [design §6](../design.md#6-suites-yaml), API: [`agentlens.suite`](../api/suite.md).
