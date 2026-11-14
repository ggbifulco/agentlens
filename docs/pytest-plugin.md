# pytest plugin

Suites can live next to your normal tests. Installing AgentLens registers a pytest plugin that
stays inactive until you use it. With the `echo_agent.py` and `suites/echo.yaml` from
[Command line](cli.md), each case becomes one test:

```console
$ pytest --agentlens suites/echo.yaml                 # repeat --agentlens for more suites
$ pytest --agentlens suites/echo.yaml -k hi           # select by case id, suite name or tag
$ pytest --agentlens suites/echo.yaml suites/echo.yaml::bye     # or by node id
$ pytest --agentlens suites/echo.yaml --agentlens-repetitions 5 --agentlens-db .agentlens/agentlens.db
```

The selected cases of a suite run once, as one experiment (with the suite's `concurrency`); a
case passes when **every** repetition passed. A failing case shows the agent's error or output
and each failed evaluator with its score and reason, not a traceback:

```text
____________________________ agentlens echo::shout _____________________________
agentlens case 'shout' of suite 'echo': 0/3 repetitions passed
  input: "hey"
  expected: "HEY"
  repetition 0: FAILED (trace 01a1d926-b458-76a4-9974-f33049ae7cdf)
    output: "hey"
    exact_match: score 0.00: output 'hey' != expected 'HEY'
  ...
```

`--agentlens-db PATH` saves the experiments (traces and scores) for `agentlens compare` and
`agentlens report`; `--agentlens-repetitions N` overrides the suites' repetitions. With
`--junitxml`, each case also records `agentlens_pass_rate`, `agentlens_score.<evaluator>` and
`agentlens_experiment` as test properties.

In Python tests, the `agentlens_run` fixture runs an agent on one case and `assert_passed` fails
with the same report:

```python
# test_agent.py
from agentlens.integrations.pytest_plugin import assert_passed
from echo_agent import run


def test_echo(agentlens_run):
    result = agentlens_run(run, {"id": "echo", "input": "hello", "expected": "hello"},
                           evaluators=["no_error", "exact_match"])
    assert_passed(result)          # on failure: each failed evaluator, score and reason
    assert result.pass_rate == 1.0
```

`agentlens_run(agent, case, *, evaluators=(), repetitions=None, model=None, timeout_s=None)`
returns a `CaseResult` (`runs`, `pass_rate`, `mean_scores()`). The agent can be a callable or an
import path. With `--agentlens-db`, the run is saved as an experiment of a suite named after the
test. The fixture is synchronous: call it from regular `def` tests.


API: [`agentlens.integrations.pytest_plugin`](api/integrations.md).
