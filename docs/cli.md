# CLI reference

`agentlens` (also `python -m agentlens`) runs suites, compares experiments and gates CI. With
an agent module and a suite like these (and the `gates.yaml` from [Gates](concepts/comparison.md#gates)):

```python
# echo_agent.py
import agentlens
from agentlens.llm import LLMRequest, Message, get_client

def run(question: str, model: str | None = None) -> str:
    llm = agentlens.wrap_llm(get_client(model or "mock"))   # "mock" echoes the question
    return llm.complete(LLMRequest(messages=[Message.user(question)])).content
```

```yaml
# suites/echo.yaml
name: echo
agent: echo_agent:run
repetitions: 3
defaults:
  evaluators: [no_error, exact_match]
cases:
  - {id: hi, input: "hi", expected: "hi"}
  - {id: bye, input: "bye", expected: "bye"}
```

```console
$ agentlens run suites/echo.yaml --name main --set-baseline      # run + mark as the baseline
$ agentlens run suites/echo.yaml --name pr-123 --model mock --gates gates.yaml
$ agentlens compare --candidate pr-123 --gates gates.yaml --json comparison.json
baseline   main (01a1d917-...) [baseline]
candidate  pr-123 (01a1d917-...)
cases      2 paired; paired bootstrap, 10000 resamples, 95% CI, seed 0

metric             n  baseline  candidate  delta     95% CI                status
pass_rate          2  1         1          +0        [+0, +0]              unchanged
...
gates passed: 2 passed, 0 failed, 0 skipped
$ agentlens report pr-123                    # or: report comparison.json, --format json
$ agentlens baseline show                    # baseline set|unset EXPERIMENT
$ agentlens experiments ls
$ agentlens traces ls --experiment pr-123    # traces show TRACE_ID [--json]
$ agentlens export traces.jsonl --experiment pr-123 && agentlens import traces.jsonl
```

| Command | What it does |
|---|---|
| `run SUITE [--name --model --repetitions --concurrency --timeout --gates --set-baseline -q --json PATH]` | Runs the suite, saves the experiment, prints a summary (per-run progress on stderr). `--gates` checks the absolute rules (`min`/`max`); `--set-baseline` marks the run as the suite's baseline unless a gate failed; `--json` also writes the report with the gate verdict. |
| `compare --candidate X [--baseline Y --gates --resamples --seed --confidence --json PATH --html PATH]` | Candidate vs baseline with bootstrap CIs. `--baseline` takes an id, a name (most recent) or `baseline` (the default: the marked baseline of the candidate's suite). `--json -` prints JSON instead of text; `--html` also writes the [HTML report](concepts/reports.md#html-report). |
| `report EXPERIMENT\|FILE [--format rich\|text\|json\|markdown\|html] [-o PATH] [--marker KEY]` | An experiment's totals, metrics and failing cases, or a saved comparison (`run`/`compare --json`) re-rendered. `markdown` is the PR-comment report (see [GitHub Action](github-action.md)); `html` is the [HTML report](concepts/reports.md#html-report) with the traces from the database. |
| `github comment --file REPORT.md [--marker KEY --pr N --repo OWNER/REPO]` | Posts the report on a pull request, or updates the comment carrying the marker (token in `$GITHUB_TOKEN`). |
| `baseline set\|unset EXPERIMENT`, `baseline show [SUITE]` | Manage the per-suite baseline. |
| `experiments ls`, `traces ls`, `traces show ID` | Browse the database. |
| `ui [--host --port --db --gates --open/--no-open]` | The [local web UI](web-ui.md) on `http://127.0.0.1:8765/`. |
| `export PATH`, `import PATH` | JSONL interchange (filters `--name/--experiment/--limit`; `import --experiment`). |

**Output:** on an interactive terminal `run`, `compare` and `report` print the coloured
**rich report**; piped, redirected, in CI logs or with `NO_COLOR` set they print the plain text
shown above, which is stable for scripts. See [Reports](concepts/reports.md).

`--db PATH` (or `$AGENTLENS_DB`) picks the database, default `.agentlens/agentlens.db`.
**Exit codes:** `0` success, `1` a gate failed, `2` usage or configuration error (bad arguments,
invalid suite or gates file, unknown experiment or trace, unreadable file), so a CI step can
tell "the agent regressed" from "the check is misconfigured". `agentlens --help` and
`agentlens COMMAND --help` document every option.
