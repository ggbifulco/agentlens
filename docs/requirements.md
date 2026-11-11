# AgentLens — Requirements

## 1. Problem

Teams shipping LLM agents change prompts, models, tools and orchestration code every week.
Each change can silently break behaviour: the agent calls the wrong tool, loops, gets more
expensive, or answers worse. Unit tests don't catch this because agent outputs are
non-deterministic and quality is fuzzy. Observability tools (LangSmith, Langfuse, Phoenix)
show *what happened*, but few tools make it easy to turn traces into a **regression test
suite that runs in CI and blocks a bad PR**.

**AgentLens** is a framework-agnostic, local-first Python toolkit to:

1. **Record** agent runs as structured traces (LLM calls, tool calls, steps, cost, latency).
2. **Replay** them deterministically (cassette-style) or live against a new model/prompt.
3. **Evaluate** them with deterministic checks and LLM-as-judge.
4. **Compare** against a baseline and detect statistically meaningful regressions.
5. **Report** in the terminal, as HTML, and as a GitHub PR comment.

## 2. Users & use cases

| Persona | Use case |
|---|---|
| AI engineer | "I changed the system prompt — did my agent get worse on my 50 golden cases?" |
| AI engineer | "Turn yesterday's production failure trace into a test case." |
| Tech lead | "Block merges that raise cost by >20% or drop task success by >5 points." |
| Researcher | "Compare model A vs model B on the same agent and dataset." |

## 3. Functional requirements

### Tracing (FR-T)
- **FR-T1** Python SDK to record a run: `with agentlens.run(...)`, `@agentlens.trace` decorator.
- **FR-T2** Span types: `agent_step`, `llm_call`, `tool_call`, `retrieval`, `custom`; spans nest.
- **FR-T3** Each span records inputs, outputs, timing, errors, token usage and estimated cost.
- **FR-T4** Traces persist to a local store (SQLite) and export/import as JSONL.
- **FR-T5** Recording overhead < 5 ms per span; recording failures never crash the agent.
- **FR-T6** (Should) Export spans following OpenTelemetry GenAI semantic conventions.

### Model adapters (FR-M)
- **FR-M1** A single `LLMClient` protocol used by judges, example agents and replay.
- **FR-M2** Adapters: `MockLLM` (scripted, deterministic), OpenAI-compatible
  (OpenAI, Ollama, vLLM, LM Studio), Anthropic.
- **FR-M3** Provider/model chosen by config or env vars; no code change to switch.
- **FR-M4** Tool/function-calling support normalised across adapters.

### Replay (FR-R)
- **FR-R1** *Cassette replay*: re-run an agent with recorded LLM/tool responses → fully deterministic.
- **FR-R2** *Live replay*: re-run the same inputs against a different model/prompt/config.
- **FR-R3** Report the first point where a replay diverges from the recording.

### Evaluation (FR-E)
- **FR-E1** Deterministic evaluators: exact/contains/regex, JSON-schema validity,
  tool-call sequence match (strict / subset / ordered), max steps, cost & latency budgets.
- **FR-E2** LLM-as-judge evaluators: rubric scoring (1–5), pass/fail with reasoning, pairwise A/B.
- **FR-E3** Custom evaluators via a simple Python interface / decorator.
- **FR-E4** Judge results are cached and reproducible (temperature 0, cached by input hash).

### Test suites & regression (FR-S)
- **FR-S1** Datasets of cases (input + expectations) defined in YAML or Python.
- **FR-S2** `agentlens run <suite>` executes the agent over the dataset with N repetitions.
- **FR-S3** Results stored as an *experiment*; any experiment can be marked as *baseline*.
- **FR-S4** `agentlens compare` diffs two experiments per case and in aggregate,
  with bootstrap confidence intervals to separate noise from real regressions.
- **FR-S5** Configurable gates (e.g. `success_rate >= 0.9`, `cost_delta <= +20%`) → non-zero exit code.
- **FR-S6** pytest plugin so suites can live next to normal tests.

### Reporting & CI (FR-C)
- **FR-C1** Rich terminal report.
- **FR-C2** Self-contained HTML report (trace viewer + comparison).
- **FR-C3** GitHub Action that runs a suite and posts/updates a PR comment with the diff.
- **FR-C4** (Could) Local web UI (`agentlens ui`) to browse traces and experiments.

### Demo (FR-D)
- **FR-D1** Example tool-using agent (e.g. support-desk agent with mock tools) + golden dataset.
- **FR-D2** A scripted "regression story": a prompt change that breaks tool usage and is caught in CI.

## 4. Non-functional requirements
- **NFR-1** Python ≥ 3.11, typed (mypy strict on `src/`), `ruff` clean.
- **NFR-2** Test coverage ≥ 85% on core packages; CI must run **without any API key** (MockLLM).
- **NFR-3** Framework-agnostic: no hard dependency on LangChain/LlamaIndex; optional integrations.
- **NFR-4** Local-first: works offline with SQLite; no SaaS required.
- **NFR-5** Secrets only from env vars; never written into traces (redaction hook).
- **NFR-6** Installable via `pip install agentlens` / `uv add agentlens`; documented public API.

## 5. Out of scope (v1)
Hosted multi-tenant backend, auth, real-time production monitoring dashboards, prompt management.

## 6. Definition of done (project)
- All *Must* requirements implemented and tested, CI green.
- README with quickstart, architecture diagram, GIF/screenshot of report, demo regression story.
- Tagged `v0.1.0` release.
