# Changelog

All notable changes to AgentLens are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the version is `0.x`,
minor releases may contain breaking changes; they are always listed here.

## [Unreleased]

## [0.1.0] - 2026-11-14

First public release: record agent runs as traces, replay them deterministically, score them with
deterministic checks and LLM judges, compare experiments with confidence intervals and fail CI on
regressions. Python 3.11-3.13, fully typed (`py.typed`), runs offline with mock models.

### Added

#### Tracing and storage
- Core schema: `Span`, `Trace`, `Usage`, `Case`, `EvalResult`, `Experiment` (pydantic v2) with
  time-ordered UUIDv7 ids and derived metrics (cost, tokens, latency, tool calls, steps) (#37).
- Tracing context and recorder: `agentlens.run()`, `@agentlens.trace`, `agentlens.span()`,
  `current_trace()` / `current_span()`; sync and async, contextvars-based, recorder failures never
  reach user code (#38).
- `wrap_llm` / `wrap_tool` auto-tracing of any LLM client and tool function (#43).
- Redaction of secrets before traces are stored: `PatternRedactor` (API keys, bearer tokens, AWS,
  GitHub, JWT, PEM keys, sensitive dict keys) and `redact_trace`, on by default, fail-closed (#41).
- `Store` protocol and `SQLiteStore` with versioned migrations and indexed filters (#39).
- JSONL export/import of traces (`export_jsonl`, `import_jsonl`), atomic and idempotent (#40).

#### LLM clients
- `LLMClient` protocol with provider-neutral `Message` / `LLMRequest` / `LLMResponse` /
  `ToolCall` / `ToolSpec` models and a scriptable `MockLLM` (#42).
- OpenAI-compatible adapter for OpenAI, Ollama, vLLM and LM Studio, with retries and backoff (#44).
- Anthropic Messages API adapter (tools, tool results, JSON mode emulation) (#45).
- `agentlens.get_client("provider:model")` factory configured from the environment, plus a packaged
  pricing table (`Pricing`, `$AGENTLENS_PRICING` overrides) that fills `usage.cost_usd` (#46).

#### Replay
- Cassettes: record and play back LLM and tool calls in `strict`, `lenient` and `fallthrough`
  modes (`Cassette`, `Cassette.from_trace`) (#47).
- Replay divergence detection with a unified diff of the changed request (`ReplayDivergence`,
  `find_divergence`), marked on the trace for reports (#48).

#### Evaluators
- `Evaluator` protocol, registry and the `@agentlens.evaluator` decorator for custom checks (#49).
- Deterministic output evaluators: `exact_match`, `contains`, `regex`, `json_schema`,
  `no_error` (#50).
- Trajectory evaluators: `tool_sequence` (strict / ordered / subset, forbidden tools),
  `max_steps`, `cost_budget`, `latency_budget` (#51).
- LLM-as-judge evaluators `llm_rubric` (1-5) and `llm_pass_fail` with retries on unparseable
  replies and an on-disk judge cache (#52).
- `PairwiseJudge`: A/B comparison judged in both orders to mitigate position bias (#53).

#### Suites and regression detection
- YAML suites and datasets (`.yaml`, `.jsonl`, `.json`) with defaults merged into cases and every
  validation problem reported with `file:line:column` (#54).
- Suite runner: repetitions, bounded async concurrency, per-run timeouts, failure isolation and
  experiment persistence with git sha, model and suite hash; baselines (#55).
- Comparator with paired bootstrap confidence intervals per metric and per-case regressions
  (`compare_experiments`, `compare_runs`) (#56).
- Gates: YAML thresholds (`min`/`max`, deltas, percentage deltas, regression counts) with CI exit
  codes (#58).

#### Command line, reports and CI
- `agentlens` CLI: `run`, `compare`, `report`, `baseline`, `experiments`, `traces`, `export`,
  `import`; exit codes 0 ok / 1 gate failed / 2 usage error (#59).
- pytest plugin: `pytest --agentlens suite.yaml` collects one test per case; `agentlens_run`
  fixture (#60).
- Rich terminal report for experiments and comparisons (#61).
- Markdown report and a composite GitHub Action that runs the suite on the PR base and head and
  posts (or updates) a PR comment; `agentlens github comment` (#62).
- Self-contained HTML report with an interactive trace viewer (`--format html`,
  `compare --html`) (#63).
- Local, read-only web UI to browse traces and experiments: `agentlens ui` (#68).
- OpenTelemetry exporter following the GenAI semantic conventions:
  `agentlens.integrations.otel`, optional extra `agentlens[otel]` (#67).

#### Example, documentation and project
- Example support-desk agent with mock tools, an offline scripted model and a 27-case golden
  dataset (`examples/support_agent`) (#64).
- End-to-end regression story checked in CI: a prompt change that drops the refund limit is
  caught by the gates (#66).
- Documentation site (mkdocs-material) with quickstart, concepts, guides, design docs and API
  reference: <https://ggbifulco.github.io/agentlens/> (#69).
- Project scaffolding (uv, src layout, ruff, mypy `--strict`, pytest with 100% coverage,
  pre-commit) and CI on Python 3.11-3.13 (#35, #36).
- Release workflow publishing to PyPI with trusted publishing and creating the GitHub release
  from this changelog (#70).

[Unreleased]: https://github.com/ggbifulco/agentlens/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ggbifulco/agentlens/releases/tag/v0.1.0
