# AgentLens

[![CI](https://github.com/ggbifulco/agentlens/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ggbifulco/agentlens/actions/workflows/ci.yml)

**Regression testing & evaluation for LLM agents.** Record traces, replay runs, score them with
LLM-as-judge and deterministic checks, and catch regressions in CI — before they ship.

> 🚧 Under active development. See the [roadmap](#roadmap) and the project board.

## Why
You changed a prompt, swapped a model or refactored a tool. Did your agent get worse?
Agent behaviour is non-deterministic and quality is fuzzy, so normal unit tests don't answer that.
AgentLens turns agent runs into a **statistically sound regression suite** that runs locally and in CI.

## Planned features
- 📼 **Record** agent runs as structured traces (LLM calls, tool calls, cost, latency) — framework-agnostic.
- 🔁 **Replay** deterministically from cassettes, or live against a new model/prompt.
- ⚖️ **Evaluate** with deterministic checks (tool sequence, JSON schema, budgets) and LLM-as-judge.
- 📊 **Compare** experiments with paired-bootstrap confidence intervals, not just averages.
- 🚦 **Gate** PRs in CI and post a report as a PR comment.
- 🔌 **Swappable models**: mock, OpenAI-compatible (OpenAI, Ollama, vLLM), Anthropic.

## Architecture
See [docs/architecture.md](docs/architecture.md) · [requirements](docs/requirements.md) ·
[design](docs/design.md) · [ADRs](docs/adr/).

## Roadmap
| Milestone | Scope |
|---|---|
| M0 Foundations | tooling, CI, board automation |
| M1 Tracing core | schema, recorder, SQLite store |
| M2 LLM adapters & replay | swappable clients, cassettes |
| M3 Evaluators | deterministic + LLM judges |
| M4 Suites & regression | runner, bootstrap comparison, gates, CLI |
| M5 Reporting & CI | terminal/HTML reports, GitHub Action |
| M6 Demo & integrations | example agent, regression story, UI, OTel |
| M7 Release | docs site, v0.1.0 on PyPI |

## How this repo is built
Developed in public by an AI agent working one backlog issue per session, with human review.
The process is documented in [AGENTS.md](AGENTS.md) and every session is logged in [docs/PROGRESS.md](docs/PROGRESS.md).

## License
Apache-2.0
