# ADR 0002: Own thin LLMClient protocol

- Status: Accepted
- Date: 2026-11-11

## Context
Judges, replay and the demo agent all need to call LLMs, and the user wants providers to be swappable (local vs paid).

## Decision
Define a small `LLMClient` Protocol with `LLMRequest`/`LLMResponse` models and write thin adapters: Mock, OpenAI-compatible (covers OpenAI, Ollama, vLLM, LM Studio via base_url), Anthropic. No LiteLLM/LangChain dependency in core.

## Consequences
Full control over normalisation (tool calls, usage) and replay hooks; small dependency surface. Cost: we maintain adapters ourselves. LiteLLM can be added later as an optional adapter.
