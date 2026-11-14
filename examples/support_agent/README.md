# Support-desk agent example

A small but real tool-calling agent for an online electronics store, plus a **golden dataset** of
27 cases to regression-test it with AgentLens. It runs **offline and deterministically** (no API
keys) through a scripted mock model, and unchanged against a real model.

```
examples/support_agent/
  agent.py             # run_agent(question, model=None): the tool-calling loop
  prompts/system_v1.txt  # the reviewed system prompt (default)
  prompts/system_v2.txt  # the "simplified" prompt of the regression story ($SUPPORT_PROMPT_VERSION=v2)
  tools.py             # the 6 tools, their JSON-schema specs, safe tool execution
  data.py              # in-memory customers, orders and FAQ ("today" is 2026-11-01)
  mock_policy.py       # the offline "model": mock:support
  cases.yaml           # the golden dataset
  suite.yaml           # deterministic suite (what CI runs)
  suite.judges.yaml    # same dataset + an llm_rubric judge (needs a judge model)
  gates.yaml           # the quality gates CI enforces on pull requests
```

## Run it

From the repository root (so that `examples.support_agent` is importable):

```bash
# The golden dataset, offline: every case passes, cost $0
agentlens run examples/support_agent/suite.yaml

# One question, printing the answer and the tools used
python -m examples.support_agent "I want a refund for order 1001"
# I have issued a refund of $49.99 for order 1001 (refund R-1001); it will reach your original
# payment method within 5 business days.
#
# tools: lookup_order, issue_refund | llm calls: 3
```

With a real model, pick it with `AGENTLENS_MODEL` or `--model` (any `get_client` spec):

```bash
export OPENAI_API_KEY=...
AGENTLENS_MODEL=openai:gpt-4o-mini agentlens run examples/support_agent/suite.yaml
agentlens run examples/support_agent/suite.yaml --model anthropic:claude-3-5-haiku-latest
agentlens run examples/support_agent/suite.yaml --model ollama:llama3.1   # local, free

# The judged suite: the same checks plus an LLM rubric on every answer
AGENTLENS_MODEL=openai:gpt-4o-mini AGENTLENS_JUDGE_MODEL=openai:gpt-4o-mini \
  agentlens run examples/support_agent/suite.judges.yaml
```

Real models word their answers differently, so the checks are deliberately tolerant
(case-insensitive `contains`/`regex` on facts such as amounts, dates and ticket numbers;
`tool_sequence` in `ordered` mode, which allows extra calls such as an initial `lookup_order`).
Expect most, not necessarily all, cases to pass with a small model: that gap is exactly what
AgentLens measures when you compare models.

## The agent

`run_agent(question, model=None)` is a plain loop: send the conversation and the tool specs to the
model; run the tool calls it asks for and append their results; stop at the first answer without
tool calls, or after `MAX_ITERATIONS` (8) model calls (then it apologises instead of looping
forever). The model is wrapped with `agentlens.wrap_llm` and each tool with `agentlens.wrap_tool`,
so `agentlens run` records every LLM and tool call in the trace. The model is `model=` (passed by
`agentlens run --model` or the suite), else `$AGENTLENS_MODEL`, else `mock:support`.

| Tool | What it does |
|---|---|
| `lookup_order(order_id)` | items, total, status, delivery date, customer |
| `issue_refund(order_id, amount, reason)` | refunds a delivered, not yet refunded order (at most its total) |
| `track_order(order_id)` | status, carrier, tracking number, ETA or delivery date |
| `update_address(order_id, new_address)` | only while the order is still processing |
| `escalate_to_human(reason, order_id)` | opens a ticket (`ESC-1001`, ...) |
| `search_faq(query)` | help-centre articles: returns, shipping, payment, warranty, ... |

Each run gets a fresh copy of the in-memory database, so refunds and address changes never leak
between runs (also when the suite runs cases concurrently). Tools return `{"error": ...}` for
business errors (unknown order, already shipped); an unknown tool, wrong arguments or a tool
exception also become an error result the model can recover from.

The **policy** — refunds only within 30 days of delivery, at most $200 without a human, escalate
when asked for a person — lives in the system prompt, `prompts/system_v1.txt`, not in the tools.
That is what makes the example useful for regression testing: a prompt edit can change behaviour.

## How the mock works

`mock:support` is a `MockLLM` whose script is a function of the request
(`mock_policy.support_policy`). On every call it re-reads the whole conversation, like a model:

1. **Intents** from the customer message: split into clauses (sentences, "and"/"also") and matched
   with keyword rules — refund, tracking, address change, store-policy question; a request for a
   human or manager escalates. An order number carries over to the next clause ("My order 1004
   arrived broken. I want my money back.").
2. **Progress** from the tool results already in the conversation: each intent is a small state
   machine ("look the order up; refund it if the policy allows, else escalate or explain"), and the
   policy emits the first tool call that has no result yet.
3. The **final answer**, one sentence per intent, built from the tool results.

The refund limit ("up to $200") and window ("within 30 days of delivery") are **parsed from the
system prompt**, so the mock reacts to prompt changes the way a model would. Try it: delete
"of up to $200" from `prompts/system_v1.txt` and run the suite again — `refund-over-limit` and
`multi-broken-over-limit` now refund the $899 laptop instead of escalating, and fail.

That change ships as `prompts/system_v2.txt`: set `SUPPORT_PROMPT_VERSION=v2` to run the agent
with it. [docs/regression-story.md](../../docs/regression-story.md) walks through how the
comparison and the gates in `gates.yaml` catch it, locally and in CI.

## The golden dataset

`cases.yaml` covers refunds (simple, paraphrased, over the limit, outside the window, already
refunded, not delivered, cancelled, unknown order, missing order number), tracking (shipped,
delayed, delivered, unknown, missing number), address changes (allowed, already shipped, missing
address), FAQ (returns, international shipping, payment, warranty), escalation, multi-step requests
and an off-topic question. Every case runs `no_error`, `max_steps: 6`, `tool_sequence` against
`expected.tools` (with `forbidden` tools for policy refusals) and `contains`/`regex` on the facts
the answer must state. Cases are tagged (`refund`, `policy`, `multi-step`, ...).
