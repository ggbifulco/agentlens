"""Ask the support agent one question and print the answer and the tools it used.

python -m examples.support_agent "I want a refund for order 1001"
python -m examples.support_agent --model openai:gpt-4o-mini "Where is order 1002?"
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import agentlens

from .agent import run_agent


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point (returns the exit code)."""
    parser = argparse.ArgumentParser(prog="python -m examples.support_agent")
    parser.add_argument("question", help="the customer's message")
    parser.add_argument("--model", help="provider:model (default: $AGENTLENS_MODEL or mock)")
    args = parser.parse_args(argv)
    with agentlens.run("support-agent", input=args.question) as trace:
        trace.output = run_agent(args.question, model=args.model)
    print(trace.output)
    print(f"\ntools: {', '.join(trace.tool_calls) or '(none)'} | llm calls: {trace.n_steps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
