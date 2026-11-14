"""Build the demo HTML report shown in the README (``docs/images/html-report.png``).

Runs the ``examples/ci_demo`` suite twice with a small traced agent (an agent
step that searches the FAQ with a tool, then asks a ``MockLLM``): the baseline
as is, the candidate with a broken refund answer and an extra "review" step on
shipping questions.
Then compares them with the demo's gates and writes the HTML report::

    uv run python scripts/html_report_demo.py /tmp/report.html
    python scripts/screenshot_html_report.py /tmp/report.html docs/images/html-report.png

No API keys and no network: everything is a ``MockLLM``.
"""

from __future__ import annotations

import re
import sys
import time
from collections.abc import Callable
from pathlib import Path

import agentlens
from agentlens.compare import compare_runs, evaluate_gates, load_gates
from agentlens.llm import LLMRequest, Message, MockLLM
from agentlens.report import ComparisonReport, render_html
from agentlens.schema import Experiment
from agentlens.suite import load_suite, run_suite

DEMO = Path(__file__).resolve().parent.parent / "examples" / "ci_demo"
sys.path.insert(0, str(DEMO))

from ci_demo_agent import FALLBACK, FAQ  # noqa: E402


def make_agent(faq: dict[str, str], *, review: bool) -> Callable[[str], str]:
    llm = agentlens.wrap_llm(MockLLM(lambda request: str(request.messages[-1].content)))

    @agentlens.wrap_tool
    def search_faq(question: str) -> str | None:
        time.sleep(0.004)
        return next(
            (answer for pattern, answer in faq.items() if re.search(pattern, question)), None
        )

    def answer(question: str) -> str:
        with agentlens.span("answer", kind="agent_step", input=question) as step:
            found = search_faq(question) or FALLBACK
            time.sleep(0.002)
            reply = llm.complete(
                LLMRequest(
                    messages=[Message.system("You are a shop assistant."), Message.user(found)]
                )
            ).content
            if step is not None:
                step.output = reply
        if review and re.search(r"(?i)ship|deliver", question):
            with agentlens.span("review", kind="agent_step", input=reply) as step:
                time.sleep(0.003)
                reply = llm.complete(LLMRequest(messages=[Message.user(reply or "")])).content
                if step is not None:
                    step.output = reply
        return reply or ""

    return answer


def main(out: str) -> None:
    suite = load_suite(DEMO / "suite.yaml")
    broken = {k: ("Refunds are not possible." if "refund" in k else v) for k, v in FAQ.items()}
    base = run_suite(suite.model_copy(update={"agent": make_agent(FAQ, review=False)}))
    cand = run_suite(suite.model_copy(update={"agent": make_agent(broken, review=True)}))
    sha = {"git_sha": "3f2c9d1e8a7b"}
    comparison = compare_runs(
        base,
        cand,
        resamples=2000,
        seed=0,
        baseline_experiment=Experiment(name="main", suite=suite.name, is_baseline=True, config=sha),
        candidate_experiment=Experiment(
            name="pr-42", suite=suite.name, config={"git_sha": "9b1e04c7d2aa"}
        ),
    )
    gates = evaluate_gates(load_gates(DEMO / "gates.yaml"), comparison=comparison, run=cand)
    html = render_html(
        ComparisonReport(comparison=comparison, gates=gates),
        runs=cand.runs,
        baseline_runs=base.runs,
    )
    Path(out).write_text(html, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "report.html")
