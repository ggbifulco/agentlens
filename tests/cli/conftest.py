"""CLI fixtures: a toy MockLLM agent, two suites and a CliRunner bound to a tmp database."""

from __future__ import annotations

import sys
import textwrap
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from agentlens.cli import app

AGENT = '''
import agentlens
from agentlens.llm import LLMRequest, Message, MockLLM


def _ask(question: str, answer) -> str:
    llm = agentlens.wrap_llm(MockLLM(lambda request: answer(request.messages[-1].content)))
    return llm.complete(LLMRequest(messages=[Message.user(question)])).content


def run_agent(question: str, model: str | None = None) -> str:
    """Baseline: shouts every answer."""
    return _ask(question, str.upper)


def new_agent(question: str) -> str:
    """Candidate: stops shouting at "hello 0"/"hello 1" (regressions), starts
    whispering "x" (an improvement), and crashes on "boom"."""
    if "boom" in question:
        raise ValueError("boom")
    quiet = question.startswith(("hello 0", "hello 1", "x"))
    return _ask(question, str.lower if quiet else str.upper)
'''

CASES = [(f"q{i}", f"hello {i}", f"HELLO {i}") for i in range(4)] + [("x", "x yz", "x yz")]


def suite_yaml(agent: str, cases: list[tuple[str, str, str]], repetitions: int = 2) -> str:
    lines = [
        "name: toy",
        f"agent: toy_agent:{agent}",
        f"repetitions: {repetitions}",
        "defaults:",
        "  evaluators: [no_error, exact_match]",
        "cases:",
    ]
    lines += [f'  - {{id: {cid}, input: "{inp}", expected: "{exp}"}}' for cid, inp, exp in cases]
    return "\n".join(lines) + "\n"


@dataclass
class Toy:
    """Files of the toy project and a runner for ``agentlens --db <db> ...``."""

    root: Path
    db: Path
    suite: Path
    """Baseline suite: run_agent, cases q0-q3, x and "gone"."""
    new_suite: Path
    """Candidate suite: new_agent, cases q0-q3, x and "boom" (added)."""
    runner: CliRunner

    def __call__(self, *args: str, db: bool = True) -> Result:
        prefix = ["--db", str(self.db)] if db else []
        return self.runner.invoke(app, [*prefix, *args])

    def write(self, name: str, text: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")
        return path


@pytest.fixture(autouse=True)
def _isolated_imports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Undo ``sys.path`` additions and forget the toy agent module after each test.

    Only modules loaded from ``tmp_path`` are dropped: forgetting library modules
    imported lazily (rich, by the help screen) would break their class identities.
    """
    monkeypatch.setattr(sys, "path", list(sys.path))
    yield
    for name, module in list(sys.modules.items()):
        if str(getattr(module, "__file__", None) or "").startswith(str(tmp_path)):
            del sys.modules[name]


@pytest.fixture
def toy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Toy:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AGENTLENS_DB", raising=False)
    monkeypatch.delenv("AGENTLENS_MODEL", raising=False)
    (tmp_path / "toy_agent.py").write_text(AGENT, encoding="utf-8")
    suite = tmp_path / "suite.yaml"
    suite.write_text(suite_yaml("run_agent", [*CASES, ("gone", "bye", "BYE")]))
    new_suite = tmp_path / "new_suite.yaml"
    new_suite.write_text(suite_yaml("new_agent", [*CASES, ("boom", "boom", "BOOM")]))
    return Toy(
        root=tmp_path,
        db=tmp_path / "runs.db",
        suite=suite,
        new_suite=new_suite,
        runner=CliRunner(),
    )


@pytest.fixture
def ran(toy: Toy) -> Toy:
    """``main`` (the baseline; only ``x`` fails) and ``pr-1`` (q0/q1 regress, x improves,
    boom crashes)."""
    first = toy("run", str(toy.suite), "--name", "main", "--set-baseline", "-q")
    assert first.exit_code == 0, first.output
    second = toy("run", str(toy.new_suite), "--name", "pr-1", "-q")
    assert second.exit_code == 0, second.output
    return toy
