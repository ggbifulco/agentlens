"""The CI demo suite (``examples/ci_demo``) and the GitHub Action that runs it.

The action itself only runs on GitHub; these tests keep its pieces consistent:
the demo passes its gates, the action's inputs/outputs are wired, and the
workflow uses inputs that exist.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from agentlens.compare import evaluate_gates, load_gates
from agentlens.suite import load_suite, run_suite

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "examples" / "ci_demo"


def _yaml(path: Path) -> dict[Any, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


@pytest.fixture
def _isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(sys, "path", list(sys.path))
    yield
    sys.modules.pop("ci_demo_agent", None)


@pytest.mark.usefixtures("_isolated")
def test_demo_suite_passes_its_gates() -> None:
    result = run_suite(load_suite(DEMO / "suite.yaml"))
    assert len(result.runs) >= 10
    assert all(run.passed for run in result.runs), [r.case_id for r in result.runs if not r.passed]
    report = evaluate_gates(load_gates(DEMO / "gates.yaml"), run=result)
    assert report.passed
    assert report.skipped  # delta gates need the baseline the action provides


def test_action_wires_inputs_and_outputs() -> None:
    action = _yaml(ROOT / "action.yml")
    inputs = set(action["inputs"])
    assert {"suite", "gates", "baseline-ref", "github-token", "marker", "comment"} <= inputs
    assert action["inputs"]["suite"]["required"] is True
    assert action["runs"]["using"] == "composite"
    steps = action["runs"]["steps"]
    text = (ROOT / "action.yml").read_text(encoding="utf-8")
    assert set(re.findall(r"inputs\.([a-z-]+)", text)) <= inputs  # no typo'd input
    unused = {name for name in inputs if f"inputs.{name}" not in text}
    assert not unused
    check = next(step for step in steps if step.get("id") == "check")
    for name, output in action["outputs"].items():
        key = re.fullmatch(r"\$\{\{ steps\.check\.outputs\.([a-z-]+) \}\}", output["value"])
        assert key, name
        assert f'echo "{key.group(1)}=' in check["run"], name
    assert all("shell" in step for step in steps if "run" in step)
    # The comment is posted before the job fails on the gates.
    names = [step.get("name") for step in steps]
    assert names.index("Comment on the pull request") < names.index("Fail if a gate failed")


def test_workflow_runs_the_action_on_the_demo() -> None:
    workflow = _yaml(ROOT / ".github" / "workflows" / "agentlens.yml")
    assert "pull_request" in workflow[True]  # YAML 1.1 reads the `on` key as True
    assert workflow["permissions"] == {"contents": "read", "pull-requests": "write"}
    steps = next(iter(workflow["jobs"].values()))["steps"]
    step = next(s for s in steps if s.get("uses") == "./")
    assert set(step["with"]) <= set(_yaml(ROOT / "action.yml")["inputs"])
    for key in ("suite", "gates"):
        assert (ROOT / step["with"][key]).is_file()
