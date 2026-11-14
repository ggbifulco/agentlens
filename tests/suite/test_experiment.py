from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

import agentlens
from agentlens.schema import Experiment
from agentlens.store import SQLiteStore
from agentlens.suite import (
    ExperimentNotFoundError,
    Suite,
    experiment_config,
    find_experiment,
    get_baseline,
    git_info,
    resolve_model,
    set_baseline,
    unset_baseline,
)
from agentlens.suite import experiment as experiment_module

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _suite(**kwargs: Any) -> Suite:
    data: dict[str, Any] = {
        "name": "toy",
        "agent": "toy_agent:run",
        "cases": [{"id": "a", "input": "x"}],
    }
    data.update(kwargs)
    return Suite.model_validate(data)


@pytest.fixture
def store() -> Iterator[SQLiteStore]:
    with SQLiteStore(":memory:") as db:
        yield db


def _save(store: SQLiteStore, name: str, suite: str = "toy", minutes: int = 0, **kw: Any) -> str:
    experiment = Experiment(
        name=name, suite=suite, created_at=T0 + timedelta(minutes=minutes), **kw
    )
    store.save_experiment(experiment)
    return experiment.id


# -- git ---------------------------------------------------------------------------

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@needs_git
def test_git_info_in_a_repository(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    (tmp_path / "f.txt").write_text("1")
    _git(tmp_path, "add", "f.txt")
    _git(tmp_path, "commit", "-q", "-m", "init")
    info = git_info(tmp_path)
    assert isinstance(info["git_sha"], str)
    assert len(info["git_sha"]) == 40
    assert info["git_dirty"] is False

    (tmp_path / "untracked.txt").write_text("ignored")
    assert git_info(tmp_path)["git_dirty"] is False
    (tmp_path / "f.txt").write_text("2")
    assert git_info(tmp_path)["git_dirty"] is True


@needs_git
def test_git_info_outside_a_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert git_info(tmp_path) == {"git_sha": None, "git_dirty": None}


def test_git_info_without_git(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError("git")

    monkeypatch.setattr(experiment_module.subprocess, "run", missing)
    assert git_info() == {"git_sha": None, "git_dirty": None}


def test_git_info_status_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    answers = iter(["abc123", None])
    monkeypatch.setattr(experiment_module, "_git", lambda args, cwd: next(answers))
    assert git_info() == {"git_sha": "abc123", "git_dirty": None}


# -- config ------------------------------------------------------------------------


def test_resolve_model_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENTLENS_MODEL", raising=False)
    assert resolve_model(_suite()) is None
    monkeypatch.setenv("AGENTLENS_MODEL", "mock:env")
    assert resolve_model(_suite()) == "mock:env"
    assert resolve_model(_suite(model="mock:suite")) == "mock:suite"
    assert resolve_model(_suite(model="mock:suite"), "mock:arg") == "mock:arg"


def test_experiment_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        experiment_module, "git_info", lambda cwd: {"git_sha": "abc", "git_dirty": False}
    )
    suite = _suite(model="mock:m", repetitions=3, timeout_s=2.5, metadata={"team": "x"})
    assert experiment_config(suite, extra={"prompt_version": "v2", "model": "pinned"}) == {
        "git_sha": "abc",
        "git_dirty": False,
        "model": "pinned",
        "suite_hash": suite.content_hash(),
        "agent": "toy_agent:run",
        "repetitions": 3,
        "concurrency": 1,
        "timeout_s": 2.5,
        "agentlens_version": agentlens.__version__,
        "suite_metadata": {"team": "x"},
        "prompt_version": "v2",
    }
    overridden = experiment_config(
        suite, model="mock:o", repetitions=1, concurrency=4, timeout_s=9.0
    )
    assert overridden["model"] == "mock:o"
    assert (overridden["repetitions"], overridden["concurrency"], overridden["timeout_s"]) == (
        1,
        4,
        9.0,
    )


def test_experiment_config_runs_git_in_the_suite_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Any] = []
    monkeypatch.setattr(experiment_module, "git_info", lambda cwd: seen.append(cwd) or {})
    suite = _suite()
    suite._source = tmp_path / "suite.yaml"
    experiment_config(suite)
    assert seen == [tmp_path]


# -- baselines ---------------------------------------------------------------------


def test_find_experiment_by_id_name_or_object(store: SQLiteStore) -> None:
    old = _save(store, "main", minutes=0)
    new = _save(store, "main", minutes=5)
    assert find_experiment(store, old).id == old
    assert find_experiment(store, "main").id == new  # most recent with that name
    stored = store.get_experiment(old)
    assert stored is not None
    assert find_experiment(store, stored).id == old


def test_find_experiment_missing(store: SQLiteStore) -> None:
    with pytest.raises(ExperimentNotFoundError, match="nope"):
        find_experiment(store, "nope")
    with pytest.raises(ExperimentNotFoundError):
        find_experiment(store, Experiment(name="unsaved", suite="toy"))


def test_set_baseline_keeps_one_per_suite(store: SQLiteStore) -> None:
    first = _save(store, "first", minutes=0)
    second = _save(store, "second", minutes=1)
    other_suite = _save(store, "other", suite="elsewhere", minutes=2, is_baseline=True)

    assert get_baseline(store, "toy") is None
    marked = set_baseline(store, first)
    assert marked.is_baseline
    assert get_baseline(store, "toy") == marked

    set_baseline(store, "second")
    baseline = get_baseline(store, "toy")
    assert baseline is not None
    assert baseline.id == second
    first_stored = store.get_experiment(first)
    assert first_stored is not None
    assert not first_stored.is_baseline
    assert len(store.list_experiments(suite="toy", is_baseline=True)) == 1
    # Other suites are untouched.
    other = get_baseline(store, "elsewhere")
    assert other is not None
    assert other.id == other_suite

    # Re-marking the current baseline is idempotent.
    set_baseline(store, second)
    assert [e.id for e in store.list_experiments(suite="toy", is_baseline=True)] == [second]


def test_get_baseline_accepts_a_suite(store: SQLiteStore) -> None:
    exp = _save(store, "main")
    set_baseline(store, exp)
    baseline = get_baseline(store, _suite())
    assert baseline is not None
    assert baseline.id == exp


def test_get_baseline_most_recent_wins(store: SQLiteStore) -> None:
    _save(store, "a", minutes=0, is_baseline=True)
    newer = _save(store, "b", minutes=1, is_baseline=True)
    baseline = get_baseline(store, "toy")
    assert baseline is not None
    assert baseline.id == newer


def test_unset_baseline(store: SQLiteStore) -> None:
    exp = _save(store, "main")
    set_baseline(store, exp)
    unmarked = unset_baseline(store, "main")
    assert not unmarked.is_baseline
    assert get_baseline(store, "toy") is None
    assert not unset_baseline(store, exp).is_baseline  # no-op when not a baseline
    with pytest.raises(ExperimentNotFoundError):
        unset_baseline(store, "missing")
