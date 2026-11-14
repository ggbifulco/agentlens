"""Experiments: what a suite run records about itself, and the baseline mark (FR-S3).

An :class:`~agentlens.schema.Experiment` is one execution of a suite. Its
``config`` says *what* ran, so two experiments can be told apart later
(:func:`experiment_config`): the git commit, the model, the suite's content
hash, the repetitions, the AgentLens version and any user-provided values.

Exactly one experiment per suite can be the **baseline** that candidates are
compared against (``docs/design.md`` §7)::

    from agentlens.suite import get_baseline, set_baseline

    set_baseline(store, "support-agent-main")   # by id or name
    baseline = get_baseline(store, "support-agent")
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

from pydantic import JsonValue

from agentlens.llm.factory import MODEL_ENV
from agentlens.schema import Experiment
from agentlens.store.base import Store
from agentlens.suite.model import Suite

__all__ = [
    "ExperimentNotFoundError",
    "experiment_config",
    "find_experiment",
    "get_baseline",
    "git_info",
    "resolve_model",
    "set_baseline",
    "unset_baseline",
]

GIT_TIMEOUT_S = 5.0
"""Upper bound for each ``git`` call made while collecting the experiment config."""


class ExperimentNotFoundError(LookupError):
    """No stored experiment has the given id or name."""


def _git(args: list[str], cwd: Path | None) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def git_info(cwd: str | os.PathLike[str] | None = None) -> dict[str, JsonValue]:
    """The current git commit of ``cwd`` (default: the process cwd).

    Returns ``{"git_sha": <sha or None>, "git_dirty": <bool or None>}``.
    ``git_dirty`` is true when tracked files have uncommitted changes (untracked
    files are ignored). Both are ``None`` outside a repository or when ``git``
    is not installed; this never raises.
    """
    where = Path(cwd) if cwd is not None else None
    sha = _git(["rev-parse", "HEAD"], where)
    if not sha:
        return {"git_sha": None, "git_dirty": None}
    status = _git(["status", "--porcelain", "--untracked-files=no"], where)
    return {"git_sha": sha, "git_dirty": None if status is None else bool(status)}


def resolve_model(suite: Suite, override: str | None = None) -> str | None:
    """The model under test: ``override``, else ``suite.model``, else ``$AGENTLENS_MODEL``."""
    return override or suite.model or os.environ.get(MODEL_ENV) or None


def experiment_config(
    suite: Suite,
    *,
    model: str | None = None,
    repetitions: int | None = None,
    concurrency: int | None = None,
    timeout_s: float | None = None,
    extra: Mapping[str, JsonValue] | None = None,
) -> dict[str, JsonValue]:
    """The ``Experiment.config`` recorded for a run of ``suite``.

    Keys: ``git_sha``, ``git_dirty`` (:func:`git_info`, run in the suite's
    folder), ``model`` (:func:`resolve_model`), ``suite_hash``
    (:meth:`Suite.content_hash`), ``agent``, ``repetitions``, ``concurrency``,
    ``timeout_s`` (the effective values), ``agentlens_version`` and
    ``suite_metadata``. ``extra`` (user-provided, e.g. ``{"prompt_version":
    "v2"}``) is merged last and wins on a clash.
    """
    from agentlens import __version__

    agent = suite.model_dump(mode="json", include={"agent"})["agent"]
    config: dict[str, JsonValue] = {
        **git_info(suite.base_dir),
        "model": resolve_model(suite, model),
        "suite_hash": suite.content_hash(),
        "agent": agent,
        "repetitions": repetitions if repetitions is not None else suite.repetitions,
        "concurrency": concurrency if concurrency is not None else suite.concurrency,
        "timeout_s": timeout_s if timeout_s is not None else suite.timeout_s,
        "agentlens_version": __version__,
        "suite_metadata": dict(suite.metadata),
    }
    config.update(extra or {})
    return config


# -- baselines -------------------------------------------------------------------


def find_experiment(store: Store, experiment: str | Experiment) -> Experiment:
    """Look up a stored experiment by id, else by name (the most recent with that name).

    Raises:
        ExperimentNotFoundError: if nothing matches.
    """
    key = experiment.id if isinstance(experiment, Experiment) else experiment
    found = store.get_experiment(key)
    if found is not None:
        return found
    if not isinstance(experiment, Experiment):
        by_name = store.list_experiments(name=key)
        if by_name:
            return by_name[0]
    raise ExperimentNotFoundError(f"no experiment with id or name {key!r}")


def set_baseline(store: Store, experiment: str | Experiment) -> Experiment:
    """Mark ``experiment`` (id, name or object) as *the* baseline of its suite.

    Every other baseline of the same suite is unmarked first, so a suite has at
    most one baseline. (If the process dies between the two steps the suite is
    left with no baseline, never with two.)

    Returns:
        The stored experiment, now with ``is_baseline=True``.

    Raises:
        ExperimentNotFoundError: if the experiment is not stored.
    """
    target = find_experiment(store, experiment)
    for other in store.list_experiments(suite=target.suite, is_baseline=True):
        if other.id != target.id:
            store.save_experiment(other.model_copy(update={"is_baseline": False}))
    marked = target.model_copy(update={"is_baseline": True})
    store.save_experiment(marked)
    return marked


def unset_baseline(store: Store, experiment: str | Experiment) -> Experiment:
    """Remove the baseline mark from ``experiment`` (a no-op if it has none).

    Raises:
        ExperimentNotFoundError: if the experiment is not stored.
    """
    target = find_experiment(store, experiment)
    unmarked = target.model_copy(update={"is_baseline": False})
    if target.is_baseline:
        store.save_experiment(unmarked)
    return unmarked


def get_baseline(store: Store, suite: str | Suite) -> Experiment | None:
    """The baseline experiment of ``suite`` (a name or a :class:`Suite`), or ``None``.

    Should several be marked (e.g. written by hand), the most recent one wins.
    """
    name = suite.name if isinstance(suite, Suite) else suite
    baselines = store.list_experiments(suite=name, is_baseline=True)
    return baselines[0] if baselines else None
