from __future__ import annotations

import functools
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from agentlens.suite import (
    AgentImportError,
    check_import_path,
    import_module,
    is_async_agent,
    resolve_agent,
)
from tests.suite.conftest import WriteFile

AGENT_MODULE = """
class Agent:
    greeting = "hello"

    def respond(self, text):
        return f"{self.greeting} {text}"

    @staticmethod
    def shout(text):
        return text.upper()


agent = Agent()
NOT_CALLABLE = 42


def run(text):
    return f"echo {text}"
"""


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("mod:func", ("mod", "func")),
        ("pkg.sub.mod:obj.method", ("pkg.sub.mod", "obj.method")),
        (" mod : func ", ("mod", "func")),
    ],
)
def test_check_import_path_valid(spec: str, expected: tuple[str, str]) -> None:
    assert check_import_path(spec) == expected


@pytest.mark.parametrize(
    "spec", ["mod.func", "mod:", ":func", "mod:func:x", "1mod:func", "mod:fu-nc", "mod..x:f", ""]
)
def test_check_import_path_invalid(spec: str) -> None:
    with pytest.raises(ValueError, match=r"package\.module:callable"):
        check_import_path(spec)


def test_resolve_function_and_attribute_paths(tmp_path: Path, write: WriteFile) -> None:
    write("myagent.py", AGENT_MODULE)
    search = [tmp_path]
    assert resolve_agent("myagent:run", search_paths=search)("hi") == "echo hi"
    assert resolve_agent("myagent:agent.respond", search_paths=search)("bob") == "hello bob"
    assert resolve_agent("myagent:Agent.shout", search_paths=search)("hi") == "HI"
    instance_cls = resolve_agent("myagent:Agent", search_paths=search)
    assert isinstance(instance_cls, type)


def test_resolve_from_package(tmp_path: Path, write: WriteFile) -> None:
    write("agents/__init__.py", "")
    write("agents/support.py", "def run(x):\n    return x * 2\n")
    assert resolve_agent("agents.support:run", search_paths=[tmp_path])(3) == 6


def test_callable_is_returned_unchanged() -> None:
    def agent(text: str) -> str:
        return text

    assert resolve_agent(agent) is agent


def test_search_paths_are_appended_once(tmp_path: Path, write: WriteFile) -> None:
    write("myagent.py", AGENT_MODULE)
    entry = os.fspath(tmp_path.resolve())
    assert entry not in sys.path
    resolve_agent("myagent:run", search_paths=[tmp_path, tmp_path])
    assert sys.path.count(entry) == 1
    assert sys.path[-1] == entry  # appended: never shadows installed modules


def test_bad_format_is_agent_import_error() -> None:
    with pytest.raises(AgentImportError, match=r"package\.module:callable"):
        resolve_agent("not a path")


def test_missing_module(tmp_path: Path) -> None:
    with pytest.raises(AgentImportError, match="cannot import module 'nope_agent_xyz'") as info:
        resolve_agent("nope_agent_xyz:run", search_paths=[tmp_path])
    assert str(tmp_path) in str(info.value)
    assert isinstance(info.value.__cause__, ModuleNotFoundError)


def test_missing_submodule_of_existing_package(tmp_path: Path, write: WriteFile) -> None:
    write("agents/__init__.py", "")
    with pytest.raises(AgentImportError, match=r"no module named 'agents\.missing'"):
        resolve_agent("agents.missing:run", search_paths=[tmp_path])


def test_missing_module_without_search_paths() -> None:
    with pytest.raises(AgentImportError, match=r"searched sys\.path and sys\.path"):
        import_module("nope_agent_xyz")


def test_module_with_broken_import_inside(tmp_path: Path, write: WriteFile) -> None:
    write("broken.py", "import does_not_exist_xyz\n")
    with pytest.raises(AgentImportError, match="importing module 'broken' failed") as info:
        resolve_agent("broken:run", search_paths=[tmp_path])
    assert "does_not_exist_xyz" in str(info.value)


def test_module_raising_at_import(tmp_path: Path, write: WriteFile) -> None:
    write("raising.py", "raise RuntimeError('no config')\n")
    with pytest.raises(AgentImportError, match="RuntimeError: no config") as info:
        resolve_agent("raising:run", search_paths=[tmp_path])
    assert isinstance(info.value.__cause__, RuntimeError)


def test_missing_attribute(tmp_path: Path, write: WriteFile) -> None:
    write("myagent.py", AGENT_MODULE)
    with pytest.raises(AgentImportError, match="module 'myagent' has no attribute 'runn'"):
        resolve_agent("myagent:runn", search_paths=[tmp_path])
    with pytest.raises(AgentImportError, match="'myagent:agent' has no attribute 'reply'"):
        resolve_agent("myagent:agent.reply", search_paths=[tmp_path])
    with pytest.raises(AgentImportError, match=r"'myagent:Agent\.shout' has no attribute 'x'"):
        resolve_agent("myagent:Agent.shout.x", search_paths=[tmp_path])


def test_not_callable(tmp_path: Path, write: WriteFile) -> None:
    write("myagent.py", AGENT_MODULE)
    with pytest.raises(AgentImportError, match=r"is not callable \(got int"):
        resolve_agent("myagent:NOT_CALLABLE", search_paths=[tmp_path])
    with pytest.raises(AgentImportError, match=r"is not callable \(got str"):
        resolve_agent("myagent:agent.greeting", search_paths=[tmp_path])


def _sync_agent(text: str) -> str:
    return text


async def _async_agent(text: str) -> str:
    return text


class _AsyncCallable:
    async def __call__(self, text: str) -> str:
        return text

    async def respond(self, text: str) -> str:
        return text


class _SyncCallable:
    def __call__(self, text: str) -> str:
        return text


@pytest.mark.parametrize(
    ("agent", "expected"),
    [
        (_sync_agent, False),
        (_async_agent, True),
        (functools.partial(_async_agent), True),
        (functools.partial(functools.partial(_sync_agent)), False),
        (_AsyncCallable(), True),
        (_AsyncCallable().respond, True),
        (_SyncCallable(), False),
        (len, False),
    ],
)
def test_is_async_agent(agent: Any, expected: bool) -> None:
    assert is_async_agent(agent) is expected
