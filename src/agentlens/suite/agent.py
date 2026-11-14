"""Resolving the suite's ``agent: "package.module:callable"`` import path.

The format is the one used by entry points and ASGI servers: a dotted module
path, a colon, then an attribute path inside the module — ``pkg.mod:run_agent``
or ``pkg.mod:Agent.run`` / ``pkg.mod:agent.respond`` for a method.

**Search path.** So that ``agentlens run suites/support.yaml`` finds the user's
own code without installing it, the current working directory and the suite
file's directory are *appended* to ``sys.path`` (when missing) before importing.
Appending, not prepending, means a stray ``json.py`` next to a suite cannot
shadow the standard library or an installed package; installed code wins on a
name clash. The entries stay in ``sys.path``, because imported modules may
import their siblings lazily later on.
"""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from types import ModuleType
from typing import Any

from agentlens.suite.errors import AgentImportError

__all__ = ["check_import_path", "import_module", "resolve_agent"]

PathLike = str | os.PathLike[str]


def _dotted(text: str) -> bool:
    return bool(text) and all(part.isidentifier() for part in text.split("."))


def check_import_path(spec: str) -> tuple[str, str]:
    """Split ``"pkg.mod:obj.attr"`` into ``("pkg.mod", "obj.attr")``.

    Raises:
        ValueError: if ``spec`` is not ``module:attribute`` with dotted
            identifiers on both sides.
    """
    module, sep, attr = spec.partition(":")
    if not sep or not _dotted(module.strip()) or not _dotted(attr.strip()):
        raise ValueError(
            f"agent import path must look like 'package.module:callable' "
            f"(or 'module:object.method'), got {spec!r}"
        )
    return module.strip(), attr.strip()


def _add_search_paths(search_paths: Iterable[PathLike]) -> None:
    for path in search_paths:
        entry = os.fspath(Path(path).resolve())
        if entry not in sys.path:
            sys.path.append(entry)


def import_module(name: str, *, search_paths: Iterable[PathLike] = ()) -> ModuleType:
    """Import module ``name``, first appending ``search_paths`` to ``sys.path``.

    Raises:
        AgentImportError: if the module does not exist, or importing it raised
            (the original exception is chained as ``__cause__``).
    """
    paths = [Path(p) for p in search_paths]
    _add_search_paths(paths)
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing and (name == missing or name.startswith(missing + ".")):
            where = ", ".join(str(p) for p in paths) or "sys.path"
            raise AgentImportError(
                f"cannot import module {name!r}: no module named {missing!r} "
                f"(searched sys.path and {where})"
            ) from exc
        raise AgentImportError(
            f"importing module {name!r} failed: {type(exc).__name__}: {exc}"
        ) from exc
    except Exception as exc:
        raise AgentImportError(
            f"importing module {name!r} failed: {type(exc).__name__}: {exc}"
        ) from exc


def resolve_agent(
    spec: str | Callable[..., Any],
    *,
    search_paths: Iterable[PathLike] = (),
) -> Callable[..., Any]:
    """Import the agent callable named by ``"package.module:callable"``.

    A callable is returned unchanged, so Python-defined suites can pass the
    function itself. Attribute paths walk into the module: ``mod:Agent.run``,
    ``mod:agent.respond``.

    Args:
        spec: The import path, or the agent callable itself.
        search_paths: Directories appended to ``sys.path`` (when missing)
            before importing — see the module docstring.

    Raises:
        AgentImportError: for a malformed path, a missing module or attribute,
            a module that fails to import, or a target that is not callable.
    """
    if callable(spec):
        return spec
    try:
        module_name, attr_path = check_import_path(spec)
    except ValueError as exc:
        raise AgentImportError(str(exc)) from None
    target: Any = import_module(module_name, search_paths=search_paths)
    walked = module_name
    for i, part in enumerate(attr_path.split(".")):
        try:
            target = getattr(target, part)
        except AttributeError:
            owner = f"module {module_name!r}" if i == 0 else repr(walked)
            raise AgentImportError(
                f"cannot resolve agent {spec!r}: {owner} has no attribute {part!r}"
            ) from None
        walked += (":" if i == 0 else ".") + part
    if not callable(target):
        raise AgentImportError(
            f"agent {spec!r} is a {type(target).__name__}, not a callable "
            "(expected a function, a method or an object with __call__)"
        )
    return target  # type: ignore[no-any-return]
