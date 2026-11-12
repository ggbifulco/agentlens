"""Best-effort conversion of arbitrary Python values into JSON for spans.

Agent code passes anything to traced functions: pydantic models, dataclasses,
client objects, open files... The recorder must never fail because a value is
not JSON-serialisable (FR-T5), so we convert what pydantic knows how to
serialise and fall back to ``repr()`` for the rest.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

from pydantic import JsonValue
from pydantic_core import from_json, to_json

__all__ = ["bind_arguments", "format_exception", "to_json_value"]


def to_json_value(value: object) -> JsonValue:
    """Return a JSON-compatible copy of ``value``.

    Round-tripping through ``to_json``/``from_json`` guarantees the result is
    plain JSON (string keys, no tuples/sets/bytes), which is exactly what the
    ``JsonValue`` fields of :class:`~agentlens.schema.Span` accept. Values that
    pydantic cannot serialise are replaced by their ``repr()``; if even that fails, by a
    ``"<unserialisable TypeName>"`` placeholder.
    """
    try:
        return from_json(to_json(value, fallback=repr, serialize_unknown=True))  # type: ignore[no-any-return]
    except Exception:  # e.g. a __repr__ that raises
        return f"<unserialisable {type(value).__name__}>"


def bind_arguments(
    signature: inspect.Signature, args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> dict[str, Any]:
    """Map a call's positional and keyword arguments to parameter names.

    ``self``/``cls`` are dropped: the bound object is rarely useful as a span
    input and is often large or unserialisable. If the call does not match the
    signature (the function is about to raise ``TypeError`` itself), the raw
    ``args``/``kwargs`` are returned instead.
    """
    try:
        bound = signature.bind(*args, **kwargs)
    except TypeError:
        return {"args": list(args), "kwargs": dict(kwargs)}
    arguments = dict(bound.arguments)
    for implicit in ("self", "cls"):
        arguments.pop(implicit, None)
    return arguments


def format_exception(exc: BaseException) -> str:
    """Compact ``"Type: message"`` description stored in ``Span.error``."""
    message = str(exc)
    name = type(exc).__name__
    return f"{name}: {message}" if message else name
