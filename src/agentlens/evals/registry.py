"""Evaluators by name: suite files say ``{max_steps: 8}``, the registry builds the object.

A *factory* is any callable ``factory(**params) -> Evaluator`` — usually the
evaluator class itself. :func:`get_evaluator` turns an
:class:`~agentlens.schema.EvaluatorSpec` (or its YAML spellings) into an
evaluator by calling the factory with the spec's parameters.

**Scalar shorthand rule.** ``{max_steps: 8}`` parses to ``params={"value": 8}``.
When the params are exactly ``{"value": x}`` and the factory has no parameter
called ``value``, ``x`` is passed to the factory's *primary parameter*: the one
named with ``register(..., primary=...)``, else the first parameter of the
factory's signature that can be passed by keyword. A factory that only takes
``**kwargs`` receives ``value=x`` unchanged.
"""

from __future__ import annotations

import difflib
import inspect
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

from pydantic import JsonValue, ValidationError

from agentlens.evals.base import Evaluator
from agentlens.evals.errors import (
    DuplicateEvaluatorError,
    EvaluatorConfigError,
    UnknownEvaluatorError,
)
from agentlens.schema import EvaluatorSpec

__all__ = [
    "EvaluatorFactory",
    "EvaluatorRegistry",
    "RegisteredEvaluator",
    "SpecLike",
    "available_evaluators",
    "default_registry",
    "get_evaluator",
    "register",
    "unregister",
]

EvaluatorFactory: TypeAlias = Callable[..., Evaluator]
"""``factory(**params) -> Evaluator``; an evaluator class is a valid factory."""

SpecLike: TypeAlias = EvaluatorSpec | str | Mapping[str, Any]
"""An :class:`EvaluatorSpec` or one of its YAML spellings (``"no_error"``, ``{"max_steps": 8}``)."""

_KEYWORD_KINDS = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)


@dataclass(frozen=True)
class RegisteredEvaluator:
    """One registry entry."""

    name: str
    factory: EvaluatorFactory
    signature: inspect.Signature | None
    """Signature used to check parameters; ``None`` when it cannot be introspected."""
    primary: str | None
    """Parameter that receives a scalar shorthand value (``{name: x}``)."""
    origin: str | None
    """Where the factory was defined (``module.qualname``); re-registering from the
    same origin (a re-run notebook cell, a reloaded module) replaces silently."""

    def build(self, params: Mapping[str, JsonValue] | None = None) -> Evaluator:
        """Call the factory with ``params`` (after applying the scalar shorthand rule).

        Raises:
            EvaluatorConfigError: if the parameters do not fit the factory, the
                factory rejects them (``TypeError``/``ValueError``) or it returns
                something that is not an :class:`Evaluator`.
        """
        kwargs = self._resolve(dict(params or {}))
        if self.signature is not None:
            try:
                self.signature.bind(**kwargs)
            except TypeError as exc:
                raise EvaluatorConfigError(
                    f"invalid parameters for evaluator {self.name!r}: {exc} "
                    f"(signature: {self.signature})"
                ) from exc
        try:
            built = self.factory(**kwargs)
        except (TypeError, ValueError) as exc:
            raise EvaluatorConfigError(
                f"cannot build evaluator {self.name!r} from {kwargs!r}: {exc}"
            ) from exc
        if not isinstance(built, Evaluator):
            raise EvaluatorConfigError(
                f"factory of evaluator {self.name!r} returned {type(built).__name__}, "
                "which has no 'name' and 'evaluate'"
            )
        return built

    def _resolve(self, params: dict[str, JsonValue]) -> dict[str, JsonValue]:
        if set(params) != {"value"} or self.primary in (None, "value"):
            return params
        return {self.primary: params["value"]}


def _signature(factory: EvaluatorFactory) -> inspect.Signature | None:
    try:
        return inspect.signature(factory)
    except (TypeError, ValueError):  # some builtins / C callables
        return None


def _primary(signature: inspect.Signature | None) -> str | None:
    if signature is None:
        return None
    for param in signature.parameters.values():
        if param.kind in _KEYWORD_KINDS:
            return param.name
    return None


def _origin(factory: EvaluatorFactory) -> str | None:
    module = getattr(factory, "__module__", None)
    qualname = getattr(factory, "__qualname__", None)
    if module is None or qualname is None:
        return None
    return f"{module}.{qualname}"


class EvaluatorRegistry:
    """A name -> factory mapping. Most code uses the module-level :data:`default_registry`."""

    def __init__(self) -> None:
        self._entries: dict[str, RegisteredEvaluator] = {}

    def register(
        self,
        name: str,
        factory: EvaluatorFactory,
        *,
        primary: str | None = None,
        replace: bool = False,
        signature: inspect.Signature | None = None,
        origin: str | None = None,
    ) -> RegisteredEvaluator:
        """Register ``factory`` under ``name``.

        Args:
            name: The name used in suite files and results.
            factory: ``factory(**params) -> Evaluator`` (an evaluator class works).
            primary: Parameter receiving a scalar shorthand (``{name: x}``);
                defaults to the first keyword-passable parameter of ``factory``.
            replace: Overwrite an existing entry instead of raising.
            signature: Override the introspected signature used to check params.
            origin: Override the definition site used to detect re-registration.

        Raises:
            DuplicateEvaluatorError: if ``name`` is taken by a factory from another
                origin and ``replace`` is false.
            ValueError: for an empty name or an unknown ``primary`` parameter.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("evaluator name must be a non-empty string")
        sig = signature if signature is not None else _signature(factory)
        if primary is not None and sig is not None:
            has_kwargs = any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            if primary not in sig.parameters and not has_kwargs:
                raise ValueError(f"factory of {name!r} has no parameter {primary!r}")
        entry = RegisteredEvaluator(
            name=name,
            factory=factory,
            signature=sig,
            primary=primary if primary is not None else _primary(sig),
            origin=origin if origin is not None else _origin(factory),
        )
        existing = self._entries.get(name)
        same_origin = (
            entry.origin is not None and existing is not None and (existing.origin == entry.origin)
        )
        if existing is not None and not replace and not same_origin:
            raise DuplicateEvaluatorError(
                f"evaluator {name!r} is already registered (from {existing.origin}); "
                "pass replace=True to override it"
            )
        self._entries[name] = entry
        return entry

    def unregister(self, name: str) -> None:
        """Remove ``name``. Raises :class:`UnknownEvaluatorError` if it is not registered."""
        self.entry(name)
        del self._entries[name]

    def entry(self, name: str) -> RegisteredEvaluator:
        """The entry for ``name``, or :class:`UnknownEvaluatorError` with suggestions."""
        try:
            return self._entries[name]
        except KeyError:
            names = self.names()
            raise UnknownEvaluatorError(
                name, names, difflib.get_close_matches(name, names, n=3)
            ) from None

    def get(self, spec: SpecLike, /, **params: JsonValue) -> Evaluator:
        """Build an evaluator from a spec (``"no_error"``, ``{"max_steps": 8}``, ...).

        Keyword ``params`` are merged over the spec's own parameters, so
        ``get("contains", text="refund")`` works without writing a spec.

        Raises:
            EvaluatorConfigError: for a malformed spec or parameters.
            UnknownEvaluatorError: if the name is not registered.
        """
        parsed = _parse_spec(spec)
        entry = self.entry(parsed.name)
        return entry.build({**entry._resolve(dict(parsed.params)), **params})

    def names(self) -> list[str]:
        """Registered names, sorted."""
        return sorted(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"EvaluatorRegistry({self.names()!r})"


def _parse_spec(spec: SpecLike) -> EvaluatorSpec:
    if isinstance(spec, EvaluatorSpec):
        return spec
    try:
        return EvaluatorSpec.model_validate(dict(spec) if isinstance(spec, Mapping) else spec)
    except ValidationError as exc:
        raise EvaluatorConfigError(f"invalid evaluator spec {spec!r}: {exc}") from exc


default_registry = EvaluatorRegistry()
"""The process-wide registry: built-ins and ``@evaluator`` functions land here."""


def register(
    name: str,
    factory: EvaluatorFactory,
    *,
    primary: str | None = None,
    replace: bool = False,
) -> RegisteredEvaluator:
    """Register ``factory`` in the :data:`default_registry`.

    See :meth:`EvaluatorRegistry.register`.
    """
    return default_registry.register(name, factory, primary=primary, replace=replace)


def unregister(name: str) -> None:
    """Remove ``name`` from the :data:`default_registry`."""
    default_registry.unregister(name)


def get_evaluator(spec: SpecLike, /, **params: JsonValue) -> Evaluator:
    """Build an evaluator from the :data:`default_registry` (see :meth:`EvaluatorRegistry.get`)."""
    return default_registry.get(spec, **params)


def available_evaluators() -> list[str]:
    """Names registered in the :data:`default_registry`, sorted."""
    return default_registry.names()
