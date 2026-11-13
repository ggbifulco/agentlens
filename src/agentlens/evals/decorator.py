"""``@agentlens.evaluator``: turn a plain function into a registered evaluator."""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import inspect
from collections.abc import Callable, Coroutine, Mapping
from typing import Any, overload

from pydantic import JsonValue

from agentlens.evals.base import DEFAULT_THRESHOLD, error_result, to_eval_result
from agentlens.evals.registry import EvaluatorRegistry, default_registry
from agentlens.schema import Case, EvalResult, Trace

__all__ = ["FunctionEvaluator", "evaluator"]

THRESHOLD_PARAM = "threshold"

_P = inspect.Parameter


class FunctionEvaluator:
    """An :class:`~agentlens.evals.Evaluator` backed by a function ``fn(trace, case, **params)``.

    Created by :func:`evaluator`. The object stays callable like the original
    function (handy in unit tests); :meth:`evaluate` calls it with the bound
    ``params`` and normalises the return value with
    :func:`~agentlens.evals.to_eval_result`. Any exception — from the function or
    from an unusable return value — becomes a failed :class:`EvalResult`
    (:func:`~agentlens.evals.error_result`), so one broken evaluator never
    crashes a suite.

    ``threshold`` is the pass threshold for bare float scores. It can also be set
    per spec (``{my_eval: {threshold: 0.8}}``); the function receives it only if
    it declares a ``threshold`` parameter itself.

    Async functions (``async def``) are supported: :meth:`aevaluate` awaits them
    and :meth:`evaluate` runs them to completion (in a worker thread when called
    from inside a running event loop).
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        params: Mapping[str, JsonValue] | None = None,
    ) -> None:
        if not callable(fn):
            raise TypeError(f"@evaluator needs a function, got {type(fn).__name__}")
        _check_threshold(threshold)
        # __name__/__doc__/__wrapped__..., but not fn.__dict__ (it could clobber our fields).
        functools.update_wrapper(self, fn, updated=())
        self.fn = fn
        self.name: str = name if name is not None else getattr(fn, "__name__", "evaluator")
        self.params: dict[str, JsonValue] = dict(params or {})
        self.threshold = threshold
        self.signature = _factory_signature(fn)
        self._fn_takes_threshold = THRESHOLD_PARAM in inspect.signature(fn).parameters
        self._is_async = inspect.iscoroutinefunction(fn)

    # -- configuration ---------------------------------------------------------

    def bind(self, **params: JsonValue) -> FunctionEvaluator:
        """A copy with ``params`` (merged over the current ones) bound for :meth:`evaluate`.

        This is the registry factory: ``{name: {k: v}}`` in a suite calls ``bind(k=v)``.
        A ``threshold`` param sets the pass threshold.
        """
        merged = {**self.params, **params}
        threshold = self.threshold
        if THRESHOLD_PARAM in merged:
            value = merged[THRESHOLD_PARAM]
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"threshold must be a number, got {value!r}")
            threshold = float(value)
            if not self._fn_takes_threshold:
                del merged[THRESHOLD_PARAM]
        return FunctionEvaluator(self.fn, name=self.name, threshold=threshold, params=merged)

    # -- Evaluator protocol ------------------------------------------------------

    def evaluate(self, trace: Trace, case: Case) -> EvalResult:
        """Run the function and normalise its result; never raises."""
        try:
            if self._is_async:
                value = _run_coroutine(self._call(trace, case))
            else:
                value = self._call(trace, case)
            return to_eval_result(self.name, value, threshold=self.threshold)
        except Exception as exc:
            return error_result(self.name, exc)

    async def aevaluate(self, trace: Trace, case: Case) -> EvalResult:
        """Async :meth:`evaluate`: awaits an async function, runs a sync one in a thread."""
        if not self._is_async:
            return await asyncio.to_thread(self.evaluate, trace, case)
        try:
            value = await self._call(trace, case)
            return to_eval_result(self.name, value, threshold=self.threshold)
        except Exception as exc:
            return error_result(self.name, exc)

    def _call(self, trace: Trace, case: Case) -> Any:
        kwargs = dict(self.params)
        if self._fn_takes_threshold:
            kwargs.setdefault(THRESHOLD_PARAM, self.threshold)
        return self.fn(trace, case, **kwargs)

    # -- plain-function behaviour -----------------------------------------------

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped function directly (bypassing normalisation)."""
        return self.fn(*args, **kwargs)

    def __repr__(self) -> str:
        params = f", params={self.params!r}" if self.params else ""
        return f"FunctionEvaluator({self.name!r}, threshold={self.threshold}{params})"


def _check_threshold(threshold: float) -> None:
    if not 0.0 <= threshold <= 1.0:  # also rejects NaN
        raise ValueError(f"threshold must be in [0, 1], got {threshold!r}")


def _factory_signature(fn: Callable[..., Any]) -> inspect.Signature:
    """Signature of ``bind``: ``fn``'s parameters after ``(trace, case)``, keyword-only.

    A ``threshold`` keyword is added when ``fn`` has none, so the scalar shorthand
    ``{my_eval: 0.8}`` of a parameterless function sets its pass threshold.
    """
    params = list(inspect.signature(fn).parameters.values())
    positional = [p for p in params[:2] if p.kind in (_P.POSITIONAL_ONLY, _P.POSITIONAL_OR_KEYWORD)]
    if len(positional) < 2 and not any(p.kind is _P.VAR_POSITIONAL for p in params[:3]):
        raise TypeError(
            f"evaluator function {getattr(fn, '__name__', fn)!r} must accept (trace, case) "
            "as its first two positional parameters"
        )
    extras: list[inspect.Parameter] = []
    var_kw: inspect.Parameter | None = None
    for param in params[len(positional) :]:
        if param.kind is _P.VAR_KEYWORD:
            var_kw = param
        elif param.kind in (_P.POSITIONAL_ONLY, _P.VAR_POSITIONAL):
            if param.kind is _P.POSITIONAL_ONLY and param.default is _P.empty:
                raise TypeError(
                    f"evaluator parameter {param.name!r} is positional-only and required; "
                    "parameters after (trace, case) are passed by keyword"
                )
        else:
            extras.append(param.replace(kind=_P.KEYWORD_ONLY))
    if all(p.name != THRESHOLD_PARAM for p in extras):
        extras.append(_P(THRESHOLD_PARAM, _P.KEYWORD_ONLY, default=None, annotation=float | None))
    if var_kw is not None:
        extras.append(var_kw)
    return inspect.Signature(extras)


def _run_coroutine(coro: Coroutine[Any, Any, Any]) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


@overload
def evaluator(fn: Callable[..., Any], /) -> FunctionEvaluator: ...


@overload
def evaluator(
    *,
    name: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    register: bool = True,
    replace: bool = False,
    registry: EvaluatorRegistry | None = None,
) -> Callable[[Callable[..., Any]], FunctionEvaluator]: ...


def evaluator(
    fn: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    register: bool = True,
    replace: bool = False,
    registry: EvaluatorRegistry | None = None,
) -> FunctionEvaluator | Callable[[Callable[..., Any]], FunctionEvaluator]:
    """Turn ``fn(trace, case, **params)`` into a registered :class:`FunctionEvaluator`.

    Usable bare or with options::

        @agentlens.evaluator
        def mentions_order(trace, case) -> bool:
            return "order" in str(trace.output)

        @agentlens.evaluator(name="short_answer", threshold=0.8)
        def short(trace, case, max_words: int = 50) -> tuple[float, str]:
            words = len(str(trace.output).split())
            return min(1.0, max_words / max(words, 1)), f"{words} words"

    The function may return an :class:`EvalResult`, a ``bool``, a score in
    ``[0, 1]`` (passes when ``>= threshold``), ``(score_or_bool, reason)`` or a
    mapping with ``score``/``passed`` (see :func:`~agentlens.evals.to_eval_result`).
    Exceptions become failed results instead of propagating.

    Parameters after ``(trace, case)`` are filled from the suite spec:
    ``{short_answer: {max_words: 20}}`` or, through the scalar shorthand,
    ``{short_answer: 20}`` (the first such parameter; ``threshold`` if there is none).

    Args:
        name: Registry/result name; defaults to the function name.
        threshold: Pass threshold for bare float scores.
        register: Add it to the registry so suites can refer to it by name.
        replace: Override an existing evaluator with the same name. Re-defining
            the *same* function (a re-run notebook cell) replaces it anyway.
        registry: Registry to use instead of :data:`~agentlens.evals.default_registry`.

    Raises:
        TypeError: if ``fn`` does not accept ``(trace, case)`` positionally.
        DuplicateEvaluatorError: if the name is taken by another evaluator.
    """

    def decorate(func: Callable[..., Any]) -> FunctionEvaluator:
        built = FunctionEvaluator(func, name=name, threshold=threshold)
        if register:
            target = registry if registry is not None else default_registry
            module = getattr(func, "__module__", None)
            qualname = getattr(func, "__qualname__", None)
            target.register(
                built.name,
                built.bind,
                replace=replace,
                signature=built.signature,
                origin=f"{module}.{qualname}",
            )
        return built

    if fn is not None:
        return decorate(fn)
    return decorate
