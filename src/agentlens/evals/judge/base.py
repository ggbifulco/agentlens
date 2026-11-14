"""The machinery every LLM judge shares: client resolution, the call, parsing, retry, cache.

:class:`LLMJudge` is an abstract evaluator. A concrete judge (``llm_rubric``,
``llm_pass_fail``, the pairwise judge of #19) only says *what to ask*
(:meth:`LLMJudge.build_prompt`), *how to read the answer*
(:meth:`LLMJudge.parse_reply`) and *what result it means*
(:meth:`LLMJudge.make_result`). The base class does the rest, the same way in
:meth:`~LLMJudge.evaluate` and :meth:`~LLMJudge.aevaluate`:

1. compute the cache key ``sha256(judge model, rendered prompt, output)`` and
   return the cached verdict on a hit (no LLM call);
2. otherwise call the judge with ``temperature=0`` and
   ``response_format={"type": "json_object"}``;
3. extract the JSON object from the reply and parse it; on a parse error ask
   again (up to ``retries`` times), showing the model its bad reply;
4. cache the raw reply only once it parsed, then build the :class:`EvalResult`.

A reply that still cannot be parsed gives a failed result whose ``details`` hold
the raw reply and the error. LLM errors (network, auth) propagate and become an
evaluator-error result through :func:`~agentlens.evals.run_evaluator`.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from pydantic import JsonValue

from agentlens.evals.errors import EvaluatorConfigError
from agentlens.evals.judge.cache import CacheSpec, JudgeCache, judge_cache_key, resolve_cache
from agentlens.evals.judge.parse import JudgeParseError, extract_json_object
from agentlens.llm import LLMClient, LLMConfigError, LLMRequest, LLMResponse, Message
from agentlens.llm.factory import MODEL_ENV, get_client, parse_spec
from agentlens.schema import Case, EvalResult, Trace, Usage

__all__ = [
    "JSON_RESPONSE_FORMAT",
    "JUDGE_MODEL_ENV",
    "JUDGE_TEMPERATURE",
    "MAX_RETRIES",
    "JudgePrompt",
    "JudgeRun",
    "LLMJudge",
    "value_text",
]

JUDGE_MODEL_ENV = "AGENTLENS_JUDGE_MODEL"
"""Judge model spec used when a judge has no ``model``/``client`` (falls back to
``$AGENTLENS_MODEL``)."""

JUDGE_TEMPERATURE = 0.0
"""Judges always sample at temperature 0 (FR-E4): as deterministic as the provider allows."""

JSON_RESPONSE_FORMAT: dict[str, JsonValue] = {"type": "json_object"}
"""Requested from every judge call (emulated by an instruction on Anthropic)."""

MAX_RETRIES = 5
"""Upper bound for a judge's ``retries`` parameter."""

RAW_REPLY_CHARS = 4000
"""Longest raw reply kept in ``details`` of an unparseable result."""

_RETRY_MESSAGE = (
    "Your previous reply could not be used: {error}. "
    "Reply again with only the JSON object in the requested format, and nothing else."
)

VerdictT = TypeVar("VerdictT")


def value_text(value: JsonValue) -> str:
    """How an input/output/expected value is shown to the judge.

    Strings as is, ``None`` as ``(none)``, anything else as JSON (sorted keys,
    non-ASCII kept) — the same text whatever the dict order, so cache keys are stable.
    """
    if value is None:
        return "(none)"
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


@dataclass(frozen=True)
class JudgePrompt:
    """A rendered judge prompt plus the output text that goes into the cache key."""

    system: str
    user: str
    output: str

    def messages(self) -> list[Message]:
        """The conversation sent to the judge."""
        return [Message.system(self.system), Message.user(self.user)]

    @property
    def text(self) -> str:
        """System and user prompt together (the ``prompt`` part of the cache key)."""
        return f"{self.system}\n\n{self.user}"


@dataclass
class JudgeRun(Generic[VerdictT]):
    """What happened when a judge was asked: the verdict (or the parse error) and metadata."""

    judge_model: str
    cache_key: str | None
    verdict: VerdictT | None = None
    error: JudgeParseError | None = None
    raw_reply: str | None = None
    cached: bool = False
    attempts: int = 0
    usage: Usage | None = None
    replies: list[str] = field(default_factory=list)

    def details(self) -> dict[str, JsonValue]:
        """Metadata every judge result carries in ``details``."""
        info: dict[str, JsonValue] = {
            "judge_model": self.judge_model,
            "cached": self.cached,
            "attempts": self.attempts,
        }
        if self.cache_key is not None:
            info["cache_key"] = self.cache_key
        if self.usage is not None:
            info["judge_usage"] = self.usage.model_dump(mode="json")
        return info


def _add_usage(total: Usage | None, extra: Usage | None) -> Usage | None:
    if extra is None:
        return total
    if total is None:
        return extra
    cost = (
        None
        if total.cost_usd is None or extra.cost_usd is None
        else total.cost_usd + extra.cost_usd
    )
    return Usage(
        input_tokens=total.input_tokens + extra.input_tokens,
        output_tokens=total.output_tokens + extra.output_tokens,
        cost_usd=cost,
    )


def _label(client: LLMClient) -> str:
    parts = [getattr(client, "provider", None), getattr(client, "model", None)]
    text = ":".join(p for p in parts if isinstance(p, str) and p)
    return text or type(client).__name__


class LLMJudge(ABC, Generic[VerdictT]):
    """Base class of LLM-as-judge evaluators (``docs/design.md`` §5, FR-E2/FR-E4).

    Args:
        model: Judge model spec (``"openai:gpt-4o-mini"``, see
            :func:`~agentlens.llm.get_client`). Default: ``$AGENTLENS_JUDGE_MODEL``,
            else ``$AGENTLENS_MODEL``. Resolved on the first evaluation, so building
            a judge (e.g. loading a suite) needs no environment.
        client: An :class:`~agentlens.llm.LLMClient` to use instead of ``model``
            (tests pass a :class:`~agentlens.llm.MockLLM`). Not both.
        cache: Where verdicts are cached (see :func:`~agentlens.evals.judge.resolve_cache`):
            default ``$AGENTLENS_JUDGE_CACHE`` or ``.agentlens/judge_cache/`` on disk;
            ``False``/``"off"`` disables it, ``"memory"`` keeps it in the process.
        retries: Extra attempts after an unparseable reply (``0``-``5``, default 1).
        max_tokens: Cap on the judge's reply (``None``: the client's default).

    Raises:
        EvaluatorConfigError: for invalid parameters.
    """

    name: str = "llm_judge"

    def __init__(
        self,
        *,
        model: str | None = None,
        client: LLMClient | None = None,
        cache: CacheSpec = None,
        retries: int = 1,
        max_tokens: int | None = None,
    ) -> None:
        if model is not None and client is not None:
            raise EvaluatorConfigError("pass either model or client to a judge, not both")
        if model is not None:
            if not isinstance(model, str) or not model.strip():
                raise EvaluatorConfigError(
                    f"model must be a '<provider>:<model>' spec, got {model!r}"
                )
            try:
                parse_spec(model)
            except LLMConfigError as exc:
                raise EvaluatorConfigError(str(exc)) from exc
        if client is not None and not isinstance(client, LLMClient):
            raise EvaluatorConfigError(
                f"client must be an LLMClient (complete/acomplete), got {type(client).__name__}"
            )
        if (
            isinstance(retries, bool)
            or not isinstance(retries, int)
            or not 0 <= retries <= MAX_RETRIES
        ):
            raise EvaluatorConfigError(
                f"retries must be an integer 0-{MAX_RETRIES}, got {retries!r}"
            )
        if max_tokens is not None and (
            isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1
        ):
            raise EvaluatorConfigError(f"max_tokens must be a positive integer, got {max_tokens!r}")
        try:
            self.cache: JudgeCache | None = resolve_cache(cache)
        except TypeError as exc:
            raise EvaluatorConfigError(str(exc)) from exc
        self.model = model
        self.retries = retries
        self.max_tokens = max_tokens
        self._client = client

    # -- client ----------------------------------------------------------------

    def _spec(self) -> str:
        spec = self.model or os.environ.get(JUDGE_MODEL_ENV) or os.environ.get(MODEL_ENV)
        if not spec or not spec.strip():
            raise LLMConfigError(
                f"no judge model configured: pass model='<provider>:<model>' (or client=...) "
                f"or set {JUDGE_MODEL_ENV} or {MODEL_ENV}"
            )
        return spec

    @property
    def judge_model(self) -> str:
        """The judge model's label, the ``judge model`` part of the cache key.

        ``"<provider>:<model>"`` for a spec; for an injected client its
        ``provider``/``model`` attributes (or its class name).

        Raises:
            LLMConfigError: if no model is configured anywhere.
        """
        if self._client is not None:
            return _label(self._client)
        provider, model = parse_spec(self._spec())
        return f"{provider}:{model}"

    @property
    def client(self) -> LLMClient:
        """The judge's LLM client, built from the model spec on first use.

        Raises:
            LLMConfigError: if no model is configured or the spec is invalid.
        """
        if self._client is None:
            self._client = get_client(self._spec())
        return self._client

    # -- what concrete judges define -------------------------------------------

    @abstractmethod
    def build_prompt(self, trace: Trace, case: Case) -> JudgePrompt:
        """Render the prompt for ``trace`` (the agent's run on ``case``)."""

    @abstractmethod
    def parse_reply(self, data: dict[str, JsonValue]) -> VerdictT:
        """Read the verdict out of the judge's JSON object.

        Raises:
            JudgeParseError: if the object does not hold a valid verdict.
        """

    @abstractmethod
    def make_result(self, verdict: VerdictT, run: JudgeRun[VerdictT]) -> EvalResult:
        """Turn a parsed verdict into the :class:`EvalResult`."""

    # -- the shared flow ---------------------------------------------------------

    def _request(self, messages: list[Message]) -> LLMRequest:
        return LLMRequest(
            messages=messages,
            temperature=JUDGE_TEMPERATURE,
            max_tokens=self.max_tokens,
            response_format=dict(JSON_RESPONSE_FORMAT),
        )

    def _parse(self, raw: str | None) -> VerdictT:
        return self.parse_reply(extract_json_object(raw))

    def _start(self, prompt: JudgePrompt) -> JudgeRun[VerdictT]:
        judge_model = self.judge_model
        key = (
            judge_cache_key(judge_model, prompt.text, prompt.output)
            if self.cache is not None
            else None
        )
        run: JudgeRun[VerdictT] = JudgeRun(judge_model=judge_model, cache_key=key)
        if self.cache is None or key is None:
            return run
        entry = self.cache.get(key)
        reply = entry.get("reply") if entry is not None else None
        if isinstance(reply, str):
            try:
                run.verdict = self._parse(reply)
            except JudgeParseError:
                return run  # stale or foreign entry: ask the judge again
            run.raw_reply = reply
            run.cached = True
        return run

    def _receive(
        self, run: JudgeRun[VerdictT], messages: list[Message], response: LLMResponse
    ) -> list[Message] | None:
        """Record one reply; the messages for a retry, or ``None`` when done."""
        raw = response.content or ""
        run.attempts += 1
        run.raw_reply = raw
        run.replies.append(raw)
        run.usage = _add_usage(run.usage, response.usage)
        try:
            run.verdict = self._parse(raw)
        except JudgeParseError as exc:
            run.error = exc
            if run.attempts > self.retries:
                return None
            retry = Message.user(_RETRY_MESSAGE.format(error=exc))
            return [*messages, Message.assistant(raw), retry]
        run.error = None
        if self.cache is not None and run.cache_key is not None:
            self.cache.set(
                run.cache_key,
                {"reply": raw, "judge_model": run.judge_model, "model": response.model},
            )
        return None

    def run_judge(self, prompt: JudgePrompt) -> JudgeRun[VerdictT]:
        """Ask the judge (or the cache) about ``prompt``; never raises on a bad reply."""
        run = self._start(prompt)
        if run.cached:
            return run
        client = self.client
        messages: list[Message] | None = prompt.messages()
        while messages is not None:
            messages = self._receive(run, messages, client.complete(self._request(messages)))
        return run

    async def arun_judge(self, prompt: JudgePrompt) -> JudgeRun[VerdictT]:
        """Async :meth:`run_judge`."""
        run = self._start(prompt)
        if run.cached:
            return run
        client = self.client
        messages: list[Message] | None = prompt.messages()
        while messages is not None:
            response = await client.acomplete(self._request(messages))
            messages = self._receive(run, messages, response)
        return run

    def _result(self, run: JudgeRun[VerdictT]) -> EvalResult:
        if run.verdict is not None:
            return self.make_result(run.verdict, run)
        error = run.error or JudgeParseError("the judge gave no verdict")
        raw = run.raw_reply or ""
        details = run.details()
        details["error"] = {"type": type(error).__name__, "message": str(error)}
        details["raw_reply"] = raw if len(raw) <= RAW_REPLY_CHARS else raw[:RAW_REPLY_CHARS] + "…"
        tries = f"{run.attempts} attempt{'s' if run.attempts != 1 else ''}"
        return EvalResult(
            evaluator=self.name,
            score=0.0,
            passed=False,
            reason=f"judge reply could not be parsed after {tries}: {error}",
            details=details,
        )

    def evaluate(self, trace: Trace, case: Case) -> EvalResult:
        """Judge ``trace`` synchronously (see the module docstring for the flow)."""
        return self._result(self.run_judge(self.build_prompt(trace, case)))

    async def aevaluate(self, trace: Trace, case: Case) -> EvalResult:
        """Judge ``trace`` with the client's async API."""
        return self._result(await self.arun_judge(self.build_prompt(trace, case)))
