"""Export AgentLens traces as OpenTelemetry spans (FR-T6), following the GenAI conventions.

Install the extra (``pip install "agentlens[otel]"``), configure OpenTelemetry as
usual (an OTLP exporter to Jaeger, Grafana Tempo, Langfuse, Phoenix, ...) and pass
an :class:`OTelExporter` as the trace sink, or call :func:`export_trace` on
traces you already have::

    import agentlens
    from agentlens.integrations.otel import OTelExporter, fan_out
    from agentlens.store import SQLiteStore

    exporter = OTelExporter()                 # global TracerProvider by default
    with SQLiteStore() as store, agentlens.run(
        "support", sink=fan_out(store.save_trace, exporter)
    ):
        ...

**Mapping** (OpenTelemetry semantic conventions for generative AI,
``gen_ai.*``):

=====================  ==========================  ========  =================================
AgentLens              OTel span name              kind      main attributes
=====================  ==========================  ========  =================================
the trace (root)       ``invoke_agent {name}``     INTERNAL  ``gen_ai.operation.name``,
                                                             ``gen_ai.agent.name``
``agent_step``         ``invoke_agent {name}``     INTERNAL  same
``llm_call``           ``chat {model}``            CLIENT    ``gen_ai.provider.name``,
                                                             ``gen_ai.request.model``,
                                                             ``gen_ai.request.temperature``,
                                                             ``gen_ai.request.max_tokens``,
                                                             ``gen_ai.response.model``,
                                                             ``gen_ai.usage.input_tokens``,
                                                             ``gen_ai.usage.output_tokens``
``tool_call``          ``execute_tool {tool}``     INTERNAL  ``gen_ai.tool.name``
``retrieval``/custom   the span name               INTERNAL  ``agentlens.span.kind``
=====================  ==========================  ========  =================================

Every span also carries ``agentlens.trace_id`` / ``agentlens.span_id`` (to find it
in the store) and, when known, ``agentlens.usage.cost_usd``. Errors set the span
status to ERROR and ``error.type`` to the exception type. Start and end times
are the recorded ones, so the exported timeline matches the trace.

**Content is not exported by default.** Prompts, outputs and tool arguments may
hold personal data; ``capture_content=True`` adds them (``gen_ai.input.messages``,
``gen_ai.output.messages``, ``gen_ai.tool.call.arguments``,
``gen_ai.tool.call.result``, as JSON strings), the opt-in the conventions
recommend. Use the redacted trace the sink receives, never the live one.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import JsonValue

from agentlens.schema import Span, SpanKind, Trace

try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.context import Context
    from opentelemetry.trace import Status, StatusCode, Tracer, TracerProvider
    from opentelemetry.util.types import AttributeValue
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        'agentlens.integrations.otel needs OpenTelemetry: pip install "agentlens[otel]"'
    ) from exc

__all__ = [
    "INSTRUMENTATION_NAME",
    "OTelExporter",
    "export_trace",
    "fan_out",
    "span_attributes",
    "span_name",
]

logger = logging.getLogger("agentlens.otel")

INSTRUMENTATION_NAME = "agentlens"
"""The instrumentation scope of the exported spans."""

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

# gen_ai.* attribute names (OpenTelemetry semantic conventions for generative AI).
OPERATION = "gen_ai.operation.name"
PROVIDER = "gen_ai.provider.name"
AGENT_NAME = "gen_ai.agent.name"
REQUEST_MODEL = "gen_ai.request.model"
REQUEST_TEMPERATURE = "gen_ai.request.temperature"
REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
RESPONSE_MODEL = "gen_ai.response.model"
INPUT_TOKENS = "gen_ai.usage.input_tokens"
OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
TOOL_NAME = "gen_ai.tool.name"
TOOL_ARGUMENTS = "gen_ai.tool.call.arguments"
TOOL_RESULT = "gen_ai.tool.call.result"
INPUT_MESSAGES = "gen_ai.input.messages"
OUTPUT_MESSAGES = "gen_ai.output.messages"
ERROR_TYPE = "error.type"

_OPERATIONS = {
    SpanKind.AGENT_STEP: "invoke_agent",
    SpanKind.LLM_CALL: "chat",
    SpanKind.TOOL_CALL: "execute_tool",
}


def _ns(value: datetime) -> int:
    """A UTC datetime as integer nanoseconds since the epoch (exact, no float rounding)."""
    return ((value - _EPOCH) // timedelta(microseconds=1)) * 1000


def _json(value: JsonValue) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _error_type(error: str) -> str:
    """``"ValueError: boom"`` -> ``"ValueError"`` (AgentLens records ``Type: message``)."""
    head, sep, _ = error.partition(":")
    return head.strip() if sep and head.strip() and " " not in head.strip() else "_OTHER"


def _str_attr(attributes: Mapping[str, JsonValue], key: str) -> str | None:
    value = attributes.get(key)
    return value if isinstance(value, str) and value else None


def _number_attr(attributes: Mapping[str, JsonValue], key: str) -> float | int | None:
    value = attributes.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return value


def span_name(span: Span) -> str:
    """The OTel span name: ``"{operation} {target}"`` for GenAI spans, else the name."""
    operation = _OPERATIONS.get(span.kind)
    if operation is None:
        return span.name
    target = span.name
    if span.kind is SpanKind.LLM_CALL:
        target = _str_attr(span.attributes, "model") or span.name
    return f"{operation} {target}"


def _llm_messages(span: Span) -> tuple[JsonValue, JsonValue]:
    """(input messages, output message) of an ``llm_call`` span, as recorded by ``wrap_llm``."""
    inputs: JsonValue = span.input
    if isinstance(span.input, dict) and "messages" in span.input:
        inputs = span.input["messages"]
    output: JsonValue = span.output
    if isinstance(span.output, dict):
        message: dict[str, JsonValue] = {"role": "assistant"}
        for key in ("content", "tool_calls"):
            if span.output.get(key):
                message[key] = span.output[key]
        output = [message]
    return inputs, output


def span_attributes(span: Span, *, capture_content: bool = False) -> dict[str, AttributeValue]:
    """The OTel attributes of one AgentLens span (see the module docstring)."""
    attrs: dict[str, AttributeValue] = {
        "agentlens.trace_id": span.trace_id,
        "agentlens.span_id": span.id,
        "agentlens.span.kind": span.kind.value,
    }
    operation = _OPERATIONS.get(span.kind)
    if operation is not None:
        attrs[OPERATION] = operation
    recorded = span.attributes

    if span.kind is SpanKind.AGENT_STEP:
        attrs[AGENT_NAME] = span.name
    elif span.kind is SpanKind.LLM_CALL:
        provider = _str_attr(recorded, "provider")
        if provider:
            attrs[PROVIDER] = provider
        model = _str_attr(recorded, "model")
        if model:
            attrs[REQUEST_MODEL] = model
        temperature = _number_attr(recorded, "temperature")
        if temperature is not None:
            attrs[REQUEST_TEMPERATURE] = float(temperature)
        max_tokens = _number_attr(recorded, "max_tokens")
        if max_tokens is not None:
            attrs[REQUEST_MAX_TOKENS] = int(max_tokens)
        response_model = _str_attr(recorded, "response_model")
        if response_model:
            attrs[RESPONSE_MODEL] = response_model
        if capture_content:
            inputs, output = _llm_messages(span)
            if inputs is not None:
                attrs[INPUT_MESSAGES] = _json(inputs)
            if output is not None:
                attrs[OUTPUT_MESSAGES] = _json(output)
    elif span.kind is SpanKind.TOOL_CALL:
        attrs[TOOL_NAME] = span.name
        if capture_content:
            if span.input is not None:
                attrs[TOOL_ARGUMENTS] = _json(span.input)
            if span.output is not None:
                attrs[TOOL_RESULT] = _json(span.output)

    if capture_content and span.kind not in (SpanKind.LLM_CALL, SpanKind.TOOL_CALL):
        if span.input is not None:
            attrs["agentlens.input"] = _json(span.input)
        if span.output is not None:
            attrs["agentlens.output"] = _json(span.output)

    if span.usage is not None:
        attrs[INPUT_TOKENS] = span.usage.input_tokens
        attrs[OUTPUT_TOKENS] = span.usage.output_tokens
        if span.usage.cost_usd is not None:
            attrs["agentlens.usage.cost_usd"] = span.usage.cost_usd
    if span.error is not None:
        attrs[ERROR_TYPE] = _error_type(span.error)
    return attrs


def _root_attributes(trace: Trace, *, capture_content: bool) -> dict[str, AttributeValue]:
    attrs: dict[str, AttributeValue] = {
        OPERATION: "invoke_agent",
        AGENT_NAME: trace.name,
        "agentlens.trace_id": trace.id,
    }
    for key, value in trace.metadata.items():
        if isinstance(value, str | bool | int | float):
            attrs[f"agentlens.metadata.{key}"] = value
    cost = trace.total_cost
    if cost is not None:
        attrs["agentlens.usage.cost_usd"] = cost
    if trace.total_tokens:
        attrs["agentlens.usage.total_tokens"] = trace.total_tokens
    if capture_content:
        if trace.input is not None:
            attrs["agentlens.input"] = _json(trace.input)
        if trace.output is not None:
            attrs["agentlens.output"] = _json(trace.output)
    if trace.error is not None:
        attrs[ERROR_TYPE] = _error_type(trace.error)
    return attrs


def _walk(trace: Trace) -> Iterator[Span]:
    """Spans parents-first (start-time order among siblings)."""
    stack = list(reversed(trace.roots()))
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(trace.children(current.id)))


def _status(error: str | None) -> Status:
    return Status(StatusCode.ERROR, error) if error is not None else Status(StatusCode.UNSET)


def export_trace(
    trace: Trace,
    *,
    tracer: Tracer | None = None,
    tracer_provider: TracerProvider | None = None,
    capture_content: bool = False,
    context: Context | None = None,
) -> int:
    """Emit ``trace`` as OpenTelemetry spans (one root span plus one per AgentLens span).

    Args:
        trace: A finished trace (unfinished spans end with the trace).
        tracer: The tracer to use; by default one named ``"agentlens"`` from
            ``tracer_provider`` (else the global provider).
        tracer_provider: Where to get the tracer from when ``tracer`` is not given.
        capture_content: Also export prompts, outputs and tool arguments/results.
        context: Parent context of the root span; by default a new trace (the
            agent run is exported after the fact, so the ambient span is not its parent).

    Returns:
        The number of OTel spans emitted.
    """
    if tracer is None:
        tracer = otel_trace.get_tracer(INSTRUMENTATION_NAME, tracer_provider=tracer_provider)
    trace_end = trace.end_time or max(
        (s.end_time or s.start_time for s in trace.spans), default=trace.start_time
    )
    root = tracer.start_span(
        f"invoke_agent {trace.name}",
        context=context if context is not None else Context(),
        kind=otel_trace.SpanKind.INTERNAL,
        attributes=_root_attributes(trace, capture_content=capture_content),
        start_time=_ns(trace.start_time),
    )
    opened: dict[str, otel_trace.Span] = {}
    emitted = 1
    try:
        for span in _walk(trace):
            parent = opened.get(span.parent_id) if span.parent_id is not None else root
            otel_span = tracer.start_span(
                span_name(span),
                context=otel_trace.set_span_in_context(parent or root),
                kind=(
                    otel_trace.SpanKind.CLIENT
                    if span.kind is SpanKind.LLM_CALL
                    else otel_trace.SpanKind.INTERNAL
                ),
                attributes=span_attributes(span, capture_content=capture_content),
                start_time=_ns(span.start_time),
            )
            otel_span.set_status(_status(span.error))
            opened[span.id] = otel_span
            emitted += 1
        for span in trace.spans:
            opened[span.id].end(end_time=_ns(span.end_time or trace_end))
    finally:
        root.set_status(_status(trace.error))
        root.end(end_time=_ns(trace_end))
    return emitted


class OTelExporter:
    """A trace sink that exports every finished trace to OpenTelemetry.

    ``agentlens.run(..., sink=OTelExporter())``; combine it with the store with
    :func:`fan_out`. Export errors are logged, never raised into the agent (FR-T5).
    """

    def __init__(
        self,
        *,
        tracer_provider: TracerProvider | None = None,
        capture_content: bool = False,
    ) -> None:
        self.tracer_provider = tracer_provider
        self.capture_content = capture_content

    def __call__(self, trace: Trace) -> None:
        try:
            export_trace(
                trace,
                tracer_provider=self.tracer_provider,
                capture_content=self.capture_content,
            )
        except Exception:
            logger.exception("AgentLens: exporting trace %s to OpenTelemetry failed", trace.id)

    def __repr__(self) -> str:
        return f"OTelExporter(capture_content={self.capture_content})"


def fan_out(*sinks: Callable[[Trace], Any]) -> Callable[[Trace], None]:
    """One sink calling every sink in order; one failing does not stop the others.

    ``agentlens.run(name, sink=fan_out(store.save_trace, OTelExporter()))``.
    """
    targets: Sequence[Callable[[Trace], Any]] = tuple(sinks)

    def sink(trace: Trace) -> None:
        for target in targets:
            try:
                target(trace)
            except Exception:
                logger.exception("AgentLens: trace sink %r failed", target)

    return sink
