"""Recorder failures must never break user code (FR-T5)."""

from __future__ import annotations

import logging
from contextvars import copy_context

import pytest

import agentlens
from agentlens.schema import Trace
from agentlens.tracing import Recorder
from agentlens.tracing._serialize import format_exception, to_json_value


@agentlens.trace
def double(x: int) -> int:
    return x * 2


def _boom(*args: object, **kwargs: object) -> None:
    raise RuntimeError("recorder bug")


def test_start_span_failure_is_logged_not_raised(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(Recorder, "start_span", _boom)
    with caplog.at_level(logging.ERROR, "agentlens.tracing"), agentlens.run("agent") as t:
        assert double(2) == 4
        with agentlens.span("s") as s:
            assert s is None
    assert t.spans == []
    assert "could not start span" in caplog.text


def test_end_span_failure_is_logged_not_raised(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(Recorder, "end_span", _boom)
    with caplog.at_level(logging.ERROR, "agentlens.tracing"), agentlens.run("agent") as t:
        assert double(2) == 4
        assert agentlens.current_span() is None  # context still restored
    assert len(t.spans) == 1
    assert "could not end span" in caplog.text


def test_sink_failure_is_logged_not_raised(caplog: pytest.LogCaptureFixture) -> None:
    def bad_sink(trace: Trace) -> None:
        raise OSError("disk full")

    with caplog.at_level(logging.ERROR, "agentlens.tracing"), agentlens.run("a", sink=bad_sink):
        pass
    assert "trace sink failed" in caplog.text


def test_finish_failure_is_logged_not_raised(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(Recorder, "finish", _boom)
    with caplog.at_level(logging.ERROR, "agentlens.tracing"), agentlens.run("agent"):
        pass
    assert "could not finish run" in caplog.text


def test_recorder_init_failure_yields_detached_trace(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(Recorder, "__init__", _boom)
    with caplog.at_level(logging.ERROR, "agentlens.tracing"), agentlens.run("agent") as t:
        assert double(3) == 6  # not recorded, but works
        assert agentlens.current_trace() is None
    assert t.name == "agent"
    assert "could not start run" in caplog.text


def test_finish_is_idempotent_and_late_spans_warn(caplog: pytest.LogCaptureFixture) -> None:
    sink: list[Trace] = []
    recorder = Recorder("agent", sink=sink.append)
    first = recorder.finish()
    assert recorder.finish() is first
    assert recorder.finished
    assert len(sink) == 1
    with caplog.at_level(logging.WARNING, "agentlens.tracing"):
        recorder.start_span(agentlens.SpanKind.CUSTOM, "late")
    assert "after run" in caplog.text


def test_exiting_in_other_context_only_warns(caplog: pytest.LogCaptureFixture) -> None:
    cm = agentlens.span("x")
    with agentlens.run("agent"):
        copy_context().run(cm.__enter__)  # token belongs to the copied context
        with caplog.at_level(logging.WARNING, "agentlens.tracing"):
            cm.__exit__(None, None, None)
    assert "different context" in caplog.text


def test_to_json_value_fallbacks() -> None:
    class BadRepr:
        def __repr__(self) -> str:
            raise RuntimeError("no repr")

    assert to_json_value(BadRepr()) == "<unserialisable BadRepr>"
    assert to_json_value({1: b"x", "s": {3}}) == {"1": "x", "s": [3]}


def test_format_exception() -> None:
    assert format_exception(ValueError()) == "ValueError"
    assert format_exception(ValueError("x")) == "ValueError: x"
