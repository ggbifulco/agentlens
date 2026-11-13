"""Unit tests for :class:`agentlens.replay.Cassette` and :class:`CassetteLLM`."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

import agentlens
from agentlens.llm import LLMClient, LLMRequest, LLMResponse, Message, MockLLM
from agentlens.replay import (
    CASSETTE_FORMAT,
    Cassette,
    CassetteEntry,
    CassetteError,
    CassetteFormatError,
    CassetteLLM,
    CassetteMissError,
    tool_key,
)
from agentlens.schema import Usage
from tests.replay.conftest import FailingLLM


def _req(text: str) -> LLMRequest:
    return LLMRequest(messages=[Message.user(text)])


def _recorded(*answers: str) -> Cassette:
    cassette = Cassette()
    llm = cassette.wrap_llm(MockLLM(list(answers)))
    for n in range(len(answers)):
        llm.complete(_req(f"q{n}"))
    return cassette


class CountingLLM:
    """A real client that answers ``live:<question>`` and counts calls."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        return LLMResponse(content=f"live:{request.last_user_message}")

    async def acomplete(self, request: LLMRequest) -> LLMResponse:
        return self.complete(request)


class FlakyLLM:
    """Fails on the first call, then answers."""

    def __init__(self) -> None:
        self.failed = False

    def complete(self, request: LLMRequest) -> LLMResponse:
        if not self.failed:
            self.failed = True
            raise ConnectionError("down")
        return LLMResponse(content="ok")

    async def acomplete(self, request: LLMRequest) -> LLMResponse:
        return self.complete(request)


# -- recording ---------------------------------------------------------------------


def test_record_stores_index_key_request_and_full_response() -> None:
    cassette = Cassette()
    llm = cassette.wrap_llm(MockLLM(["a", "b"]))
    assert isinstance(llm, LLMClient)
    first = llm.complete(_req("q0"))
    llm.complete(_req("q1"))
    entries = cassette.entries
    assert [(e.kind, e.index) for e in entries] == [("llm", 0), ("llm", 1)]
    assert entries[0].key == _req("q0").fingerprint()
    assert entries[0].request == {"messages": [{"role": "user", "content": "q0"}]}
    assert LLMResponse.model_validate(entries[0].response) == first
    assert entries[0].response["usage"] is not None  # type: ignore[index]
    assert cassette.metadata == {"model": "mock"}
    assert len(cassette) == 2
    assert repr(cassette) == "Cassette(mode='record', entries=2)"


def test_record_needs_a_real_client() -> None:
    llm = Cassette().wrap_llm()
    with pytest.raises(CassetteError, match="needs a real client"):
        llm.complete(_req("q"))


def test_failed_real_call_is_not_recorded_and_frees_the_index() -> None:
    cassette = Cassette()
    llm = cassette.wrap_llm(FlakyLLM())
    with pytest.raises(ConnectionError):
        llm.complete(_req("q"))
    assert llm.complete(_req("q")).content == "ok"
    assert [e.index for e in cassette.entries] == [0]


def test_record_over_existing_entries_replaces_them_in_place() -> None:
    cassette = _recorded("a", "b")
    cassette.rewind()
    cassette.wrap_llm(MockLLM(["c"])).complete(_req("other"))
    assert [e.index for e in cassette.entries] == [0, 1]
    assert cassette.entries[0].response["content"] == "c"  # type: ignore[index]


# -- playback modes ------------------------------------------------------------------


def test_strict_hit_returns_recorded_response() -> None:
    cassette = _recorded("a", "b")
    cassette.mode = "strict"
    cassette.rewind()
    llm = cassette.wrap_llm(FailingLLM())
    assert [llm.complete(_req(f"q{n}")).content for n in range(2)] == ["a", "b"]
    assert cassette.unplayed == []


def test_strict_key_mismatch_raises_miss_error_with_details() -> None:
    cassette = _recorded("a")
    cassette.mode = "strict"
    cassette.rewind()
    llm = cassette.wrap_llm()
    with pytest.raises(CassetteMissError) as info:
        llm.complete(_req("changed"))
    err = info.value
    assert (err.kind, err.index, err.mode, err.actual_name) == ("llm", 0, "strict", None)
    assert err.expected is cassette.entries[0]
    assert err.expected_request == {"messages": [{"role": "user", "content": "q0"}]}
    assert err.actual_request == {"messages": [{"role": "user", "content": "changed"}]}
    assert err.actual_key == _req("changed").fingerprint()
    assert "strict cassette miss: llm call #0: request key" in str(err)
    # the index is not consumed: the recorded call can still be played
    assert llm.complete(_req("q0")).content == "a"


def test_strict_beyond_the_end_raises_miss_error() -> None:
    cassette = _recorded("a")
    cassette.mode = "strict"
    llm = cassette.wrap_llm()
    with pytest.raises(CassetteMissError, match="no recorded llm call #1") as info:
        llm.complete(_req("q1"))
    assert info.value.expected is None
    assert info.value.expected_request is None


def test_lenient_ignores_the_request_but_not_the_end() -> None:
    cassette = _recorded("a")
    cassette.mode = "lenient"
    cassette.rewind()
    llm = cassette.wrap_llm()
    assert llm.complete(_req("something else")).content == "a"
    with pytest.raises(CassetteMissError, match="lenient cassette miss"):
        llm.complete(_req("q1"))


def test_fallthrough_calls_real_client_on_miss_without_recording() -> None:
    cassette = _recorded("a", "b")
    cassette.mode = "fallthrough"
    cassette.rewind()
    real = CountingLLM()
    llm = cassette.wrap_llm(real)
    assert llm.complete(_req("q0")).content == "a"
    assert llm.complete(_req("new")).content == "live:new"
    assert llm.complete(_req("extra")).content == "live:extra"
    assert real.calls == 2
    assert [e.response["content"] for e in cassette.entries] == ["a", "b"]  # type: ignore[index]


def test_fallthrough_record_new_replaces_and_appends() -> None:
    cassette = _recorded("a", "b")
    cassette.mode = "fallthrough"
    cassette.record_new = True
    cassette.rewind()
    llm = cassette.wrap_llm(CountingLLM())
    llm.complete(_req("q0"))
    llm.complete(_req("new"))
    llm.complete(_req("extra"))
    contents = [e.response["content"] for e in cassette.entries]  # type: ignore[index]
    assert contents == ["a", "live:new", "live:extra"]
    assert cassette.entries[1].key == _req("new").fingerprint()


def test_fallthrough_miss_without_real_client() -> None:
    cassette = _recorded("a")
    cassette.mode = "fallthrough"
    llm = cassette.wrap_llm()
    with pytest.raises(CassetteError, match="fallthrough miss on llm call #1"):
        llm.complete(_req("q1"))
    # the index was given back
    cassette.mode = "strict"
    with pytest.raises(CassetteMissError, match="#1"):
        llm.complete(_req("q1"))


def test_fallthrough_real_error_gives_index_back() -> None:
    cassette = Cassette(mode="fallthrough", record_new=True)
    llm = cassette.wrap_llm(FlakyLLM())
    with pytest.raises(ConnectionError):
        llm.complete(_req("q"))
    llm.complete(_req("q"))
    assert [e.index for e in cassette.entries] == [0]


def test_async_modes() -> None:
    cassette = _recorded("a")
    cassette.mode = "strict"
    cassette.rewind()
    llm = cassette.wrap_llm(FailingLLM())

    async def main() -> None:
        assert (await llm.acomplete(_req("q0"))).content == "a"
        with pytest.raises(CassetteMissError):
            await llm.acomplete(_req("q1"))
        cassette.mode = "fallthrough"
        cassette.record_new = True
        flaky = cassette.wrap_llm(FlakyLLM())
        with pytest.raises(ConnectionError):
            await flaky.acomplete(_req("q1"))
        assert (await flaky.acomplete(_req("q1"))).content == "ok"

    asyncio.run(main())
    assert len(cassette) == 2


def test_rewind_and_unplayed() -> None:
    cassette = _recorded("a", "b")
    cassette.mode = "lenient"
    assert cassette.unplayed == []
    cassette.rewind()
    assert len(cassette.unplayed) == 2
    cassette.wrap_llm().complete(_req("x"))
    assert [e.index for e in cassette.unplayed] == [1]


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown cassette mode"):
        Cassette(mode="replay")  # type: ignore[arg-type]
    cassette = Cassette()
    with pytest.raises(ValueError, match="unknown cassette mode"):
        cassette.mode = "nope"  # type: ignore[assignment]


def test_duplicate_entries_are_rejected() -> None:
    entry = _recorded("a").entries[0]
    with pytest.raises(CassetteFormatError, match="<cassette>: invalid cassette: duplicate"):
        Cassette([entry, entry])


# -- CassetteLLM attributes ------------------------------------------------------------


def test_model_and_provider_resolution() -> None:
    class Provided:
        model = "gpt-x"
        provider = "acme"
        extra = 42

        def complete(self, request: LLMRequest) -> LLMResponse:
            return LLMResponse(content="x")

        async def acomplete(self, request: LLMRequest) -> LLMResponse:
            return self.complete(request)

    cassette = Cassette()
    llm = cassette.wrap_llm(Provided())
    assert (llm.model, llm.provider, llm.extra) == ("gpt-x", "acme", 42)
    assert cassette.metadata == {"model": "gpt-x", "provider": "acme"}
    assert "Provided" in repr(llm)
    assert "mode='record'" in repr(llm)

    bare = cassette.wrap_llm()  # playback default from metadata
    assert (bare.model, bare.provider) == ("gpt-x", "acme")
    with pytest.raises(AttributeError):
        _ = bare.extra
    explicit = cassette.wrap_llm(Provided(), model="m", provider="p")
    assert (explicit.model, explicit.provider) == ("m", "p")
    assert Cassette().wrap_llm().model is None


def test_traced_cassette_llm_records_provider_from_metadata() -> None:
    cassette = Cassette(metadata={"model": "gpt-x", "provider": "acme"}, mode="lenient")
    cassette._put(_recorded("a").entries[0])
    llm = agentlens.wrap_llm(cassette.wrap_llm())
    with agentlens.run("t") as trace:
        llm.complete(_req("q"))
    span = trace.spans[0]
    assert span.name == "gpt-x"
    assert span.attributes["provider"] == "acme"
    assert span.attributes["client"] == f"{CassetteLLM.__module__}.CassetteLLM"


def test_getattr_before_init_does_not_recurse() -> None:
    llm = CassetteLLM.__new__(CassetteLLM)
    with pytest.raises(AttributeError):
        _ = llm.anything


# -- tools ---------------------------------------------------------------------------


def test_tool_record_and_strict_playback() -> None:
    calls: list[tuple[str, int]] = []

    def search(query: str, limit: int = 3) -> list[str]:
        calls.append((query, limit))
        return [f"{query}-{n}" for n in range(limit)]

    cassette = Cassette()
    tool = cassette.wrap_tool(search)
    assert tool.__name__ == "search"
    assert tool("x", limit=2) == ["x-0", "x-1"]
    entry = cassette.entries[0]
    assert (entry.kind, entry.index, entry.name) == ("tool", 0, "search")
    assert entry.request == {"query": "x", "limit": 2}
    assert entry.key == tool_key("search", {"query": "x", "limit": 2})

    cassette.mode = "strict"
    cassette.rewind()
    assert tool(query="x", limit=2) == ["x-0", "x-1"]  # same bound arguments
    assert len(calls) == 1
    with pytest.raises(CassetteMissError, match=r"tool call #1 \('search'\)"):
        tool("y")


def test_tool_name_mismatch_is_a_strict_miss_but_lenient_hit() -> None:
    cassette = Cassette()
    cassette.wrap_tool(lambda: "a", name="first")()
    cassette.mode = "strict"
    cassette.rewind()
    other = cassette.wrap_tool(lambda: "b", name="second")
    with pytest.raises(CassetteMissError, match="recorded tool is 'first'") as info:
        other()
    assert info.value.actual_name == "second"
    cassette.mode = "lenient"
    assert other() == "a"


def test_tool_decorator_forms_and_json_results() -> None:
    cassette = Cassette()

    @cassette.wrap_tool
    def pair(a: int) -> tuple[int, int]:
        return (a, a)

    @cassette.wrap_tool(name="renamed")
    def anything() -> str:
        return "x"

    assert pair(1) == (1, 1)
    assert anything.__name__ == "renamed"
    anything()
    cassette.mode = "strict"
    cassette.rewind()
    assert pair(1) == [1, 1]  # played back as JSON
    assert anything() == "x"


def test_tool_errors_are_not_recorded() -> None:
    cassette = Cassette()

    @cassette.wrap_tool
    def boom() -> None:
        raise KeyError("x")

    with pytest.raises(KeyError):
        boom()
    assert len(cassette) == 0


def test_async_tool_record_playback_and_errors() -> None:
    cassette = Cassette()

    @cassette.wrap_tool
    async def fetch(url: str) -> str:
        if url == "bad":
            raise ValueError(url)
        return f"<{url}>"

    async def main() -> None:
        with pytest.raises(ValueError, match="bad"):
            await fetch("bad")
        assert await fetch("a") == "<a>"
        cassette.mode = "strict"
        cassette.rewind()
        assert await fetch("a") == "<a>"

    asyncio.run(main())
    assert [e.index for e in cassette.entries] == [0]


def test_llm_and_tool_indexes_are_independent() -> None:
    cassette = Cassette()
    llm = cassette.wrap_llm(MockLLM(["a", "b"]))
    tool = cassette.wrap_tool(lambda x: x * 2, name="double")
    llm.complete(_req("q0"))
    tool(1)
    llm.complete(_req("q1"))
    assert [(e.kind, e.index) for e in cassette.entries] == [("llm", 0), ("tool", 0), ("llm", 1)]
    assert [e.index for e in cassette.entries_of("llm")] == [0, 1]


def test_calls_inside_a_real_tool_are_passed_through() -> None:
    cassette = Cassette()
    real = CountingLLM()
    llm = cassette.wrap_llm(real)

    @cassette.wrap_tool
    def summarise(text: str) -> str | None:
        return llm.complete(_req(text)).content

    @cassette.wrap_tool
    async def aupper(text: str) -> str:
        return text.upper()

    @cassette.wrap_tool
    async def asummarise(text: str) -> str | None:
        inner = await llm.acomplete(_req(text))
        return f"{inner.content}|{summarise(text)}|{await aupper(text)}"

    assert summarise("doc") == "live:doc"
    assert asyncio.run(asummarise("doc")) == "live:doc|live:doc|DOC"
    assert [(e.kind, e.name) for e in cassette.entries] == [
        ("tool", "summarise"),
        ("tool", "asummarise"),
    ]
    assert real.calls == 3

    cassette.mode = "strict"
    cassette.rewind()
    assert summarise("doc") == "live:doc"
    assert real.calls == 3


def test_call_inside_a_real_tool_needs_a_real_client() -> None:
    cassette = Cassette()
    llm = cassette.wrap_llm()

    @cassette.wrap_tool
    def ask() -> str | None:
        return llm.complete(_req("q")).content

    with pytest.raises(CassetteError, match="needs a real client"):
        ask()
    assert len(cassette) == 0


# -- persistence -----------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path: Path) -> None:
    cassette = _recorded("a")
    cassette.wrap_tool(lambda: {"k": 1}, name="t")()
    cassette.metadata["note"] = "hi"
    path = tmp_path / "nested" / "c.jsonl"
    assert cassette.save(path) == 2
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0]) == {
        "format": CASSETTE_FORMAT,
        "version": 1,
        "metadata": {"model": "mock", "note": "hi"},
    }
    assert json.loads(lines[2])["kind"] == "tool"

    loaded = Cassette.load(path)
    assert loaded.mode == "strict"
    assert loaded.metadata == cassette.metadata
    assert loaded.entries == cassette.entries
    fall = Cassette.load(path, mode="fallthrough", record_new=True)
    assert (fall.mode, fall.record_new) == ("fallthrough", True)
    assert list(tmp_path.joinpath("nested").iterdir()) == [path]


def test_load_without_header_and_with_blank_lines(tmp_path: Path) -> None:
    entry = _recorded("a").entries[0]
    path = tmp_path / "c.jsonl"
    path.write_text(f"\n{entry.model_dump_json()}\n\n", encoding="utf-8")
    loaded = Cassette.load(path)
    assert loaded.entries == [entry]
    assert loaded.metadata == {}


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        (["not json"], r"c\.jsonl:1: invalid cassette"),
        (['{"format": "agentlens-cassette", "version": 2}'], r":1: invalid cassette: version"),
        (["{ENTRY}", '{"format": "agentlens-cassette"}'], r":2: invalid cassette"),
        (["{ENTRY}", "{ENTRY}"], r":2: invalid cassette: duplicate llm call #0"),
        (['{"index": 0, "kind": "tool", "key": "' + "0" * 64 + '"}'], r":1: .*need a name"),
        (['{"index": 0, "kind": "llm", "key": "abc"}'], r":1: invalid cassette: key"),
    ],
)
def test_load_errors(tmp_path: Path, lines: list[str], message: str) -> None:
    entry = _recorded("a").entries[0].model_dump_json()
    path = tmp_path / "c.jsonl"
    path.write_text("\n".join(line.replace("{ENTRY}", entry) for line in lines), encoding="utf-8")
    with pytest.raises(CassetteFormatError, match=message) as info:
        Cassette.load(path)
    assert info.value.path == str(path)
    assert isinstance(info.value, ValueError)


def test_save_failure_keeps_existing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text("old", encoding="utf-8")
    cassette = _recorded("a")

    def broken(self: Any, **kwargs: Any) -> str:
        raise RuntimeError("boom")

    monkeypatch.setattr(CassetteEntry, "model_dump_json", broken)
    with pytest.raises(RuntimeError):
        cassette.save(path)
    assert path.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.iterdir()) == [path]


def test_entry_validates_llm_payloads() -> None:
    with pytest.raises(ValueError, match="messages"):
        CassetteEntry(index=0, kind="llm", key="0" * 64, request={"messages": []}, response={})
    entry = CassetteEntry(
        index=0,
        kind="llm",
        key="0" * 64,
        request={"messages": "hi"},
        response={"content": "x", "usage": Usage(input_tokens=1).model_dump()},
    )
    assert entry.name is None
