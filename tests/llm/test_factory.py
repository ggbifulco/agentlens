"""Tests for get_client: spec parsing, env configuration and pricing (HTTP mocked with respx)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

import agentlens
from agentlens.llm import (
    MODEL_ENV,
    PRICING_ENV,
    PROVIDERS,
    AnthropicClient,
    LLMConfigError,
    LLMRequest,
    MockLLM,
    ModelPrice,
    OpenAICompatClient,
    PricedClient,
    Pricing,
    get_client,
    parse_spec,
)
from agentlens.schema import SpanKind

ENV_VARS = [
    MODEL_ENV,
    PRICING_ENV,
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "OLLAMA_BASE_URL",
    "VLLM_BASE_URL",
    "VLLM_API_KEY",
    "LMSTUDIO_BASE_URL",
    "OPENAI_COMPAT_BASE_URL",
    "OPENAI_COMPAT_API_KEY",
]

REQUEST = LLMRequest(messages="hello")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def inner(client: object) -> Any:
    assert isinstance(client, PricedClient)
    return client.client


def openai_body(model: str = "gpt-4o-mini-2024-07-18") -> dict[str, Any]:
    return {
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}],
        "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
    }


class TestParseSpec:
    @pytest.mark.parametrize(
        ("spec", "expected"),
        [
            ("openai:gpt-4o-mini", ("openai", "gpt-4o-mini")),
            (" OpenAI : gpt-4o ", ("openai", "gpt-4o")),
            ("ollama:llama3.1:8b", ("ollama", "llama3.1:8b")),
            ("openai_compat:m", ("openai-compat", "m")),
            ("lm-studio:m", ("lmstudio", "m")),
            ("mock", ("mock", "mock")),
            ("mock:", ("mock", "mock")),
            ("mock:judge", ("mock", "judge")),
        ],
    )
    def test_valid(self, spec: str, expected: tuple[str, str]) -> None:
        assert parse_spec(spec) == expected

    @pytest.mark.parametrize(
        ("spec", "message"),
        [
            ("", "empty model spec"),
            ("   ", "empty model spec"),
            ("gpt-4o-mini", "expected '<provider>:<model>'"),
            ("bedrock:claude", "unknown LLM provider 'bedrock'"),
            ("openai:", "has no model"),
            ("anthropic:  ", "has no model"),
        ],
    )
    def test_invalid(self, spec: str, message: str) -> None:
        with pytest.raises(LLMConfigError, match=message) as info:
            parse_spec(spec)
        assert isinstance(info.value, ValueError)

    def test_error_lists_known_providers(self) -> None:
        with pytest.raises(LLMConfigError, match="anthropic, lmstudio, mock, ollama"):
            parse_spec("nope:x")


class TestDefaultSpec:
    def test_missing_spec_and_env(self) -> None:
        with pytest.raises(LLMConfigError, match=f"no model configured.*{MODEL_ENV}"):
            get_client()

    def test_empty_env_counts_as_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(MODEL_ENV, "")
        with pytest.raises(LLMConfigError, match="no model configured"):
            get_client()

    def test_reads_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(MODEL_ENV, "ollama:qwen2.5")
        assert inner(get_client()).model == "qwen2.5"

    def test_explicit_spec_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(MODEL_ENV, "ollama:qwen2.5")
        assert inner(get_client("ollama:llama3.1")).model == "llama3.1"

    def test_top_level_export(self) -> None:
        assert agentlens.get_client is get_client


class TestMock:
    def test_echoes_without_network(self) -> None:
        client = get_client("mock")
        assert isinstance(client, MockLLM)
        response = client.complete(REQUEST)
        assert response.content == "hello"
        assert response.model == "mock"
        assert response.usage is not None
        assert response.usage.cost_usd == 0.0

    def test_echo_without_user_message(self) -> None:
        response = get_client("mock").complete(
            LLMRequest.model_validate({"messages": [{"role": "system", "content": "s"}]})
        )
        assert response.content == ""

    def test_name_and_script_override(self) -> None:
        client = get_client("mock:judge", script=["PASS"], cycle=True)
        assert isinstance(client, MockLLM)
        assert [client.complete(REQUEST).content for _ in range(2)] == ["PASS", "PASS"]
        assert client.complete(REQUEST).model == "judge"


class TestOpenAI:
    def test_defaults(self) -> None:
        client = get_client("openai:gpt-4o-mini")
        assert isinstance(client, PricedClient)
        adapter = inner(client)
        assert isinstance(adapter, OpenAICompatClient)
        assert adapter.base_url == "https://api.openai.com/v1"
        assert client.provider == "openai"
        assert client.model == "gpt-4o-mini"

    @respx.mock
    def test_env_key_base_url_and_cost(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://proxy.test/v1")
        route = respx.post("http://proxy.test/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=openai_body())
        )
        response = get_client("openai:gpt-4o-mini").complete(REQUEST)
        assert route.calls.last.request.headers["authorization"] == "Bearer sk-test"
        assert response.usage is not None
        # Dated id priced as gpt-4o-mini: 0.15 + 0.60 per 1M.
        assert response.usage.cost_usd == pytest.approx(0.75)

    def test_overrides_win(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_BASE_URL", "http://env.test/v1")
        adapter = inner(get_client("openai:gpt-4o", base_url="http://arg.test/v1", timeout=5))
        assert adapter.base_url == "http://arg.test/v1"

    @respx.mock
    def test_unknown_model_cost_none(self) -> None:
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=openai_body("my-finetune"))
        )
        response = get_client("openai:my-finetune", api_key="k").complete(REQUEST)
        assert response.usage is not None
        assert response.usage.cost_usd is None


class TestLocalProviders:
    @pytest.mark.parametrize(
        ("provider", "env", "default"),
        [
            ("ollama", "OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            ("vllm", "VLLM_BASE_URL", "http://localhost:8000/v1"),
            ("lmstudio", "LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
        ],
    )
    def test_base_url_from_env_or_default(
        self, monkeypatch: pytest.MonkeyPatch, provider: str, env: str, default: str
    ) -> None:
        client = get_client(f"{provider}:m")
        assert isinstance(client, PricedClient)
        assert inner(client).base_url == default
        assert client.provider == provider
        monkeypatch.setenv(env, "http://box.test:9/v1/")
        assert inner(get_client(f"{provider}:m")).base_url == "http://box.test:9/v1"

    @respx.mock
    def test_ollama_never_gets_openai_key_and_is_free(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama.test/v1")
        route = respx.post("http://ollama.test/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=openai_body("llama3.1:8b"))
        )
        response = get_client("ollama:llama3.1:8b").complete(REQUEST)
        assert "authorization" not in route.calls.last.request.headers
        assert response.usage is not None
        assert response.usage.cost_usd == 0.0

    @respx.mock
    def test_vllm_optional_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VLLM_API_KEY", "vllm-key")
        route = respx.post("http://localhost:8000/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=openai_body("m"))
        )
        get_client("vllm:m").complete(REQUEST)
        assert route.calls.last.request.headers["authorization"] == "Bearer vllm-key"


class TestOpenAICompat:
    def test_requires_base_url(self) -> None:
        with pytest.raises(LLMConfigError, match="OPENAI_COMPAT_BASE_URL"):
            get_client("openai-compat:m")

    @respx.mock
    def test_env_base_url_and_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://together.test/v1")
        monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "tg-key")
        route = respx.post("http://together.test/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=openai_body("meta-llama/x"))
        )
        client = get_client("openai-compat:meta-llama/x")
        response = client.complete(REQUEST)
        assert route.calls.last.request.headers["authorization"] == "Bearer tg-key"
        assert client.provider == "openai_compat"
        assert response.usage is not None
        assert response.usage.cost_usd is None  # not a free provider, not priced

    def test_base_url_override(self) -> None:
        adapter = inner(get_client("openai_compat:m", base_url="http://x.test/v1"))
        assert adapter.base_url == "http://x.test/v1"


class TestAnthropic:
    def test_defaults(self) -> None:
        adapter = inner(get_client("anthropic:claude-3-5-haiku-latest", max_tokens=99))
        assert isinstance(adapter, AnthropicClient)
        assert adapter.base_url == "https://api.anthropic.com"
        assert adapter.max_tokens == 99

    @respx.mock
    def test_env_and_cost(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ak-test")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://gw.test")
        route = respx.post("http://gw.test/v1/messages").mock(
            return_value=httpx.Response(
                200,
                json={
                    "model": "claude-3-5-haiku-20241022",
                    "content": [{"type": "text", "text": "hi"}],
                    "usage": {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
                },
            )
        )
        response = get_client("anthropic:claude-3-5-haiku-latest").complete(REQUEST)
        assert route.calls.last.request.headers["x-api-key"] == "ak-test"
        assert response.usage is not None
        assert response.usage.cost_usd == pytest.approx(0.8 + 4.0)


class TestPricingOption:
    def test_pricing_instance(self) -> None:
        table = Pricing({"m": ModelPrice(input=1, output=1)})
        client = get_client("openai:m", pricing=table)
        assert isinstance(client, PricedClient)
        assert client.pricing is table

    def test_pricing_path_merged_over_packaged(self, tmp_path: Path) -> None:
        path = tmp_path / "p.yaml"
        path.write_text("models:\n  my-model: {input: 1, output: 2}\n")
        client = get_client("openai:my-model", pricing=str(path))
        assert isinstance(client, PricedClient)
        assert client.pricing.lookup("my-model") == ModelPrice(input=1, output=2)
        assert client.pricing.lookup("gpt-4o") is not None

    def test_pricing_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = tmp_path / "p.yaml"
        path.write_text("models:\n  my-model: {input: 1, output: 2}\n")
        monkeypatch.setenv(PRICING_ENV, str(path))
        client = get_client("openai:my-model")
        assert isinstance(client, PricedClient)
        assert client.pricing.lookup("my-model") is not None


def test_every_provider_builds() -> None:
    for name in PROVIDERS:
        client = get_client(f"{name}:m", base_url="http://x.test/v1")
        assert isinstance(client, PricedClient)


@respx.mock
def test_wrap_llm_records_provider_and_cost() -> None:
    respx.post("http://localhost:11434/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=openai_body("llama3.1"))
    )
    llm = agentlens.wrap_llm(get_client("ollama:llama3.1"))
    with agentlens.run("factory") as trace:
        llm.complete(REQUEST)
    (span,) = [s for s in trace.spans if s.kind == SpanKind.LLM_CALL]
    assert span.attributes["provider"] == "ollama"
    assert span.name == "llama3.1"
    assert span.usage is not None
    assert span.usage.cost_usd == 0.0
