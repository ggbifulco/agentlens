"""Tests for the pricing table and the PricedClient wrapper."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentlens.llm import (
    PRICING_ENV,
    LLMClient,
    LLMConfigError,
    LLMRequest,
    LLMResponse,
    MockLLM,
    ModelPrice,
    PricedClient,
    Pricing,
)
from agentlens.schema import Usage

TABLE = Pricing(
    {
        "gpt-4o": ModelPrice(input=2.5, output=10),
        "gpt-4o-mini": {"input": 0.15, "output": 0.6},
        "Llama3": {"input": 1, "output": 1},
    },
    aliases={"chatgpt-4o-latest": "gpt-4o", "broken": "missing"},
    free_providers=["Ollama"],
    updated="2026-11-13",
)


def usage_response(model: str | None = None, cost: float | None = None) -> LLMResponse:
    return LLMResponse(
        content="ok",
        usage=Usage(input_tokens=1_000_000, output_tokens=2_000_000, cost_usd=cost),
        model=model,
    )


class TestModelPrice:
    def test_cost_is_per_million_tokens(self) -> None:
        assert ModelPrice(input=2, output=8).cost(500_000, 250_000) == pytest.approx(3.0)

    def test_negative_price_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelPrice(input=-1, output=1)


class TestLookup:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("gpt-4o", "gpt-4o"),
            ("GPT-4o-Mini", "gpt-4o-mini"),  # case-insensitive
            ("gpt-4o-mini-2024-07-18", "gpt-4o-mini"),  # longest prefix wins
            ("gpt-4o-2024-08-06", "gpt-4o"),
            ("llama3:8b", "llama3"),
            ("llama3@q4", "llama3"),
            ("chatgpt-4o-latest", "gpt-4o"),  # alias
        ],
    )
    def test_matches(self, model: str, expected: str) -> None:
        assert TABLE.lookup(model) == TABLE.models[expected]

    @pytest.mark.parametrize("model", ["gpt-4", "gpt-4omni", "llama3.1", "unknown", "broken"])
    def test_unknown(self, model: str) -> None:
        assert TABLE.lookup(model) is None

    def test_cost(self) -> None:
        assert TABLE.cost("gpt-4o-mini", 1_000_000, 1_000_000) == pytest.approx(0.75)

    def test_unknown_model_has_no_cost(self) -> None:
        assert TABLE.cost("mystery", 10, 10) is None
        assert TABLE.cost(None, 10, 10) is None

    def test_free_provider_costs_zero_for_any_model(self) -> None:
        assert TABLE.cost("anything", 10, 10, provider="ollama") == 0.0
        assert TABLE.cost(None, 10, 10, provider="OLLAMA") == 0.0
        assert TABLE.cost("mystery", 10, 10, provider="openai") is None

    def test_repr(self) -> None:
        assert repr(TABLE) == "Pricing(3 models, updated='2026-11-13')"


class TestApply:
    def test_fills_missing_cost_from_response_model(self) -> None:
        out = TABLE.apply(usage_response("gpt-4o-mini-2024-07-18"), model="gpt-4o")
        assert out.usage is not None
        assert out.usage.cost_usd == pytest.approx(0.15 + 1.2)

    def test_falls_back_to_given_model(self) -> None:
        out = TABLE.apply(usage_response(None), model="gpt-4o")
        assert out.usage is not None
        assert out.usage.cost_usd == pytest.approx(22.5)

    def test_keeps_reported_cost(self) -> None:
        response = usage_response("gpt-4o", cost=1.0)
        assert TABLE.apply(response) is response

    def test_no_usage_unchanged(self) -> None:
        response = LLMResponse(content="x", model="gpt-4o")
        assert TABLE.apply(response) is response

    def test_unknown_model_unchanged(self) -> None:
        response = usage_response("mystery")
        assert TABLE.apply(response) is response

    def test_free_provider(self) -> None:
        out = TABLE.apply(usage_response("llama3.1"), provider="ollama")
        assert out.usage is not None
        assert out.usage.cost_usd == 0.0


class TestLoading:
    def test_from_yaml(self) -> None:
        table = Pricing.from_yaml(
            "updated: '2026-01-01'\nfree_providers: [local]\n"
            "models:\n  m: {input: 1, output: 2}\naliases:\n  a: m\n"
        )
        assert table.updated == "2026-01-01"
        assert table.lookup("a") == ModelPrice(input=1, output=2)
        assert table.free_providers == frozenset({"local"})

    def test_empty_yaml_is_empty_table(self) -> None:
        assert Pricing.from_yaml("").models == {}

    @pytest.mark.parametrize(
        ("text", "message"),
        [
            ("models: [", "invalid YAML"),
            ("- a\n- b\n", "mapping at the top level"),
            ("models:\n  m: {input: 1}\n", "invalid pricing table"),
            ("modles: {}\n", "invalid pricing table"),
            ("models:\n  m: {input: -1, output: 1}\n", "invalid pricing table"),
        ],
    )
    def test_invalid_yaml(self, text: str, message: str) -> None:
        with pytest.raises(LLMConfigError, match=message) as info:
            Pricing.from_yaml(text, source="x.yaml")
        assert isinstance(info.value, ValueError)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(LLMConfigError, match="cannot read pricing file"):
            Pricing.from_file(tmp_path / "nope.yaml")

    def test_packaged_table(self) -> None:
        table = Pricing.packaged()
        assert table.updated
        for model in ("gpt-4o-mini", "gpt-4o", "claude-3-5-haiku", "claude-sonnet-4-5"):
            price = table.lookup(model)
            assert price is not None
            assert price.input > 0
            assert price.output > price.input
        assert {"ollama", "vllm", "lmstudio", "mock"} <= table.free_providers
        assert table.lookup("claude-3-5-haiku-latest") == table.lookup("claude-3-5-haiku")
        assert table.lookup("claude-sonnet-4-5-20250929") == table.lookup("claude-sonnet-4-5")
        assert table.lookup("chatgpt-4o-latest") == table.lookup("gpt-4o")

    def test_packaged_returns_independent_copies(self) -> None:
        first = Pricing.packaged()
        first.models.clear()
        assert Pricing.packaged().models

    def test_load_without_override_is_packaged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(PRICING_ENV, raising=False)
        assert Pricing.load().models == Pricing.packaged().models

    def test_load_merges_file_over_packaged(self, tmp_path: Path) -> None:
        path = tmp_path / "prices.yaml"
        path.write_text(
            "updated: mine\nfree_providers: [corp-gateway]\nmodels:\n"
            "  gpt-4o-mini: {input: 9, output: 9}\n  my-model: {input: 1, output: 1}\n"
        )
        table = Pricing.load(path)
        assert table.lookup("gpt-4o-mini") == ModelPrice(input=9, output=9)
        assert table.lookup("my-model") is not None
        assert table.lookup("gpt-4o") == Pricing.packaged().lookup("gpt-4o")
        assert "corp-gateway" in table.free_providers
        assert "ollama" in table.free_providers
        assert table.updated == "mine"

    def test_load_reads_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = tmp_path / "prices.yaml"
        path.write_text("models:\n  my-model: {input: 1, output: 1}\n")
        monkeypatch.setenv(PRICING_ENV, str(path))
        table = Pricing.load()
        assert table.lookup("my-model") is not None
        assert table.updated == Pricing.packaged().updated

    def test_load_bad_env_file_fails_loudly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(PRICING_ENV, str(tmp_path / "missing.yaml"))
        with pytest.raises(LLMConfigError):
            Pricing.load()


class _Unpriced:
    """A client reporting usage without cost, and a ``provider`` attribute."""

    def __init__(self, model: str | None = "gpt-4o", provider: object = "openai") -> None:
        self.model = model
        self.provider = provider
        self.closed = False

    def _response(self) -> LLMResponse:
        return LLMResponse(content="hi", usage=Usage(input_tokens=1_000_000, output_tokens=0))

    def complete(self, request: LLMRequest) -> LLMResponse:
        return self._response()

    async def acomplete(self, request: LLMRequest) -> LLMResponse:
        return self._response()

    def close(self) -> None:
        self.closed = True


REQUEST = LLMRequest(messages="hello")


class TestPricedClient:
    def test_is_llm_client_and_fills_cost_from_client_model(self) -> None:
        client = PricedClient(_Unpriced(), TABLE)
        assert isinstance(client, LLMClient)
        usage = client.complete(REQUEST).usage
        assert usage is not None
        assert usage.cost_usd == pytest.approx(2.5)

    def test_request_model_wins_over_client_model(self) -> None:
        client = PricedClient(_Unpriced(), TABLE)
        usage = client.complete(LLMRequest(messages="hi", model="gpt-4o-mini")).usage
        assert usage is not None
        assert usage.cost_usd == pytest.approx(0.15)

    def test_async(self) -> None:
        usage = asyncio.run(PricedClient(_Unpriced(), TABLE).acomplete(REQUEST)).usage
        assert usage is not None
        assert usage.cost_usd == pytest.approx(2.5)

    def test_unknown_model_keeps_none(self) -> None:
        usage = PricedClient(_Unpriced(model=None), TABLE).complete(REQUEST).usage
        assert usage is not None
        assert usage.cost_usd is None

    def test_provider_from_client_or_argument(self) -> None:
        assert PricedClient(_Unpriced(provider="ollama"), TABLE).provider == "ollama"
        assert PricedClient(_Unpriced(provider=42), TABLE).provider is None
        free = PricedClient(_Unpriced(model="mystery"), TABLE, provider="ollama")
        usage = free.complete(REQUEST).usage
        assert usage is not None
        assert usage.cost_usd == 0.0

    def test_delegates_attributes(self) -> None:
        inner = _Unpriced()
        client = PricedClient(inner, TABLE)
        assert client.model == "gpt-4o"
        client.close()
        assert inner.closed
        assert repr(client).startswith("PricedClient(")
        with pytest.raises(AttributeError):
            _ = client.nope

    def test_client_attribute_missing_does_not_recurse(self) -> None:
        bare = PricedClient.__new__(PricedClient)
        with pytest.raises(AttributeError):
            _ = bare.client

    def test_default_table_is_loaded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(PRICING_ENV, raising=False)
        client = PricedClient(MockLLM(["x"]))
        assert client.pricing.models == Pricing.packaged().models
        # MockLLM already reports cost 0: kept as is.
        usage = client.complete(REQUEST).usage
        assert usage is not None
        assert usage.cost_usd == 0.0
