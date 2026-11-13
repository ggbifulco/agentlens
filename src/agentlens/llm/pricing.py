"""Model pricing: estimate ``usage.cost_usd`` from token counts (FR-T3).

A :class:`Pricing` table maps model ids to USD prices per 1M input/output tokens.
The packaged table (``agentlens/llm/pricing.yaml``) lists common OpenAI and Anthropic
models at approximate list prices; local providers (Ollama, vLLM, LM Studio, mock)
cost 0. Your own YAML file of the same shape is merged over it::

    from agentlens.llm import Pricing

    pricing = Pricing.load()  # packaged table + $AGENTLENS_PRICING, if set
    pricing.cost("gpt-4o-mini-2024-07-18", input_tokens=1_000, output_tokens=500)
    # -> 0.00045  (matched "gpt-4o-mini")

Unknown models get ``None``, never 0: an unknown cost is not a free call.
:class:`PricedClient` applies a table to every response of an
:class:`~agentlens.llm.LLMClient` that did not report a cost itself.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, NonNegativeFloat, ValidationError

from agentlens.llm.base import LLMClient
from agentlens.llm.errors import LLMConfigError
from agentlens.llm.types import LLMRequest, LLMResponse
from agentlens.schema._base import SchemaModel

__all__ = ["PRICING_ENV", "ModelPrice", "PricedClient", "Pricing"]

PRICING_ENV = "AGENTLENS_PRICING"
"""Environment variable naming a YAML pricing file merged over the packaged table."""

_PREFIX_SEPARATORS = ("-", ":", "@")
_TOKENS_PER_UNIT = 1_000_000


class ModelPrice(SchemaModel):
    """Price of one model, in USD per 1M tokens."""

    input: NonNegativeFloat
    """USD per 1M input (prompt) tokens."""
    output: NonNegativeFloat
    """USD per 1M output (completion) tokens."""

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        """USD cost of a call with these token counts."""
        return (input_tokens * self.input + output_tokens * self.output) / _TOKENS_PER_UNIT


class _PricingFile(SchemaModel):
    updated: str | None = None
    free_providers: list[str] = Field(default_factory=list)
    models: dict[str, ModelPrice] = Field(default_factory=dict)
    aliases: dict[str, str] = Field(default_factory=dict)


def _norm(name: str) -> str:
    return name.strip().lower()


class Pricing:
    """A table of model prices with alias and prefix matching.

    Args:
        models: Model id -> :class:`ModelPrice` (or a ``{"input": .., "output": ..}`` mapping).
        aliases: Model id -> the priced id it costs the same as.
        free_providers: Providers whose models all cost 0 (local servers).
        updated: Free-form date of the prices, for display.

    Lookup (case-insensitive) of a model id: exact key, then alias, then the longest key
    that is a prefix of the id followed by ``-``, ``:`` or ``@`` — so dated or tagged ids
    (``gpt-4o-mini-2024-07-18``, ``claude-3-5-haiku-latest``) use their base model's price.
    """

    def __init__(
        self,
        models: Mapping[str, ModelPrice | Mapping[str, float]] | None = None,
        *,
        aliases: Mapping[str, str] | None = None,
        free_providers: Iterable[str] = (),
        updated: str | None = None,
    ) -> None:
        self.models: dict[str, ModelPrice] = {
            _norm(name): price
            if isinstance(price, ModelPrice)
            else ModelPrice.model_validate(price)
            for name, price in (models or {}).items()
        }
        self.aliases: dict[str, str] = {
            _norm(alias): _norm(target) for alias, target in (aliases or {}).items()
        }
        self.free_providers: frozenset[str] = frozenset(_norm(p) for p in free_providers)
        self.updated = updated

    def __repr__(self) -> str:
        return f"{type(self).__name__}({len(self.models)} models, updated={self.updated!r})"

    # -- construction ------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Pricing:
        """Build a table from parsed YAML/JSON (keys ``models``, ``aliases``,
        ``free_providers``, ``updated``).

        Raises:
            LLMConfigError: The data does not have that shape.
        """
        try:
            parsed = _PricingFile.model_validate(data)
        except ValidationError as exc:
            raise LLMConfigError(f"invalid pricing table: {exc}") from exc
        return cls(
            parsed.models,
            aliases=parsed.aliases,
            free_providers=parsed.free_providers,
            updated=parsed.updated,
        )

    @classmethod
    def from_yaml(cls, text: str, *, source: str = "<string>") -> Pricing:
        """Parse a YAML pricing table (an empty document is an empty table).

        Raises:
            LLMConfigError: Invalid YAML or an unexpected shape.
        """
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise LLMConfigError(f"invalid YAML in pricing file {source}: {exc}") from exc
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise LLMConfigError(f"pricing file {source} must be a mapping at the top level")
        try:
            return cls.from_dict(data)
        except LLMConfigError as exc:
            raise LLMConfigError(f"{source}: {exc}") from exc

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> Pricing:
        """Read a YAML pricing table from ``path``.

        Raises:
            LLMConfigError: The file is missing, unreadable or invalid.
        """
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            raise LLMConfigError(f"cannot read pricing file {path}: {exc}") from exc
        return cls.from_yaml(text, source=str(path))

    @classmethod
    def packaged(cls) -> Pricing:
        """The table shipped with AgentLens (approximate prices, see its ``updated`` date)."""
        return _packaged().merged(Pricing())  # a copy: callers may mutate their table

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> Pricing:
        """The packaged table with ``path`` (else ``$AGENTLENS_PRICING``, if set) merged over it."""
        table = cls.packaged()
        override = path if path is not None else os.environ.get(PRICING_ENV) or None
        if override is not None:
            table = table.merged(cls.from_file(override))
        return table

    def merged(self, other: Pricing) -> Pricing:
        """A new table: this one with ``other``'s models, aliases and free providers on top."""
        return Pricing(
            {**self.models, **other.models},
            aliases={**self.aliases, **other.aliases},
            free_providers=self.free_providers | other.free_providers,
            updated=other.updated or self.updated,
        )

    # -- lookup --------------------------------------------------------------------

    def lookup(self, model: str) -> ModelPrice | None:
        """The price of ``model`` (exact, alias, then prefix match), or ``None`` if unknown."""
        name = _norm(model)
        if name in self.models:
            return self.models[name]
        if name in self.aliases:
            return self.models.get(self.aliases[name])
        best: str | None = None
        for key in self.models:
            if (
                len(name) > len(key)
                and name.startswith(key)
                and name[len(key)] in _PREFIX_SEPARATORS
                and (best is None or len(key) > len(best))
            ):
                best = key
        return self.models[best] if best is not None else None

    def cost(
        self,
        model: str | None,
        input_tokens: int,
        output_tokens: int,
        *,
        provider: str | None = None,
    ) -> float | None:
        """USD cost of a call, ``0.0`` for a free provider, ``None`` for an unknown model."""
        if provider is not None and _norm(provider) in self.free_providers:
            return 0.0
        if not model:
            return None
        price = self.lookup(model)
        return price.cost(input_tokens, output_tokens) if price is not None else None

    def apply(
        self, response: LLMResponse, *, model: str | None = None, provider: str | None = None
    ) -> LLMResponse:
        """``response`` with ``usage.cost_usd`` filled in, when it can be.

        The response's own ``model`` (the one that actually answered) wins over
        ``model``. Returned unchanged when there is no usage, the cost is already set,
        or the model is unknown.
        """
        usage = response.usage
        if usage is None or usage.cost_usd is not None:
            return response
        cost = self.cost(
            response.model or model, usage.input_tokens, usage.output_tokens, provider=provider
        )
        if cost is None:
            return response
        return response.model_copy(update={"usage": usage.model_copy(update={"cost_usd": cost})})


@lru_cache(maxsize=1)
def _packaged() -> Pricing:
    text = resources.files("agentlens.llm").joinpath("pricing.yaml").read_text(encoding="utf-8")
    return Pricing.from_yaml(text, source="agentlens/llm/pricing.yaml")


class PricedClient:
    """Wrap an :class:`~agentlens.llm.LLMClient` so responses carry ``usage.cost_usd``.

    Costs the provider already reported are kept; otherwise the cost is computed from
    ``pricing`` using the response's model (else the request's, else the client's
    ``model``). Unknown models keep ``cost_usd=None``. Attributes not defined here
    (``model``, ``provider``, ``close``...) are looked up on the wrapped client, so
    :func:`agentlens.wrap_llm` still sees the provider and default model.

    Args:
        client: The client to wrap.
        pricing: The table to use (default: :meth:`Pricing.load`).
        provider: Provider name for the free-provider check; defaults to the client's
            ``provider`` attribute when it is a string.
    """

    def __init__(
        self,
        client: LLMClient,
        pricing: Pricing | None = None,
        *,
        provider: str | None = None,
    ) -> None:
        self.client = client
        """The wrapped client."""
        self.pricing = pricing if pricing is not None else Pricing.load()
        """The pricing table in use."""
        if provider is None:
            client_provider = getattr(client, "provider", None)
            provider = client_provider if isinstance(client_provider, str) else None
        self.provider = provider
        """Provider used for the free-provider check (``None`` if unknown)."""

    def __getattr__(self, name: str) -> Any:
        if name == "client":  # not yet set (e.g. during unpickling): avoid recursion
            raise AttributeError(name)
        return getattr(self.client, name)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.client!r})"

    def _price(self, request: LLMRequest, response: LLMResponse) -> LLMResponse:
        default_model = getattr(self.client, "model", None)
        model = request.model or (default_model if isinstance(default_model, str) else None)
        return self.pricing.apply(response, model=model, provider=self.provider)

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Call the wrapped client and fill in the response's cost."""
        return self._price(request, self.client.complete(request))

    async def acomplete(self, request: LLMRequest) -> LLMResponse:
        """Async variant of :meth:`complete`."""
        return self._price(request, await self.client.acomplete(request))
