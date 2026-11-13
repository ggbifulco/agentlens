"""Tests for the shared HTTP/retry helper of the LLM adapters."""

from __future__ import annotations

import asyncio
import email.utils
import logging
from collections.abc import Callable

import httpx
import pytest
import respx

from agentlens.llm import (
    LLMConnectionError,
    LLMError,
    LLMHTTPError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    RetryPolicy,
)
from agentlens.llm._http import (
    JSONHTTPClient,
    equal_jitter,
    is_retryable_status,
    parse_retry_after,
)

URL = "https://llm.test/v1/chat"
SECRET = "sk-test-secret-key-123456"


class Sleeps:
    """Records requested delays instead of sleeping."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, delay: float) -> None:
        self.delays.append(delay)

    async def asleep(self, delay: float) -> None:
        self.delays.append(delay)


def policy(sleeps: Sleeps, **kwargs: object) -> RetryPolicy:
    return RetryPolicy(
        jitter=lambda d: d,
        sleep=sleeps,
        async_sleep=sleeps.asleep,
        **kwargs,  # type: ignore[arg-type]
    )


def make(sleeps: Sleeps, **kwargs: object) -> JSONHTTPClient:
    return JSONHTTPClient(timeout=5.0, retry=policy(sleeps, **kwargs))


def post(client: JSONHTTPClient) -> dict[str, object]:
    return client.post_json(URL, {"q": 1}, {"Authorization": f"Bearer {SECRET}"})


def apost(client: JSONHTTPClient) -> dict[str, object]:
    return asyncio.run(client.apost_json(URL, {"q": 1}, {"Authorization": f"Bearer {SECRET}"}))


Post = Callable[[JSONHTTPClient], dict[str, object]]
POSTS = [pytest.param(post, id="sync"), pytest.param(apost, id="async")]


class TestRetryPolicy:
    def test_defaults(self) -> None:
        retry = RetryPolicy()
        assert retry.max_retries == 2
        assert retry.jitter is equal_jitter

    def test_exponential_backoff_capped(self) -> None:
        retry = RetryPolicy(initial_delay=1.0, max_delay=5.0, jitter=lambda d: d)
        assert [retry.backoff(i) for i in range(5)] == [1.0, 2.0, 4.0, 5.0, 5.0]

    def test_retry_after_wins_and_is_capped(self) -> None:
        retry = RetryPolicy(max_retry_after=10.0, jitter=lambda d: 999.0)
        assert retry.delay(0, 3.0) == 3.0
        assert retry.delay(0, 30.0) == 10.0
        assert retry.delay(0) == 999.0

    @pytest.mark.parametrize(
        "kwargs", [{"max_retries": -1}, {"initial_delay": -1.0}, {"max_retry_after": -0.1}]
    )
    def test_rejects_negative_values(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError, match=">= 0"):
            RetryPolicy(**kwargs)  # type: ignore[arg-type]

    def test_equal_jitter_bounds(self) -> None:
        for _ in range(200):
            assert 2.0 <= equal_jitter(4.0) <= 4.0


class TestRetryableStatus:
    @pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504, 529])
    def test_retryable(self, status: int) -> None:
        assert is_retryable_status(status)

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422])
    def test_not_retryable(self, status: int) -> None:
        assert not is_retryable_status(status)


class TestParseRetryAfter:
    def test_absent(self) -> None:
        assert parse_retry_after({}) is None

    def test_seconds_case_insensitive(self) -> None:
        assert parse_retry_after({"Retry-After": "2.5"}) == 2.5

    def test_milliseconds_preferred(self) -> None:
        assert parse_retry_after({"retry-after-ms": "250", "retry-after": "9"}) == 0.25

    def test_bad_milliseconds_fall_back(self) -> None:
        assert parse_retry_after({"retry-after-ms": "soon", "retry-after": "3"}) == 3.0
        assert parse_retry_after({"retry-after-ms": "-5"}) is None

    def test_http_date(self) -> None:
        now = 1_700_000_000.0
        header = email.utils.formatdate(now + 7, usegmt=True)
        assert parse_retry_after({"retry-after": header}, now=now) == pytest.approx(7.0)

    def test_http_date_in_past_means_now(self) -> None:
        header = email.utils.formatdate(1_000.0, usegmt=True)
        assert parse_retry_after({"retry-after": header}) == 0.0

    @pytest.mark.parametrize("value", ["-3", "tomorrow-ish", ""])
    def test_invalid(self, value: str) -> None:
        assert parse_retry_after({"retry-after": value}) is None


@pytest.mark.parametrize("call", POSTS)
class TestJSONHTTPClient:
    def test_success_sends_json_and_headers(self, call: Post, respx_mock: respx.Router) -> None:
        route = respx_mock.post(URL).respond(json={"ok": True})
        sleeps = Sleeps()
        assert call(make(sleeps)) == {"ok": True}
        request = route.calls.last.request
        assert request.headers["content-type"] == "application/json"
        assert request.headers["authorization"] == f"Bearer {SECRET}"
        assert request.content == b'{"q": 1}'
        assert sleeps.delays == []

    def test_retries_5xx_with_backoff(self, call: Post, respx_mock: respx.Router) -> None:
        route = respx_mock.post(URL).mock(
            side_effect=[httpx.Response(500), httpx.Response(503), httpx.Response(200, json={})]
        )
        sleeps = Sleeps()
        assert call(make(sleeps, initial_delay=1.0)) == {}
        assert route.call_count == 3
        assert sleeps.delays == [1.0, 2.0]

    def test_honours_retry_after(self, call: Post, respx_mock: respx.Router) -> None:
        respx_mock.post(URL).mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "7"}),
                httpx.Response(200, json={"n": 1}),
            ]
        )
        sleeps = Sleeps()
        assert call(make(sleeps)) == {"n": 1}
        assert sleeps.delays == [7.0]

    def test_rate_limit_exhausted(self, call: Post, respx_mock: respx.Router) -> None:
        route = respx_mock.post(URL).respond(
            429, json={"error": {"message": "Rate limit reached", "type": "requests"}}
        )
        sleeps = Sleeps()
        with pytest.raises(LLMRateLimitError, match="HTTP 429: Rate limit reached") as info:
            call(make(sleeps, max_retries=3))
        assert route.call_count == 4
        assert len(sleeps.delays) == 3
        assert info.value.status_code == 429
        assert "Rate limit reached" in info.value.body
        assert isinstance(info.value, LLMHTTPError)

    def test_client_error_not_retried(self, call: Post, respx_mock: respx.Router) -> None:
        route = respx_mock.post(URL).respond(401, json={"error": {"message": "bad key"}})
        sleeps = Sleeps()
        with pytest.raises(LLMHTTPError) as info:
            call(make(sleeps))
        assert route.call_count == 1
        assert info.value.status_code == 401
        assert not isinstance(info.value, LLMRateLimitError)
        assert SECRET not in str(info.value)

    def test_no_retries_when_disabled(self, call: Post, respx_mock: respx.Router) -> None:
        route = respx_mock.post(URL).respond(500, text="boom")
        with pytest.raises(LLMHTTPError, match="HTTP 500: boom"):
            call(make(Sleeps(), max_retries=0))
        assert route.call_count == 1

    def test_transport_error_retried(self, call: Post, respx_mock: respx.Router) -> None:
        respx_mock.post(URL).mock(
            side_effect=[httpx.ConnectError("refused"), httpx.Response(200, json={"a": 1})]
        )
        sleeps = Sleeps()
        assert call(make(sleeps)) == {"a": 1}
        assert sleeps.delays == [0.5]

    def test_connection_error_exhausted(self, call: Post, respx_mock: respx.Router) -> None:
        route = respx_mock.post(URL).mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(LLMConnectionError, match="ConnectError: refused") as info:
            call(make(Sleeps(), max_retries=1))
        assert route.call_count == 2
        assert not isinstance(info.value, LLMTimeoutError)
        assert isinstance(info.value.__cause__, httpx.ConnectError)

    def test_timeout(self, call: Post, respx_mock: respx.Router) -> None:
        respx_mock.post(URL).mock(side_effect=httpx.ReadTimeout("slow"))
        with pytest.raises(LLMTimeoutError):
            call(make(Sleeps(), max_retries=1))

    def test_unsupported_protocol_not_retried(self, call: Post, respx_mock: respx.Router) -> None:
        route = respx_mock.post(URL).mock(side_effect=httpx.UnsupportedProtocol("ftp?"))
        with pytest.raises(LLMConnectionError):
            call(make(Sleeps()))
        assert route.call_count == 1

    def test_non_json_body(self, call: Post, respx_mock: respx.Router) -> None:
        respx_mock.post(URL).respond(200, text="<html>proxy</html>")
        with pytest.raises(LLMResponseError, match="non-JSON"):
            call(make(Sleeps()))

    def test_non_object_json(self, call: Post, respx_mock: respx.Router) -> None:
        respx_mock.post(URL).respond(200, json=[1, 2])
        with pytest.raises(LLMResponseError, match="expected an object"):
            call(make(Sleeps()))

    def test_errors_are_llm_errors(self, call: Post, respx_mock: respx.Router) -> None:
        respx_mock.post(URL).respond(400, json={"error": "plain string error"})
        with pytest.raises(LLMError, match="plain string error"):
            call(make(Sleeps()))


class TestErrorDetail:
    @pytest.mark.parametrize(
        ("kwargs", "detail"),
        [
            ({"json": {"message": "top-level"}}, "top-level"),
            ({"json": {"error": {"code": 1}}}, '{"error":{"code":1}}'),
            ({"text": ""}, "Bad Request"),
        ],
    )
    def test_detail_variants(
        self, kwargs: dict[str, object], detail: str, respx_mock: respx.Router
    ) -> None:
        respx_mock.post(URL).respond(400, **kwargs)  # type: ignore[arg-type]
        with pytest.raises(LLMHTTPError) as info:
            post(make(Sleeps()))
        assert str(info.value).endswith(detail)

    def test_query_string_not_in_message(self, respx_mock: respx.Router) -> None:
        respx_mock.post(URL).respond(404, text="nope")
        with pytest.raises(LLMHTTPError) as info:
            make(Sleeps()).post_json(f"{URL}?key=secret", {}, {})
        assert "secret" not in str(info.value)
        assert URL in str(info.value)


class TestLogging:
    def test_retry_logged_without_secrets(
        self, respx_mock: respx.Router, caplog: pytest.LogCaptureFixture
    ) -> None:
        respx_mock.post(URL).mock(side_effect=[httpx.Response(502), httpx.Response(200, json={})])
        with caplog.at_level(logging.DEBUG, logger="agentlens.llm"):
            post(make(Sleeps()))
        assert "retrying POST" in caplog.text
        assert "HTTP 502" in caplog.text
        assert SECRET not in caplog.text


class TestClientLifecycle:
    def test_reuses_and_closes_owned_client(self, respx_mock: respx.Router) -> None:
        respx_mock.post(URL).respond(json={})
        client = make(Sleeps())
        post(client)
        first = client._client
        post(client)
        assert client._client is first
        client.close()
        assert client._client is None
        assert first is not None
        assert first.is_closed
        client.close()  # idempotent

    def test_injected_clients_not_closed(self, respx_mock: respx.Router) -> None:
        respx_mock.post(URL).respond(json={"x": 1})
        sync = httpx.Client()

        async def scenario() -> dict[str, object]:
            async with httpx.AsyncClient() as async_client:
                client = JSONHTTPClient(
                    timeout=1.0,
                    retry=RetryPolicy(),
                    http_client=sync,
                    async_http_client=async_client,
                )
                assert client.post_json(URL, {}, {}) == {"x": 1}
                result = await client.apost_json(URL, {}, {})
                client.close()
                assert not async_client.is_closed
                return result

        assert asyncio.run(scenario()) == {"x": 1}
        assert not sync.is_closed
        sync.close()
