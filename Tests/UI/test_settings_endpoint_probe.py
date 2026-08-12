"""Focused contract tests for the Settings provider endpoint probe."""

from __future__ import annotations

import httpx
import pytest

from tldw_chatbook.UI.Screens.settings_endpoint_probe import (
    SettingsEndpointProbeOutcome,
    probe_settings_endpoint,
)

pytestmark = pytest.mark.local_server_probe


def _client(handler, *, follow_redirects: bool = False) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        follow_redirects=follow_redirects,
    )


def test_outcome_keeps_legacy_constructor_and_properties() -> None:
    outcome = SettingsEndpointProbeOutcome(
        reachable=True,
        summary="reachable (3 models)",
        model_count=3,
    )

    assert outcome.state == "reachable"
    assert outcome.reachable is True
    assert outcome.model_count == 3
    assert outcome.model_ids == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("entered", "expected"),
    (
        (
            "https://example.test/v1/chat/completions",
            "https://example.test/v1/models",
        ),
        ("https://example.test/v1", "https://example.test/v1/models"),
        (
            "https://example.test/proxy/v1/chat/completions",
            "https://example.test/proxy/v1/models",
        ),
        (
            "https://example.test/proxy/v1/models",
            "https://example.test/proxy/v1/models",
        ),
    ),
)
async def test_probe_derives_models_route_exactly_once(
    entered: str, expected: str
) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(404)

    async with _client(handler) as client:
        outcome = await probe_settings_endpoint(
            entered,
            provider="custom",
            http_client=client,
        )

    assert seen == [expected]
    assert outcome.state == "model_listing_unavailable"
    assert outcome.category == "http_status"
    assert outcome.summary == "Model listing unavailable; chat endpoint not tested"
    assert outcome.reachable is False


@pytest.mark.asyncio
async def test_probe_parses_and_sanitizes_bounded_openai_model_ids() -> None:
    hostile = " bad\x1b[31m\nmodel " + "x" * 300
    payload = {
        "data": [
            {"id": hostile},
            {"id": "model-b"},
            {"id": "model-b"},
        ]
    }

    async with _client(lambda request: httpx.Response(200, json=payload)) as client:
        outcome = await probe_settings_endpoint(
            "http://127.0.0.1:9099/v1",
            http_client=client,
        )

    assert outcome.state == "reachable"
    assert outcome.category is None
    assert outcome.model_count == 2
    assert outcome.model_ids[1] == "model-b"
    assert len(outcome.model_ids[0]) <= 120
    assert all(character.isprintable() for character in outcome.model_ids[0])


@pytest.mark.asyncio
async def test_probe_accepts_valid_empty_model_listing() -> None:
    async with _client(
        lambda request: httpx.Response(200, json={"object": "list", "data": []})
    ) as client:
        outcome = await probe_settings_endpoint(
            "http://127.0.0.1:9099",
            http_client=client,
        )

    assert outcome.state == "reachable"
    assert outcome.model_ids == ()
    assert outcome.model_count == 0
    assert outcome.summary == "reachable (0 models)"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "category", "summary"),
    (
        (401, "unauthorized", "unreachable: unauthorized"),
        (403, "forbidden", "unreachable: forbidden"),
        (500, "http_status", "unreachable: HTTP 500"),
    ),
)
async def test_probe_classifies_bounded_http_failures(
    status: int, category: str, summary: str
) -> None:
    async with _client(
        lambda request: httpx.Response(status, text="secret response body")
    ) as client:
        outcome = await probe_settings_endpoint(
            "https://example.test/v1",
            http_client=client,
        )

    assert outcome.state == "unreachable"
    assert outcome.category == category
    assert outcome.summary == summary
    assert "secret response body" not in outcome.summary


@pytest.mark.asyncio
async def test_probe_does_not_follow_redirects_even_with_redirecting_client() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(302, headers={"location": "https://secret.test/models"})

    async with _client(handler, follow_redirects=True) as client:
        outcome = await probe_settings_endpoint(
            "https://example.test/v1",
            http_client=client,
        )

    assert seen == ["https://example.test/v1/models"]
    assert outcome.state == "unreachable"
    assert outcome.category == "http_status"
    assert outcome.summary == "unreachable: HTTP 302"
    assert "secret.test" not in outcome.summary


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "category", "summary"),
    (
        (httpx.ReadTimeout("secret timeout URL"), "timeout", "unreachable: timeout"),
        (
            httpx.ConnectError("secret refused URL"),
            "connection_refused",
            "unreachable: connection refused",
        ),
        (
            httpx.RemoteProtocolError("secret protocol body"),
            "connection_error",
            "unreachable: connection error",
        ),
    ),
)
async def test_probe_classifies_transport_failures_without_leaking_details(
    failure: Exception, category: str, summary: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise failure

    async with _client(handler) as client:
        outcome = await probe_settings_endpoint(
            "https://example.test/v1",
            http_client=client,
        )

    assert outcome.state == "unreachable"
    assert outcome.category == category
    assert "secret" not in outcome.summary
    assert "example.test" not in outcome.summary
    assert outcome.summary == summary


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    (
        "not json",
        {"status": "ok", "body": "secret body"},
    ),
)
async def test_probe_rejects_malformed_or_unrecognized_payload(payload: object) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(payload, str):
            return httpx.Response(200, text=payload)
        return httpx.Response(200, json=payload)

    async with _client(handler) as client:
        outcome = await probe_settings_endpoint(
            "https://example.test/v1",
            http_client=client,
        )

    assert outcome.state == "unreachable"
    assert outcome.category == "invalid_payload"
    assert outcome.summary == "unreachable: invalid models response"
    assert "secret" not in outcome.summary


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ("ollama", "local_ollama"))
async def test_probe_ollama_uses_contract_root_for_api_tags_fallback(
    provider: str,
) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.path.endswith("/v1/models"):
            return httpx.Response(404)
        return httpx.Response(200, json={"models": [{"name": "llama3:latest"}]})

    async with _client(handler) as client:
        outcome = await probe_settings_endpoint(
            "https://example.test/proxy/v1/chat/completions",
            provider=provider,
            http_client=client,
        )

    assert seen == [
        "https://example.test/proxy/v1/models",
        "https://example.test/proxy/api/tags",
    ]
    assert outcome.state == "reachable"
    assert outcome.model_ids == ("llama3:latest",)


@pytest.mark.asyncio
async def test_invalid_input_never_makes_a_request_or_echoes_input() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("invalid input must not be requested")

    entered = "https://user:password@example.test/v1?token=secret"
    async with _client(handler) as client:
        outcome = await probe_settings_endpoint(entered, http_client=client)

    assert outcome.state == "unreachable"
    assert outcome.category is None
    assert outcome.summary == "unreachable: invalid endpoint URL"
    assert not any(part in outcome.summary for part in ("password", "token", "secret"))
