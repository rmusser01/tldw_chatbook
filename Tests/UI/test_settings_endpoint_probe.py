"""Focused contract tests for the Settings provider endpoint probe."""

from __future__ import annotations

import errno

import httpx
import pytest

import tldw_chatbook.UI.Screens.settings_endpoint_probe as settings_probe_module
from tldw_chatbook.UI.Screens.settings_endpoint_probe import (
    SettingsEndpointProbeOutcome,
    probe_settings_endpoint,
)

pytestmark = pytest.mark.local_server_probe
_EXPECTED_MODEL_PROBE_RESPONSE_MAX_BYTES = 1024 * 1024


def _client(handler, *, follow_redirects: bool = False) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        follow_redirects=follow_redirects,
    )


class _TrackingAsyncStream(httpx.AsyncByteStream):
    def __init__(self, *chunks: bytes) -> None:
        self.chunks = chunks
        self.iterated_chunks = 0
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            self.iterated_chunks += 1
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


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


def test_outcome_legacy_model_count_participates_in_equality() -> None:
    one_model = SettingsEndpointProbeOutcome(
        reachable=True,
        summary="reachable",
        model_count=1,
    )
    two_models = SettingsEndpointProbeOutcome(
        reachable=True,
        summary="reachable",
        model_count=2,
    )

    assert one_model != two_models
    assert one_model == SettingsEndpointProbeOutcome(
        reachable=True,
        summary="reachable",
        model_count=1,
    )


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    (
        (
            {"state": "reachable", "reachable": False},
            "Endpoint probe state conflicts with reachable.",
        ),
        (
            {"state": "unreachable", "reachable": True},
            "Endpoint probe state conflicts with reachable.",
        ),
        (
            {"state": "reachable", "category": "timeout"},
            "Reachable probe cannot include a failure category.",
        ),
        (
            {"state": "reachable", "model_count": -1},
            "Model count must be a non-negative integer.",
        ),
        (
            {"state": "reachable", "model_count": True},
            "Model count must be a non-negative integer.",
        ),
        (
            {
                "state": "reachable",
                "model_ids": ("model-a",),
                "model_count": 2,
            },
            "Reachable probe model data is inconsistent.",
        ),
        (
            {"state": "reachable", "model_ids": (), "model_count": 3},
            "Reachable probe model data is inconsistent.",
        ),
        (
            {"state": "unreachable", "model_ids": ("model-a",)},
            "Unreachable probe data must be empty.",
        ),
        (
            {"state": "unreachable", "model_count": 0},
            "Unreachable probe data must be empty.",
        ),
        (
            {"state": "model_listing_unavailable", "model_count": 0},
            "Model listing probe data must be empty.",
        ),
        (
            {"state": "model_listing_unavailable", "model_ids": ("model-a",)},
            "Model listing probe data must be empty.",
        ),
        (
            {"state": "model_listing_unavailable", "category": "timeout"},
            "Model listing probe category is invalid.",
        ),
        (
            {"state": "unreachable", "category": "not-bounded"},
            "Endpoint probe category is invalid.",
        ),
    ),
)
def test_outcome_rejects_impossible_state_category_and_model_data(
    kwargs: dict[str, object],
    expected_message: str,
) -> None:
    with pytest.raises(ValueError) as exc_info:
        SettingsEndpointProbeOutcome(summary="bounded", **kwargs)

    assert str(exc_info.value) == expected_message
    assert "model-a" not in str(exc_info.value)


def test_outcome_accepts_consistent_structured_and_legacy_data() -> None:
    structured = SettingsEndpointProbeOutcome(
        state="reachable",
        summary="reachable",
        model_ids=("model-a", "model-b"),
        model_count=2,
    )
    unavailable = SettingsEndpointProbeOutcome(
        state="model_listing_unavailable",
        category="http_status",
        summary="unavailable",
    )

    assert structured.model_count == 2
    assert unavailable.category == "http_status"


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    (
        ({}, "Provide endpoint probe state or reachable."),
        (
            {"reachable": "yes", "summary": "bounded"},
            "Reachable must be a boolean.",
        ),
        (
            {"reachable": 1, "summary": "bounded"},
            "Reachable must be a boolean.",
        ),
        (
            {"state": "reachable", "summary": 123},
            "Endpoint probe summary must be text.",
        ),
        (
            {
                "state": "reachable",
                "summary": "bounded",
                "model_ids": "model-a",
            },
            "Model IDs are invalid.",
        ),
        (
            {
                "state": "reachable",
                "summary": "bounded",
                "model_ids": b"model-a",
            },
            "Model IDs are invalid.",
        ),
        (
            {
                "state": "reachable",
                "summary": "bounded",
                "model_ids": ("model-a", 1),
            },
            "Model IDs are invalid.",
        ),
    ),
)
def test_outcome_rejects_invalid_constructor_types(
    kwargs: dict[str, object],
    expected_message: str,
) -> None:
    with pytest.raises(ValueError) as exc_info:
        SettingsEndpointProbeOutcome(**kwargs)

    assert str(exc_info.value) == expected_message


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
async def test_probe_accepts_bare_list_string_model_entry() -> None:
    async with _client(
        lambda request: httpx.Response(200, json=["model-a"])
    ) as client:
        outcome = await probe_settings_endpoint(
            "http://127.0.0.1:9099",
            http_client=client,
        )

    assert outcome.state == "reachable"
    assert outcome.model_ids == ("model-a",)


@pytest.mark.asyncio
async def test_settings_probe_accepts_json_body_at_exact_byte_limit() -> None:
    prefix = b'{"data":[]}'
    body = prefix + b" " * (
        _EXPECTED_MODEL_PROBE_RESPONSE_MAX_BYTES - len(prefix)
    )

    async with _client(lambda request: httpx.Response(200, content=body)) as client:
        outcome = await probe_settings_endpoint(
            "https://example.test/v1",
            http_client=client,
        )

    assert outcome.state == "reachable"
    assert outcome.model_count == 0


@pytest.mark.asyncio
async def test_settings_probe_rejects_oversized_content_length_without_reading() -> None:
    stream = _TrackingAsyncStream(b"secret oversized response body")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-length": str(
                    _EXPECTED_MODEL_PROBE_RESPONSE_MAX_BYTES + 1
                )
            },
            stream=stream,
        )

    async with _client(handler) as client:
        outcome = await probe_settings_endpoint(
            "https://example.test/v1",
            http_client=client,
        )

    assert outcome.state == "unreachable"
    assert outcome.category == "invalid_payload"
    assert outcome.summary == "unreachable: models response too large"
    assert "secret" not in outcome.summary
    assert stream.iterated_chunks == 0
    assert stream.closed is True


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
            "connection_error",
            "unreachable: connection error",
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
async def test_probe_classifies_only_errno_refused_connect_errors_as_refused() -> None:
    async def refused_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("secret refused URL") from OSError(
            errno.ECONNREFUSED,
            "secret refused detail",
        )

    async def routing_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("secret routing URL") from OSError(
            errno.EHOSTUNREACH,
            "secret routing detail",
        )

    async with _client(refused_handler) as client:
        refused = await probe_settings_endpoint(
            "https://example.test/v1",
            http_client=client,
        )
    async with _client(routing_handler) as client:
        routing = await probe_settings_endpoint(
            "https://example.test/v1",
            http_client=client,
        )

    assert refused.category == "connection_refused"
    assert refused.summary == "unreachable: connection refused"
    assert routing.category == "connection_error"
    assert routing.summary == "unreachable: connection error"
    assert "secret" not in repr((refused, routing))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    (
        "not json",
        {"status": "ok", "body": "secret body"},
        {"data": [{"foo": "bar"}]},
        [[]],
        [123],
        {"models": [{"foo": "bar"}]},
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
async def test_probe_classifies_recursive_json_as_invalid_payload(monkeypatch) -> None:
    def recursive_loads(body: bytes) -> object:
        raise RecursionError("secret recursive decoder detail")

    monkeypatch.setattr(settings_probe_module.json, "loads", recursive_loads)

    async with _client(
        lambda request: httpx.Response(200, content=b'{"data": []}')
    ) as client:
        outcome = await probe_settings_endpoint(
            "https://example.test/v1",
            http_client=client,
        )

    assert outcome.state == "unreachable"
    assert outcome.category == "invalid_payload"
    assert outcome.summary == "unreachable: invalid models response"
    assert "secret" not in repr(outcome)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider",
    ("ollama", "Ollama", "local_ollama", "Local Ollama"),
)
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
