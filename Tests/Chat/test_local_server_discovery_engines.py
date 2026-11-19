"""TASK-34100.6 slice E: discovery finds more engines and names what failed.

Review findings provider-13 (LM Studio, Jan and KoboldCpp ports were never
probed; the Custom rows' own endpoints were skipped), provider-22 (one server
listed twice, as 127.0.0.1 and as localhost), coverage-19 (a certificate
failure read like an outage) and provider-15 (https against a plain-HTTP
server read 'could not reach the server').

The exception shapes below are copied from real failures captured on
2026-10-04 (lessons-live-verification: build a fake failure from a real one):
a self-signed llama-server on :8443 raised ``ConnectError <-
SSLCertVerificationError``; https against a proxy that answers at once raised
``ConnectError <- SSLError(WRONG_VERSION_NUMBER)``; https against llama-server
on :8181 timed out with ``SSLWantReadError`` in the chain.
"""

from __future__ import annotations

import asyncio
import ssl

import httpx
import pytest

from tldw_chatbook.Chat.local_server_discovery import (
    DiscoveredLocalServer,
    build_local_server_candidates,
    connect_error_is_plain_http,
    connect_error_is_refused,
    connect_error_is_untrusted_certificate,
    discover_local_servers,
    one_server_per_port,
)

pytestmark = pytest.mark.local_server_probe


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _models(*model_ids: str) -> dict:
    return {"object": "list", "data": [{"id": model_id} for model_id in model_ids]}


def _chained(outer: BaseException, inner: BaseException) -> BaseException:
    outer.__cause__ = inner
    return outer


def test_lm_studio_jan_and_koboldcpp_ports_are_candidates_named_by_engine():
    """provider-13: each sets up the Custom row, named by its engine."""
    candidates = {
        (c.provider_key, c.base_url, c.engine) for c in build_local_server_candidates({})
    }
    assert ("custom", "http://127.0.0.1:1234", "LM Studio") in candidates
    assert ("custom", "http://127.0.0.1:1337", "Jan") in candidates
    assert ("custom", "http://127.0.0.1:5001", "KoboldCpp") in candidates


def test_the_custom_rows_configured_endpoints_are_probed():
    candidates = build_local_server_candidates(
        {
            "api_settings": {
                "custom": {"api_url": "http://localhost:4321/v1/chat/completions"},
                "custom_2": {"api_url": "http://127.0.0.1:4322/v1"},
            }
        }
    )
    pairs = {(c.provider_key, c.base_url) for c in candidates}
    assert ("custom", "http://localhost:4321") in pairs
    assert ("custom_2", "http://127.0.0.1:4322") in pairs


@pytest.mark.asyncio
async def test_new_ports_are_probed_in_parallel_within_one_timeout():
    """provider-13: more candidates never lengthen the scan."""

    in_flight = 0
    most = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, most
        in_flight += 1
        most = max(most, in_flight)
        try:
            await asyncio.sleep(0.3)
        finally:
            in_flight -= 1
        raise httpx.ConnectError("refused", request=request)

    candidates = build_local_server_candidates({})
    assert len(candidates) >= 6
    assert await discover_local_servers({}, http_client=_client(handler)) == ()
    # Structural, not wall-clock (review round 1, F9: a 0.9 s bound flaked
    # under xdist): every new port's probe was in flight at once.
    assert most >= 6, f"only {most} probes overlapped; candidates were probed serially"


@pytest.mark.asyncio
async def test_every_probe_keeps_the_existing_short_timeout(monkeypatch):
    """provider-13: the new ports ride the same short per-request timeout.

    With every probe in flight at once (above), the scan's length is bounded
    by this one timeout, unchanged from before :1234/:1337/:5001 were added.
    """
    from tldw_chatbook.Chat import local_server_discovery as discovery

    seen: dict[str, float] = {}

    async def fake_probe(base_url, *, provider_key, http_client, timeout):
        seen[base_url] = timeout
        return None

    monkeypatch.setattr(discovery, "probe_models_endpoint", fake_probe)
    assert discovery.DISCOVERY_PROBE_TIMEOUT_SECONDS == 2.5, "the pre-existing value"

    assert await discover_local_servers({}, http_client=_client(None)) == ()

    assert {"http://127.0.0.1:1234", "http://127.0.0.1:1337"} <= set(seen)
    assert set(seen.values()) == {discovery.DISCOVERY_PROBE_TIMEOUT_SECONDS}


@pytest.mark.asyncio
async def test_lm_studio_found_on_1234_is_the_custom_row_named_lm_studio():
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.port == 1234:
            return httpx.Response(200, json=_models("qwen2.5-7b-instruct"))
        raise httpx.ConnectError("refused", request=request)

    servers = await discover_local_servers({}, http_client=_client(handler))
    assert servers == (
        DiscoveredLocalServer(
            provider_key="custom",
            base_url="http://127.0.0.1:1234",
            model_ids=("qwen2.5-7b-instruct",),
            engine="LM Studio",
        ),
    )


@pytest.mark.asyncio
async def test_one_server_answering_on_both_spellings_is_one_result():
    """provider-22: both spellings probed, one row, the first that answered."""
    probed: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        probed.append(f"{request.url.host}:{request.url.port}")
        if request.url.port == 11434:
            return httpx.Response(200, json=_models("llama3.2"))
        raise httpx.ConnectError("refused", request=request)

    config = {"api_settings": {"ollama": {"api_url": "http://localhost:11434"}}}
    servers = await discover_local_servers(config, http_client=_client(handler))
    assert "127.0.0.1:11434" in probed and "localhost:11434" in probed
    assert [server.base_url for server in servers] == ["http://127.0.0.1:11434"]


@pytest.mark.asyncio
async def test_a_server_bound_only_to_localhost_keeps_that_spelling():
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "localhost" and request.url.port == 11434:
            return httpx.Response(200, json=_models("llama3.2"))
        raise httpx.ConnectError("refused", request=request)

    config = {"api_settings": {"ollama": {"api_url": "http://localhost:11434"}}}
    servers = await discover_local_servers(config, http_client=_client(handler))
    assert [server.base_url for server in servers] == ["http://localhost:11434"]


def test_one_server_per_port_merges_loopback_spellings_only():
    servers = (
        DiscoveredLocalServer("llama_cpp", "http://127.0.0.1:9099"),
        DiscoveredLocalServer("llama_cpp", "http://localhost:9099"),
        DiscoveredLocalServer("llama_cpp", "http://localhost:8181"),
    )
    assert [s.base_url for s in one_server_per_port(servers)] == [
        "http://127.0.0.1:9099",
        "http://localhost:8181",
    ]


def test_a_certificate_verification_failure_is_untrusted_not_unreachable():
    """coverage-19: the live shape from a self-signed llama-server."""
    error = _chained(
        httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"),
        ssl.SSLCertVerificationError(1, "certificate verify failed: self-signed"),
    )
    assert connect_error_is_untrusted_certificate(error)
    assert not connect_error_is_refused(error)
    assert not connect_error_is_plain_http(error)


def test_https_against_plain_http_is_recognised_when_the_server_answers_at_once():
    """provider-15: the answered-at-once shape is evidence on its own.

    A stalled handshake is not (review round 1, F2): anyio awaits every TLS
    read inside ``except SSLWantReadError``, so every https timeout carries it.
    ``classify_transport_failure`` confirms that case with a plain GET instead
    (``test_provider_tls_failures.py``).
    """
    wrong_version = ssl.SSLError(1, "[SSL: WRONG_VERSION_NUMBER] wrong version number")
    wrong_version.reason = "WRONG_VERSION_NUMBER"
    answered = _chained(httpx.ConnectError("wrong version number"), wrong_version)
    stalled = _chained(
        httpx.ConnectTimeout(""),
        _chained(TimeoutError(), ssl.SSLWantReadError(2, "The operation did not complete")),
    )
    read_stall = _chained(
        httpx.ReadTimeout(""),
        _chained(TimeoutError(), ssl.SSLWantReadError(2, "The operation did not complete")),
    )
    assert connect_error_is_plain_http(answered)
    assert not connect_error_is_plain_http(stalled)
    assert not connect_error_is_plain_http(read_stall)
    assert not connect_error_is_untrusted_certificate(answered)
    refused = _chained(httpx.ConnectError("refused"), ConnectionRefusedError(61, "x"))
    assert not connect_error_is_plain_http(refused)
