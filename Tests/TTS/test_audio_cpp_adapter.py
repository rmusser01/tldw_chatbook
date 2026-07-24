from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from time import monotonic
from typing import Any

import httpx
import pytest

from tldw_chatbook.TTS.adapter_types import (
    ProviderHealth,
    TTSModelInfo,
    TTSRequest,
)
from tldw_chatbook.TTS.adapters.audio_cpp import AudioCppAdapter
from tldw_chatbook.TTS.audio_cpp_config import AudioCppConfig


MAX_METADATA_BYTES = 512
AVAILABLE_HEALTH = ProviderHealth(state="available", fresh=True)
NOT_CONFIGURED_HEALTH = ProviderHealth(
    state="not_configured",
    fresh=True,
    diagnostic="No audio.cpp TTS models are configured",
    recovery_action="check_server",
)
TRANSIENT_HEALTH = ProviderHealth(
    state="unavailable",
    fresh=False,
    diagnostic="The audio.cpp server is unavailable",
    retryable=True,
    recovery_action="retry",
)
CONTRACT_HEALTH = ProviderHealth(
    state="unavailable",
    fresh=False,
    diagnostic="The audio.cpp server response is incompatible",
    recovery_action="check_server",
)
CLOSED_HEALTH = ProviderHealth(
    state="closed",
    fresh=False,
    diagnostic="The audio.cpp adapter is closed",
)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode()


def _health_body(models: int = 1) -> bytes:
    return _json_bytes({"status": "ok", "backend": "cpu", "models": models})


def _model(
    model_id: str = "model",
    *,
    family: str = "family",
    task: str = "tts",
    mode: str = "offline",
) -> dict[str, str]:
    return {
        "id": model_id,
        "object": "model",
        "owned_by": "engine",
        "family": family,
        "task": task,
        "mode": mode,
    }


def _models_body(*models: dict[str, str]) -> bytes:
    return _json_bytes({"object": "list", "data": list(models)})


def _voices_body(*voices: str) -> bytes:
    return _json_bytes({"voices": list(voices)})


def _config(**updates: Any) -> AudioCppConfig:
    values: dict[str, Any] = {
        "max_metadata_bytes": MAX_METADATA_BYTES,
        "max_catalog_models": 16,
        "max_voices_per_model": 16,
        "max_identifier_characters": 128,
    }
    values.update(updates)
    return AudioCppConfig.from_mapping(values)


@asynccontextmanager
async def _adapter(
    handler: Callable[[httpx.Request], Any],
    **config_updates: Any,
) -> AsyncIterator[AudioCppAdapter]:
    adapter = AudioCppAdapter(
        _config(**config_updates),
        transport=httpx.MockTransport(handler),
    )
    try:
        yield adapter
    finally:
        await adapter.close()


class TrackingStream(httpx.AsyncByteStream):
    def __init__(
        self,
        *items: bytes | BaseException,
    ) -> None:
        self.items = items
        self.read_count = 0
        self.close_count = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for item in self.items:
            self.read_count += 1
            if isinstance(item, BaseException):
                raise item
            yield item

    async def aclose(self) -> None:
        self.close_count += 1


def _streaming_response(
    body: bytes,
    *,
    status: int = 200,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    return httpx.Response(
        status,
        headers=headers,
        stream=TrackingStream(body),
    )


class BlockingStream(httpx.AsyncByteStream):
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.close_count = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.started.set()
        await self.release.wait()
        yield self.body

    async def aclose(self) -> None:
        self.close_count += 1


@pytest.mark.asyncio
async def test_construction_is_lazy_and_first_readiness_uses_required_order() -> None:
    requests: list[httpx.Request] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body = (
            _health_body() if request.url.path == "/health" else _models_body(_model())
        )
        headers = {"Content-Encoding": " Identity "}
        if request.url.path == "/health":
            headers["Set-Cookie"] = "remote_session=REMOTE_COOKIE_SENTINEL"
        return httpx.Response(
            200,
            headers=headers,
            stream=TrackingStream(body),
        )

    async with _adapter(respond) as adapter:
        assert requests == []

        await adapter.ensure_ready()

        assert [request.url.path for request in requests] == [
            "/health",
            "/v1/models",
        ]
        assert all(
            request.headers["accept-encoding"] == "identity" for request in requests
        )
        assert all("authorization" not in request.headers for request in requests)
        assert all("cookie" not in request.headers for request in requests)
        assert requests[0].extensions["timeout"] == {
            "connect": 5.0,
            "read": None,
            "write": None,
            "pool": None,
        }


@pytest.mark.asyncio
async def test_catalog_maps_only_tts_models_and_caches_until_forced_refresh() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/health":
            return _streaming_response(_health_body(2))
        return _streaming_response(
            _models_body(
                _model("speech", family="pocket_tts", mode="native"),
                _model("transcriber", family="whisper", task="stt"),
            ),
        )

    async with _adapter(respond) as adapter:
        first = await adapter.get_catalog()
        cached = await adapter.get_catalog()
        assert first is cached
        assert first.provider_id == "audio_cpp"
        assert first.revision == 1
        assert first.health == AVAILABLE_HEALTH
        assert first.approximate is False
        assert first.models == (
            TTSModelInfo(
                model_id="speech",
                display_name="speech",
                family="pocket_tts",
                upstream_mode="native",
                formats=("wav",),
                voices=(),
                supports_speed=False,
                supports_options=(),
                omit_voice_uses_server_default=True,
            ),
        )
        assert len(requests) == 2

        refreshed = await adapter.get_catalog(refresh=True)

        assert refreshed.revision == 2
        assert refreshed.models == first.models
        assert len(requests) == 4


@pytest.mark.asyncio
async def test_zero_tts_models_is_fresh_not_configured() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        return _streaming_response(
            _models_body(_model("transcriber", task="stt")),
        )

    async with _adapter(respond) as adapter:
        catalog = await adapter.get_catalog()

    assert catalog.revision == 1
    assert catalog.models == ()
    assert catalog.health == NOT_CONFIGURED_HEALTH


@pytest.mark.asyncio
async def test_concurrent_refreshes_from_one_revision_are_coalesced() -> None:
    requests: list[str] = []
    refresh_started = asyncio.Event()
    release_refresh = asyncio.Event()

    async def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if len(requests) > 2 and request.url.path == "/health":
            refresh_started.set()
            await release_refresh.wait()
        body = (
            _health_body() if request.url.path == "/health" else _models_body(_model())
        )
        return _streaming_response(body)

    async with _adapter(respond) as adapter:
        assert (await adapter.get_catalog()).revision == 1
        first = asyncio.create_task(adapter.get_catalog(refresh=True))
        await refresh_started.wait()
        second = asyncio.create_task(adapter.get_catalog(refresh=True))
        await asyncio.sleep(0)
        release_refresh.set()

        first_result, second_result = await asyncio.gather(first, second)

        assert first_result.revision == second_result.revision == 2
        assert requests == ["/health", "/v1/models", "/health", "/v1/models"]


@pytest.mark.asyncio
async def test_first_and_stale_transport_failures_have_safe_retryable_health() -> None:
    phase = "first_failure"
    requests: list[str] = []
    voice_requests = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal voice_requests
        requests.append(request.url.path)
        if (
            phase in {"first_failure", "stale_failure"}
            and request.url.path == "/health"
        ):
            raise httpx.ConnectError("REMOTE TRANSPORT SENTINEL", request=request)
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        if request.url.path == "/v1/models":
            return _streaming_response(_models_body(_model()))
        voice_requests += 1
        return _streaming_response(_voices_body("voice"))

    async with _adapter(respond) as adapter:
        first_failure = await adapter.get_catalog()
        assert first_failure.revision == 0
        assert first_failure.models == ()
        assert first_failure.health == TRANSIENT_HEALTH
        assert requests == ["/health", "/health"]

        phase = "ready"
        ready = await adapter.get_catalog()
        assert ready.revision == 1
        assert await adapter.get_voices("model") == ("voice",)

        phase = "stale_failure"
        stale = await adapter.get_catalog(refresh=True)
        assert stale.revision == ready.revision
        assert stale.models == ready.models
        assert stale.health == TRANSIENT_HEALTH

        assert await adapter.get_voices("model") == ("voice",)
        assert voice_requests == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [408, 425, 429, 500, 502, 503, 504])
async def test_transient_status_retries_exactly_once(status: int) -> None:
    requests = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(status, content=b"REMOTE STATUS BODY")

    async with _adapter(respond) as adapter:
        catalog = await adapter.get_catalog()

    assert requests == 2
    assert catalog.health == TRANSIENT_HEALTH


@pytest.mark.asyncio
async def test_eligible_transport_and_status_failures_can_recover_on_retry() -> None:
    health_attempts = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal health_attempts
        if request.url.path == "/health":
            health_attempts += 1
            if health_attempts == 1:
                raise httpx.ReadTimeout("REMOTE TIMEOUT", request=request)
            if health_attempts == 2:
                return httpx.Response(503, content=b"REMOTE BUSY")
            return _streaming_response(_health_body())
        return _streaming_response(_models_body(_model()))

    async with _adapter(respond) as adapter:
        first = await adapter.get_catalog()
        second = await adapter.get_catalog()

    assert first.health == TRANSIENT_HEALTH
    assert second.health == AVAILABLE_HEALTH
    assert second.revision == 1
    assert health_attempts == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "malformed",
        "oversized",
        "compressed",
        "malformed_length",
        "duplicate_length",
        "length_mismatch",
        "preconsumed_stream",
        "redirect",
        "unexpected_status",
    ],
)
async def test_required_contract_failures_are_nonretryable_and_retain_stale_catalog(
    case: str,
) -> None:
    phase = "ready"
    requests: list[str] = []
    failure_streams: list[TrackingStream] = []

    def failure_response() -> httpx.Response:
        if case == "malformed":
            return _streaming_response(b"{REMOTE MALFORMED")
        if case == "oversized":
            return _streaming_response(b"x" * (MAX_METADATA_BYTES + 1))
        if case == "compressed":
            return _streaming_response(
                _health_body(),
                headers={"Content-Encoding": "gzip"},
            )
        if case == "malformed_length":
            stream = TrackingStream(_health_body())
            failure_streams.append(stream)
            return httpx.Response(
                200,
                headers={"Content-Length": "12x"},
                stream=stream,
            )
        if case == "duplicate_length":
            stream = TrackingStream(_health_body())
            failure_streams.append(stream)
            return httpx.Response(
                200,
                headers=[
                    (b"Content-Length", b"1"),
                    (b"Content-Length", b"1"),
                ],
                stream=stream,
            )
        if case == "length_mismatch":
            stream = TrackingStream(_health_body())
            failure_streams.append(stream)
            return httpx.Response(
                200,
                headers={"Content-Length": "1"},
                stream=stream,
            )
        if case == "preconsumed_stream":
            return httpx.Response(200, content=_health_body())
        if case == "redirect":
            return httpx.Response(
                302,
                headers={"Location": "https://redirect-sentinel.invalid/private"},
            )
        return httpx.Response(418, content=b"REMOTE STATUS BODY")

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if phase == "failure" and request.url.path == "/health":
            return failure_response()
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        return _streaming_response(_models_body(_model("retained")))

    async with _adapter(respond) as adapter:
        ready = await adapter.get_catalog()
        phase = "failure"
        failed = await adapter.get_catalog(refresh=True)

    assert failed.revision == ready.revision == 1
    assert failed.models == ready.models
    assert failed.health == CONTRACT_HEALTH
    assert requests[-1] == "/health"
    assert requests.count("/health") == 2
    assert all(stream.close_count == 1 for stream in failure_streams)


@pytest.mark.asyncio
async def test_required_parser_failure_is_not_retried() -> None:
    requests: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        return _streaming_response(b'{"status":"not-ok"}')

    async with _adapter(respond) as adapter:
        catalog = await adapter.get_catalog()

    assert requests == ["/health"]
    assert catalog.health == CONTRACT_HEALTH


@pytest.mark.asyncio
async def test_bounded_reader_stops_at_first_oversized_chunk_and_closes() -> None:
    stream = TrackingStream(
        b"x" * MAX_METADATA_BYTES,
        b"y",
        b"UNREAD REMOTE SENTINEL",
    )

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    async with _adapter(respond) as adapter:
        catalog = await adapter.get_catalog()

    assert catalog.health == CONTRACT_HEALTH
    assert stream.read_count == 2
    assert stream.close_count == 1


@pytest.mark.asyncio
async def test_required_operation_has_one_deadline_for_health_models_and_retry() -> (
    None
):
    requests: list[str] = []
    blocked = asyncio.Event()

    async def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if request.url.path == "/health" and len(requests) == 1:
            await asyncio.sleep(0.07)
            raise httpx.ConnectError("REMOTE CONNECT", request=request)
        await blocked.wait()
        raise AssertionError("unreachable")

    started = monotonic()
    async with _adapter(respond, connect_timeout_seconds=0.1) as adapter:
        catalog = await adapter.get_catalog()
    elapsed = monotonic() - started

    assert requests == ["/health", "/health"]
    assert elapsed < 0.15
    assert catalog.health == TRANSIENT_HEALTH


@pytest.mark.asyncio
async def test_required_cancellation_closes_response_and_leaves_state_untouched() -> (
    None
):
    phase = "ready"
    blocking_stream = BlockingStream(_health_body())
    voice_requests = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal voice_requests
        if request.url.path == "/health":
            if phase == "blocked":
                return httpx.Response(200, stream=blocking_stream)
            return _streaming_response(_health_body())
        if request.url.path == "/v1/models":
            return _streaming_response(_models_body(_model()))
        voice_requests += 1
        return _streaming_response(_voices_body("cached"))

    async with _adapter(respond) as adapter:
        ready = await adapter.get_catalog()
        assert await adapter.get_voices("model") == ("cached",)
        phase = "blocked"
        refresh = asyncio.create_task(adapter.get_catalog(refresh=True))
        await blocking_stream.started.wait()
        refresh.cancel()

        with pytest.raises(asyncio.CancelledError):
            await refresh

        assert await adapter.get_catalog() is ready
        assert await adapter.get_voices("model") == ("cached",)

    assert blocking_stream.close_count == 1
    assert voice_requests == 1


@pytest.mark.asyncio
async def test_known_model_voices_are_lazy_query_encoded_cached_and_refreshable() -> (
    None
):
    model_id = "voice model/+?"
    requests: list[httpx.Request] = []
    voice_attempts = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal voice_attempts
        requests.append(request)
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        if request.url.path == "/v1/models":
            return _streaming_response(_models_body(_model(model_id)))
        voice_attempts += 1
        return _streaming_response(
            _voices_body(f"voice-{voice_attempts}", "second"),
        )

    async with _adapter(respond) as adapter:
        await adapter.ensure_ready()
        assert all(request.url.path != "/v1/audio/voices" for request in requests)

        first = await adapter.get_voices(model_id)
        cached = await adapter.get_voices(model_id)
        refreshed = await adapter.get_voices(model_id, refresh=True)

    assert first == cached == ("voice-1", "second")
    assert refreshed == ("voice-2", "second")
    voice_requests = [
        request for request in requests if request.url.path == "/v1/audio/voices"
    ]
    assert len(voice_requests) == 2
    assert all(request.url.params["model"] == model_id for request in voice_requests)
    assert all(
        request.headers["accept-encoding"] == "identity" for request in voice_requests
    )


@pytest.mark.asyncio
async def test_unknown_and_removed_model_ids_never_fetch_voices() -> None:
    models = (_model("current"),)
    voice_requests = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal voice_requests
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        if request.url.path == "/v1/models":
            return _streaming_response(_models_body(*models))
        voice_requests += 1
        return _streaming_response(_voices_body("voice"))

    async with _adapter(respond) as adapter:
        assert await adapter.get_voices("unknown") == ()
        assert await adapter.get_voices("current") == ("voice",)
        models = (_model("replacement"),)
        await adapter.get_catalog(refresh=True)
        assert await adapter.get_voices("current") == ()

    assert voice_requests == 1


@pytest.mark.asyncio
async def test_catalog_revision_invalidates_even_unchanged_voice_cache() -> None:
    voice_attempts = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal voice_attempts
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        if request.url.path == "/v1/models":
            return _streaming_response(_models_body(_model()))
        voice_attempts += 1
        return _streaming_response(
            _voices_body(f"revision-voice-{voice_attempts}"),
        )

    async with _adapter(respond) as adapter:
        first_catalog = await adapter.get_catalog()
        first_voices = await adapter.get_voices("model")
        second_catalog = await adapter.get_catalog(refresh=True)
        second_voices = await adapter.get_voices("model")

    assert first_catalog.revision == 1
    assert second_catalog.revision == 2
    assert first_catalog.models == second_catalog.models
    assert first_voices == ("revision-voice-1",)
    assert second_voices == ("revision-voice-2",)


@pytest.mark.asyncio
async def test_concurrent_voice_fetches_for_one_model_revision_are_coalesced() -> None:
    voice_requests = 0
    voice_started = asyncio.Event()
    release_voice = asyncio.Event()

    async def respond(request: httpx.Request) -> httpx.Response:
        nonlocal voice_requests
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        if request.url.path == "/v1/models":
            return _streaming_response(_models_body(_model()))
        voice_requests += 1
        voice_started.set()
        await release_voice.wait()
        return _streaming_response(_voices_body("coalesced"))

    async with _adapter(respond) as adapter:
        first = asyncio.create_task(adapter.get_voices("model", refresh=True))
        await voice_started.wait()
        second = asyncio.create_task(adapter.get_voices("model", refresh=True))
        await asyncio.sleep(0)
        release_voice.set()
        results = await asyncio.gather(first, second)

    assert results == [("coalesced",), ("coalesced",)]
    assert voice_requests == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "expected_requests"),
    [
        ("not_found", 1),
        ("transient_status", 2),
        ("redirect", 1),
        ("compressed", 1),
        ("malformed_length", 1),
        ("oversized", 1),
        ("invalid", 1),
        ("transport", 2),
        ("timeout", 2),
    ],
)
async def test_optional_voice_failures_cache_empty_without_health_mutation(
    case: str,
    expected_requests: int,
) -> None:
    voice_requests = 0
    streams: list[TrackingStream] = []

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal voice_requests
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        if request.url.path == "/v1/models":
            return _streaming_response(_models_body(_model()))
        voice_requests += 1
        if case == "not_found":
            return httpx.Response(404, content=b"REMOTE MISSING")
        if case == "transient_status":
            return httpx.Response(503, content=b"REMOTE BUSY")
        if case == "redirect":
            return httpx.Response(
                307,
                headers={"Location": "https://voice-redirect.invalid/private"},
            )
        if case == "compressed":
            return _streaming_response(
                _voices_body("remote"),
                headers={"Content-Encoding": "br"},
            )
        if case == "malformed_length":
            stream = TrackingStream(_voices_body("remote"))
            streams.append(stream)
            return httpx.Response(
                200,
                headers={"Content-Length": "nope"},
                stream=stream,
            )
        if case == "oversized":
            return _streaming_response(b"x" * (MAX_METADATA_BYTES + 1))
        if case == "invalid":
            return _streaming_response(b'{"voices":["duplicate","duplicate"]}')
        if case == "transport":
            raise httpx.ConnectError("REMOTE VOICE CONNECT", request=request)
        raise httpx.ReadTimeout("REMOTE VOICE TIMEOUT", request=request)

    async with _adapter(respond) as adapter:
        before = await adapter.get_catalog()
        assert await adapter.get_voices("model") == ()
        assert await adapter.get_voices("model") == ()
        after = await adapter.get_catalog()

    assert after is before
    assert after.health == AVAILABLE_HEALTH
    assert after.revision == 1
    assert voice_requests == expected_requests
    assert all(stream.close_count == 1 for stream in streams)


@pytest.mark.asyncio
async def test_voice_cancellation_is_not_cached_and_closes_response() -> None:
    voice_requests = 0
    blocked_stream = BlockingStream(_voices_body("cancelled"))

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal voice_requests
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        if request.url.path == "/v1/models":
            return _streaming_response(_models_body(_model()))
        voice_requests += 1
        if voice_requests == 1:
            return httpx.Response(200, stream=blocked_stream)
        return _streaming_response(_voices_body("retried"))

    async with _adapter(respond) as adapter:
        first = asyncio.create_task(adapter.get_voices("model"))
        await blocked_stream.started.wait()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        assert await adapter.get_voices("model") == ("retried",)

    assert voice_requests == 2
    assert blocked_stream.close_count == 1


class CountingTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.close_count = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise AssertionError("Construction and close must not make a request")

    async def aclose(self) -> None:
        self.close_count += 1


@pytest.mark.asyncio
async def test_close_is_idempotent_closes_client_once_and_marks_health_closed() -> None:
    transport = CountingTransport()
    adapter = AudioCppAdapter(_config(), transport=transport)

    await adapter.close()
    await adapter.close()
    catalog = await adapter.get_catalog()

    assert transport.close_count == 1
    assert catalog.revision == 0
    assert catalog.health == CLOSED_HEALTH


@pytest.mark.asyncio
async def test_synthesize_is_explicitly_deferred_and_never_sends_post() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise AssertionError("Synthesis must not make an HTTP request in this slice")

    async with _adapter(respond) as adapter:
        with pytest.raises(NotImplementedError, match="synthesis"):
            await adapter.synthesize(
                TTSRequest(
                    provider_id="audio_cpp",
                    model_id="model",
                    text="text",
                    voice=None,
                    response_format="wav",
                )
            )

    assert requests == []


@pytest.mark.asyncio
async def test_diagnostics_and_logs_do_not_echo_remote_values(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinel_url = "http://adapter-url-sentinel.invalid:8181"
    remote_values = (
        sentinel_url,
        "REMOTE_BODY_SENTINEL",
        "REMOTE_MODEL_SENTINEL",
        "REMOTE_VOICE_SENTINEL",
    )

    def respond(request: httpx.Request) -> httpx.Response:
        logging.getLogger("httpcore").debug(
            "connect url=%s",
            request.url,
        )
        return _streaming_response(
            (
                b'{"status":"ok","backend":"REMOTE_MODEL_SENTINEL","models":1,'
                b'"private":"REMOTE_BODY_SENTINEL"}garbage'
            ),
        )

    caplog.set_level(logging.DEBUG)
    async with _adapter(respond, base_url=sentinel_url) as adapter:
        catalog = await adapter.get_catalog()
        voices = await adapter.get_voices("REMOTE_MODEL_SENTINEL")

    public_output = " ".join(
        (
            str(catalog.health),
            str(catalog.revision),
            str(catalog.models),
            str(voices),
            caplog.text,
        )
    )
    assert catalog.health == CONTRACT_HEALTH
    assert all(value not in public_output for value in remote_values)
