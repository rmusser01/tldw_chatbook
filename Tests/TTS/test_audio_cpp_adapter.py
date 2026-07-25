from __future__ import annotations

import asyncio
import json
import logging
import math
import struct
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from fractions import Fraction
from time import monotonic
from typing import Any

import httpx
import pytest

from tldw_chatbook.TTS.adapter_types import (
    ProviderHealth,
    TTSAudioResponse,
    TTSModelInfo,
    TTSOperationError,
    TTSProgress,
    TTSProviderDescriptor,
    TTSProviderSpec,
    TTSRequest,
)
from tldw_chatbook.TTS.adapter_registry import TTSAdapterRegistry
from tldw_chatbook.TTS.adapters import audio_cpp as audio_cpp_module
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


def _wav(
    *,
    channels: int = 1,
    sample_rate: int = 24_000,
    frames: int = 2,
) -> bytes:
    block_align = channels * 2
    data = b"\x00\x00" * channels * frames
    fmt = struct.pack(
        "<4sIHHIIHH",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        sample_rate * block_align,
        block_align,
        16,
    )
    data_chunk = struct.pack("<4sI", b"data", len(data)) + data
    payload = b"WAVE" + fmt + data_chunk
    return b"RIFF" + struct.pack("<I", len(payload)) + payload


def _speech_request(**updates: Any) -> TTSRequest:
    values: dict[str, Any] = {
        "provider_id": "audio_cpp",
        "model_id": "model",
        "text": " Preserve exact whitespace ",
        "voice": None,
        "response_format": "wav",
        "speed": 1.0,
        "options": {},
    }
    values.update(updates)
    return TTSRequest(**values)


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


@asynccontextmanager
async def _registered_adapter(
    handler: Callable[[httpx.Request], Any],
    **config_updates: Any,
) -> AsyncIterator[tuple[TTSAdapterRegistry, AudioCppAdapter]]:
    adapter = AudioCppAdapter(
        _config(**config_updates),
        transport=httpx.MockTransport(handler),
    )
    registry = TTSAdapterRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor(
                    provider_id="audio_cpp",
                    display_name="audio.cpp",
                    native=True,
                ),
                factory=lambda _config: adapter,
                initial_config={},
            ),
        ),
        aliases={},
    )
    try:
        yield registry, adapter
    finally:
        await registry.close()
        await registry.wait_closed()


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


def _exception_graph(error: BaseException) -> list[BaseException]:
    pending = [error]
    seen: set[int] = set()
    graph: list[BaseException] = []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        graph.append(current)
        for linked in (current.__context__, current.__cause__):
            if linked is not None:
                pending.append(linked)
    return graph


def _assert_operation_error(
    error: TTSOperationError,
    *,
    code: str,
    message: str,
    retryable: bool,
    recovery_action: str,
) -> None:
    assert error.code == code
    assert error.args == (message,)
    assert str(error) == message
    assert error.retryable is retryable
    assert error.recovery_action == recovery_action
    assert len(error.operation_id) == 32
    assert error.operation_id.isascii()
    assert error.operation_id.isalnum()
    assert _exception_graph(error) == [error]
    assert error.__context__ is None
    assert error.__cause__ is None


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
@pytest.mark.parametrize("refresh", [False, True])
async def test_registry_first_catalog_uses_one_authoritative_refresh(
    refresh: bool,
) -> None:
    requests: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        return _streaming_response(_models_body(_model()))

    async with _registered_adapter(respond) as (registry, _adapter_instance):
        catalog = await registry.get_catalog("audio_cpp", refresh=refresh)

    assert catalog.revision == 1
    assert requests == ["/health", "/v1/models"]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["catalog", "voices"])
async def test_registry_first_failed_discovery_uses_one_refresh_operation(
    operation: str,
) -> None:
    requests: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        raise httpx.ConnectError("REMOTE CONNECT", request=request)

    async with _registered_adapter(respond) as (registry, _adapter_instance):
        if operation == "catalog":
            catalog = await registry.get_catalog("audio_cpp")
            assert catalog.health == TRANSIENT_HEALTH
        else:
            assert await registry.get_voices("audio_cpp", "model") == ()

    assert requests == ["/health", "/health"]


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
async def test_bounded_reader_rejects_one_huge_chunk_before_accumulating_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase = "ready"
    huge_stream = TrackingStream(b"x" * (MAX_METADATA_BYTES * 4096))
    peak_accumulated = 0

    class GuardedBytearray(bytearray):
        def extend(self, value: Any) -> None:
            nonlocal peak_accumulated
            proposed_size = len(self) + len(value)
            peak_accumulated = max(peak_accumulated, proposed_size)
            if proposed_size > MAX_METADATA_BYTES:
                raise AssertionError("oversized chunk was copied")
            super().extend(value)

    monkeypatch.setattr(
        audio_cpp_module,
        "bytearray",
        GuardedBytearray,
        raising=False,
    )

    def respond(request: httpx.Request) -> httpx.Response:
        if phase == "oversized" and request.url.path == "/health":
            return httpx.Response(200, stream=huge_stream)
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        return _streaming_response(_models_body(_model("retained")))

    async with _adapter(respond) as adapter:
        ready = await adapter.get_catalog()
        phase = "oversized"
        failed = await adapter.get_catalog(refresh=True)

    assert failed.revision == ready.revision
    assert failed.models == ready.models
    assert failed.health == CONTRACT_HEALTH
    assert peak_accumulated <= MAX_METADATA_BYTES
    assert huge_stream.close_count == 1


@pytest.mark.asyncio
async def test_unreasonably_long_content_lengths_fail_safely_without_value_leakage(
    caplog: pytest.LogCaptureFixture,
) -> None:
    declared_length = "7" * 5000

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(
                200,
                headers={"Content-Length": declared_length},
                stream=TrackingStream(_health_body()),
            )
        raise AssertionError("models must not be requested after invalid length")

    caplog.set_level(logging.DEBUG)
    async with _adapter(respond) as adapter:
        catalog = await adapter.get_catalog()

    assert catalog.revision == 0
    assert catalog.models == ()
    assert catalog.health == CONTRACT_HEALTH
    assert declared_length not in caplog.text


@pytest.mark.asyncio
async def test_long_content_length_failures_have_value_independent_stale_health() -> (
    None
):
    phase = "ready"
    digit_count = 5000

    def respond(request: httpx.Request) -> httpx.Response:
        if phase == "failure" and request.url.path == "/health":
            return httpx.Response(
                200,
                headers={"Content-Length": "9" * digit_count},
                stream=TrackingStream(_health_body()),
            )
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        return _streaming_response(_models_body(_model("retained")))

    async with _adapter(respond) as adapter:
        ready = await adapter.get_catalog()
        phase = "failure"
        failures = []
        for digit_count in (5000, 5003):
            failures.append(await adapter.get_catalog(refresh=True))

    assert all(failure.revision == ready.revision for failure in failures)
    assert all(failure.models == ready.models for failure in failures)
    assert [failure.health for failure in failures] == [
        CONTRACT_HEALTH,
        CONTRACT_HEALTH,
    ]


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
        assert adapter._voice_cache == {}
        assert adapter._voice_cache_bytes == 0
        assert adapter._voice_generation == {}
        assert adapter._voice_locks == {}
        second_voices = await adapter.get_voices("model")

    assert first_catalog.revision == 1
    assert second_catalog.revision == 2
    assert first_catalog.models == second_catalog.models
    assert first_voices == ("revision-voice-1",)
    assert second_voices == ("revision-voice-2",)


@pytest.mark.asyncio
async def test_voice_cache_evicts_least_recent_model_at_entry_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audio_cpp_module, "_MAX_VOICE_CACHE_ENTRIES", 2)
    monkeypatch.setattr(audio_cpp_module, "_MAX_VOICE_CACHE_BYTES", 8 * 1024 * 1024)
    requests: dict[str, int] = {}

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _streaming_response(_health_body(3))
        if request.url.path == "/v1/models":
            return _streaming_response(
                _models_body(_model("a"), _model("b"), _model("c"))
            )
        model_id = request.url.params["model"]
        requests[model_id] = requests.get(model_id, 0) + 1
        return _streaming_response(_voices_body(f"voice-{model_id}"))

    async with _adapter(respond) as adapter:
        assert await adapter.get_voices("a") == ("voice-a",)
        assert await adapter.get_voices("b") == ("voice-b",)
        assert await adapter.get_voices("c") == ("voice-c",)
        assert await adapter.get_voices("a") == ("voice-a",)

    assert requests == {"a": 2, "b": 1, "c": 1}


@pytest.mark.asyncio
async def test_voice_cache_reads_touch_lru_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audio_cpp_module, "_MAX_VOICE_CACHE_ENTRIES", 2)
    monkeypatch.setattr(audio_cpp_module, "_MAX_VOICE_CACHE_BYTES", 8 * 1024 * 1024)
    requests: dict[str, int] = {}

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _streaming_response(_health_body(3))
        if request.url.path == "/v1/models":
            return _streaming_response(
                _models_body(_model("a"), _model("b"), _model("c"))
            )
        model_id = request.url.params["model"]
        requests[model_id] = requests.get(model_id, 0) + 1
        return _streaming_response(_voices_body(f"voice-{model_id}"))

    async with _adapter(respond) as adapter:
        await adapter.get_voices("a")
        await adapter.get_voices("b")
        await adapter.get_voices("a")
        await adapter.get_voices("c")
        await adapter.get_voices("a")
        await adapter.get_voices("b")

    assert requests == {"a": 1, "b": 2, "c": 1}


@pytest.mark.asyncio
async def test_voice_cache_evicts_by_aggregate_byte_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    one_entry_bytes = audio_cpp_module._estimate_voice_cache_entry_bytes(
        (1, "a"),
        ("voice-a",),
    )
    monkeypatch.setattr(audio_cpp_module, "_MAX_VOICE_CACHE_ENTRIES", 32)
    monkeypatch.setattr(audio_cpp_module, "_MAX_VOICE_CACHE_BYTES", one_entry_bytes)
    requests: dict[str, int] = {}

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _streaming_response(_health_body(2))
        if request.url.path == "/v1/models":
            return _streaming_response(_models_body(_model("a"), _model("b")))
        model_id = request.url.params["model"]
        requests[model_id] = requests.get(model_id, 0) + 1
        return _streaming_response(_voices_body(f"voice-{model_id}"))

    async with _adapter(respond) as adapter:
        await adapter.get_voices("a")
        await adapter.get_voices("b")
        await adapter.get_voices("a")

    assert requests == {"a": 2, "b": 1}


@pytest.mark.asyncio
async def test_oversized_single_voice_result_is_returned_but_not_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audio_cpp_module, "_MAX_VOICE_CACHE_ENTRIES", 32)
    monkeypatch.setattr(audio_cpp_module, "_MAX_VOICE_CACHE_BYTES", 1)
    voice_requests = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal voice_requests
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        if request.url.path == "/v1/models":
            return _streaming_response(_models_body(_model()))
        voice_requests += 1
        return _streaming_response(_voices_body("large-result"))

    async with _adapter(respond) as adapter:
        assert await adapter.get_voices("model") == ("large-result",)
        assert await adapter.get_voices("model") == ("large-result",)

    assert voice_requests == 2


@pytest.mark.asyncio
async def test_voice_cache_eviction_preserves_an_in_use_coalescing_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audio_cpp_module, "_MAX_VOICE_CACHE_ENTRIES", 1)
    monkeypatch.setattr(audio_cpp_module, "_MAX_VOICE_CACHE_BYTES", 8 * 1024 * 1024)
    refresh_started = asyncio.Event()
    release_refresh = asyncio.Event()
    a_requests = 0

    async def respond(request: httpx.Request) -> httpx.Response:
        nonlocal a_requests
        if request.url.path == "/health":
            return _streaming_response(_health_body(2))
        if request.url.path == "/v1/models":
            return _streaming_response(_models_body(_model("a"), _model("b")))
        model_id = request.url.params["model"]
        if model_id == "a":
            a_requests += 1
            if a_requests == 2:
                refresh_started.set()
                await release_refresh.wait()
                return _streaming_response(_voices_body("refreshed-a"))
        return _streaming_response(_voices_body(f"voice-{model_id}"))

    async with _adapter(respond) as adapter:
        assert await adapter.get_voices("a") == ("voice-a",)
        first_refresh = asyncio.create_task(adapter.get_voices("a", refresh=True))
        await refresh_started.wait()
        assert await adapter.get_voices("b") == ("voice-b",)
        second_refresh = asyncio.create_task(adapter.get_voices("a", refresh=True))
        await asyncio.sleep(0)
        release_refresh.set()
        results = await asyncio.gather(first_refresh, second_refresh)

    assert results == [("refreshed-a",), ("refreshed-a",)]
    assert a_requests == 2


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


class CountingMockTransport(httpx.MockTransport):
    def __init__(self, handler: Callable[[httpx.Request], Any]) -> None:
        super().__init__(handler)
        self.close_count = 0

    async def aclose(self) -> None:
        self.close_count += 1
        await super().aclose()


class BlockingCloseTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.close_count = 0
        self.close_started = asyncio.Event()
        self.allow_close = asyncio.Event()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise AssertionError("Construction and close must not make a request")

    async def aclose(self) -> None:
        self.close_count += 1
        self.close_started.set()
        await self.allow_close.wait()


@pytest.mark.asyncio
async def test_cancelled_close_during_refresh_retains_cleanup_to_completion() -> None:
    phase = "ready"
    refresh_stream = BlockingStream(_health_body())

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            if phase == "refresh":
                return httpx.Response(200, stream=refresh_stream)
            return _streaming_response(_health_body())
        if request.url.path == "/v1/models":
            return _streaming_response(_models_body(_model()))
        return _streaming_response(_voices_body("cached"))

    transport = CountingMockTransport(respond)
    adapter = AudioCppAdapter(_config(), transport=transport)
    privacy_filter = adapter._httpx_privacy_filter
    try:
        await adapter.ensure_ready()
        assert await adapter.get_voices("model") == ("cached",)
        phase = "refresh"
        refresh = asyncio.create_task(adapter.get_catalog(refresh=True))
        await refresh_stream.started.wait()

        close = asyncio.create_task(adapter.close())
        await asyncio.sleep(0)
        close.cancel("caller cancelled")
        await asyncio.sleep(0)
        close_returned_before_cleanup = close.done()

        refresh_stream.release.set()
        await refresh
        with pytest.raises(asyncio.CancelledError, match="caller cancelled"):
            await close

        await adapter.close()
        catalog = await adapter.get_catalog()

        assert close_returned_before_cleanup is False
        assert transport.close_count == 1
        assert adapter._client.is_closed is True
        assert catalog.health == CLOSED_HEALTH
        assert adapter._voice_cache == {}
        assert adapter._voice_cache_bytes == 0
        assert adapter._voice_generation == {}
        assert adapter._voice_locks == {}
        assert adapter._voice_lock_users == {}
        assert adapter._voice_shared_results == {}
        assert all(
            privacy_filter not in logging.getLogger(logger_name).filters
            for logger_name in audio_cpp_module._HTTP_LOGGER_NAMES
        )
    finally:
        if not adapter._client.is_closed:
            await adapter._client.aclose()
        for logger_name in audio_cpp_module._HTTP_LOGGER_NAMES:
            logger = logging.getLogger(logger_name)
            if privacy_filter in logger.filters:
                logger.removeFilter(privacy_filter)


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
async def test_concurrent_close_calls_join_one_client_cleanup() -> None:
    transport = BlockingCloseTransport()
    adapter = AudioCppAdapter(_config(), transport=transport)

    first = asyncio.create_task(adapter.close())
    await transport.close_started.wait()
    second = asyncio.create_task(adapter.close())
    try:
        await asyncio.sleep(0)

        assert first.done() is False
        assert second.done() is False
        assert transport.close_count == 1
    finally:
        transport.allow_close.set()
        await asyncio.gather(first, second)

    assert transport.close_count == 1
    assert adapter._client.is_closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["get", "post"])
async def test_close_retains_privacy_filter_until_active_request_scope_exits(
    operation: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinel_url = "http://active-request-sentinel.invalid:8181"
    active_log = "ACTIVE_HTTP_LOG_AFTER_CLOSE_SENTINEL"
    private_value = "ACTIVE_REMOTE_VALUE_AFTER_CLOSE_SENTINEL"
    outside_log = "OUTSIDE_HTTP_LOG_WHILE_ACTIVE_SENTINEL"
    request_active = asyncio.Event()
    release_request = asyncio.Event()

    async def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        if request.url.path == "/v1/models":
            return _streaming_response(_models_body(_model()))

        expected_path = "/v1/audio/voices" if operation == "get" else "/v1/audio/speech"
        assert request.url.path == expected_path
        request_active.set()
        await release_request.wait()
        logging.getLogger("httpx").info(
            "%s url=%s private=%s",
            active_log,
            request.url,
            private_value,
        )
        logging.getLogger("httpcore.http11").debug(
            "%s private=%s",
            active_log,
            private_value,
        )
        if operation == "get":
            raise httpx.ReadError(private_value, request=request)
        return httpx.Response(
            200,
            headers={"Content-Type": "audio/wav"},
            stream=TrackingStream(
                httpx.ReadError(private_value, request=request),
            ),
        )

    caplog.set_level(logging.DEBUG)
    adapter = AudioCppAdapter(
        _config(base_url=sentinel_url),
        transport=httpx.MockTransport(respond),
    )
    privacy_filter = adapter._httpx_privacy_filter
    request_task: asyncio.Task[Any] | None = None
    try:
        await adapter.ensure_ready()
        if operation == "post":
            request_task = asyncio.create_task(adapter.synthesize(_speech_request()))
        else:
            request_task = asyncio.create_task(adapter.get_voices("model"))
        await asyncio.wait_for(request_active.wait(), timeout=1)

        await asyncio.wait_for(adapter.close(), timeout=1)
        assert all(
            privacy_filter in logging.getLogger(logger_name).filters
            for logger_name in audio_cpp_module._HTTP_LOGGER_NAMES
        )
        logging.getLogger("httpcore.http11").debug(outside_log)

        release_request.set()
        if operation == "get":
            assert await request_task == ()
        else:
            with pytest.raises(TTSOperationError) as captured:
                await request_task
            _assert_operation_error(
                captured.value,
                code="connection_unavailable",
                message="The audio.cpp server is unavailable",
                retryable=False,
                recovery_action="check_server",
            )
        assert adapter._catalog.health == CLOSED_HEALTH
        assert all(
            privacy_filter not in logging.getLogger(logger_name).filters
            for logger_name in audio_cpp_module._HTTP_LOGGER_NAMES
        )
    finally:
        release_request.set()
        if request_task is not None and not request_task.done():
            request_task.cancel()
        if request_task is not None:
            await asyncio.gather(request_task, return_exceptions=True)
        await adapter.close()
        for logger_name in audio_cpp_module._HTTP_LOGGER_NAMES:
            logger = logging.getLogger(logger_name)
            if privacy_filter in logger.filters:
                logger.removeFilter(privacy_filter)

    assert outside_log in caplog.text
    assert active_log not in caplog.text
    assert sentinel_url not in caplog.text
    assert private_value not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "speech_request",
    [
        _speech_request(provider_id="Audio_CPP"),
        _speech_request(text=""),
        _speech_request(text=" \t\n"),
        _speech_request(text="x" * 129),
        _speech_request(response_format="WAV"),
        _speech_request(speed=True),
        _speech_request(speed=0),
        _speech_request(speed=1.0001),
        _speech_request(speed=math.inf),
        _speech_request(speed=math.nan),
        pytest.param(
            _speech_request(speed=10**10_000),
            id="huge-integer-speed",
        ),
        pytest.param(
            _speech_request(speed=Fraction(10**10_000, 1)),
            id="huge-rational-speed",
        ),
        _speech_request(options={"REMOTE_OPTION_SENTINEL": True}),
        _speech_request(voice=""),
        _speech_request(voice=" voice"),
        _speech_request(voice="voice "),
        _speech_request(voice="x" * 129),
        _speech_request(voice="unsafe\u0000voice"),
        _speech_request(voice="unsafe\u200bvoice"),
        _speech_request(voice="\ue000"),
        _speech_request(voice="\ud800"),
        _speech_request(voice="\u0378"),
    ],
)
async def test_synthesize_rejects_every_local_request_category_before_http(
    speech_request: TTSRequest,
) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise AssertionError("Local request validation must precede HTTP")

    async with _adapter(
        respond,
        max_input_characters=128,
        max_identifier_characters=128,
    ) as adapter:
        with pytest.raises(TTSOperationError) as captured:
            await adapter.synthesize(speech_request)

    assert requests == []
    _assert_operation_error(
        captured.value,
        code="request_invalid",
        message="The audio.cpp speech request is invalid",
        retryable=False,
        recovery_action="edit_request",
    )


@pytest.mark.asyncio
async def test_separate_synthesis_failures_use_distinct_private_operation_ids() -> None:
    sentinel_model = "OPERATION_MODEL_SENTINEL"
    sentinel_text = "OPERATION_TEXT_SENTINEL"
    sentinel_voice = "OPERATION_VOICE_SENTINEL"
    remote_messages = iter(("REMOTE_OPERATION_BODY_ONE", "REMOTE_OPERATION_BODY_TWO"))

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        if request.url.path == "/v1/models":
            return _streaming_response(_models_body(_model(sentinel_model)))
        return _streaming_response(
            _json_bytes(
                {
                    "error": {
                        "message": next(remote_messages),
                        "type": "server_busy",
                    }
                }
            ),
            status=503,
        )

    errors: list[TTSOperationError] = []
    async with _adapter(respond) as adapter:
        for _ in range(2):
            with pytest.raises(TTSOperationError) as captured:
                await adapter.synthesize(
                    _speech_request(
                        model_id=sentinel_model,
                        text=sentinel_text,
                        voice=sentinel_voice,
                    )
                )
            errors.append(captured.value)

    assert errors[0].operation_id != errors[1].operation_id
    public_output = " ".join(
        component
        for error in errors
        for component in (
            repr(error.args),
            str(error),
            error.operation_id,
            repr(_exception_graph(error)),
        )
    )
    assert all(
        value not in public_output
        for value in (
            sentinel_model,
            sentinel_text,
            sentinel_voice,
            "REMOTE_OPERATION_BODY_ONE",
            "REMOTE_OPERATION_BODY_TWO",
        )
    )


@pytest.mark.asyncio
async def test_oversized_text_is_rejected_before_whitespace_scanning() -> None:
    class OversizedText(str):
        def strip(self, chars: str | None = None) -> str:
            del chars
            raise AssertionError("Oversized text must not be scanned")

    async with _adapter(
        lambda _request: pytest.fail("Oversized text must not use HTTP"),
        max_input_characters=8,
    ) as adapter:
        with pytest.raises(TTSOperationError) as captured:
            await adapter.synthesize(
                _speech_request(text=OversizedText("x" * 9)),
            )

    _assert_operation_error(
        captured.value,
        code="request_invalid",
        message="The audio.cpp speech request is invalid",
        retryable=False,
        recovery_action="edit_request",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("voice", [None, "native-cached-voice"])
async def test_synthesize_posts_exact_payload_and_reports_complete_progress(
    voice: str | None,
) -> None:
    requests: list[httpx.Request] = []
    progress: list[TTSProgress] = []
    wav = _wav(channels=2, sample_rate=48_000, frames=3)

    async def report(item: TTSProgress) -> None:
        progress.append(item)

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        if request.url.path == "/v1/models":
            return _streaming_response(_models_body(_model()))
        assert request.url.path == "/v1/audio/speech"
        return _streaming_response(
            wav,
            headers={
                "Content-Type": " Audio/WAV ; codec=pcm ",
                "X-AudioCpp-Wall-Ms": "12.5",
                "X-AudioCpp-Audio-Duration-Ms": "125",
                "X-AudioCpp-RTF": "0.1",
                "X-Remote-Secret": "REMOTE_HEADER_SENTINEL",
            },
        )

    async with _adapter(respond) as adapter:
        response = await adapter.synthesize(
            _speech_request(voice=voice),
            report,
        )
        chunks = [chunk async for chunk in response.byte_stream]

        assert isinstance(response, TTSAudioResponse)
        assert response.provider_id == "audio_cpp"
        assert response.model_id == "model"
        assert response.audio_format == "wav"
        assert response.content_type == "audio/wav"
        assert response.sample_rate == 48_000
        assert chunks == [wav]
        assert response.metadata == {
            "adapter": "audio_cpp",
            "contract": "audio_cpp_http_v1",
            "delivery": "complete_wav",
            "channels": 2,
            "frame_count": 3,
            "data_size": 12,
            "wall_ms": 12.5,
            "audio_duration_ms": 125.0,
            "rtf": 0.1,
        }
        with pytest.raises(TypeError):
            response.metadata["adapter"] = "changed"  # type: ignore[index]
        await response.aclose()

    posts = [request for request in requests if request.url.path == "/v1/audio/speech"]
    assert len(posts) == 1
    assert posts[0].method == "POST"
    expected_payload: dict[str, Any] = {
        "model": "model",
        "input": " Preserve exact whitespace ",
        "response_format": "wav",
    }
    if voice is not None:
        expected_payload["voice"] = voice
    assert json.loads(posts[0].content) == expected_payload
    assert set(json.loads(posts[0].content)) == set(expected_payload)
    assert progress == [
        TTSProgress(status="Generating", fraction=None),
        TTSProgress(status="Complete", fraction=1.0),
    ]


@pytest.mark.asyncio
async def test_synthesize_accepts_exactly_one_rational_speed() -> None:
    posts = 0
    wav = _wav()

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        if request.url.path == "/v1/models":
            return _streaming_response(_models_body(_model()))
        posts += 1
        return _streaming_response(
            wav,
            headers={"Content-Type": "audio/wav"},
        )

    async with _adapter(respond) as adapter:
        response = await adapter.synthesize(
            _speech_request(speed=Fraction(1, 1)),
        )
        assert [chunk async for chunk in response.byte_stream] == [wav]
        await response.aclose()

    assert posts == 1


@pytest.mark.asyncio
async def test_missing_model_forces_one_refresh_then_returns_model_invalid() -> None:
    requests: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        if request.url.path == "/v1/models":
            return _streaming_response(_models_body(_model("other")))
        raise AssertionError("An unknown model must never be posted")

    async with _adapter(respond) as adapter:
        await adapter.ensure_ready()
        requests.clear()
        with pytest.raises(TTSOperationError) as captured:
            await adapter.synthesize(_speech_request())

    assert requests == ["/health", "/v1/models"]
    _assert_operation_error(
        captured.value,
        code="model_invalid",
        message="The requested audio.cpp model is unavailable",
        retryable=False,
        recovery_action="refresh_models",
    )


@pytest.mark.asyncio
async def test_zero_tts_models_returns_not_configured_without_post() -> None:
    requests: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        return _streaming_response(_models_body(_model(task="stt")))

    async with _adapter(respond) as adapter:
        with pytest.raises(TTSOperationError) as captured:
            await adapter.synthesize(_speech_request())

    assert requests == ["/health", "/v1/models"]
    _assert_operation_error(
        captured.value,
        code="not_configured",
        message="No audio.cpp TTS models are configured",
        retryable=False,
        recovery_action="configure_server",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content_type",
    ["audio/wav", "AUDIO/X-WAV; codec=pcm", "application/octet-stream"],
)
async def test_synthesize_accepts_only_documented_wav_media_types(
    content_type: str,
) -> None:
    wav = _wav()

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        if request.url.path == "/v1/models":
            return _streaming_response(_models_body(_model()))
        return _streaming_response(
            wav,
            headers={
                "Content-Type": content_type,
                "X-AudioCpp-Wall-Ms": "-1",
                "X-AudioCpp-Audio-Duration-Ms": "not-a-number",
                "X-AudioCpp-RTF": "1e9",
            },
        )

    async with _adapter(respond, max_response_bytes=len(wav)) as adapter:
        response = await adapter.synthesize(_speech_request())
        assert [chunk async for chunk in response.byte_stream] == [wav]
        assert "wall_ms" not in response.metadata
        assert "audio_duration_ms" not in response.metadata
        assert "rtf" not in response.metadata
        await response.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        [],
        [("Content-Type", "text/plain")],
        [("Content-Type", "binary/octet-stream")],
        [("Content-Type", "audio/wav, application/octet-stream")],
        [("Content-Type", "audio/wav"), ("Content-Type", "audio/x-wav")],
    ],
)
async def test_synthesize_rejects_missing_multiple_or_other_media_types(
    headers: list[tuple[str, str]],
) -> None:
    progress: list[TTSProgress] = []
    speech_stream = TrackingStream(_wav())

    async def report(item: TTSProgress) -> None:
        progress.append(item)

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        if request.url.path == "/v1/models":
            return _streaming_response(_models_body(_model()))
        return httpx.Response(200, headers=headers, stream=speech_stream)

    async with _adapter(respond) as adapter:
        with pytest.raises(TTSOperationError) as captured:
            await adapter.synthesize(_speech_request(), report)
        assert (await adapter.get_catalog()).health == AVAILABLE_HEALTH

    _assert_operation_error(
        captured.value,
        code="audio_response_invalid",
        message="audio.cpp returned invalid audio",
        retryable=False,
        recovery_action="check_server",
    )
    assert progress == [TTSProgress(status="Generating", fraction=None)]
    assert speech_stream.read_count == 0
    assert speech_stream.close_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"not-a-wave",
        _wav()[:-1],
        _wav().replace(b"RIFF", b"RIFX", 1),
        _wav().replace(b"\x01\x00\x01\x00", b"\x03\x00\x01\x00", 1),
    ],
)
async def test_synthesize_rejects_malformed_complete_wav_matrix(body: bytes) -> None:
    stream = TrackingStream(body)

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        if request.url.path == "/v1/models":
            return _streaming_response(_models_body(_model()))
        return httpx.Response(
            200,
            headers={"Content-Type": "audio/wav"},
            stream=stream,
        )

    async with _adapter(respond) as adapter:
        with pytest.raises(TTSOperationError) as captured:
            await adapter.synthesize(_speech_request())
        assert (await adapter.get_catalog()).health == AVAILABLE_HEALTH

    _assert_operation_error(
        captured.value,
        code="audio_response_invalid",
        message="audio.cpp returned invalid audio",
        retryable=False,
        recovery_action="check_server",
    )
    assert stream.close_count == 1


@pytest.mark.asyncio
async def test_preconsumed_http_200_body_is_invalid_audio_not_health_failure() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        if request.url.path == "/v1/models":
            return _streaming_response(_models_body(_model()))
        return httpx.Response(
            200,
            headers={"Content-Type": "audio/wav"},
            content=_wav(),
        )

    async with _adapter(respond) as adapter:
        with pytest.raises(TTSOperationError) as captured:
            await adapter.synthesize(_speech_request())
        catalog = adapter._catalog

    _assert_operation_error(
        captured.value,
        code="audio_response_invalid",
        message="audio.cpp returned invalid audio",
        retryable=False,
        recovery_action="check_server",
    )
    assert catalog.health == AVAILABLE_HEALTH


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("headers", "chunks", "max_response_bytes", "expected_reads"),
    [
        (
            {"Content-Type": "audio/wav", "Content-Length": "999"},
            (_wav(),),
            len(_wav()),
            0,
        ),
        (
            {
                "Content-Type": "audio/wav",
                "Content-Length": "9" * (len(str(sys.maxsize)) + 1),
            },
            (_wav(),),
            len(_wav()),
            0,
        ),
        (
            {
                "Content-Type": "audio/wav",
                "Content-Length": str(len(_wav()) + 1),
            },
            (_wav(),),
            len(_wav()) + 1,
            1,
        ),
        (
            {"Content-Type": "audio/wav", "Content-Encoding": "gzip"},
            (_wav(),),
            len(_wav()),
            0,
        ),
        (
            {"Content-Type": "audio/wav"},
            (_wav()[:-1], b"x"),
            len(_wav()) - 1,
            2,
        ),
        (
            {"Content-Type": "audio/wav"},
            (_wav(),),
            len(_wav()) - 1,
            1,
        ),
    ],
)
async def test_synthesize_enforces_audio_headers_and_body_bound_before_copy(
    headers: dict[str, str],
    chunks: tuple[bytes, ...],
    max_response_bytes: int,
    expected_reads: int,
) -> None:
    stream = TrackingStream(*chunks)

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        if request.url.path == "/v1/models":
            return _streaming_response(_models_body(_model()))
        return httpx.Response(200, headers=headers, stream=stream)

    async with _adapter(
        respond,
        max_response_bytes=max_response_bytes,
    ) as adapter:
        with pytest.raises(TTSOperationError) as captured:
            await adapter.synthesize(_speech_request())
        assert (await adapter.get_catalog()).health == AVAILABLE_HEALTH

    _assert_operation_error(
        captured.value,
        code="audio_response_invalid",
        message="audio.cpp returned invalid audio",
        retryable=False,
        recovery_action="check_server",
    )
    assert stream.read_count == expected_reads
    assert stream.close_count == 1


@pytest.mark.asyncio
async def test_synthesize_joins_incremental_wav_once_and_returns_one_chunk() -> None:
    wav = _wav(channels=2, sample_rate=44_100, frames=4)
    stream = TrackingStream(wav[:13], wav[13:37], wav[37:])

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        if request.url.path == "/v1/models":
            return _streaming_response(_models_body(_model()))
        return httpx.Response(
            200,
            headers={
                "Content-Type": "audio/x-wav",
                "Content-Length": str(len(wav)),
            },
            stream=stream,
        )

    async with _adapter(respond, max_response_bytes=len(wav)) as adapter:
        response = await adapter.synthesize(_speech_request())
        assert stream.close_count == 1
        chunks = [chunk async for chunk in response.byte_stream]
        assert chunks == [wav]
        assert response.sample_rate == 44_100
        assert response.metadata["channels"] == 2
        assert response.metadata["frame_count"] == 4
        assert response.metadata["data_size"] == 16
        await response.aclose()

    assert stream.read_count == 3
    assert stream.close_count == 1


@pytest.mark.asyncio
async def test_synthesize_ignores_empty_tiny_chunks_and_preserves_exact_wav() -> None:
    wav = _wav(channels=2, sample_rate=44_100, frames=4)
    chunks = tuple(chunk for byte in wav for chunk in (b"", bytes((byte,))))
    stream = TrackingStream(*chunks)

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        if request.url.path == "/v1/models":
            return _streaming_response(_models_body(_model()))
        return httpx.Response(
            200,
            headers={
                "Content-Type": "audio/wav",
                "Content-Length": str(len(wav)),
            },
            stream=stream,
        )

    async with _adapter(respond, max_response_bytes=len(wav)) as adapter:
        response = await adapter.synthesize(_speech_request())
        assert [chunk async for chunk in response.byte_stream] == [wav]
        await response.aclose()

    assert stream.read_count == len(chunks)
    assert stream.close_count == 1


@pytest.mark.asyncio
async def test_synthesize_stops_on_first_tiny_chunk_beyond_body_bound() -> None:
    wav = _wav()
    max_response_bytes = len(wav) - 1
    chunks = tuple(chunk for byte in wav for chunk in (b"", bytes((byte,)))) + (
        b"UNREAD_REMOTE_BODY_SENTINEL",
    )
    stream = TrackingStream(*chunks)

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        if request.url.path == "/v1/models":
            return _streaming_response(_models_body(_model()))
        return httpx.Response(
            200,
            headers={"Content-Type": "audio/wav"},
            stream=stream,
        )

    async with _adapter(
        respond,
        max_response_bytes=max_response_bytes,
    ) as adapter:
        with pytest.raises(TTSOperationError) as captured:
            await adapter.synthesize(_speech_request())

    _assert_operation_error(
        captured.value,
        code="audio_response_invalid",
        message="audio.cpp returned invalid audio",
        retryable=False,
        recovery_action="check_server",
    )
    assert stream.read_count == (max_response_bytes * 2) + 2
    assert stream.close_count == 1


@pytest.mark.asyncio
async def test_structured_server_busy_is_retryable_without_staling_catalog() -> None:
    remote_message = "REMOTE_BUSY_MESSAGE_SENTINEL"
    requests: list[str] = []
    stream = TrackingStream(
        _json_bytes(
            {
                "error": {
                    "message": remote_message,
                    "type": "server_busy",
                }
            }
        )
    )

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        if request.url.path == "/v1/models":
            return _streaming_response(_models_body(_model()))
        return httpx.Response(503, stream=stream)

    async with _adapter(respond) as adapter:
        with pytest.raises(TTSOperationError) as captured:
            await adapter.synthesize(_speech_request())
        catalog = adapter._catalog

    _assert_operation_error(
        captured.value,
        code="server_busy",
        message="The audio.cpp server is busy",
        retryable=True,
        recovery_action="retry",
    )
    assert remote_message not in repr(captured.value.args)
    assert catalog.health == AVAILABLE_HEALTH
    assert requests.count("/v1/audio/speech") == 1
    assert stream.close_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "headers", "max_metadata_bytes"),
    [
        (b"{", {}, MAX_METADATA_BYTES),
        (
            _json_bytes(
                {"error": {"message": "REMOTE_SENTINEL", "type": "server_error"}}
            ),
            {},
            MAX_METADATA_BYTES,
        ),
        (
            b'{"error":{"message":"REMOTE_SENTINEL","type":"server_busy"},'
            b'"extreme":' + b"7" * 129 + b"}",
            {},
            MAX_METADATA_BYTES,
        ),
        (b"x" * (MAX_METADATA_BYTES + 1), {}, MAX_METADATA_BYTES),
        (
            _json_bytes(
                {"error": {"message": "REMOTE_SENTINEL", "type": "server_busy"}}
            ),
            {"Content-Encoding": "gzip"},
            MAX_METADATA_BYTES,
        ),
    ],
)
async def test_invalid_503_envelopes_map_to_transient_unavailable(
    body: bytes,
    headers: dict[str, str],
    max_metadata_bytes: int,
) -> None:
    posts = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        if request.url.path == "/v1/models":
            return _streaming_response(_models_body(_model()))
        posts += 1
        return _streaming_response(body, status=503, headers=headers)

    async with _adapter(
        respond,
        max_metadata_bytes=max_metadata_bytes,
    ) as adapter:
        with pytest.raises(TTSOperationError) as captured:
            await adapter.synthesize(_speech_request())
        catalog = adapter._catalog

    _assert_operation_error(
        captured.value,
        code="connection_unavailable",
        message="The audio.cpp server is unavailable",
        retryable=True,
        recovery_action="retry",
    )
    assert "REMOTE_SENTINEL" not in repr(captured.value.args)
    assert catalog.health == TRANSIENT_HEALTH
    assert catalog.models
    assert posts == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [408, 429, 502, 504])
async def test_transient_speech_statuses_stale_health_without_post_retry(
    status: int,
) -> None:
    posts = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        if request.url.path == "/v1/models":
            return _streaming_response(_models_body(_model()))
        posts += 1
        return _streaming_response(b"REMOTE_STATUS_BODY", status=status)

    async with _adapter(respond) as adapter:
        with pytest.raises(TTSOperationError) as captured:
            await adapter.synthesize(_speech_request())
        catalog = adapter._catalog

    _assert_operation_error(
        captured.value,
        code="connection_unavailable",
        message="The audio.cpp server is unavailable",
        retryable=True,
        recovery_action="retry",
    )
    assert catalog.health == TRANSIENT_HEALTH
    assert posts == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [301, 302, 307, 308, 404])
async def test_missing_or_redirected_speech_endpoint_is_incompatible(
    status: int,
) -> None:
    posts = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        if request.url.path == "/v1/models":
            return _streaming_response(_models_body(_model()))
        posts += 1
        return httpx.Response(
            status,
            headers={"Location": "https://REMOTE_REDIRECT_SENTINEL.invalid/"},
            content=b"REMOTE_CONTRACT_BODY",
        )

    async with _adapter(respond) as adapter:
        with pytest.raises(TTSOperationError) as captured:
            await adapter.synthesize(_speech_request())
        catalog = adapter._catalog

    _assert_operation_error(
        captured.value,
        code="contract_incompatible",
        message="The audio.cpp server response is incompatible",
        retryable=False,
        recovery_action="check_server",
    )
    assert catalog.health == CONTRACT_HEALTH
    assert posts == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 403, 405, 422, 425, 501])
async def test_unexpected_speech_errors_are_generation_failures(
    status: int,
) -> None:
    posts = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        if request.url.path == "/v1/models":
            return _streaming_response(_models_body(_model()))
        posts += 1
        return _streaming_response(b"REMOTE_GENERATION_BODY", status=status)

    async with _adapter(respond) as adapter:
        with pytest.raises(TTSOperationError) as captured:
            await adapter.synthesize(_speech_request())
        catalog = adapter._catalog

    _assert_operation_error(
        captured.value,
        code="generation_failed",
        message="audio.cpp could not generate speech",
        retryable=False,
        recovery_action="check_server",
    )
    assert catalog.health == AVAILABLE_HEALTH
    assert posts == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("refresh_result", "expected_code", "expected_message", "expected_recovery"),
    [
        (
            "model_remains",
            "generation_failed",
            "audio.cpp could not generate speech",
            "check_server",
        ),
        (
            "model_vanished",
            "model_invalid",
            "The requested audio.cpp model is unavailable",
            "refresh_models",
        ),
        (
            "no_models",
            "model_invalid",
            "The requested audio.cpp model is unavailable",
            "refresh_models",
        ),
        (
            "unavailable",
            "connection_unavailable",
            "The audio.cpp server is unavailable",
            "retry",
        ),
        (
            "incompatible",
            "contract_incompatible",
            "The audio.cpp server response is incompatible",
            "check_server",
        ),
    ],
)
async def test_speech_500_refreshes_models_once_without_post_retry(
    refresh_result: str,
    expected_code: str,
    expected_message: str,
    expected_recovery: str,
) -> None:
    posts = 0
    model_reads = 0
    post_seen = False
    paths: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal model_reads, post_seen, posts
        paths.append(request.url.path)
        if request.url.path == "/v1/audio/speech":
            posts += 1
            post_seen = True
            return _streaming_response(b"REMOTE_500_BODY", status=500)
        if request.url.path == "/health":
            if post_seen and refresh_result == "unavailable":
                raise httpx.ConnectError("REMOTE_REFRESH_TRANSPORT", request=request)
            if post_seen and refresh_result == "incompatible":
                return _streaming_response(b'{"status":"REMOTE_BAD"}')
            return _streaming_response(_health_body())

        model_reads += 1
        if not post_seen or refresh_result == "model_remains":
            return _streaming_response(_models_body(_model()))
        if refresh_result == "model_vanished":
            return _streaming_response(_models_body(_model("other")))
        if refresh_result == "no_models":
            return _streaming_response(_models_body(_model(task="stt")))
        raise AssertionError("Failed refresh must not request models")

    async with _adapter(respond) as adapter:
        with pytest.raises(TTSOperationError) as captured:
            await adapter.synthesize(_speech_request())

    _assert_operation_error(
        captured.value,
        code=expected_code,
        message=expected_message,
        retryable=expected_code == "connection_unavailable",
        recovery_action=expected_recovery,
    )
    assert posts == 1
    assert model_reads == (
        2 if refresh_result in {"model_remains", "model_vanished", "no_models"} else 1
    )
    assert paths.count("/v1/audio/speech") == 1


@pytest.mark.asyncio
async def test_speech_transport_failure_stales_health_without_retry() -> None:
    posts = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        if request.url.path == "/v1/models":
            return _streaming_response(_models_body(_model()))
        posts += 1
        raise httpx.ConnectError("REMOTE_TRANSPORT_SENTINEL", request=request)

    async with _adapter(respond) as adapter:
        with pytest.raises(TTSOperationError) as captured:
            await adapter.synthesize(_speech_request())
        catalog = adapter._catalog

    _assert_operation_error(
        captured.value,
        code="connection_unavailable",
        message="The audio.cpp server is unavailable",
        retryable=True,
        recovery_action="retry",
    )
    assert catalog.health == TRANSIENT_HEALTH
    assert posts == 1


@pytest.mark.asyncio
async def test_speech_body_transport_failure_closes_and_stales_without_retry() -> None:
    transport_sentinel = "REMOTE_BODY_TRANSPORT_SENTINEL"
    posts = 0
    streams: list[TrackingStream] = []

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        if request.url.path == "/v1/models":
            return _streaming_response(_models_body(_model()))
        posts += 1
        stream = TrackingStream(
            _wav()[:16],
            httpx.ReadError(transport_sentinel, request=request),
            b"UNREAD_REMOTE_BODY_SENTINEL",
        )
        streams.append(stream)
        return httpx.Response(
            200,
            headers={"Content-Type": "audio/wav"},
            stream=stream,
        )

    async with _adapter(respond) as adapter:
        with pytest.raises(TTSOperationError) as captured:
            await adapter.synthesize(_speech_request())
        catalog = adapter._catalog

    _assert_operation_error(
        captured.value,
        code="connection_unavailable",
        message="The audio.cpp server is unavailable",
        retryable=True,
        recovery_action="retry",
    )
    assert catalog.health == TRANSIENT_HEALTH
    assert catalog.revision == 1
    assert catalog.models
    assert posts == 1
    assert len(streams) == 1
    assert streams[0].read_count == 2
    assert streams[0].close_count == 1
    assert transport_sentinel not in repr(captured.value.args)
    assert transport_sentinel not in repr(_exception_graph(captured.value))


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["headers", "body"])
async def test_speech_total_deadline_covers_headers_and_body_without_staling(
    phase: str,
) -> None:
    posts = 0
    blocked_headers = asyncio.Event()
    speech_stream = BlockingStream(_wav())

    async def respond(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        if request.url.path == "/v1/models":
            return _streaming_response(_models_body(_model()))
        posts += 1
        if phase == "headers":
            blocked_headers.set()
            await asyncio.Event().wait()
        return httpx.Response(
            200,
            headers={"Content-Type": "audio/wav"},
            stream=speech_stream,
        )

    async with _adapter(
        respond,
        synthesis_timeout_seconds=0.01,
    ) as adapter:
        with pytest.raises(TTSOperationError) as captured:
            await adapter.synthesize(_speech_request())
        catalog = adapter._catalog

    _assert_operation_error(
        captured.value,
        code="generation_timeout",
        message="audio.cpp speech generation timed out",
        retryable=True,
        recovery_action="retry",
    )
    assert catalog.health == AVAILABLE_HEALTH
    assert posts == 1
    if phase == "headers":
        assert blocked_headers.is_set()
    else:
        assert speech_stream.started.is_set()
        assert speech_stream.close_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["headers", "body"])
async def test_speech_cancellation_propagates_and_closes_without_health_poisoning(
    phase: str,
) -> None:
    posts = 0
    headers_started = asyncio.Event()
    release_headers = asyncio.Event()
    stream = BlockingStream(_wav())

    async def respond(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        if request.url.path == "/v1/models":
            return _streaming_response(_models_body(_model()))
        posts += 1
        if phase == "headers":
            headers_started.set()
            await release_headers.wait()
        return httpx.Response(
            200,
            headers={"Content-Type": "audio/wav"},
            stream=stream,
        )

    async with _adapter(respond) as adapter:
        generation = asyncio.create_task(adapter.synthesize(_speech_request()))
        if phase == "headers":
            await headers_started.wait()
        else:
            await stream.started.wait()
        generation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await generation
        catalog = adapter._catalog

    assert catalog.health == AVAILABLE_HEALTH
    assert posts == 1
    if phase == "body":
        assert stream.close_count == 1


@pytest.mark.asyncio
async def test_failed_readiness_maps_retryability_and_never_posts() -> None:
    mode = "transient"
    posts = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if request.url.path == "/v1/audio/speech":
            posts += 1
        if mode == "transient":
            raise httpx.ConnectError("REMOTE_READY_SENTINEL", request=request)
        return _streaming_response(b"{}")

    async with _adapter(respond) as adapter:
        with pytest.raises(TTSOperationError) as transient:
            await adapter.synthesize(_speech_request())
        mode = "contract"
        with pytest.raises(TTSOperationError) as contract:
            await adapter.synthesize(_speech_request())

    _assert_operation_error(
        transient.value,
        code="connection_unavailable",
        message="The audio.cpp server is unavailable",
        retryable=True,
        recovery_action="retry",
    )
    _assert_operation_error(
        contract.value,
        code="contract_incompatible",
        message="The audio.cpp server response is incompatible",
        retryable=False,
        recovery_action="check_server",
    )
    assert posts == 0


@pytest.mark.asyncio
async def test_closed_adapter_maps_to_nonretryable_connection_error() -> None:
    adapter = AudioCppAdapter(
        _config(),
        transport=httpx.MockTransport(
            lambda _request: pytest.fail("Closed synthesis must not use HTTP")
        ),
    )
    await adapter.close()

    with pytest.raises(TTSOperationError) as captured:
        await adapter.synthesize(_speech_request())

    _assert_operation_error(
        captured.value,
        code="connection_unavailable",
        message="The audio.cpp server is unavailable",
        retryable=False,
        recovery_action="check_server",
    )


@pytest.mark.asyncio
async def test_close_from_progress_before_post_maps_to_closed_without_http() -> None:
    posts = 0
    progress: list[TTSProgress] = []

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        if request.url.path == "/v1/models":
            return _streaming_response(_models_body(_model()))
        posts += 1
        raise AssertionError("Close before POST must seal speech admission")

    adapter = AudioCppAdapter(
        _config(),
        transport=httpx.MockTransport(respond),
    )

    async def close_on_generating(item: TTSProgress) -> None:
        progress.append(item)
        await adapter.close()

    try:
        with pytest.raises(TTSOperationError) as captured:
            await adapter.synthesize(
                _speech_request(),
                close_on_generating,
            )
    finally:
        await adapter.close()

    _assert_operation_error(
        captured.value,
        code="connection_unavailable",
        message="The audio.cpp server is unavailable",
        retryable=False,
        recovery_action="check_server",
    )
    assert progress == [TTSProgress(status="Generating", fraction=None)]
    assert posts == 0
    assert adapter._catalog.health == CLOSED_HEALTH


@pytest.mark.asyncio
async def test_synthesis_error_and_http_logs_never_retain_request_or_remote_values(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinel_url = "http://synthesis-origin-sentinel.invalid:8181"
    sentinel_text = "SYNTHESIS_TEXT_SENTINEL"
    sentinel_model = "SYNTHESIS_MODEL_SENTINEL"
    sentinel_voice = "SYNTHESIS_VOICE_SENTINEL"
    sentinel_body = "SYNTHESIS_BODY_SENTINEL"
    sentinel_header = "SYNTHESIS_HEADER_SENTINEL"
    sentinel_cookie = "SYNTHESIS_COOKIE_SENTINEL"
    sentinel_reason = "SYNTHESIS_REASON_SENTINEL"
    outside_log = "OUTSIDE_SYNTHESIS_LOG_SENTINEL"
    request_active = asyncio.Event()
    release_request = asyncio.Event()

    async def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return _streaming_response(_health_body())
        if request.url.path == "/v1/models":
            return _streaming_response(_models_body(_model(sentinel_model)))
        logging.getLogger("httpx").info(
            "HTTP Request: POST %s body=%s model=%s voice=%s",
            sentinel_url,
            sentinel_body,
            sentinel_model,
            sentinel_voice,
        )
        logging.getLogger("httpcore.http11").debug(
            "receive_response_headers.complete headers=%r reason=%s",
            (
                (b"x-private", sentinel_header.encode()),
                (b"set-cookie", sentinel_cookie.encode()),
            ),
            sentinel_reason,
        )
        request_active.set()
        await release_request.wait()
        return httpx.Response(
            503,
            headers={
                "Set-Cookie": f"remote={sentinel_cookie}; Path=/",
                "X-Private": sentinel_header,
            },
            extensions={"reason_phrase": sentinel_reason.encode("ascii")},
            stream=TrackingStream(
                b'{"error":{"message":"'
                + sentinel_body.encode()
                + b'","type":"server_error"}}'
            ),
        )

    caplog.set_level(logging.DEBUG)
    async with _adapter(respond, base_url=sentinel_url) as adapter:
        generation = asyncio.create_task(
            adapter.synthesize(
                _speech_request(
                    model_id=sentinel_model,
                    text=sentinel_text,
                    voice=sentinel_voice,
                )
            )
        )
        await request_active.wait()
        logging.getLogger("httpcore.http11").debug(outside_log)
        release_request.set()
        with pytest.raises(TTSOperationError) as captured:
            await generation
        assert len(adapter._client.cookies) == 0

    _assert_operation_error(
        captured.value,
        code="connection_unavailable",
        message="The audio.cpp server is unavailable",
        retryable=True,
        recovery_action="retry",
    )
    public_output = " ".join(
        (
            repr(captured.value.args),
            str(captured.value),
            repr(_exception_graph(captured.value)),
            caplog.text,
        )
    )
    assert all(
        value not in public_output
        for value in (
            sentinel_url,
            sentinel_text,
            sentinel_model,
            sentinel_voice,
            sentinel_body,
            sentinel_header,
            sentinel_cookie,
            sentinel_reason,
        )
    )
    assert outside_log in caplog.text


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
        "REMOTE_RESPONSE_HEADER_SENTINEL",
        "REMOTE_COOKIE_SENTINEL",
    )
    request_active = asyncio.Event()
    release_request = asyncio.Event()

    async def respond(request: httpx.Request) -> httpx.Response:
        logging.getLogger("httpcore.http11").debug(
            "receive_response_headers.complete headers=%r",
            (
                (b"x-model", b"REMOTE_RESPONSE_HEADER_SENTINEL"),
                (b"set-cookie", b"REMOTE_COOKIE_SENTINEL"),
            ),
        )
        request_active.set()
        await release_request.wait()
        return _streaming_response(
            (
                b'{"status":"ok","backend":"REMOTE_MODEL_SENTINEL","models":1,'
                b'"private":"REMOTE_BODY_SENTINEL"}garbage'
            ),
        )

    caplog.set_level(logging.DEBUG)
    async with _adapter(respond, base_url=sentinel_url) as adapter:
        request = asyncio.create_task(adapter.get_catalog())
        await request_active.wait()
        logging.getLogger("httpcore.http11").debug(
            "OUTSIDE_UNRELATED_HTTP_LOG_SENTINEL"
        )
        release_request.set()
        catalog = await request

    public_output = " ".join(
        (
            str(catalog.health),
            str(catalog.revision),
            str(catalog.models),
            caplog.text,
        )
    )
    assert catalog.health == CONTRACT_HEALTH
    assert all(value not in public_output for value in remote_values)
    assert "OUTSIDE_UNRELATED_HTTP_LOG_SENTINEL" in caplog.text
