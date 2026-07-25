from __future__ import annotations

import asyncio
import stat
from uuid import UUID
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from loguru import logger
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Select, Static, TextArea

from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSEventHandler,
    STTSPlaygroundGenerateEvent,
)
from tldw_chatbook.TTS import (
    STTSGeneratedAudio,
    STTSPlaygroundRequest,
    ProviderHealth,
    TTSOperationError,
    TTSModelInfo,
    TTSProviderCatalog,
    TTSProviderDescriptor,
    TTSRequest,
)
from tldw_chatbook.TTS.adapter_types import (
    TTSProviderReconfiguringError,
    TTSRegistryClosedError,
)
from tldw_chatbook.UI import STTS_Window
from tldw_chatbook.UI.STTS_Window import TTSPlaygroundWidget


def _snapshot(
    *,
    provider_id: str = "audio_cpp",
    response_format: str = "wav",
    options: Mapping[str, Any] | None = None,
    model_id: str = "model-1",
    text: str = "private source text",
) -> STTSPlaygroundRequest:
    return STTSPlaygroundRequest(
        operation_id="local-operation-1",
        provider_id=provider_id,
        model_id=model_id,
        text=text,
        voice_id=None if provider_id == "audio_cpp" else "alloy",
        response_format=response_format,
        speed=1.0,
        options=options or {},
    )


class _CountingStream:
    def __init__(
        self,
        chunks: tuple[bytes, ...],
        *,
        failure: BaseException | None = None,
        blocked: asyncio.Event | None = None,
    ) -> None:
        self.chunks = chunks
        self.failure = failure
        self.blocked = blocked
        self.iterations = 0
        self.close_calls = 0

    def __aiter__(self) -> AsyncIterator[bytes]:
        self.iterations += 1
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk
        if self.blocked is not None:
            await self.blocked.wait()
        if self.failure is not None:
            raise self.failure

    async def aclose(self) -> None:
        self.close_calls += 1


class _Response:
    def __init__(
        self,
        stream: _CountingStream,
        *,
        provider_id: str = "audio_cpp",
        model_id: str = "server-model",
        audio_format: str = "wav",
        content_type: str = "audio/wav",
        metadata: Mapping[str, str | int | float | bool | None] | None = None,
    ) -> None:
        self.provider_id = provider_id
        self.model_id = model_id
        self.audio_format = audio_format
        self.content_type = content_type
        self.byte_stream = stream
        self.metadata = metadata or {}
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1
        await self.byte_stream.aclose()


class _NativeService:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.requests: list[TTSRequest] = []
        self.legacy_calls = 0

    async def synthesize(
        self,
        request: TTSRequest,
        progress_sink: object = None,
    ) -> _Response:
        del progress_sink
        self.requests.append(request)
        return self.response

    async def generate_audio_stream(
        self,
        *_args: object,
        **_kwargs: object,
    ) -> AsyncIterator[bytes]:
        self.legacy_calls += 1
        raise AssertionError("audio.cpp must not use the legacy stream bridge")
        yield b""  # pragma: no cover


class _LegacyService:
    def __init__(self) -> None:
        self.stream_calls: list[tuple[object, str, object]] = []
        self.native_calls = 0

    async def synthesize(self, *_args: object, **_kwargs: object) -> None:
        self.native_calls += 1
        raise AssertionError("legacy generation must retain the bridge")

    async def generate_audio_stream(
        self,
        request: object,
        internal_model_id: str,
        progress_sink: object = None,
    ) -> AsyncIterator[bytes]:
        self.stream_calls.append((request, internal_model_id, progress_sink))
        yield b"RIFF"
        yield b"legacy"


class _DeliveryPlayground:
    def __init__(self) -> None:
        self.completions: list[STTSGeneratedAudio | None] = []
        self.log = SimpleNamespace(write=Mock())
        self.progress = SimpleNamespace(update=Mock())
        self.container = SimpleNamespace(remove_class=Mock(), add_class=Mock())
        self.status = SimpleNamespace(update=Mock())
        self.button = SimpleNamespace(disabled=True)

    def query_one(self, selector: str, _widget_type: object = None) -> object:
        return {
            "#generation-status-container": self.container,
            "#generation-progress": self.progress,
            "#generation-status-text": self.status,
            "#tts-generation-log": self.log,
            "#tts-generate-btn": self.button,
        }[selector]

    def call_from_thread(self, callback: object, *args: object) -> None:
        assert callable(callback)
        callback(*args)

    def _generation_complete(
        self,
        artifact: STTSGeneratedAudio | None,
    ) -> None:
        self.completions.append(artifact)


class _DeliveryApp:
    def __init__(self, playground: _DeliveryPlayground | None = None) -> None:
        self.playground = playground
        self.notifications: list[tuple[str, str]] = []

    def query_one(self, _widget_type: object) -> _DeliveryPlayground:
        if self.playground is None:
            raise LookupError("Playground is not mounted")
        return self.playground

    def notify(self, message: str, *, severity: str = "information") -> None:
        self.notifications.append((message, severity))


def _handler(service: object) -> STTSEventHandler:
    handler = STTSEventHandler(
        app=SimpleNamespace(notify=lambda *_args, **_kwargs: None)
    )
    handler._stts_service = service
    return handler


class _SnapshotService:
    def provider_descriptors(self) -> tuple[TTSProviderDescriptor, ...]:
        return (
            TTSProviderDescriptor(
                provider_id="audio_cpp",
                display_name="audio.cpp",
                native=True,
            ),
        )

    def configuration_revision(self, _provider_id: str) -> int:
        return 1

    async def get_catalog(
        self,
        _provider_id: str,
        refresh: bool = False,
    ) -> TTSProviderCatalog:
        del refresh
        return TTSProviderCatalog(
            provider_id="audio_cpp",
            revision=1,
            health=ProviderHealth(state="available", fresh=True),
            models=(
                TTSModelInfo(
                    model_id="model-1",
                    display_name="Model 1",
                    family="test",
                    upstream_mode="tts",
                    formats=("wav",),
                    voices=(),
                    supports_speed=False,
                    omit_voice_uses_server_default=True,
                ),
                TTSModelInfo(
                    model_id="model-2",
                    display_name="Model 2",
                    family="test",
                    upstream_mode="tts",
                    formats=("wav",),
                    voices=(),
                    supports_speed=False,
                    omit_voice_uses_server_default=True,
                ),
            ),
        )

    async def get_voices(
        self,
        _provider_id: str,
        _model_id: str,
        refresh: bool = False,
    ) -> tuple[str, ...]:
        del refresh
        return ()


class _SnapshotHost(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.generation_events: list[STTSPlaygroundGenerateEvent] = []

    def compose(self) -> ComposeResult:
        yield TTSPlaygroundWidget()

    @on(STTSPlaygroundGenerateEvent)
    def capture_generation(self, event: STTSPlaygroundGenerateEvent) -> None:
        self.generation_events.append(event)


class _EndToEndHost(App[None]):
    def __init__(self, native_service: object) -> None:
        super().__init__()
        self.notifications: list[tuple[str, str]] = []
        self._stts_handler = STTSEventHandler(app=self)
        self._stts_handler._stts_service = native_service

    def compose(self) -> ComposeResult:
        yield TTSPlaygroundWidget()

    @on(STTSPlaygroundGenerateEvent)
    def generate(self, event: STTSPlaygroundGenerateEvent) -> None:
        self._stts_handler.start_playground_generation(event)

    def notify(
        self,
        message: str,
        *,
        title: str = "",
        severity: str = "information",
        timeout: float | None = None,
    ) -> None:
        del title, timeout
        self.notifications.append((message, severity))


async def _resolved(value: object) -> object:
    return value


def test_generation_event_owns_one_immutable_request_snapshot() -> None:
    request = _snapshot(options={"nested": {"value": 1}})

    event = STTSPlaygroundGenerateEvent(request)

    assert event.request is request
    with pytest.raises(TypeError):
        event.request.options["new"] = "value"  # type: ignore[index]
    with pytest.raises(TypeError):
        event.request.options["nested"]["value"] = 2  # type: ignore[index]


@pytest.mark.asyncio
async def test_playground_captures_audio_cpp_request_before_controls_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _SnapshotService()
    monkeypatch.setattr(
        STTS_Window,
        "get_cli_setting",
        lambda section, key, default=None: (
            "audio_cpp"
            if (section, key) == ("app_tts", "default_provider")
            else default
        ),
    )
    monkeypatch.setattr(
        STTS_Window,
        "get_tts_service",
        lambda: _resolved(service),
    )
    app = _SnapshotHost()

    async with app.run_test(size=(160, 60)) as pilot:
        await app.workers.wait_for_complete()
        text_area = app.query_one("#tts-text-input", TextArea)
        model_select = app.query_one("#tts-model-select", Select)
        speed_input = app.query_one("#tts-speed-input", Input)
        text_area.text = "original private text"
        await pilot.pause()

        app.query_one(TTSPlaygroundWidget)._generate_tts()
        await pilot.pause()
        assert len(app.generation_events) == 1
        request = app.generation_events[0].request

        text_area.text = "changed after post"
        model_select.value = "model-2"
        speed_input.value = "9.0"
        await pilot.pause()

        UUID(request.operation_id)
        assert request.provider_id == "audio_cpp"
        assert request.model_id == "model-1"
        assert request.text == "original private text"
        assert request.voice_id is None
        assert request.response_format == "wav"
        assert request.speed == 1.0
        assert dict(request.options) == {}


@pytest.mark.asyncio
async def test_playground_stores_delivered_artifact_not_current_selectors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = _SnapshotService()
    monkeypatch.setattr(
        STTS_Window,
        "get_cli_setting",
        lambda section, key, default=None: (
            "audio_cpp"
            if (section, key) == ("app_tts", "default_provider")
            else default
        ),
    )
    monkeypatch.setattr(
        STTS_Window,
        "get_tts_service",
        lambda: _resolved(service),
    )
    app = _SnapshotHost()
    artifact_path = tmp_path / "actual-response.wav"
    artifact_path.write_bytes(b"RIFF")
    artifact = STTSGeneratedAudio(
        path=artifact_path,
        provider_id="audio_cpp",
        model_id="response-model",
        voice_id=None,
        source_text="original text",
        operation_id="operation-2",
        audio_format="wav",
        content_type="audio/wav",
        metadata={"sample_rate": 24_000},
    )

    async with app.run_test(size=(160, 60)) as pilot:
        await app.workers.wait_for_complete()
        widget = app.query_one(TTSPlaygroundWidget)
        app.query_one("#tts-model-select", Select).value = "model-2"
        widget._generation_complete(artifact)
        await pilot.pause()

        assert widget.current_audio_artifact is artifact
        assert widget.current_audio_file == artifact_path
        assert app.query_one("#audio-play-btn", Button).disabled is False
        assert app.query_one("#audio-export-btn", Button).disabled is False
        assert (
            str(app.query_one("#audio-player-status", Static).render())
            == "WAV audio ready to play"
        )


@pytest.mark.asyncio
async def test_audio_cpp_playground_runs_end_to_end_through_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_service = _SnapshotService()
    native_service = _NativeService(
        _Response(_CountingStream((b"RIFF", b"end-to-end")))
    )
    monkeypatch.setattr(
        STTS_Window,
        "get_cli_setting",
        lambda section, key, default=None: (
            "audio_cpp"
            if (section, key) == ("app_tts", "default_provider")
            else default
        ),
    )
    monkeypatch.setattr(
        STTS_Window,
        "get_tts_service",
        lambda: _resolved(catalog_service),
    )
    app = _EndToEndHost(native_service)

    async with app.run_test(size=(160, 60)) as pilot:
        await app.workers.wait_for_complete()
        widget = app.query_one(TTSPlaygroundWidget)
        app.query_one("#tts-text-input", TextArea).text = "synthesize this"
        await pilot.pause()

        widget._generate_tts()
        for _ in range(100):
            if app._stts_handler.playground_state().artifact is not None:
                break
            await pilot.pause(0.02)
        await pilot.pause()

        artifact = app._stts_handler.playground_state().artifact
        assert artifact is not None
        assert artifact.path.read_bytes() == b"RIFFend-to-end"
        assert widget.current_audio_artifact is artifact
        assert widget.current_audio_file == artifact.path
        assert app.query_one("#audio-play-btn", Button).disabled is False
        assert app.query_one("#audio-export-btn", Button).disabled is False

    await app._stts_handler.cleanup_tts_resources()


@pytest.mark.asyncio
async def test_catalog_refresh_does_not_cancel_handler_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_service = _SnapshotService()
    release = asyncio.Event()
    native_service = _NativeService(_Response(_CountingStream((), blocked=release)))
    monkeypatch.setattr(
        STTS_Window,
        "get_cli_setting",
        lambda section, key, default=None: (
            "audio_cpp"
            if (section, key) == ("app_tts", "default_provider")
            else default
        ),
    )
    monkeypatch.setattr(
        STTS_Window,
        "get_tts_service",
        lambda: _resolved(catalog_service),
    )
    app = _EndToEndHost(native_service)

    async with app.run_test(size=(160, 60)) as pilot:
        await app.workers.wait_for_complete()
        widget = app.query_one(TTSPlaygroundWidget)
        app.query_one("#tts-text-input", TextArea).text = "synthesize this"
        await pilot.pause()
        widget._generate_tts()
        for _ in range(100):
            if app._stts_handler.playground_state().generation_active:
                break
            await pilot.pause(0.02)
        generation_task = app._stts_handler._generation_task
        assert generation_task is not None

        widget._load_provider_catalog("audio_cpp", refresh=True)
        await app.workers.wait_for_complete()

        assert app._stts_handler._generation_task is generation_task
        assert generation_task.done() is False
        assert generation_task.cancelled() is False

        release.set()
        await generation_task

    await app._stts_handler.cleanup_tts_resources()


@pytest.mark.asyncio
async def test_audio_cpp_native_generation_consumes_and_closes_one_wav_response() -> (
    None
):
    stream = _CountingStream((b"RIFF", b"audio"))
    response = _Response(
        stream,
        metadata={"sample_rate": 24_000, "engine": "audio.cpp"},
    )
    service = _NativeService(response)
    handler = _handler(service)
    handler._convert_audio_format = AsyncMock(
        side_effect=AssertionError("audio.cpp must not convert")
    )
    snapshot = _snapshot()

    artifact = await handler._generate_audio_cpp(snapshot, None)

    try:
        assert service.requests == [
            TTSRequest(
                provider_id="audio_cpp",
                model_id="model-1",
                text="private source text",
                voice=None,
                response_format="wav",
                speed=1.0,
                options={},
            )
        ]
        assert service.legacy_calls == 0
        handler._convert_audio_format.assert_not_awaited()
        assert stream.iterations == 1
        assert stream.close_calls == 1
        assert response.close_calls == 1
        assert artifact == STTSGeneratedAudio(
            path=artifact.path,
            provider_id="audio_cpp",
            model_id="server-model",
            voice_id=None,
            source_text="private source text",
            operation_id="local-operation-1",
            audio_format="wav",
            content_type="audio/wav",
            metadata={"sample_rate": 24_000, "engine": "audio.cpp"},
        )
        assert artifact.path.read_bytes() == b"RIFFaudio"
        assert artifact.path.suffix == ".wav"
        assert stat.S_IMODE(artifact.path.stat().st_mode) == 0o600
        assert artifact.path in handler._playground_audio_files
    finally:
        artifact.path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_audio_cpp_native_generation_preserves_stream_error_and_closes() -> None:
    primary = RuntimeError("primary stream failure")
    stream = _CountingStream((b"partial",), failure=primary)

    class CloseFailingResponse(_Response):
        async def aclose(self) -> None:
            await super().aclose()
            raise RuntimeError("secondary close failure")

    response = CloseFailingResponse(stream)
    handler = _handler(_NativeService(response))

    with pytest.raises(RuntimeError) as raised:
        await handler._generate_audio_cpp(_snapshot(), None)

    assert raised.value is primary
    assert response.close_calls == 1
    assert stream.close_calls == 1
    assert handler._playground_audio_files == set()


@pytest.mark.asyncio
async def test_audio_cpp_native_generation_propagates_cancellation_and_closes() -> None:
    blocked = asyncio.Event()
    stream = _CountingStream((), blocked=blocked)
    response = _Response(stream)
    handler = _handler(_NativeService(response))
    generation = asyncio.create_task(handler._generate_audio_cpp(_snapshot(), None))
    await asyncio.sleep(0)

    generation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await generation

    assert response.close_calls == 1
    assert stream.close_calls == 1
    assert handler._playground_audio_files == set()


@pytest.mark.asyncio
async def test_legacy_generation_retains_stream_bridge_and_requested_conversion(
    tmp_path: Path,
) -> None:
    service = _LegacyService()
    handler = _handler(service)

    async def convert(input_path: Path, output_format: str) -> Path:
        assert input_path.read_bytes() == b"RIFFlegacy"
        assert output_format == "mp3"
        output = input_path.with_suffix(".mp3")
        output.write_bytes(b"converted")
        return output

    handler._convert_audio_format = AsyncMock(side_effect=convert)

    artifact = await handler._generate_legacy(
        _snapshot(provider_id="openai", response_format="mp3"),
        None,
    )

    try:
        assert len(service.stream_calls) == 1
        request, internal_model_id, _progress_sink = service.stream_calls[0]
        assert request.model == "model-1"
        assert request.input == "private source text"
        assert request.voice == "alloy"
        assert request.response_format == "wav"
        assert internal_model_id == "openai_official_model1"
        assert service.native_calls == 0
        handler._convert_audio_format.assert_awaited_once()
        assert artifact.audio_format == "mp3"
        assert artifact.path.read_bytes() == b"converted"
        assert artifact.provider_id == "openai"
        assert artifact.model_id == "model-1"
        assert artifact.voice_id == "alloy"
        assert artifact.path in handler._playground_audio_files
    finally:
        for path in tuple(handler._playground_audio_files):
            path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_handler_dispatches_native_generation_and_delivers_artifact() -> None:
    response = _Response(_CountingStream((b"RIFF", b"audio")))
    playground = _DeliveryPlayground()
    app = _DeliveryApp(playground)
    handler = STTSEventHandler(app=app)
    service = _NativeService(response)
    handler._stts_service = service

    await handler.handle_playground_generate(STTSPlaygroundGenerateEvent(_snapshot()))

    artifact = handler._current_playground_artifact
    try:
        assert artifact is not None
        assert handler._current_audio_file == artifact.path
        assert playground.completions == [artifact]
        assert app.notifications == [
            ("TTS generation complete!", "information"),
        ]
        assert handler._is_generating is False
    finally:
        if artifact is not None:
            artifact.path.unlink(missing_ok=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code",
    (
        "configuration_invalid",
        "connection_unavailable",
        "contract_incompatible",
        "not_configured",
        "request_invalid",
        "model_invalid",
        "server_busy",
        "generation_failed",
        "audio_response_invalid",
        "generation_timeout",
    ),
)
async def test_native_operation_errors_display_only_stable_safe_contract(
    code: str,
) -> None:
    operation_error = TTSOperationError(
        code=code,  # type: ignore[arg-type]
        message="[safe provider message]",
        retryable=True,
        operation_id="provider-operation",
        recovery_action="Retry from STTS",
    )
    service = SimpleNamespace(
        synthesize=AsyncMock(side_effect=operation_error),
    )
    app = _DeliveryApp()
    handler = STTSEventHandler(app=app)
    handler._stts_service = service

    await handler.handle_playground_generate(STTSPlaygroundGenerateEvent(_snapshot()))

    assert app.notifications == [
        (
            "TTS generation failed: \\[safe provider message] Retry from STTS",
            "error",
        )
    ]
    assert handler._is_generating is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected"),
    (
        (
            TTSProviderReconfiguringError("PRIVATE_RECONFIGURING"),
            "TTS settings are being applied; retry shortly",
        ),
        (
            TTSRegistryClosedError("PRIVATE_CLOSED"),
            "The TTS service is unavailable",
        ),
        (
            ValueError("PRIVATE_CONFIGURATION"),
            "TTS is not configured; open STTS Settings",
        ),
    ),
)
async def test_native_local_failures_use_fixed_recovery_copy(
    failure: Exception,
    expected: str,
) -> None:
    service = SimpleNamespace(synthesize=AsyncMock(side_effect=failure))
    app = _DeliveryApp()
    handler = STTSEventHandler(app=app)
    handler._stts_service = service

    await handler.handle_playground_generate(STTSPlaygroundGenerateEvent(_snapshot()))

    assert app.notifications == [(f"TTS generation failed: {expected}", "error")]
    assert str(failure) not in app.notifications[0][0]


@pytest.mark.asyncio
async def test_unknown_native_failure_never_leaks_exception_or_request_details() -> (
    None
):
    private_values = (
        "UNKNOWN_FAILURE_SECRET",
        "private source text",
        "https://user:password@example.invalid",
        "PRIVATE_MODEL_ID",
    )
    failure = RuntimeError(" ".join(private_values))
    service = SimpleNamespace(synthesize=AsyncMock(side_effect=failure))
    app = _DeliveryApp()
    handler = STTSEventHandler(app=app)
    handler._stts_service = service
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="DEBUG", format="{message}")
    try:
        await handler.handle_playground_generate(
            STTSPlaygroundGenerateEvent(
                _snapshot(
                    model_id=private_values[3],
                    text=private_values[1],
                    options={"origin": private_values[2]},
                )
            )
        )
    finally:
        logger.remove(sink_id)

    rendered = "\n".join(messages)
    notices = "\n".join(message for message, _severity in app.notifications)
    assert app.notifications == [
        (
            "TTS generation failed: Unexpected TTS generation failure; retry",
            "error",
        )
    ]
    for private_value in private_values:
        assert private_value not in rendered
        assert private_value not in notices


@pytest.mark.asyncio
async def test_cancelled_native_handler_closes_without_failure_notice() -> None:
    blocked = asyncio.Event()
    response = _Response(_CountingStream((), blocked=blocked))
    service = _NativeService(response)
    app = _DeliveryApp()
    handler = STTSEventHandler(app=app)
    handler._stts_service = service
    generation = asyncio.create_task(
        handler.handle_playground_generate(STTSPlaygroundGenerateEvent(_snapshot()))
    )
    await asyncio.sleep(0)

    generation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await generation

    assert response.close_calls == 1
    assert app.notifications == []
    assert handler._is_generating is False


@pytest.mark.asyncio
async def test_repeated_generate_is_rejected_without_replacing_active_work() -> None:
    service = SimpleNamespace(synthesize=AsyncMock())
    app = _DeliveryApp()
    handler = STTSEventHandler(app=app)
    handler._stts_service = service
    handler._is_generating = True

    await handler.handle_playground_generate(STTSPlaygroundGenerateEvent(_snapshot()))

    service.synthesize.assert_not_awaited()
    assert handler._is_generating is True
    assert app.notifications == [
        ("TTS generation already in progress", "warning"),
    ]


@pytest.mark.asyncio
async def test_handler_retains_one_generation_task_and_exposes_read_only_state() -> (
    None
):
    release = asyncio.Event()
    first_response = _Response(_CountingStream((), blocked=release))
    service = _NativeService(first_response)
    app = _DeliveryApp()
    handler = STTSEventHandler(app=app)
    handler._stts_service = service
    first_event = STTSPlaygroundGenerateEvent(_snapshot())
    second_event = STTSPlaygroundGenerateEvent(
        STTSPlaygroundRequest(
            operation_id="local-operation-2",
            provider_id="audio_cpp",
            model_id="model-2",
            text="second",
            voice_id=None,
            response_format="wav",
        )
    )

    handler.start_playground_generation(first_event)
    await asyncio.sleep(0)
    retained_task = handler._generation_task
    state = handler.playground_state()
    handler.start_playground_generation(second_event)

    assert retained_task is not None
    assert handler._generation_task is retained_task
    assert state.active_operation_id == "local-operation-1"
    assert state.generation_active is True
    assert state.artifact is None
    assert app.notifications == [
        ("TTS generation already in progress", "warning"),
    ]

    release.set()
    await retained_task
    finished_state = handler.playground_state()
    try:
        assert finished_state.active_operation_id is None
        assert finished_state.generation_active is False
        assert finished_state.artifact is not None
        assert finished_state.artifact.operation_id == "local-operation-1"
    finally:
        await handler.cleanup_tts_resources()


@pytest.mark.asyncio
async def test_successful_replacement_and_shutdown_delete_owned_artifacts() -> None:
    app = _DeliveryApp()
    handler = STTSEventHandler(app=app)
    handler._stts_service = _NativeService(
        _Response(_CountingStream((b"RIFF", b"first")))
    )
    await handler.handle_playground_generate(STTSPlaygroundGenerateEvent(_snapshot()))
    first = handler._current_playground_artifact
    assert first is not None
    assert first.path.exists()

    handler._stts_service = _NativeService(
        _Response(_CountingStream((b"RIFF", b"second")))
    )
    await handler.handle_playground_generate(
        STTSPlaygroundGenerateEvent(
            STTSPlaygroundRequest(
                operation_id="local-operation-2",
                provider_id="audio_cpp",
                model_id="model-2",
                text="second",
                voice_id=None,
                response_format="wav",
            )
        )
    )
    second = handler._current_playground_artifact

    assert second is not None
    assert second.path.exists()
    assert second.path.read_bytes() == b"RIFFsecond"
    assert not first.path.exists()
    assert handler._playground_audio_files == {second.path}
    assert handler._playground_operation_files == {
        "local-operation-2": {second.path},
    }

    await handler.cleanup_tts_resources()

    assert not second.path.exists()
    assert handler._current_playground_artifact is None
    assert handler._current_audio_file is None
    assert handler._playground_audio_files == set()
    assert handler._playground_operation_files == {}


@pytest.mark.asyncio
async def test_generation_survives_broken_or_removed_progress_view() -> None:
    class BrokenPlayground:
        def query_one(self, *_args: object, **_kwargs: object) -> object:
            raise LookupError("PRIVATE_REMOVED_WIDGET")

        def call_from_thread(self, callback: object, *args: object) -> None:
            assert callable(callback)
            callback(*args)

    response = _Response(_CountingStream((b"RIFF", b"audio")))
    app = _DeliveryApp(BrokenPlayground())  # type: ignore[arg-type]
    handler = STTSEventHandler(app=app)
    handler._stts_service = _NativeService(response)

    await handler.handle_playground_generate(STTSPlaygroundGenerateEvent(_snapshot()))

    artifact = handler._current_playground_artifact
    try:
        assert artifact is not None
        assert artifact.path.read_bytes() == b"RIFFaudio"
        assert app.notifications == [
            ("TTS generation complete!", "information"),
        ]
    finally:
        await handler.cleanup_tts_resources()


@pytest.mark.asyncio
async def test_unmounted_view_is_not_retained_or_called_on_completion() -> None:
    release = asyncio.Event()
    response = _Response(_CountingStream((), blocked=release))
    first_view = _DeliveryPlayground()
    app = _DeliveryApp(first_view)
    handler = STTSEventHandler(app=app)
    handler._stts_service = _NativeService(response)
    handler.start_playground_generation(STTSPlaygroundGenerateEvent(_snapshot()))
    await asyncio.sleep(0)
    retained_task = handler._generation_task
    assert retained_task is not None

    app.playground = None
    release.set()
    await retained_task

    state = handler.playground_state()
    try:
        assert first_view.completions == []
        assert state.generation_active is False
        assert state.active_operation_id is None
        assert state.artifact is not None
        assert state.artifact.path.exists()
    finally:
        await handler.cleanup_tts_resources()


@pytest.mark.asyncio
async def test_unavailable_service_releases_playground_generation_state() -> None:
    playground = _DeliveryPlayground()
    app = _DeliveryApp(playground)
    handler = STTSEventHandler(app=app)

    await handler.handle_playground_generate(STTSPlaygroundGenerateEvent(_snapshot()))

    state = handler.playground_state()
    assert playground.completions == [None]
    assert state.active_operation_id is None
    assert state.generation_active is False
    assert state.artifact is None
    assert app.notifications == [
        ("TTS service not initialized", "error"),
    ]
