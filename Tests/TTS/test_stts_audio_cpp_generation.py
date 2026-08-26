from __future__ import annotations

import asyncio
import io
import stat
import struct
import wave
from collections.abc import AsyncIterator, Mapping
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock
from uuid import UUID

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
    CanonicalTTSCloneReference,
    ProviderHealth,
    STTSGeneratedAudio,
    STTSPlaygroundCloneSnapshot,
    STTSPlaygroundProfilePreview,
    STTSPlaygroundRequest,
    STTSPlaygroundResultProjection,
    TTSModelInfo,
    TTSOperationError,
    TTSProviderCatalog,
    TTSProviderDescriptor,
    TTSRequest,
    TTSRequestedSelectionSnapshot,
)
from tldw_chatbook.TTS.adapter_types import (
    _TTS_CLONE_GENERATION_EVIDENCE_TOKEN,
    TTSCloneGenerationEvidence,
    TTSProviderReconfiguringError,
    TTSRegistryClosedError,
)
from tldw_chatbook.TTS.effective_settings import (
    TTSSelectionSource,
    TTSSelectionOverrides,
    TTSStudioDraftSelection,
)
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.studio_preferences import (
    StudioTTSPreferencesSnapshot,
    StudioTTSPreferenceStore,
)
from tldw_chatbook.UI.Screens.settings_speech_tts import (
    build_provider_test_fingerprint,
    load_global_speech_tts_state,
)
from tldw_chatbook.UI.Speech.speech_playground_pane import SpeechPlaygroundPane
from tldw_chatbook.UI.Speech.speech_settings_contracts import (
    SpeechTTSConnectionState,
)


def _snapshot(
    *,
    provider_id: str = "audio_cpp",
    response_format: str = "wav",
    options: Mapping[str, Any] | None = None,
    model_id: str = "model-1",
    text: str = "private source text",
    voice_id: str | None = None,
    speed: float = 1.0,
) -> STTSPlaygroundRequest:
    return STTSPlaygroundRequest(
        operation_id="local-operation-1",
        provider_id=provider_id,
        model_id=model_id,
        text=text,
        voice_id=(
            None
            if provider_id == "audio_cpp"
            else ("alloy" if voice_id is None else voice_id)
        ),
        response_format=response_format,
        speed=speed,
        options=options or {},
    )


def _sample_wav(payload: bytes = b"\x00\x00") -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24_000)
        wav.writeframes(payload)
    return output.getvalue()


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
        sample_rate: int | None = None,
    ) -> None:
        self.provider_id = provider_id
        self.model_id = model_id
        self.audio_format = audio_format
        self.content_type = content_type
        self.byte_stream = stream
        self.metadata = metadata or {}
        self.sample_rate = sample_rate
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1
        await self.byte_stream.aclose()


class _NativeService:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.requests: list[TTSRequest] = []
        self.legacy_calls = 0
        self.saved_revision = 3

    async def synthesize(
        self,
        request: TTSRequest,
        progress_sink: object = None,
    ) -> _Response:
        del progress_sink
        self.requests.append(request)
        return self.response

    async def synthesize_exact(
        self,
        request: TTSRequest,
        progress_sink: object = None,
    ) -> tuple[_Response, TTSRequestedSelectionSnapshot]:
        del progress_sink
        self.requests.append(request)
        return (
            self.response,
            TTSRequestedSelectionSnapshot(
                provider_id=request.provider_id,
                model_id=request.model_id,
                voice_id=request.voice,
                response_format=request.response_format,
                speed=request.speed,
                options=request.options,
                configuration_revision=3,
            ),
        )

    async def generate_audio_stream(
        self,
        *_args: object,
        **_kwargs: object,
    ) -> AsyncIterator[bytes]:
        self.legacy_calls += 1
        raise AssertionError("audio.cpp must not use the legacy stream bridge")
        yield b""  # pragma: no cover

    def saved_configuration_revision(self, provider_id: str) -> int:
        assert provider_id == "audio_cpp"
        return self.saved_revision

    def applied_configuration_revision(self, provider_id: str) -> int:
        assert provider_id == "audio_cpp"
        return self.saved_revision

    def configuration_revision(self, provider_id: str) -> int:
        assert provider_id == "audio_cpp"
        return self.saved_revision


class _LegacyService:
    def __init__(self) -> None:
        self.stream_calls: list[tuple[object, str, object]] = []
        self.native_calls = 0
        self.revision = 5
        self.revision_error: BaseException | None = None
        self.stream_body = b"RIFFlegacy"
        self.saved_preferences = TTSPreferencesSnapshot(
            provider_id="openai",
            model_mode="exact",
            model_id="model-1",
            voice_mode="exact",
            voice_id="alloy",
            response_format="mp3",
            speed=1.0,
        )

    async def synthesize(self, *_args: object, **_kwargs: object) -> None:
        self.native_calls += 1
        raise AssertionError("legacy generation must retain the bridge")

    def configuration_revision(self, _provider_id: str) -> int:
        if self.revision_error is not None:
            raise self.revision_error
        return self.revision

    def saved_configuration_revision(self, _provider_id: str) -> int:
        return self.revision

    def applied_configuration_revision(self, _provider_id: str) -> int:
        return self.revision

    def preferences_snapshot(self) -> TTSPreferencesSnapshot:
        return self.saved_preferences

    async def generate_audio_stream(
        self,
        request: object,
        internal_model_id: str,
        progress_sink: object = None,
    ) -> AsyncIterator[bytes]:
        self.stream_calls.append((request, internal_model_id, progress_sink))
        yield self.stream_body


class _RevisionSeparatedLegacyService(_LegacyService):
    def __init__(
        self,
        *,
        saved_revision: int,
        applied_revision: int,
        runtime_revision: int,
    ) -> None:
        super().__init__()
        self.saved_revision = saved_revision
        self.applied_revision = applied_revision
        self.runtime_revision = runtime_revision

    def saved_configuration_revision(self, _provider_id: str) -> int:
        return self.saved_revision

    def applied_configuration_revision(self, _provider_id: str) -> int:
        return self.applied_revision

    def configuration_revision(self, _provider_id: str) -> int:
        return self.runtime_revision


class _StudioService:
    def __init__(
        self,
        response: _Response,
        *,
        effective_provider_id: str = "audio_cpp",
        effective_model_id: str = "draft/model",
        effective_voice_id: str | None = None,
        effective_response_format: str = "wav",
        effective_speed: float = 1.0,
        effective_configuration_revision: int = 9,
        # `TTSEffectiveSelection` always carries this axis; a fake that omits
        # it invents a shape the real system never produces.
        effective_provider_options: Mapping[str, Any] | None = None,
        clone_evidence: TTSCloneGenerationEvidence | None = None,
        effective_uses_draft: bool = False,
    ) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []
        self._effective_provider_id = effective_provider_id
        self._effective_model_id = effective_model_id
        self._effective_voice_id = effective_voice_id
        self._effective_response_format = effective_response_format
        self._effective_speed = effective_speed
        self._effective_configuration_revision = effective_configuration_revision
        self._effective_provider_options = (
            {} if effective_provider_options is None else effective_provider_options
        )
        self._clone_evidence = clone_evidence
        self._effective_uses_draft = effective_uses_draft

    async def synthesize_effective(self, **kwargs: object) -> tuple[object, object]:
        self.calls.append(kwargs)
        return self.response, SimpleNamespace(
            provider_id=self._effective_provider_id,
            model_id=self._effective_model_id,
            voice_id=self._effective_voice_id,
            response_format=self._effective_response_format,
            speed=self._effective_speed,
            provider_options=self._effective_provider_options,
            revisions=SimpleNamespace(
                provider_configuration=self._effective_configuration_revision
            ),
            sources={
                "provider_id": (
                    TTSSelectionSource.STUDIO_DRAFT
                    if self._effective_uses_draft
                    else TTSSelectionSource.STUDIO_SAVED
                )
            },
            provider_option_sources={},
            studio_preview=False,
        )

    async def synthesize_effective_with_evidence(
        self,
        **kwargs: object,
    ) -> tuple[object, object, TTSCloneGenerationEvidence | None]:
        response, effective = await self.synthesize_effective(**kwargs)
        return response, effective, self._clone_evidence

    def saved_configuration_revision(self, provider_id: str) -> int:
        assert provider_id == self._effective_provider_id
        return self._effective_configuration_revision

    def applied_configuration_revision(self, provider_id: str) -> int:
        assert provider_id == self._effective_provider_id
        return self._effective_configuration_revision

    def configuration_revision(self, provider_id: str) -> int:
        assert provider_id == self._effective_provider_id
        return self._effective_configuration_revision


class _DeliveryPlayground:
    def __init__(self) -> None:
        self.completions: list[STTSPlaygroundResultProjection | None] = []
        self.log = SimpleNamespace(write=Mock())
        self.progress = SimpleNamespace(update=Mock())
        self.container = SimpleNamespace(remove_class=Mock(), add_class=Mock())
        self.status = SimpleNamespace(update=Mock())
        self.button = SimpleNamespace(disabled=True)
        self.accepted_clone_results: list[tuple[str, int]] = []

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
        artifact: STTSPlaygroundResultProjection | None,
    ) -> None:
        self.completions.append(artifact)

    def _accept_clone_generation_result(
        self,
        operation_id: str,
        draft_revision: int,
    ) -> None:
        self.accepted_clone_results.append((operation_id, draft_revision))


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
    def __init__(self) -> None:
        self.revision = 1

    def provider_descriptors(self) -> tuple[TTSProviderDescriptor, ...]:
        return (
            TTSProviderDescriptor(
                provider_id="audio_cpp",
                display_name="audio.cpp",
                native=True,
            ),
        )

    def configuration_revision(self, _provider_id: str) -> int:
        return self.revision

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
        yield SpeechPlaygroundPane(id="speech-playground-pane", provider="audio_cpp")

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
        yield SpeechPlaygroundPane(id="speech-playground-pane", provider="audio_cpp")

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
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    app = _SnapshotHost()

    async with app.run_test(size=(160, 60)) as pilot:
        await app.workers.wait_for_complete()
        text_area = app.query_one("#tts-text-input", TextArea)
        model_select = app.query_one("#tts-model-select", Select)
        speed_input = app.query_one("#tts-speed-input", Input)
        text_area.text = "original private text"
        await pilot.pause()

        app.query_one(SpeechPlaygroundPane)._generate_tts()
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
async def test_playground_shortcut_does_not_replace_active_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _SnapshotService()
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    app = _SnapshotHost()

    async with app.run_test(size=(160, 60)) as pilot:
        await app.workers.wait_for_complete()
        widget = app.query_one(SpeechPlaygroundPane)
        app.query_one("#tts-text-input", TextArea).text = "only once"
        await pilot.pause()

        widget._generate_tts()
        await pilot.pause()
        first_operation_id = widget._generation_operation_id

        widget.action_generate_tts()
        await pilot.pause()

        assert first_operation_id is not None
        assert widget._generation_operation_id == first_operation_id
        assert len(app.generation_events) == 1


@pytest.mark.asyncio
async def test_playground_shortcut_rejects_stale_configuration_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _SnapshotService()
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
    )
    app = _SnapshotHost()

    async with app.run_test(size=(160, 60)) as pilot:
        await app.workers.wait_for_complete()
        widget = app.query_one(SpeechPlaygroundPane)
        app.query_one("#tts-text-input", TextArea).text = "stale request"
        await pilot.pause()

        service.revision = 2
        widget.action_generate_tts()
        await pilot.pause()

        assert widget._generation_operation_id is None
        assert app.generation_events == []


@pytest.mark.asyncio
async def test_playground_stores_delivered_artifact_not_current_selectors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = _SnapshotService()
    monkeypatch.setattr(
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(service),
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
        await pilot.pause()
        widget = app.query_one(SpeechPlaygroundPane)
        app.query_one("#tts-model-select", Select).value = "model-2"
        widget._generation_complete(artifact)
        await pilot.pause()

        assert type(widget.current_audio_artifact) is STTSPlaygroundResultProjection
        assert widget.current_audio_artifact.operation_id == artifact.operation_id
        assert not hasattr(widget.current_audio_artifact, "clone_evidence")
        assert widget.current_audio_file == artifact_path
        assert app.query_one("#audio-play-btn", Button).disabled is False
        assert app.query_one("#audio-export-btn", Button).disabled is False
        # "WAV audio ready to play" was the retired legacy widget's copy;
        # the pane's `_current_result_status_copy` (SpeechPlaybackMixin,
        # already covered on its own by
        # test_current_result_reports_only_known_artifact_facts) renders
        # "Ready · WAV[ · duration]" instead. Asserted here too because this
        # is an end-to-end delivery path, not just the copy function.
        assert (
            str(app.query_one("#audio-player-status", Static).render()) == "Ready · WAV"
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
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(catalog_service),
    )
    app = _EndToEndHost(native_service)

    async with app.run_test(size=(160, 60)) as pilot:
        await app.workers.wait_for_complete()
        widget = app.query_one(SpeechPlaygroundPane)
        app.query_one("#tts-text-input", TextArea).text = "synthesize this"
        await pilot.pause()

        widget._generate_tts()
        for _ in range(100):
            if app._stts_handler.playground_state().artifact is not None:
                break
            await pilot.pause(0.02)
        await pilot.pause()

        projection = app._stts_handler.playground_state().artifact
        assert projection is not None
        assert projection.path.read_bytes() == b"RIFFend-to-end"
        assert widget.current_audio_artifact == projection
        assert type(widget.current_audio_artifact) is STTSPlaygroundResultProjection
        assert widget.current_audio_file == projection.path
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
        SpeechPlaygroundPane,
        "_tts_service_factory",
        lambda self: _resolved(catalog_service),
    )
    app = _EndToEndHost(native_service)

    async with app.run_test(size=(160, 60)) as pilot:
        await app.workers.wait_for_complete()
        widget = app.query_one(SpeechPlaygroundPane)
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
        metadata={"engine": "audio.cpp"},
        sample_rate=24_000,
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
            requested_selection=TTSRequestedSelectionSnapshot(
                provider_id="audio_cpp",
                model_id="model-1",
                voice_id=None,
                response_format="wav",
                speed=1.0,
                options={},
                configuration_revision=3,
            ),
        )
        assert not hasattr(artifact.requested_selection, "text")
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
        assert artifact.requested_selection == TTSRequestedSelectionSnapshot(
            provider_id="openai",
            model_id="model-1",
            voice_id="alloy",
            response_format="mp3",
            speed=1.0,
            options={},
            configuration_revision=5,
        )
        assert artifact.profile_save_eligible is True
        assert artifact.path in handler._playground_audio_files
    finally:
        for path in tuple(handler._playground_audio_files):
            path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_legacy_generation_with_provider_options_is_not_save_eligible(
    tmp_path: Path,
) -> None:
    """Mirrors the Studio-effective coverage for the standalone legacy
    bridge (`_generate_legacy`, taken when a Playground request carries no
    Studio preferences): it must also pass the real `options` through to
    `_build_requested_selection` rather than the pre-fix hardcoded `{}`, so
    a Higgs/Chatterbox result that used `extra_params` is refused
    provenance -- and states why -- instead of saving a profile that
    silently drops what the user configured."""

    service = _LegacyService()
    handler = _handler(service)

    async def convert(input_path: Path, output_format: str) -> Path:
        output = input_path.with_suffix(".mp3")
        output.write_bytes(b"converted")
        return output

    handler._convert_audio_format = AsyncMock(side_effect=convert)

    artifact = await handler._generate_legacy(
        _snapshot(
            provider_id="higgs",
            response_format="mp3",
            options={"temperature": 0.8},
        ),
        None,
    )

    try:
        assert artifact.provider_id == "higgs"
        assert artifact.requested_selection is None
        assert artifact.profile_save_eligible is False
        assert artifact.profile_save_block_code == "provider_options"
    finally:
        for path in tuple(handler._playground_audio_files):
            path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_legacy_generation_survives_configuration_revision_failure(
    tmp_path: Path,
) -> None:
    """A `configuration_revision` failure must degrade provenance, not the
    generation: the artifact still returns with the audio it produced, just
    not profile-save eligible."""

    service = _LegacyService()
    service.revision_error = RuntimeError("registry unavailable")
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
        handler._convert_audio_format.assert_awaited_once()
        assert artifact.audio_format == "mp3"
        assert artifact.path.read_bytes() == b"converted"
        assert artifact.provider_id == "openai"
        assert artifact.requested_selection is None
        assert artifact.profile_save_eligible is False
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
        assert playground.completions == [
            STTSPlaygroundResultProjection.from_artifact(artifact)
        ]
        assert app.notifications == [
            ("TTS generation complete!", "information"),
        ]
        assert handler._is_generating is False
    finally:
        if artifact is not None:
            artifact.path.unlink(missing_ok=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "expected"),
    (
        ("valid", SpeechTTSConnectionState.REACHABLE),
        ("empty", SpeechTTSConnectionState.NOT_TESTED),
        ("oversized", SpeechTTSConnectionState.NOT_TESTED),
        ("invalid", SpeechTTSConnectionState.NOT_TESTED),
        ("invalid_content_type", SpeechTTSConnectionState.NOT_TESTED),
        ("failed", SpeechTTSConnectionState.NOT_TESTED),
    ),
)
async def test_playground_records_only_successful_bounded_valid_sample_evidence(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    expected: SpeechTTSConnectionState,
) -> None:
    monkeypatch.setattr(
        "tldw_chatbook.Event_Handlers.STTS_Events.stts_events.get_runtime_config_snapshot",
        lambda: SimpleNamespace(values={}),
        raising=False,
    )
    state = load_global_speech_tts_state({}, environment={})
    fingerprint = build_provider_test_fingerprint(
        state,
        provider_id="audio_cpp",
        saved_revision=3,
    )
    sample = {
        "valid": _sample_wav(),
        "empty": b"",
        "oversized": _sample_wav(b"\x00" * (8 * 1024 * 1024)),
        "invalid": b"not audio",
        "invalid_content_type": _sample_wav(),
        "failed": b"",
    }[case]
    response = _Response(
        _CountingStream((sample,)),
        content_type="text/html" if case == "invalid_content_type" else "audio/wav",
    )
    service = _NativeService(response)
    if case == "failed":
        service.synthesize_exact = AsyncMock(
            side_effect=RuntimeError("provider failed")
        )
    handler = STTSEventHandler(app=_DeliveryApp())
    handler._stts_service = service

    await handler.handle_playground_generate(STTSPlaygroundGenerateEvent(_snapshot()))

    artifact = handler._current_playground_artifact
    try:
        assert handler.provider_test_evidence.sample_state(fingerprint) is expected
    finally:
        if artifact is not None:
            artifact.path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_cancelled_playground_sample_does_not_record_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tldw_chatbook.Event_Handlers.STTS_Events.stts_events.get_runtime_config_snapshot",
        lambda: SimpleNamespace(values={}),
        raising=False,
    )
    state = load_global_speech_tts_state({}, environment={})
    fingerprint = build_provider_test_fingerprint(
        state,
        provider_id="audio_cpp",
        saved_revision=3,
    )
    blocked = asyncio.Event()
    response = _Response(_CountingStream((), blocked=blocked))
    handler = STTSEventHandler(app=_DeliveryApp())
    handler._stts_service = _NativeService(response)
    generation = asyncio.create_task(
        handler.handle_playground_generate(STTSPlaygroundGenerateEvent(_snapshot()))
    )
    await asyncio.sleep(0)

    generation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await generation

    assert (
        handler.provider_test_evidence.sample_state(fingerprint)
        is SpeechTTSConnectionState.NOT_TESTED
    )


@pytest.mark.asyncio
async def test_evidence_lookup_failure_does_not_fail_valid_generation() -> None:
    response = _Response(_CountingStream((_sample_wav(),)))
    service = _NativeService(response)
    service.saved_configuration_revision = Mock(
        side_effect=RuntimeError("revision unavailable")
    )
    handler = STTSEventHandler(app=_DeliveryApp())
    handler._stts_service = service

    await handler.handle_playground_generate(STTSPlaygroundGenerateEvent(_snapshot()))

    artifact = handler._current_playground_artifact
    try:
        assert artifact is not None
    finally:
        if artifact is not None:
            artifact.path.unlink(missing_ok=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "expected"),
    (
        ("saved", SpeechTTSConnectionState.REACHABLE),
        ("model", SpeechTTSConnectionState.NOT_TESTED),
        ("voice", SpeechTTSConnectionState.NOT_TESTED),
        ("speed", SpeechTTSConnectionState.NOT_TESTED),
        ("options", SpeechTTSConnectionState.NOT_TESTED),
    ),
)
async def test_legacy_sample_certifies_only_exact_saved_effective_selection(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    expected: SpeechTTSConnectionState,
) -> None:
    monkeypatch.setattr(
        "tldw_chatbook.Event_Handlers.STTS_Events.stts_events.get_runtime_config_snapshot",
        lambda: SimpleNamespace(values={}),
    )
    service = _LegacyService()
    service.stream_body = _sample_wav()
    service.saved_preferences = TTSPreferencesSnapshot(
        provider_id="openai",
        model_mode="exact",
        model_id="model-1",
        voice_mode="exact",
        voice_id="alloy",
        response_format="wav",
        speed=1.0,
    )
    request = _snapshot(
        provider_id="openai",
        response_format="wav",
        model_id="unsaved-model" if case == "model" else "model-1",
        voice_id="echo" if case == "voice" else "alloy",
        speed=1.25 if case == "speed" else 1.0,
        options={"temperature": 0.8} if case == "options" else {},
    )
    state = load_global_speech_tts_state({}, environment={})
    fingerprint = build_provider_test_fingerprint(
        state,
        provider_id="openai",
        saved_revision=5,
    )
    handler = STTSEventHandler(app=_DeliveryApp())
    handler._stts_service = service

    await handler.handle_playground_generate(STTSPlaygroundGenerateEvent(request))

    artifact = handler._current_playground_artifact
    try:
        assert artifact is not None
        assert handler.provider_test_evidence.sample_state(fingerprint) is expected
    finally:
        if artifact is not None:
            artifact.path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_sample_evidence_accepts_distinct_matching_publication_and_runtime_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tldw_chatbook.Event_Handlers.STTS_Events.stts_events.get_runtime_config_snapshot",
        lambda: SimpleNamespace(values={}),
    )
    service = _RevisionSeparatedLegacyService(
        saved_revision=7,
        applied_revision=7,
        runtime_revision=41,
    )
    service.stream_body = _sample_wav()
    service.saved_preferences = TTSPreferencesSnapshot(
        provider_id="openai",
        model_mode="exact",
        model_id="model-1",
        voice_mode="exact",
        voice_id="alloy",
        response_format="wav",
        speed=1.0,
    )
    state = load_global_speech_tts_state({}, environment={})
    fingerprint = build_provider_test_fingerprint(
        state,
        provider_id="openai",
        saved_revision=7,
    )
    handler = STTSEventHandler(app=_DeliveryApp())
    handler._stts_service = service

    await handler.handle_playground_generate(
        STTSPlaygroundGenerateEvent(
            _snapshot(provider_id="openai", response_format="wav")
        )
    )

    artifact = handler._current_playground_artifact
    try:
        assert artifact is not None
        assert (
            handler.provider_test_evidence.sample_state(fingerprint)
            is SpeechTTSConnectionState.REACHABLE
        )
    finally:
        if artifact is not None:
            artifact.path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_coincidental_runtime_revision_cannot_mask_unapplied_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tldw_chatbook.Event_Handlers.STTS_Events.stts_events.get_runtime_config_snapshot",
        lambda: SimpleNamespace(values={}),
    )
    service = _RevisionSeparatedLegacyService(
        saved_revision=7,
        applied_revision=6,
        runtime_revision=7,
    )
    service.stream_body = _sample_wav()
    service.saved_preferences = TTSPreferencesSnapshot(
        provider_id="openai",
        model_mode="exact",
        model_id="model-1",
        voice_mode="exact",
        voice_id="alloy",
        response_format="wav",
        speed=1.0,
    )
    state = load_global_speech_tts_state({}, environment={})
    fingerprint = build_provider_test_fingerprint(
        state,
        provider_id="openai",
        saved_revision=7,
    )
    handler = STTSEventHandler(app=_DeliveryApp())
    handler._stts_service = service

    await handler.handle_playground_generate(
        STTSPlaygroundGenerateEvent(
            _snapshot(provider_id="openai", response_format="wav")
        )
    )

    artifact = handler._current_playground_artifact
    try:
        assert artifact is not None
        assert (
            handler.provider_test_evidence.sample_state(fingerprint)
            is SpeechTTSConnectionState.NOT_TESTED
        )
    finally:
        if artifact is not None:
            artifact.path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_sample_evidence_rejects_runtime_identity_change_during_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tldw_chatbook.Event_Handlers.STTS_Events.stts_events.get_runtime_config_snapshot",
        lambda: SimpleNamespace(values={}),
    )
    service = _RevisionSeparatedLegacyService(
        saved_revision=7,
        applied_revision=7,
        runtime_revision=41,
    )
    service.saved_preferences = TTSPreferencesSnapshot(
        provider_id="openai",
        model_mode="exact",
        model_id="model-1",
        voice_mode="exact",
        voice_id="alloy",
        response_format="wav",
        speed=1.0,
    )

    async def changing_stream(
        *_args: object, **_kwargs: object
    ) -> AsyncIterator[bytes]:
        yield _sample_wav()
        service.runtime_revision = 42

    service.generate_audio_stream = changing_stream  # type: ignore[method-assign]
    state = load_global_speech_tts_state({}, environment={})
    fingerprint = build_provider_test_fingerprint(
        state,
        provider_id="openai",
        saved_revision=7,
    )
    handler = STTSEventHandler(app=_DeliveryApp())
    handler._stts_service = service

    await handler.handle_playground_generate(
        STTSPlaygroundGenerateEvent(
            _snapshot(provider_id="openai", response_format="wav")
        )
    )

    artifact = handler._current_playground_artifact
    try:
        assert artifact is not None
        assert (
            handler.provider_test_evidence.sample_state(fingerprint)
            is SpeechTTSConnectionState.NOT_TESTED
        )
    finally:
        if artifact is not None:
            artifact.path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_sample_evidence_rejects_stale_publication_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tldw_chatbook.Event_Handlers.STTS_Events.stts_events.get_runtime_config_snapshot",
        lambda: SimpleNamespace(values={}),
    )
    service = _RevisionSeparatedLegacyService(
        saved_revision=7,
        applied_revision=7,
        runtime_revision=41,
    )
    service.saved_preferences = TTSPreferencesSnapshot(
        provider_id="openai",
        model_mode="exact",
        model_id="model-1",
        voice_mode="exact",
        voice_id="alloy",
        response_format="wav",
        speed=1.0,
    )

    async def changing_stream(
        *_args: object, **_kwargs: object
    ) -> AsyncIterator[bytes]:
        yield _sample_wav()
        service.saved_revision = 8
        service.applied_revision = 8

    service.generate_audio_stream = changing_stream  # type: ignore[method-assign]
    state = load_global_speech_tts_state({}, environment={})
    stale = build_provider_test_fingerprint(
        state,
        provider_id="openai",
        saved_revision=7,
    )
    current = build_provider_test_fingerprint(
        state,
        provider_id="openai",
        saved_revision=8,
    )
    handler = STTSEventHandler(app=_DeliveryApp())
    handler._stts_service = service

    await handler.handle_playground_generate(
        STTSPlaygroundGenerateEvent(
            _snapshot(provider_id="openai", response_format="wav")
        )
    )

    artifact = handler._current_playground_artifact
    try:
        assert artifact is not None
        assert (
            handler.provider_test_evidence.sample_state(stale)
            is SpeechTTSConnectionState.NOT_TESTED
        )
        assert (
            handler.provider_test_evidence.sample_state(current)
            is SpeechTTSConnectionState.NOT_TESTED
        )
    finally:
        if artifact is not None:
            artifact.path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_sample_evidence_rejects_effective_selection_mismatched_to_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tldw_chatbook.Event_Handlers.STTS_Events.stts_events.get_runtime_config_snapshot",
        lambda: SimpleNamespace(values={}),
    )
    response = _Response(_CountingStream((_sample_wav(),)))
    service = _NativeService(response)
    service.saved_revision = 7
    service.configuration_revision = Mock(return_value=41)  # type: ignore[method-assign]
    service.synthesize_exact = AsyncMock(
        return_value=(
            response,
            TTSRequestedSelectionSnapshot(
                provider_id="audio_cpp",
                model_id="different-model",
                voice_id=None,
                response_format="wav",
                speed=1.0,
                options={},
                configuration_revision=41,
            ),
        )
    )
    state = load_global_speech_tts_state({}, environment={})
    fingerprint = build_provider_test_fingerprint(
        state,
        provider_id="audio_cpp",
        saved_revision=7,
    )
    handler = STTSEventHandler(app=_DeliveryApp())
    handler._stts_service = service

    await handler.handle_playground_generate(STTSPlaygroundGenerateEvent(_snapshot()))

    artifact = handler._current_playground_artifact
    try:
        assert artifact is not None
        assert (
            handler.provider_test_evidence.sample_state(fingerprint)
            is SpeechTTSConnectionState.NOT_TESTED
        )
    finally:
        if artifact is not None:
            artifact.path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_legacy_sample_revision_change_during_stream_prevents_attribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tldw_chatbook.Event_Handlers.STTS_Events.stts_events.get_runtime_config_snapshot",
        lambda: SimpleNamespace(values={}),
    )
    service = _LegacyService()
    service.saved_preferences = TTSPreferencesSnapshot(
        provider_id="openai",
        model_mode="exact",
        model_id="model-1",
        voice_mode="exact",
        voice_id="alloy",
        response_format="wav",
        speed=1.0,
    )

    async def changing_stream(
        *_args: object, **_kwargs: object
    ) -> AsyncIterator[bytes]:
        yield _sample_wav()
        service.revision = 6

    service.generate_audio_stream = changing_stream  # type: ignore[method-assign]
    state = load_global_speech_tts_state({}, environment={})
    original = build_provider_test_fingerprint(
        state,
        provider_id="openai",
        saved_revision=5,
    )
    changed = build_provider_test_fingerprint(
        state,
        provider_id="openai",
        saved_revision=6,
    )
    handler = STTSEventHandler(app=_DeliveryApp())
    handler._stts_service = service

    await handler.handle_playground_generate(
        STTSPlaygroundGenerateEvent(
            _snapshot(provider_id="openai", response_format="wav")
        )
    )

    artifact = handler._current_playground_artifact
    try:
        assert artifact is not None
        assert (
            handler.provider_test_evidence.sample_state(original)
            is SpeechTTSConnectionState.NOT_TESTED
        )
        assert (
            handler.provider_test_evidence.sample_state(changed)
            is SpeechTTSConnectionState.NOT_TESTED
        )
    finally:
        if artifact is not None:
            artifact.path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_legacy_saved_selection_change_during_stream_prevents_attribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tldw_chatbook.Event_Handlers.STTS_Events.stts_events.get_runtime_config_snapshot",
        lambda: SimpleNamespace(values={}),
    )
    service = _LegacyService()
    service.saved_preferences = TTSPreferencesSnapshot(
        provider_id="openai",
        model_mode="exact",
        model_id="model-1",
        voice_mode="exact",
        voice_id="alloy",
        response_format="wav",
        speed=1.0,
    )

    async def changing_stream(
        *_args: object, **_kwargs: object
    ) -> AsyncIterator[bytes]:
        yield _sample_wav()
        service.saved_preferences = TTSPreferencesSnapshot(
            provider_id="openai",
            model_mode="exact",
            model_id="model-2",
            voice_mode="exact",
            voice_id="alloy",
            response_format="wav",
            speed=1.0,
        )

    service.generate_audio_stream = changing_stream  # type: ignore[method-assign]
    state = load_global_speech_tts_state({}, environment={})
    fingerprint = build_provider_test_fingerprint(
        state,
        provider_id="openai",
        saved_revision=5,
    )
    handler = STTSEventHandler(app=_DeliveryApp())
    handler._stts_service = service

    await handler.handle_playground_generate(
        STTSPlaygroundGenerateEvent(
            _snapshot(provider_id="openai", response_format="wav")
        )
    )

    artifact = handler._current_playground_artifact
    try:
        assert artifact is not None
        assert (
            handler.provider_test_evidence.sample_state(fingerprint)
            is SpeechTTSConnectionState.NOT_TESTED
        )
    finally:
        if artifact is not None:
            artifact.path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_concurrent_saved_revision_change_prevents_sample_attribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tldw_chatbook.Event_Handlers.STTS_Events.stts_events.get_runtime_config_snapshot",
        lambda: SimpleNamespace(values={}),
    )
    service: _NativeService

    async def changing_stream() -> AsyncIterator[bytes]:
        yield _sample_wav()
        service.saved_revision = 4

    response = _Response(changing_stream())  # type: ignore[arg-type]
    service = _NativeService(response)
    state = load_global_speech_tts_state({}, environment={})
    original = build_provider_test_fingerprint(
        state,
        provider_id="audio_cpp",
        saved_revision=3,
    )
    changed = build_provider_test_fingerprint(
        state,
        provider_id="audio_cpp",
        saved_revision=4,
    )
    handler = STTSEventHandler(app=_DeliveryApp())
    handler._stts_service = service

    await handler.handle_playground_generate(STTSPlaygroundGenerateEvent(_snapshot()))

    artifact = handler._current_playground_artifact
    try:
        assert artifact is not None
        assert (
            handler.provider_test_evidence.sample_state(original)
            is SpeechTTSConnectionState.NOT_TESTED
        )
        assert (
            handler.provider_test_evidence.sample_state(changed)
            is SpeechTTSConnectionState.NOT_TESTED
        )
    finally:
        if artifact is not None:
            artifact.path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_studio_draft_generation_does_not_certify_saved_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tldw_chatbook.Event_Handlers.STTS_Events.stts_events.get_runtime_config_snapshot",
        lambda: SimpleNamespace(values={}),
    )
    response = _Response(_CountingStream((_sample_wav(),)))
    service = _StudioService(
        response,
        effective_configuration_revision=9,
        effective_uses_draft=True,
    )
    preferences = StudioTTSPreferencesSnapshot(revision=5)
    draft = TTSStudioDraftSelection(
        selection=TTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="draft/model",
            voice_mode="server_default",
            response_format="wav",
            speed=1.0,
            provider_options={},
        ),
        base_revision=5,
    )
    request = STTSPlaygroundRequest(
        operation_id="studio-draft-evidence",
        provider_id="audio_cpp",
        model_id="draft/model",
        text="draft sample",
        voice_id=None,
        response_format="wav",
        studio_draft=draft,
        studio_preferences=preferences,
    )
    state = load_global_speech_tts_state({}, environment={})
    fingerprint = build_provider_test_fingerprint(
        state,
        provider_id="audio_cpp",
        saved_revision=9,
    )
    handler = STTSEventHandler(app=_DeliveryApp())
    handler._stts_service = service

    await handler.handle_playground_generate(STTSPlaygroundGenerateEvent(request))

    artifact = handler._current_playground_artifact
    try:
        assert artifact is not None
        assert (
            handler.provider_test_evidence.sample_state(fingerprint)
            is SpeechTTSConnectionState.NOT_TESTED
        )
    finally:
        if artifact is not None:
            artifact.path.unlink(missing_ok=True)


def test_clone_success_acknowledges_only_the_accepted_draft_revision(
    tmp_path: Path,
) -> None:
    path = tmp_path / "clone-result.wav"
    path.write_bytes(b"RIFF")
    artifact = STTSGeneratedAudio(
        path=path,
        provider_id="audio_cpp",
        model_id="clone-model",
        voice_id=None,
        source_text="target text",
        operation_id="clone-operation",
        audio_format="wav",
        content_type="audio/wav",
    )
    playground = _DeliveryPlayground()
    handler = STTSEventHandler(app=_DeliveryApp(playground))
    handler._active_playground_operation_id = artifact.operation_id

    handler._deliver_generation_success(
        artifact.operation_id,
        artifact,
        accepted_clone_draft_revision=6,
    )

    assert playground.accepted_clone_results == [(artifact.operation_id, 6)]
    assert playground.completions == [
        STTSPlaygroundResultProjection.from_artifact(artifact)
    ]


@pytest.mark.asyncio
async def test_studio_profile_preview_forwards_only_identity_and_private_resolver() -> (
    None
):
    response = _Response(_CountingStream((b"RIFF", b"audio")))
    service = _StudioService(response, effective_model_id="clone-model")
    profile_reference = object()
    profile_service = SimpleNamespace(
        get_profile=AsyncMock(
            return_value=SimpleNamespace(
                repository_generation=8,
                profile=SimpleNamespace(
                    revision=5,
                    provider_id="audio_cpp",
                    model_id="clone-model",
                    reference=object(),
                ),
            )
        ),
        get_reference=AsyncMock(return_value=profile_reference),
    )
    app = SimpleNamespace(
        _ensure_tts_profile_service=AsyncMock(return_value=profile_service),
        notify=lambda *_args, **_kwargs: None,
    )
    handler = STTSEventHandler(app=app)
    handler._stts_service = service
    preview = STTSPlaygroundProfilePreview(
        profile_id=UUID("77777777-7777-4777-8777-777777777777"),
        repository_generation=8,
        profile_revision=5,
    )
    preferences = StudioTTSPreferencesSnapshot(revision=3)
    snapshot = STTSPlaygroundRequest(
        operation_id="profile-preview-op",
        provider_id="audio_cpp",
        model_id="clone-model",
        text="preview text",
        voice_id=None,
        response_format="wav",
        studio_draft=TTSStudioDraftSelection(
            selection=TTSSelectionOverrides(
                provider_id="audio_cpp",
                model_mode="exact",
                model_id="clone-model",
                voice_mode="server_default",
                response_format="wav",
                speed=1.0,
                provider_options={},
            ),
            base_revision=3,
            preview=True,
        ),
        studio_preferences=preferences,
        profile_preview=preview,
    )

    artifact = await handler._generate_studio_effective(snapshot, None)

    try:
        assert len(service.calls) == 1
        call = service.calls[0]
        assert call["profile_preview"] is preview
        resolver = call["profile_reference_resolver"]
        assert callable(resolver)
        assert profile_reference not in call.values()
        resolved = await resolver(  # type: ignore[operator]
            preview.profile_id,
            preview.repository_generation,
            preview.profile_revision,
        )
        assert resolved is profile_reference
        profile_service.get_profile.assert_awaited_once_with(preview.profile_id)
        profile_service.get_reference.assert_awaited_once_with(
            preview.profile_id,
            expected_generation=preview.repository_generation,
            expected_revision=preview.profile_revision,
        )
        profile_service.get_profile.return_value.profile.model_id = "other-model"
        with pytest.raises(RuntimeError, match="profile preview is stale"):
            await resolver(  # type: ignore[operator]
                preview.profile_id,
                preview.repository_generation,
                preview.profile_revision,
            )
        assert profile_service.get_reference.await_count == 1
    finally:
        artifact.path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_studio_clone_audition_forwards_exact_canonical_snapshot() -> None:
    frames = 32
    sample_rate = 16_000
    pcm = struct.pack("<h", 4) * frames
    fmt = struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16)
    body = (
        b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(pcm))
        + pcm
    )
    wav = b"RIFF" + struct.pack("<I", len(body)) + body
    canonical = CanonicalTTSCloneReference(
        wav_bytes=wav,
        reference_text="Private transcript",
        sha256=sha256(wav).hexdigest(),
        byte_length=len(wav),
        duration_ms=2,
        sample_rate_hz=sample_rate,
        channels=1,
        sample_encoding="pcm_s16le",
    )
    evidence = TTSCloneGenerationEvidence(
        _TTS_CLONE_GENERATION_EVIDENCE_TOKEN,
        canonical_reference=canonical,
        model_id="clone-model",
        recipe_id="pocket_tts",
        recipe_revision=1,
        provider_configuration_revision=9,
        applied_provider_generation=2,
        process_generation=7,
    )
    response = _Response(_CountingStream((wav,)))
    service = _StudioService(
        response,
        effective_model_id="clone-model",
        clone_evidence=evidence,
    )
    handler = _handler(service)
    clone = STTSPlaygroundCloneSnapshot(
        draft_revision=6,
        canonical_reference=canonical,
    )
    preferences = StudioTTSPreferencesSnapshot(revision=3)
    snapshot = STTSPlaygroundRequest(
        operation_id="clone-audition-op",
        provider_id="audio_cpp",
        model_id="clone-model",
        text="preview text",
        voice_id=None,
        response_format="wav",
        studio_draft=TTSStudioDraftSelection(
            selection=TTSSelectionOverrides(
                provider_id="audio_cpp",
                model_mode="exact",
                model_id="clone-model",
                voice_mode="server_default",
                response_format="wav",
                speed=1.0,
                provider_options={},
            ),
            base_revision=3,
        ),
        studio_preferences=preferences,
        clone_audition=clone,
    )

    artifact = await handler._generate_studio_effective(snapshot, None)

    try:
        assert service.calls[0]["clone_audition"] is clone
        assert "profile_reference_resolver" in service.calls[0]
        assert service.calls[0]["profile_reference_resolver"] is None
        assert artifact.clone_evidence is evidence
        assert "Private transcript" not in repr(artifact)
        projection = STTSPlaygroundResultProjection.from_artifact(artifact)
        assert projection.clone_profile_save_eligible is True
        assert not hasattr(projection, "clone_evidence")
        assert not hasattr(projection, "source_text")
        assert "Private transcript" not in repr(projection)
        assert canonical.sha256 not in repr(projection)
    finally:
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
        recovery_action="retry",
    )
    service = SimpleNamespace(
        synthesize_exact=AsyncMock(side_effect=operation_error),
    )
    app = _DeliveryApp()
    handler = STTSEventHandler(app=app)
    handler._stts_service = service

    await handler.handle_playground_generate(STTSPlaygroundGenerateEvent(_snapshot()))

    assert app.notifications == [
        (
            "TTS generation failed: \\[safe provider message] "
            "Retry from the STTS Playground.",
            "error",
        )
    ]
    assert handler._is_generating is False


@pytest.mark.parametrize(
    ("recovery_action", "expected"),
    (
        ("check_server", "Check the configured audio.cpp server and retry."),
        (
            "configure_server",
            "Open STTS Settings and configure the audio.cpp server.",
        ),
        ("refresh_models", "Refresh models in the STTS Playground and retry."),
        ("edit_request", "Adjust the text or selected options and retry."),
        ("retry", "Retry from the STTS Playground."),
        ("PRIVATE_UNKNOWN_ACTION", "Retry from the STTS Playground."),
    ),
)
def test_native_recovery_identifiers_map_to_fixed_ui_copy(
    recovery_action: str,
    expected: str,
) -> None:
    error = TTSOperationError(
        code="generation_failed",
        message="Safe failure",
        retryable=True,
        operation_id="operation",
        recovery_action=recovery_action,
    )

    copy = STTSEventHandler._generation_error_copy(error)

    assert copy == f"Safe failure {expected}"
    assert "PRIVATE_UNKNOWN_ACTION" not in copy


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
    service = SimpleNamespace(synthesize_exact=AsyncMock(side_effect=failure))
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
    service = SimpleNamespace(synthesize_exact=AsyncMock(side_effect=failure))
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
async def test_studio_generation_uses_current_draft_without_persisting_it() -> None:
    response = _Response(_CountingStream((b"RIFF", b"studio")))
    service = _StudioService(response)
    app = _DeliveryApp()
    handler = STTSEventHandler(app=app)
    handler._stts_service = service
    preferences = StudioTTSPreferencesSnapshot(revision=5)
    draft = TTSStudioDraftSelection(
        selection=TTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="draft/model",
            voice_mode="server_default",
            response_format="wav",
            speed=1.0,
            provider_options={},
        ),
        base_revision=5,
    )
    request = STTSPlaygroundRequest(
        operation_id="studio-operation",
        provider_id="audio_cpp",
        model_id="draft/model",
        text="private Studio text",
        voice_id=None,
        response_format="wav",
        studio_draft=draft,
        studio_preferences=preferences,
    )

    await handler.handle_playground_generate(STTSPlaygroundGenerateEvent(request))

    assert len(service.calls) == 1
    assert service.calls[0]["studio_draft"] is draft
    assert service.calls[0]["studio_preferences"] is preferences
    assert service.calls[0]["text"] == "private Studio text"
    assert response.close_calls == 1
    artifact = handler._current_playground_artifact
    try:
        assert artifact is not None
        assert artifact.path.read_bytes() == b"RIFFstudio"
        assert artifact.model_id == "draft/model"
        assert artifact.requested_selection is not None
        assert artifact.requested_selection.configuration_revision == 9
        assert artifact.profile_save_eligible is True
        assert preferences.revision == 5
    finally:
        await handler.cleanup_tts_resources()


@pytest.mark.asyncio
async def test_studio_generation_attaches_provenance_for_legacy_effective_provider() -> (
    None
):
    """The real Playground dispatch always routes Generate through the Studio-
    effective path (`studio_preferences` is never None once a pane is mounted).
    `_generate_legacy`'s own provenance construction is therefore unreachable
    from the live UI -- this proves the path the UI actually takes attaches
    provenance for a legacy effective provider too, not just audio_cpp."""

    response = _Response(
        _CountingStream((b"ID3", b"mp3-bytes")),
        provider_id="openai",
        model_id="tts-1",
        audio_format="mp3",
        content_type="audio/mpeg",
    )
    service = _StudioService(
        response,
        effective_provider_id="openai",
        effective_model_id="tts-1",
        effective_voice_id="alloy",
        effective_response_format="mp3",
        effective_speed=1.25,
        effective_configuration_revision=7,
    )
    app = _DeliveryApp()
    handler = STTSEventHandler(app=app)
    handler._stts_service = service
    preferences = StudioTTSPreferencesSnapshot(revision=5)
    draft = TTSStudioDraftSelection(
        selection=TTSSelectionOverrides(
            provider_id="openai",
            model_mode="exact",
            model_id="tts-1",
            voice_mode="exact",
            voice_id="alloy",
            response_format="mp3",
            speed=1.25,
            provider_options={},
        ),
        base_revision=5,
    )
    request = STTSPlaygroundRequest(
        operation_id="studio-legacy-operation",
        provider_id="openai",
        model_id="tts-1",
        text="private Studio legacy text",
        voice_id="alloy",
        response_format="mp3",
        speed=1.25,
        studio_draft=draft,
        studio_preferences=preferences,
    )

    await handler.handle_playground_generate(STTSPlaygroundGenerateEvent(request))

    artifact = handler._current_playground_artifact
    try:
        assert artifact is not None
        assert artifact.provider_id == "openai"
        assert artifact.requested_selection == TTSRequestedSelectionSnapshot(
            provider_id="openai",
            model_id="tts-1",
            voice_id="alloy",
            response_format="mp3",
            speed=1.25,
            options={},
            configuration_revision=7,
        )
        assert artifact.profile_save_eligible is True
    finally:
        await handler.cleanup_tts_resources()


@pytest.mark.asyncio
async def test_studio_generation_with_provider_options_is_not_save_eligible() -> None:
    """A generation that used provider options cannot be reproduced by a
    slice-1 profile (options are fixed empty), so `_build_requested_selection`
    must pass the real options and let the type's guard refuse -- and the
    artifact must say WHY, not just go quiet."""

    response = _Response(
        _CountingStream((b"ID3", b"mp3-bytes")),
        provider_id="higgs",
        model_id="higgs-v2",
        audio_format="mp3",
        content_type="audio/mpeg",
    )
    service = _StudioService(
        response,
        effective_provider_id="higgs",
        effective_model_id="higgs-v2",
        effective_voice_id="narrator",
        effective_response_format="mp3",
        effective_speed=1.0,
        effective_configuration_revision=7,
        effective_provider_options={"temperature": 0.8},
    )
    app = _DeliveryApp()
    handler = STTSEventHandler(app=app)
    handler._stts_service = service
    preferences = StudioTTSPreferencesSnapshot(revision=5)
    draft = TTSStudioDraftSelection(
        selection=TTSSelectionOverrides(
            provider_id="higgs",
            model_mode="exact",
            model_id="higgs-v2",
            voice_mode="exact",
            voice_id="narrator",
            response_format="mp3",
            speed=1.0,
            provider_options={"temperature": 0.8},
        ),
        base_revision=5,
    )
    request = STTSPlaygroundRequest(
        operation_id="studio-options-operation",
        provider_id="higgs",
        model_id="higgs-v2",
        text="private Studio options text",
        voice_id="narrator",
        response_format="mp3",
        options={"temperature": 0.8},
        studio_draft=draft,
        studio_preferences=preferences,
    )

    await handler.handle_playground_generate(STTSPlaygroundGenerateEvent(request))

    artifact = handler._current_playground_artifact
    try:
        assert artifact is not None
        assert artifact.path.read_bytes() == b"ID3mp3-bytes"
        assert artifact.requested_selection is None
        assert artifact.profile_save_eligible is False
        assert artifact.profile_save_block_code == "provider_options"
    finally:
        await handler.cleanup_tts_resources()


@pytest.mark.asyncio
async def test_studio_generation_without_provider_options_stays_save_eligible() -> None:
    response = _Response(
        _CountingStream((b"ID3", b"mp3-bytes")),
        provider_id="higgs",
        model_id="higgs-v2",
        audio_format="mp3",
        content_type="audio/mpeg",
    )
    service = _StudioService(
        response,
        effective_provider_id="higgs",
        effective_model_id="higgs-v2",
        effective_voice_id="narrator",
        effective_response_format="mp3",
        effective_speed=1.0,
        effective_configuration_revision=7,
    )
    app = _DeliveryApp()
    handler = STTSEventHandler(app=app)
    handler._stts_service = service
    preferences = StudioTTSPreferencesSnapshot(revision=5)
    draft = TTSStudioDraftSelection(
        selection=TTSSelectionOverrides(
            provider_id="higgs",
            model_mode="exact",
            model_id="higgs-v2",
            voice_mode="exact",
            voice_id="narrator",
            response_format="mp3",
            speed=1.0,
            provider_options={},
        ),
        base_revision=5,
    )
    request = STTSPlaygroundRequest(
        operation_id="studio-no-options-operation",
        provider_id="higgs",
        model_id="higgs-v2",
        text="private Studio text",
        voice_id="narrator",
        response_format="mp3",
        studio_draft=draft,
        studio_preferences=preferences,
    )

    await handler.handle_playground_generate(STTSPlaygroundGenerateEvent(request))

    artifact = handler._current_playground_artifact
    try:
        assert artifact is not None
        assert artifact.requested_selection is not None
        assert artifact.profile_save_eligible is True
        assert artifact.profile_save_block_code is None
    finally:
        await handler.cleanup_tts_resources()


@pytest.mark.asyncio
async def test_studio_generation_survives_invalid_provenance_for_legacy_effective_provider() -> (
    None
):
    """Mirrors `test_legacy_generation_survives_configuration_revision_failure`
    for the path the live UI actually takes: a provenance snapshot that fails
    to construct must degrade the artifact to not-save-eligible, never fail
    the generation that already produced real audio."""

    response = _Response(
        _CountingStream((b"ID3", b"mp3-bytes")),
        provider_id="openai",
        model_id="tts-1",
        audio_format="mp3",
        content_type="audio/mpeg",
    )
    service = _StudioService(
        response,
        effective_provider_id="openai",
        effective_model_id="tts-1",
        effective_voice_id="alloy",
        # Not in PROFILE_PROVIDER_FORMATS["openai"] -- TTSRequestedSelectionSnapshot
        # construction must raise and be swallowed, not propagate.
        effective_response_format="not-a-real-format",
        effective_speed=1.25,
        effective_configuration_revision=7,
    )
    app = _DeliveryApp()
    handler = STTSEventHandler(app=app)
    handler._stts_service = service
    preferences = StudioTTSPreferencesSnapshot(revision=5)
    draft = TTSStudioDraftSelection(
        selection=TTSSelectionOverrides(
            provider_id="openai",
            model_mode="exact",
            model_id="tts-1",
            voice_mode="exact",
            voice_id="alloy",
            response_format="mp3",
            speed=1.25,
            provider_options={},
        ),
        base_revision=5,
    )
    request = STTSPlaygroundRequest(
        operation_id="studio-legacy-degrade-operation",
        provider_id="openai",
        model_id="tts-1",
        text="private Studio legacy degrade text",
        voice_id="alloy",
        response_format="mp3",
        speed=1.25,
        studio_draft=draft,
        studio_preferences=preferences,
    )

    await handler.handle_playground_generate(STTSPlaygroundGenerateEvent(request))

    artifact = handler._current_playground_artifact
    try:
        assert artifact is not None
        assert artifact.provider_id == "openai"
        assert artifact.path.read_bytes() == b"ID3mp3-bytes"
        assert artifact.requested_selection is None
        assert artifact.profile_save_eligible is False
    finally:
        await handler.cleanup_tts_resources()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True], ids=("failure", "cancellation"))
async def test_failed_or_cancelled_studio_generation_never_persists_the_draft(
    monkeypatch: pytest.MonkeyPatch,
    cancel: bool,
) -> None:
    blocked = asyncio.Event() if cancel else None
    response = _Response(
        _CountingStream(
            (),
            blocked=blocked,
            failure=None if cancel else RuntimeError("generation failed"),
        )
    )
    service = _StudioService(response)
    app = _DeliveryApp()
    handler = STTSEventHandler(app=app)
    handler._stts_service = service
    persist = Mock(side_effect=AssertionError("generation persisted Studio draft"))
    monkeypatch.setattr(StudioTTSPreferenceStore, "save", persist)
    preferences = StudioTTSPreferencesSnapshot(revision=6)
    draft = TTSStudioDraftSelection(
        selection=TTSSelectionOverrides(
            provider_id="audio_cpp",
            model_mode="exact",
            model_id="draft/model",
            voice_mode="server_default",
            response_format="wav",
            speed=1.0,
            provider_options={},
        ),
        base_revision=6,
    )
    request = STTSPlaygroundRequest(
        operation_id=f"studio-{'cancel' if cancel else 'failure'}",
        provider_id="audio_cpp",
        model_id="draft/model",
        text="ephemeral draft",
        voice_id=None,
        response_format="wav",
        studio_draft=draft,
        studio_preferences=preferences,
    )

    generation = asyncio.create_task(
        handler.handle_playground_generate(STTSPlaygroundGenerateEvent(request))
    )
    if cancel:
        await asyncio.sleep(0)
        generation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await generation
    else:
        await generation

    persist.assert_not_called()
    assert preferences.revision == 6
    assert response.close_calls == 1


@pytest.mark.asyncio
async def test_repeated_generate_is_rejected_without_replacing_active_work() -> None:
    service = SimpleNamespace(synthesize_exact=AsyncMock())
    app = _DeliveryApp()
    handler = STTSEventHandler(app=app)
    handler._stts_service = service
    handler._is_generating = True

    await handler.handle_playground_generate(STTSPlaygroundGenerateEvent(_snapshot()))

    service.synthesize_exact.assert_not_awaited()
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
async def test_retiring_playground_context_discards_current_artifact() -> None:
    app = _DeliveryApp()
    handler = STTSEventHandler(app=app)
    handler._stts_service = _NativeService(
        _Response(_CountingStream((b"RIFF", b"current")))
    )
    await handler.handle_playground_generate(STTSPlaygroundGenerateEvent(_snapshot()))
    artifact = handler.playground_state().artifact
    assert artifact is not None
    assert artifact.path.exists()

    handler.retire_playground_context()

    state = handler.playground_state()
    assert state.artifact is None
    assert handler._current_audio_file is None
    assert not artifact.path.exists()
    assert handler._playground_audio_files == set()
    assert handler._playground_operation_files == {}


@pytest.mark.asyncio
async def test_retiring_playground_context_fences_in_flight_completion() -> None:
    release = asyncio.Event()
    response = _Response(_CountingStream((b"RIFF",), blocked=release))
    playground = _DeliveryPlayground()
    app = _DeliveryApp(playground)
    handler = STTSEventHandler(app=app)
    handler._stts_service = _NativeService(response)
    handler.start_playground_generation(STTSPlaygroundGenerateEvent(_snapshot()))
    await asyncio.sleep(0)
    generation_task = handler._generation_task
    assert generation_task is not None
    assert handler.playground_state().generation_active is True

    handler.retire_playground_context()
    release.set()
    await asyncio.gather(generation_task, return_exceptions=True)

    state = handler.playground_state()
    assert state.generation_active is False
    assert state.active_operation_id is None
    assert state.artifact is None
    assert playground.completions == []
    assert app.notifications == []
    assert handler._playground_audio_files == set()
    assert handler._playground_operation_files == {}


@pytest.mark.asyncio
async def test_retiring_only_generation_preserves_completed_artifact() -> None:
    app = _DeliveryApp()
    handler = STTSEventHandler(app=app)
    handler._stts_service = _NativeService(
        _Response(_CountingStream((b"RIFF", b"completed")))
    )
    await handler.handle_playground_generate(STTSPlaygroundGenerateEvent(_snapshot()))
    completed = handler.playground_state().artifact
    assert completed is not None

    release = asyncio.Event()
    handler._stts_service = _NativeService(
        _Response(_CountingStream((b"RIFF", b"replacement"), blocked=release))
    )
    replacement = STTSPlaygroundRequest(
        operation_id="replacement-in-flight",
        provider_id="audio_cpp",
        model_id="model-2",
        text="replacement",
        voice_id=None,
        response_format="wav",
    )
    handler.start_playground_generation(STTSPlaygroundGenerateEvent(replacement))
    await asyncio.sleep(0)
    generation_task = handler._generation_task
    assert generation_task is not None

    handler.retire_playground_generation()
    release.set()
    await asyncio.gather(generation_task, return_exceptions=True)

    state = handler.playground_state()
    assert state.artifact == completed
    assert completed.path.exists()
    assert handler._current_audio_file == completed.path

    await handler.cleanup_tts_resources()


@pytest.mark.asyncio
async def test_retiring_queued_generation_before_first_task_turn_fences_it() -> None:
    app = _DeliveryApp()
    handler = STTSEventHandler(app=app)
    service = _NativeService(_Response(_CountingStream((b"RIFF", b"queued"))))
    handler._stts_service = service
    event = STTSPlaygroundGenerateEvent(
        STTSPlaygroundRequest(
            operation_id="queued-profile-generation",
            provider_id="audio_cpp",
            model_id="model-2",
            text="queued",
            voice_id=None,
            response_format="wav",
        )
    )

    handler.start_playground_generation(event)
    generation_task = handler._generation_task
    assert generation_task is not None
    handler.retire_playground_generation("queued-profile-generation")
    await generation_task

    assert service.requests == []
    assert handler.playground_state().artifact is None
    assert handler._playground_audio_files == set()


@pytest.mark.asyncio
async def test_leased_artifact_survives_replacement_until_release() -> None:
    app = _DeliveryApp()
    handler = STTSEventHandler(app=app)
    handler._stts_service = _NativeService(
        _Response(_CountingStream((b"RIFF", b"first")))
    )
    await handler.handle_playground_generate(STTSPlaygroundGenerateEvent(_snapshot()))
    first = handler._current_playground_artifact
    assert first is not None
    assert handler.lease_playground_artifact(first) is True

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
    assert first.path.exists()
    assert first.path in handler._playground_audio_files

    handler.release_playground_artifact(first)

    assert not first.path.exists()
    assert first.path not in handler._playground_audio_files
    assert second.path.exists()

    await handler.cleanup_tts_resources()


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
