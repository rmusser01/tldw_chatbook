from __future__ import annotations

import asyncio
import stat
import threading
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Event_Handlers.TTS_Events import tts_events as tts_events_module
from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
    TTSCompleteEvent,
    TTSEventHandler,
    TTSMessageSpeechRequestEvent,
    TTSProgressEvent,
    TTSRequestEvent,
    TTSStreamingEvent,
)
from tldw_chatbook.TTS.adapter_registry import TTSAdapterRegistry
from tldw_chatbook.TTS.adapter_types import (
    ProgressSink,
    TTSAudioResponse,
    TTSProviderCatalog,
    TTSProviderDescriptor,
    TTSProviderSpec,
    TTSRequest,
)
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.playground_types import TTSRequestedSelectionSnapshot
from tldw_chatbook.TTS.profile_service import LoadedCharacterTTSAssignment
from tldw_chatbook.TTS.profile_types import (
    AssignedTTSProfileSnapshot,
    CharacterRef,
    CharacterTTSAssignment,
    TTSGenerationProfile,
    TTSProfileDraft,
)
from tldw_chatbook.TTS.TTS_Generation import TTSService

_WAIT_SECONDS = 1.0
_WAV_CHUNKS = (
    b"RIFF",
    b"\x24\x00\x00\x00WAVEfmt ",
    b"\x10\x00\x00\x00\x01\x00\x01\x00",
    b"\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00",
)


class _RecordingStream:
    def __init__(
        self,
        chunks: tuple[bytes, ...],
        timeline: list[str],
        *,
        failure: BaseException | None = None,
        blocked: asyncio.Event | None = None,
        started: asyncio.Event | None = None,
    ) -> None:
        self._chunks = chunks
        self._timeline = timeline
        self._failure = failure
        self._blocked = blocked
        self._started = started
        self.close_calls = 0

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk
        if self._started is not None:
            self._started.set()
        if self._blocked is not None:
            await self._blocked.wait()
        if self._failure is not None:
            raise self._failure
        self._timeline.append("stream-drained")

    async def aclose(self) -> None:
        self.close_calls += 1


class _Response:
    def __init__(
        self,
        stream: _RecordingStream,
        *,
        provider_id: str = "audio_cpp",
        model_id: str = "<Opaque:Model>",
        audio_format: str = "wav",
    ) -> None:
        self.provider_id = provider_id
        self.model_id = model_id
        self.audio_format = audio_format
        self.content_type = "audio/wav"
        self.byte_stream = stream
        self.metadata: Mapping[str, str | int | float | bool | None] = {}
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1
        await self.byte_stream.aclose()


class _CapturingAdapter:
    def __init__(self, timeline: list[str]) -> None:
        self.requests: list[TTSRequest] = []
        self.timeline = timeline
        self.response_close_calls = 0

    async def ensure_ready(self) -> None:
        return

    async def get_catalog(self, refresh: bool = False) -> TTSProviderCatalog:
        del refresh
        raise AssertionError("exact Console preferences must not require a catalog")

    async def get_voices(
        self,
        model_id: str,
        refresh: bool = False,
    ) -> tuple[str, ...]:
        del model_id, refresh
        return ()

    async def synthesize(
        self,
        request: TTSRequest,
        progress_sink: ProgressSink | None = None,
    ) -> TTSAudioResponse:
        self.requests.append(request)
        if progress_sink is not None:
            from tldw_chatbook.TTS.adapter_types import TTSProgress

            await progress_sink(TTSProgress(status="Generating", fraction=0.5))

        async def cleanup() -> None:
            self.response_close_calls += 1

        stream = _RecordingStream(_WAV_CHUNKS, self.timeline)
        return TTSAudioResponse(
            provider_id="audio_cpp",
            model_id=request.model_id,
            audio_format="wav",
            content_type="audio/wav",
            byte_stream=stream,
            cleanup=cleanup,
        )

    async def close(self) -> None:
        return


class _Handler(TTSEventHandler):
    def __init__(
        self,
        timeline: list[str] | None = None,
        profile_service_loader=None,
    ) -> None:
        super().__init__(profile_service_loader=profile_service_loader)
        self.messages: list[object] = []
        self.completion_posted = asyncio.Event()
        self.timeline = timeline

    async def post_message(self, message: object) -> None:
        self.messages.append(message)
        if isinstance(message, TTSCompleteEvent):
            if self.timeline is not None:
                self.timeline.append("completion-posted")
            self.completion_posted.set()


class _RecordingTempManager:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.paths: list[Path] = []
        self.suffixes: list[str] = []

    def create_temp_file(
        self,
        content: str | bytes,
        suffix: str = "",
        prefix: str = "tmp",
        dir: str | None = None,
    ) -> str:
        del dir
        path = self.directory / f"{prefix}{len(self.paths)}{suffix}"
        path.write_bytes(content.encode() if isinstance(content, str) else content)
        path.chmod(0o600)
        self.paths.append(path)
        self.suffixes.append(suffix)
        return str(path)


class _HeartbeatBlocker:
    """Record whether the asyncio loop runs while one sync phase is blocked."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self.entered = threading.Event()
        self.heartbeat = threading.Event()
        self.release = threading.Event()
        self.observed_heartbeat: bool | None = None
        self._coordinator = threading.Thread(target=self._coordinate)

    def start(self) -> None:
        self._coordinator.start()

    def block(self) -> None:
        self.entered.set()
        assert self.release.wait(timeout=1.0)
        self.observed_heartbeat = self.heartbeat.is_set()

    def join(self) -> None:
        self._coordinator.join(timeout=1.0)
        assert not self._coordinator.is_alive()

    def _coordinate(self) -> None:
        assert self.entered.wait(timeout=1.0)
        self._loop.call_soon_threadsafe(self._heartbeat)
        if not self.heartbeat.wait(timeout=0.2):
            # Unstick the pre-fix synchronous path without leaking a test thread.
            self.release.set()

    def _heartbeat(self) -> None:
        self.heartbeat.set()
        self.release.set()


class _BlockingFile:
    def __init__(
        self,
        wrapped: Any,
        blocker: _HeartbeatBlocker,
        phase: str,
    ) -> None:
        self._wrapped = wrapped
        self._blocker = blocker
        self._phase = phase

    def __enter__(self) -> _BlockingFile:
        self._wrapped.__enter__()
        return self

    def __exit__(self, *exc_info: object) -> object:
        if self._phase == "close":
            self._blocker.block()
        return self._wrapped.__exit__(*exc_info)

    def write(self, content: bytes) -> int:
        if self._phase == "write":
            self._blocker.block()
        return self._wrapped.write(content)

    def flush(self) -> None:
        if self._phase == "flush":
            self._blocker.block()
        self._wrapped.flush()


class _DefaultService:
    def __init__(
        self,
        response: _Response,
        *,
        snapshot_provider_id: str | None = None,
    ) -> None:
        self.response = response
        self.snapshot_provider_id = (
            response.provider_id
            if snapshot_provider_id is None
            else snapshot_provider_id
        )
        self.calls: list[tuple[str, str | None, object]] = []
        self.exact_calls: list[tuple[TTSRequest, object]] = []
        self.exact_error: BaseException | None = None

    def preferences_snapshot(self) -> SimpleNamespace:
        return SimpleNamespace(provider_id=self.snapshot_provider_id)

    async def synthesize_default(
        self,
        *,
        text: str,
        voice_override: str | None = None,
        progress_sink: object = None,
    ) -> _Response:
        self.calls.append((text, voice_override, progress_sink))
        return self.response

    async def synthesize_exact(
        self,
        request: TTSRequest,
        progress_sink: object = None,
    ) -> tuple[_Response, TTSRequestedSelectionSnapshot]:
        self.exact_calls.append((request, progress_sink))
        if self.exact_error is not None:
            raise self.exact_error
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
        raise AssertionError("audio.cpp must not use the legacy stream bridge")
        yield b""  # pragma: no cover


def _external_preferences() -> dict[str, Any]:
    return {
        "app_tts": {
            "default_provider": "audio_cpp",
            "default_model_mode": "exact",
            "default_model": "<Opaque:Model>",
            "default_voice_mode": "exact",
            "default_voice": "[Voice]",
            "default_format": "wav",
            "default_speed": 1.0,
            "audio_cpp": {
                "mode": "external",
                "base_url": "http://127.0.0.1:8080",
            },
        }
    }


class _StaticProfileService:
    def __init__(self, result: LoadedCharacterTTSAssignment) -> None:
        self.result = result
        self.calls: list[CharacterRef] = []

    async def get_assigned_profile(
        self,
        character_ref: CharacterRef,
    ) -> LoadedCharacterTTSAssignment:
        self.calls.append(character_ref)
        return self.result


def _assigned_profile(character_ref: CharacterRef) -> LoadedCharacterTTSAssignment:
    draft = TTSProfileDraft(
        display_name="Mira voice",
        provider_id="audio_cpp",
        model_id="assigned-model",
        voice_id="assigned-voice",
        response_format="wav",
        speed=1.0,
        options={},
    )
    profile = TTSGenerationProfile(
        profile_id=UUID("11111111-1111-4111-8111-111111111111"),
        display_name=draft.display_name,
        normalized_name=draft.normalized_name,
        provider_id=draft.provider_id,
        model_id=draft.model_id,
        voice_id=draft.voice_id,
        response_format=draft.response_format,
        speed=draft.speed,
        options=draft.options,
        revision=5,
        created_at=datetime(2026, 7, 31, tzinfo=UTC),
        updated_at=datetime(2026, 7, 31, tzinfo=UTC),
    )
    return LoadedCharacterTTSAssignment(
        repository_generation=3,
        snapshot=AssignedTTSProfileSnapshot(
            assignment=CharacterTTSAssignment(
                character_ref=character_ref,
                profile_id=profile.profile_id,
            ),
            profile=profile,
        ),
    )


@pytest.mark.asyncio
async def test_console_audio_cpp_snapshot_uses_native_default_without_rewriting_ids() -> (
    None
):
    timeline: list[str] = []
    adapter = _CapturingAdapter(timeline)
    registry = TTSAdapterRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor(
                    provider_id="audio_cpp",
                    display_name="audio.cpp",
                    native=True,
                ),
                factory=lambda _config: adapter,
                initial_config=_external_preferences()["app_tts"]["audio_cpp"],
                exclusive_reconfigure=True,
            ),
        ),
        aliases={},
    )
    service = TTSService(
        registry,
        preferences_snapshot=TTSPreferencesSnapshot.from_settings(
            _external_preferences()
        ),
    )
    legacy_calls = 0

    async def reject_legacy(
        *_args: object,
        **_kwargs: object,
    ) -> AsyncIterator[bytes]:
        nonlocal legacy_calls
        legacy_calls += 1
        raise AssertionError("audio.cpp must not use generate_audio_stream")
        yield b""  # pragma: no cover

    service.generate_audio_stream = reject_legacy  # type: ignore[method-assign]
    profile_service = _StaticProfileService(
        LoadedCharacterTTSAssignment(
            repository_generation=3,
            snapshot=None,
        )
    )

    async def load_profile_service() -> _StaticProfileService:
        return profile_service

    handler = _Handler(timeline, load_profile_service)
    handler._request_cooldown = {}
    handler._tts_service = service
    store = ConsoleChatStore()
    session = store.create_session(
        runtime_backend="local",
        assistant_kind="character",
        assistant_id="7",
        assistant_authority_id="local-authority",
        character_id=7,
        character_name="Mira",
    )
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Character response",
    )
    artifact: Path | None = None
    try:
        await handler.handle_tts_request(
            TTSMessageSpeechRequestEvent(
                store.issue_tts_message_speech_snapshot(message.id),
                store.validate_tts_message_speech_snapshot,
            )
        )
        await asyncio.wait_for(
            handler.completion_posted.wait(),
            timeout=_WAIT_SECONDS,
        )
        completion = next(
            message
            for message in handler.messages
            if isinstance(message, TTSCompleteEvent)
        )
        artifact = completion.audio_file

        assert completion.message_id == message.id
        assert adapter.requests == [
            TTSRequest(
                provider_id="audio_cpp",
                model_id="<Opaque:Model>",
                text="Character response",
                voice="[Voice]",
                response_format="wav",
                speed=1.0,
                options={},
            )
        ]
        assert profile_service.calls == [
            CharacterRef(
                source="local",
                authority_id="local-authority",
                character_id="7",
            )
        ]
        assert legacy_calls == 0
        assert adapter.response_close_calls == 1
        assert artifact is not None
        assert artifact.suffix == ".wav"
        assert artifact.read_bytes() == b"".join(_WAV_CHUNKS)
        assert stat.S_IMODE(artifact.stat().st_mode) == 0o600
        assert timeline == ["stream-drained", "completion-posted"]
        progress = [
            message
            for message in handler.messages
            if isinstance(message, TTSProgressEvent)
        ]
        assert progress[0].progress == 0.0
        assert progress[-1].progress == 1.0
        assert not any(
            isinstance(message, TTSStreamingEvent) for message in handler.messages
        )
    finally:
        await asyncio.sleep(0)
        await handler.cleanup_tts_resources()
        await service.close()
        await service.wait_closed()

    assert artifact is not None
    assert not artifact.exists()


@pytest.mark.asyncio
async def test_assigned_console_snapshot_uses_exact_profile_and_complete_wav(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeline: list[str] = []
    response = _Response(
        _RecordingStream(_WAV_CHUNKS, timeline),
        model_id="assigned-model",
    )
    service = _DefaultService(response)
    metric_calls: list[tuple[str, dict[str, Any]]] = []

    def capture_counter(
        name: str,
        value: int = 1,
        labels: dict[str, Any] | None = None,
    ) -> None:
        del value
        metric_calls.append((name, dict(labels or {})))

    def capture_histogram(
        name: str,
        value: float,
        labels: dict[str, Any] | None = None,
    ) -> None:
        del value
        metric_calls.append((name, dict(labels or {})))

    monkeypatch.setattr(
        "tldw_chatbook.Metrics.metrics_logger.log_counter",
        capture_counter,
    )
    monkeypatch.setattr(
        "tldw_chatbook.Metrics.metrics_logger.log_histogram",
        capture_histogram,
    )
    store = ConsoleChatStore()
    session = store.create_session(
        runtime_backend="local",
        assistant_kind="character",
        assistant_id="7",
        assistant_authority_id="local-authority",
        character_id=7,
        character_name="Mira",
    )
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Assigned character response",
    )
    snapshot = store.issue_tts_message_speech_snapshot(message.id)
    assert snapshot.character_ref is not None
    profile_service = _StaticProfileService(_assigned_profile(snapshot.character_ref))

    async def load_profile_service() -> _StaticProfileService:
        return profile_service

    handler = _Handler(profile_service_loader=load_profile_service)
    handler._tts_service = service
    handler._temp_manager = _RecordingTempManager(tmp_path)
    artifact: Path | None = None
    try:
        await handler.handle_tts_request(
            TTSMessageSpeechRequestEvent(
                snapshot,
                store.validate_tts_message_speech_snapshot,
            )
        )
        await asyncio.wait_for(
            handler.completion_posted.wait(),
            timeout=_WAIT_SECONDS,
        )
        completion = next(
            event for event in handler.messages if isinstance(event, TTSCompleteEvent)
        )
        artifact = completion.audio_file

        assert completion.error is None
        assert service.calls == []
        assert len(service.exact_calls) == 1
        request, progress_sink = service.exact_calls[0]
        assert request == TTSRequest(
            provider_id="audio_cpp",
            model_id="assigned-model",
            text="Assigned character response",
            voice="assigned-voice",
            response_format="wav",
            speed=1.0,
            options={},
        )
        assert callable(progress_sink)
        assert profile_service.calls == [snapshot.character_ref]
        assert artifact is not None
        assert artifact.suffix == ".wav"
        assert artifact.read_bytes() == b"".join(_WAV_CHUNKS)
        assert response.close_calls == 1
        assert response.byte_stream.close_calls == 1
        assert [labels for _name, labels in metric_calls] == [
            {
                "provider_id": "audio_cpp",
                "resolution_source": "assigned",
                "outcome_code": "success",
            },
            {
                "provider_id": "audio_cpp",
                "resolution_source": "assigned",
                "outcome_code": "success",
            },
        ]
    finally:
        await handler.cleanup_tts_resources()

    assert artifact is not None
    assert not artifact.exists()


@pytest.mark.asyncio
async def test_assigned_exact_failure_never_calls_global_or_offers_fallback(
    tmp_path: Path,
) -> None:
    response = _Response(_RecordingStream(_WAV_CHUNKS, []))
    service = _DefaultService(response)
    service.exact_error = RuntimeError(
        "https://user:credential@example.test/private/path"
    )
    store = ConsoleChatStore()
    session = store.create_session(
        runtime_backend="local",
        assistant_kind="character",
        assistant_id="7",
        assistant_authority_id="local-authority",
        character_id=7,
    )
    message = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="Assigned character response",
    )
    snapshot = store.issue_tts_message_speech_snapshot(message.id)
    assert snapshot.character_ref is not None
    profile_service = _StaticProfileService(_assigned_profile(snapshot.character_ref))

    async def load_profile_service() -> _StaticProfileService:
        return profile_service

    handler = _Handler(profile_service_loader=load_profile_service)
    handler._tts_service = service
    handler._temp_manager = _RecordingTempManager(tmp_path)
    try:
        await handler.handle_tts_request(
            TTSMessageSpeechRequestEvent(
                snapshot,
                store.validate_tts_message_speech_snapshot,
            )
        )
        await asyncio.wait_for(
            handler.completion_posted.wait(),
            timeout=_WAIT_SECONDS,
        )
        completion = next(
            event for event in handler.messages if isinstance(event, TTSCompleteEvent)
        )

        assert completion.audio_file is None
        assert completion.error
        assert "credential" not in completion.error
        assert completion.global_override_token is None
        assert len(service.exact_calls) == 1
        assert service.calls == []
    finally:
        await handler.cleanup_tts_resources()


@pytest.mark.asyncio
async def test_non_console_tts_request_event_keeps_global_default_path(
    tmp_path: Path,
) -> None:
    timeline: list[str] = []
    response = _Response(_RecordingStream(_WAV_CHUNKS, timeline))
    service = _DefaultService(response)
    handler = _Handler()
    handler._tts_service = service
    handler._temp_manager = _RecordingTempManager(tmp_path)
    artifact: Path | None = None
    try:
        await handler.handle_tts_request(
            TTSRequestEvent(
                text="Global caller response",
                message_id="global-native-1",
                voice="Global voice override",
            )
        )
        await asyncio.wait_for(
            handler.completion_posted.wait(),
            timeout=_WAIT_SECONDS,
        )
        completion = next(
            message
            for message in handler.messages
            if isinstance(message, TTSCompleteEvent)
        )
        artifact = completion.audio_file

        assert len(service.calls) == 1
        text, voice, progress_sink = service.calls[0]
        assert text == "Global caller response"
        assert voice == "Global voice override"
        assert callable(progress_sink)
        assert artifact is not None
        assert artifact.read_bytes() == b"".join(_WAV_CHUNKS)
        assert response.close_calls == 1
        assert response.byte_stream.close_calls == 1
    finally:
        await handler.cleanup_tts_resources()

    assert artifact is not None
    assert not artifact.exists()


@pytest.mark.asyncio
async def test_console_accepts_admitted_response_when_provider_snapshot_is_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeline: list[str] = []
    response = _Response(_RecordingStream(_WAV_CHUNKS, timeline))
    service = _DefaultService(response, snapshot_provider_id="openai")
    handler = _Handler()
    handler._tts_service = service
    handler._temp_manager = _RecordingTempManager(tmp_path)
    metric_calls: list[tuple[str, dict[str, Any]]] = []

    def capture_counter(
        name: str,
        value: int = 1,
        labels: dict[str, Any] | None = None,
    ) -> None:
        del value
        metric_calls.append((name, dict(labels or {})))

    def capture_histogram(
        name: str,
        value: float,
        labels: dict[str, Any] | None = None,
    ) -> None:
        del value
        metric_calls.append((name, dict(labels or {})))

    monkeypatch.setattr(
        "tldw_chatbook.Metrics.metrics_logger.log_counter",
        capture_counter,
    )
    monkeypatch.setattr(
        "tldw_chatbook.Metrics.metrics_logger.log_histogram",
        capture_histogram,
    )
    artifact: Path | None = None
    try:
        await handler._generate_tts(
            "Character response",
            "console-stale-provider-snapshot",
            None,
        )
        completion = next(
            message
            for message in handler.messages
            if isinstance(message, TTSCompleteEvent)
        )
        artifact = completion.audio_file

        assert completion.error is None
        assert artifact is not None
        assert artifact.read_bytes() == b"".join(_WAV_CHUNKS)
        assert response.close_calls == 1
        assert response.byte_stream.close_calls == 1
        assert metric_calls == [
            (
                "tts_generation_total",
                {
                    "provider_id": "audio_cpp",
                    "resolution_source": "global",
                    "outcome_code": "success",
                },
            ),
            (
                "tts_generation_latency_seconds",
                {
                    "provider_id": "audio_cpp",
                    "resolution_source": "global",
                    "outcome_code": "success",
                },
            ),
        ]
    finally:
        await handler.cleanup_tts_resources()

    assert artifact is not None
    assert not artifact.exists()


@pytest.mark.asyncio
async def test_console_stream_failure_deletes_partial_artifact_and_closes_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_error = RuntimeError("PRIVATE_UPSTREAM_STREAM_FAILURE")
    timeline: list[str] = []
    response = _Response(
        _RecordingStream(
            _WAV_CHUNKS[:1],
            timeline,
            failure=private_error,
        )
    )
    handler = _Handler()
    handler._tts_service = _DefaultService(response)
    temp_manager = _RecordingTempManager(tmp_path)
    handler._temp_manager = temp_manager
    deleted: list[Path] = []

    def delete(path: str | Path) -> bool:
        candidate = Path(path)
        deleted.append(candidate)
        if candidate.exists():
            candidate.write_bytes(b"\x00" * candidate.stat().st_size)
            candidate.unlink()
        return True

    monkeypatch.setattr(tts_events_module, "secure_delete_file", delete)

    await handler._generate_tts(
        "Character response",
        "console-native-failure",
        None,
    )

    assert len(temp_manager.paths) == 1
    assert deleted == temp_manager.paths
    assert not temp_manager.paths[0].exists()
    assert handler._audio_files == {}
    assert response.close_calls == 1
    assert response.byte_stream.close_calls == 1
    completions = [
        message for message in handler.messages if isinstance(message, TTSCompleteEvent)
    ]
    assert len(completions) == 1
    assert completions[0].audio_file is None
    assert completions[0].error
    assert "PRIVATE_UPSTREAM_STREAM_FAILURE" not in completions[0].error
    assert not any(
        isinstance(message, TTSStreamingEvent) for message in handler.messages
    )


@pytest.mark.asyncio
async def test_console_cancellation_deletes_partial_artifact_and_closes_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeline: list[str] = []
    started = asyncio.Event()
    never_release = asyncio.Event()
    response = _Response(
        _RecordingStream(
            _WAV_CHUNKS[:1],
            timeline,
            blocked=never_release,
            started=started,
        )
    )
    handler = _Handler()
    handler._tts_service = _DefaultService(response)
    temp_manager = _RecordingTempManager(tmp_path)
    handler._temp_manager = temp_manager
    deleted: list[Path] = []

    def delete(path: str | Path) -> bool:
        candidate = Path(path)
        deleted.append(candidate)
        if candidate.exists():
            candidate.write_bytes(b"\x00" * candidate.stat().st_size)
            candidate.unlink()
        return True

    monkeypatch.setattr(tts_events_module, "secure_delete_file", delete)
    generation = asyncio.create_task(
        handler._generate_tts(
            "Character response",
            "console-native-cancel",
            None,
        )
    )
    await asyncio.wait_for(started.wait(), timeout=_WAIT_SECONDS)
    generation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await generation

    assert len(temp_manager.paths) == 1
    assert deleted == temp_manager.paths
    assert not temp_manager.paths[0].exists()
    assert handler._audio_files == {}
    assert response.close_calls == 1
    assert response.byte_stream.close_calls == 1
    assert not any(
        isinstance(message, TTSCompleteEvent) for message in handler.messages
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("create", "write", "flush", "close"))
async def test_console_artifact_io_keeps_event_loop_responsive(
    phase: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = asyncio.get_running_loop()
    blocker = _HeartbeatBlocker(loop)
    timeline: list[str] = []
    response = _Response(_RecordingStream(_WAV_CHUNKS[:1], timeline))
    handler = _Handler()
    handler._tts_service = _DefaultService(response)

    class TempManager(_RecordingTempManager):
        def create_temp_file(
            self,
            content: str | bytes,
            suffix: str = "",
            prefix: str = "tmp",
            dir: str | None = None,
        ) -> str:
            if phase == "create":
                blocker.block()
            return super().create_temp_file(content, suffix, prefix, dir)

    temp_manager = TempManager(tmp_path)
    handler._temp_manager = temp_manager
    original_open = Path.open

    if phase != "create":

        def blocking_open(path: Path, *args: object, **kwargs: object) -> Any:
            wrapped = original_open(path, *args, **kwargs)
            if path.parent == tmp_path and path.name.startswith("tts_audio_"):
                return _BlockingFile(wrapped, blocker, phase)
            return wrapped

        monkeypatch.setattr(Path, "open", blocking_open)

    blocker.start()
    try:
        await handler._generate_tts(
            "Character response",
            f"console-native-{phase}",
            None,
        )
    finally:
        blocker.join()

    assert blocker.observed_heartbeat is True
    assert response.close_calls == 1
    assert any(
        isinstance(message, TTSCompleteEvent) and message.audio_file is not None
        for message in handler.messages
    )
    await handler.cleanup_tts_resources()


@pytest.mark.asyncio
async def test_console_batches_small_stream_chunks_into_one_artifact_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeline: list[str] = []
    response = _Response(_RecordingStream(_WAV_CHUNKS, timeline))
    handler = _Handler()
    handler._tts_service = _DefaultService(response)
    handler._temp_manager = _RecordingTempManager(tmp_path)
    written_batches: list[bytes] = []
    append_chunk = handler._append_tts_artifact_chunk

    def capture_batch(artifact_path: Path, content: bytes) -> None:
        written_batches.append(content)
        append_chunk(artifact_path, content)

    monkeypatch.setattr(handler, "_append_tts_artifact_chunk", capture_batch)
    try:
        await handler._generate_tts(
            "Character response",
            "console-native-batched-write",
            None,
        )

        assert written_batches == [b"".join(_WAV_CHUNKS)]
        completion = next(
            message
            for message in handler.messages
            if isinstance(message, TTSCompleteEvent)
        )
        assert completion.audio_file is not None
        assert completion.audio_file.read_bytes() == b"".join(_WAV_CHUNKS)
    finally:
        await handler.cleanup_tts_resources()


@pytest.mark.asyncio
async def test_console_artifact_batches_preserve_order_across_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = (b"abc", b"def", b"ghi")
    timeline: list[str] = []
    response = _Response(_RecordingStream(chunks, timeline))
    handler = _Handler()
    handler._tts_service = _DefaultService(response)
    handler._temp_manager = _RecordingTempManager(tmp_path)
    written_batches: list[bytes] = []
    append_chunk = handler._append_tts_artifact_chunk

    def capture_batch(artifact_path: Path, content: bytes) -> None:
        written_batches.append(content)
        append_chunk(artifact_path, content)

    monkeypatch.setattr(handler, "_append_tts_artifact_chunk", capture_batch)
    monkeypatch.setattr(
        tts_events_module,
        "_TTS_ARTIFACT_WRITE_BATCH_BYTES",
        5,
    )
    try:
        await handler._generate_tts(
            "Character response",
            "console-native-threshold-batches",
            None,
        )

        assert written_batches == [b"abcdef", b"ghi"]
        completion = next(
            message
            for message in handler.messages
            if isinstance(message, TTSCompleteEvent)
        )
        assert completion.audio_file is not None
        assert completion.audio_file.read_bytes() == b"".join(chunks)
    finally:
        await handler.cleanup_tts_resources()


@pytest.mark.asyncio
async def test_console_cancellation_joins_blocking_artifact_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks_before = asyncio.all_tasks()
    entered = threading.Event()
    release = threading.Event()
    worker_exited = threading.Event()
    fallback_used = threading.Event()
    timeline: list[str] = []
    response = _Response(_RecordingStream(_WAV_CHUNKS[:1], timeline))
    handler = _Handler()
    handler._tts_service = _DefaultService(response)
    temp_manager = _RecordingTempManager(tmp_path)
    handler._temp_manager = temp_manager
    original_open = Path.open

    class BlockingWriteFile:
        def __init__(self, wrapped: Any) -> None:
            self._wrapped = wrapped

        def __enter__(self) -> BlockingWriteFile:
            self._wrapped.__enter__()
            return self

        def __exit__(self, *exc_info: object) -> object:
            return self._wrapped.__exit__(*exc_info)

        def write(self, content: bytes) -> int:
            entered.set()
            try:
                assert release.wait(timeout=1.0)
                return self._wrapped.write(content)
            finally:
                worker_exited.set()

        def flush(self) -> None:
            self._wrapped.flush()

    def blocking_open(path: Path, *args: object, **kwargs: object) -> Any:
        wrapped = original_open(path, *args, **kwargs)
        if path.parent == tmp_path and path.name.startswith("tts_audio_"):
            return BlockingWriteFile(wrapped)
        return wrapped

    def unblock_synchronous_red_path() -> None:
        assert entered.wait(timeout=1.0)
        if not release.wait(timeout=0.2):
            fallback_used.set()
            release.set()

    monkeypatch.setattr(Path, "open", blocking_open)
    watchdog = threading.Thread(target=unblock_synchronous_red_path)
    watchdog.start()
    generation = asyncio.create_task(
        handler._generate_tts(
            "Character response",
            "console-native-worker-cancel",
            None,
        )
    )

    async def wait_for_thread_entry() -> None:
        while not entered.is_set():
            await asyncio.sleep(0)

    await asyncio.wait_for(wait_for_thread_entry(), timeout=1.0)
    generation.cancel()
    await asyncio.sleep(0)
    assert not generation.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await generation
    watchdog.join(timeout=1.0)

    assert not watchdog.is_alive()
    assert not fallback_used.is_set()
    assert worker_exited.is_set()
    assert response.close_calls == 1
    assert handler._audio_files == {}
    assert handler._artifact_cleanup_retry == set()
    assert len(temp_manager.paths) == 1
    assert not temp_manager.paths[0].exists()
    assert asyncio.all_tasks() - tasks_before == set()


@pytest.mark.asyncio
@pytest.mark.parametrize("late_failure", (False, True))
async def test_console_cancellation_timeout_returns_before_stalled_write_and_retries_cleanup(
    late_failure: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    worker_exited = threading.Event()
    timeline: list[str] = []
    response = _Response(_RecordingStream(_WAV_CHUNKS[:1], timeline))
    handler = _Handler()
    handler._tts_service = _DefaultService(response)
    original_open = Path.open

    class DirectTempManager(_RecordingTempManager):
        def create_temp_file(
            self,
            content: str | bytes,
            suffix: str = "",
            prefix: str = "tmp",
            dir: str | None = None,
        ) -> str:
            del dir
            path = self.directory / f"{prefix}{len(self.paths)}{suffix}"
            payload = content.encode() if isinstance(content, str) else content
            with original_open(path, "wb") as artifact:
                artifact.write(payload)
                artifact.flush()
            path.chmod(0o600)
            self.paths.append(path)
            self.suffixes.append(suffix)
            return str(path)

    temp_manager = DirectTempManager(tmp_path)
    handler._temp_manager = temp_manager

    class StalledWriteFile:
        def __init__(self, wrapped: Any) -> None:
            self._wrapped = wrapped

        def __enter__(self) -> StalledWriteFile:
            self._wrapped.__enter__()
            return self

        def __exit__(self, *exc_info: object) -> object:
            return self._wrapped.__exit__(*exc_info)

        def write(self, content: bytes) -> int:
            entered.set()
            try:
                assert release.wait(timeout=2.0)
                if late_failure:
                    raise OSError("PRIVATE_LATE_WRITE_FAILURE")
                return self._wrapped.write(content)
            finally:
                worker_exited.set()

        def flush(self) -> None:
            self._wrapped.flush()

    def blocking_open(path: Path, *args: object, **kwargs: object) -> Any:
        wrapped = original_open(path, *args, **kwargs)
        if path.parent == tmp_path and path.name.startswith("tts_audio_"):
            return StalledWriteFile(wrapped)
        return wrapped

    def delete_after_writer(candidate: str | Path) -> bool:
        path = Path(candidate)
        if not worker_exited.is_set():
            return False
        path.unlink(missing_ok=True)
        return True

    async def wait_for_thread_event(event: threading.Event) -> None:
        while not event.is_set():
            await asyncio.sleep(0)

    async def wait_for_late_cleanup(path: Path) -> None:
        while (
            path.exists()
            or handler._artifact_cleanup_retry
            or handler._retained_tts_io_tasks
            or handler._retained_tts_cleanup_tasks
        ):
            await asyncio.sleep(0)

    monkeypatch.setattr(Path, "open", blocking_open)
    monkeypatch.setattr(tts_events_module, "secure_delete_file", delete_after_writer)
    monkeypatch.setattr(
        tts_events_module,
        "_TTS_IO_CANCELLATION_JOIN_TIMEOUT_SECONDS",
        0.01,
        raising=False,
    )
    generation = asyncio.create_task(
        handler._generate_tts(
            "Character response",
            "console-native-stalled-write",
            None,
        )
    )

    try:
        await asyncio.wait_for(wait_for_thread_event(entered), timeout=1.0)
        generation.cancel("caller shutdown")
        done, _pending = await asyncio.wait({generation}, timeout=0.2)
        assert generation in done
        with pytest.raises(asyncio.CancelledError, match="caller shutdown"):
            await generation

        assert len(temp_manager.paths) == 1
        artifact = temp_manager.paths[0]
        assert artifact in handler._artifact_cleanup_retry
        assert artifact.exists()

        release.set()
        await asyncio.wait_for(wait_for_thread_event(worker_exited), timeout=1.0)
        await asyncio.wait_for(wait_for_late_cleanup(artifact), timeout=1.0)
        assert not artifact.exists()
    finally:
        release.set()
        await asyncio.gather(generation, return_exceptions=True)
        await handler.cleanup_tts_resources()


@pytest.mark.asyncio
async def test_console_cancellation_does_not_wait_for_stalled_secure_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream_started = asyncio.Event()
    hold_stream = asyncio.Event()
    delete_entered = threading.Event()
    delete_release = threading.Event()
    timeline: list[str] = []
    response = _Response(
        _RecordingStream(
            _WAV_CHUNKS[:1],
            timeline,
            blocked=hold_stream,
            started=stream_started,
        )
    )
    handler = _Handler()
    handler._tts_service = _DefaultService(response)
    temp_manager = _RecordingTempManager(tmp_path)
    handler._temp_manager = temp_manager

    def stalled_delete(candidate: str | Path) -> bool:
        path = Path(candidate)
        delete_entered.set()
        assert delete_release.wait(timeout=2.0)
        path.unlink(missing_ok=True)
        return True

    async def wait_for_thread_event(event: threading.Event) -> None:
        while not event.is_set():
            await asyncio.sleep(0)

    async def wait_for_cleanup(path: Path) -> None:
        while (
            path.exists()
            or handler._artifact_cleanup_retry
            or handler._retained_tts_io_tasks
            or handler._retained_tts_cleanup_tasks
        ):
            await asyncio.sleep(0)

    monkeypatch.setattr(tts_events_module, "secure_delete_file", stalled_delete)
    monkeypatch.setattr(
        tts_events_module,
        "_TTS_SECURE_DELETE_TIMEOUT_SECONDS",
        0.01,
        raising=False,
    )
    generation = asyncio.create_task(
        handler._generate_tts(
            "Character response",
            "console-native-stalled-delete",
            None,
        )
    )

    try:
        await asyncio.wait_for(stream_started.wait(), timeout=1.0)
        generation.cancel("caller shutdown")
        await asyncio.wait_for(wait_for_thread_event(delete_entered), timeout=1.0)
        done, _pending = await asyncio.wait({generation}, timeout=0.2)
        assert generation in done
        with pytest.raises(asyncio.CancelledError, match="caller shutdown"):
            await generation

        assert len(temp_manager.paths) == 1
        artifact = temp_manager.paths[0]
        assert artifact in handler._artifact_cleanup_retry

        delete_release.set()
        await asyncio.wait_for(wait_for_cleanup(artifact), timeout=1.0)
        assert not artifact.exists()
    finally:
        delete_release.set()
        hold_stream.set()
        await asyncio.gather(generation, return_exceptions=True)
        await handler.cleanup_tts_resources()


@pytest.mark.asyncio
async def test_console_late_cancelled_creation_is_eventually_deleted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    timeline: list[str] = []
    response = _Response(_RecordingStream(_WAV_CHUNKS[:1], timeline))
    handler = _Handler()
    handler._tts_service = _DefaultService(response)

    class StalledTempManager(_RecordingTempManager):
        def create_temp_file(
            self,
            content: str | bytes,
            suffix: str = "",
            prefix: str = "tmp",
            dir: str | None = None,
        ) -> str:
            entered.set()
            assert release.wait(timeout=2.0)
            return super().create_temp_file(content, suffix, prefix, dir)

    temp_manager = StalledTempManager(tmp_path)
    handler._temp_manager = temp_manager
    deleted: list[Path] = []

    def delete(candidate: str | Path) -> bool:
        path = Path(candidate)
        deleted.append(path)
        path.unlink(missing_ok=True)
        return True

    async def wait_for_thread_event(event: threading.Event) -> None:
        while not event.is_set():
            await asyncio.sleep(0)

    async def wait_for_late_cleanup() -> None:
        while (
            not temp_manager.paths
            or temp_manager.paths[0].exists()
            or handler._artifact_cleanup_retry
            or handler._retained_tts_io_tasks
            or handler._retained_tts_cleanup_tasks
        ):
            await asyncio.sleep(0)

    monkeypatch.setattr(tts_events_module, "secure_delete_file", delete)
    monkeypatch.setattr(
        tts_events_module,
        "_TTS_IO_CANCELLATION_JOIN_TIMEOUT_SECONDS",
        0.01,
        raising=False,
    )
    generation = asyncio.create_task(
        handler._generate_tts(
            "Character response",
            "console-native-stalled-create",
            None,
        )
    )

    try:
        await asyncio.wait_for(wait_for_thread_event(entered), timeout=1.0)
        generation.cancel()
        done, _pending = await asyncio.wait({generation}, timeout=0.2)
        assert generation in done
        with pytest.raises(asyncio.CancelledError):
            await generation
        assert temp_manager.paths == []

        cleanup = asyncio.create_task(handler.cleanup_tts_resources())
        await asyncio.sleep(0)
        assert not cleanup.done()
        release.set()
        await asyncio.wait_for(cleanup, timeout=1.0)
        assert deleted == temp_manager.paths
        assert not temp_manager.paths[0].exists()
        assert handler._retained_tts_io_tasks == set()
        assert handler._retained_tts_cleanup_tasks == set()
    finally:
        release.set()
        await asyncio.gather(generation, return_exceptions=True)
        await asyncio.wait_for(wait_for_late_cleanup(), timeout=1.0)
        await handler.cleanup_tts_resources()


@pytest.mark.asyncio
async def test_console_cancellation_wins_over_joined_artifact_worker_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks_before = asyncio.all_tasks()
    cancel_reason = "PRIVATE_CANCEL_REASON"
    io_failure = "PRIVATE_IO_FAILURE"
    entered = threading.Event()
    release = threading.Event()
    worker_exited = threading.Event()
    fallback_used = threading.Event()
    file_exit_calls = 0
    timeline: list[str] = []
    metric_calls: list[tuple[str, str, float | int, dict[str, Any]]] = []
    log_messages: list[str] = []
    response = _Response(_RecordingStream(_WAV_CHUNKS[:1], timeline))
    handler = _Handler()
    handler._tts_service = _DefaultService(response)
    original_open = Path.open

    class SuccessfulTempManager(_RecordingTempManager):
        def create_temp_file(
            self,
            content: str | bytes,
            suffix: str = "",
            prefix: str = "tmp",
            dir: str | None = None,
        ) -> str:
            del dir
            path = self.directory / f"{prefix}{len(self.paths)}{suffix}"
            payload = content.encode() if isinstance(content, str) else content
            with original_open(path, "wb") as artifact:
                artifact.write(payload)
                artifact.flush()
            path.chmod(0o600)
            self.paths.append(path)
            self.suffixes.append(suffix)
            return str(path)

    class FailingWriteFile:
        def __init__(self, wrapped: Any) -> None:
            self._wrapped = wrapped

        def __enter__(self) -> FailingWriteFile:
            self._wrapped.__enter__()
            return self

        def __exit__(self, *exc_info: object) -> object:
            nonlocal file_exit_calls
            file_exit_calls += 1
            return self._wrapped.__exit__(*exc_info)

        def write(self, content: bytes) -> int:
            del content
            entered.set()
            try:
                assert release.wait(timeout=1.0)
                raise OSError(io_failure)
            finally:
                worker_exited.set()

        def flush(self) -> None:
            self._wrapped.flush()

    def blocking_open(path: Path, *args: object, **kwargs: object) -> Any:
        wrapped = original_open(path, *args, **kwargs)
        if path.parent == tmp_path and path.name.startswith("tts_audio_"):
            return FailingWriteFile(wrapped)
        return wrapped

    def unblock_synchronous_regression() -> None:
        assert entered.wait(timeout=1.0)
        if not release.wait(timeout=0.2):
            fallback_used.set()
            release.set()

    deleted: list[Path] = []

    def delete(path: str | Path) -> bool:
        candidate = Path(path)
        deleted.append(candidate)
        candidate.unlink()
        return True

    def capture_counter(
        name: str,
        value: int = 1,
        labels: dict[str, Any] | None = None,
    ) -> None:
        metric_calls.append((name, "counter", value, dict(labels or {})))

    def capture_histogram(
        name: str,
        value: float,
        labels: dict[str, Any] | None = None,
    ) -> None:
        metric_calls.append((name, "histogram", value, dict(labels or {})))

    temp_manager = SuccessfulTempManager(tmp_path)
    handler._temp_manager = temp_manager
    monkeypatch.setattr(Path, "open", blocking_open)
    monkeypatch.setattr(tts_events_module, "secure_delete_file", delete)
    monkeypatch.setattr(
        "tldw_chatbook.Metrics.metrics_logger.log_counter",
        capture_counter,
    )
    monkeypatch.setattr(
        "tldw_chatbook.Metrics.metrics_logger.log_histogram",
        capture_histogram,
    )
    sink_id = tts_events_module.logger.add(
        log_messages.append,
        level="DEBUG",
        format="{message}",
    )
    watchdog = threading.Thread(target=unblock_synchronous_regression)
    watchdog.start()
    generation = asyncio.create_task(
        handler._generate_tts(
            "Character response",
            "console-native-worker-error-cancel",
            None,
        )
    )

    async def wait_for_thread_entry() -> None:
        while not entered.is_set():
            await asyncio.sleep(0)

    try:
        await asyncio.wait_for(wait_for_thread_entry(), timeout=1.0)
        generation.cancel(cancel_reason)
        await asyncio.sleep(0)
        assert not generation.done()
        release.set()
        with pytest.raises(asyncio.CancelledError) as cancellation:
            await generation
    finally:
        release.set()
        watchdog.join(timeout=1.0)
        tts_events_module.logger.remove(sink_id)

    assert cancellation.value.args == (cancel_reason,)
    assert not watchdog.is_alive()
    assert not fallback_used.is_set()
    assert worker_exited.is_set()
    assert file_exit_calls == 1
    assert response.close_calls == 1
    assert response.byte_stream.close_calls == 1
    assert len(temp_manager.paths) == 1
    assert deleted == temp_manager.paths
    assert not temp_manager.paths[0].exists()
    assert handler._audio_files == {}
    assert handler._artifact_cleanup_retry == set()
    assert not any(
        isinstance(message, (TTSCompleteEvent, TTSStreamingEvent))
        for message in handler.messages
    )
    assert asyncio.all_tasks() - tasks_before == set()
    assert len(metric_calls) == 2
    for _name, _kind, _value, labels in metric_calls:
        assert labels == {
            "provider_id": "audio_cpp",
            "resolution_source": "global",
            "outcome_code": "cancelled",
        }
    rendered = repr(metric_calls) + "\n".join(log_messages) + repr(handler.messages)
    for private_value in (cancel_reason, io_failure, str(temp_manager.paths[0])):
        assert private_value not in rendered


@pytest.mark.asyncio
async def test_console_secure_delete_keeps_event_loop_responsive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "partial.wav"
    path.write_bytes(b"partial audio")
    handler = _Handler()
    handler._audio_files["message"] = path
    blocker = _HeartbeatBlocker(asyncio.get_running_loop())

    def blocking_delete(candidate: str | Path) -> bool:
        assert Path(candidate) == path
        blocker.block()
        path.unlink()
        return True

    monkeypatch.setattr(tts_events_module, "secure_delete_file", blocking_delete)
    blocker.start()
    try:
        await handler._discard_tts_artifact("message", path)
    finally:
        blocker.join()

    assert blocker.observed_heartbeat is True
    assert not path.exists()
    assert handler._audio_files == {}


@pytest.mark.parametrize("replace_during_delete", [False, True])
@pytest.mark.asyncio
async def test_console_late_secure_delete_success_releases_only_matching_cache_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replace_during_delete: bool,
) -> None:
    path = tmp_path / "late-delete.wav"
    replacement = tmp_path / "replacement.wav"
    path.write_bytes(b"audio")
    replacement.write_bytes(b"replacement")
    handler = _Handler()
    handler._audio_files["message"] = path
    entered = threading.Event()
    release = threading.Event()

    def delayed_delete(candidate: str | Path) -> bool:
        assert Path(candidate) == path
        entered.set()
        assert release.wait(timeout=2.0)
        path.unlink()
        return True

    async def wait_for_thread_event(event: threading.Event) -> None:
        while not event.is_set():
            await asyncio.sleep(0)

    async def wait_for_late_delete() -> None:
        while (
            path.exists()
            or handler._retained_tts_io_tasks
            or handler._retained_tts_cleanup_tasks
        ):
            await asyncio.sleep(0)

    monkeypatch.setattr(tts_events_module, "secure_delete_file", delayed_delete)
    monkeypatch.setattr(
        tts_events_module,
        "_TTS_SECURE_DELETE_TIMEOUT_SECONDS",
        0.01,
        raising=False,
    )

    try:
        await handler._cleanup_audio_file("message")
        await asyncio.wait_for(wait_for_thread_event(entered), timeout=1.0)
        assert handler._audio_files == {"message": path}

        if replace_during_delete:
            handler._audio_files["message"] = replacement

        release.set()
        await asyncio.wait_for(wait_for_late_delete(), timeout=1.0)
    finally:
        release.set()
        replacement.unlink(missing_ok=True)

    expected = {"message": replacement} if replace_during_delete else {}
    assert handler._audio_files == expected
    assert not path.exists()


@pytest.mark.asyncio
async def test_console_failed_secure_delete_retains_retry_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "partial-private-path.wav"
    path.write_bytes(b"partial audio")
    handler = _Handler()
    handler._audio_files["message"] = path
    delete_results = iter((False, True))
    log_messages: list[str] = []

    def delete(candidate: str | Path) -> bool:
        assert Path(candidate) == path
        result = next(delete_results)
        if result:
            path.unlink()
        return result

    monkeypatch.setattr(tts_events_module, "secure_delete_file", delete)
    sink_id = tts_events_module.logger.add(
        log_messages.append,
        level="DEBUG",
        format="{message}",
    )
    try:
        await handler._discard_tts_artifact("message", path)
        assert handler._audio_files == {}
        assert handler._artifact_cleanup_retry == {path}
        assert path.exists()

        await handler.cleanup_tts_resources()
    finally:
        tts_events_module.logger.remove(sink_id)

    assert handler._artifact_cleanup_retry == set()
    assert not path.exists()
    rendered = "\n".join(log_messages)
    assert "Incomplete TTS artifact cleanup will be retried" in rendered
    assert str(path) not in rendered
