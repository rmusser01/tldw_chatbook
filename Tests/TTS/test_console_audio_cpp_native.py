from __future__ import annotations

import asyncio
import stat
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_chatbook.Event_Handlers.TTS_Events import tts_events as tts_events_module
from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
    TTSCompleteEvent,
    TTSEventHandler,
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
    def __init__(self, timeline: list[str] | None = None) -> None:
        super().__init__()
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


class _DefaultService:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.calls: list[tuple[str, str | None, object]] = []

    def preferences_snapshot(self) -> SimpleNamespace:
        return SimpleNamespace(provider_id=self.response.provider_id)

    async def synthesize_default(
        self,
        *,
        text: str,
        voice_override: str | None = None,
        progress_sink: object = None,
    ) -> _Response:
        self.calls.append((text, voice_override, progress_sink))
        return self.response

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


@pytest.mark.asyncio
async def test_console_audio_cpp_request_uses_native_default_without_rewriting_ids() -> (
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
    handler = _Handler(timeline)
    handler._request_cooldown = {}
    handler._tts_service = service
    artifact: Path | None = None
    try:
        await handler.handle_tts_request(
            TTSRequestEvent(
                text="Character response",
                message_id="console-native-1",
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
