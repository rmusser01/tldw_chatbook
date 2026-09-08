"""Real legacy admission and PCM fallback, with only synthesis/device I/O replaced."""

from __future__ import annotations

import asyncio
import io
import struct
import threading
import wave
from pathlib import Path

import pytest

from Tests.TTS_Events.test_spoken_feedback_streaming import _RecordingSink
from tldw_chatbook.Event_Handlers.TTS_Events import tts_events
from tldw_chatbook.TTS import pcm_playback
from tldw_chatbook.TTS.adapter_registry import TTSAdapterRegistry
from tldw_chatbook.TTS.legacy_bridge import legacy_provider_specs
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.TTS_Generation import TTSService

PCM = struct.pack("<hh", 2000, -2000) * 1200
SELECTIONS = {
    "openai": ("tts-1", "alloy"),
    "elevenlabs": ("eleven_multilingual_v2", "21m00Tcm4TlvDq8ikWAM"),
    "kokoro": ("kokoro", "af_alloy"),
    "chatterbox": ("chatterbox", "default"),
    "higgs": ("higgs-audio-v2", "professional_female"),
    "alltalk": ("alltalk", "female_01.wav"),
}


def _service(provider, *, endpoint=None, audio_format="pcm", body=PCM):
    requests = []

    class Backend:
        def set_progress_callback(self, callback):
            pass

        async def generate_speech_stream(self, request):
            requests.append(request)
            assert request.response_format == audio_format
            # A transport split may bisect one PCM frame.
            yield body[:3]
            yield body[3:]

    class Manager:
        async def get_backend(self, internal_id):
            return Backend()

        async def close_all_backends(self):
            pass

    registry = TTSAdapterRegistry(
        specs=legacy_provider_specs(
            {"app_tts": {"OPENAI_BASE_URL": endpoint} if endpoint else {}},
            manager_factory=lambda _provider, _config: Manager(),
        ),
        aliases={},
    )
    model, voice = SELECTIONS[provider]
    service = TTSService(
        registry,
        preferences_snapshot=TTSPreferencesSnapshot(
            provider_id=provider,
            model_mode="exact",
            model_id=model,
            voice_mode="exact",
            voice_id=voice,
            response_format=audio_format,
            speed=1.0,
        ),
    )
    return service, requests


def _assert_complete_wav(path):
    assert path.suffix == ".wav"
    with wave.open(io.BytesIO(path.read_bytes()), "rb") as audio:
        assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (
            24000,
            1,
            2,
        )
        assert audio.getnframes() == 2400
        assert audio.readframes(2400) == PCM


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", SELECTIONS)
async def test_known_legacy_pcm_declares_exact_samples_and_preserves_export(provider):
    service, requests = _service(provider)
    try:
        response = await service.synthesize_default(text="A synthetic PCM reply.")
        try:
            assert response.sample_rate == 24000
            assert dict(response.metadata) == {
                "sample_rate": 24000,
                "channels": 1,
                "sample_encoding": "pcm_s16le",
            }
            assert response.audio_format == "pcm"
            assert response.content_type == "application/octet-stream"
            assert b"".join([part async for part in response.byte_stream]) == PCM
            assert len(requests) == 1
        finally:
            await response.aclose()
    finally:
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("audio_format", ["pcm", "wav"])
async def test_custom_openai_endpoint_does_not_claim_an_unknown_pcm_rate(audio_format):
    service, _requests = _service(
        "openai", endpoint="http://127.0.0.1:18992/v1", audio_format=audio_format
    )
    try:
        response = await service.synthesize_default(text="Custom endpoint.")
        try:
            assert response.sample_rate is None
            assert dict(response.metadata) == {}
        finally:
            await response.aclose()
    finally:
        await service.close()
        await service.wait_closed()


class _Handler(tts_events.TTSEventHandler):
    def __init__(self, service):
        super().__init__()
        self._tts_service = service
        self.messages = []

    async def post_message(self, message):
        self.messages.append(message)

    def notify(self, message, severity="information"):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", SELECTIONS)
async def test_explicit_legacy_pcm_uses_the_existing_sink_without_a_file(
    provider, monkeypatch
):
    service, requests = _service(provider)
    handler = _Handler(service)
    sinks = []

    class Sink(_RecordingSink):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            sinks.append(self)

    monkeypatch.setattr(tts_events, "sink_available", lambda: True)
    monkeypatch.setattr(tts_events, "StreamingPcmSink", Sink)
    try:
        for _turn in range(2):
            await handler._generate_tts("Explicit PCM request.", "pcm-stream", None)
        assert len(requests) == len(sinks) == 2
        assert all(request.response_format == "pcm" for request in requests)
        for sink in sinks:
            assert sink.opened_with == (24000, 1)
            assert b"".join(sink.fed) == PCM
        complete = [
            event
            for event in handler.messages
            if isinstance(event, tts_events.TTSCompleteEvent)
        ]
        assert len(complete) == 2
        assert all(
            event.error is None and event.audio_file is None for event in complete
        )
        assert handler._audio_files == {}
    finally:
        await handler.cleanup_tts_resources()
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", SELECTIONS)
@pytest.mark.parametrize("sink_present", [False, True])
async def test_pcm_file_fallback_is_complete_wav_and_repeated_playable(
    provider, sink_present, monkeypatch
):
    service, requests = _service(provider)
    handler = _Handler(service)
    played = []

    class CannotOpenSink(_RecordingSink):
        _open_should_fail = True

    def play(path):
        _assert_complete_wav(path)
        played.append(path)

    monkeypatch.setattr(tts_events, "sink_available", lambda: sink_present)
    monkeypatch.setattr(tts_events, "StreamingPcmSink", CannotOpenSink)
    monkeypatch.setattr(tts_events, "play_audio_file", play)
    try:
        for turn in range(2):
            await handler._generate_tts("A synthetic PCM reply.", "pcm-turn", None)
            complete = [
                event
                for event in handler.messages
                if isinstance(event, tts_events.TTSCompleteEvent)
            ]
            assert len(complete) == turn + 1
            assert complete[-1].error is None
            path = complete[-1].audio_file
            assert path is not None
            _assert_complete_wav(path)
            await handler.handle_tts_playback(
                tts_events.TTSPlaybackEvent(message_id="pcm-turn", action="play")
            )
            assert len(played) == turn + 1
        assert len(requests) == 2
        assert all(request.response_format == "pcm" for request in requests)
        assert not played[0].exists()
    finally:
        await handler.cleanup_tts_resources()
        await service.close()
        await service.wait_closed()
    assert all(not path.exists() for path in played)


@pytest.mark.asyncio
async def test_unknown_custom_pcm_never_publishes_an_unplayable_console_file(
    monkeypatch,
):
    service, _requests = _service("openai", endpoint="http://127.0.0.1:18992/v1")
    handler = _Handler(service)
    monkeypatch.setattr(tts_events, "sink_available", lambda: False)
    try:
        await handler._generate_tts("Unknown PCM shape.", "unknown", None)
        complete = [
            event
            for event in handler.messages
            if isinstance(event, tts_events.TTSCompleteEvent)
        ]
        assert len(complete) == 1
        assert complete[0].error
        assert complete[0].audio_file is None
        assert handler._audio_files == {}
    finally:
        await handler.cleanup_tts_resources()
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [b"", PCM + b"\x00"])
async def test_malformed_pcm_does_not_complete_as_playable_audio(body, monkeypatch):
    service, _requests = _service("kokoro", body=body)
    handler = _Handler(service)
    monkeypatch.setattr(tts_events, "sink_available", lambda: False)
    try:
        await handler._generate_tts("Invalid PCM frames.", "bad-pcm", None)
        complete = [
            event
            for event in handler.messages
            if isinstance(event, tts_events.TTSCompleteEvent)
        ]
        assert len(complete) == 1
        assert complete[0].error and complete[0].audio_file is None
        assert handler._audio_files == {}
    finally:
        await handler.cleanup_tts_resources()
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_cancelled_late_pcm_copy_cannot_delete_the_next_playback(monkeypatch):
    service, _requests = _service("kokoro")
    handler = _Handler(service)
    original_copy = tts_events.create_pcm16_wav_copy
    entered = threading.Event()
    allow_finish = threading.Event()
    first_copy = []
    copies = []

    def held_copy(*args):
        path = original_copy(*args)
        copies.append(path)
        if not first_copy:
            first_copy.append(path)
            entered.set()
            assert allow_finish.wait(5)
        return path

    monkeypatch.setattr(tts_events, "create_pcm16_wav_copy", held_copy)
    monkeypatch.setattr(tts_events, "sink_available", lambda: False)
    monkeypatch.setattr(tts_events, "_TTS_IO_CANCELLATION_JOIN_TIMEOUT_SECONDS", 0.01)
    task = asyncio.create_task(handler._generate_tts("Old PCM.", "same-owner", None))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await handler._generate_tts("Next PCM.", "same-owner", None)
        current = handler._audio_files["same-owner"]
        _assert_complete_wav(current)
        allow_finish.set()
        await handler._drain_retained_tts_artifact_work()
        assert not first_copy[0].exists()
        assert handler._audio_files["same-owner"] == current
        _assert_complete_wav(current)
    finally:
        allow_finish.set()
        await handler.cleanup_tts_resources()
        await service.close()
        await service.wait_closed()
    assert all(not path.exists() for path in copies)


@pytest.mark.parametrize(
    "rate,channels",
    [(None, 1), (True, 1), (0, 1), (24000.0, 1), (24000, 0), (24000, 3)],
)
def test_pcm_playback_copy_refuses_unknown_or_invalid_sample_metadata(
    rate, channels, tmp_path
):
    source = tmp_path / "source.pcm"
    source.write_bytes(PCM)
    with pytest.raises(ValueError, match="known sample rate"):
        pcm_playback.create_pcm16_wav_copy(source, rate, channels)
    assert source.read_bytes() == PCM


def test_pcm_playback_copy_is_complete_stereo_and_preserves_source(tmp_path):
    source = tmp_path / "stereo.pcm"
    source.write_bytes(PCM * 20)
    copied = pcm_playback.create_pcm16_wav_copy(source, 48000, 2)
    try:
        with wave.open(str(copied), "rb") as audio:
            assert audio.getframerate() == 48000
            assert audio.getnchannels() == 2
            assert audio.getnframes() == len(PCM) * 20 // 4
            assert audio.readframes(audio.getnframes()) == PCM * 20
        assert source.read_bytes() == PCM * 20
        assert copied.stat().st_mode & 0o777 == 0o600
    finally:
        copied.unlink()


def test_pcm_copy_writer_failure_removes_only_its_partial_copy(tmp_path, monkeypatch):
    source = tmp_path / "source.pcm"
    source.write_bytes(PCM * 20)
    destinations = []
    create = pcm_playback.create_secure_temp_file

    def track(*args, **kwargs):
        result = create(*args, **kwargs, dir=str(tmp_path))
        destinations.append(result)
        return result

    write = wave.Wave_write.writeframesraw

    def broken_write(self, chunk):
        write(self, chunk)
        raise OSError("Synthetic local copy failure")

    monkeypatch.setattr(pcm_playback, "create_secure_temp_file", track)
    monkeypatch.setattr(wave.Wave_write, "writeframesraw", broken_write)
    with pytest.raises(OSError, match="Synthetic"):
        pcm_playback.create_pcm16_wav_copy(source, 24000)
    assert destinations and all(not Path(path).exists() for path in destinations)
    assert source.read_bytes() == PCM * 20
