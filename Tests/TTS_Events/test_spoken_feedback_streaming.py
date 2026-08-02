"""Task 4 (streaming-pcm-sink plan): Console spoken feedback streams through
the sink.

`_generate_tts` (`Event_Handlers/TTS_Events/tts_events.py`) is the shared
generation worker body every TTS request runs through -- Console spoken
feedback (`chat_screen.py`'s `_speak_status`, via `TTSRequestEvent`) is one
caller among several (character speech snapshots, ad-hoc requests), and the
streaming seam this module wires in branches on the RESPONSE alone (format /
rate / bytes), never on which caller asked -- see the plan's Global
Constraints. These tests drive `_generate_tts` directly, the same harness
pattern `Tests/TTS/test_tts_improvements.py`'s `test_progress_events` and
`test_console_retained_provider_defaults_use_legacy_adapter` already use: a
fake `TTSAudioResponse`-shaped object plus a fake service exposing
`synthesize_default`/`preferences_snapshot`, with no `Tests/TTS_Events/`
harness pre-existing to build on (this is the first file in the directory).

`StreamingPcmSink` itself is monkeypatched, in the `tts_events` module's own
namespace, with `_RecordingSink` -- a minimal fake implementing just enough
of the real class's interface (`open`/`feed`/`close`/`stop`/`state`/
`terminal_reason`/`bytes_per_second`/`buffered_seconds`) for the REAL
`pump()` (`Audio/streaming_sink.py`, untouched here) to drive it
synchronously, with no real audio device or background thread. Patching
`streaming_sink._import_sounddevice` would not be enough on its own --
`open()` still needs a `stream_factory`-shaped success/failure outcome to
react to, which is exactly what patching the class provides.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

from tldw_chatbook.Event_Handlers.TTS_Events import tts_events as tts_events_module
from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
    TTSCompleteEvent,
    TTSEventHandler,
    TTSPlaybackEvent,
    TTSProgressEvent,
)
from tldw_chatbook.TTS.adapter_types import TTSAudioResponse
from Tests.TTS.test_pcm_stream_plan import _wav_with_trailing_chunk

RATE = 24000


# ---------------------------------------------------------------------------
# `_LIVE_SINK` is a process-global shared across the whole test session (see
# `Tests/Audio/test_streaming_sink.py`'s identical fixture) -- force-clear
# before AND after every test in this file too, so a sink another test left
# registered can never leak in here, and nothing here leaks forward either.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _reset_live_sink_registry():
    import tldw_chatbook.Audio.streaming_sink as streaming_sink_module

    def _force_clear() -> None:
        live = streaming_sink_module._LIVE_SINK
        if live is not None:
            try:
                live.stop()
            except Exception:
                pass
        with streaming_sink_module._LIVE_SINK_LOCK:
            streaming_sink_module._LIVE_SINK = None

    _force_clear()
    yield
    _force_clear()


def test_fake_response_mirrors_the_real_ttsaudioresponse_field_names():
    """Pin against the fake drifting from the real class's shape (brief
    requirement): every attribute `_FakeResponse` exposes below must also
    exist on a real `TTSAudioResponse`.
    """

    async def _empty() -> AsyncIterator[bytes]:
        return
        yield b""  # pragma: no cover -- makes this an async generator

    real = TTSAudioResponse(
        provider_id="openai",
        model_id="tts-1",
        audio_format="pcm",
        content_type="audio/pcm",
        byte_stream=_empty(),
        sample_rate=RATE,
    )
    for field in ("provider_id", "model_id", "audio_format", "content_type",
                  "byte_stream", "sample_rate", "metadata"):
        assert hasattr(real, field), f"_FakeResponse's {field!r} is not a real field"


class _FakeResponse:
    """Minimal stand-in for `TTSAudioResponse` (adapter_types.py:352):
    same field names (asserted against the real class above), driven by a
    plain list of chunks instead of a live provider connection.
    """

    def __init__(
        self,
        chunks: list[bytes],
        *,
        audio_format: str,
        sample_rate: int | None = None,
        provider_id: str = "openai",
        model_id: str = "tts-1",
        content_type: str = "application/octet-stream",
    ) -> None:
        self.provider_id = provider_id
        self.model_id = model_id
        self.audio_format = audio_format
        self.content_type = content_type
        self.sample_rate = sample_rate
        self.metadata: dict = {}
        self._chunks = list(chunks)
        self.byte_stream = self._make_stream()
        self.close_calls = 0

    async def _make_stream(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self.close_calls += 1


class _FakeService:
    """Mirrors the two `TTSService` entry points `_generate_tts`'s default
    (non-exact) path calls: `preferences_snapshot()` and
    `synthesize_default()`. No `synthesize_exact`/legacy machinery -- every
    test here drives the plain adhoc/spoken-feedback path (`resolution=None`),
    matching `_speak_status`'s own `TTSRequestEvent(text=text)`.
    """

    def __init__(self, response: _FakeResponse, *, provider_id: str = "openai") -> None:
        self._response = response
        self._provider_id = provider_id
        self.synthesize_default_calls: list[tuple[str, str | None]] = []

    def preferences_snapshot(self):
        return SimpleNamespace(provider_id=self._provider_id)

    async def synthesize_default(self, *, text, voice_override=None, progress_sink=None):
        self.synthesize_default_calls.append((text, voice_override))
        return self._response


class _RecordingSink:
    """Fake `StreamingPcmSink` (Audio/streaming_sink.py) driven by the REAL
    `pump()` -- implements only the surface `pump()` and `_generate_tts`
    actually touch (`open`, `feed`, `close`, `stop`, `state`,
    `terminal_reason`, `bytes_per_second`, `buffered_seconds`), with no
    device/thread of its own so tests stay synchronous and deterministic.
    """

    #: Subclassed per-test (`_open_should_fail = True`) to simulate an
    #: `open()` failure without touching any real audio backend.
    _open_should_fail = False

    def __init__(self, *, on_event, blocksize_ms: int = 20, stream_factory=None) -> None:
        self.on_event = on_event
        self.opened_with: tuple[int, int] | None = None
        self.state = "idle"
        self.terminal_reason: str | None = None
        self.fail_reason: str | None = None
        self.fed: list[bytes] = []

    def open(self, sample_rate: int, channels: int = 1) -> None:
        self.opened_with = (sample_rate, channels)
        if self._open_should_fail:
            self.state = "failed"
            self.terminal_reason = "failed"
            self.fail_reason = "test-forced open failure"
            return
        self.state = "open"

    def feed(self, pcm: bytes) -> bool:
        if self.state not in ("open", "draining"):
            return False
        self.fed.append(pcm)
        return True

    def close(self) -> None:
        if self.state != "open":
            return
        self.state = "stopped"
        self.terminal_reason = "drained"

    def stop(self) -> None:
        if self.terminal_reason is None:
            self.terminal_reason = "stopped"
        self.state = "stopped"

    @property
    def bytes_per_second(self) -> int:
        rate, channels = self.opened_with or (RATE, 1)
        return rate * 2 * channels

    @property
    def buffered_seconds(self) -> float:
        return 0.0


@pytest.fixture
def handler():
    class _Handler(TTSEventHandler):
        def __init__(self) -> None:
            super().__init__()
            self.messages: list = []

        async def post_message(self, message) -> None:
            self.messages.append(message)

        def notify(self, message, severity: str = "info") -> None:
            pass

    return _Handler()


def _forbid_legacy_artifact_creation(handler_instance) -> None:
    """Pin against the streaming branch ALSO falling through to the legacy
    file-write path (the double-audio mutation the brief calls out):
    `_create_tts_artifact` must never run for a response the sink consumed.
    """

    def _fail(*_args, **_kwargs):
        raise AssertionError(
            "legacy artifact creation must not run for a streaming-eligible response"
        )

    handler_instance._create_tts_artifact = _fail


def _spy_sink_class(holder: dict) -> type[_RecordingSink]:
    class _Sink(_RecordingSink):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            holder["sink"] = self

    return _Sink


# ---------------------------------------------------------------------------
# (a) pcm response streams through a fake sink -- no file playback.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pcm_response_streams_through_the_sink_with_no_file_playback(handler, monkeypatch):
    chunks = [bytes([1, 0]) * 50, bytes([2, 0]) * 50]
    response = _FakeResponse(chunks, audio_format="pcm", sample_rate=RATE)
    service = _FakeService(response)
    handler._tts_service = service
    _forbid_legacy_artifact_creation(handler)

    sink_holder: dict = {}
    monkeypatch.setattr(tts_events_module, "StreamingPcmSink", _spy_sink_class(sink_holder))
    monkeypatch.setattr(tts_events_module, "sink_available", lambda: True)

    await handler._generate_tts("Capture ended.", "adhoc", None)

    sink = sink_holder["sink"]
    assert sink.opened_with == (RATE, 1)
    assert b"".join(sink.fed) == b"".join(chunks)
    assert handler._audio_files == {}, "no artifact should ever be tracked for a streamed response"

    complete_events = [m for m in handler.messages if isinstance(m, TTSCompleteEvent)]
    assert len(complete_events) == 1
    assert complete_events[0].audio_file is None
    assert complete_events[0].error is None
    progress_events = [m for m in handler.messages if isinstance(m, TTSProgressEvent)]
    assert progress_events[-1].progress == 1.0
    assert response.close_calls == 1


# ---------------------------------------------------------------------------
# (b) mp3 response -> legacy `play_audio_file` path, no sink touched.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mp3_response_uses_the_legacy_path_and_never_constructs_a_sink(handler, monkeypatch):
    chunks = [b"ID3", b"restofmp3bytes"]
    response = _FakeResponse(chunks, audio_format="mp3", sample_rate=None)
    service = _FakeService(response)
    handler._tts_service = service

    sink_holder: dict = {}
    monkeypatch.setattr(tts_events_module, "StreamingPcmSink", _spy_sink_class(sink_holder))
    # Sink IS available -- eligibility must still be refused for a
    # compressed format regardless, proving the branch keys off the
    # response's own declared format, not just availability.
    monkeypatch.setattr(tts_events_module, "sink_available", lambda: True)

    try:
        await handler._generate_tts("Discarded.", "adhoc", None)

        assert sink_holder == {}, "a compressed format must never even construct a sink"
        complete_events = [m for m in handler.messages if isinstance(m, TTSCompleteEvent)]
        assert len(complete_events) == 1
        assert complete_events[0].error is None
        artifact_path = complete_events[0].audio_file
        assert artifact_path is not None and artifact_path.exists()
        assert artifact_path.read_bytes() == b"".join(chunks)
    finally:
        await handler.cleanup_tts_resources()


# ---------------------------------------------------------------------------
# (c) sink open-failure -> legacy path used, exactly one completion posted.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sink_open_failure_falls_through_to_the_legacy_path(handler, monkeypatch):
    chunks = [bytes([3, 0]) * 40, bytes([4, 0]) * 40]
    response = _FakeResponse(chunks, audio_format="pcm", sample_rate=RATE)
    service = _FakeService(response)
    handler._tts_service = service

    class _FailingSink(_RecordingSink):
        _open_should_fail = True

    monkeypatch.setattr(tts_events_module, "StreamingPcmSink", _FailingSink)
    monkeypatch.setattr(tts_events_module, "sink_available", lambda: True)

    try:
        await handler._generate_tts("Still responding.", "adhoc", None)

        complete_events = [m for m in handler.messages if isinstance(m, TTSCompleteEvent)]
        assert len(complete_events) == 1, (
            "a sink open failure must fall through silently -- exactly one "
            "completion (the legacy path's own) may ever surface, no phantom "
            "second toast for the failed open() itself"
        )
        assert complete_events[0].error is None
        artifact_path = complete_events[0].audio_file
        assert artifact_path is not None and artifact_path.exists()
        assert artifact_path.read_bytes() == b"".join(chunks)
    finally:
        await handler.cleanup_tts_resources()


# ---------------------------------------------------------------------------
# (d) TTSPlaybackEvent("stop") stops the live sink, for both the
# message-scoped and the global/bare stop.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("message_id", [None, "adhoc"])
async def test_stop_action_stops_whatever_sink_is_currently_registered_live(
    handler, message_id,
):
    import tldw_chatbook.Audio.streaming_sink as streaming_sink_module

    stop_calls: list[bool] = []

    class _FakeLiveSink:
        def stop(self) -> None:
            stop_calls.append(True)

    with streaming_sink_module._LIVE_SINK_LOCK:
        streaming_sink_module._LIVE_SINK = _FakeLiveSink()

    await handler.handle_tts_playback(TTSPlaybackEvent(action="stop", message_id=message_id))

    assert stop_calls == [True]


# ---------------------------------------------------------------------------
# (f) a WAV plan with a trailing chunk -- pumped bytes stop at data's end
# (the review-contracted `data_bytes` pin).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_wav_response_with_a_trailing_chunk_stops_pumped_bytes_at_datas_end(
    handler, monkeypatch,
):
    """WAV eligibility can only be decided from the COMPLETE body (see
    `pcm_stream.sink_plan`'s docstring), so -- unlike pcm -- the wav half of
    this seam decides AFTER the unchanged legacy write loop has already
    written the response to a temp artifact (verified separately: deciding
    BEFORE writing, by eagerly draining the response up front, reproduced
    four real regressions in `test_console_audio_cpp_native.py`'s
    cancellation/partial-artifact/batching pins -- eager draining changes
    observable timing for every wav response whenever a sink is merely
    available, regardless of the eventual eligibility verdict). An eligible
    response is then played through the sink from the bytes already
    collected, and the now-redundant artifact is deleted -- this test pins
    both: the artifact existed and then stopped existing, AND the bytes fed
    to the sink stop exactly at `data`'s end despite the trailing chunk.
    """
    data = bytes(range(200))
    body = _wav_with_trailing_chunk(data=data)
    # Split mid-trailer, across two response chunks -- the harder case: a
    # `max_bytes` bound that only trimmed the FIRST oversized chunk (without
    # also refusing to read further ones) would still leak trailer bytes
    # fed from a later chunk.
    split = len(body) - 5
    response = _FakeResponse([body[:split], body[split:]], audio_format="wav", sample_rate=None)
    service = _FakeService(response)
    handler._tts_service = service

    created_paths: list = []
    original_create_artifact = handler._create_tts_artifact

    def _spy_create_artifact(audio_format):
        path = original_create_artifact(audio_format)
        created_paths.append(path)
        return path

    handler._create_tts_artifact = _spy_create_artifact

    sink_holder: dict = {}
    monkeypatch.setattr(tts_events_module, "StreamingPcmSink", _spy_sink_class(sink_holder))
    monkeypatch.setattr(tts_events_module, "sink_available", lambda: True)

    await handler._generate_tts("Should be spoken.", "adhoc", None)

    sink = sink_holder["sink"]
    fed = b"".join(sink.fed)
    assert fed == data, "pumped bytes must stop exactly at the data chunk's end"
    assert b"LIST" not in fed
    assert b"INFOtest" not in fed

    assert len(created_paths) == 1, "the legacy write loop must still run, unmodified, for wav"
    assert not created_paths[0].exists(), (
        "the now-redundant artifact must be deleted once it was played live"
    )
    assert handler._audio_files == {}
    complete_events = [m for m in handler.messages if isinstance(m, TTSCompleteEvent)]
    assert len(complete_events) == 1
    assert complete_events[0].audio_file is None
    assert complete_events[0].error is None


# ---------------------------------------------------------------------------
# A WAV sink failure (open OR mid-stream) has a complete, valid,
# already-written artifact to fall back to -- unlike pcm, it must silently
# use it rather than surface a user-facing error for an opportunistic
# upgrade that simply didn't pan out.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_wav_sink_failure_falls_back_to_the_already_written_file_silently(
    handler, monkeypatch,
):
    data = bytes(range(64))
    body = _wav_with_trailing_chunk(data=data)
    response = _FakeResponse([body], audio_format="wav", sample_rate=None)
    service = _FakeService(response)
    handler._tts_service = service

    class _FailingSink(_RecordingSink):
        _open_should_fail = True

    monkeypatch.setattr(tts_events_module, "StreamingPcmSink", _FailingSink)
    monkeypatch.setattr(tts_events_module, "sink_available", lambda: True)

    try:
        await handler._generate_tts("Discarded.", "adhoc", None)

        complete_events = [m for m in handler.messages if isinstance(m, TTSCompleteEvent)]
        assert len(complete_events) == 1, "no phantom error toast for a silent fallback"
        assert complete_events[0].error is None
        artifact_path = complete_events[0].audio_file
        assert artifact_path is not None and artifact_path.exists()
        assert artifact_path.read_bytes() == body
    finally:
        await handler.cleanup_tts_resources()


# ---------------------------------------------------------------------------
# Fallback pins: sink unavailable at all -> byte-identical legacy behavior,
# even for an otherwise-eligible pcm response.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sink_unavailable_leaves_an_eligible_pcm_response_on_the_legacy_path(
    handler, monkeypatch,
):
    chunks = [bytes([5, 0]) * 30]
    response = _FakeResponse(chunks, audio_format="pcm", sample_rate=RATE)
    service = _FakeService(response)
    handler._tts_service = service

    sink_holder: dict = {}
    monkeypatch.setattr(tts_events_module, "StreamingPcmSink", _spy_sink_class(sink_holder))
    monkeypatch.setattr(tts_events_module, "sink_available", lambda: False)

    try:
        await handler._generate_tts("Nothing to read yet.", "adhoc", None)

        assert sink_holder == {}
        complete_events = [m for m in handler.messages if isinstance(m, TTSCompleteEvent)]
        assert len(complete_events) == 1
        artifact_path = complete_events[0].audio_file
        assert artifact_path is not None and artifact_path.read_bytes() == chunks[0]
    finally:
        await handler.cleanup_tts_resources()


# ---------------------------------------------------------------------------
# A mid-stream sink failure (as opposed to an open() failure) must not be
# silently dropped -- it has already committed to the streaming branch (the
# legacy path cannot un-silence itself at this point), so it surfaces
# through the same TTSCompleteEvent(error=...) channel, exactly once.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mid_stream_sink_failure_surfaces_exactly_one_error_completion(
    handler, monkeypatch,
):
    class _FailingMidStreamSink(_RecordingSink):
        def feed(self, pcm: bytes) -> bool:
            self.state = "failed"
            self.terminal_reason = "failed"
            self.fail_reason = "device error mid-stream"
            return False

    chunks = [bytes([6, 0]) * 30]
    response = _FakeResponse(chunks, audio_format="pcm", sample_rate=RATE)
    service = _FakeService(response)
    handler._tts_service = service
    _forbid_legacy_artifact_creation(handler)

    monkeypatch.setattr(tts_events_module, "StreamingPcmSink", _FailingMidStreamSink)
    monkeypatch.setattr(tts_events_module, "sink_available", lambda: True)

    await handler._generate_tts("The TTS service returned invalid audio", "adhoc", None)

    complete_events = [m for m in handler.messages if isinstance(m, TTSCompleteEvent)]
    assert len(complete_events) == 1
    assert complete_events[0].error is not None
    assert complete_events[0].audio_file is None
