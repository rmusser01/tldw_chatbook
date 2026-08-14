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
`terminal_reason`/`fail_reason`/`bytes_per_second`/`buffered_seconds`) for
the REAL `pump()` (`Audio/streaming_sink.py`, untouched here) to drive it
synchronously, with no real audio device or background thread. Patching
`streaming_sink._import_sounddevice` would not be enough on its own --
`open()` still needs a `stream_factory`-shaped success/failure outcome to
react to, which is exactly what patching the class provides.

pcm and wav responses take deliberately different branches in `_generate_tts`
and are tested accordingly. pcm decides sink eligibility BEFORE any file
write (needs only `response.sample_rate`, no bytes read) and, when eligible,
never touches disk. wav can only be validated against its COMPLETE body
(`pcm_stream.sink_plan`'s own docstring), so it decides AFTER the legacy
write loop has already run unmodified -- an earlier version that drained wav
responses up front, to decide before writing, reproduced four real
regressions in `test_console_audio_cpp_native.py`'s cancellation/partial-
artifact/write-batching pins (eager draining changes observable timing for
every wav response whenever a sink is merely available, independent of the
eventual eligibility verdict). An eligible wav response is played from the
bytes the write loop already collected, then the now-redundant artifact is
deleted; an ineligible one, or one where the sink itself fails afterward,
is reported via the file the write loop already produced -- see
`test_wav_response_with_a_trailing_chunk_stops_pumped_bytes_at_datas_end`
and `test_wav_sink_failure_falls_back_to_the_already_written_file_silently`.
"""
from __future__ import annotations

import threading
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from loguru import logger as loguru_logger

import tldw_chatbook.Audio.streaming_sink as streaming_sink_module
from tldw_chatbook.Audio.streaming_sink import SinkUnderrun
from tldw_chatbook.Event_Handlers.TTS_Events import tts_events as tts_events_module
from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
    TTSCompleteEvent,
    TTSEventHandler,
    TTSPlaybackEvent,
    TTSProgressEvent,
)
from tldw_chatbook.TTS.adapter_types import TTSAudioResponse
from Tests.Audio.test_streaming_sink import _mk as _mk_real_sink
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
        #: Fix-round F2 pin: which thread actually called `open()`, so a
        #: test can assert it ran off the event-loop thread.
        self.opened_on_thread: threading.Thread | None = None
        self.state = "idle"
        self.terminal_reason: str | None = None
        self.fail_reason: str | None = None
        self.fed: list[bytes] = []

    def open(self, sample_rate: int, channels: int = 1) -> None:
        self.opened_on_thread = threading.current_thread()
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
# (d) A lifecycle-less stop can only stop an unowned live sink globally.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message_id", "expected_stop_count"),
    [(None, 1), ("adhoc", 0)],
)
async def test_stop_action_stops_whatever_sink_is_currently_registered_live(
    handler, message_id, expected_stop_count,
):
    stop_calls: list[bool] = []

    class _FakeLiveSink:
        def stop(self) -> None:
            stop_calls.append(True)

    with streaming_sink_module._LIVE_SINK_LOCK:
        streaming_sink_module._LIVE_SINK = _FakeLiveSink()

    await handler.handle_tts_playback(TTSPlaybackEvent(action="stop", message_id=message_id))

    assert stop_calls == [True] * expected_stop_count


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


# ---------------------------------------------------------------------------
# Fix-round F1 (task-4 review): a legacy `play` action must stop a live sink
# first, symmetrically with `_stop_prior_legacy_clip` (which silences a
# legacy clip before a NEW streaming utterance starts) -- otherwise a
# `TTSPlaybackEvent(action="play")` for a different, file-based message
# starts an overlapping second voice on top of live sink playback. This was
# a newly-reachable hole (no sink existed in this path before task-4).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_legacy_play_action_stops_a_live_sink_first(handler, monkeypatch, tmp_path):
    stop_calls: list[bool] = []

    class _FakeLiveSink:
        def stop(self) -> None:
            stop_calls.append(True)

    with streaming_sink_module._LIVE_SINK_LOCK:
        streaming_sink_module._LIVE_SINK = _FakeLiveSink()

    test_audio = tmp_path / "clip.mp3"
    test_audio.write_bytes(b"fake audio data")
    async with handler._audio_files_lock:
        handler._audio_files["msg-1"] = test_audio

    fake_player = MagicMock()
    fake_player.play.return_value = True
    monkeypatch.setattr(
        "tldw_chatbook.TTS.audio_player.get_audio_player", lambda: fake_player
    )

    await handler.handle_tts_playback(TTSPlaybackEvent(action="play", message_id="msg-1"))

    assert stop_calls == [True], "a live sink must be stopped before legacy file playback starts"
    fake_player.play.assert_called_once()


# ---------------------------------------------------------------------------
# Fix-round F2 (task-4 review): `sink.open()` must run off the event loop,
# through the SAME `_run_blocking_tts_io` offload seam every other blocking
# call in `_generate_tts` already uses -- measured ~65-110ms on a quiet
# machine, worse when another process holds the audio device, which is a
# whole-UI stall in a Textual TUI on every spoken utterance otherwise.
# Pinned via the offload seam itself (spying on `_run_blocking_tts_io`) AND
# via thread identity on the fake sink -- NOT via wall-clock timing, which
# would be flaky.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sink_open_runs_off_the_event_loop_via_the_existing_offload_seam(
    handler, monkeypatch,
):
    chunks = [bytes([7, 0]) * 10]
    response = _FakeResponse(chunks, audio_format="pcm", sample_rate=RATE)
    service = _FakeService(response)
    handler._tts_service = service

    sink_holder: dict = {}
    monkeypatch.setattr(tts_events_module, "StreamingPcmSink", _spy_sink_class(sink_holder))
    monkeypatch.setattr(tts_events_module, "sink_available", lambda: True)

    offload_calls: list[str] = []
    original_run_blocking = handler._run_blocking_tts_io

    async def _spy_run_blocking(operation, **kwargs):
        offload_calls.append("called")
        return await original_run_blocking(operation, **kwargs)

    handler._run_blocking_tts_io = _spy_run_blocking

    await handler._generate_tts("Capture ended.", "adhoc", None)

    sink = sink_holder["sink"]
    assert sink.opened_with == (RATE, 1)
    assert offload_calls, (
        "sink.open() must run through the existing _run_blocking_tts_io "
        "offload seam, not directly on the event loop -- this is the ONLY "
        "possible _run_blocking_tts_io call on the pcm streaming-success "
        "path (no artifact is ever created for it), so any hit here can "
        "only be the open() call"
    )
    assert sink.opened_on_thread is not None
    assert sink.opened_on_thread is not threading.main_thread(), (
        "sink.open() ran on the event-loop (main) thread instead of being "
        "offloaded"
    )


# ---------------------------------------------------------------------------
# Fix-round F3 (task-4 review): wav-body collection for the post-write
# sink-eligibility check must be gated on `sink_available()` too, not format
# alone -- otherwise a machine with no `sounddevice` at all still retains
# the WHOLE response body in memory for every wav generation, regressing
# the bounded-memory write-batching the legacy loop was designed to keep.
# ---------------------------------------------------------------------------

def test_wants_wav_collection_is_gated_on_sink_availability_too(monkeypatch):
    monkeypatch.setattr(tts_events_module, "sink_available", lambda: False)
    assert tts_events_module._wants_wav_collection("wav") is False
    assert tts_events_module._wants_wav_collection("mp3") is False

    monkeypatch.setattr(tts_events_module, "sink_available", lambda: True)
    assert tts_events_module._wants_wav_collection("wav") is True
    assert tts_events_module._wants_wav_collection("mp3") is False
    assert tts_events_module._wants_wav_collection("pcm") is False


# ---------------------------------------------------------------------------
# Fix-round F4 (task-4 review): the bare/global stop (`chat_screen.py`'s
# dictation-start, unconditionally posted before opening the mic to protect
# the mic/speaker mutual-exclusion invariant) must ALSO silence legacy file
# playback, not just the sink -- every branch of the pre-task-4 `if/elif`
# chain required a truthy `message_id`, so a bare stop was always a no-op
# for the legacy player. Deliberately scoped to ONLY the bare stop: a
# message-scoped stop must keep its own more careful message-id-matched
# logic (task-559 unit 2) -- stopping message A must never silence a
# different, still-playing message B -- pinned by the second test below as
# a regression guard against widening the fix too far.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bare_stop_action_also_silences_legacy_file_playback(handler, monkeypatch):
    stop_calls: list = []
    clip = Path("clip.mp3")

    class _FakePlayer:
        def get_current_file(self):
            return clip

        def stop(self):
            stop_calls.append(True)
            return True

    monkeypatch.setattr(
        "tldw_chatbook.TTS.audio_player.get_audio_player", lambda: _FakePlayer()
    )
    async with handler._audio_files_lock:
        handler._last_played = ("some-other-message", clip)

    await handler.handle_tts_playback(TTSPlaybackEvent(action="stop", message_id=None))

    assert stop_calls == [True], "a bare stop must also silence legacy file playback"
    async with handler._audio_files_lock:
        assert handler._last_played is None


@pytest.mark.asyncio
async def test_message_scoped_stop_does_not_silence_a_different_messages_legacy_playback(
    handler, monkeypatch,
):
    """Regression guard (task-559 unit 2): F4's fix is scoped to ONLY the
    bare/global stop. A message-scoped stop for message A must still never
    silence a DIFFERENT, actively-playing message B's legacy clip -- this
    would have been a real regression if F4's `_stop_prior_legacy_clip`
    call had been placed unconditionally instead of gated on
    `event.message_id is None`.
    """
    stop_calls: list = []
    clip = Path("clip.mp3")

    class _FakePlayer:
        def get_current_file(self):
            return clip

        def stop(self):
            stop_calls.append(True)
            return True

    monkeypatch.setattr(
        "tldw_chatbook.TTS.audio_player.get_audio_player", lambda: _FakePlayer()
    )
    async with handler._audio_files_lock:
        handler._last_played = ("message-B", clip)

    await handler.handle_tts_playback(
        TTSPlaybackEvent(action="stop", message_id="message-A")
    )

    assert stop_calls == [], (
        "stopping message A must never silence a different, "
        "still-playing message B"
    )
    async with handler._audio_files_lock:
        assert handler._last_played == ("message-B", clip)


# ---------------------------------------------------------------------------
# Fix-round F6 (task-4 review): promotes the reviewer's real-chain probe
# into the suite. Unlike `test_stop_action_stops_whatever_sink_is_currently_
# registered_live` above (which plants a bare `_FakeLiveSink` directly into
# the registry, proving only that `stop_live_sink()` gets called), this
# constructs a REAL `StreamingPcmSink` (`Audio/streaming_sink.py`,
# untouched by this task) against a fake device stream and drives it
# through the REAL `_register_live_sink` registry AND the REAL
# `handle_tts_playback` -> `stop_live_sink()` -> `sink.stop()` chain --
# proving the real registry/stop machinery interrupts a real sink for a
# global stop (`stream.abort()`, not `.stop()`, which would drain), while a
# lifecycle-less message stop cannot claim an unowned sink.
#
# Corrected docstring (fix-round N4, re-review): this test opens the sink
# ITSELF, in the test body -- it does NOT drive `_generate_tts`, so it does
# NOT prove that the CONSUMER's own opened sink is what ends up in the
# registry (an earlier version of this comment overclaimed exactly that).
# What it proves is narrower and still real: a sink that IS registered as
# `_LIVE_SINK`, by whatever means, is only interrupted by a global stop.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message_id", "expected_stopped"),
    [(None, True), ("some-other-message", False)],
)
async def test_real_sink_end_to_end_stop_wiring(handler, message_id, expected_stopped):
    events: list = []
    sink, sink_test_holder = _mk_real_sink(events)
    sink.open(sample_rate=RATE)
    assert sink.state == "open"

    await handler.handle_tts_playback(
        TTSPlaybackEvent(action="stop", message_id=message_id)
    )

    assert (sink.terminal_reason == "stopped") is expected_stopped
    assert sink_test_holder["s"].aborted is expected_stopped
    if not expected_stopped:
        sink.stop()


# ---------------------------------------------------------------------------
# Re-review N1: `stop_live_sink()` in the `play` branch must only fire when
# a replacement is actually about to play -- a play request for a message
# whose cached artifact is already gone (the 5s cache cleanup, or one that
# was simply never generated) must not silence live streaming audio and
# then play nothing back.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_play_for_a_missing_cached_file_does_not_stop_a_live_sink(handler):
    stop_calls: list[bool] = []

    class _FakeLiveSink:
        def stop(self) -> None:
            stop_calls.append(True)

    with streaming_sink_module._LIVE_SINK_LOCK:
        streaming_sink_module._LIVE_SINK = _FakeLiveSink()

    # No `_audio_files` entry for this message -- nothing will actually play.
    await handler.handle_tts_playback(
        TTSPlaybackEvent(action="play", message_id="missing-msg")
    )

    assert stop_calls == [], (
        "a play with nothing to actually play back must not stop a live sink"
    )


# ---------------------------------------------------------------------------
# Re-review N2: F3's fix was pinned only at the pure `_wants_wav_collection`
# function -- a mutation reverting the CALL SITE back to an inline
# format-only check (`wav_collect = [] if audio_format == "wav" else None`)
# was invisible to the whole suite, since nothing asserted the seam actually
# calls the gate. Spying on the gate itself closes that: any call site that
# stops calling it fails this test regardless of whether any other
# behavioral assertion happens to still pass.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_tts_calls_the_wav_collection_gate_at_its_call_site(
    handler, monkeypatch,
):
    calls: list[str] = []
    original = tts_events_module._wants_wav_collection

    def _spy(audio_format: str) -> bool:
        calls.append(audio_format)
        return original(audio_format)

    monkeypatch.setattr(tts_events_module, "_wants_wav_collection", _spy)
    monkeypatch.setattr(tts_events_module, "sink_available", lambda: False)

    chunks = [b"RIFFnotavalidatedwavbodyatall"]
    response = _FakeResponse(chunks, audio_format="wav", sample_rate=None)
    service = _FakeService(response)
    handler._tts_service = service

    try:
        await handler._generate_tts("Nothing to read yet.", "adhoc", None)

        assert calls == ["wav"], (
            "_generate_tts must call _wants_wav_collection at its call "
            "site, not inline the format-only check it wraps"
        )
        complete_events = [m for m in handler.messages if isinstance(m, TTSCompleteEvent)]
        assert len(complete_events) == 1
        artifact_path = complete_events[0].audio_file
        assert artifact_path is not None and artifact_path.read_bytes() == chunks[0]
    finally:
        await handler.cleanup_tts_resources()


# ---------------------------------------------------------------------------
# Re-review N3: `TTSPlaybackEvent(action="stop", message_id="")` is neither
# `None` (the bare-stop branch) nor truthy (the message-scoped branch) --
# it used to fall between both, silencing nothing. Both the sink and legacy
# playback must be silenced, same as a bare/global stop.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stop_with_empty_string_message_id_is_treated_as_a_bare_stop(
    handler, monkeypatch,
):
    sink_stop_calls: list[bool] = []

    class _FakeLiveSink:
        def stop(self) -> None:
            sink_stop_calls.append(True)

    with streaming_sink_module._LIVE_SINK_LOCK:
        streaming_sink_module._LIVE_SINK = _FakeLiveSink()

    player_stop_calls: list[bool] = []
    clip = Path("clip.mp3")

    class _FakePlayer:
        def get_current_file(self):
            return clip

        def stop(self):
            player_stop_calls.append(True)
            return True

    monkeypatch.setattr(
        "tldw_chatbook.TTS.audio_player.get_audio_player", lambda: _FakePlayer()
    )
    async with handler._audio_files_lock:
        handler._last_played = ("some-message", clip)

    await handler.handle_tts_playback(TTSPlaybackEvent(action="stop", message_id=""))

    assert sink_stop_calls == [True]
    assert player_stop_calls == [True]
    async with handler._audio_files_lock:
        assert handler._last_played is None


# ---------------------------------------------------------------------------
# Re-review N5: the new "interrupted" outcome label (F8) had no test --
# nothing in the suite reached the actual metric to check a regression that
# folds it back into "success".
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_barged_in_utterance_reports_interrupted_not_success_in_metrics(
    handler, monkeypatch,
):
    class _BargedInSink(_RecordingSink):
        def feed(self, pcm: bytes) -> bool:
            accepted = super().feed(pcm)
            # Simulate an external stop() landing right after this piece
            # was accepted -- e.g. a later utterance's one-voice
            # displacement, or a barge-in.
            self.state = "stopped"
            self.terminal_reason = "stopped"
            return accepted

    chunks = [bytes([9, 0]) * 20]
    response = _FakeResponse(chunks, audio_format="pcm", sample_rate=RATE)
    service = _FakeService(response)
    handler._tts_service = service

    monkeypatch.setattr(tts_events_module, "StreamingPcmSink", _BargedInSink)
    monkeypatch.setattr(tts_events_module, "sink_available", lambda: True)

    metric_calls: list[tuple[str, dict]] = []

    def capture_counter(name, value=1, labels=None):
        metric_calls.append((name, dict(labels or {})))

    def capture_histogram(name, value, labels=None):
        metric_calls.append((name, dict(labels or {})))

    monkeypatch.setattr(
        "tldw_chatbook.Metrics.metrics_logger.log_counter", capture_counter
    )
    monkeypatch.setattr(
        "tldw_chatbook.Metrics.metrics_logger.log_histogram", capture_histogram
    )

    await handler._generate_tts("Interrupted.", "adhoc", None)

    outcome_codes = {
        labels["outcome_code"] for _, labels in metric_calls if "outcome_code" in labels
    }
    assert outcome_codes == {"interrupted"}, (
        "a displaced/interrupted utterance must not be labeled success"
    )
    complete_events = [m for m in handler.messages if isinstance(m, TTSCompleteEvent)]
    assert len(complete_events) == 1
    assert complete_events[0].error is None
    assert complete_events[0].audio_file is None


# ---------------------------------------------------------------------------
# Re-review REQUIRED merge-gate item: `Tests/conftest.py`'s autouse
# `_no_real_audio_device` fixture must neutralize the reproduced hazard --
# an eligible response reaching `_generate_tts` with NEITHER
# `StreamingPcmSink` NOR `sink_available` patched used to construct a real
# `sounddevice.OutputStream` (confirmed by the reviewer: the unguarded test
# still PASSED while doing so, a silent failure mode invisible to CI,
# review, or the suite itself).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_real_audio_device_guard_blocks_the_reproduced_hazard(
    handler, monkeypatch,
):
    """Deliberately reproduces the reviewer's exact hazard shape: the REAL
    `StreamingPcmSink` class, the REAL `sink_available()` probe -- no
    per-test mock of either. `Tests/conftest.py`'s autouse
    `_no_real_audio_device` fixture is already active for every test in
    this session, this one included, and must be what saves this one:
    `sink.open()` reaches "failed" via the already-tested `_fail("audio
    output unavailable ...")` path (driven by `_import_sounddevice()`
    returning `None`, which is exactly the chokepoint that fixture
    patches), never a real device, and generation falls through to the
    legacy artifact path.
    """
    # Cheap, non-invasive precondition check: if the conftest guard were
    # ever removed or broken, this fails loudly and immediately here,
    # rather than the test instead falling through to the
    # `_CountingOutputStream` subclass below actually opening a real
    # device (which the guard being broken would otherwise let happen).
    assert streaming_sink_module._import_sounddevice() is None, (
        "Tests/conftest.py's autouse _no_real_audio_device guard is not "
        "active -- this test cannot safely proceed"
    )

    import sounddevice

    construction_count = 0
    real_output_stream = sounddevice.OutputStream

    class _CountingOutputStream(real_output_stream):
        def __init__(self, *args, **kwargs):
            nonlocal construction_count
            construction_count += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(sounddevice, "OutputStream", _CountingOutputStream)

    chunks = [bytes([1, 0]) * 10]
    response = _FakeResponse(chunks, audio_format="pcm", sample_rate=RATE)
    service = _FakeService(response)
    handler._tts_service = service
    # Deliberately NOT patching StreamingPcmSink or sink_available here --
    # that is the point of this test.

    try:
        await handler._generate_tts("Guard check.", "adhoc", None)

        assert construction_count == 0, (
            "the conftest _no_real_audio_device guard failed to prevent a "
            "real sounddevice.OutputStream construction"
        )
        complete_events = [m for m in handler.messages if isinstance(m, TTSCompleteEvent)]
        assert len(complete_events) == 1
        assert complete_events[0].error is None
        assert complete_events[0].audio_file is not None, (
            "the open() failure must fall through to the legacy artifact "
            "path, exactly like any other sink open failure"
        )
    finally:
        await handler.cleanup_tts_resources()


# ---------------------------------------------------------------------------
# Final whole-branch review M2: `SinkUnderrun` -- the only signal that live
# playback is stuttering -- used to die silently in `_post_sink_event`'s
# debug-only recorder, invisible to both the user and metrics on the one
# feature whose entire premise is live playback quality. Minimal honest
# fix: on utterance end (once `pump()` returns), if any underrun was
# observed, log it at INFO with the frame count and bump the existing
# `tts_generation_total` metric with an `outcome_code="underrun"` label
# (same pattern F8 already established for `"interrupted"`) -- no UI.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_an_underrun_is_logged_at_info_and_bumps_the_metric_on_utterance_end(
    handler, monkeypatch,
):
    class _UnderrunningSink(_RecordingSink):
        def feed(self, pcm: bytes) -> bool:
            accepted = super().feed(pcm)
            # Simulate the device callback running dry mid-utterance --
            # SinkUnderrun always originates from the sink's own notify
            # thread in production; calling on_event directly here is the
            # same seam the real sink uses, just synchronous for the test.
            self.on_event(SinkUnderrun(frames=42))
            return accepted

    chunks = [bytes([1, 0]) * 10]
    response = _FakeResponse(chunks, audio_format="pcm", sample_rate=RATE)
    service = _FakeService(response)
    handler._tts_service = service

    monkeypatch.setattr(tts_events_module, "StreamingPcmSink", _UnderrunningSink)
    monkeypatch.setattr(tts_events_module, "sink_available", lambda: True)

    metric_calls: list[tuple[str, dict]] = []

    def capture_counter(name, value=1, labels=None):
        metric_calls.append((name, dict(labels or {})))

    monkeypatch.setattr(
        "tldw_chatbook.Metrics.metrics_logger.log_counter", capture_counter
    )

    captured_logs: list[str] = []
    handler_id = loguru_logger.add(
        lambda message: captured_logs.append(message.record["message"]),
        level="INFO",
    )
    try:
        await handler._generate_tts("Underrun check.", "adhoc", None)
    finally:
        loguru_logger.remove(handler_id)

    assert any(
        "42" in line and "underrun" in line.lower() for line in captured_logs
    ), captured_logs
    assert any(
        name == "tts_generation_total" and labels.get("outcome_code") == "underrun"
        for name, labels in metric_calls
    ), metric_calls
    # The underrun is a stutter signal, not a failure -- the utterance
    # itself still completes normally.
    complete_events = [m for m in handler.messages if isinstance(m, TTSCompleteEvent)]
    assert len(complete_events) == 1
    assert complete_events[0].error is None


@pytest.mark.asyncio
async def test_no_underrun_never_logs_or_bumps_the_underrun_metric(handler, monkeypatch):
    chunks = [bytes([1, 0]) * 10]
    response = _FakeResponse(chunks, audio_format="pcm", sample_rate=RATE)
    service = _FakeService(response)
    handler._tts_service = service

    sink_holder: dict = {}
    monkeypatch.setattr(tts_events_module, "StreamingPcmSink", _spy_sink_class(sink_holder))
    monkeypatch.setattr(tts_events_module, "sink_available", lambda: True)

    metric_calls: list[tuple[str, dict]] = []

    def capture_counter(name, value=1, labels=None):
        metric_calls.append((name, dict(labels or {})))

    monkeypatch.setattr(
        "tldw_chatbook.Metrics.metrics_logger.log_counter", capture_counter
    )

    await handler._generate_tts("No underrun.", "adhoc", None)

    assert not any(
        labels.get("outcome_code") == "underrun" for _, labels in metric_calls
    ), metric_calls
