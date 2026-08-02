# ruff: noqa: F811
"""Task 4 (hands-free-loop plan): the cooldown-free utterance speech entry.

`speak_utterance` (`Event_Handlers/TTS_Events/tts_events.py`) is the public
entry `Chat/reply_sentence_sequencer.py`'s `SentenceSequencer` will be wired
through (task 5): one call per sentence-sized utterance, with `on_finished`
threaded back into `utterance_finished(ok, token)`. It reuses the SAME
generation + playback machinery `Tests/TTS_Events/test_spoken_feedback_
streaming.py` already covers end to end (`_generate_tts`, streaming-sink
branch included) -- these tests exercise the ENTRY itself (cooldown bypass,
single-fire completion signalling on every path), not that shared machinery
a second time, reusing its fake-response/fake-sink harness rather than
duplicating it (imported below, per the brief).

Binding-carrier pin (task 2/3 reviews, restated in the task-4 brief): a
production caller MUST capture `sequencer.current_utterance_token` at
`speak()` time and thread it into `utterance_finished(ok, token)` -- a
`None` token reopens the double-voice defect (task-2 review F2). Task 4
does not own that wiring (task 5 does), but its contract -- `on_finished`
firing exactly once, on every path, including an interrupted mid-utterance
stop -- is what makes that wiring safe. The one integration-shaped test
below proves the pattern end to end with a REAL `SentenceSequencer`.
"""
from __future__ import annotations

import asyncio

import pytest

import tldw_chatbook.Event_Handlers.TTS_Events.tts_events as tts_events_module
from tldw_chatbook.Chat.reply_sentence_sequencer import SentenceSequencer
from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import TTSPlaybackEvent
from tldw_chatbook.TTS.adapter_types import TTSProviderUnavailableError
from Tests.TTS_Events.test_spoken_feedback_streaming import (  # noqa: F401,E402
    RATE,
    _FakeResponse,
    _FakeService,
    _RecordingSink,
    _reset_live_sink_registry,
    handler,
)


# ---------------------------------------------------------------------------
# Extra fakes this file needs that the shared harness doesn't provide:
# a service that can be called more than once (case a: two utterances back
# to back need two fresh responses -- `_FakeResponse.byte_stream` is a
# single-use async generator) and one that fails synthesis outright (case
# d: a genuine generation failure, not just a validation rejection).
# ---------------------------------------------------------------------------


class _MultiCallService:
    """Like `_FakeService`, but produces a FRESH response per call."""

    def __init__(self, response_factory, *, provider_id: str = "openai") -> None:
        self._response_factory = response_factory
        self._provider_id = provider_id
        self.synthesize_default_calls: list[tuple[str, str | None]] = []

    def preferences_snapshot(self):
        from types import SimpleNamespace

        return SimpleNamespace(provider_id=self._provider_id)

    async def synthesize_default(self, *, text, voice_override=None, progress_sink=None):
        self.synthesize_default_calls.append((text, voice_override))
        return self._response_factory()


class _FailingService:
    """Fails synthesis itself -- distinct from a `_tts_service is None` or
    text-validation rejection, both of which never reach generation at all.
    """

    def preferences_snapshot(self):
        from types import SimpleNamespace

        return SimpleNamespace(provider_id="openai")

    async def synthesize_default(self, *, text, voice_override=None, progress_sink=None):
        raise TTSProviderUnavailableError("synthesis unavailable")


def _counting_on_finished():
    """A completion callback that records every call, for asserting
    "exactly once" precisely (not just "truthy"/"falsy" once observed)."""
    calls: list[bool] = []
    return calls.append, calls


# ---------------------------------------------------------------------------
# (a) two utterances back-to-back both play -- no cooldown throttle.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_utterances_back_to_back_never_touch_the_cooldown_gate(
    handler, monkeypatch,
):
    """RED against a naive TTSRequestEvent-based approach: routing through
    the ad-hoc branch's `_enforce_cooldown_limit()` maintenance call (or
    the real per-message-id cooldown gate in `_admit_tts_generation`, which
    also calls it) would make this spy see at least one call. Neither the
    admission gate nor its maintenance call may ever run for this entry.
    """
    calls: list[str] = []
    original = tts_events_module.TTSEventHandler._enforce_cooldown_limit

    def _spy(self) -> None:
        calls.append("called")
        return original(self)

    monkeypatch.setattr(tts_events_module.TTSEventHandler, "_enforce_cooldown_limit", _spy)
    monkeypatch.setattr(tts_events_module, "sink_available", lambda: True)
    monkeypatch.setattr(
        tts_events_module,
        "StreamingPcmSink",
        lambda **kwargs: _RecordingSink(**kwargs),
    )

    def _fresh_response():
        return _FakeResponse([bytes([1, 0]) * 10], audio_format="pcm", sample_rate=RATE)

    service = _MultiCallService(_fresh_response)
    handler._tts_service = service

    on_finished_1, results_1 = _counting_on_finished()
    on_finished_2, results_2 = _counting_on_finished()

    await handler.speak_utterance("First utterance.", on_finished=on_finished_1)
    await handler.speak_utterance("Second utterance.", on_finished=on_finished_2)

    assert calls == [], "the cooldown gate/maintenance call must never run for this entry"
    assert len(service.synthesize_default_calls) == 2, "both utterances must actually generate"
    assert results_1 == [True]
    assert results_2 == [True]


# ---------------------------------------------------------------------------
# (b) completion fires exactly once on the legacy (file) path.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_legacy_path_completion_fires_exactly_once(handler, monkeypatch):
    chunks = [b"ID3", b"restofmp3bytes"]
    response = _FakeResponse(chunks, audio_format="mp3", sample_rate=None)
    service = _FakeService(response)
    handler._tts_service = service

    play_calls: list = []

    class _FakePlayer:
        def play(self, file_path) -> bool:
            play_calls.append(file_path)
            return True

    monkeypatch.setattr(
        "tldw_chatbook.TTS.audio_player.get_audio_player", lambda: _FakePlayer()
    )

    on_finished, results = _counting_on_finished()
    try:
        await handler.speak_utterance("Discarded.", on_finished=on_finished)

        assert results == [True], "must fire exactly once, with the player's own success"
        assert len(play_calls) == 1, "the legacy artifact must actually be played, not just written"
    finally:
        await handler.cleanup_tts_resources()


@pytest.mark.asyncio
async def test_legacy_path_completion_fires_false_once_when_the_player_fails(
    handler, monkeypatch,
):
    chunks = [b"ID3", b"restofmp3bytes"]
    response = _FakeResponse(chunks, audio_format="mp3", sample_rate=None)
    service = _FakeService(response)
    handler._tts_service = service

    class _FailingPlayer:
        def play(self, file_path) -> bool:
            return False

    monkeypatch.setattr(
        "tldw_chatbook.TTS.audio_player.get_audio_player", lambda: _FailingPlayer()
    )

    on_finished, results = _counting_on_finished()
    try:
        await handler.speak_utterance("Discarded.", on_finished=on_finished)

        assert results == [False], "a player failure must still fire exactly once, as False"
    finally:
        await handler.cleanup_tts_resources()


# ---------------------------------------------------------------------------
# (c) completion fires exactly once on the sink path (drained).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sink_path_completion_fires_exactly_once_on_drain(handler, monkeypatch):
    chunks = [bytes([2, 0]) * 40, bytes([3, 0]) * 40]
    response = _FakeResponse(chunks, audio_format="pcm", sample_rate=RATE)
    service = _FakeService(response)
    handler._tts_service = service

    sink_holder: dict = {}

    def _spy_sink(**kwargs):
        sink = _RecordingSink(**kwargs)
        sink_holder["sink"] = sink
        return sink

    monkeypatch.setattr(tts_events_module, "StreamingPcmSink", _spy_sink)
    monkeypatch.setattr(tts_events_module, "sink_available", lambda: True)

    on_finished, results = _counting_on_finished()
    await handler.speak_utterance("Spoken live.", on_finished=on_finished)

    assert results == [True]
    sink = sink_holder["sink"]
    assert b"".join(sink.fed) == b"".join(chunks)
    assert handler._audio_files == {}, "a streamed response must never touch the legacy artifact path"


# ---------------------------------------------------------------------------
# (d) a genuine synthesis failure fires on_finished(False) exactly once.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesis_failure_fires_on_finished_false_exactly_once(handler):
    handler._tts_service = _FailingService()

    on_finished, results = _counting_on_finished()
    await handler.speak_utterance("Will fail.", on_finished=on_finished)

    assert results == [False]


# ---------------------------------------------------------------------------
# (e) an interrupted/stopped sink still fires completion exactly once, ok=False.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stopped_sink_fires_completion_exactly_once_as_ok_false(handler, monkeypatch):
    """Same deterministic technique `test_spoken_feedback_streaming.py`'s
    `_BargedInSink` already established for this exact scenario (a real
    concurrent stop landing mid-pump can't be driven synchronously without
    a live audio device): the fake sink flips itself to "stopped" the
    instant it accepts bytes, simulating an external stop (one-voice
    displacement or a barge-in) landing mid-utterance. The REAL
    both-ways stop routine (`handle_tts_playback`) is exercised separately,
    end to end, by the integration test below.
    """
    class _BargedInSink(_RecordingSink):
        def feed(self, pcm: bytes) -> bool:
            accepted = super().feed(pcm)
            self.state = "stopped"
            self.terminal_reason = "stopped"
            return accepted

    chunks = [bytes([4, 0]) * 20]
    response = _FakeResponse(chunks, audio_format="pcm", sample_rate=RATE)
    service = _FakeService(response)
    handler._tts_service = service

    monkeypatch.setattr(tts_events_module, "StreamingPcmSink", _BargedInSink)
    monkeypatch.setattr(tts_events_module, "sink_available", lambda: True)

    on_finished, results = _counting_on_finished()
    await handler.speak_utterance("Interrupted.", on_finished=on_finished)

    assert results == [False]


# ---------------------------------------------------------------------------
# Integration pin (brief's binding-carrier requirement): a REAL
# SentenceSequencer wired through the REAL speak_utterance entry, driven by
# a fake TTS service/sink standing in for the audio backend, must survive a
# barge-in landing MID first utterance -- the token-threaded completion for
# the abandoned utterance lands safely (sequencer._inflight already False
# by then, so utterance_finished() no-ops), the second queued sentence is
# never dispatched (no double-advance), and the sequencer still reaches
# drained (no stall).
# ---------------------------------------------------------------------------


class _SequencerDriver:
    """Bridges the sequencer's plain-sync `speak`/`stop_speech` callables to
    the async `speak_utterance`/`handle_tts_playback` entries, the same way
    a real production caller (task 5) will -- `speak()` reads
    `current_utterance_token` synchronously, at dispatch time (the
    binding-carrier rule), and closes it into `on_finished`.
    """

    def __init__(self, handler) -> None:
        self.handler = handler
        self.sequencer: SentenceSequencer | None = None
        self.speak_calls: list[str] = []
        self.finish_calls: list[tuple[str, bool, int | None]] = []
        self.tasks: list[asyncio.Task] = []

    def speak(self, text: str) -> None:
        sequencer = self.sequencer
        assert sequencer is not None
        token = sequencer.current_utterance_token
        self.speak_calls.append(text)

        def _on_finished(ok: bool) -> None:
            self.finish_calls.append((text, ok, token))
            sequencer.utterance_finished(ok, token=token)

        self.tasks.append(
            asyncio.create_task(
                self.handler.speak_utterance(text, on_finished=_on_finished)
            )
        )

    def stop_speech(self) -> None:
        self.tasks.append(
            asyncio.create_task(
                self.handler.handle_tts_playback(
                    TTSPlaybackEvent(action="stop", message_id=None)
                )
            )
        )


async def _drain(driver: _SequencerDriver) -> None:
    """Keep awaiting tasks until no more get scheduled -- a barge-in landing
    mid-flight schedules a NEW `stop_speech()` task from inside an
    already-running `speak()` task, which a single `gather()` call would
    miss (its argument list is captured once, at call time)."""
    while driver.tasks:
        pending, driver.tasks = driver.tasks, []
        await asyncio.gather(*pending)


@pytest.mark.asyncio
async def test_sentence_sequencer_wired_through_speak_utterance_survives_a_mid_utterance_stop(
    handler, monkeypatch,
):
    driver = _SequencerDriver(handler)
    sequencer = SentenceSequencer(speak=driver.speak, stop_speech=driver.stop_speech)
    driver.sequencer = sequencer
    drained_calls: list[bool] = []
    sequencer.on_drained = lambda: drained_calls.append(True)

    class _BargingSink(_RecordingSink):
        def feed(self, pcm: bytes) -> bool:
            accepted = super().feed(pcm)
            # The barge-in signal (Esc / VAD interrupt) landing mid-first-
            # utterance, from the sequencer's own perspective -- BEFORE
            # this utterance's own speak_utterance call has resolved.
            sequencer.flush()
            self.state = "stopped"
            self.terminal_reason = "stopped"
            return accepted

    chunks = [bytes([5, 0]) * 20]
    response = _FakeResponse(chunks, audio_format="pcm", sample_rate=RATE)
    service = _FakeService(response)
    handler._tts_service = service

    monkeypatch.setattr(tts_events_module, "StreamingPcmSink", _BargingSink)
    monkeypatch.setattr(tts_events_module, "sink_available", lambda: True)

    sequencer.feed("First sentence. Second sentence. ")
    sequencer.reply_completed()

    await _drain(driver)

    assert driver.speak_calls == ["First sentence."], (
        "the second sentence must never be dispatched after a barge-in "
        "(no double-advance)"
    )
    assert driver.finish_calls == [("First sentence.", False, 1)], (
        "exactly one completion, carrying the token captured at speak "
        "time, for the interrupted utterance"
    )
    assert sequencer.drained is True, "the sequencer must not stall after the barge-in"
    assert drained_calls == [True]
