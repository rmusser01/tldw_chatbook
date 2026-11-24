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

## Review fix-round (task-4-review.md)

The reviewer's probes drove the app's real routing (`app.py`'s
`TTSCompleteEvent`/`TTSPlaybackEvent` handlers) on top of the real handler,
which the `handler` fixture's plain-list `post_message` cannot see on its
own -- that is how F1 (a real double voice) escaped the original suite.
These additions stay at the handler level (matching the rest of this file)
but assert on the OBSERVABLE CONTRACT the app's routing depends on
(`audio_file` on the posted `TTSCompleteEvent`) rather than re-deriving the
app's own routing logic here.

The file tests exercise real per-attempt runs and app-wide ownership.
Their native players control when exact attempts start, finish and retire.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

import tldw_chatbook.Event_Handlers.TTS_Events.tts_events as tts_events_module
from tldw_chatbook.Chat.reply_sentence_sequencer import SentenceSequencer
from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
    TTSCompleteEvent,
    TTSPlaybackEvent,
)
from Tests.TTS.owned_output_fakes import (  # noqa: F401
    ConsoleFilePlayer,
    forbid_native_audio,
    wait_for_condition,
)
from tldw_chatbook.TTS import audio_output_runs, audio_player, playback_capability
from tldw_chatbook.TTS.audio_player import PlaybackState
from tldw_chatbook.TTS.adapter_types import TTSProviderUnavailableError
from Tests.TTS_Events.test_spoken_feedback_streaming import (  # noqa: F401,E402
    RATE,
    _FakeResponse,
    _FakeService,
    _RecordingSink,
    _reset_live_sink_registry,
    handler,
)


@pytest.fixture(autouse=True)
async def _retire_utterance_handler(handler, monkeypatch):
    """Finish owned artifact cleanup while the fake player is still installed."""
    monkeypatch.setattr(
        playback_capability, "format_playable_locally", lambda _format: True
    )
    monkeypatch.setattr(
        audio_player, "find_player_for_format", lambda _format: "test-player"
    )
    yield
    await handler.cleanup_tts_resources()
    assert not handler._audio_files
    assert not handler._pending_legacy_cleanup_timers


# ---------------------------------------------------------------------------
# Extra fakes this file needs that the shared harness doesn't provide:
# a service that can be called more than once (case a: two utterances back
# to back need two fresh responses -- `_FakeResponse.byte_stream` is a
# single-use async generator), one that fails synthesis outright (case d),
# one that pauses generation on demand (F4: a reliable in-flight window),
# and a legacy-player fake implementing the poll surface F2's fix needs.
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

    async def synthesize_default(
        self,
        *,
        text,
        voice_override=None,
        response_format_override=None,
        progress_sink=None,
    ):
        self.synthesize_default_calls.append((text, voice_override))
        return self._response_factory()


class _FailingService:
    """Fails synthesis itself -- distinct from a `_tts_service is None` or
    text-validation rejection, both of which never reach generation at all.
    """

    def preferences_snapshot(self):
        from types import SimpleNamespace

        return SimpleNamespace(provider_id="openai")

    async def synthesize_default(
        self,
        *,
        text,
        voice_override=None,
        response_format_override=None,
        progress_sink=None,
    ):
        raise TTSProviderUnavailableError("synthesis unavailable")


class _PausableService:
    """Blocks inside `synthesize_default` until released -- gives a
    reliable, deterministic in-flight window (task-4 review F4) without
    real-time sleeps or scheduling guesswork."""

    def __init__(self, response) -> None:
        self._response = response
        self.proceed = asyncio.Event()

    def preferences_snapshot(self):
        from types import SimpleNamespace

        return SimpleNamespace(provider_id="openai")

    async def synthesize_default(
        self,
        *,
        text,
        voice_override=None,
        response_format_override=None,
        progress_sink=None,
    ):
        await self.proceed.wait()
        return self._response


class _FakeLegacyPlayer(ConsoleFilePlayer):
    """Report natural completion on a chosen observation of this exact attempt."""

    def __init__(self, *, play_succeeds=True, finishes_after_polls=1, events=None):
        super().__init__(PlaybackState.PLAYING if play_succeeds else None)
        self.finish_after = finishes_after_polls
        self.polls = 0
        self.events = events if events is not None else []
        self.play_calls = self.paths

    def get_state(self, *, expected_token=None):
        state = super().get_state(expected_token=expected_token)
        if state is PlaybackState.PLAYING:
            self.polls += 1
            if self.polls >= self.finish_after:
                self.state = PlaybackState.FINISHED
                self.events.append("observed_finished")
        return self.state if expected_token is self.token else state


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
    handler,
    monkeypatch,
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

    monkeypatch.setattr(
        tts_events_module.TTSEventHandler, "_enforce_cooldown_limit", _spy
    )
    monkeypatch.setattr(tts_events_module, "sink_available", lambda: True)
    monkeypatch.setattr(
        audio_output_runs,
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

    assert calls == [], (
        "the cooldown gate/maintenance call must never run for this entry"
    )
    assert len(service.synthesize_default_calls) == 2, (
        "both utterances must actually generate"
    )
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

    fake_player = _FakeLegacyPlayer(finishes_after_polls=1)
    monkeypatch.setattr(audio_output_runs, "SimpleAudioPlayer", lambda: fake_player)

    on_finished, results = _counting_on_finished()
    try:
        await handler.speak_utterance("Discarded.", on_finished=on_finished)

        assert results == [True], (
            "must fire exactly once, once playback actually finished"
        )
        assert len(fake_player.play_calls) == 1, (
            "the legacy artifact must actually be played, not just written"
        )
    finally:
        await handler.cleanup_tts_resources()


@pytest.mark.asyncio
async def test_legacy_path_completion_fires_false_once_when_the_player_fails(
    handler,
    monkeypatch,
):
    chunks = [b"ID3", b"restofmp3bytes"]
    response = _FakeResponse(chunks, audio_format="mp3", sample_rate=None)
    service = _FakeService(response)
    handler._tts_service = service

    fake_player = _FakeLegacyPlayer(play_succeeds=False)
    monkeypatch.setattr(audio_output_runs, "SimpleAudioPlayer", lambda: fake_player)

    on_finished, results = _counting_on_finished()
    try:
        await handler.speak_utterance("Discarded.", on_finished=on_finished)

        assert results == [False], (
            "a player failure must still fire exactly once, as False"
        )
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

    monkeypatch.setattr(audio_output_runs, "StreamingPcmSink", _spy_sink)
    monkeypatch.setattr(tts_events_module, "sink_available", lambda: True)

    on_finished, results = _counting_on_finished()
    await handler.speak_utterance("Spoken live.", on_finished=on_finished)

    assert results == [True]
    sink = sink_holder["sink"]
    assert b"".join(sink.fed) == b"".join(chunks)
    assert handler._audio_files == {}, (
        "a streamed response must never touch the legacy artifact path"
    )


# ---------------------------------------------------------------------------
# (d) a genuine synthesis failure fires on_finished(False) exactly once.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesis_failure_fires_on_finished_false_exactly_once(handler):
    handler._tts_service = _FailingService()

    on_finished, results = _counting_on_finished()
    await handler.speak_utterance("Will fail.", on_finished=on_finished)

    assert results == [False]
    complete_events = [m for m in handler.messages if isinstance(m, TTSCompleteEvent)]
    assert len(complete_events) == 1 and complete_events[0].error is not None, (
        "by default (quiet=False) the failure toast must still post"
    )


# ---------------------------------------------------------------------------
# (e) an interrupted/stopped sink still fires completion exactly once, ok=False.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stopped_sink_fires_completion_exactly_once_as_ok_false(
    handler, monkeypatch
):
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

    monkeypatch.setattr(audio_output_runs, "StreamingPcmSink", _BargedInSink)
    monkeypatch.setattr(tts_events_module, "sink_available", lambda: True)

    on_finished, results = _counting_on_finished()
    await handler.speak_utterance("Interrupted.", on_finished=on_finished)

    assert results == [False]


# ---------------------------------------------------------------------------
# F1 (BLOCKER, task-4-review.md) -- every legacy hands-free utterance played
# TWICE: `_generate_tts` posted `TTSCompleteEvent(audio_file=artifact_path)`
# before playing it directly; no widget ever claims a `handsfree-<uuid4>`
# id, so the app's own `TTSCompleteEvent` handler (`app.py:6699-6706`)
# auto-played the SAME file a second time. Fixed: `audio_file=None` when
# `on_finished is not None`, exactly like the sink branch already does.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_legacy_completion_advertises_no_audio_file_so_the_app_cannot_double_play(
    handler,
    monkeypatch,
):
    chunks = [b"ID3", b"restofmp3bytes"]
    response = _FakeResponse(chunks, audio_format="mp3", sample_rate=None)
    service = _FakeService(response)
    handler._tts_service = service

    fake_player = _FakeLegacyPlayer(finishes_after_polls=1)
    monkeypatch.setattr(audio_output_runs, "SimpleAudioPlayer", lambda: fake_player)

    on_finished, results = _counting_on_finished()
    await handler.speak_utterance("Discarded.", on_finished=on_finished)

    complete_events = [m for m in handler.messages if isinstance(m, TTSCompleteEvent)]
    assert len(complete_events) == 1
    assert complete_events[0].audio_file is None, (
        "posting the real artifact path here is exactly what let the app's "
        "own TTSCompleteEvent handler auto-play it a SECOND time on top of "
        "the direct play below (task-4 review F1)"
    )
    # The direct play must still have genuinely happened -- F1's fix must
    # not degrade into silently dropping playback altogether.
    assert len(fake_player.play_calls) == 1
    assert results == [True]


@pytest.mark.asyncio
async def test_non_handsfree_legacy_completion_still_advertises_its_audio_file(
    handler,
    monkeypatch,
):
    """Regression guard: F1's fix is gated on `on_finished is not None` --
    every OTHER caller (spoken feedback, character speech, ad-hoc requests)
    goes through `_generate_tts` with `on_finished=None` and must keep
    seeing the real artifact path, unchanged."""
    chunks = [b"ID3", b"restofmp3bytes"]
    response = _FakeResponse(chunks, audio_format="mp3", sample_rate=None)
    service = _FakeService(response)
    handler._tts_service = service

    try:
        await handler._generate_tts("Discarded.", "adhoc", None)

        complete_events = [
            m for m in handler.messages if isinstance(m, TTSCompleteEvent)
        ]
        assert len(complete_events) == 1
        assert complete_events[0].audio_file is not None
        assert complete_events[0].audio_file.exists()
    finally:
        await handler.cleanup_tts_resources()


# ---------------------------------------------------------------------------
# F2 (Important, task-4-review.md, THE HEADLINE) -- completion meant "handed
# off", not "played": `on_finished(True)` fired the instant `play()` handed
# the clip to a background process, not when it actually finished. Since
# `play()` stops whatever is currently loaded first (a single-slot global
# singleton), the very next utterance's handoff killed the current one
# mid-word. Fixed: poll `get_state()`/`get_current_file()` inside the
# existing `_run_blocking_tts_io` worker until the clip is no longer
# current, bounded, before firing.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_completion_waits_for_the_poll_to_observe_finished_before_firing(
    handler,
    monkeypatch,
):
    chunks = [b"ID3", b"restofmp3bytes"]
    response = _FakeResponse(chunks, audio_format="mp3", sample_rate=None)
    service = _FakeService(response)
    handler._tts_service = service

    events: list[str] = []
    fake_player = _FakeLegacyPlayer(finishes_after_polls=4, events=events)
    monkeypatch.setattr(audio_output_runs, "SimpleAudioPlayer", lambda: fake_player)

    def _on_finished(ok: bool) -> None:
        events.append("on_finished")

    await handler.speak_utterance("Discarded.", on_finished=_on_finished)

    assert events == ["observed_finished", "on_finished"], (
        "on_finished must not fire before the poll observes the clip "
        "reaching FINISHED (task-4 review F2, the headline finding: "
        "firing at handoff instead of completion truncated every "
        "sentence but the last)"
    )


@pytest.mark.asyncio
async def test_completion_reports_false_when_other_output_takes_over(
    handler, monkeypatch, tmp_path
):
    handler._tts_service = _FakeService(
        _FakeResponse([b"ID3", b"audio"], audio_format="mp3", sample_rate=None)
    )
    old_player = ConsoleFilePlayer(PlaybackState.PLAYING)
    monkeypatch.setattr(audio_output_runs, "SimpleAudioPlayer", lambda: old_player)
    outcomes = []
    task = asyncio.create_task(
        handler.speak_utterance("Discarded.", on_finished=outcomes.append)
    )
    await wait_for_condition(lambda: old_player.launches == 1)

    new_player = ConsoleFilePlayer(PlaybackState.PLAYING)
    run = await audio_output_runs.FileOutputRun.prepare(
        tmp_path / "unrelated.wav", player=new_player
    )
    host = handler.app.audio_output
    lease = await host.acquire("Other surface", run.retire, intent=host.begin_intent())
    try:
        assert old_player.stops == [old_player.token]
        assert await run.start(lease)
        await task
        assert outcomes == [False]
        assert host.owner == "Other surface"
        assert not new_player.stops
    finally:
        await host.stop(lease)


@pytest.mark.asyncio
async def test_utterance_is_registered_before_native_start_and_stop_fences_it(
    handler, monkeypatch, tmp_path
):
    path = tmp_path / "clip.mp3"
    path.write_bytes(b"fake audio")
    entered, release = threading.Event(), threading.Event()

    class HeldPlayer(ConsoleFilePlayer):
        def play_prepared(self, plan, *, attempt_token, start_authority):
            entered.set()
            assert release.wait(5)
            if not start_authority():
                self.refused_token = attempt_token
                return False
            return super().play_prepared(
                plan, attempt_token=attempt_token, start_authority=start_authority
            )

    player = HeldPlayer()
    monkeypatch.setattr(audio_output_runs, "SimpleAudioPlayer", lambda: player)
    outcomes, stops = [], []
    speech = asyncio.create_task(
        handler._play_utterance_legacy_artifact("msg", path, "Hello.", outcomes.append)
    )
    stop = None
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        (attempt,) = handler._speech_outputs
        assert (attempt.message_id, attempt.path) == ("msg", path)
        assert attempt.run.lease.is_current()
        assert outcomes == []

        stop = asyncio.create_task(
            handler.handle_tts_playback(
                TTSPlaybackEvent(action="stop", outcome_callback=stops.append)
            )
        )
        await wait_for_condition(lambda: attempt.run.lease.revoked)
        assert not stop.done()
        release.set()
        await asyncio.gather(stop, speech)
        assert player.launches == 0
        assert stops == [True]
        assert outcomes == [False]
        assert not handler._speech_outputs
        assert handler.app.audio_output.owner is None
    finally:
        release.set()
        await asyncio.gather(speech, *([stop] if stop is not None else []))


@pytest.mark.asyncio
@pytest.mark.parametrize("message_id", [None, "unrelated"])
async def test_stop_without_tracked_speech_preserves_other_surface(
    handler, tmp_path, message_id
):
    player = ConsoleFilePlayer(PlaybackState.PLAYING)
    run = await audio_output_runs.FileOutputRun.prepare(
        tmp_path / "unrelated.wav", player=player
    )
    host = handler.app.audio_output
    lease = await host.acquire("Other surface", run.retire)
    try:
        assert await run.start(lease)
        outcomes = []
        await handler.handle_tts_playback(
            TTSPlaybackEvent(
                action="stop", message_id=message_id, outcome_callback=outcomes.append
            )
        )
        assert outcomes == [False]
        assert await run.get_state() is PlaybackState.PLAYING
        assert lease.is_current()
        assert not player.stops
    finally:
        await host.stop(lease)


@pytest.mark.asyncio
async def test_older_preparation_finally_cannot_untrack_new_attempt(
    handler, monkeypatch, tmp_path
):
    old_path, next_path = tmp_path / "old.mp3", tmp_path / "new.mp3"
    for path in (old_path, next_path):
        path.write_bytes(b"fake audio")
    entered, release = threading.Event(), threading.Event()

    class Player(ConsoleFilePlayer):
        def prepare(self, path):
            if path == old_path:
                entered.set()
                assert release.wait(5)
            return super().prepare(path)

    players = [Player(PlaybackState.PLAYING), Player(PlaybackState.PLAYING)]
    monkeypatch.setattr(audio_output_runs, "SimpleAudioPlayer", iter(players).__next__)
    old_outcomes, next_outcomes = [], []
    old = asyncio.create_task(
        handler._play_utterance_legacy_artifact(
            "same", old_path, "Old", old_outcomes.append
        )
    )
    new = None
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        new = asyncio.create_task(
            handler._play_utterance_legacy_artifact(
                "same", next_path, "New", next_outcomes.append
            )
        )
        await wait_for_condition(lambda: players[1].launches == 1)
        release.set()
        await old
        assert old_outcomes == [False] and players[0].launches == 0
        (attempt,) = handler._speech_outputs
        assert attempt.path == next_path
        assert attempt.run.lease.is_current()

        await handler.handle_tts_playback(TTSPlaybackEvent(action="stop"))
        await new
        assert next_outcomes == [False]
        assert players[1].stops == [players[1].token]
    finally:
        release.set()
        await handler.handle_tts_playback(TTSPlaybackEvent(action="stop"))
        await asyncio.gather(old, *([new] if new is not None else []))


# ---------------------------------------------------------------------------
# N3 (Minor, task-4-review.md) -- the completion bound was rate-blind:
# `default_speed` is user-configurable and only validated as "finite
# positive". Fixed: fold the resolved speed into the estimate (a slower
# speed widens the bound), raise the absolute ceiling, and log distinctly
# when the bound is what ended the poll (not a natural finish) rather than
# leave both cases returning the same `True`.
# ---------------------------------------------------------------------------


def test_legacy_playback_timeout_seconds_widens_for_a_slower_configured_speed():
    text_length = 200
    normal = tts_events_module._legacy_playback_timeout_seconds(text_length, speed=1.0)
    half_speed = tts_events_module._legacy_playback_timeout_seconds(
        text_length, speed=0.5
    )

    expected_half = (
        text_length / (tts_events_module._LEGACY_PLAYBACK_MIN_CHARS_PER_SECOND * 0.5)
        + tts_events_module._LEGACY_PLAYBACK_POLL_MARGIN_SECONDS
    )
    assert half_speed == pytest.approx(expected_half)
    assert half_speed > normal, (
        "a slower configured speed must widen the bound, not leave it "
        "unchanged (task-4 review N3)"
    )


def test_legacy_playback_timeout_seconds_floors_non_positive_speed_to_normal():
    """Defensive only -- `TTS/preferences.py`'s `_require_speed` already
    guarantees "finite positive" upstream, but this function must never
    divide by zero or go negative regardless of what a caller passes."""
    baseline = tts_events_module._legacy_playback_timeout_seconds(100, speed=1.0)
    assert (
        tts_events_module._legacy_playback_timeout_seconds(100, speed=0.0) == baseline
    )
    assert (
        tts_events_module._legacy_playback_timeout_seconds(100, speed=-2.0) == baseline
    )


@pytest.mark.asyncio
async def test_generate_tts_extracts_speed_from_preferences_and_threads_it_through(
    handler,
    monkeypatch,
):
    chunks = [b"ID3", b"restofmp3bytes"]
    response = _FakeResponse(chunks, audio_format="mp3", sample_rate=None)

    class _SlowSpeedService:
        def preferences_snapshot(self):
            from types import SimpleNamespace

            return SimpleNamespace(provider_id="openai", speed=0.5)

        async def synthesize_default(
            self,
            *,
            text,
            voice_override=None,
            response_format_override=None,
            progress_sink=None,
        ):
            return response

    handler._tts_service = _SlowSpeedService()

    fake_player = _FakeLegacyPlayer(finishes_after_polls=1)
    monkeypatch.setattr(audio_output_runs, "SimpleAudioPlayer", lambda: fake_player)

    captured_speeds: list[float] = []
    original = tts_events_module._legacy_playback_timeout_seconds

    def _spy(text_length, speed=1.0):
        captured_speeds.append(speed)
        return original(text_length, speed)

    monkeypatch.setattr(tts_events_module, "_legacy_playback_timeout_seconds", _spy)

    on_finished, results = _counting_on_finished()
    await handler.speak_utterance("Discarded.", on_finished=on_finished)

    assert captured_speeds == [0.5], (
        "the preferences snapshot's speed must reach the timeout estimate "
        "(task-4 review N3)"
    )
    assert results == [True]


@pytest.mark.asyncio
async def test_a_timed_out_poll_logs_a_distinguishing_warning(handler, monkeypatch):
    from loguru import logger

    handler._tts_service = _FakeService(
        _FakeResponse([b"ID3", b"audio"], audio_format="mp3", sample_rate=None)
    )
    player = ConsoleFilePlayer(PlaybackState.PLAYING)
    monkeypatch.setattr(audio_output_runs, "SimpleAudioPlayer", lambda: player)
    monkeypatch.setattr(
        tts_events_module, "_legacy_playback_timeout_seconds", lambda *_args: 0.02
    )
    monkeypatch.setattr(
        tts_events_module, "_LEGACY_PLAYBACK_POLL_INTERVAL_SECONDS", 0.005
    )
    logs = []
    log_id = logger.add(
        lambda message: logs.append(message.record["message"]), level="WARNING"
    )
    try:
        outcomes = []
        await handler.speak_utterance("Wait.", on_finished=outcomes.append)
        assert outcomes == [True]
        assert not player.stops
        assert handler.app.audio_output.owner is not None
        assert any("timed out" in line.lower() for line in logs), logs
        await handler.handle_tts_playback(TTSPlaybackEvent(action="stop"))
        assert player.stops == [player.token]
    finally:
        logger.remove(log_id)


@pytest.mark.asyncio
async def test_cancelling_mid_poll_promptly_stops_the_player(handler, monkeypatch):
    handler._tts_service = _FakeService(
        _FakeResponse([b"ID3", b"audio"], audio_format="mp3", sample_rate=None)
    )
    player = ConsoleFilePlayer(PlaybackState.PLAYING)
    monkeypatch.setattr(audio_output_runs, "SimpleAudioPlayer", lambda: player)
    outcomes = []
    task = asyncio.create_task(
        handler.speak_utterance("Discarded.", on_finished=outcomes.append)
    )
    await wait_for_condition(lambda: player.launches == 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)
    assert player.stops == [player.token]
    assert outcomes == [False]
    assert handler.app.audio_output.owner is None


# ---------------------------------------------------------------------------
# N5 (Minor, task-4-review.md) -- `cleanup_tts_resources()` used to AWAIT
# (boundedly) the legacy cleanup timer, buying nothing since the same
# method deletes the artifact directly moments later regardless -- a
# measured ~2s shutdown cost for zero benefit. Fixed: cancel it outright.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_tts_resources_cancels_the_legacy_timer_instead_of_awaiting_it(
    handler,
    monkeypatch,
):
    chunks = [b"ID3", b"restofmp3bytes"]
    response = _FakeResponse(chunks, audio_format="mp3", sample_rate=None)
    service = _FakeService(response)
    handler._tts_service = service

    fake_player = _FakeLegacyPlayer(finishes_after_polls=1)
    monkeypatch.setattr(audio_output_runs, "SimpleAudioPlayer", lambda: fake_player)

    on_finished, results = _counting_on_finished()
    await handler.speak_utterance("Discarded.", on_finished=on_finished)
    assert len(handler._pending_legacy_cleanup_timers) == 1, (
        "the delayed cleanup timer must be tracked in its own set, not "
        "the awaited/drained _retained_tts_cleanup_tasks"
    )

    started = asyncio.get_event_loop().time()
    await handler.cleanup_tts_resources()
    elapsed = asyncio.get_event_loop().time() - started

    assert elapsed < 0.5, (
        f"cleanup_tts_resources() must not wait out the 5s timer "
        f"(task-4 review N5); took {elapsed:.2f}s"
    )
    assert handler._pending_legacy_cleanup_timers == set()


# ---------------------------------------------------------------------------
# F4 (Minor, task-4-review.md) -- hands-free generation was not registered
# in `_active_tasks`, so `cleanup_tts_resources()` could not cancel an
# in-flight utterance at shutdown. Fixed: `speak_utterance` now registers
# its generation task the same way `_admit_tts_generation` does for every
# other caller.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generation_is_registered_in_active_tasks_while_in_flight(
    handler,
    monkeypatch,
):
    chunks = [b"ID3", b"restofmp3bytes"]
    response = _FakeResponse(chunks, audio_format="mp3", sample_rate=None)
    service = _PausableService(response)
    handler._tts_service = service

    fake_player = _FakeLegacyPlayer(finishes_after_polls=1)
    monkeypatch.setattr(audio_output_runs, "SimpleAudioPlayer", lambda: fake_player)

    on_finished, results = _counting_on_finished()
    task = asyncio.create_task(
        handler.speak_utterance("Registered?", on_finished=on_finished)
    )
    for _ in range(5):
        await asyncio.sleep(0)

    async with handler._active_tasks_lock:
        active_count = len(handler._active_tasks)
    assert active_count == 1, (
        "the generation task must be registered in _active_tasks while "
        "in flight, so cleanup_tts_resources() can cancel it (task-4 "
        "review F4)"
    )

    service.proceed.set()
    await task

    async with handler._active_tasks_lock:
        assert len(handler._active_tasks) == 0, (
            "must be removed once generation completes"
        )
    assert results == [True]


# ---------------------------------------------------------------------------
# F5 (Minor, task-4-review.md) -- one user-visible error toast per failed
# sentence would be a toast storm proportional to reply length. Fixed: an
# opt-in `quiet` flag suppresses the toast (completion/logging unaffected);
# `speak_utterance` stays stateless -- a stateful caller (task 5) decides
# WHEN to pass it.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_quiet_suppresses_the_error_toast_but_not_completion(handler):
    handler._tts_service = _FailingService()
    on_finished, results = _counting_on_finished()

    await handler.speak_utterance("Will fail.", on_finished=on_finished, quiet=True)

    assert results == [False], "on_finished must still fire normally"
    complete_events = [m for m in handler.messages if isinstance(m, TTSCompleteEvent)]
    assert complete_events == [], "the error toast must be suppressed when quiet=True"


@pytest.mark.asyncio
async def test_quiet_suppresses_the_text_validation_toast_too(handler):
    handler._tts_service = None  # _prepare_tts_text's own rejection path
    on_finished, results = _counting_on_finished()

    await handler.speak_utterance("Anything.", on_finished=on_finished, quiet=True)

    assert results == [False]
    complete_events = [m for m in handler.messages if isinstance(m, TTSCompleteEvent)]
    assert complete_events == []


# ---------------------------------------------------------------------------
# F6 (Minor, task-4-review.md) -- a raising `on_finished` was swallowed
# inside `_generate_tts`'s own `try`, misreported as a TTS generation
# failure. In production that callback is real caller code
# (`utterance_finished -> speak -> ...`). Fixed: isolated inside `fire()`,
# logged, never allowed to unwind into generation's own exception handling.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_raising_on_finished_is_not_reported_as_a_generation_failure(
    handler,
    monkeypatch,
):
    chunks = [bytes([1, 0]) * 10]
    response = _FakeResponse(chunks, audio_format="pcm", sample_rate=RATE)
    service = _FakeService(response)
    handler._tts_service = service

    monkeypatch.setattr(audio_output_runs, "StreamingPcmSink", _RecordingSink)
    monkeypatch.setattr(tts_events_module, "sink_available", lambda: True)

    def _raising_on_finished(ok: bool) -> None:
        raise RuntimeError("caller bug, not a TTS failure")

    # Must not raise out of speak_utterance itself.
    await handler.speak_utterance("Spoken live.", on_finished=_raising_on_finished)

    complete_events = [m for m in handler.messages if isinstance(m, TTSCompleteEvent)]
    assert all(event.error is None for event in complete_events), (
        "a raising on_finished must never be misreported as a TTS "
        "generation failure (task-4 review F6)"
    )


# ---------------------------------------------------------------------------
# Integration pin (brief's binding-carrier requirement): a REAL
# SentenceSequencer wired through the REAL speak_utterance entry, driven by
# a fake TTS service/sink standing in for the audio backend, must survive a
# barge-in landing MID first utterance -- the token-threaded completion for
# the abandoned utterance lands safely (sequencer._inflight already False
# by then, so utterance_finished() no-ops), the second queued sentence is
# never dispatched (no double-advance), and the sequencer still reaches
# drained (no stall).
#
# F8 (Nit, task-4-review.md): the mechanism that catches the double-advance
# here is `flush()`'s `_suppressed` latch + `_inflight = False`
# (`utterance_finished` short-circuits at `if not self._inflight: return`),
# NOT the token itself -- token NECESSITY is already pinned separately by
# task 2's `test_late_completion_with_stale_token_does_not_advance_a_newer_
# utterance` (`Tests/Chat/test_reply_sentence_sequencer.py`). This test
# proves token PRESENCE: the production-shaped caller (`_SequencerDriver`,
# mirroring task 5's real wiring) actually threads
# `current_utterance_token` through end to end -- not that a `None` token
# would break this specific scenario.
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
    handler,
    monkeypatch,
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

    monkeypatch.setattr(audio_output_runs, "StreamingPcmSink", _BargingSink)
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


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
@pytest.mark.parametrize("check", ["presentation", "cleanup", "superseded"])
async def test_guarded_legacy_utterance_owns_delivery_and_cleanup(
    handler, monkeypatch, check
):
    from types import SimpleNamespace

    from tldw_chatbook import app_speech
    from tldw_chatbook.Audio.speech_presentation import (
        SpeechPresentation,
        SpeechPresentationOwner,
    )

    class GuardedService(_FakeService):
        async def synthesize_default(self, *, admission_authorizer=None, **kwargs):
            return await super().synthesize_default(**kwargs)

    handler._tts_service = GuardedService(
        _FakeResponse([b"ID3", b"audio"], audio_format="mp3", sample_rate=None)
    )
    player = _FakeLegacyPlayer(finishes_after_polls=2)
    monkeypatch.setattr(audio_output_runs, "SimpleAudioPlayer", lambda: player)
    speech = SpeechPresentation(
        SpeechPresentationOwner("request", object(), "session", 1, 7, "reply", 1)
    )
    observed, outcomes, created = [], [], []
    current = [True]
    transport = SpeechPresentation.set_transport

    def observe_transport(channel, state):
        transport(channel, state)
        if channel is speech and channel.read() is not None:
            observed.append(channel.read().state)

    monkeypatch.setattr(SpeechPresentation, "set_transport", observe_transport)
    create = handler._create_tts_artifact

    def remember_artifact(audio_format):
        path = create(audio_format)
        created.append(path)
        return path

    monkeypatch.setattr(handler, "_create_tts_artifact", remember_artifact)
    cleanup = handler._cleanup_audio_file

    async def immediate_cleanup(message_id, delay=0, **kwargs):
        await cleanup(message_id, delay=0, **kwargs)

    monkeypatch.setattr(handler, "_cleanup_audio_file", immediate_cleanup)
    post = handler.post_message
    app = SimpleNamespace(loguru_logger=tts_events_module.logger)

    async def deliver(message):
        await post(message)
        if isinstance(message, TTSCompleteEvent):
            await app_speech._deliver_tts_complete_event(app, message)
            if check == "superseded":
                current[0] = False

    monkeypatch.setattr(handler, "post_message", deliver)
    await handler.speak_utterance(
        "Guarded.",
        on_finished=outcomes.append,
        presentation=speech,
        validator=lambda: current[0],
    )
    await asyncio.gather(*tuple(handler._pending_legacy_cleanup_timers))
    errors = [
        event.error
        for event in handler.messages
        if isinstance(event, TTSCompleteEvent) and event.error
    ]
    assert not errors, errors
    if check == "presentation":
        assert "playing" in observed and outcomes == [True]
        assert len(player.play_calls) == 1
    elif check == "cleanup":
        assert outcomes == [True] and created
        assert not handler._audio_files and not handler._audio_file_owners
        assert all(not path.exists() for path in created)
    else:
        assert outcomes == [False] and not player.play_calls
        assert not handler._audio_files and all(not path.exists() for path in created)
