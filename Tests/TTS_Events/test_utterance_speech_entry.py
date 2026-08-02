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

`_FakeLegacyPlayer` replaces the ad-hoc per-test fake players the original
version of this file used: F2's fix (`_play_legacy_clip_and_await_completion`
polling `get_state()`/`get_current_file()`) needs a fake that implements
that surface, not just `play()`. It is deterministic by CALL COUNT, not
wall-clock, so these tests carry no real-time sleep dependency despite
exercising the actual poll loop.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import tldw_chatbook.Event_Handlers.TTS_Events.tts_events as tts_events_module
from tldw_chatbook.Chat.reply_sentence_sequencer import SentenceSequencer
from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
    TTSCompleteEvent,
    TTSPlaybackEvent,
)
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

    async def synthesize_default(self, *, text, voice_override=None, progress_sink=None):
        await self.proceed.wait()
        return self._response


class _FakeLegacyPlayer:
    """Minimal fake for the `SimpleAudioPlayer`-shaped surface
    `_play_legacy_clip_and_await_completion` (task-4 review F2) actually
    touches: `play`, `get_state`, `get_current_file`, `stop`.

    Deterministic by CALL COUNT, not wall-clock: `get_state()` reports
    PLAYING for the first `finishes_after_polls - 1` calls, then FINISHED
    from the `finishes_after_polls`'th call onward (or, if `stop_after_polls`
    is set and reached first, simulates an externally-landed stop instead --
    current file cleared, state IDLE, exactly like the real `stop()`).
    """

    def __init__(
        self,
        *,
        play_succeeds: bool = True,
        finishes_after_polls: int = 1,
        stop_after_polls: int | None = None,
        events: list[str] | None = None,
    ) -> None:
        self._play_succeeds = play_succeeds
        self._finishes_after_polls = finishes_after_polls
        self._stop_after_polls = stop_after_polls
        self._poll_calls = 0
        self._current_file = None
        self._finished = False
        self.play_calls: list = []
        self.stop_calls = 0
        self.events = events if events is not None else []

    def play(self, file_path) -> bool:
        self.play_calls.append(file_path)
        if not self._play_succeeds:
            return False
        self._current_file = file_path
        self._finished = False
        self._poll_calls = 0
        return True

    def get_current_file(self):
        return self._current_file

    def get_state(self):
        from tldw_chatbook.TTS.audio_player import PlaybackState

        if self._current_file is None:
            return PlaybackState.IDLE
        if self._finished:
            return PlaybackState.FINISHED
        self._poll_calls += 1
        if self._stop_after_polls is not None and self._poll_calls >= self._stop_after_polls:
            self.stop()
            return PlaybackState.IDLE
        if self._poll_calls >= self._finishes_after_polls:
            self._finished = True
            self.events.append("observed_finished")
            return PlaybackState.FINISHED
        return PlaybackState.PLAYING

    def stop(self):
        self.stop_calls += 1
        self._current_file = None
        self._finished = False
        return True


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

    fake_player = _FakeLegacyPlayer(finishes_after_polls=1)
    monkeypatch.setattr(
        "tldw_chatbook.TTS.audio_player.get_audio_player", lambda: fake_player
    )

    on_finished, results = _counting_on_finished()
    try:
        await handler.speak_utterance("Discarded.", on_finished=on_finished)

        assert results == [True], "must fire exactly once, once playback actually finished"
        assert len(fake_player.play_calls) == 1, (
            "the legacy artifact must actually be played, not just written"
        )
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

    fake_player = _FakeLegacyPlayer(play_succeeds=False)
    monkeypatch.setattr(
        "tldw_chatbook.TTS.audio_player.get_audio_player", lambda: fake_player
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
    complete_events = [m for m in handler.messages if isinstance(m, TTSCompleteEvent)]
    assert len(complete_events) == 1 and complete_events[0].error is not None, (
        "by default (quiet=False) the failure toast must still post"
    )


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
# F1 (BLOCKER, task-4-review.md) -- every legacy hands-free utterance played
# TWICE: `_generate_tts` posted `TTSCompleteEvent(audio_file=artifact_path)`
# before playing it directly; no widget ever claims a `handsfree-<uuid4>`
# id, so the app's own `TTSCompleteEvent` handler (`app.py:6699-6706`)
# auto-played the SAME file a second time. Fixed: `audio_file=None` when
# `on_finished is not None`, exactly like the sink branch already does.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_legacy_completion_advertises_no_audio_file_so_the_app_cannot_double_play(
    handler, monkeypatch,
):
    chunks = [b"ID3", b"restofmp3bytes"]
    response = _FakeResponse(chunks, audio_format="mp3", sample_rate=None)
    service = _FakeService(response)
    handler._tts_service = service

    fake_player = _FakeLegacyPlayer(finishes_after_polls=1)
    monkeypatch.setattr(
        "tldw_chatbook.TTS.audio_player.get_audio_player", lambda: fake_player
    )

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
    handler, monkeypatch,
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

        complete_events = [m for m in handler.messages if isinstance(m, TTSCompleteEvent)]
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
    handler, monkeypatch,
):
    chunks = [b"ID3", b"restofmp3bytes"]
    response = _FakeResponse(chunks, audio_format="mp3", sample_rate=None)
    service = _FakeService(response)
    handler._tts_service = service

    events: list[str] = []
    fake_player = _FakeLegacyPlayer(finishes_after_polls=4, events=events)
    monkeypatch.setattr(
        "tldw_chatbook.TTS.audio_player.get_audio_player", lambda: fake_player
    )

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
async def test_completion_reports_false_when_the_clip_is_displaced_before_finishing(
    handler, monkeypatch,
):
    """A clip that stops being current for any reason OTHER than reaching
    FINISHED (an explicit stop, or displacement by a different clip) must
    report `ok=False` -- it did not play through. Deterministic: the fake's
    `get_state()` simulates an externally-landed stop at a fixed poll
    count, rather than racing a real concurrent stop against a real
    background thread."""
    chunks = [b"ID3", b"restofmp3bytes"]
    response = _FakeResponse(chunks, audio_format="mp3", sample_rate=None)
    service = _FakeService(response)
    handler._tts_service = service

    fake_player = _FakeLegacyPlayer(finishes_after_polls=1000, stop_after_polls=2)
    monkeypatch.setattr(
        "tldw_chatbook.TTS.audio_player.get_audio_player", lambda: fake_player
    )

    on_finished, results = _counting_on_finished()
    await handler.speak_utterance("Discarded.", on_finished=on_finished)

    assert results == [False]
    assert fake_player.stop_calls == 1


# ---------------------------------------------------------------------------
# F3+N2 (re-review round 2, task-4-review.md) -- round 1's fix (registering
# `_last_played` before the handoff, plus an unconditional fallback gated
# on `_last_played is None`) did NOT close the barge-in window: moving the
# registration earlier made `_stop_prior_legacy_clip` take its TRACKED
# branch during the handoff, whose `stop_audio_playback_if_current`
# identity check fails until the player actually owns the file (it does
# not yet, mid-handoff) -- a silent no-op, and the fallback (gated on
# NOTHING tracked) was now unreachable in exactly the window it targeted.
# Worse, that same blanket fallback reached from `_stream_response_via_
# sink` (the shared path EVERY TTS caller uses) could stop a completely
# UNRELATED clip playing on the process-global player singleton (N2).
#
# Redesigned: `_legacy_handoff_in_flight`, a handler-level flag set for the
# full duration of the play-and-poll call, gates an unconditional stop --
# but ONLY for the bare/global-stop call site (`bare_stop=True`, passed
# only by `handle_tts_playback`'s bare-stop branch); `_stream_response_via_
# sink`'s call keeps the ORIGINAL tracked-only, deliberate-no-op-when-
# nothing-tracked behavior unchanged (N2's fix), while the bare-stop path
# now genuinely interrupts a handoff-in-progress clip (F3's fix).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_last_played_is_still_registered_before_the_file_reaches_the_player(
    handler, monkeypatch, tmp_path,
):
    """Not the fix itself (see below) -- but this ordering is still worth
    keeping: `_last_played` reflects the truth as early as possible for
    anything else that reads it directly (the message-scoped stop branch,
    the 5s cleanup)."""
    audio_file = tmp_path / "clip.mp3"
    audio_file.write_bytes(b"fake audio data")
    fake_player = _FakeLegacyPlayer(finishes_after_polls=1)
    monkeypatch.setattr(
        "tldw_chatbook.TTS.audio_player.get_audio_player", lambda: fake_player
    )

    on_finished, results = _counting_on_finished()
    task = asyncio.create_task(
        handler._play_utterance_legacy_artifact(
            "msg-1", audio_file, "Hi there.", on_finished
        )
    )
    # Let the task run up to its own first real suspension point
    # (`_run_blocking_tts_io`'s executor hop) -- everything before that,
    # including the `_last_played` assignment and setting the handoff
    # flag, runs synchronously with no other await in between, so a
    # handful of no-op yields reliably lands here without any wall-clock
    # dependency.
    for _ in range(5):
        await asyncio.sleep(0)

    async with handler._audio_files_lock:
        assert handler._last_played == ("msg-1", audio_file)
    assert handler._legacy_handoff_in_flight is True, (
        "the handoff flag must be set for the duration of the play-and-"
        "poll call -- this is what F3's actual fix gates on"
    )

    await task
    assert results == [True]


@pytest.mark.asyncio
async def test_bare_stop_during_an_in_flight_handoff_stops_even_when_the_tracked_identity_check_would_fail(
    handler, monkeypatch, tmp_path,
):
    """The core F3 fix, isolated deterministically. A real end-to-end
    version of this probe (drive `_play_utterance_legacy_artifact` as a
    background task, then race a real `handle_tts_playback` stop against
    it) turned out NOT to be reliably mutation-sensitive: with only a
    handful of `asyncio.sleep(0)` yields, the real background thread
    (dispatched via `asyncio.to_thread`) frequently already called
    `player.play()` by the time the stop landed, so the PRE-EXISTING
    tracked branch's identity check (`stop_audio_playback_if_current`)
    would ALSO have stopped it -- masking a reverted flag-gate mutation
    (confirmed: that mutation survived the end-to-end version, 45/45).
    OS-thread scheduling cannot be pinned deterministically from the event
    loop side without real synchronization primitives, which would
    reintroduce exactly the wall-clock-flakiness this whole review round
    has been eliminating.

    Modeled directly instead: `_last_played` is set (round 1's fix, still
    in place) but the fake player's `get_current_file()` returns something
    else -- EXACTLY what the real `SimpleAudioPlayer` reports before
    `play()` has actually run (`get_current_file()` only ever reflects
    the LAST clip `play()`/`stop()` touched). This reproduces the failure
    mode of the tracked branch precisely, without needing real thread
    timing at all.
    """
    audio_file = tmp_path / "clip.mp3"
    stop_calls: list = []

    class _FakePlayerNotYetOwningTheFile:
        def get_current_file(self):
            return None  # what the real player reports before play() runs

        def stop(self):
            stop_calls.append("stop")
            return True

    monkeypatch.setattr(
        "tldw_chatbook.TTS.audio_player.get_audio_player",
        lambda: _FakePlayerNotYetOwningTheFile(),
    )
    async with handler._audio_files_lock:
        handler._last_played = ("msg-1", audio_file)  # round 1's early registration
    handler._legacy_handoff_in_flight = True  # round 2's flag

    await handler._stop_prior_legacy_clip(bare_stop=True)

    assert stop_calls == ["stop"], (
        "a bare stop during an in-flight handoff must silence the player "
        "unconditionally, even though the tracked branch's identity "
        "check would fail here (task-4 review F3)"
    )


@pytest.mark.asyncio
async def test_non_bare_stop_never_touches_unrelated_audio_when_nothing_tracked(
    handler, monkeypatch,
):
    """N2: `_stream_response_via_sink`'s own call shape (`bare_stop`
    defaults False) must remain a deliberate no-op when `_last_played is
    None` -- exactly the pre-task-4 behavior. Round 1's blanket fallback
    broke this: the reviewer's probe showed an ordinary streaming
    utterance (Console spoken feedback, `on_finished=None`) stopping a
    completely UNRELATED clip on the shared process-global player
    singleton (started by, e.g., watchlists or the STTS playground calling
    `TTS.audio_player.play_audio_file` directly).
    """
    fake_player = _FakeLegacyPlayer()
    fake_player.play(Path("unrelated-clip-started-elsewhere.mp3"))
    monkeypatch.setattr(
        "tldw_chatbook.TTS.audio_player.get_audio_player", lambda: fake_player
    )
    async with handler._audio_files_lock:
        handler._last_played = None

    await handler._stop_prior_legacy_clip()  # bare_stop=False, the sink call's own shape

    assert fake_player.stop_calls == 0, (
        "must never stop an unrelated clip when nothing is tracked "
        "(task-4 review N2)"
    )


@pytest.mark.asyncio
async def test_bare_stop_without_an_in_flight_handoff_keeps_the_tracked_only_behavior(
    handler, monkeypatch,
):
    """Regression guard: the unconditional branch must be gated on BOTH
    `bare_stop=True` AND `_legacy_handoff_in_flight` -- a bare stop with
    nothing tracked and no handoff in progress (the ordinary idle case)
    must stay a no-op, not reach for the player unconditionally."""
    fake_player = _FakeLegacyPlayer()
    monkeypatch.setattr(
        "tldw_chatbook.TTS.audio_player.get_audio_player", lambda: fake_player
    )
    async with handler._audio_files_lock:
        handler._last_played = None
    assert handler._legacy_handoff_in_flight is False

    await handler._stop_prior_legacy_clip(bare_stop=True)

    assert fake_player.stop_calls == 0


@pytest.mark.asyncio
async def test_bare_stop_still_uses_the_tracked_branch_when_something_is_tracked(
    handler, monkeypatch, tmp_path,
):
    """Regression guard: a bare stop for an ORDINARY (non-hands-free,
    no in-flight handoff) tracked clip must still go through the
    identity-checked tracked branch, exactly as before -- the new
    unconditional branch must not shadow it."""
    stop_calls: list = []
    clip = tmp_path / "clip.mp3"

    class _FakePlayer:
        def get_current_file(self):
            return clip

        def stop(self):
            stop_calls.append("stop")
            return True

    monkeypatch.setattr(
        "tldw_chatbook.TTS.audio_player.get_audio_player", lambda: _FakePlayer()
    )
    async with handler._audio_files_lock:
        handler._last_played = ("msg-1", clip)
    assert handler._legacy_handoff_in_flight is False

    await handler._stop_prior_legacy_clip(bare_stop=True)

    assert stop_calls == ["stop"], "exactly one stop call, via the tracked-clip path"


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
    half_speed = tts_events_module._legacy_playback_timeout_seconds(text_length, speed=0.5)

    expected_half = (
        text_length
        / (tts_events_module._LEGACY_PLAYBACK_MIN_CHARS_PER_SECOND * 0.5)
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
    assert tts_events_module._legacy_playback_timeout_seconds(100, speed=0.0) == baseline
    assert tts_events_module._legacy_playback_timeout_seconds(100, speed=-2.0) == baseline


@pytest.mark.asyncio
async def test_generate_tts_extracts_speed_from_preferences_and_threads_it_through(
    handler, monkeypatch,
):
    chunks = [b"ID3", b"restofmp3bytes"]
    response = _FakeResponse(chunks, audio_format="mp3", sample_rate=None)

    class _SlowSpeedService:
        def preferences_snapshot(self):
            from types import SimpleNamespace

            return SimpleNamespace(provider_id="openai", speed=0.5)

        async def synthesize_default(self, *, text, voice_override=None, progress_sink=None):
            return response

    handler._tts_service = _SlowSpeedService()

    fake_player = _FakeLegacyPlayer(finishes_after_polls=1)
    monkeypatch.setattr(
        "tldw_chatbook.TTS.audio_player.get_audio_player", lambda: fake_player
    )

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


def test_a_timed_out_poll_logs_a_distinguishing_warning(monkeypatch):
    """N3: a timeout must be OBSERVABLE, not silently indistinguishable
    from a natural finish (both currently still return True -- narrowing
    that gap further is future work; this pins what shipped)."""
    fake_player = _FakeLegacyPlayer(finishes_after_polls=1_000_000)
    audio_file = Path("clip.mp3")

    captured_logs: list[str] = []
    from loguru import logger as loguru_logger

    handler_id = loguru_logger.add(
        lambda message: captured_logs.append(message.record["message"]),
        level="WARNING",
    )
    try:
        result = tts_events_module._play_legacy_clip_and_await_completion(
            fake_player,
            audio_file,
            timeout_seconds=0.05,
            poll_interval_seconds=0.01,
        )
    finally:
        loguru_logger.remove(handler_id)

    assert result is True
    assert any("timed out" in line.lower() for line in captured_logs), captured_logs


# ---------------------------------------------------------------------------
# N4 (Minor, task-4-review.md) -- `_run_blocking_tts_io`'s `asyncio.shield`
# meant cancellation only stopped the player once the shielded worker
# actually RETURNED (bounded by a separate ~1s internal join), not
# promptly. Fixed: a bespoke offload (bare `asyncio.to_thread` + shield)
# for THIS call specifically, so cancellation stops the player immediately.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancelling_mid_poll_promptly_stops_the_player(handler, monkeypatch):
    chunks = [b"ID3", b"restofmp3bytes"]
    response = _FakeResponse(chunks, audio_format="mp3", sample_rate=None)
    service = _FakeService(response)
    handler._tts_service = service

    # Never finishes on its own within any realistic test timeframe --
    # isolates the assertion to "did cancellation itself stop it".
    fake_player = _FakeLegacyPlayer(finishes_after_polls=1_000_000)
    monkeypatch.setattr(
        "tldw_chatbook.TTS.audio_player.get_audio_player", lambda: fake_player
    )

    on_finished, results = _counting_on_finished()
    task = asyncio.create_task(
        handler.speak_utterance("Discarded.", on_finished=on_finished)
    )
    # `speak_utterance` crosses several real await points (text prep,
    # artifact write, THEN the legacy play-and-poll) before reaching the
    # handoff -- loop rather than a fixed count, bounded so a genuine
    # regression fails fast instead of hanging.
    for _ in range(500):
        if handler._legacy_handoff_in_flight:
            break
        await asyncio.sleep(0)
    assert handler._legacy_handoff_in_flight is True

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert fake_player.stop_calls >= 1, (
        "cancellation must promptly stop the player, not wait out the "
        "poll's own bound (task-4 review N4)"
    )
    assert results == [False], "on_finished must still fire exactly once via the finally net"


# ---------------------------------------------------------------------------
# N5 (Minor, task-4-review.md) -- `cleanup_tts_resources()` used to AWAIT
# (boundedly) the legacy cleanup timer, buying nothing since the same
# method deletes the artifact directly moments later regardless -- a
# measured ~2s shutdown cost for zero benefit. Fixed: cancel it outright.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_tts_resources_cancels_the_legacy_timer_instead_of_awaiting_it(
    handler, monkeypatch,
):
    chunks = [b"ID3", b"restofmp3bytes"]
    response = _FakeResponse(chunks, audio_format="mp3", sample_rate=None)
    service = _FakeService(response)
    handler._tts_service = service

    fake_player = _FakeLegacyPlayer(finishes_after_polls=1)
    monkeypatch.setattr(
        "tldw_chatbook.TTS.audio_player.get_audio_player", lambda: fake_player
    )

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
    handler, monkeypatch,
):
    chunks = [b"ID3", b"restofmp3bytes"]
    response = _FakeResponse(chunks, audio_format="mp3", sample_rate=None)
    service = _PausableService(response)
    handler._tts_service = service

    fake_player = _FakeLegacyPlayer(finishes_after_polls=1)
    monkeypatch.setattr(
        "tldw_chatbook.TTS.audio_player.get_audio_player", lambda: fake_player
    )

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
        assert len(handler._active_tasks) == 0, "must be removed once generation completes"
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
    handler, monkeypatch,
):
    chunks = [bytes([1, 0]) * 10]
    response = _FakeResponse(chunks, audio_format="pcm", sample_rate=RATE)
    service = _FakeService(response)
    handler._tts_service = service

    monkeypatch.setattr(tts_events_module, "StreamingPcmSink", _RecordingSink)
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
